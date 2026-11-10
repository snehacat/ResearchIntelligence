"""Groq structured-generation backend using the official Python SDK."""

import os
import time
from typing import Any, Callable, Mapping, Optional

from analysis.llm_backend import LLMBackend, LLMGenerationResult


class GroqBackendError(RuntimeError):
    """Sanitized Groq request or response failure."""


class GroqAuthenticationError(GroqBackendError):
    """Groq rejected the configured API key."""


class GroqRateLimitError(GroqBackendError):
    """Groq rate limit was exhausted after bounded retries."""

    def __init__(self, retry_after: Optional[str], attempts: int) -> None:
        self.retry_after = retry_after
        self.attempts = attempts
        message = f"Groq rate limit reached after {attempts} attempt(s)"
        if retry_after is not None:
            message += f" (Retry-After: {retry_after})"
        super().__init__(message)


class GroqTimeoutError(TimeoutError):
    """Groq request timed out."""

    def __init__(self, attempts: int) -> None:
        self.attempts = attempts
        super().__init__(f"Groq request timed out after {attempts} attempt(s).")


class GroqNetworkError(GroqBackendError):
    """Groq could not be reached after bounded retries."""


class GroqAPIError(GroqBackendError):
    """Groq returned an unsuccessful HTTP response."""

    def __init__(
        self,
        status_code: Optional[int],
        attempts: int,
        retry_after: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        self.status_code = status_code
        self.attempts = attempts
        self.retry_after = retry_after
        self.detail = detail
        message = f"Groq request failed after {attempts} attempt(s)"
        if status_code is not None:
            message += f" with HTTP status {status_code}"
        if retry_after is not None:
            message += f" (Retry-After: {retry_after})"
        if detail:
            message += f": {detail}"
        super().__init__(message)


class GroqResponseError(GroqBackendError):
    """Groq returned an unexpected response structure."""


class GroqBackend(LLMBackend):
    """Generate schema-constrained JSON through the Groq chat completions API."""

    name = "groq"
    is_real = True
    DEFAULT_MODEL = "openai/gpt-oss-20b"
    API_KEY_ENV_VAR = "GROQ_API_KEY"
    MAX_RETRIES = 1
    MAX_RETRY_DELAY_SECONDS = 5.0
    SCHEMA = {
        "type": "object",
        "properties": {
            "title": {"type": ["string", "null"]},
            "authors": {
                "type": ["array", "null"],
                "items": {"type": "string"},
            },
            "publication_year": {"type": ["string", "null"]},
            "abstract": {"type": ["string", "null"]},
            "research_problem": {"type": ["string", "null"]},
            "methodology": {"type": ["string", "null"]},
            "dataset": {"type": ["string", "null"]},
            "evaluation_metrics": {
                "type": ["array", "null"],
                "items": {"type": "string"},
            },
            "reported_results": {"type": ["string", "null"]},
            "limitations": {
                "type": ["array", "null"],
                "items": {"type": "string"},
            },
        },
        "required": [
            "title",
            "authors",
            "publication_year",
            "abstract",
            "research_problem",
            "methodology",
            "dataset",
            "evaluation_metrics",
            "reported_results",
            "limitations",
        ],
        "additionalProperties": False,
    }

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key_env_var: str = API_KEY_ENV_VAR,
        client_factory: Optional[Callable[..., Any]] = None,
        max_retries: int = MAX_RETRIES,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(api_key_env_var, str) or not api_key_env_var.strip():
            raise ValueError("api_key_env_var must be a non-empty string")
        if (
            not isinstance(max_retries, int)
            or isinstance(max_retries, bool)
            or not 0 <= max_retries <= self.MAX_RETRIES
        ):
            raise ValueError(f"max_retries must be between 0 and {self.MAX_RETRIES}")
        self.model_name = model_name
        self.api_key_env_var = api_key_env_var
        self.client_factory = (
            self._create_client if client_factory is None else client_factory
        )
        self.max_retries = max_retries
        self.sleep = sleep

    def generate_json(
        self,
        prompt: str,
        *,
        timeout_seconds: float,
        options: Mapping[str, Any],
    ) -> LLMGenerationResult:
        api_key = os.environ.get(self.api_key_env_var)
        if not api_key:
            raise GroqBackendError(
                f"Missing API key environment variable: {self.api_key_env_var}"
            )

        try:
            client = self.client_factory(
                api_key=api_key,
                timeout=timeout_seconds,
                max_retries=0,
            )
        except GroqBackendError:
            raise
        except Exception:
            raise GroqBackendError("Could not initialize the Groq client.") from None
        started_at = time.perf_counter()
        attempt = 0
        request_count = 0
        used_json_mode_fallback = False
        last_error_detail = None
        last_status_code = None
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "research_paper_analysis",
                "strict": True,
                "schema": self.SCHEMA,
            },
        }
        while True:
            request_count += 1
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=response_format,
                    temperature=0,
                    max_completion_tokens=options.get("num_predict", 768),
                )
                break
            except Exception as error:
                status_code = self._status_code(error)
                last_status_code = status_code
                retry_after = self._retry_after(error)
                if retry_after is not None:
                    retry_after = retry_after.replace(api_key, "[REDACTED]")
                error_detail = self._safe_error_detail(error, api_key)
                last_error_detail = error_detail
                if (
                    not used_json_mode_fallback
                    and status_code == 400
                    and error_detail is not None
                    and any(
                        message in error_detail.casefold()
                        for message in (
                            "failed to generate json",
                            "failed to validate json",
                        )
                    )
                ):
                    response_format = {"type": "json_object"}
                    used_json_mode_fallback = True
                    continue
                error_name = type(error).__name__.casefold()
                timed_out = isinstance(error, TimeoutError) or "timeout" in error_name
                network_error = "connection" in error_name
                retryable = (
                    status_code == 429
                    or (status_code is not None and 500 <= status_code < 600)
                    or timed_out
                    or network_error
                )
                if retryable and attempt < self.max_retries:
                    self.sleep(self._retry_delay(retry_after, attempt))
                    attempt += 1
                    continue
                if status_code in (401, 403) or "authentication" in error_name:
                    raise GroqAuthenticationError(
                        "Groq authentication failed; check GROQ_API_KEY."
                    ) from None
                if status_code == 429 or "ratelimit" in error_name:
                    raise GroqRateLimitError(
                        retry_after, attempts=request_count
                    ) from None
                if timed_out:
                    raise GroqTimeoutError(attempts=request_count) from None
                if network_error:
                    raise GroqNetworkError(
                        f"Groq connection failed after {request_count} attempt(s)."
                    ) from None
                raise GroqAPIError(
                    status_code,
                    attempts=request_count,
                    retry_after=retry_after,
                    detail=error_detail,
                ) from None

        duration = time.perf_counter() - started_at
        try:
            choice = response.choices[0]
            message = choice.message
            text = message.content or ""
            if not isinstance(text, str):
                raise TypeError
        except (AttributeError, IndexError, TypeError):
            raise GroqResponseError(
                "Groq returned an unexpected chat completion response."
            ) from None

        text = text.replace(api_key, "[REDACTED]")
        usage = getattr(response, "usage", None)
        metadata = {
            "done_reason": self._redact(
                getattr(choice, "finish_reason", None), api_key
            ),
            "prompt_eval_count": self._token_count(
                getattr(usage, "prompt_tokens", None)
            ),
            "eval_count": self._token_count(
                getattr(usage, "completion_tokens", None)
            ),
            "duration": duration,
            "response_format": (
                "json_object" if used_json_mode_fallback else "json_schema"
            ),
            "used_json_mode_fallback": used_json_mode_fallback,
        }
        return LLMGenerationResult(text=text, metadata=metadata)

    @staticmethod
    def _create_client(
        *, api_key: str, timeout: float, max_retries: int
    ) -> Any:
        try:
            from groq import Groq
        except ImportError as error:
            raise GroqBackendError(
                "The Groq SDK is missing. Install dependencies with "
                "'pip install -r requirements.txt'."
            ) from error
        return Groq(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    @staticmethod
    def _status_code(error: Exception) -> Optional[int]:
        status_code = getattr(error, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
        return status_code if isinstance(status_code, int) else None

    @staticmethod
    def _retry_after(error: Exception) -> Optional[str]:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            headers = getattr(error, "headers", None)
        value = None
        if headers is not None:
            value = headers.get("retry-after")
            if value is None:
                value = next(
                    (
                        header_value
                        for header_name, header_value in headers.items()
                        if str(header_name).casefold() == "retry-after"
                    ),
                    None,
                )
        return str(value) if value is not None else None

    @staticmethod
    def _safe_error_detail(error: Exception, secret: str) -> Optional[str]:
        body = getattr(error, "body", None)
        if not isinstance(body, Mapping):
            return None
        provider_error = body.get("error", body)
        if isinstance(provider_error, Mapping):
            detail = provider_error.get("message")
        else:
            detail = None
        if not isinstance(detail, str) or not detail.strip():
            return None
        safe_detail = detail.replace(secret, "[REDACTED]")
        safe_detail = " ".join(safe_detail.split())
        return safe_detail[:500]

    @classmethod
    def _retry_delay(
        cls, retry_after: Optional[str], attempt: int
    ) -> float:
        if retry_after is not None:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = 0.5 * (2 ** attempt)
        else:
            delay = 0.5 * (2 ** attempt)
        return min(max(delay, 0.0), cls.MAX_RETRY_DELAY_SECONDS)

    @staticmethod
    def _redact(value: Any, secret: str) -> Any:
        if isinstance(value, str):
            return value.replace(secret, "[REDACTED]")
        return value

    @staticmethod
    def _token_count(value: Any) -> Optional[int]:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return None
