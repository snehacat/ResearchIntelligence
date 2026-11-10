"""Google Gemini structured-generation backend.

The endpoint, authentication header, request payload field names, and response
field names are assumed, to be verified by the owner at the first real call.
"""

import json
import os
import socket
import time
from typing import Any, Callable, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from analysis.llm_backend import LLMBackend, LLMGenerationResult


class GeminiBackendError(RuntimeError):
    """Sanitized Gemini request or response failure."""


class GeminiHTTPError(GeminiBackendError):
    """Gemini HTTP failure with status and optional retry guidance."""

    def __init__(self, status_code: int, retry_after: Optional[str] = None) -> None:
        self.status_code = status_code
        self.retry_after = retry_after
        message = f"Gemini request failed with HTTP status {status_code}"
        if retry_after is not None:
            message += f" (Retry-After: {retry_after})"
        super().__init__(message)


class GeminiTimeoutError(TimeoutError):
    """Sanitized Gemini request timeout."""

    def __init__(self) -> None:
        super().__init__("Gemini request timed out.")


class GeminiBackend(LLMBackend):
    """Gemini backend using an injectable standard-library HTTP transport."""

    name = "gemini"
    is_real = True
    BASE_URL = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "{model}:generateContent"
    )

    def __init__(
        self,
        model_name: str,
        api_key_env_var: str = "GEMINI_API_KEY",
        http_post: Optional[Callable[..., bytes]] = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(api_key_env_var, str) or not api_key_env_var.strip():
            raise ValueError("api_key_env_var must be a non-empty string")
        self.model_name = model_name
        self.api_key_env_var = api_key_env_var
        self.http_post = self._http_post if http_post is None else http_post

    def generate_json(
        self,
        prompt: str,
        *,
        timeout_seconds: float,
        options: Mapping[str, Any],
    ) -> LLMGenerationResult:
        api_key = os.environ.get(self.api_key_env_var)
        if not api_key:
            raise GeminiBackendError(
                f"Missing API key environment variable: {self.api_key_env_var}"
            )

        # These endpoint, header, and field names are assumed and must be
        # verified by the owner at the first real call.
        url = self.BASE_URL.format(model=quote(self.model_name, safe=""))
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0,
                "maxOutputTokens": options.get("num_predict", 768),
            },
        }
        started_at = time.perf_counter()
        try:
            response_body = self.http_post(
                url,
                headers=headers,
                body=body,
                timeout=timeout_seconds,
            )
        except HTTPError as error:
            retry_after = error.headers.get("Retry-After") if error.headers else None
            if isinstance(retry_after, str) and api_key in retry_after:
                retry_after = retry_after.replace(api_key, "[REDACTED]")
            raise GeminiHTTPError(error.code, retry_after) from None
        except (TimeoutError, socket.timeout):
            raise GeminiTimeoutError() from None
        except URLError as error:
            if isinstance(error.reason, (TimeoutError, socket.timeout)):
                raise GeminiTimeoutError() from None
            raise GeminiBackendError("Gemini request failed.") from None
        except Exception:
            raise GeminiBackendError("Gemini request failed.") from None

        duration = time.perf_counter() - started_at
        try:
            if isinstance(response_body, bytes):
                response_text = response_body.decode("utf-8")
            elif isinstance(response_body, str):
                response_text = response_body
            else:
                raise TypeError("HTTP transport must return bytes or text")
            response = json.loads(response_text)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
            raise GeminiBackendError("Gemini returned an invalid response.") from None
        if not isinstance(response, dict):
            raise GeminiBackendError("Gemini returned an invalid response.")

        candidates = response.get("candidates", [])
        candidate = candidates[0] if isinstance(candidates, list) and candidates else {}
        content = candidate.get("content", {}) if isinstance(candidate, dict) else {}
        parts = content.get("parts", []) if isinstance(content, dict) else []
        text = ""
        if isinstance(parts, list) and parts and isinstance(parts[0], dict):
            candidate_text = parts[0].get("text", "")
            if isinstance(candidate_text, str):
                text = candidate_text
        if api_key in text:
            text = text.replace(api_key, "[REDACTED]")

        usage = response.get("usageMetadata", {})
        if not isinstance(usage, dict):
            usage = {}
        metadata = {
            "done_reason": self._redact(
                candidate.get("finishReason"), api_key
            ) if isinstance(candidate, dict) else None,
            "prompt_eval_count": self._token_count(
                usage.get("promptTokenCount")
            ),
            "eval_count": self._token_count(
                usage.get("candidatesTokenCount")
            ),
            "duration": duration,
        }
        return LLMGenerationResult(text=text, metadata=metadata)

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

    @staticmethod
    def _http_post(
        url: str,
        *,
        headers: Mapping[str, str],
        body: Mapping[str, Any],
        timeout: float,
    ) -> bytes:
        request = Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=dict(headers),
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            return response.read()
