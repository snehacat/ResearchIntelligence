from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from analysis.groq_backend import (
    GroqAPIError,
    GroqAuthenticationError,
    GroqBackend,
    GroqBackendError,
    GroqNetworkError,
    GroqRateLimitError,
    GroqResponseError,
    GroqTimeoutError,
)
from analysis.paper_analyzer import PaperAnalyzer
from paper_analysis_errors import ExtractionError
from validation.step5.run_validation import (
    _prepare_output_directory,
    create_evidence_audit,
    parse_arguments,
)


API_KEY = "unit-test-groq-secret"


def completion(
    text='{"title": "Example"}',
    *,
    finish_reason="stop",
    prompt_tokens=21,
    completion_tokens=13,
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=text),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


def client_with_results(*results):
    request = Mock(side_effect=list(results))
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=request)
        )
    )
    return client, request


def test_groq_backend_sends_strict_schema_request_and_reports_metadata(
    monkeypatch,
):
    monkeypatch.setenv("GROQ_API_KEY", API_KEY)
    client, request = client_with_results(completion())
    factories = []

    def factory(**kwargs):
        factories.append(kwargs)
        return client

    backend = GroqBackend(client_factory=factory)
    result = backend.generate_json(
        "extract this excerpt",
        timeout_seconds=27,
        options={"num_predict": 321, "num_ctx": 4096},
    )

    assert backend.name == "groq"
    assert backend.is_real is True
    assert backend.model_name == "openai/gpt-oss-20b"
    assert factories == [{
        "api_key": API_KEY,
        "timeout": 27,
        "max_retries": 0,
    }]
    kwargs = request.call_args.kwargs
    assert kwargs["model"] == "openai/gpt-oss-20b"
    assert kwargs["messages"] == [
        {"role": "user", "content": "extract this excerpt"}
    ]
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["strict"] is True
    schema = kwargs["response_format"]["json_schema"]["schema"]
    assert schema["required"] == list(schema["properties"])
    assert schema["additionalProperties"] is False
    assert kwargs["temperature"] == 0
    assert kwargs["max_completion_tokens"] == 321
    assert "num_ctx" not in kwargs
    assert result.text == '{"title": "Example"}'
    assert result.metadata["done_reason"] == "stop"
    assert result.metadata["prompt_eval_count"] == 21
    assert result.metadata["eval_count"] == 13
    assert result.metadata["duration"] >= 0


def test_groq_backend_requires_key_without_exposing_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    backend = GroqBackend(client_factory=Mock())

    with pytest.raises(GroqBackendError, match="GROQ_API_KEY") as error:
        backend.generate_json("prompt", timeout_seconds=3, options={})

    assert API_KEY not in str(error.value)


def test_groq_backend_reads_api_key_at_call_time(monkeypatch):
    client, _ = client_with_results(completion())
    factory = Mock(return_value=client)
    backend = GroqBackend(client_factory=factory)
    monkeypatch.setenv("GROQ_API_KEY", API_KEY)

    backend.generate_json("prompt", timeout_seconds=3, options={})

    assert factory.call_args.kwargs["api_key"] == API_KEY


def test_groq_backend_retries_rate_limit_once_and_honors_bounded_delay(
    monkeypatch,
):
    monkeypatch.setenv("GROQ_API_KEY", API_KEY)

    class RateLimitError(Exception):
        status_code = 429
        response = SimpleNamespace(headers={"retry-after": "9"})

    client, request = client_with_results(RateLimitError(), completion())
    sleeps = []
    backend = GroqBackend(
        client_factory=lambda **kwargs: client,
        sleep=sleeps.append,
    )

    result = backend.generate_json("prompt", timeout_seconds=3, options={})

    assert result.text == '{"title": "Example"}'
    assert request.call_count == 2
    assert sleeps == [backend.MAX_RETRY_DELAY_SECONDS]


@pytest.mark.parametrize(
    ("exception", "expected_type", "expected_message"),
    [
        (
            type(
                "AuthenticationError",
                (Exception,),
                {"status_code": 401},
            )("bad credentials"),
            GroqAuthenticationError,
            "GROQ_API_KEY",
        ),
        (
            type(
                "RateLimitError",
                (Exception,),
                {
                    "status_code": 429,
                    "response": SimpleNamespace(
                        headers={"retry-after": "3"}
                    ),
                },
            )("rate limited"),
            GroqRateLimitError,
            "after 1 attempt",
        ),
        (
            type(
                "InternalServerError",
                (Exception,),
                {"status_code": 503},
            )("server error"),
            GroqAPIError,
            "503",
        ),
        (TimeoutError("secret timeout"), GroqTimeoutError, "timed out"),
        (
            type("APIConnectionError", (Exception,), {})("secret connection"),
            GroqNetworkError,
            "connection failed",
        ),
    ],
)
def test_groq_backend_maps_failures_without_leaking_provider_messages(
    monkeypatch, exception, expected_type, expected_message
):
    monkeypatch.setenv("GROQ_API_KEY", API_KEY)
    client, _ = client_with_results(exception)
    backend = GroqBackend(
        client_factory=lambda **kwargs: client,
        max_retries=0,
    )

    with pytest.raises(expected_type, match=expected_message) as error:
        backend.generate_json("prompt", timeout_seconds=3, options={})

    assert "secret" not in str(error.value)
    assert API_KEY not in str(error.value)


def test_groq_backend_redacts_echoed_secret_from_completion(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", API_KEY)
    client, _ = client_with_results(
        completion(f"echoed {API_KEY}", finish_reason=API_KEY)
    )
    backend = GroqBackend(client_factory=lambda **kwargs: client)

    result = backend.generate_json("prompt", timeout_seconds=3, options={})

    assert API_KEY not in result.text
    assert API_KEY not in repr(result.metadata)
    assert "[REDACTED]" in result.text
    assert result.metadata["done_reason"] == "[REDACTED]"


def test_groq_secret_is_redacted_from_analyzer_diagnostics(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", API_KEY)
    client, _ = client_with_results(
        completion(f"malformed {API_KEY}"),
        completion(f"malformed {API_KEY}"),
    )
    backend = GroqBackend(
        client_factory=lambda **kwargs: client,
        max_retries=0,
    )
    analyzer = PaperAnalyzer(
        Mock(),
        Mock(),
        Mock(),
        llm_backend=backend,
        model_name=backend.model_name,
    )

    with pytest.raises(ExtractionError) as error:
        analyzer._extract_structured_json([{"text": "source excerpt"}])

    assert API_KEY not in str(error.value)
    assert API_KEY not in repr(error.value.diagnostics)
    assert "[REDACTED]" in error.value.diagnostics["raw_response"]


def test_groq_backend_handles_empty_and_malformed_completions(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", API_KEY)
    empty_client, _ = client_with_results(completion(None))
    empty_result = GroqBackend(
        client_factory=lambda **kwargs: empty_client
    ).generate_json("prompt", timeout_seconds=3, options={})
    assert empty_result.text == ""

    malformed_client, _ = client_with_results(
        SimpleNamespace(choices=[], usage=None)
    )
    with pytest.raises(GroqResponseError, match="unexpected"):
        GroqBackend(
            client_factory=lambda **kwargs: malformed_client
        ).generate_json("prompt", timeout_seconds=3, options={})


def test_groq_backend_reports_exhausted_transient_retries(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", API_KEY)

    class RateLimitError(Exception):
        status_code = 429

    client, request = client_with_results(RateLimitError(), RateLimitError())
    backend = GroqBackend(
        client_factory=lambda **kwargs: client,
        sleep=lambda _: None,
    )

    with pytest.raises(GroqRateLimitError) as error:
        backend.generate_json("prompt", timeout_seconds=3, options={})

    assert request.call_count == 2
    assert error.value.attempts == 2


def test_groq_backend_exposes_server_retry_after_without_provider_body(
    monkeypatch,
):
    monkeypatch.setenv("GROQ_API_KEY", API_KEY)

    class ServerError(Exception):
        status_code = 503
        response = SimpleNamespace(headers={"Retry-After": "4"})

    client, _ = client_with_results(ServerError("secret response"))
    backend = GroqBackend(
        client_factory=lambda **kwargs: client,
        max_retries=0,
    )

    with pytest.raises(GroqAPIError) as error:
        backend.generate_json("prompt", timeout_seconds=3, options={})

    assert error.value.status_code == 503
    assert error.value.retry_after == "4"
    assert "Retry-After: 4" in str(error.value)
    assert "secret response" not in str(error.value)
    assert API_KEY not in str(error.value)


def test_groq_backend_includes_bounded_sanitized_provider_error_detail(
    monkeypatch,
):
    monkeypatch.setenv("GROQ_API_KEY", API_KEY)

    class BadRequestError(Exception):
        status_code = 400
        body = {
            "error": {
                "message": f"Invalid option; echoed {API_KEY} and extra details.",
            }
        }

    client, _ = client_with_results(BadRequestError())
    backend = GroqBackend(
        client_factory=lambda **kwargs: client,
        max_retries=0,
    )

    with pytest.raises(GroqAPIError) as error:
        backend.generate_json("prompt", timeout_seconds=3, options={})

    assert error.value.detail == (
        "Invalid option; echoed [REDACTED] and extra details."
    )
    assert API_KEY not in str(error.value)
    assert len(error.value.detail) <= 500


@pytest.mark.parametrize(
    "provider_message",
    [
        "Failed to generate JSON. Adjust your prompt.",
        "Failed to validate JSON. Adjust your prompt.",
    ],
)
def test_groq_backend_falls_back_to_json_mode_for_schema_rejection(
    monkeypatch, provider_message
):
    monkeypatch.setenv("GROQ_API_KEY", API_KEY)

    class SchemaValidationError(Exception):
        status_code = 400

        def __init__(self):
            super().__init__(provider_message)
            self.body = {"error": {"message": provider_message}}

    client, request = client_with_results(SchemaValidationError(), completion())
    backend = GroqBackend(client_factory=lambda **kwargs: client)

    result = backend.generate_json(
        "Return a JSON object with a title field.",
        timeout_seconds=3,
        options={},
    )

    assert request.call_count == 2
    assert request.call_args_list[0].kwargs["response_format"]["type"] == (
        "json_schema"
    )
    assert request.call_args_list[1].kwargs["response_format"] == {
        "type": "json_object"
    }
    assert result.metadata["response_format"] == "json_object"


def test_groq_backend_does_not_fallback_on_unrelated_bad_request(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", API_KEY)

    class BadRequestError(Exception):
        status_code = 400
        body = {"error": {"message": "Model is unavailable."}}

    client, request = client_with_results(BadRequestError())
    backend = GroqBackend(
        client_factory=lambda **kwargs: client,
        max_retries=0,
    )

    with pytest.raises(GroqAPIError, match="Model is unavailable"):
        backend.generate_json("prompt", timeout_seconds=3, options={})

    assert request.call_count == 1


def test_groq_backend_retries_transient_server_error_once(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", API_KEY)

    class ServerError(Exception):
        status_code = 500
        response = SimpleNamespace(headers={"retry-after": "2"})

    client, request = client_with_results(ServerError(), completion())
    sleeps = []
    backend = GroqBackend(
        client_factory=lambda **kwargs: client,
        sleep=sleeps.append,
    )

    result = backend.generate_json("prompt", timeout_seconds=3, options={})

    assert result.text == '{"title": "Example"}'
    assert request.call_count == 2
    assert sleeps == [2.0]


def test_analyzer_maps_groq_timeout_to_actionable_extraction_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", API_KEY)
    client, _ = client_with_results(TimeoutError("network timeout"))
    backend = GroqBackend(
        client_factory=lambda **kwargs: client,
        sleep=lambda _: None,
        max_retries=0,
    )
    analyzer = PaperAnalyzer(
        Mock(),
        Mock(),
        Mock(),
        llm_backend=backend,
        model_name=backend.model_name,
    )

    with pytest.raises(ExtractionError, match="timed out after 90 seconds"):
        analyzer._extract_structured_json([{"text": "source excerpt"}])


def test_backend_configuration_defaults_to_groq(
    monkeypatch,
):
    monkeypatch.delenv("RESEARCHINTELLIGENCE_LLM_BACKEND", raising=False)
    monkeypatch.delenv("RESEARCHINTELLIGENCE_LLM_MODEL", raising=False)
    analyzer = PaperAnalyzer(Mock(), Mock(), Mock())

    assert analyzer.model_name == GroqBackend.DEFAULT_MODEL
    assert isinstance(analyzer.llm_backend, GroqBackend)

    monkeypatch.setenv("RESEARCHINTELLIGENCE_LLM_BACKEND", "invalid")
    with pytest.raises(ValueError, match="must be one of"):
        PaperAnalyzer(Mock(), Mock(), Mock())


def test_cli_groq_requires_data_and_cost_confirmation():
    with pytest.raises(SystemExit):
        parse_arguments(["--backend", "groq"])
    with pytest.raises(SystemExit):
        parse_arguments([
            "--backend",
            "groq",
            "--confirm-public-papers",
        ])

    args = parse_arguments([
        "--backend",
        "groq",
        "--confirm-public-papers",
        "--confirm-groq-costs",
    ])
    assert args.backend == "groq"
    assert args.model == GroqBackend.DEFAULT_MODEL


def test_cli_rejects_removed_ollama_backend():
    with pytest.raises(SystemExit):
        parse_arguments(["--backend", "ollama"])


def test_output_directory_creation_never_overwrites_existing_directory(
    tmp_path,
):
    existing = tmp_path / "previous-validation"
    existing.mkdir()

    with pytest.raises(FileExistsError):
        _prepare_output_directory(existing)

    created = _prepare_output_directory(tmp_path / "new-validation")
    assert created.is_dir()


def test_evidence_audit_preserves_document_and_flags_unverified_support():
    first_passage = "The original source passage. " * 15
    second_passage = "A second supporting source passage."
    audit = create_evidence_audit([{
        "status": "success",
        "pdf_path": "example.pdf",
        "document_id": "paper-1",
        "analysis": {
            "title": "Example title",
            "evidence_citations": {
                "title": {
                    "evidence_chunks": [{
                        "chunk_id": "chunk-1",
                        "document_id": "paper-1",
                        "start_page": 2,
                        "end_page": 2,
                        "text": first_passage,
                    }, {
                        "chunk_id": "chunk-2",
                        "document_id": "paper-1",
                        "start_page": 3,
                        "end_page": 3,
                        "text": second_passage,
                    }],
                    "citations": [{
                        "document_id": "paper-1",
                        "chunk_id": "chunk-1",
                        "start_page": 2,
                        "end_page": 2,
                    }, {
                        "document_id": "paper-1",
                        "chunk_id": "chunk-2",
                        "start_page": 3,
                        "end_page": 3,
                    }],
                    "evidence_type": "explicit_fact",
                    "evidence_sufficiency": "sufficient",
                    "requires_human_review": False,
                    "lexical_support": 1.0,
                    "retrieval_similarity": 0.99,
                }
            },
        },
    }])

    title_rows = [row for row in audit if row["field_name"] == "title"]
    assert len(title_rows) == 2
    assert [row["chunk_id"] for row in title_rows] == ["chunk-1", "chunk-2"]
    assert title_rows[0]["document_id"] == "paper-1"
    assert title_rows[0]["cited_page"] == "2-2"
    assert title_rows[0]["supporting_text"] == first_passage
    assert title_rows[1]["cited_page"] == "3-3"
    assert title_rows[1]["supporting_text"] == second_passage
    for row in title_rows:
        assert row["automated_assessment"] == "reference_consistent_unverified"
        assert row["requires_human_review"] is True
        assert row["human_verdict"] == ""
        assert row["human_notes"] == ""


def test_evidence_audit_rejects_cross_document_citation():
    audit = create_evidence_audit([{
        "status": "success",
        "pdf_path": "example.pdf",
        "document_id": "paper-1",
        "analysis": {
            "title": "Example title",
            "evidence_citations": {
                "title": {
                    "evidence_chunks": [{
                        "chunk_id": "chunk-1",
                        "document_id": "paper-1",
                        "start_page": 2,
                        "end_page": 2,
                        "text": "The source passage.",
                    }],
                    "citations": [{
                        "document_id": "paper-2",
                        "chunk_id": "chunk-1",
                        "start_page": 2,
                        "end_page": 2,
                    }],
                }
            },
        },
    }])

    title_row = next(row for row in audit if row["field_name"] == "title")
    assert title_row["citation_status"] == "wrong_document"
    assert title_row["supporting_text"] == ""
    assert title_row["automated_assessment"] == "unsupported_reference"


class TestGroqRateLimitHandlingPriority4:
    """Priority 4: Focused tests for HTTP 429 rate-limit handling."""

    def test_429_without_retry_after_uses_exponential_backoff(self, monkeypatch):
        """Test 429 without Retry-After header uses exponential backoff."""
        monkeypatch.setenv("GROQ_API_KEY", API_KEY)

        class RateLimitError(Exception):
            status_code = 429
            response = SimpleNamespace(headers={})

        client, request = client_with_results(RateLimitError(), completion())
        sleeps = []
        backend = GroqBackend(
            client_factory=lambda **kwargs: client,
            sleep=sleeps.append,
        )

        result = backend.generate_json("prompt", timeout_seconds=3, options={})

        assert result.text == '{"title": "Example"}'
        assert request.call_count == 2
        assert sleeps == [0.5]  # Exponential backoff: 0.5 * (2^0)

    def test_429_with_invalid_retry_after_uses_exponential_backoff(self, monkeypatch):
        """Test 429 with invalid Retry-After header falls back to exponential backoff."""
        monkeypatch.setenv("GROQ_API_KEY", API_KEY)

        class RateLimitError(Exception):
            status_code = 429
            response = SimpleNamespace(headers={"retry-after": "invalid"})

        client, request = client_with_results(RateLimitError(), completion())
        sleeps = []
        backend = GroqBackend(
            client_factory=lambda **kwargs: client,
            sleep=sleeps.append,
        )

        result = backend.generate_json("prompt", timeout_seconds=3, options={})

        assert result.text == '{"title": "Example"}'
        assert request.call_count == 2
        assert sleeps == [0.5]  # Falls back to exponential backoff

    def test_429_with_large_retry_after_is_capped(self, monkeypatch):
        """Test 429 with large Retry-After value is capped at MAX_RETRY_DELAY_SECONDS."""
        monkeypatch.setenv("GROQ_API_KEY", API_KEY)

        class RateLimitError(Exception):
            status_code = 429
            response = SimpleNamespace(headers={"retry-after": "1000"})

        client, request = client_with_results(RateLimitError(), completion())
        sleeps = []
        backend = GroqBackend(
            client_factory=lambda **kwargs: client,
            sleep=sleeps.append,
        )

        result = backend.generate_json("prompt", timeout_seconds=3, options={})

        assert result.text == '{"title": "Example"}'
        assert request.call_count == 2
        assert sleeps == [GroqBackend.MAX_RETRY_DELAY_SECONDS]  # Capped at 5.0

    def test_429_exhausted_retries_reported_clearly(self, monkeypatch):
        """Test that exhausted 429 retries are reported with clear error message."""
        monkeypatch.setenv("GROQ_API_KEY", API_KEY)

        class RateLimitError(Exception):
            status_code = 429
            response = SimpleNamespace(headers={"retry-after": "1"})

        client, request = client_with_results(RateLimitError(), RateLimitError())
        backend = GroqBackend(
            client_factory=lambda **kwargs: client,
            sleep=lambda _: None,
        )

        with pytest.raises(GroqRateLimitError) as error:
            backend.generate_json("prompt", timeout_seconds=3, options={})

        assert request.call_count == 2
        assert error.value.attempts == 2
        assert error.value.retry_after == "1"
        assert "rate limit reached after 2 attempt" in str(error.value)
        assert "Retry-After: 1" in str(error.value)
        assert API_KEY not in str(error.value)

    def test_429_with_retry_after_exposes_header_in_error(self, monkeypatch):
        """Test that Retry-After header is exposed in final error when retries exhausted."""
        monkeypatch.setenv("GROQ_API_KEY", API_KEY)

        class RateLimitError(Exception):
            status_code = 429
            response = SimpleNamespace(headers={"Retry-After": "7"})

        client, _ = client_with_results(RateLimitError(), RateLimitError())
        backend = GroqBackend(
            client_factory=lambda **kwargs: client,
            sleep=lambda _: None,
        )

        with pytest.raises(GroqRateLimitError) as error:
            backend.generate_json("prompt", timeout_seconds=3, options={})

        assert error.value.retry_after == "7"
        assert "Retry-After: 7" in str(error.value)

    def test_429_without_retry_after_error_message(self, monkeypatch):
        """Test error message when 429 is exhausted without Retry-After header."""
        monkeypatch.setenv("GROQ_API_KEY", API_KEY)

        class RateLimitError(Exception):
            status_code = 429
            response = SimpleNamespace(headers={})

        client, _ = client_with_results(RateLimitError(), RateLimitError())
        backend = GroqBackend(
            client_factory=lambda **kwargs: client,
            sleep=lambda _: None,
        )

        with pytest.raises(GroqRateLimitError) as error:
            backend.generate_json("prompt", timeout_seconds=3, options={})

        assert error.value.retry_after is None
        assert "rate limit reached after 2 attempt" in str(error.value)
        assert "Retry-After" not in str(error.value)

    def test_429_case_insensitive_retry_after_header(self, monkeypatch):
        """Test that Retry-After header is matched case-insensitively."""
        monkeypatch.setenv("GROQ_API_KEY", API_KEY)

        class RateLimitError(Exception):
            status_code = 429
            response = SimpleNamespace(headers={"Retry-After": "3"})

        client, request = client_with_results(RateLimitError(), completion())
        sleeps = []
        backend = GroqBackend(
            client_factory=lambda **kwargs: client,
            sleep=sleeps.append,
        )

        result = backend.generate_json("prompt", timeout_seconds=3, options={})

        assert result.text == '{"title": "Example"}'
        assert request.call_count == 2
        assert sleeps == [3.0]  # Uses server-provided delay

    def test_429_does_not_affect_timeout_handling(self, monkeypatch):
        """Test that 429 retry logic does not interfere with timeout handling."""
        monkeypatch.setenv("GROQ_API_KEY", API_KEY)

        class RateLimitError(Exception):
            status_code = 429
            response = SimpleNamespace(headers={"retry-after": "1"})

        class TimeoutError(Exception):
            pass

        # First request: 429, second request: timeout
        client, request = client_with_results(RateLimitError(), TimeoutError())
        sleeps = []
        backend = GroqBackend(
            client_factory=lambda **kwargs: client,
            sleep=sleeps.append,
        )

        with pytest.raises(GroqTimeoutError) as error:
            backend.generate_json("prompt", timeout_seconds=3, options={})

        assert request.call_count == 2
        assert sleeps == [1.0]  # Retry delay applied before timeout
        assert error.value.attempts == 2

    def test_429_retry_respects_max_retries_config(self, monkeypatch):
        """Test that 429 retry respects configured max_retries."""
        monkeypatch.setenv("GROQ_API_KEY", API_KEY)

        class RateLimitError(Exception):
            status_code = 429
            response = SimpleNamespace(headers={})

        # With max_retries=0, should not retry
        client, request = client_with_results(RateLimitError())
        backend = GroqBackend(
            client_factory=lambda **kwargs: client,
            max_retries=0,
        )

        with pytest.raises(GroqRateLimitError) as error:
            backend.generate_json("prompt", timeout_seconds=3, options={})

        assert request.call_count == 1
        assert error.value.attempts == 1


class TestGroqErrorDetailPreservationBatch7:
    """Batch 7: Test preservation of Groq API error details for diagnostics."""

    def test_http_400_with_provider_error_message(self, monkeypatch):
        """Test that HTTP 400 with provider error message is preserved."""
        monkeypatch.setenv("GROQ_API_KEY", API_KEY)

        class BadRequestError(Exception):
            status_code = 400
            body = {
                "error": {
                    "message": "Model parameter is invalid for this request.",
                }
            }

        client, _ = client_with_results(BadRequestError())
        backend = GroqBackend(
            client_factory=lambda **kwargs: client,
            max_retries=0,
        )

        with pytest.raises(GroqAPIError) as error:
            backend.generate_json("prompt", timeout_seconds=3, options={})

        assert error.value.status_code == 400
        assert error.value.detail == "Model parameter is invalid for this request."
        assert "400" in str(error.value)

    def test_structured_output_fallback_attempted_and_failed(self, monkeypatch):
        """Test that fallback is attempted but still fails."""
        monkeypatch.setenv("GROQ_API_KEY", API_KEY)

        class SchemaValidationError(Exception):
            status_code = 400

            def __init__(self):
                super().__init__("Failed to generate JSON. Adjust your prompt.")
                self.body = {"error": {"message": "Failed to generate JSON. Adjust your prompt."}}

        class JsonModeError(Exception):
            status_code = 400
            body = {"error": {"message": "Invalid request format."}}

        client, request = client_with_results(SchemaValidationError(), JsonModeError())
        backend = GroqBackend(
            client_factory=lambda **kwargs: client,
            max_retries=0,
        )

        with pytest.raises(GroqAPIError) as error:
            backend.generate_json("prompt", timeout_seconds=3, options={})

        assert request.call_count == 2
        assert error.value.status_code == 400
        assert error.value.detail == "Invalid request format."

    def test_successful_fallback_records_metadata(self, monkeypatch):
        """Test that successful fallback is recorded in metadata."""
        monkeypatch.setenv("GROQ_API_KEY", API_KEY)

        class SchemaValidationError(Exception):
            status_code = 400

            def __init__(self):
                super().__init__("Failed to generate JSON. Adjust your prompt.")
                self.body = {"error": {"message": "Failed to generate JSON. Adjust your prompt."}}

        client, request = client_with_results(SchemaValidationError(), completion())
        backend = GroqBackend(client_factory=lambda **kwargs: client)

        result = backend.generate_json("prompt", timeout_seconds=3, options={})

        assert request.call_count == 2
        assert result.metadata["response_format"] == "json_object"
        assert result.metadata["used_json_mode_fallback"] is True

    def test_missing_provider_error_detail_handling(self, monkeypatch):
        """Test handling of missing provider error detail."""
        monkeypatch.setenv("GROQ_API_KEY", API_KEY)

        class BadRequestError(Exception):
            status_code = 400
            body = {}  # No error message

        client, _ = client_with_results(BadRequestError())
        backend = GroqBackend(
            client_factory=lambda **kwargs: client,
            max_retries=0,
        )

        with pytest.raises(GroqAPIError) as error:
            backend.generate_json("prompt", timeout_seconds=3, options={})

        assert error.value.status_code == 400
        assert error.value.detail is None

    def test_malformed_provider_error_detail_handling(self, monkeypatch):
        """Test handling of malformed provider error detail."""
        monkeypatch.setenv("GROQ_API_KEY", API_KEY)

        class BadRequestError(Exception):
            status_code = 400
            body = {"error": 123}  # Not a dict, not a string

        client, _ = client_with_results(BadRequestError())
        backend = GroqBackend(
            client_factory=lambda **kwargs: client,
            max_retries=0,
        )

        with pytest.raises(GroqAPIError) as error:
            backend.generate_json("prompt", timeout_seconds=3, options={})

        assert error.value.status_code == 400
        assert error.value.detail is None

    def test_secret_redaction_in_error_detail(self, monkeypatch):
        """Test that secrets are redacted from error details."""
        monkeypatch.setenv("GROQ_API_KEY", API_KEY)

        class BadRequestError(Exception):
            status_code = 400
            body = {
                "error": {
                    "message": f"Authentication failed for key {API_KEY}",
                }
            }

        client, _ = client_with_results(BadRequestError())
        backend = GroqBackend(
            client_factory=lambda **kwargs: client,
            max_retries=0,
        )

        with pytest.raises(GroqAPIError) as error:
            backend.generate_json("prompt", timeout_seconds=3, options={})

        assert API_KEY not in error.value.detail
        assert "[REDACTED]" in error.value.detail

    def test_error_detail_length_bounded(self, monkeypatch):
        """Test that error detail is bounded to 500 characters."""
        monkeypatch.setenv("GROQ_API_KEY", API_KEY)

        long_message = "A" * 600
        class BadRequestError(Exception):
            status_code = 400
            body = {"error": {"message": long_message}}

        client, _ = client_with_results(BadRequestError())
        backend = GroqBackend(
            client_factory=lambda **kwargs: client,
            max_retries=0,
        )

        with pytest.raises(GroqAPIError) as error:
            backend.generate_json("prompt", timeout_seconds=3, options={})

        assert len(error.value.detail) <= 500


class TestPaperAnalyzerErrorDiagnosticsBatch7:
    """Batch 7: Test that PaperAnalyzer preserves error diagnostics."""

    def test_analyzer_preserves_groq_error_status_code(self, monkeypatch):
        """Test that Groq error status code is preserved in analyzer diagnostics."""
        monkeypatch.setenv("GROQ_API_KEY", API_KEY)

        class BadRequestError(Exception):
            status_code = 400
            body = {"error": {"message": "Invalid model parameter."}}

        client, _ = client_with_results(BadRequestError())
        backend = GroqBackend(
            client_factory=lambda **kwargs: client,
            max_retries=0,
        )
        analyzer = PaperAnalyzer(
            Mock(),
            Mock(),
            Mock(),
            llm_backend=backend,
            model_name=backend.model_name,
        )

        with pytest.raises(ExtractionError) as error:
            analyzer._extract_structured_json([{"text": "source excerpt"}])

        assert error.value.diagnostics["error_status_code"] == 400

    def test_analyzer_preserves_groq_error_detail(self, monkeypatch):
        """Test that Groq error detail is preserved in analyzer diagnostics."""
        monkeypatch.setenv("GROQ_API_KEY", API_KEY)

        class BadRequestError(Exception):
            status_code = 400
            body = {"error": {"message": "Invalid model parameter."}}

        client, _ = client_with_results(BadRequestError())
        backend = GroqBackend(
            client_factory=lambda **kwargs: client,
            max_retries=0,
        )
        analyzer = PaperAnalyzer(
            Mock(),
            Mock(),
            Mock(),
            llm_backend=backend,
            model_name=backend.model_name,
        )

        with pytest.raises(ExtractionError) as error:
            analyzer._extract_structured_json([{"text": "source excerpt"}])

        assert error.value.diagnostics["error_detail"] == "Invalid model parameter."

    def test_analyzer_increments_fallback_calls_on_successful_fallback(
        self, monkeypatch
    ):
        """Test that fallback_calls is incremented when fallback succeeds."""
        monkeypatch.setenv("GROQ_API_KEY", API_KEY)

        class SchemaValidationError(Exception):
            status_code = 400

            def __init__(self):
                super().__init__("Failed to generate JSON. Adjust your prompt.")
                self.body = {"error": {"message": "Failed to generate JSON. Adjust your prompt."}}

        client, _ = client_with_results(SchemaValidationError(), completion())
        backend = GroqBackend(client_factory=lambda **kwargs: client)
        analyzer = PaperAnalyzer(
            Mock(),
            Mock(),
            Mock(),
            llm_backend=backend,
            model_name=backend.model_name,
        )

        # This should succeed with fallback
        result = analyzer._extract_structured_json([{"text": "source excerpt"}])

        assert analyzer._last_extraction_diagnostics["fallback_calls"] == 1
        assert result is not None
