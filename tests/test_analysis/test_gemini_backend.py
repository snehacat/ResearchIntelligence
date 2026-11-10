import json
from urllib.error import HTTPError
from unittest.mock import Mock

import pytest

from analysis.paper_analyzer import PaperAnalyzer
from analysis.gemini_backend import (
    GeminiBackend,
    GeminiBackendError,
    GeminiHTTPError,
)
from paper_analysis_errors import ExtractionError
from validation.step5.run_validation import parse_arguments


API_KEY = "test-secret-key-value"


@pytest.fixture
def paper_analyzer():
    return PaperAnalyzer(Mock(), Mock(), Mock())


def response_body(
    text=None,
    *,
    finish_reason="STOP",
    prompt_tokens=12,
    candidate_tokens=7,
):
    candidates = []
    if finish_reason is not None or text is not None:
        candidate = {}
        if finish_reason is not None:
            candidate["finishReason"] = finish_reason
        if text is not None:
            candidate["content"] = {"parts": [{"text": text}]}
        candidates.append(candidate)
    return json.dumps({
        "candidates": candidates,
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": candidate_tokens,
        },
    }).encode("utf-8")


def set_api_key(monkeypatch, key=API_KEY):
    monkeypatch.setenv("GEMINI_API_KEY", key)


def test_gemini_backend_success_sends_assumed_request_shape(monkeypatch):
    set_api_key(monkeypatch)
    captured = {}

    def fake_http_post(url, *, headers, body, timeout):
        captured.update(url=url, headers=headers, body=body, timeout=timeout)
        return response_body('{"title": "Paper"}')

    result = GeminiBackend("test-model", http_post=fake_http_post).generate_json(
        "Extract this paper",
        timeout_seconds=23,
        options={"num_predict": 321, "num_ctx": 8192},
    )

    assert captured["url"].endswith("/models/test-model:generateContent")
    assert captured["headers"]["x-goog-api-key"] == API_KEY
    assert captured["body"] == {
        "contents": [{"parts": [{"text": "Extract this paper"}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0,
            "maxOutputTokens": 321,
        },
    }
    assert captured["timeout"] == 23
    assert result.text == '{"title": "Paper"}'
    assert result.metadata["done_reason"] == "STOP"
    assert result.metadata["prompt_eval_count"] == 12
    assert result.metadata["eval_count"] == 7
    assert result.metadata["duration"] >= 0
    assert "num_ctx" not in captured["body"]["generationConfig"]


def test_gemini_backend_preserves_markdown_fenced_response(monkeypatch):
    set_api_key(monkeypatch)
    content = '```json\n{"title": "Paper"}\n```'
    backend = GeminiBackend(
        "test-model",
        http_post=lambda *args, **kwargs: response_body(content),
    )

    result = backend.generate_json(
        "prompt", timeout_seconds=5, options={}
    )

    assert result.text == content


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (response_body(None, finish_reason="SAFETY"), "SAFETY"),
        (response_body(None, finish_reason=None), None),
    ],
)
def test_gemini_backend_returns_empty_text_for_blocked_or_empty_candidate(
    monkeypatch, body, reason
):
    set_api_key(monkeypatch)
    backend = GeminiBackend(
        "test-model",
        http_post=lambda *args, **kwargs: body,
    )

    result = backend.generate_json("prompt", timeout_seconds=5, options={})

    assert result.text == ""
    assert result.metadata["done_reason"] == reason


def test_gemini_backend_429_includes_retry_after_without_secret(monkeypatch):
    set_api_key(monkeypatch)

    def fake_http_post(*args, **kwargs):
        raise HTTPError(
            "https://example.invalid",
            429,
            "rate limited",
            {"Retry-After": "17"},
            None,
        )

    backend = GeminiBackend("test-model", http_post=fake_http_post)
    with pytest.raises(GeminiHTTPError) as error:
        backend.generate_json("prompt", timeout_seconds=5, options={})

    assert error.value.status_code == 429
    assert error.value.retry_after == "17"
    assert "17" in str(error.value)
    assert API_KEY not in str(error.value)


def test_gemini_backend_500_raises_typed_http_error(monkeypatch):
    set_api_key(monkeypatch)

    def fake_http_post(*args, **kwargs):
        raise HTTPError(
            "https://example.invalid", 500, "server error", {}, None
        )

    with pytest.raises(GeminiHTTPError) as error:
        GeminiBackend("test-model", http_post=fake_http_post).generate_json(
            "prompt", timeout_seconds=5, options={}
        )

    assert error.value.status_code == 500
    assert API_KEY not in str(error.value)


def test_gemini_backend_missing_key_names_variable_only(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    backend = GeminiBackend("test-model", http_post=lambda *args, **kwargs: b"{}")

    with pytest.raises(GeminiBackendError) as error:
        backend.generate_json("prompt", timeout_seconds=5, options={})

    assert "GEMINI_API_KEY" in str(error.value)
    assert API_KEY not in str(error.value)


def test_echoed_api_key_is_redacted_from_result_and_analyzer_diagnostics(
    monkeypatch, paper_analyzer
):
    set_api_key(monkeypatch)
    backend = GeminiBackend(
        "test-model",
        http_post=lambda *args, **kwargs: response_body(
            API_KEY,
            finish_reason=API_KEY,
            prompt_tokens=API_KEY,
        ),
    )

    backend_result = backend.generate_json(
        "prompt", timeout_seconds=5, options={}
    )
    assert API_KEY not in repr(backend_result.metadata)
    paper_analyzer.llm_backend = backend

    with pytest.raises(ExtractionError) as error:
        paper_analyzer._extract_structured_json([{"text": "Paper"}])

    assert API_KEY not in str(error.value)
    assert API_KEY not in repr(error.value.diagnostics)
    assert "[REDACTED]" in error.value.diagnostics["raw_response"]
    assert API_KEY not in repr(paper_analyzer._last_extraction_diagnostics)


def test_gemini_timeout_maps_to_analyzer_timeout_handling(
    monkeypatch, paper_analyzer
):
    set_api_key(monkeypatch)

    def fake_http_post(*args, **kwargs):
        raise TimeoutError("transport timeout")

    paper_analyzer.llm_backend = GeminiBackend(
        "test-model", http_post=fake_http_post
    )

    with pytest.raises(ExtractionError, match="timed out after 90 seconds"):
        paper_analyzer._extract_structured_json([{"text": "Paper"}])


def test_validation_cli_requires_gemini_model_and_confirmation():
    with pytest.raises(SystemExit):
        parse_arguments(["--backend", "gemini", "--confirm-public-papers"])
    with pytest.raises(SystemExit):
        parse_arguments(["--backend", "gemini", "--model", "test-model"])

    args = parse_arguments([
        "--backend", "gemini", "--model", "test-model",
        "--confirm-public-papers", "--delay-seconds", "2",
    ])
    assert args.backend == "gemini"
    assert args.model == "test-model"
    assert args.delay_seconds == 2


def test_validation_cli_defaults_to_groq_and_ten_second_delay():
    with pytest.raises(SystemExit):
        parse_arguments([])

    args = parse_arguments([
        "--confirm-public-papers",
        "--confirm-groq-costs",
    ])

    assert args.backend == "groq"
    assert args.delay_seconds == 10
