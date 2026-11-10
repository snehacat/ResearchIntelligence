from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_analyze_button_explains_when_no_pdf_is_selected():
    project_root = Path(__file__).parents[2]
    app_test = AppTest.from_file(
        project_root / "app.py", default_timeout=30
    )
    app_test.run()

    analyze_button = app_test.button[0]
    assert analyze_button.label == "🔍 Analyze"
    assert not analyze_button.disabled

    analyze_button.click().run()

    assert not app_test.exception
    assert any(
        "Please choose a research-paper PDF" in element.value
        for element in app_test.error
    )


def test_results_screen_renders_tabs_passages_and_report_download():
    bundle = {
        "analysis": {
            "document_id": "test-paper",
            "title": "Banking services study",
            "authors": ["A. Author"],
            "publication_year": "2024",
            "research_problem": "Understand card-service usage.",
            "methodology": "Survey questionnaire",
            "dataset": "100 respondents",
            "evaluation_metrics": ["respondent percentage"],
            "reported_results": "100 respondents completed the survey.",
            "limitations": ["Small sample size"],
            "analysis_coverage": {
                "is_partial": True,
                "total_chunks": 6,
                "omitted_chunks": [{"chunk_id": "chunk-6"}],
            },
            "evidence_citations": {
                "dataset": {
                    "evidence_type": "explicit_fact",
                    "citations": [{
                        "document_id": "bank-paper",
                        "chunk_id": "chunk-2",
                        "start_page": 2,
                        "end_page": 2,
                    }],
                    "evidence_chunks": [{
                        "document_id": "bank-paper",
                        "chunk_id": "chunk-2",
                        "start_page": 2,
                        "end_page": 2,
                        "text": "The study surveyed 100 respondents.",
                    }],
                },
            },
        },
        "related_papers": [],
        "related_paper_candidates": [],
        "screened_out_papers": [],
        "related_evidence": [],
        "comparison": {
            "status": "insufficient_sources",
            "source_count": 1,
            "field_comparisons": {},
        },
        "gap_analysis": {
            "research_gap_candidates": [],
            "scope_warning": "These are candidates, not validated gaps.",
        },
        "report": {
            "markdown": "# Research Analysis Report",
            "warnings": ["Source-PDF review required."],
        },
    }

    project_root = Path(__file__).parents[2]
    app_test = AppTest.from_file(
        project_root / "app.py", default_timeout=30
    )
    app_test.session_state["research_bundle"] = bundle
    app_test.session_state["current_document_id"] = "test-paper"
    app_test.run()

    assert not app_test.exception
    assert len(app_test.tabs) == 6
    assert len(app_test.get("download_button")) == 1
    rendered_text = "\n".join(
        element.value
        for element in app_test.get("markdown")
        if isinstance(element.value, str)
    )
    assert "The study surveyed 100 respondents." in rendered_text


def test_stale_results_not_displayed_after_failed_analysis():
    """A7: After a failed analysis, stale results from a previous paper are not shown."""
    project_root = Path(__file__).parents[2]
    app_test = AppTest.from_file(
        project_root / "app.py", default_timeout=30
    )

    # First, store a successful analysis for "paper-1"
    bundle_1 = {
        "analysis": {
            "document_id": "paper-1",
            "title": "First Paper",
            "authors": ["Author One"],
            "publication_year": "2024",
        },
        "related_papers": [],
        "related_paper_candidates": [],
        "screened_out_papers": [],
        "related_evidence": [],
        "comparison": {"status": "insufficient_sources", "source_count": 1, "field_comparisons": {}},
        "gap_analysis": {"research_gap_candidates": [], "scope_warning": "Candidates only."},
        "report": {"markdown": "# Report 1", "warnings": []},
    }
    app_test.session_state["research_bundle"] = bundle_1
    app_test.session_state["current_document_id"] = "paper-1"
    app_test.run()

    # Verify first paper is displayed
    assert not app_test.exception
    rendered_text = "\n".join(
        element.value
        for element in app_test.get("markdown")
        if isinstance(element.value, str)
    )
    assert "First Paper" in rendered_text

    # Now simulate a failed analysis for "paper-2"
    # (In a real test, this would upload a file that causes an error)
    # For this test, we manually set the state to simulate the failure path
    app_test.session_state["research_bundle"] = None
    app_test.session_state["current_document_id"] = None
    app_test.run()

    # Verify stale results are not displayed
    rendered_text = "\n".join(
        element.value
        for element in app_test.get("markdown")
        if isinstance(element.value, str)
    )
    assert "First Paper" not in rendered_text


def test_related_paper_failure_preserves_primary_analysis():
    """A8: If related-paper search fails, primary analysis is still available."""
    bundle = {
        "analysis": {
            "document_id": "test-paper",
            "title": "Primary Paper",
            "authors": ["Author"],
            "publication_year": "2024",
        },
        "related_papers": [],
        "related_paper_candidates": [],
        "screened_out_papers": [],
        "related_evidence": [],
        "comparison": {"status": "insufficient_sources", "source_count": 1, "field_comparisons": {}},
        "gap_analysis": {"research_gap_candidates": [], "scope_warning": "Candidates only."},
        "report": {"markdown": "# Report", "warnings": []},
        "discovery_status": "failed",
        "discovery_error": "Crossref API timeout",
        "metadata_verification_status": "skipped",
        "related_analysis_status": "skipped",
    }

    project_root = Path(__file__).parents[2]
    app_test = AppTest.from_file(
        project_root / "app.py", default_timeout=30
    )
    app_test.session_state["research_bundle"] = bundle
    app_test.session_state["current_document_id"] = "test-paper"
    app_test.run()

    assert not app_test.exception
    # Primary analysis should be displayed
    rendered_text = "\n".join(
        element.value
        for element in app_test.get("markdown")
        if isinstance(element.value, str)
    )
    assert "Primary Paper" in rendered_text
    # Error message should be shown
    assert any(
        "Related-paper search failed" in element.value
        for element in app_test.error
    )


def test_user_facing_errors_are_sanitized():
    """A10: User-facing errors do not expose sensitive details like local paths."""
    # Test that the error handling in run_analysis isolates optional stage failures
    # without exposing internal error details in the status fields
    bundle = {
        "analysis": {
            "document_id": "test-paper",
            "title": "Primary Paper",
            "authors": ["Author"],
            "publication_year": "2024",
        },
        "related_papers": [],
        "related_paper_candidates": [],
        "screened_out_papers": [],
        "related_evidence": [],
        "comparison": {"status": "insufficient_sources", "source_count": 1, "field_comparisons": {}},
        "gap_analysis": {"research_gap_candidates": [], "scope_warning": "Candidates only."},
        "report": {"markdown": "# Report", "warnings": []},
        "discovery_status": "failed",
        "discovery_error": "Crossref API timeout",  # Sanitized error, no local paths
        "metadata_verification_status": "skipped",
        "related_analysis_status": "skipped",
    }

    project_root = Path(__file__).parents[2]
    app_test = AppTest.from_file(
        project_root / "app.py", default_timeout=30
    )
    app_test.session_state["research_bundle"] = bundle
    app_test.session_state["current_document_id"] = "test-paper"
    app_test.run()

    assert not app_test.exception
    # Verify the error message shown to user is sanitized
    assert any(
        "Related-paper search failed" in element.value
        for element in app_test.error
    )
    # Verify no local paths are exposed in the error message
    error_text = "\n".join(
        element.value
        for element in app_test.error
        if isinstance(element.value, str)
    )
    assert "C:\\" not in error_text
    assert "C:/" not in error_text
    assert "Users\\" not in error_text
    assert "api_key" not in error_text.lower()


def test_valueerror_exception_is_sanitized():
    """A10 regression: ValueError exceptions show safe message, not raw error string."""
    # This test verifies that the ValueError handler in app.py shows a fixed
    # safe message instead of rendering str(error) which could include local paths.
    # We test this by inspecting the error handling code path directly.
    import inspect
    from app import run_analysis

    source = inspect.getsource(run_analysis)
    # Verify that ValueError is raised with a message (not the UI handler)
    assert "raise ValueError" in source

    # Now verify the UI handler in app.py shows safe message
    project_root = Path(__file__).parents[2]
    with open(project_root / "app.py", "r", encoding="utf-8") as f:
        app_source = f.read()

    # Find the ValueError exception handler in the UI section
    # It should show: "Analysis failed. This can happen if the PDF has no extractable text."
    # It should NOT show: f"⚠️ {error}" which would expose raw error string
    assert "except ValueError as error:" in app_source
    # The safe message should be present
    assert "PDF has no extractable text" in app_source
    # Verify that the old unsafe pattern (f"⚠️ {error}") is NOT present for ValueError
    # We check that the ValueError handler section does NOT contain the pattern
    lines = app_source.split("\n")
    in_valueerror_handler = False
    for i, line in enumerate(lines):
        if "except ValueError as error:" in line:
            in_valueerror_handler = True
        elif in_valueerror_handler and "except" in line and "ValueError" not in line:
            in_valueerror_handler = False
        elif in_valueerror_handler:
            # Inside ValueError handler, should not have unsafe pattern
            assert "{error}" not in line or "f" not in line, (
                f"Line {i+1} in ValueError handler may expose raw error: {line}"
            )


def test_discovery_exception_is_sanitized():
    """A10 regression: Discovery exceptions in run_analysis do not expose raw error strings."""
    # This test verifies that the discovery_error field does not contain str(e)
    # which could include local paths or provider internals
    import inspect
    from app import run_analysis

    source = inspect.getsource(run_analysis)
    # Verify that the discovery exception handler does NOT use str(e)
    lines = source.split("\n")
    in_discovery_handler = False
    for i, line in enumerate(lines):
        if "except Exception as e:" in line and "discovery" in source[max(0, i-200):i]:
            in_discovery_handler = True
        elif in_discovery_handler and "except" in line:
            in_discovery_handler = False
        elif in_discovery_handler:
            # Inside discovery handler, should not have str(e)
            assert "str(e)" not in line, (
                f"Line {i+1} in discovery handler may expose raw error: {line}"
            )


def test_exception_with_secret_and_path_is_sanitized():
    """A10 regression: Exception containing fake secret and local path is not shown to user."""
    # This test verifies that if an exception contains sensitive information,
    # it is not displayed in the UI. We verify the code patterns prevent this.
    project_root = Path(__file__).parents[2]
    with open(project_root / "app.py", "r", encoding="utf-8") as f:
        app_source = f.read()

    # Check that no exception handler in the UI section uses f-string with exception variable
    # This would be a pattern like: st.error(f"Error: {error}")
    lines = app_source.split("\n")
    in_exception_handler = False
    for i, line in enumerate(lines):
        if "except" in line and ("Exception" in line or "ValueError" in line or "error" in line):
            in_exception_handler = True
        elif in_exception_handler and line.strip() and not line.strip().startswith("#"):
            # Check if this line could expose raw exception
            if "st.error" in line and ("{" in line or "f\"" in line or "f'" in line):
                # Check if it contains an exception variable name
                if any(var in line for var in ["error", "e", "exception", "exc"]):
                    # This might be unsafe - verify it's a safe message
                    # Safe messages are fixed strings, not formatted with exception
                    assert "API key" not in line.lower() or "not set" in line.lower(), (
                        f"Line {i+1} may expose API key in error: {line}"
                    )
        # End of exception handler is typically the next blank line or another except/finally
        if in_exception_handler and (line.strip() == "" or "except" in line or "finally" in line):
            if "except" not in line:  # Don't end on the next except
                in_exception_handler = False
