"""Tests for persisted background analysis jobs."""

import os
import sqlite3
import subprocess
import sys
import threading

import pytest

import analysis.job_store as job_store_module
from analysis.analysis_job_manager import AnalysisJobManager
from analysis.job_store import (
    AnalysisJobStateError,
    AnalysisJobStore,
)


@pytest.fixture
def job_store(tmp_path):
    return AnalysisJobStore(str(tmp_path / "jobs.sqlite3"))


class FakeAnalyzer:
    def __init__(self, *, failure=None, observed_checkpoints=None, gate=None):
        self.failure = failure
        self.observed_checkpoints = observed_checkpoints
        self.gate = gate

    def analyze_paper(
        self,
        extraction_result,
        document_id,
        progress_callback=None,
        extracted_data=None,
        checkpoint_callback=None,
    ):
        if progress_callback:
            progress_callback("chunking", "Preparing test job.")
        if self.gate:
            self.gate.wait(timeout=2)
        if self.failure:
            raise RuntimeError(self.failure)
        if self.observed_checkpoints is not None:
            self.observed_checkpoints.append(extracted_data)
        if extracted_data is None:
            extracted_data = {"title": "Checkpointed title"}
            if checkpoint_callback:
                checkpoint_callback("llm_extracted", extracted_data)
        if progress_callback:
            progress_callback("validating", "Using checkpointed extraction.")
        return {
            "document_id": document_id,
            "title": extracted_data["title"],
            "source_pages": len(extraction_result.get("pages", [])),
            "evidence_citations": {"title": {"page": 1}},
        }


class TestAnalysisJobStore:
    @pytest.mark.skipif(os.name != "nt", reason="Windows process API behavior")
    def test_process_is_alive_for_current_process(self):
        assert AnalysisJobStore._process_is_alive(os.getpid()) is True

    @pytest.mark.skipif(os.name != "nt", reason="Windows process API behavior")
    def test_process_is_not_alive_after_subprocess_exits(self):
        process = subprocess.Popen([sys.executable, "-c", "pass"])
        process.wait(timeout=10)

        assert AnalysisJobStore._process_is_alive(process.pid) is False

    @pytest.mark.skipif(os.name != "nt", reason="Windows process API behavior")
    def test_process_liveness_check_does_not_call_os_kill(self, monkeypatch):
        def unexpected_kill(*args, **kwargs):
            raise AssertionError("os.kill must not be used on Windows")

        monkeypatch.setattr(job_store_module.os, "kill", unexpected_kill)

        assert AnalysisJobStore._process_is_alive(os.getpid()) is True

    def test_round_trips_source_result_and_status(self, job_store):
        job = job_store.create_job(
            {"pages": [{"page_number": 1, "text": "Paper text"}]},
            "paper-1",
        )

        assert job["status"] == "queued"
        assert "extraction_result" not in job
        assert job["result"] is None

        assert job_store.claim_job(job["job_id"])
        job_store.update_progress(job["job_id"], "extracting", "Running model.")
        job_store.save_extracted_data(job["job_id"], {"title": "Example"})
        job_store.complete_job(job["job_id"], {"title": "Example"})
        completed = job_store.get_job(job["job_id"], include_internal=True)

        assert completed["status"] == "completed"
        assert completed["result"] == {"title": "Example"}
        assert completed["extracted_data"] == {"title": "Example"}
        assert completed["extraction_result"]["pages"][0]["text"] == "Paper text"
        summaries = job_store.list_jobs()
        assert "extraction_result" not in summaries[0]
        assert "result" not in summaries[0]

    def test_rejects_nonserializable_input_and_invalid_limits(self, job_store):
        with pytest.raises(ValueError, match="JSON-serializable"):
            job_store.create_job({"invalid": object()}, "paper")
        with pytest.raises(ValueError, match="document_id"):
            job_store.create_job({}, " ")
        with pytest.raises(ValueError, match="limit"):
            job_store.list_jobs(0)

    def test_recovers_running_jobs_and_preserves_extraction_checkpoint(self, job_store):
        job = job_store.create_job({"pages": []}, "paper")
        assert job_store.claim_job(job["job_id"])
        job_store.save_extracted_data(job["job_id"], {"title": "Saved title"})
        assert job_store.recover_interrupted_jobs() == 0

        with sqlite3.connect(job_store.database_path) as connection:
            connection.execute(
                "UPDATE analysis_jobs SET worker_pid = -1 WHERE job_id = ?",
                (job["job_id"],),
            )

        assert job_store.recover_interrupted_jobs() == 1
        recovered = job_store.get_job(job["job_id"], include_internal=True)

        assert recovered["status"] == "queued"
        assert recovered["extracted_data"] == {"title": "Saved title"}
        assert "process interruption" in recovered["message"]

    def test_delete_is_explicit_and_disallowed_while_running(self, job_store):
        job = job_store.create_job({}, "paper")
        assert job_store.claim_job(job["job_id"])

        with pytest.raises(AnalysisJobStateError, match="currently running"):
            job_store.delete_job(job["job_id"])

        job_store.fail_job(job["job_id"], "test failure")
        job_store.delete_job(job["job_id"])
        assert job_store.get_job(job["job_id"]) is None


class TestAnalysisJobManager:
    def test_background_job_completes_and_persists_checkpoint(self, job_store):
        manager = AnalysisJobManager(
            job_store,
            analyzer_factory=lambda: FakeAnalyzer(),
        )
        try:
            job = manager.submit({"pages": []}, "paper-2")
            completed = manager.wait(job["job_id"], timeout=3)

            assert completed["status"] == "completed"
            assert completed["result"]["title"] == "Checkpointed title"
            assert "extracted_data" not in completed
            persisted = job_store.get_job(job["job_id"], include_internal=True)
            assert persisted["extracted_data"] == {
                "title": "Checkpointed title"
            }
        finally:
            manager.close()

    def test_restart_resumes_from_saved_llm_checkpoint(self, job_store):
        job = job_store.create_job({"pages": []}, "paper-3")
        assert job_store.claim_job(job["job_id"])
        job_store.save_extracted_data(
            job["job_id"], {"title": "Saved before interruption"}
        )
        with sqlite3.connect(job_store.database_path) as connection:
            connection.execute(
                "UPDATE analysis_jobs SET worker_pid = -1 WHERE job_id = ?",
                (job["job_id"],),
            )

        observed = []
        manager = AnalysisJobManager(
            job_store,
            analyzer_factory=lambda: FakeAnalyzer(
                observed_checkpoints=observed
            ),
        )
        try:
            resumed = manager.wait(job["job_id"], timeout=3)
            assert resumed["status"] == "completed"
            assert observed == [{"title": "Saved before interruption"}]
            assert resumed["attempt_count"] == 2
        finally:
            manager.close()

    def test_failures_are_reported_and_retryable(self, job_store):
        attempts = []

        def analyzer_factory():
            attempts.append(len(attempts) + 1)
            if len(attempts) == 1:
                return FakeAnalyzer(failure="model unavailable")
            return FakeAnalyzer()

        manager = AnalysisJobManager(job_store, analyzer_factory)
        try:
            job = manager.submit({"pages": []}, "paper-4")
            failed = manager.wait(job["job_id"], timeout=3)
            assert failed["status"] == "failed"
            assert "model unavailable" in failed["error"]

            manager.retry(job["job_id"])
            retried = manager.wait(job["job_id"], timeout=3)
            assert retried["status"] == "completed"
            assert retried["attempt_count"] == 2
        finally:
            manager.close()

    def test_running_job_cannot_be_deleted(self, job_store):
        gate = threading.Event()
        manager = AnalysisJobManager(
            job_store,
            analyzer_factory=lambda: FakeAnalyzer(gate=gate),
        )
        try:
            job = manager.submit({"pages": []}, "paper-5")
            running = threading.Event()

            def wait_until_running():
                for _ in range(100):
                    current = manager.get_job(job["job_id"])
                    if current and current["status"] == "running":
                        running.set()
                        return
                    threading.Event().wait(0.01)

            watcher = threading.Thread(target=wait_until_running)
            watcher.start()
            assert running.wait(timeout=2)
            watcher.join(timeout=1)
            with pytest.raises(AnalysisJobStateError):
                manager.delete(job["job_id"])
            gate.set()
            assert manager.wait(job["job_id"], timeout=3)["status"] == "completed"
        finally:
            gate.set()
            manager.close()
