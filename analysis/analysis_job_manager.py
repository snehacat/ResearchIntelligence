"""Single-worker background execution and restart recovery for analysis jobs."""

from concurrent.futures import Future, ThreadPoolExecutor
from threading import RLock
from typing import Any, Callable, Dict, Mapping, Optional, Protocol

from analysis.job_store import AnalysisJobStore


class _AnalyzablePaper(Protocol):
    def analyze_paper(
        self,
        extraction_result: Dict[str, Any],
        document_id: str,
        progress_callback: Optional[Callable[[str, str], None]] = None,
        extracted_data: Optional[Dict[str, Any]] = None,
        checkpoint_callback: Optional[
            Callable[[str, Dict[str, Any]], None]
        ] = None,
    ) -> Dict[str, Any]: ...


class AnalysisJobManager:
    """Run persisted paper analyses off-thread with one model call at a time."""

    def __init__(
        self,
        store: AnalysisJobStore,
        analyzer_factory: Callable[[], _AnalyzablePaper],
        auto_resume: bool = True,
    ) -> None:
        if not callable(analyzer_factory):
            raise ValueError("analyzer_factory must be callable")
        self.store = store
        self.analyzer_factory = analyzer_factory
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="paper-analysis",
        )
        self._futures: Dict[str, Future[None]] = {}
        self._lock = RLock()
        self._closed = False

        if auto_resume:
            self.store.recover_interrupted_jobs()
            for job_id in self.store.list_queued_job_ids():
                self._schedule(job_id)

    def submit(
        self,
        extraction_result: Mapping[str, Any],
        document_id: str,
    ) -> Dict[str, Any]:
        self._ensure_open()
        job = self.store.create_job(extraction_result, document_id)
        self._schedule(job["job_id"])
        return job

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.store.get_job(job_id)

    def list_jobs(self, limit: int = 100) -> list[Dict[str, Any]]:
        return self.store.list_jobs(limit=limit)

    def retry(self, job_id: str) -> Dict[str, Any]:
        self._ensure_open()
        self.store.retry_job(job_id)
        self._schedule(job_id)
        job = self.store.get_job(job_id)
        if job is None:
            raise RuntimeError(f"Retried analysis job {job_id} could not be read")
        return job

    def delete(self, job_id: str) -> None:
        self.store.delete_job(job_id)

    def wait(self, job_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for one job up to the caller's own UI/test budget."""
        with self._lock:
            future = self._futures.get(job_id)
        if future is not None:
            future.result(timeout=timeout)
        job = self.store.get_job(job_id)
        if job is None:
            raise KeyError(f"Analysis job {job_id} does not exist")
        return job

    def close(self, wait: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("AnalysisJobManager is closed")

    def _schedule(self, job_id: str) -> None:
        self._ensure_open()
        with self._lock:
            current = self._futures.get(job_id)
            if current is not None and not current.done():
                return
            future = self._executor.submit(self._run_job, job_id)
            self._futures[job_id] = future
            future.add_done_callback(
                lambda completed, key=job_id: self._discard_future(key, completed)
            )

    def _discard_future(self, job_id: str, future: Future[None]) -> None:
        with self._lock:
            if self._futures.get(job_id) is future:
                del self._futures[job_id]

    def _run_job(self, job_id: str) -> None:
        try:
            if not self.store.claim_job(job_id):
                return
            job = self.store.get_job(job_id, include_internal=True)
            if job is None:
                raise RuntimeError(f"Claimed analysis job {job_id} no longer exists")

            analyzer = self.analyzer_factory()
            result = analyzer.analyze_paper(
                job["extraction_result"],
                job["document_id"],
                progress_callback=lambda stage, message: self.store.update_progress(
                    job_id, stage, message
                ),
                extracted_data=job["extracted_data"],
                checkpoint_callback=lambda stage, data: self._save_checkpoint(
                    job_id, stage, data
                ),
            )
            self.store.complete_job(job_id, result)
        except Exception as error:
            job = self.store.get_job(job_id)
            if job is not None and job["status"] == "running":
                message = f"{type(error).__name__}: {error}"
                self.store.fail_job(job_id, message)

    def _save_checkpoint(
        self, job_id: str, stage: str, data: Dict[str, Any]
    ) -> None:
        if stage != "llm_extracted":
            raise ValueError(f"Unsupported analysis checkpoint stage: {stage}")
        self.store.save_extracted_data(job_id, data)
        self.store.update_progress(
            job_id,
            "extraction_checkpointed",
            "Structured LLM output saved; later stages can resume without rerunning it.",
        )
