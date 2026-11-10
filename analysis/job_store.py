"""SQLite persistence for background paper-analysis jobs."""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Mapping, Optional
from uuid import uuid4


class AnalysisJobStoreError(Exception):
    """Raised when analysis job persistence fails."""


class AnalysisJobStateError(AnalysisJobStoreError):
    """Raised when a job cannot make the requested state transition."""


class AnalysisJobStore:
    """Persist job inputs, checkpoints, status, and results in local SQLite."""

    MAX_LIST_LIMIT = 1000
    MAX_EXTRACTION_JSON_BYTES = 25 * 1024 * 1024

    def __init__(self, database_path: Optional[str] = None) -> None:
        self.database_path = Path(
            self._default_database_path()
            if database_path is None
            else database_path
        )
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    @staticmethod
    def _default_database_path() -> Path:
        import os

        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            data_directory = Path(local_app_data) / "ResearchIntelligence"
        else:
            data_directory = Path.home() / ".local" / "share" / "researchintelligence"
        return data_directory / "analysis_jobs.sqlite3"

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.database_path),
            timeout=10.0,
            isolation_level="IMMEDIATE",
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize_database(self) -> None:
        try:
            with self._connection() as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS analysis_jobs (
                        job_id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        status TEXT NOT NULL CHECK (
                            status IN ('queued', 'running', 'completed', 'failed')
                        ),
                        stage TEXT NOT NULL,
                        message TEXT NOT NULL,
                        extraction_json TEXT NOT NULL,
                        extracted_data_json TEXT,
                        result_json TEXT,
                        error TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        started_at TEXT,
                        completed_at TEXT,
                        worker_pid INTEGER,
                        attempt_count INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_analysis_jobs_created
                    ON analysis_jobs(created_at DESC)
                    """
                )
                columns = {
                    row["name"]
                    for row in connection.execute(
                        "PRAGMA table_info(analysis_jobs)"
                    ).fetchall()
                }
                if "worker_pid" not in columns:
                    connection.execute(
                        "ALTER TABLE analysis_jobs ADD COLUMN worker_pid INTEGER"
                    )
        except sqlite3.Error as error:
            raise AnalysisJobStoreError(
                f"Could not initialize analysis job database: {error}"
            ) from error

    @staticmethod
    def _encode_json(value: Any, label: str) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} must contain JSON-serializable values") from error

    @staticmethod
    def _decode_job(
        row: sqlite3.Row,
        include_internal: bool = False,
        include_result: bool = True,
    ) -> Dict[str, Any]:
        job = dict(row)
        extraction_json = job.pop("extraction_json")
        extracted_data_json = job.pop("extracted_data_json")
        result_json = job.pop("result_json")
        if include_internal:
            job["extraction_result"] = (
                json.loads(extraction_json) if extraction_json is not None else None
            )
            job["extracted_data"] = (
                json.loads(extracted_data_json)
                if extracted_data_json is not None
                else None
            )
        if include_result:
            job["result"] = (
                json.loads(result_json) if result_json is not None else None
            )
        return job

    def create_job(
        self,
        extraction_result: Mapping[str, Any],
        document_id: str,
    ) -> Dict[str, Any]:
        if not isinstance(extraction_result, Mapping):
            raise ValueError("extraction_result must be a mapping")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        extraction_json = self._encode_json(extraction_result, "extraction_result")
        if len(extraction_json.encode("utf-8")) > self.MAX_EXTRACTION_JSON_BYTES:
            raise ValueError(
                "extraction_result exceeds the 25 MiB persistence limit"
            )
        job_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO analysis_jobs (
                        job_id, document_id, status, stage, message,
                        extraction_json, created_at, updated_at
                    ) VALUES (?, ?, 'queued', 'queued', ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        document_id,
                        "Analysis job is queued.",
                        extraction_json,
                        now,
                        now,
                    ),
                )
        except sqlite3.Error as error:
            raise AnalysisJobStoreError(
                f"Could not create analysis job: {error}"
            ) from error
        job = self.get_job(job_id)
        if job is None:
            raise AnalysisJobStoreError("Created analysis job could not be read")
        return job

    def get_job(
        self,
        job_id: str,
        include_internal: bool = False,
    ) -> Optional[Dict[str, Any]]:
        try:
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT * FROM analysis_jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
            return (
                self._decode_job(row, include_internal=include_internal)
                if row is not None
                else None
            )
        except (sqlite3.Error, json.JSONDecodeError) as error:
            raise AnalysisJobStoreError(
                f"Could not read analysis job: {error}"
            ) from error

    def list_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= self.MAX_LIST_LIMIT
        ):
            raise ValueError(f"limit must be between 1 and {self.MAX_LIST_LIMIT}")
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    """
                    SELECT * FROM analysis_jobs
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            return [
                self._decode_job(row, include_result=False)
                for row in rows
            ]
        except (sqlite3.Error, json.JSONDecodeError) as error:
            raise AnalysisJobStoreError(
                f"Could not list analysis jobs: {error}"
            ) from error

    def list_queued_job_ids(self) -> List[str]:
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    "SELECT job_id FROM analysis_jobs WHERE status = 'queued'"
                ).fetchall()
            return [row["job_id"] for row in rows]
        except sqlite3.Error as error:
            raise AnalysisJobStoreError(
                f"Could not load queued analysis jobs: {error}"
            ) from error

    def recover_interrupted_jobs(self) -> int:
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connection() as connection:
                running = connection.execute(
                    "SELECT job_id, worker_pid FROM analysis_jobs "
                    "WHERE status = 'running'"
                ).fetchall()
                interrupted_ids = [
                    row["job_id"]
                    for row in running
                    if not self._process_is_alive(row["worker_pid"])
                ]
                for job_id in interrupted_ids:
                    connection.execute(
                        """
                        UPDATE analysis_jobs
                        SET status = 'queued',
                            stage = 'queued',
                            message = ?,
                            updated_at = ?,
                            worker_pid = NULL
                        WHERE job_id = ? AND status = 'running'
                        """,
                        (
                            "Recovered after process interruption; resuming from the "
                            "latest saved extraction checkpoint.",
                            now,
                            job_id,
                        ),
                    )
            return len(interrupted_ids)
        except sqlite3.Error as error:
            raise AnalysisJobStoreError(
                f"Could not recover interrupted analysis jobs: {error}"
            ) from error

    @staticmethod
    def _process_is_alive(process_id: Optional[int]) -> bool:
        if process_id is None or process_id <= 0:
            return False
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            process_query_limited_information = 0x1000
            still_active = 259
            error_access_denied = 5
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = (
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            )
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.GetExitCodeProcess.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(wintypes.DWORD),
            )
            kernel32.GetExitCodeProcess.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel32.CloseHandle.restype = wintypes.BOOL

            handle = kernel32.OpenProcess(
                process_query_limited_information,
                False,
                process_id,
            )
            if not handle:
                return ctypes.get_last_error() == error_access_denied

            try:
                exit_code = wintypes.DWORD()
                if not kernel32.GetExitCodeProcess(
                    handle, ctypes.byref(exit_code)
                ):
                    return False
                return exit_code.value == still_active
            finally:
                kernel32.CloseHandle(handle)

        try:
            os.kill(process_id, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False

    def claim_job(self, job_id: str) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connection() as connection:
                cursor = connection.execute(
                    """
                    UPDATE analysis_jobs
                    SET status = 'running',
                        stage = 'starting',
                        message = 'Analysis started.',
                        started_at = ?,
                        completed_at = NULL,
                        error = NULL,
                        updated_at = ?,
                        worker_pid = ?,
                        attempt_count = attempt_count + 1
                    WHERE job_id = ? AND status = 'queued'
                    """,
                    (now, now, os.getpid(), job_id),
                )
            return cursor.rowcount == 1
        except sqlite3.Error as error:
            raise AnalysisJobStoreError(
                f"Could not claim analysis job: {error}"
            ) from error

    def update_progress(self, job_id: str, stage: str, message: str) -> None:
        self._update_running_job(
            job_id,
            "stage = ?, message = ?, updated_at = ?",
            (stage, message, datetime.now(timezone.utc).isoformat()),
            "update progress for",
        )

    def save_extracted_data(
        self, job_id: str, extracted_data: Mapping[str, Any]
    ) -> None:
        extracted_json = self._encode_json(extracted_data, "extracted_data")
        self._update_running_job(
            job_id,
            "extracted_data_json = ?, updated_at = ?",
            (extracted_json, datetime.now(timezone.utc).isoformat()),
            "save extraction checkpoint for",
        )

    def complete_job(self, job_id: str, result: Mapping[str, Any]) -> None:
        result_json = self._encode_json(result, "result")
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connection() as connection:
                cursor = connection.execute(
                    """
                    UPDATE analysis_jobs
                    SET status = 'completed',
                        stage = 'complete',
                        message = 'Paper analysis completed.',
                        result_json = ?,
                        error = NULL,
                        completed_at = ?,
                        updated_at = ?,
                        worker_pid = NULL
                    WHERE job_id = ? AND status = 'running'
                    """,
                    (result_json, now, now, job_id),
                )
            if cursor.rowcount != 1:
                raise AnalysisJobStateError(
                    f"Analysis job {job_id} is not running"
                )
        except sqlite3.Error as error:
            raise AnalysisJobStoreError(
                f"Could not complete analysis job: {error}"
            ) from error

    def fail_job(self, job_id: str, error_message: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connection() as connection:
                cursor = connection.execute(
                    """
                    UPDATE analysis_jobs
                    SET status = 'failed',
                        stage = 'failed',
                        message = ?,
                        error = ?,
                        completed_at = ?,
                        updated_at = ?,
                        worker_pid = NULL
                    WHERE job_id = ? AND status = 'running'
                    """,
                    (error_message, error_message, now, now, job_id),
                )
            if cursor.rowcount != 1:
                raise AnalysisJobStateError(
                    f"Analysis job {job_id} is not running"
                )
        except sqlite3.Error as error:
            raise AnalysisJobStoreError(
                f"Could not fail analysis job: {error}"
            ) from error

    def retry_job(self, job_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connection() as connection:
                cursor = connection.execute(
                    """
                    UPDATE analysis_jobs
                    SET status = 'queued',
                        stage = 'queued',
                        message = 'Retry queued.',
                        result_json = NULL,
                        error = NULL,
                        completed_at = NULL,
                        updated_at = ?
                    WHERE job_id = ? AND status = 'failed'
                    """,
                    (now, job_id),
                )
            if cursor.rowcount != 1:
                raise AnalysisJobStateError(
                    f"Analysis job {job_id} is not in a retryable failed state"
                )
        except sqlite3.Error as error:
            raise AnalysisJobStoreError(
                f"Could not retry analysis job: {error}"
            ) from error

    def delete_job(self, job_id: str) -> None:
        try:
            with self._connection() as connection:
                cursor = connection.execute(
                    "DELETE FROM analysis_jobs WHERE job_id = ? AND status != 'running'",
                    (job_id,),
                )
            if cursor.rowcount != 1:
                raise AnalysisJobStateError(
                    f"Analysis job {job_id} does not exist or is currently running"
                )
        except sqlite3.Error as error:
            raise AnalysisJobStoreError(
                f"Could not delete analysis job: {error}"
            ) from error

    def _update_running_job(
        self,
        job_id: str,
        assignments: str,
        values: tuple,
        operation: str,
    ) -> None:
        try:
            with self._connection() as connection:
                cursor = connection.execute(
                    f"UPDATE analysis_jobs SET {assignments} "
                    "WHERE job_id = ? AND status = 'running'",
                    (*values, job_id),
                )
            if cursor.rowcount != 1:
                raise AnalysisJobStateError(
                    f"Cannot {operation} analysis job {job_id}: it is not running"
                )
        except sqlite3.Error as error:
            raise AnalysisJobStoreError(
                f"Could not {operation} analysis job: {error}"
            ) from error
