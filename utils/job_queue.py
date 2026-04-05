"""
Redis-based async job queue for managing try-on inference jobs.
"""

import json
import logging
import redis

logger = logging.getLogger(__name__)

JOB_TTL = 3600  # seconds


class JobQueue:
    """
    Thin wrapper around Redis for storing job status.

    Job states: pending → processing → completed | failed
    """

    def __init__(self, host: str = "localhost", port: int = 6379):
        self.r = redis.Redis(host=host, port=port, decode_responses=True)
        self.r.ping()
        logger.info(f"JobQueue connected to Redis at {host}:{port}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_job(self, job_id: str) -> bool:
        """Create a new job entry with status 'pending'. Returns True on success."""
        try:
            data = {"status": "pending", "result_url": None, "error": None}
            self.r.set(self._key(job_id), json.dumps(data), ex=JOB_TTL)
            return True
        except Exception as e:
            logger.error(f"Failed to create job {job_id}: {e}")
            return False

    def update_job_status(
        self,
        job_id: str,
        status: str,
        result_url: str = None,
        error: str = None,
    ) -> None:
        """Update job status. status must be 'pending'|'processing'|'completed'|'failed'."""
        data = {"status": status, "result_url": result_url, "error": error}
        self.r.set(self._key(job_id), json.dumps(data), ex=JOB_TTL)

    def get_job_status(self, job_id: str) -> dict | None:
        """Return job dict or None if not found."""
        raw = self.r.get(self._key(job_id))
        if raw is None:
            return None
        return json.loads(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(job_id: str) -> str:
        return f"vton:job:{job_id}"
