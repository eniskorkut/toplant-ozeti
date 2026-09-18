"""Dedicated processing worker: claims queued meetings and runs the CPU pipeline.

Run inside the backend image (compose service `worker`):

    uv run --locked python -m app.worker
"""

from __future__ import annotations

import asyncio
import logging
import os

from app.config import Settings, get_settings
from app.db import Database
from app.logging_config import configure_logging
from app.services.pipeline import claim_next_meeting, process_meeting

logger = logging.getLogger(__name__)


async def run_once(database: Database, settings: Settings) -> bool:
    """Process at most one queued meeting. Returns True when work was done."""
    async with database.session_factory() as session:
        meeting = await claim_next_meeting(session)
        if meeting is None:
            return False
        await process_meeting(session, meeting, settings)
        return True


async def worker_loop(settings: Settings, *, max_jobs: int | None = None) -> int:
    """Poll the queue forever, or drain it and exit when `max_jobs` is set."""
    database = Database(settings.database_url)
    await database.init()
    logger.info("worker ready (database %s)", settings.database_url)

    processed = 0
    try:
        while max_jobs is None or processed < max_jobs:
            did_work = await run_once(database, settings)
            if did_work:
                processed += 1
            elif max_jobs is not None:
                break
            else:
                await asyncio.sleep(settings.worker_poll_seconds)
    finally:
        await database.dispose()
    return processed


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    configured_max_jobs = os.environ.get("MEETING_WORKER_MAX_JOBS")
    max_jobs = int(configured_max_jobs) if configured_max_jobs else None
    asyncio.run(worker_loop(settings, max_jobs=max_jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
