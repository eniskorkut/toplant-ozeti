"""Dedicated processing worker: claims queued meetings and runs the CPU pipeline.

Run inside the backend image (compose service `worker`):

    uv run --locked python -m app.worker

`MEETING_WORKER_CONCURRENCY` slots poll the queue in parallel (default 1). Every
claim is an atomic state transition, so running several workers/replicas is safe;
stale jobs are recovered by lease, not by "requeue everything on startup".
"""

from __future__ import annotations

import asyncio
import logging
import os

from app.config import Settings, get_settings
from app.db import Database
from app.logging_config import configure_logging
from app.services.analysis_pipeline import (
    claim_next_analysis,
    process_analysis,
    requeue_stale_analyses,
)
from app.services.pipeline import claim_next_meeting, process_meeting, requeue_stale_meetings

logger = logging.getLogger(__name__)


async def recover_stale_jobs(database: Database, settings: Settings) -> tuple[int, int]:
    """Lease-based startup recovery: requeue jobs whose lease has expired."""
    async with database.session_factory() as session:
        meetings = await requeue_stale_meetings(session, settings.worker_lease_seconds)
        analyses = await requeue_stale_analyses(session, settings.worker_lease_seconds)
    if meetings or analyses:
        logger.info("recovered stale jobs: %d meetings, %d analyses", meetings, analyses)
    return meetings, analyses


async def run_once(database: Database, settings: Settings) -> bool:
    """Process one queued job. Transcription has priority over analysis."""
    async with database.session_factory() as session:
        meeting = await claim_next_meeting(session)
        if meeting is not None:
            await process_meeting(session, meeting, settings)
            return True

        analysis = await claim_next_analysis(session)
        if analysis is not None:
            await process_analysis(session, analysis, settings)
            return True

        return False


async def _worker_slot(settings: Settings, database: Database) -> None:
    """One concurrent slot: poll, claim and process until cancelled."""
    while True:
        did_work = await run_once(database, settings)
        if not did_work:
            await asyncio.sleep(settings.worker_poll_seconds)


async def worker_loop(settings: Settings, *, max_jobs: int | None = None) -> int:
    """Run the queue.

    With `max_jobs` the queue is drained sequentially and the loop exits (used by
    tests and one-shot runs). Otherwise `MEETING_WORKER_CONCURRENCY` slots poll in
    parallel and the loop runs until cancelled.
    """
    database = Database(settings.database_url)
    await database.init()
    await recover_stale_jobs(database, settings)

    concurrency = 1 if max_jobs is not None else max(1, settings.worker_concurrency)
    logger.info(
        "worker ready (database %s, concurrency %d)", settings.database_url, concurrency
    )

    try:
        if max_jobs is not None:
            processed = 0
            while processed < max_jobs:
                if not await run_once(database, settings):
                    break
                processed += 1
            return processed

        await asyncio.gather(
            *(_worker_slot(settings, database) for _ in range(concurrency))
        )
        return 0
    finally:
        await database.dispose()


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    configured_max_jobs = os.environ.get("MEETING_WORKER_MAX_JOBS")
    max_jobs = int(configured_max_jobs) if configured_max_jobs else None
    asyncio.run(worker_loop(settings, max_jobs=max_jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
