"""Dedicated, bounded thread pool for LLM calls.

LLM providers are synchronous (``httpx.post``). Running them in the shared default
executor would let a burst of chat questions starve uploads (ffmpeg) and other
``asyncio.to_thread`` work, so they get their own pool sized by
``MEETING_LLM_MAX_CONCURRENCY``. Extra calls wait in the pool queue instead of
exhausting the event loop's threads, and the pool size is a hard cap on concurrent
provider traffic.
"""

from __future__ import annotations

import asyncio
import functools
import threading
from concurrent.futures import ThreadPoolExecutor

from app.config import Settings
from app.services.llm_provider import MeetingAnalysisProvider, ProviderResponse

_executor: ThreadPoolExecutor | None = None
_lock = threading.Lock()


def get_llm_executor(settings: Settings) -> ThreadPoolExecutor:
    """The process-wide LLM pool (created once, sized from the first settings)."""
    global _executor
    if _executor is None:
        with _lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(
                    max_workers=max(1, settings.llm_max_concurrency),
                    thread_name_prefix="meeting-llm",
                )
    return _executor


async def run_llm(
    provider: MeetingAnalysisProvider,
    *,
    system_prompt: str,
    user_prompt: str,
    session_id: str,
    settings: Settings,
) -> ProviderResponse:
    """Run one provider call in the dedicated pool without blocking the event loop."""
    loop = asyncio.get_running_loop()
    executor = get_llm_executor(settings)
    call = functools.partial(
        provider.analyze,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        session_id=session_id,
    )
    return await loop.run_in_executor(executor, call)
