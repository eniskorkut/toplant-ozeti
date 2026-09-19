from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import get_settings
from app.db import get_database
from app.logging_config import configure_logging
from app.routers.live_transcription import router as live_transcription_router
from app.routers.meetings import router as meetings_router
from app.routers.recordings import router as recordings_router
from app.routers.transcription import router as transcription_router


class HealthResponse(BaseModel):
    status: str


settings = get_settings()
configure_logging(settings.log_level)

@asynccontextmanager
async def lifespan(_: FastAPI):
    # The API process never runs inference; it only reads/writes the job state.
    database = get_database()
    await database.init()
    yield
    await database.dispose()


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)

# Centralized CORS configuration: the frontend runs natively on the host while
# the backend runs in Docker, so they are always different origins in dev.
# DELETE (meeting deletion) and PUT (speaker aliases) are real browser calls and
# must pass the preflight, otherwise fetch fails with "Failed to fetch".
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Accept"],
)

app.include_router(recordings_router)
app.include_router(meetings_router)
app.include_router(live_transcription_router)
app.include_router(transcription_router)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe used by tests, deployment checks and the frontend."""
    return HealthResponse(status="ok")
