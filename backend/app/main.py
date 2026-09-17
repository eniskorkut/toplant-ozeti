from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import get_settings
from app.logging_config import configure_logging
from app.routers.recordings import router as recordings_router


class HealthResponse(BaseModel):
    status: str


settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(title=settings.app_name, version=settings.app_version)

# Centralized CORS configuration: the frontend runs natively on the host while
# the backend runs in Docker, so they are always different origins in dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)

app.include_router(recordings_router)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe used by tests, deployment checks and the frontend."""
    return HealthResponse(status="ok")
