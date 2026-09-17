from fastapi import FastAPI
from pydantic import BaseModel

from app.config import get_settings


class HealthResponse(BaseModel):
    status: str


settings = get_settings()

app = FastAPI(title=settings.app_name, version=settings.app_version)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe used by tests, deployment checks and the frontend."""
    return HealthResponse(status="ok")
