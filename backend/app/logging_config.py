"""Standard-library logging setup for the application.

Uvicorn configures only its own loggers, so without this the root logger has no
handler and application log records never reach `docker compose logs`.
"""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Attach a single stdout handler to the root logger and set its level."""
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(handler)
    root.setLevel(level.upper())
