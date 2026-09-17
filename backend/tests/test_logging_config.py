import logging

from app.logging_config import LOG_FORMAT, configure_logging


def test_configure_logging_sets_level_and_single_handler() -> None:
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level

    try:
        root.handlers.clear()

        configure_logging("debug")
        assert root.level == logging.DEBUG
        assert len(root.handlers) == 1

        configure_logging("INFO")
        assert root.level == logging.INFO
        assert len(root.handlers) == 1

        handler = root.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        assert handler.formatter is not None
        assert handler.formatter._fmt == LOG_FORMAT
    finally:
        root.handlers[:] = original_handlers
        root.setLevel(original_level)


def test_application_logger_records_reach_root(caplog) -> None:
    logger = logging.getLogger("app.services.recordings")

    with caplog.at_level(logging.INFO):
        logger.info("stored recording test-id")

    assert "stored recording test-id" in caplog.text
