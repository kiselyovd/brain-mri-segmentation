"""Structured logging configuration."""

from __future__ import annotations

import logging
import os
import sys

import structlog


def configure_logging(level: str | None = None, json_output: bool = False) -> None:
    level = level or os.environ.get("LOG_LEVEL", "INFO")
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level.upper())
    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper())),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
