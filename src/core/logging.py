# src/core/logging.py
# Lightweight, production-friendly logging with optional Rich console output
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

try:  # Rich is optional; if absent, fall back to basic logging
    from rich.logging import RichHandler  # type: ignore
except Exception:  # pragma: no cover
    RichHandler = None  # type: ignore


_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


def _level_from_str(level: str) -> int:
    return _LEVELS.get(level.upper(), logging.INFO)


class _PlainFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover
        # Attach module.function:lineno for quick tracing
        record.module_line = f"{record.module}.{record.funcName}:{record.lineno}"
        record.message = record.getMessage()
        return f"{record.levelname} | {record.module_line} | {record.message}"


def configure_logging(
    *,
    level: str = "INFO",
    log_dir: Path | str = Path("logs"),
    file_name: str = "runtime.log",
    rotate_mb: int = 50,
    use_rich: bool = True,
) -> None:
    """Configure root logger with console + rotating file handlers.

    Args:
        level: min log level name
        log_dir: directory for log files
        file_name: base log file name
        rotate_mb: max size per file before rotation
        use_rich: use Rich console handler if available
    """
    root = logging.getLogger()
    root.setLevel(_level_from_str(level))

    # Clear existing handlers to allow re-config at hot-reload
    for h in list(root.handlers):
        root.removeHandler(h)

    # Console handler
    if use_rich and RichHandler is not None:
        console = RichHandler(rich_tracebacks=False, show_time=False, show_path=False)
        console.setLevel(_level_from_str(level))
        root.addHandler(console)
    else:
        console = logging.StreamHandler()
        console.setFormatter(_PlainFormatter())
        console.setLevel(_level_from_str(level))
        root.addHandler(console)

    # Rotating file handler
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / file_name, maxBytes=rotate_mb * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(_level_from_str(level))
    file_handler.setFormatter(_PlainFormatter())
    root.addHandler(file_handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a module-level logger. Call configure_logging() once at program start."""
    return logging.getLogger(name)
