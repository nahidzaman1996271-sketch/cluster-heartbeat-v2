"""Structured console + file logging."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    log_dir: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a logger with a console handler and (optionally) a file handler.

    Handlers are installed once per logger name, so calling this repeatedly
    is cheap and never duplicates output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(_FMT, _DATEFMT))
        logger.addHandler(console)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{name.replace('.', '_')}.log"
        if not any(isinstance(h, logging.FileHandler) and
                   Path(h.baseFilename) == log_file for h in logger.handlers):
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter(_FMT, _DATEFMT))
            logger.addHandler(fh)

    return logger
