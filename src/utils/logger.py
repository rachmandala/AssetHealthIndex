"""Centralized logging configuration for the Asset Health Index platform.

Logging is configured from ``config/logging.yaml`` via
:func:`logging.config.dictConfig`. If the YAML file cannot be found or
parsed, a safe basic configuration is used instead so the application
never fails purely because of a logging misconfiguration.
"""

from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path
from typing import Optional

import yaml

_DEFAULT_LOGGING_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "logging.yaml"
)
_LOGGER_NAME = "asset_health_index"
_configured = False


def setup_logging(config_path: Optional[Path | str] = None) -> None:
    """Configure application-wide logging from a YAML file.

    Safe to call multiple times; subsequent calls are no-ops unless
    ``force`` semantics are needed in the future.

    Args:
        config_path: Optional path to a logging YAML configuration file.
            Defaults to ``config/logging.yaml`` at the repository root.
    """
    global _configured
    if _configured:
        return

    path = Path(config_path) if config_path else _DEFAULT_LOGGING_CONFIG_PATH

    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                logging_config = yaml.safe_load(f)

            # Ensure the log directory referenced by file handlers exists.
            for handler in logging_config.get("handlers", {}).values():
                filename = handler.get("filename")
                if filename:
                    os.makedirs(Path(filename).parent, exist_ok=True)

            logging.config.dictConfig(logging_config)
        else:
            _configure_basic_logging()
    except (OSError, yaml.YAMLError, ValueError):
        _configure_basic_logging()

    _configured = True


def _configure_basic_logging() -> None:
    """Fallback to a minimal console-only logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Return a configured logger instance.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` instance, configuring the logging
        subsystem on first use if it has not already been configured.
    """
    if not _configured:
        setup_logging()
    return logging.getLogger(name)
