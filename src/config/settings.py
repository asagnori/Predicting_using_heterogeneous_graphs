"""Central project configuration loaded from environment variables."""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.utils.device import get_device


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists() and re.search(r"(?m)^\s*DB_[A-Z_]+\s*=", _env_path.read_text()):
    load_dotenv(_env_path)


def _legacy_database_config() -> dict[str, str]:
    """Read the legacy ``DatabaseConfig = {...}`` format if present."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return {}

    content = env_path.read_text(encoding="utf-8")
    marker = "DatabaseConfig"
    start = content.find("{", content.find(marker))
    end = content.find("}", start)
    if start < 0 or end < 0:
        return {}

    try:
        value: Any = ast.literal_eval(content[start : end + 1])
    except (SyntaxError, ValueError):
        return {}

    if not isinstance(value, dict):
        return {}
    return {
        str(key): str(item)
        for key, item in value.items()
        if key in {"host", "database", "user", "password"}
    }


def _database_config() -> dict[str, str]:
    legacy = _legacy_database_config()
    config = {
        "host": os.getenv("DB_HOST", legacy.get("host", "")),
        "port": os.getenv("DB_PORT", ""),
        "database": os.getenv("DB_NAME", legacy.get("database", "")),
        "user": os.getenv("DB_USER", legacy.get("user", "")),
        "password": os.getenv("DB_PASSWORD", legacy.get("password", "")),
    }
    missing = [
        key for key, value in config.items()
        if key != "port" and not value
    ]
    if missing:
        raise RuntimeError(
            "Missing database configuration: "
            + ", ".join(
                "DB_PORT" if key == "port" else f"DB_{key.upper()}"
                for key in missing
            )
        )
    return config


DEVICE = get_device()
EPOCHS = int(os.getenv("GNN_EPOCHS", "100"))
LEARNING_RATE = float(os.getenv("GNN_LEARNING_RATE", "0.01"))
HIDDEN_CHANNELS = int(os.getenv("GNN_HIDDEN_CHANNELS", "64"))
OUT_CHANNELS = int(os.getenv("GNN_OUT_CHANNELS", "2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEMPORAL_CUTOFF = os.getenv("TEMPORAL_CUTOFF", "2018-01-01")

DATABASE_CONFIG = _database_config()