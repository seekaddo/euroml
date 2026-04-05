from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

CACHE_VERSION = 1
DEFAULT_CACHE_DIR = Path(".engine-cache")


def cache_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / f"{name}.pkl"


def load_cached_object(cache_dir: Path, name: str, fingerprint: str) -> Any | None:
    path = cache_path(cache_dir, name)
    if not path.exists():
        return None

    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        LOGGER.warning("Ignoring unreadable cache file %s: %s", path, exc)
        return None

    if payload.get("cache_version") != CACHE_VERSION:
        return None
    if payload.get("fingerprint") != fingerprint:
        return None
    return payload.get("data")


def save_cached_object(cache_dir: Path, name: str, fingerprint: str, data: Any) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir, name)
    payload = {
        "cache_version": CACHE_VERSION,
        "fingerprint": fingerprint,
        "data": data,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
