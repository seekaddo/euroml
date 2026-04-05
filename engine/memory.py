from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engine.data import DrawRecord


def dataset_fingerprint(records: list[DrawRecord]) -> str:
    payload = [
        {
            "draw_date": record.draw_date.isoformat(),
            "main_numbers": record.main_numbers,
            "star_numbers": record.star_numbers,
            "regime": record.spec.regime_name,
        }
        for record in records
    ]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest


def load_memory(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_memory(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched_payload = dict(payload)
    enriched_payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(enriched_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def select_confidence_calibration(
    existing: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if candidate is None:
        return existing
    if existing is None:
        return candidate

    existing_fingerprint = existing.get("dataset_fingerprint")
    candidate_fingerprint = candidate.get("dataset_fingerprint")
    existing_signature = existing.get("engine_signature")
    candidate_signature = candidate.get("engine_signature")
    if existing_fingerprint != candidate_fingerprint or existing_signature != candidate_signature:
        return candidate

    existing_observations = int(existing.get("observation_count", 0))
    candidate_observations = int(candidate.get("observation_count", 0))
    if candidate_observations > existing_observations:
        return candidate
    if candidate_observations < existing_observations:
        return existing

    existing_draws = int(existing.get("source_summary", {}).get("draw_count", 0))
    candidate_draws = int(candidate.get("source_summary", {}).get("draw_count", 0))
    if candidate_draws > existing_draws:
        return candidate
    return existing


def calibration_matches_dataset(
    calibration: dict[str, Any] | None,
    fingerprint: str,
    engine_signature: str | None = None,
) -> bool:
    if not calibration:
        return False
    if calibration.get("dataset_fingerprint") != fingerprint:
        return False
    if engine_signature is not None and calibration.get("engine_signature") != engine_signature:
        return False
    return True
