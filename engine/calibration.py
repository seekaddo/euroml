from __future__ import annotations

from typing import Any

import numpy as np


def build_confidence_calibration(report: dict[str, Any], bin_count: int = 10) -> dict[str, Any] | None:
    observations: list[dict[str, float | int]] = []
    for draw in report.get("draws", []):
        for prediction in draw.get("predictions", []):
            observations.append(
                {
                    "score": float(prediction["score"]),
                    "weighted_hits": int(prediction["weighted_hits"]),
                    "main_hits": int(prediction["main_hits"]),
                    "star_hits": int(prediction["star_hits"]),
                }
            )

    if len(observations) < max(20, bin_count):
        return None

    scores = np.array([item["score"] for item in observations], dtype=float)
    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, bin_count + 1))
    unique_edges = np.unique(quantiles)
    if len(unique_edges) < 3:
        return None

    bins: list[dict[str, Any]] = []
    for index in range(len(unique_edges) - 1):
        lower = float(unique_edges[index])
        upper = float(unique_edges[index + 1])
        if index == len(unique_edges) - 2:
            members = [item for item in observations if lower <= item["score"] <= upper]
        else:
            members = [item for item in observations if lower <= item["score"] < upper]

        if not members:
            continue

        avg_weighted_hits = float(np.mean([item["weighted_hits"] for item in members]))
        avg_main_hits = float(np.mean([item["main_hits"] for item in members]))
        avg_star_hits = float(np.mean([item["star_hits"] for item in members]))
        hit_any_rate = float(np.mean([item["weighted_hits"] > 0 for item in members]))
        hit_two_plus_rate = float(np.mean([item["weighted_hits"] >= 2 for item in members]))
        hit_four_plus_rate = float(np.mean([item["weighted_hits"] >= 4 for item in members]))

        bins.append(
            {
                "score_min": lower,
                "score_max": upper,
                "count": len(members),
                "avg_weighted_hits": avg_weighted_hits,
                "avg_main_hits": avg_main_hits,
                "avg_star_hits": avg_star_hits,
                "hit_any_rate": hit_any_rate,
                "hit_two_plus_rate": hit_two_plus_rate,
                "hit_four_plus_rate": hit_four_plus_rate,
            }
        )

    if len(bins) < 2:
        return None

    signals = [item["avg_weighted_hits"] + item["hit_two_plus_rate"] + 1.5 * item["hit_four_plus_rate"] for item in bins]
    min_signal = min(signals)
    max_signal = max(signals)
    span = max(max_signal - min_signal, 1e-9)
    for item, signal in zip(bins, signals, strict=True):
        relative = (signal - min_signal) / span
        item["confidence"] = max(1, min(10, int(round(1 + 9 * relative))))

    return {
        "bin_count": len(bins),
        "observation_count": len(observations),
        "bins": bins,
        "source_summary": report.get("summary", {}),
    }


def calibrate_confidence(score: float, calibration: dict[str, Any] | None) -> int | None:
    if not calibration:
        return None

    bins = calibration.get("bins", [])
    if not bins:
        return None

    for item in bins:
        if item["score_min"] <= score <= item["score_max"]:
            return int(item["confidence"])

    if score < bins[0]["score_min"]:
        return int(bins[0]["confidence"])
    return int(bins[-1]["confidence"])
