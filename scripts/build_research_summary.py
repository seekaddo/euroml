from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble research outputs into metadata and markdown summary files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-strategy", required=True)
    parser.add_argument("--strategies", nargs="+", required=True)
    return parser.parse_args()


def _best_result(payload: dict[str, object]) -> dict[str, object]:
    results = payload["results"]
    assert isinstance(results, list)
    return max(
        results,
        key=lambda row: (
            float(row["4+2_count"]),
            float(row["3+2_count"]),
            float(row.get("useful_rate", 0.0)),
            float(row.get("3+1_count", 0.0)),
            float(row["2+2_rate"]),
            float(row["best5_w"]),
            float(row["lift_best5"]),
        ),
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir

    compare_2025 = json.loads((output_dir / "compare_2025.json").read_text(encoding="utf-8"))
    compare_2026 = json.loads((output_dir / "compare_2026.json").read_text(encoding="utf-8"))
    next_prediction = json.loads((output_dir / "next_draw_prediction.json").read_text(encoding="utf-8"))

    best_2025 = _best_result(compare_2025)
    best_2026 = _best_result(compare_2026)
    details_2025 = compare_2025.get("details", {})
    details_2026 = compare_2026.get("details", {})
    best_2025_events = details_2025.get(best_2025["strategy"], {}).get("best5_events", [])
    best_2026_events = details_2026.get(best_2026["strategy"], {}).get("best5_events", [])
    predictions = next_prediction.get("predictions", [])
    top_prediction = predictions[0] if predictions else None

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "strategies": args.strategies,
        "samples": next_prediction.get("samples") or os.environ.get("SAMPLES"),
        "prediction_strategy": args.prediction_strategy,
        "next_draw_date": next_prediction.get("next_draw_date"),
        "best_2025": best_2025,
        "best_2026": best_2026,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Research Cycle Summary",
        "",
        f"- Generated: `{metadata['generated_at_utc']}`",
        f"- Samples: `{metadata['samples']}`",
        f"- Prediction strategy: `{args.prediction_strategy}`",
        f"- Strategies compared: `{', '.join(args.strategies)}`",
        "",
        "## Best 2025 Window Result",
        "",
        f"- Strategy: `{best_2025['strategy']}`",
        f"- `best5_w`: `{best_2025['best5_w']}`",
        f"- `lift_best5`: `{best_2025['lift_best5']}`",
        f"- `useful_rate`: `{best_2025.get('useful_rate', 0.0)}`",
        f"- `2+2_rate`: `{best_2025['2+2_rate']}`",
        f"- `3+1_count`: `{best_2025.get('3+1_count', 0)}`",
        f"- `3+2_count`: `{best_2025['3+2_count']}`",
        f"- `4+2_count`: `{best_2025['4+2_count']}`",
        "",
        "### Best 2025 Useful Hits",
        "",
    ]

    if best_2025_events:
        for event in best_2025_events:
            lines.append(f"- `{event['draw_date']}` `{event['hit']}`")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Best 2026 Window Result",
            "",
            f"- Strategy: `{best_2026['strategy']}`",
            f"- `best5_w`: `{best_2026['best5_w']}`",
            f"- `lift_best5`: `{best_2026['lift_best5']}`",
            f"- `useful_rate`: `{best_2026.get('useful_rate', 0.0)}`",
            f"- `2+2_rate`: `{best_2026['2+2_rate']}`",
            f"- `3+1_count`: `{best_2026.get('3+1_count', 0)}`",
            f"- `3+2_count`: `{best_2026['3+2_count']}`",
            f"- `4+2_count`: `{best_2026['4+2_count']}`",
            "",
            "### Best 2026 Useful Hits",
            "",
        ]
    )

    if best_2026_events:
        for event in best_2026_events:
            lines.append(f"- `{event['draw_date']}` `{event['hit']}`")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Next Prediction",
            "",
            f"- Target draw date: `{next_prediction.get('next_draw_date')}`",
        ]
    )

    if top_prediction is not None:
        main_numbers = " ".join(f"{number:02d}" for number in top_prediction["main_numbers"])
        star_numbers = " ".join(f"{number:02d}" for number in top_prediction["star_numbers"])
        lines.extend(
            [
                f"- Top ticket: `{main_numbers} | {star_numbers}`",
                f"- Confidence: `{top_prediction['confidence']}`",
                f"- Score: `{top_prediction['score']}`",
            ]
        )

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
