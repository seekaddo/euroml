from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.memory import load_memory, save_memory, select_confidence_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare prediction memory from the latest compare artifact.")
    parser.add_argument("--existing", type=Path, required=True, help="Existing committed engine state file.")
    parser.add_argument("--compare", type=Path, required=True, help="compare_2025.json artifact path.")
    parser.add_argument("--strategy", required=True, help="Prediction strategy name.")
    parser.add_argument("--output", type=Path, required=True, help="Output memory file path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    existing_memory = load_memory(args.existing)
    compare_payload = json.loads(args.compare.read_text(encoding="utf-8"))
    details = compare_payload.get("details", {}).get(args.strategy, {})
    calibration = details.get("confidence_calibration")

    merged_memory = dict(existing_memory)
    merged_memory["prediction_strategy"] = args.strategy
    if calibration is not None:
        merged_memory["confidence_calibration"] = select_confidence_calibration(
            existing_memory.get("confidence_calibration"),
            calibration,
        )

    save_memory(args.output, merged_memory)


if __name__ == "__main__":
    main()
