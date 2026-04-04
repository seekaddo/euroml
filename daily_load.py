"""
Copyright (c) Dennis Kwame Addo

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import date, datetime
from pathlib import Path

from download import DATASET_DIR, dataset_file_for_year, refresh_range

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_TRAIN_CSV = DATASET_DIR / "train_data.csv"
DEFAULT_TEST_CSV = DATASET_DIR / "test_data.csv"
DATE_FORMAT = "%d.%m.%Y"


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, DATE_FORMAT)


def find_existing_years(dataset_dir: Path) -> list[int]:
    years: list[int] = []
    for path in dataset_dir.glob("*_eml.json"):
        stem = path.stem
        year_str = stem.split("_", maxsplit=1)[0]
        if year_str.isdigit():
            years.append(int(year_str))
    return sorted(years)


def resolve_refresh_window(dataset_dir: Path, start_year: int | None, end_year: int | None) -> tuple[int, int]:
    current_year = date.today().year
    existing_years = find_existing_years(dataset_dir)

    resolved_start = start_year
    if resolved_start is None:
        # Daily refreshes should update the latest local archive year and the current year.
        resolved_start = max(existing_years) if existing_years else current_year

    resolved_end = end_year if end_year is not None else current_year
    if resolved_start > resolved_end:
        raise ValueError("start_year cannot be greater than end_year")

    return resolved_start, resolved_end


def latest_training_draw(train_csv: Path) -> datetime:
    latest = datetime.min
    with train_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            latest = max(latest, parse_date(row[0]))

    if latest == datetime.min:
        raise ValueError(f"No training rows found in {train_csv}")

    return latest


def load_draw_map(path: Path) -> dict[str, list[list[int]]]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_test_rows(dataset_dir: Path, cutoff: datetime) -> list[list[str]]:
    rows: list[list[str]] = []
    for year in find_existing_years(dataset_dir):
        draw_map = load_draw_map(dataset_file_for_year(year, dataset_dir))
        for draw_date, values in draw_map.items():
            parsed_date = parse_date(draw_date)
            if parsed_date <= cutoff:
                continue
            rows.append([draw_date, str(values[0]), str(values[1])])

    rows.sort(key=lambda row: parse_date(row[0]))
    return rows


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Date", "main-numbers", "lucky-stars"])
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cron-style updater for the latest EuroMillions yearly archives.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        help="First year to refresh. Defaults to the latest locally available year.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        help="Last year to refresh. Defaults to the current year.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help=f"Directory that contains the yearly *_eml.json files. Default: {DATASET_DIR}",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_TRAIN_CSV,
        help=f"Training CSV used to determine the train/test cutoff. Default: {DEFAULT_TRAIN_CSV}",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=DEFAULT_TEST_CSV,
        help=f"Output CSV rebuilt from the refreshed yearly JSON files. Default: {DEFAULT_TEST_CSV}",
    )
    csv_group = parser.add_mutually_exclusive_group()
    csv_group.add_argument(
        "--rebuild-test-csv",
        dest="rebuild_test_csv",
        action="store_true",
        help="Rebuild test_data.csv after refreshing the yearly archives.",
    )
    csv_group.add_argument(
        "--skip-test-csv",
        dest="rebuild_test_csv",
        action="store_false",
        help="Refresh only the yearly JSON archives and leave test_data.csv unchanged.",
    )
    parser.set_defaults(rebuild_test_csv=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_year, end_year = resolve_refresh_window(
        dataset_dir=args.dataset_dir,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    logging.info("Refreshing yearly archives from %s to %s", start_year, end_year)
    refresh_range(start_year, end_year, dataset_dir=args.dataset_dir)

    if not args.rebuild_test_csv:
        logging.info("Skipping test CSV rebuild by request")
        return

    cutoff = latest_training_draw(args.train_csv)
    rows = build_test_rows(args.dataset_dir, cutoff)
    write_csv(args.test_csv, rows)
    logging.info("Rebuilt %s with %s rows newer than %s", args.test_csv, len(rows), cutoff.strftime(DATE_FORMAT))


if __name__ == "__main__":
    main()
