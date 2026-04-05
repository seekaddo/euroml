from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from engine.spec import GameSpec, infer_next_draw_date, spec_for_draw_date

DATE_FORMAT = "%d.%m.%Y"
DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"


@dataclass(frozen=True)
class DrawRecord:
    draw_id: int
    draw_date: date
    main_numbers: tuple[int, ...]
    star_numbers: tuple[int, ...]
    spec: GameSpec

    @property
    def weekday(self) -> str:
        return self.draw_date.strftime("%A")


def parse_draw_date(value: str) -> date:
    return datetime.strptime(value, DATE_FORMAT).date()


def format_draw_date(value: date) -> str:
    return value.strftime(DATE_FORMAT)


def load_draw_records(dataset_dir: Path = DEFAULT_DATASET_DIR) -> list[DrawRecord]:
    raw_draws: dict[date, tuple[tuple[int, ...], tuple[int, ...]]] = {}

    for path in sorted(dataset_dir.glob("*_eml.json")):
        year_prefix = path.stem.split("_", maxsplit=1)[0]
        if not year_prefix.isdigit():
            continue

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        for draw_date_str, values in payload.items():
            draw_date = parse_draw_date(draw_date_str)
            main_numbers = tuple(sorted(int(number) for number in values[0]))
            star_numbers = tuple(sorted(int(number) for number in values[1]))
            raw_draws[draw_date] = (main_numbers, star_numbers)

    records: list[DrawRecord] = []
    for draw_id, draw_date in enumerate(sorted(raw_draws), start=1):
        main_numbers, star_numbers = raw_draws[draw_date]
        spec = spec_for_draw_date(draw_date)
        spec.validate(main_numbers, star_numbers)
        records.append(
            DrawRecord(
                draw_id=draw_id,
                draw_date=draw_date,
                main_numbers=main_numbers,
                star_numbers=star_numbers,
                spec=spec,
            )
        )

    return records


def split_records(records: list[DrawRecord], train_end: date, test_start: date, test_end: date) -> tuple[list[DrawRecord], list[DrawRecord]]:
    train_records = [record for record in records if record.draw_date <= train_end]
    test_records = [record for record in records if test_start <= record.draw_date <= test_end]
    return train_records, test_records


def latest_draw_date(records: list[DrawRecord]) -> date:
    return max(record.draw_date for record in records)


def next_draw_date(records: list[DrawRecord]) -> date:
    return infer_next_draw_date(latest_draw_date(records))
