from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

FIRST_TUESDAY_DRAW = date(2011, 5, 10)
TWELVE_STAR_DRAW_START = date(2016, 9, 27)

TUESDAY = 1
FRIDAY = 4
WEEKDAY_NAMES = {
    TUESDAY: "Tuesday",
    FRIDAY: "Friday",
}


@dataclass(frozen=True)
class GameSpec:
    draw_date: date
    main_pool_size: int = 50
    main_pick_count: int = 5
    star_pick_count: int = 2
    star_pool_size: int = 12
    allowed_weekdays: tuple[int, ...] = (TUESDAY, FRIDAY)
    regime_name: str = "modern"

    def validate(self, main_numbers: tuple[int, ...], star_numbers: tuple[int, ...]) -> None:
        if self.draw_date.weekday() not in self.allowed_weekdays:
            allowed = ", ".join(WEEKDAY_NAMES[value] for value in self.allowed_weekdays)
            raise ValueError(f"{self.draw_date.isoformat()} is not a legal EuroMillions draw day. Expected: {allowed}")

        self._validate_component(main_numbers, self.main_pick_count, self.main_pool_size, "main")
        self._validate_component(star_numbers, self.star_pick_count, self.star_pool_size, "star")

    @staticmethod
    def _validate_component(numbers: tuple[int, ...], pick_count: int, pool_size: int, label: str) -> None:
        if len(numbers) != pick_count:
            raise ValueError(f"Expected {pick_count} {label} numbers, got {len(numbers)}")
        if tuple(sorted(numbers)) != numbers:
            raise ValueError(f"{label.capitalize()} numbers must be stored in sorted order")
        if len(set(numbers)) != len(numbers):
            raise ValueError(f"{label.capitalize()} numbers must be unique")
        if numbers[0] < 1 or numbers[-1] > pool_size:
            raise ValueError(f"{label.capitalize()} numbers must be between 1 and {pool_size}")


def spec_for_draw_date(draw_date: date) -> GameSpec:
    if draw_date < FIRST_TUESDAY_DRAW:
        return GameSpec(
            draw_date=draw_date,
            star_pool_size=9,
            allowed_weekdays=(FRIDAY,),
            regime_name="friday_9_star",
        )

    if draw_date < TWELVE_STAR_DRAW_START:
        return GameSpec(
            draw_date=draw_date,
            star_pool_size=11,
            allowed_weekdays=(TUESDAY, FRIDAY),
            regime_name="tuesday_11_star",
        )

    return GameSpec(
        draw_date=draw_date,
        star_pool_size=12,
        allowed_weekdays=(TUESDAY, FRIDAY),
        regime_name="modern_12_star",
    )


def infer_next_draw_date(last_draw_date: date) -> date:
    weekday = last_draw_date.weekday()
    if weekday == TUESDAY:
        return last_draw_date + timedelta(days=3)
    if weekday == FRIDAY:
        return last_draw_date + timedelta(days=4)
    raise ValueError(f"Unsupported last draw weekday: {weekday}")
