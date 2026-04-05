from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine.data import DrawRecord
from engine.spec import spec_for_draw_date

WINDOWS = (5, 10, 25, 50, 100, 250)
CONDITIONAL_WINDOWS = (10, 25, 50)
RECENT_STEPS = 3
FEATURE_SET_VERSION = "v2_weekday_regime"


@dataclass(frozen=True)
class ComponentConfig:
    name: str
    max_pool_size: int
    pick_count: int


@dataclass
class SegmentState:
    draw_count: int
    counts_total: np.ndarray
    last_seen: np.ndarray
    window_queues: dict[int, deque[set[int]]] = field(default_factory=dict)
    window_counts: dict[int, np.ndarray] = field(default_factory=dict)


MAIN_CONFIG = ComponentConfig(name="main", max_pool_size=50, pick_count=5)
STAR_CONFIG = ComponentConfig(name="star", max_pool_size=12, pick_count=2)


def feature_signature() -> str:
    return FEATURE_SET_VERSION


def feature_columns() -> list[str]:
    columns = [
        "candidate",
        "candidate_norm",
        "history_len",
        "weekday_tuesday",
        "weekday_friday",
        "pool_size",
        "pick_count",
        "global_rate",
        "gap",
        "gap_norm",
        "ewma_short",
        "ewma_long",
        "ewma_diff",
        "deviation_from_uniform",
        "zscore_from_uniform",
        "same_weekday_history_len",
        "same_weekday_global_rate",
        "same_weekday_gap",
        "same_weekday_gap_norm",
        "same_weekday_rate_lift",
        "regime_history_len",
        "regime_global_rate",
        "regime_gap",
        "regime_gap_norm",
        "regime_rate_lift",
    ]
    columns.extend(f"recent_hit_{step}" for step in range(1, RECENT_STEPS + 1))
    columns.extend(f"count_last_{window}" for window in WINDOWS)
    columns.extend(f"rate_last_{window}" for window in WINDOWS)
    columns.extend(f"same_weekday_rate_last_{window}" for window in CONDITIONAL_WINDOWS)
    columns.extend(f"regime_rate_last_{window}" for window in CONDITIONAL_WINDOWS)
    columns.extend(
        [
            "hotness_25_100",
            "hotness_10_50",
            "hotness_5_25",
            "same_weekday_hotness_10_50",
            "regime_hotness_10_50",
        ]
    )
    return columns


def build_feature_tables(records: list[DrawRecord]) -> dict[str, pd.DataFrame]:
    return {
        MAIN_CONFIG.name: _build_component_table(records, MAIN_CONFIG),
        STAR_CONFIG.name: _build_component_table(records, STAR_CONFIG),
    }


def build_prediction_rows(records: list[DrawRecord], draw_date) -> dict[str, pd.DataFrame]:
    return {
        MAIN_CONFIG.name: _build_component_prediction_rows(records, MAIN_CONFIG, draw_date),
        STAR_CONFIG.name: _build_component_prediction_rows(records, STAR_CONFIG, draw_date),
    }


def _make_segment_state(max_pool_size: int, windows: tuple[int, ...]) -> SegmentState:
    return SegmentState(
        draw_count=0,
        counts_total=np.zeros(max_pool_size + 1, dtype=float),
        last_seen=np.full(max_pool_size + 1, -1, dtype=int),
        window_queues={window: deque() for window in windows},
        window_counts={window: np.zeros(max_pool_size + 1, dtype=float) for window in windows},
    )


def _segment_stats(
    state: SegmentState,
    candidate: int,
    expected_rate: float,
    windows: tuple[int, ...],
) -> tuple[int, float, int, float, dict[int, float]]:
    history_len = state.draw_count
    gap = history_len - state.last_seen[candidate] if state.last_seen[candidate] >= 0 else history_len + 1
    global_rate = state.counts_total[candidate] / history_len if history_len else expected_rate
    rate_last = {
        window: (
            state.window_counts[window][candidate] / min(history_len, window)
            if history_len
            else expected_rate
        )
        for window in windows
    }
    return history_len, global_rate, gap, gap / max(1, history_len), rate_last


def _update_segment_state(state: SegmentState, current_values: set[int], windows: tuple[int, ...]) -> None:
    current_index = state.draw_count
    for value in current_values:
        state.counts_total[value] += 1.0
        state.last_seen[value] = current_index

    for window in windows:
        state.window_queues[window].append(current_values)
        for value in current_values:
            state.window_counts[window][value] += 1.0
        if len(state.window_queues[window]) > window:
            expired = state.window_queues[window].popleft()
            for value in expired:
                state.window_counts[window][value] -= 1.0

    state.draw_count += 1
def _build_component_table(records: list[DrawRecord], config: ComponentConfig) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    overall_state = _make_segment_state(config.max_pool_size, WINDOWS)
    weekday_states = {
        1: _make_segment_state(config.max_pool_size, CONDITIONAL_WINDOWS),
        4: _make_segment_state(config.max_pool_size, CONDITIONAL_WINDOWS),
    }
    regime_states: dict[int, SegmentState] = {}

    ewma_short = np.zeros(config.max_pool_size + 1, dtype=float)
    ewma_long = np.zeros(config.max_pool_size + 1, dtype=float)
    alpha_short = 0.25
    alpha_long = 0.08
    history_values: list[set[int]] = []

    for record in records:
        current_values = set(record.main_numbers if config.name == MAIN_CONFIG.name else record.star_numbers)
        weekday_key = record.draw_date.weekday()
        pool_size = record.spec.main_pool_size if config.name == MAIN_CONFIG.name else record.spec.star_pool_size
        regime_state = regime_states.setdefault(pool_size, _make_segment_state(config.max_pool_size, CONDITIONAL_WINDOWS))
        weekday_state = weekday_states[weekday_key]
        expected_rate = config.pick_count / pool_size

        for candidate in range(1, pool_size + 1):
            history_len, global_rate, gap, gap_norm, rate_last = _segment_stats(
                overall_state,
                candidate,
                expected_rate,
                WINDOWS,
            )
            same_weekday_history_len, same_weekday_rate, same_weekday_gap, same_weekday_gap_norm, same_weekday_rate_last = _segment_stats(
                weekday_state,
                candidate,
                expected_rate,
                CONDITIONAL_WINDOWS,
            )
            regime_history_len, regime_rate, regime_gap, regime_gap_norm, regime_rate_last = _segment_stats(
                regime_state,
                candidate,
                expected_rate,
                CONDITIONAL_WINDOWS,
            )

            variance = history_len * expected_rate * (1.0 - expected_rate)
            zscore = 0.0
            if variance > 0:
                zscore = (overall_state.counts_total[candidate] - history_len * expected_rate) / np.sqrt(variance)

            row: dict[str, float | int | str] = {
                "component": config.name,
                "draw_id": record.draw_id,
                "draw_date": record.draw_date.isoformat(),
                "draw_ordinal": record.draw_date.toordinal(),
                "weekday": record.weekday,
                "candidate": candidate,
                "candidate_norm": candidate / pool_size,
                "history_len": history_len,
                "weekday_tuesday": int(weekday_key == 1),
                "weekday_friday": int(weekday_key == 4),
                "pool_size": pool_size,
                "pick_count": config.pick_count,
                "global_rate": global_rate,
                "gap": gap,
                "gap_norm": gap_norm,
                "ewma_short": ewma_short[candidate],
                "ewma_long": ewma_long[candidate],
                "ewma_diff": ewma_short[candidate] - ewma_long[candidate],
                "deviation_from_uniform": global_rate - expected_rate,
                "zscore_from_uniform": zscore,
                "same_weekday_history_len": same_weekday_history_len,
                "same_weekday_global_rate": same_weekday_rate,
                "same_weekday_gap": same_weekday_gap,
                "same_weekday_gap_norm": same_weekday_gap_norm,
                "same_weekday_rate_lift": same_weekday_rate - global_rate,
                "regime_history_len": regime_history_len,
                "regime_global_rate": regime_rate,
                "regime_gap": regime_gap,
                "regime_gap_norm": regime_gap_norm,
                "regime_rate_lift": regime_rate - global_rate,
                "target": int(candidate in current_values),
            }

            for step in range(1, RECENT_STEPS + 1):
                row[f"recent_hit_{step}"] = int(len(history_values) >= step and candidate in history_values[-step])

            for window in WINDOWS:
                row[f"count_last_{window}"] = overall_state.window_counts[window][candidate]
                row[f"rate_last_{window}"] = rate_last[window]

            for window in CONDITIONAL_WINDOWS:
                row[f"same_weekday_rate_last_{window}"] = same_weekday_rate_last[window]
                row[f"regime_rate_last_{window}"] = regime_rate_last[window]

            row["hotness_25_100"] = row["rate_last_25"] - row["rate_last_100"]
            row["hotness_10_50"] = row["rate_last_10"] - row["rate_last_50"]
            row["hotness_5_25"] = row["rate_last_5"] - row["rate_last_25"]
            row["same_weekday_hotness_10_50"] = row["same_weekday_rate_last_10"] - row["same_weekday_rate_last_50"]
            row["regime_hotness_10_50"] = row["regime_rate_last_10"] - row["regime_rate_last_50"]
            rows.append(row)

        history_values.append(current_values)

        indicator = np.zeros(config.max_pool_size + 1, dtype=float)
        for value in current_values:
            indicator[value] = 1.0

        ewma_short = alpha_short * indicator + (1.0 - alpha_short) * ewma_short
        ewma_long = alpha_long * indicator + (1.0 - alpha_long) * ewma_long

        _update_segment_state(overall_state, current_values, WINDOWS)
        _update_segment_state(weekday_state, current_values, CONDITIONAL_WINDOWS)
        _update_segment_state(regime_state, current_values, CONDITIONAL_WINDOWS)

    return pd.DataFrame(rows)


def _build_component_prediction_rows(records: list[DrawRecord], config: ComponentConfig, draw_date) -> pd.DataFrame:
    spec = spec_for_draw_date(draw_date)
    overall_state = _make_segment_state(config.max_pool_size, WINDOWS)
    weekday_states = {
        1: _make_segment_state(config.max_pool_size, CONDITIONAL_WINDOWS),
        4: _make_segment_state(config.max_pool_size, CONDITIONAL_WINDOWS),
    }
    regime_states: dict[int, SegmentState] = {}

    ewma_short = np.zeros(config.max_pool_size + 1, dtype=float)
    ewma_long = np.zeros(config.max_pool_size + 1, dtype=float)
    alpha_short = 0.25
    alpha_long = 0.08
    history_values: list[set[int]] = []

    for record in records:
        current_values = set(record.main_numbers if config.name == MAIN_CONFIG.name else record.star_numbers)
        weekday_key = record.draw_date.weekday()
        pool_size = record.spec.main_pool_size if config.name == MAIN_CONFIG.name else record.spec.star_pool_size
        regime_state = regime_states.setdefault(pool_size, _make_segment_state(config.max_pool_size, CONDITIONAL_WINDOWS))
        weekday_state = weekday_states[weekday_key]

        indicator = np.zeros(config.max_pool_size + 1, dtype=float)
        for value in current_values:
            indicator[value] = 1.0
        ewma_short = alpha_short * indicator + (1.0 - alpha_short) * ewma_short
        ewma_long = alpha_long * indicator + (1.0 - alpha_long) * ewma_long

        history_values.append(current_values)
        _update_segment_state(overall_state, current_values, WINDOWS)
        _update_segment_state(weekday_state, current_values, CONDITIONAL_WINDOWS)
        _update_segment_state(regime_state, current_values, CONDITIONAL_WINDOWS)

    target_weekday = draw_date.weekday()
    pool_size = spec.main_pool_size if config.name == MAIN_CONFIG.name else spec.star_pool_size
    expected_rate = config.pick_count / pool_size
    weekday_state = weekday_states[target_weekday]
    regime_state = regime_states.setdefault(pool_size, _make_segment_state(config.max_pool_size, CONDITIONAL_WINDOWS))
    rows: list[dict[str, float | int | str]] = []

    for candidate in range(1, pool_size + 1):
        history_len, global_rate, gap, gap_norm, rate_last = _segment_stats(
            overall_state,
            candidate,
            expected_rate,
            WINDOWS,
        )
        same_weekday_history_len, same_weekday_rate, same_weekday_gap, same_weekday_gap_norm, same_weekday_rate_last = _segment_stats(
            weekday_state,
            candidate,
            expected_rate,
            CONDITIONAL_WINDOWS,
        )
        regime_history_len, regime_rate, regime_gap, regime_gap_norm, regime_rate_last = _segment_stats(
            regime_state,
            candidate,
            expected_rate,
            CONDITIONAL_WINDOWS,
        )

        variance = history_len * expected_rate * (1.0 - expected_rate)
        zscore = 0.0
        if variance > 0:
            zscore = (overall_state.counts_total[candidate] - history_len * expected_rate) / np.sqrt(variance)

        row: dict[str, float | int | str] = {
            "component": config.name,
            "draw_id": history_len + 1,
            "draw_date": draw_date.isoformat(),
            "draw_ordinal": draw_date.toordinal(),
            "weekday": draw_date.strftime("%A"),
            "candidate": candidate,
            "candidate_norm": candidate / pool_size,
            "history_len": history_len,
            "weekday_tuesday": int(target_weekday == 1),
            "weekday_friday": int(target_weekday == 4),
            "pool_size": pool_size,
            "pick_count": config.pick_count,
            "global_rate": global_rate,
            "gap": gap,
            "gap_norm": gap_norm,
            "ewma_short": ewma_short[candidate],
            "ewma_long": ewma_long[candidate],
            "ewma_diff": ewma_short[candidate] - ewma_long[candidate],
            "deviation_from_uniform": global_rate - expected_rate,
            "zscore_from_uniform": zscore,
            "same_weekday_history_len": same_weekday_history_len,
            "same_weekday_global_rate": same_weekday_rate,
            "same_weekday_gap": same_weekday_gap,
            "same_weekday_gap_norm": same_weekday_gap_norm,
            "same_weekday_rate_lift": same_weekday_rate - global_rate,
            "regime_history_len": regime_history_len,
            "regime_global_rate": regime_rate,
            "regime_gap": regime_gap,
            "regime_gap_norm": regime_gap_norm,
            "regime_rate_lift": regime_rate - global_rate,
            "target": 0,
        }

        for step in range(1, RECENT_STEPS + 1):
            row[f"recent_hit_{step}"] = int(len(history_values) >= step and candidate in history_values[-step])

        for window in WINDOWS:
            row[f"count_last_{window}"] = overall_state.window_counts[window][candidate]
            row[f"rate_last_{window}"] = rate_last[window]

        for window in CONDITIONAL_WINDOWS:
            row[f"same_weekday_rate_last_{window}"] = same_weekday_rate_last[window]
            row[f"regime_rate_last_{window}"] = regime_rate_last[window]

        row["hotness_25_100"] = row["rate_last_25"] - row["rate_last_100"]
        row["hotness_10_50"] = row["rate_last_10"] - row["rate_last_50"]
        row["hotness_5_25"] = row["rate_last_5"] - row["rate_last_25"]
        row["same_weekday_hotness_10_50"] = row["same_weekday_rate_last_10"] - row["same_weekday_rate_last_50"]
        row["regime_hotness_10_50"] = row["regime_rate_last_10"] - row["regime_rate_last_50"]
        rows.append(row)

    return pd.DataFrame(rows)
