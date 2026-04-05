from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import logging
from statistics import mean
from typing import Literal

import numpy as np
import pandas as pd

from engine.data import DrawRecord
from engine.strategies import StrategySpec
from engine.tickets import CandidateTicket, evaluate_ticket

BacktestMode = Literal["frozen", "rolling"]
LOGGER = logging.getLogger(__name__)
MILESTONE_PATTERNS: tuple[tuple[str, int, int], ...] = (
    ("0+2", 0, 2),
    ("1+2", 1, 2),
    ("2+2", 2, 2),
    ("3+1", 3, 1),
    ("3+2", 3, 2),
    ("4+1", 4, 1),
    ("4+2", 4, 2),
    ("5+2", 5, 2),
)
RELEASE_PATTERNS: tuple[str, ...] = ("3+2", "4+2")


@dataclass(frozen=True)
class BacktestConfig:
    train_end: date
    test_start: date
    test_end: date
    mode: BacktestMode = "frozen"
    top_k: int = 5
    sample_count: int = 5000
    random_state: int = 7


@dataclass
class DrawBacktestResult:
    draw_date: str
    actual_main_numbers: tuple[int, ...]
    actual_star_numbers: tuple[int, ...]
    predictions: list[dict[str, object]]


def _pattern_counts(scores: list[dict[str, int]]) -> dict[str, int]:
    counts = {label: 0 for label, _, _ in MILESTONE_PATTERNS}
    for score in scores:
        label = f"{score['main_hits']}+{score['star_hits']}"
        if label in counts:
            counts[label] += 1
    return counts


def _is_useful_hit(score: dict[str, int]) -> bool:
    return (
        (score["star_hits"] == 2 and score["main_hits"] >= 2)
        or (score["main_hits"] >= 3 and score["star_hits"] >= 1)
    )


def _aggregate_milestones(
    top1_scores: list[dict[str, int]],
    best5_scores: list[dict[str, int]],
    random_top1_scores: list[dict[str, int]],
    random_best5_scores: list[dict[str, int]],
) -> dict[str, object]:
    def _pack(scores: list[dict[str, int]]) -> dict[str, object]:
        draw_count = len(scores)
        patterns = _pattern_counts(scores)
        rates = {
            label: round(count / draw_count, 4) if draw_count else 0.0
            for label, count in patterns.items()
        }
        useful_hits = sum(1 for score in scores if _is_useful_hit(score))
        return {
            "count": patterns,
            "rate": rates,
            "useful_hit_count": useful_hits,
            "useful_hit_rate": round(useful_hits / draw_count, 4) if draw_count else 0.0,
        }

    return {
        "top1": _pack(top1_scores),
        "best5": _pack(best5_scores),
        "random_top1": _pack(random_top1_scores),
        "random_best5": _pack(random_best5_scores),
    }


def _build_release_gate(milestones: dict[str, object]) -> dict[str, object]:
    release_gate: dict[str, object] = {
        "target_patterns": list(RELEASE_PATTERNS),
        "window_pass_rule": "Pass this backtest window only if both 3+2 and 4+2 occur at least once and beat the random baseline for the same portfolio.",
    }

    for portfolio, random_portfolio in (("top1", "random_top1"), ("best5", "random_best5")):
        portfolio_row = milestones.get(portfolio, {})
        random_row = milestones.get(random_portfolio, {})
        portfolio_counts = portfolio_row.get("count", {})
        random_counts = random_row.get("count", {})
        portfolio_rates = portfolio_row.get("rate", {})
        random_rates = random_row.get("rate", {})

        pattern_rows: dict[str, object] = {}
        for pattern in RELEASE_PATTERNS:
            count = int(portfolio_counts.get(pattern, 0))
            random_count = int(random_counts.get(pattern, 0))
            rate = float(portfolio_rates.get(pattern, 0.0))
            random_rate = float(random_rates.get(pattern, 0.0))
            pattern_rows[pattern] = {
                "count": count,
                "random_count": random_count,
                "rate": rate,
                "random_rate": random_rate,
                "lift_count": count - random_count,
                "lift_rate": round(rate - random_rate, 4),
                "window_pass": count > 0 and count > random_count,
            }

        release_gate[portfolio] = {
            "patterns": pattern_rows,
            "window_pass": all(pattern_rows[pattern]["window_pass"] for pattern in RELEASE_PATTERNS),
        }

    return release_gate


def run_backtest(
    records: list[DrawRecord],
    feature_tables: dict[str, pd.DataFrame],
    config: BacktestConfig,
    strategy: StrategySpec,
) -> dict[str, object]:
    test_records = [
        record
        for record in records
        if config.test_start <= record.draw_date <= config.test_end
    ]
    if not test_records:
        raise ValueError("No test records were found for the requested backtest window")

    frozen_context = None
    if config.mode == "frozen":
        frozen_train_records = [record for record in records if record.draw_date <= config.train_end]
        LOGGER.info(
            "Training frozen %s strategy on %s draws through %s",
            strategy.name,
            len(frozen_train_records),
            config.train_end.isoformat(),
        )
        frozen_context = strategy.build_context(
            records=records,
            feature_tables=feature_tables,
            train_cutoff=config.train_end,
            random_state=config.random_state,
        )

    draw_results: list[DrawBacktestResult] = []
    top1_weighted_hits: list[int] = []
    top1_main_hits: list[int] = []
    top1_star_hits: list[int] = []
    best5_weighted_hits: list[int] = []
    best5_main_hits: list[int] = []
    best5_star_hits: list[int] = []
    top1_scores: list[dict[str, int]] = []
    best5_scores: list[dict[str, int]] = []
    random_top1_weighted_hits: list[int] = []
    random_top1_main_hits: list[int] = []
    random_top1_star_hits: list[int] = []
    random_best5_weighted_hits: list[int] = []
    random_best5_main_hits: list[int] = []
    random_best5_star_hits: list[int] = []
    random_top1_scores: list[dict[str, int]] = []
    random_best5_scores: list[dict[str, int]] = []

    for draw_index, record in enumerate(test_records, start=1):
        if draw_index == 1 or draw_index % 10 == 0 or draw_index == len(test_records):
            LOGGER.info(
                "Backtest progress %s/%s draws: scoring %s",
                draw_index,
                len(test_records),
                record.draw_date.isoformat(),
            )
        if config.mode == "rolling":
            train_cutoff = date.fromordinal(record.draw_date.toordinal() - 1)
            LOGGER.info("Retraining rolling models through %s", train_cutoff.isoformat())
            context = strategy.build_context(
                records=records,
                feature_tables=feature_tables,
                train_cutoff=train_cutoff,
                random_state=config.random_state,
            )
        else:
            context = frozen_context

        assert context is not None

        tickets = strategy.generate_tickets_from_context(
            context=context,
            feature_tables=feature_tables,
            target_record=record,
            top_k=config.top_k,
            sample_count=config.sample_count,
            random_state=config.random_state + record.draw_id,
        )

        evaluated_predictions: list[dict[str, object]] = []
        per_ticket_scores: list[dict[str, int]] = []
        for rank, ticket in enumerate(tickets, start=1):
            metrics = evaluate_ticket(ticket, record)
            prediction = {
                "rank": rank,
                "main_numbers": list(ticket.main_numbers),
                "star_numbers": list(ticket.star_numbers),
                "score": round(ticket.score, 6),
                "confidence": ticket.confidence,
                **metrics,
            }
            evaluated_predictions.append(prediction)
            per_ticket_scores.append(metrics)

        draw_results.append(
            DrawBacktestResult(
                draw_date=record.draw_date.isoformat(),
                actual_main_numbers=record.main_numbers,
                actual_star_numbers=record.star_numbers,
                predictions=evaluated_predictions,
            )
        )

        if not per_ticket_scores:
            continue

        top1 = per_ticket_scores[0]
        top1_main_hits.append(top1["main_hits"])
        top1_star_hits.append(top1["star_hits"])
        top1_weighted_hits.append(top1["weighted_hits"])
        top1_scores.append(top1)

        best5 = max(
            per_ticket_scores,
            key=lambda item: (item["weighted_hits"], item["main_hits"], item["star_hits"]),
        )
        best5_main_hits.append(best5["main_hits"])
        best5_star_hits.append(best5["star_hits"])
        best5_weighted_hits.append(best5["weighted_hits"])
        best5_scores.append(best5)

        random_scores = _random_ticket_scores(record, config.top_k, config.random_state + record.draw_id)
        random_top1 = random_scores[0]
        random_top1_main_hits.append(random_top1["main_hits"])
        random_top1_star_hits.append(random_top1["star_hits"])
        random_top1_weighted_hits.append(random_top1["weighted_hits"])
        random_top1_scores.append(random_top1)
        random_best5 = max(
            random_scores,
            key=lambda item: (item["weighted_hits"], item["main_hits"], item["star_hits"]),
        )
        random_best5_main_hits.append(random_best5["main_hits"])
        random_best5_star_hits.append(random_best5["star_hits"])
        random_best5_weighted_hits.append(random_best5["weighted_hits"])
        random_best5_scores.append(random_best5)

    summary = {
        "strategy": strategy.name,
        "mode": config.mode,
        "train_end": config.train_end.isoformat(),
        "test_start": config.test_start.isoformat(),
        "test_end": config.test_end.isoformat(),
        "draw_count": len(draw_results),
        "top1_avg_main_hits": round(mean(top1_main_hits), 4) if top1_main_hits else 0.0,
        "top1_avg_star_hits": round(mean(top1_star_hits), 4) if top1_star_hits else 0.0,
        "top1_avg_weighted_hits": round(mean(top1_weighted_hits), 4) if top1_weighted_hits else 0.0,
        "best5_avg_main_hits": round(mean(best5_main_hits), 4) if best5_main_hits else 0.0,
        "best5_avg_star_hits": round(mean(best5_star_hits), 4) if best5_star_hits else 0.0,
        "best5_avg_weighted_hits": round(mean(best5_weighted_hits), 4) if best5_weighted_hits else 0.0,
        "random_top1_avg_main_hits": round(mean(random_top1_main_hits), 4) if random_top1_main_hits else 0.0,
        "random_top1_avg_star_hits": round(mean(random_top1_star_hits), 4) if random_top1_star_hits else 0.0,
        "random_top1_avg_weighted_hits": round(mean(random_top1_weighted_hits), 4) if random_top1_weighted_hits else 0.0,
        "random_best5_avg_main_hits": round(mean(random_best5_main_hits), 4) if random_best5_main_hits else 0.0,
        "random_best5_avg_star_hits": round(mean(random_best5_star_hits), 4) if random_best5_star_hits else 0.0,
        "random_best5_avg_weighted_hits": round(mean(random_best5_weighted_hits), 4) if random_best5_weighted_hits else 0.0,
        "lift_top1_weighted_hits": round((mean(top1_weighted_hits) - mean(random_top1_weighted_hits)), 4) if top1_weighted_hits else 0.0,
        "lift_best5_weighted_hits": round((mean(best5_weighted_hits) - mean(random_best5_weighted_hits)), 4) if best5_weighted_hits else 0.0,
        "top1_exact_5_2_hits": sum(
            1
            for result in draw_results
            if result.predictions
            and result.predictions[0]["main_hits"] == 5
            and result.predictions[0]["star_hits"] == 2
        ),
        "best5_exact_5_2_hits": sum(
            1
            for result in draw_results
            if any(
                prediction["main_hits"] == 5 and prediction["star_hits"] == 2
                for prediction in result.predictions
            )
        ),
    }

    milestones = _aggregate_milestones(
        top1_scores=top1_scores,
        best5_scores=best5_scores,
        random_top1_scores=random_top1_scores,
        random_best5_scores=random_best5_scores,
    )

    report = {
        "summary": summary,
        "milestones": milestones,
        "release_gate": _build_release_gate(milestones),
        "draws": [
            {
                "draw_date": result.draw_date,
                "actual_main_numbers": list(result.actual_main_numbers),
                "actual_star_numbers": list(result.actual_star_numbers),
                "predictions": result.predictions,
            }
            for result in draw_results
        ],
    }
    if config.mode == "frozen" and frozen_context is not None:
        report["model_diagnostics"] = strategy.diagnostics_from_context(frozen_context)
    return report


def _random_ticket_scores(record: DrawRecord, top_k: int, seed: int) -> list[dict[str, int]]:
    rng = np.random.default_rng(seed)
    scores: list[dict[str, int]] = []
    for rank in range(top_k):
        main_numbers = tuple(sorted(int(value) for value in rng.choice(np.arange(1, record.spec.main_pool_size + 1), size=record.spec.main_pick_count, replace=False)))
        star_numbers = tuple(sorted(int(value) for value in rng.choice(np.arange(1, record.spec.star_pool_size + 1), size=record.spec.star_pick_count, replace=False)))
        ticket = CandidateTicket(
            main_numbers=main_numbers,
            star_numbers=star_numbers,
            score=0.0,
            confidence=1,
        )
        scores.append(evaluate_ticket(ticket, record))
    return scores


def predict_for_draw(
    records: list[DrawRecord],
    feature_tables: dict[str, pd.DataFrame],
    target_record: DrawRecord,
    train_cutoff: date,
    strategy: StrategySpec,
    top_k: int = 5,
    sample_count: int = 5000,
    random_state: int = 7,
) -> list[CandidateTicket]:
    context = strategy.build_context(
        records=records,
        feature_tables=feature_tables,
        train_cutoff=train_cutoff,
        random_state=random_state,
    )
    return strategy.generate_tickets_from_context(
        context=context,
        feature_tables=feature_tables,
        target_record=target_record,
        top_k=top_k,
        sample_count=sample_count,
        random_state=random_state + target_record.draw_id,
    )
