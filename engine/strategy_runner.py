from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

import pandas as pd

from engine.data import DrawRecord
from engine.tickets import CandidateTicket

ModelFactory = Callable[[int], Any]
TicketGenerator = Callable[..., list[CandidateTicket]]


@dataclass(frozen=True)
class DefaultStrategyContext:
    train_cutoff: date
    train_records: list[DrawRecord]
    main_model: Any
    star_model: Any


def _train_rows(feature_table: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    return feature_table[feature_table["draw_ordinal"] <= cutoff.toordinal()]


def _draw_rows(feature_table: pd.DataFrame, draw_date: date) -> pd.DataFrame:
    return feature_table[feature_table["draw_ordinal"] == draw_date.toordinal()]


def build_default_context(
    records: list[DrawRecord],
    feature_tables: dict[str, pd.DataFrame],
    train_cutoff: date,
    main_factory: ModelFactory,
    star_factory: ModelFactory,
    random_state: int,
) -> DefaultStrategyContext:
    train_records = [record for record in records if record.draw_date <= train_cutoff]
    main_model = main_factory(random_state=random_state).fit(_train_rows(feature_tables["main"], train_cutoff))
    star_model = star_factory(random_state=random_state).fit(_train_rows(feature_tables["star"], train_cutoff))
    return DefaultStrategyContext(
        train_cutoff=train_cutoff,
        train_records=train_records,
        main_model=main_model,
        star_model=star_model,
    )


def default_context_diagnostics(context: DefaultStrategyContext) -> dict[str, object]:
    return {
        "main": context.main_model.diagnostics,
        "star": context.star_model.diagnostics,
    }


def build_probability_maps(
    context: DefaultStrategyContext,
    feature_tables: dict[str, pd.DataFrame],
    target_record: DrawRecord,
) -> tuple[dict[int, float], dict[int, float]]:
    main_rows = _draw_rows(feature_tables["main"], target_record.draw_date)
    star_rows = _draw_rows(feature_tables["star"], target_record.draw_date)
    main_probabilities = dict(
        zip(
            main_rows["candidate"].astype(int),
            context.main_model.predict_proba(main_rows),
            strict=True,
        )
    )
    star_probabilities = dict(
        zip(
            star_rows["candidate"].astype(int),
            context.star_model.predict_proba(star_rows),
            strict=True,
        )
    )
    return main_probabilities, star_probabilities


def generate_tickets_from_default_context(
    context: DefaultStrategyContext,
    feature_tables: dict[str, pd.DataFrame],
    target_record: DrawRecord,
    ticket_generator: TicketGenerator,
    top_k: int,
    sample_count: int,
    random_state: int,
) -> list[CandidateTicket]:
    main_probabilities, star_probabilities = build_probability_maps(context, feature_tables, target_record)
    return ticket_generator(
        main_probabilities=main_probabilities,
        star_probabilities=star_probabilities,
        spec=target_record.spec,
        train_records=context.train_records,
        top_k=top_k,
        sample_count=sample_count,
        random_state=random_state,
    )
