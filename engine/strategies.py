from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from engine.models import FamilyBlendProbabilityModel, InclusionProbabilityModel, MultiHistoryInclusionProbabilityModel
from engine.strategy_runner import (
    build_default_context,
    default_context_diagnostics,
    generate_tickets_from_default_context,
)
from engine.tickets import (
    CandidateTicket,
    generate_candidate_tickets,
    generate_conditional_two_stage_tickets,
    generate_conditional_rerank_tickets,
    generate_core_plus_guard_tickets,
    generate_selector_ensemble_tickets,
    generate_hybrid_conditional_tickets,
    generate_adaptive_soft_star_guard_screen_tickets,
    generate_soft_star_guard_screen_tickets,
    generate_soft_star_guard_rerank_tickets,
    generate_support_gated_star_guard_screen_050_tickets,
    generate_support_gated_star_guard_screen_060_tickets,
    generate_star_guard1_rerank_tickets,
    generate_star_guard1_soft_screen_tickets,
    generate_star_guard1_soft_rerank_tickets,
    generate_star_guard_rerank_tickets,
    generate_two_stage_star_tickets,
)
from engine.data import DrawRecord

ProbabilityModel = InclusionProbabilityModel | MultiHistoryInclusionProbabilityModel | FamilyBlendProbabilityModel
ModelFactory = Callable[[int], ProbabilityModel]
TicketGenerator = Callable[..., list[CandidateTicket]]
ContextBuilder = Callable[[list[DrawRecord], dict[str, Any], date, int], Any]
ContextTicketGenerator = Callable[[Any, dict[str, Any], DrawRecord, int, int, int], list[CandidateTicket]]
ContextDiagnostics = Callable[[Any], dict[str, object]]


@dataclass(frozen=True)
class StrategySpec:
    name: str
    description: str
    signature: str
    main_factory: ModelFactory
    star_factory: ModelFactory
    ticket_generator: TicketGenerator = generate_candidate_tickets
    context_builder: ContextBuilder | None = None
    context_ticket_generator: ContextTicketGenerator | None = None
    context_diagnostics: ContextDiagnostics | None = None

    def create_main_model(self, random_state: int) -> ProbabilityModel:
        return self.main_factory(random_state)

    def create_star_model(self, random_state: int) -> ProbabilityModel:
        return self.star_factory(random_state)

    def build_context(
        self,
        records: list[DrawRecord],
        feature_tables: dict[str, Any],
        train_cutoff: date,
        random_state: int,
    ) -> Any:
        if self.context_builder is not None:
            return self.context_builder(records, feature_tables, train_cutoff, random_state)
        return build_default_context(
            records=records,
            feature_tables=feature_tables,
            train_cutoff=train_cutoff,
            main_factory=self.main_factory,
            star_factory=self.star_factory,
            random_state=random_state,
        )

    def generate_tickets_from_context(
        self,
        context: Any,
        feature_tables: dict[str, Any],
        target_record: DrawRecord,
        top_k: int,
        sample_count: int,
        random_state: int,
    ) -> list[CandidateTicket]:
        if self.context_ticket_generator is not None:
            return self.context_ticket_generator(
                context,
                feature_tables,
                target_record,
                top_k,
                sample_count,
                random_state,
            )
        return generate_tickets_from_default_context(
            context=context,
            feature_tables=feature_tables,
            target_record=target_record,
            ticket_generator=self.ticket_generator,
            top_k=top_k,
            sample_count=sample_count,
            random_state=random_state,
        )

    def diagnostics_from_context(self, context: Any) -> dict[str, object]:
        if self.context_diagnostics is not None:
            return self.context_diagnostics(context)
        return default_context_diagnostics(context)


def _baseline_factory(random_state: int) -> InclusionProbabilityModel:
    return InclusionProbabilityModel(random_state=random_state)


def _multi_history_factory(random_state: int) -> MultiHistoryInclusionProbabilityModel:
    return MultiHistoryInclusionProbabilityModel(random_state=random_state)


def _star_focus_factory(random_state: int) -> MultiHistoryInclusionProbabilityModel:
    return MultiHistoryInclusionProbabilityModel(
        random_state=random_state,
        history_windows=(None, 260, 104, 52, 26),
    )


def _hybrid_main_factory(random_state: int) -> FamilyBlendProbabilityModel:
    return FamilyBlendProbabilityModel(
        random_state=random_state,
        factories=(
            ("baseline", _baseline_factory),
            ("multi_history", _multi_history_factory),
        ),
    )


def _hybrid_star_factory(random_state: int) -> FamilyBlendProbabilityModel:
    return FamilyBlendProbabilityModel(
        random_state=random_state,
        factories=(
            ("baseline", _baseline_factory),
            ("star_focus", _star_focus_factory),
        ),
    )


def _ticket_key(ticket: CandidateTicket) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return ticket.main_numbers, ticket.star_numbers


def _overlap_cost(left: CandidateTicket, right: CandidateTicket) -> float:
    main_overlap = len(set(left.main_numbers).intersection(right.main_numbers))
    star_overlap = len(set(left.star_numbers).intersection(right.star_numbers))
    return 0.35 * main_overlap + 0.65 * star_overlap


def _select_committee_from_source(
    candidates: list[CandidateTicket],
    selected: list[CandidateTicket],
    quota: int,
) -> list[CandidateTicket]:
    chosen: list[CandidateTicket] = []
    remaining = list(candidates)
    used_keys = {_ticket_key(ticket) for ticket in selected}

    while remaining and len(chosen) < quota:
        best_ticket = max(
            remaining,
            key=lambda ticket: (
                ticket.score
                - 0.42 * max((_overlap_cost(ticket, existing) for existing in selected + chosen), default=0.0),
                -sum(ticket.main_numbers),
                -sum(ticket.star_numbers),
            ),
        )
        remaining.remove(best_ticket)
        ticket_key = _ticket_key(best_ticket)
        if ticket_key in used_keys:
            continue
        chosen.append(best_ticket)
        used_keys.add(ticket_key)

    return chosen


def _build_committee_context(
    records: list[DrawRecord],
    feature_tables: dict[str, Any],
    train_cutoff: date,
    random_state: int,
) -> dict[str, Any]:
    component_specs = {
        "guarded": STRATEGIES["star_guard1_soft_screen_multi_history"],
        "hybrid": STRATEGIES["hybrid_main_star_focus_soft_guard_screen"],
        "baseline": STRATEGIES["baseline"],
    }
    component_contexts: dict[str, Any] = {}
    component_diagnostics: dict[str, object] = {}
    for index, (label, spec) in enumerate(component_specs.items(), start=1):
        context = build_default_context(
            records=records,
            feature_tables=feature_tables,
            train_cutoff=train_cutoff,
            main_factory=spec.main_factory,
            star_factory=spec.star_factory,
            random_state=random_state + 101 * index,
        )
        component_contexts[label] = {
            "strategy": spec,
            "context": context,
        }
        diagnostics = default_context_diagnostics(context)
        for component_name, component_diag in diagnostics.items():
            component_diagnostics[f"{label}.{component_name}"] = component_diag

    return {
        "components": component_contexts,
        "diagnostics": component_diagnostics,
    }


def _generate_committee_guarded_hybrid_tickets(
    context: dict[str, Any],
    feature_tables: dict[str, Any],
    target_record: DrawRecord,
    top_k: int,
    sample_count: int,
    random_state: int,
) -> list[CandidateTicket]:
    component_requests = (
        ("guarded", 2, 4),
        ("hybrid", 2, 4),
        ("baseline", 1, 3),
    )
    source_candidates: dict[str, list[CandidateTicket]] = {}
    for index, (label, _, component_top_k) in enumerate(component_requests, start=1):
        payload = context["components"][label]
        strategy = payload["strategy"]
        source_candidates[label] = generate_tickets_from_default_context(
            context=payload["context"],
            feature_tables=feature_tables,
            target_record=target_record,
            ticket_generator=strategy.ticket_generator,
            top_k=component_top_k,
            sample_count=sample_count,
            random_state=random_state + 503 * index,
        )

    selected: list[CandidateTicket] = []
    for label, quota, _ in component_requests:
        selected.extend(_select_committee_from_source(source_candidates[label], selected, quota))
        if len(selected) >= top_k:
            return selected[:top_k]

    fallback_pool: list[CandidateTicket] = []
    selected_keys = {_ticket_key(ticket) for ticket in selected}
    for label, _, _ in component_requests:
        for ticket in source_candidates[label]:
            ticket_key = _ticket_key(ticket)
            if ticket_key in selected_keys:
                continue
            fallback_pool.append(ticket)
            selected_keys.add(ticket_key)

    selected.extend(_select_committee_from_source(fallback_pool, selected, max(0, top_k - len(selected))))
    return selected[:top_k]


def _select_specialist_candidate(
    candidates: list[CandidateTicket],
    selected: list[CandidateTicket],
    fallback_ticket: CandidateTicket,
    score_drop_limit: float,
) -> CandidateTicket | None:
    chosen: CandidateTicket | None = None
    chosen_score = float("-inf")
    selected_star_pairs = {ticket.star_numbers for ticket in selected}
    selected_keys = {_ticket_key(ticket) for ticket in selected}

    for candidate in candidates:
        candidate_key = _ticket_key(candidate)
        if candidate_key in selected_keys:
            continue
        novelty_bonus = 0.12 if candidate.star_numbers not in selected_star_pairs else 0.0
        overlap_penalty = max((_overlap_cost(candidate, existing) for existing in selected), default=0.0)
        adjusted_score = candidate.score + novelty_bonus - 0.45 * overlap_penalty
        if adjusted_score > chosen_score:
            chosen = candidate
            chosen_score = adjusted_score

    if chosen is None:
        return None
    if chosen.score + 0.08 < fallback_ticket.score - score_drop_limit:
        return None
    return chosen


def _generate_baseline_core_dual_specialist_tickets(
    context: dict[str, Any],
    feature_tables: dict[str, Any],
    target_record: DrawRecord,
    top_k: int,
    sample_count: int,
    random_state: int,
) -> list[CandidateTicket]:
    component_top_k = {
        "baseline": max(top_k, 5),
        "hybrid": 3,
        "guarded": 3,
    }
    source_candidates: dict[str, list[CandidateTicket]] = {}
    for index, label in enumerate(("baseline", "hybrid", "guarded"), start=1):
        payload = context["components"][label]
        strategy = payload["strategy"]
        source_candidates[label] = generate_tickets_from_default_context(
            context=payload["context"],
            feature_tables=feature_tables,
            target_record=target_record,
            ticket_generator=strategy.ticket_generator,
            top_k=component_top_k[label],
            sample_count=sample_count,
            random_state=random_state + 607 * index,
        )

    baseline_portfolio = source_candidates["baseline"][:top_k]
    if len(baseline_portfolio) <= 3:
        return baseline_portfolio

    selected = list(baseline_portfolio[:3])
    fallback_tail = list(baseline_portfolio[3:])

    hybrid_fallback = fallback_tail[0] if fallback_tail else baseline_portfolio[-1]
    hybrid_choice = _select_specialist_candidate(
        candidates=source_candidates["hybrid"],
        selected=selected,
        fallback_ticket=hybrid_fallback,
        score_drop_limit=0.24,
    )
    selected.append(hybrid_choice or hybrid_fallback)

    remaining_fallback = [ticket for ticket in fallback_tail if _ticket_key(ticket) not in {_ticket_key(item) for item in selected}]
    guard_fallback = remaining_fallback[0] if remaining_fallback else baseline_portfolio[-1]
    guard_choice = _select_specialist_candidate(
        candidates=source_candidates["guarded"],
        selected=selected,
        fallback_ticket=guard_fallback,
        score_drop_limit=0.18,
    )
    selected.append(guard_choice or guard_fallback)

    selected_keys = {_ticket_key(ticket) for ticket in selected}
    for ticket in baseline_portfolio:
        ticket_key = _ticket_key(ticket)
        if ticket_key in selected_keys:
            continue
        selected.append(ticket)
        selected_keys.add(ticket_key)
        if len(selected) >= top_k:
            break

    return selected[:top_k]


def _committee_context_diagnostics(context: dict[str, Any]) -> dict[str, object]:
    return context["diagnostics"]


STRATEGIES: dict[str, StrategySpec] = {
    "baseline": StrategySpec(
        name="baseline",
        description="Promoted baseline: inclusion ensemble for mains and stars with the default ticket selector.",
        signature="strategy-baseline-v1",
        main_factory=_baseline_factory,
        star_factory=_baseline_factory,
    ),
    "multi_history": StrategySpec(
        name="multi_history",
        description="Blend long and recent history windows for both mains and stars.",
        signature="strategy-multi-history-v1",
        main_factory=_multi_history_factory,
        star_factory=_multi_history_factory,
    ),
    "star_focus": StrategySpec(
        name="star_focus",
        description="Keep the baseline main-number model and use a multi-history specialist for Lucky Stars.",
        signature="strategy-star-focus-v1",
        main_factory=_baseline_factory,
        star_factory=_star_focus_factory,
    ),
    "two_stage_baseline": StrategySpec(
        name="two_stage_baseline",
        description="Use the promoted inclusion models but generate the ticket portfolio through a star-first two-stage search.",
        signature="strategy-two-stage-baseline-v1",
        main_factory=_baseline_factory,
        star_factory=_baseline_factory,
        ticket_generator=generate_two_stage_star_tickets,
    ),
    "two_stage_star": StrategySpec(
        name="two_stage_star",
        description="Combine the star-first two-stage search with the multi-history Lucky Star specialist.",
        signature="strategy-two-stage-star-v1",
        main_factory=_baseline_factory,
        star_factory=_star_focus_factory,
        ticket_generator=generate_two_stage_star_tickets,
    ),
    "two_stage_multi_history": StrategySpec(
        name="two_stage_multi_history",
        description="Multi-history mains and stars with star-pair-first beam portfolio search.",
        signature="strategy-two-stage-multi-history-v1",
        main_factory=_multi_history_factory,
        star_factory=_multi_history_factory,
        ticket_generator=generate_two_stage_star_tickets,
    ),
    "conditional_two_stage_multi_history": StrategySpec(
        name="conditional_two_stage_multi_history",
        description="Multi-history mains and stars with star-conditioned main rescoring and star-pair-first beam search.",
        signature="strategy-conditional-two-stage-multi-history-v1",
        main_factory=_multi_history_factory,
        star_factory=_multi_history_factory,
        ticket_generator=generate_conditional_two_stage_tickets,
    ),
    "hybrid_conditional_multi_history": StrategySpec(
        name="hybrid_conditional_multi_history",
        description="Blend the stronger global multi-history portfolio with conditional star-conditioned candidates before final selection.",
        signature="strategy-hybrid-conditional-multi-history-v1",
        main_factory=_multi_history_factory,
        star_factory=_multi_history_factory,
        ticket_generator=generate_hybrid_conditional_tickets,
    ),
    "conditional_rerank_multi_history": StrategySpec(
        name="conditional_rerank_multi_history",
        description="Use the strong global multi-history pool and rerank it with a star-conditioned main-number bonus.",
        signature="strategy-conditional-rerank-multi-history-v1",
        main_factory=_multi_history_factory,
        star_factory=_multi_history_factory,
        ticket_generator=generate_conditional_rerank_tickets,
    ),
    "star_guard_rerank_multi_history": StrategySpec(
        name="star_guard_rerank_multi_history",
        description="Reserve strong distinct-star global tickets, then fill the rest with the conditional reranker.",
        signature="strategy-star-guard-rerank-multi-history-v1",
        main_factory=_multi_history_factory,
        star_factory=_multi_history_factory,
        ticket_generator=generate_star_guard_rerank_tickets,
    ),
    "star_guard1_rerank_multi_history": StrategySpec(
        name="star_guard1_rerank_multi_history",
        description="Keep only one guarded distinct-star global ticket before the conditional reranker fills the portfolio.",
        signature="strategy-star-guard1-rerank-multi-history-v1",
        main_factory=_multi_history_factory,
        star_factory=_multi_history_factory,
        ticket_generator=generate_star_guard1_rerank_tickets,
    ),
    "star_guard1_soft_rerank_multi_history": StrategySpec(
        name="star_guard1_soft_rerank_multi_history",
        description="One guarded distinct-star global ticket with a softer conditional rerank weight.",
        signature="strategy-star-guard1-soft-rerank-multi-history-v1",
        main_factory=_multi_history_factory,
        star_factory=_multi_history_factory,
        ticket_generator=generate_star_guard1_soft_rerank_tickets,
    ),
    "soft_star_guard_rerank_multi_history": StrategySpec(
        name="soft_star_guard_rerank_multi_history",
        description="Boost top distinct-star global tickets during conditional reranking instead of forcing them into the final portfolio.",
        signature="strategy-soft-star-guard-rerank-multi-history-v1",
        main_factory=_multi_history_factory,
        star_factory=_multi_history_factory,
        ticket_generator=generate_soft_star_guard_rerank_tickets,
    ),
    "star_guard1_soft_screen_multi_history": StrategySpec(
        name="star_guard1_soft_screen_multi_history",
        description="Experiment screen: one guarded distinct-star ticket with softer rerank weight and a smaller rerank pool.",
        signature="strategy-star-guard1-soft-screen-multi-history-v1",
        main_factory=_multi_history_factory,
        star_factory=_multi_history_factory,
        ticket_generator=generate_star_guard1_soft_screen_tickets,
    ),
    "soft_star_guard_screen_multi_history": StrategySpec(
        name="soft_star_guard_screen_multi_history",
        description="Experiment screen: softly boost distinct-star candidates with a smaller rerank pool.",
        signature="strategy-soft-star-guard-screen-multi-history-v1",
        main_factory=_multi_history_factory,
        star_factory=_multi_history_factory,
        ticket_generator=generate_soft_star_guard_screen_tickets,
    ),
    "support_gated_star_guard_screen_050_multi_history": StrategySpec(
        name="support_gated_star_guard_screen_050_multi_history",
        description="Experiment screen: only activate the one-guard soft rerank path when the guarded star pair support is at least 0.50.",
        signature="strategy-support-gated-star-guard-screen-050-multi-history-v1",
        main_factory=_multi_history_factory,
        star_factory=_multi_history_factory,
        ticket_generator=generate_support_gated_star_guard_screen_050_tickets,
    ),
    "support_gated_star_guard_screen_060_multi_history": StrategySpec(
        name="support_gated_star_guard_screen_060_multi_history",
        description="Experiment screen: only activate the one-guard soft rerank path when the guarded star pair support is at least 0.60.",
        signature="strategy-support-gated-star-guard-screen-060-multi-history-v1",
        main_factory=_multi_history_factory,
        star_factory=_multi_history_factory,
        ticket_generator=generate_support_gated_star_guard_screen_060_tickets,
    ),
    "support_gated_star_guard_screen_050_baseline": StrategySpec(
        name="support_gated_star_guard_screen_050_baseline",
        description="Baseline probability models with the support-gated one-guard soft screen selector at a 0.50 threshold.",
        signature="strategy-support-gated-star-guard-screen-050-baseline-v1",
        main_factory=_baseline_factory,
        star_factory=_baseline_factory,
        ticket_generator=generate_support_gated_star_guard_screen_050_tickets,
    ),
    "support_gated_star_guard_screen_060_baseline": StrategySpec(
        name="support_gated_star_guard_screen_060_baseline",
        description="Baseline probability models with the support-gated one-guard soft screen selector at a 0.60 threshold.",
        signature="strategy-support-gated-star-guard-screen-060-baseline-v1",
        main_factory=_baseline_factory,
        star_factory=_baseline_factory,
        ticket_generator=generate_support_gated_star_guard_screen_060_tickets,
    ),
    "adaptive_soft_star_guard_screen_multi_history": StrategySpec(
        name="adaptive_soft_star_guard_screen_multi_history",
        description="Activate the softer star-guard screen only when exact star-pair evidence and pair separation are both strong enough.",
        signature="strategy-adaptive-soft-star-guard-screen-multi-history-v1",
        main_factory=_multi_history_factory,
        star_factory=_multi_history_factory,
        ticket_generator=generate_adaptive_soft_star_guard_screen_tickets,
    ),
    "adaptive_soft_star_guard_screen_baseline": StrategySpec(
        name="adaptive_soft_star_guard_screen_baseline",
        description="Baseline probability models with the adaptive soft star-guard screen.",
        signature="strategy-adaptive-soft-star-guard-screen-baseline-v1",
        main_factory=_baseline_factory,
        star_factory=_baseline_factory,
        ticket_generator=generate_adaptive_soft_star_guard_screen_tickets,
    ),
    "core_plus_guard_multi_history": StrategySpec(
        name="core_plus_guard_multi_history",
        description="Keep a baseline-ranked core portfolio and allow one guarded specialist ticket when star-pair evidence is strong enough.",
        signature="strategy-core-plus-guard-multi-history-v1",
        main_factory=_multi_history_factory,
        star_factory=_multi_history_factory,
        ticket_generator=generate_core_plus_guard_tickets,
    ),
    "core_plus_guard_baseline": StrategySpec(
        name="core_plus_guard_baseline",
        description="Baseline probability models with a baseline-core plus one guarded specialist ticket.",
        signature="strategy-core-plus-guard-baseline-v1",
        main_factory=_baseline_factory,
        star_factory=_baseline_factory,
        ticket_generator=generate_core_plus_guard_tickets,
    ),
    "star_focus_soft_guard_screen": StrategySpec(
        name="star_focus_soft_guard_screen",
        description="Baseline main-number model with a multi-history Lucky Star specialist and the one-guard soft screen selector.",
        signature="strategy-star-focus-soft-guard-screen-v1",
        main_factory=_baseline_factory,
        star_factory=_star_focus_factory,
        ticket_generator=generate_star_guard1_soft_screen_tickets,
    ),
    "star_focus_core_plus_guard": StrategySpec(
        name="star_focus_core_plus_guard",
        description="Baseline main-number model with a multi-history Lucky Star specialist and a baseline-core plus guarded-ticket portfolio.",
        signature="strategy-star-focus-core-plus-guard-v1",
        main_factory=_baseline_factory,
        star_factory=_star_focus_factory,
        ticket_generator=generate_core_plus_guard_tickets,
    ),
    "hybrid_main_star_focus_soft_guard_screen": StrategySpec(
        name="hybrid_main_star_focus_soft_guard_screen",
        description="Blend baseline and multi-history main models, keep the star specialist, and use the one-guard soft screen selector.",
        signature="strategy-hybrid-main-star-focus-soft-guard-screen-v1",
        main_factory=_hybrid_main_factory,
        star_factory=_star_focus_factory,
        ticket_generator=generate_star_guard1_soft_screen_tickets,
    ),
    "hybrid_main_hybrid_star_soft_guard_screen": StrategySpec(
        name="hybrid_main_hybrid_star_soft_guard_screen",
        description="Blend baseline and multi-history models for mains, blend baseline and star-focused models for stars, and use the one-guard soft screen selector.",
        signature="strategy-hybrid-main-hybrid-star-soft-guard-screen-v1",
        main_factory=_hybrid_main_factory,
        star_factory=_hybrid_star_factory,
        ticket_generator=generate_star_guard1_soft_screen_tickets,
    ),
    "hybrid_main_star_focus_core_plus_guard": StrategySpec(
        name="hybrid_main_star_focus_core_plus_guard",
        description="Blend baseline and multi-history main models with the star specialist and a baseline-core plus guarded-ticket portfolio.",
        signature="strategy-hybrid-main-star-focus-core-plus-guard-v1",
        main_factory=_hybrid_main_factory,
        star_factory=_star_focus_factory,
        ticket_generator=generate_core_plus_guard_tickets,
    ),
    "committee_guarded_hybrid": StrategySpec(
        name="committee_guarded_hybrid",
        description="Committee portfolio: guarded 2025 expert, hybrid 2026 expert, and one conservative baseline ticket.",
        signature="strategy-committee-guarded-hybrid-v1",
        main_factory=_baseline_factory,
        star_factory=_baseline_factory,
        context_builder=_build_committee_context,
        context_ticket_generator=_generate_committee_guarded_hybrid_tickets,
        context_diagnostics=_committee_context_diagnostics,
    ),
    "baseline_core_dual_specialist": StrategySpec(
        name="baseline_core_dual_specialist",
        description="Keep a baseline 3-ticket core, then allow one hybrid star-focused specialist and one guarded specialist if they clear score and diversity gates.",
        signature="strategy-baseline-core-dual-specialist-v1",
        main_factory=_baseline_factory,
        star_factory=_baseline_factory,
        context_builder=_build_committee_context,
        context_ticket_generator=_generate_baseline_core_dual_specialist_tickets,
        context_diagnostics=_committee_context_diagnostics,
    ),
    "hybrid_main_star_focus_selector_ensemble": StrategySpec(
        name="hybrid_main_star_focus_selector_ensemble",
        description="Blend baseline and multi-history main models, keep the star specialist, and combine safe, conditional, and guarded selector experts into one portfolio.",
        signature="strategy-hybrid-main-star-focus-selector-ensemble-v1",
        main_factory=_hybrid_main_factory,
        star_factory=_star_focus_factory,
        ticket_generator=generate_selector_ensemble_tickets,
    ),
}


def available_strategy_names() -> list[str]:
    return sorted(STRATEGIES)


def get_strategy(name: str) -> StrategySpec:
    try:
        return STRATEGIES[name]
    except KeyError as exc:
        available = ", ".join(available_strategy_names())
        raise ValueError(f"Unknown strategy '{name}'. Available strategies: {available}") from exc
