from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from engine.models import InclusionProbabilityModel, MultiHistoryInclusionProbabilityModel
from engine.tickets import (
    CandidateTicket,
    generate_candidate_tickets,
    generate_conditional_two_stage_tickets,
    generate_conditional_rerank_tickets,
    generate_hybrid_conditional_tickets,
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

ProbabilityModel = InclusionProbabilityModel | MultiHistoryInclusionProbabilityModel
ModelFactory = Callable[[int], ProbabilityModel]
TicketGenerator = Callable[..., list[CandidateTicket]]


@dataclass(frozen=True)
class StrategySpec:
    name: str
    description: str
    signature: str
    main_factory: ModelFactory
    star_factory: ModelFactory
    ticket_generator: TicketGenerator = generate_candidate_tickets

    def create_main_model(self, random_state: int) -> ProbabilityModel:
        return self.main_factory(random_state)

    def create_star_model(self, random_state: int) -> ProbabilityModel:
        return self.star_factory(random_state)


def _baseline_factory(random_state: int) -> InclusionProbabilityModel:
    return InclusionProbabilityModel(random_state=random_state)


def _multi_history_factory(random_state: int) -> MultiHistoryInclusionProbabilityModel:
    return MultiHistoryInclusionProbabilityModel(random_state=random_state)


def _star_focus_factory(random_state: int) -> MultiHistoryInclusionProbabilityModel:
    return MultiHistoryInclusionProbabilityModel(
        random_state=random_state,
        history_windows=(None, 260, 104, 52, 26),
    )


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
}


def available_strategy_names() -> list[str]:
    return sorted(STRATEGIES)


def get_strategy(name: str) -> StrategySpec:
    try:
        return STRATEGIES[name]
    except KeyError as exc:
        available = ", ".join(available_strategy_names())
        raise ValueError(f"Unknown strategy '{name}'. Available strategies: {available}") from exc
