from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from engine.data import DrawRecord
from engine.spec import GameSpec

RECENT_PRIOR_DRAWS = 520
DETERMINISTIC_MAIN_POOL = 14
DETERMINISTIC_STAR_POOL = 7
SELECTOR_VERSION = "v2_overlap_penalty"
TWO_STAGE_SELECTOR_VERSION = "v1_star_beam"
BEAM_MAIN_POOL = 18
BEAM_WIDTH = 48
STAR_PAIR_POOL = 6
CONDITIONAL_PRIOR_STRENGTH = 18.0
CONDITIONAL_SUPPORT_DRAWS = 60.0
EXACT_STAR_SUPPORT_DRAWS = 10.0
CONDITIONAL_RERANK_POOL = 2500
CONDITIONAL_RERANK_WEIGHT = 0.42
EXPERIMENTAL_RERANK_POOL = 800
STAR_GUARD_COUNT = 2
SUPPORT_GATE_THRESHOLD_050 = 0.50
SUPPORT_GATE_THRESHOLD_060 = 0.60


@dataclass(frozen=True)
class CandidateTicket:
    main_numbers: tuple[int, ...]
    star_numbers: tuple[int, ...]
    score: float
    confidence: int


@dataclass(frozen=True)
class ShapePriors:
    main_sum_log_probs: dict[int, float]
    star_sum_log_probs: dict[int, float]
    main_odd_log_probs: dict[int, float]
    star_odd_log_probs: dict[int, float]
    main_low_log_probs: dict[int, float]
    main_consecutive_log_probs: dict[int, float]
    main_range_log_probs: dict[int, float]

    def score(self, main_numbers: tuple[int, ...], star_numbers: tuple[int, ...]) -> float:
        main_sum = sum(main_numbers)
        star_sum = sum(star_numbers)
        main_odd = sum(number % 2 for number in main_numbers)
        star_odd = sum(number % 2 for number in star_numbers)
        main_low = sum(number <= 25 for number in main_numbers)
        main_consecutive = sum(
            1 for left, right in zip(main_numbers, main_numbers[1:]) if right - left == 1
        )
        main_range = main_numbers[-1] - main_numbers[0]

        score = 0.0
        score += 0.12 * self.main_sum_log_probs[main_sum]
        score += 0.08 * self.star_sum_log_probs[star_sum]
        score += 0.07 * self.main_odd_log_probs[main_odd]
        score += 0.05 * self.star_odd_log_probs[star_odd]
        score += 0.05 * self.main_low_log_probs[main_low]
        score += 0.04 * self.main_consecutive_log_probs[main_consecutive]
        score += 0.05 * self.main_range_log_probs[main_range]
        return score


def ticket_generator_signature() -> str:
    return SELECTOR_VERSION


def build_pair_bonus(records: list[DrawRecord], component: str, pool_size: int) -> np.ndarray:
    filtered_records = _filter_records_for_component(records, component, pool_size)
    pair_counts = np.ones((pool_size + 1, pool_size + 1), dtype=float)
    single_counts = np.ones(pool_size + 1, dtype=float)
    total_draws = max(1, len(filtered_records))

    for record in filtered_records:
        values = record.main_numbers if component == "main" else record.star_numbers
        for value in values:
            single_counts[value] += 1.0
        for left, right in combinations(values, 2):
            pair_counts[left, right] += 1.0
            pair_counts[right, left] += 1.0

    pair_bonus = np.zeros_like(pair_counts)
    for left in range(1, pool_size + 1):
        for right in range(left + 1, pool_size + 1):
            empirical_pair_rate = pair_counts[left, right] / total_draws
            empirical_independence = (single_counts[left] / total_draws) * (single_counts[right] / total_draws)
            bonus = np.log(empirical_pair_rate / empirical_independence)
            pair_bonus[left, right] = bonus
            pair_bonus[right, left] = bonus
    return pair_bonus


def build_shape_priors(records: list[DrawRecord], spec: GameSpec) -> ShapePriors:
    recent_records = records[-RECENT_PRIOR_DRAWS:] if len(records) > RECENT_PRIOR_DRAWS else records
    compatible_star_records = [record for record in recent_records if record.spec.star_pool_size == spec.star_pool_size]
    if not compatible_star_records:
        compatible_star_records = recent_records

    main_sum_counter = Counter(sum(record.main_numbers) for record in recent_records)
    star_sum_counter = Counter(sum(record.star_numbers) for record in compatible_star_records)
    main_odd_counter = Counter(sum(number % 2 for number in record.main_numbers) for record in recent_records)
    star_odd_counter = Counter(sum(number % 2 for number in record.star_numbers) for record in compatible_star_records)
    main_low_counter = Counter(sum(number <= 25 for number in record.main_numbers) for record in recent_records)
    main_consecutive_counter = Counter(
        sum(1 for left, right in zip(record.main_numbers, record.main_numbers[1:]) if right - left == 1)
        for record in recent_records
    )
    main_range_counter = Counter(record.main_numbers[-1] - record.main_numbers[0] for record in recent_records)

    return ShapePriors(
        main_sum_log_probs=_counter_to_log_probs(main_sum_counter, range(15, 241)),
        star_sum_log_probs=_counter_to_log_probs(star_sum_counter, range(3, spec.star_pool_size * 2)),
        main_odd_log_probs=_counter_to_log_probs(main_odd_counter, range(0, 6)),
        star_odd_log_probs=_counter_to_log_probs(star_odd_counter, range(0, 3)),
        main_low_log_probs=_counter_to_log_probs(main_low_counter, range(0, 6)),
        main_consecutive_log_probs=_counter_to_log_probs(main_consecutive_counter, range(0, 5)),
        main_range_log_probs=_counter_to_log_probs(main_range_counter, range(4, 50)),
    )


def _counter_to_log_probs(counter: Counter[int], values: range) -> dict[int, float]:
    smoothed_total = sum(counter.values()) + len(values)
    return {
        value: float(np.log((counter.get(value, 0) + 1) / smoothed_total))
        for value in values
    }


def _filter_records_for_component(records: list[DrawRecord], component: str, pool_size: int) -> list[DrawRecord]:
    recent_records = records[-RECENT_PRIOR_DRAWS:] if len(records) > RECENT_PRIOR_DRAWS else records
    if component == "main":
        return recent_records

    filtered = [record for record in recent_records if record.spec.star_pool_size == pool_size]
    return filtered or recent_records


def _sample_without_replacement(weights: np.ndarray, values: np.ndarray, sample_size: int, rng: np.random.Generator) -> tuple[int, ...]:
    local_weights = weights.astype(float).copy()
    chosen: list[int] = []
    for _ in range(sample_size):
        probabilities = local_weights / local_weights.sum()
        index = int(rng.choice(len(values), p=probabilities))
        chosen.append(int(values[index]))
        local_weights[index] = 0.0
    return tuple(sorted(chosen))


def _ticket_score(
    main_numbers: tuple[int, ...],
    star_numbers: tuple[int, ...],
    main_probabilities: dict[int, float],
    star_probabilities: dict[int, float],
    main_pair_bonus: np.ndarray,
    star_pair_bonus: np.ndarray,
    shape_priors: ShapePriors,
) -> float:
    score = 0.0
    score += sum(np.log(main_probabilities[number]) for number in main_numbers)
    score += sum(np.log(star_probabilities[number]) for number in star_numbers)
    score += 0.15 * sum(main_pair_bonus[left, right] for left, right in combinations(main_numbers, 2))
    score += 0.10 * sum(star_pair_bonus[left, right] for left, right in combinations(star_numbers, 2))
    score += shape_priors.score(main_numbers, star_numbers)
    return float(score)


def _overlap_penalty(left: CandidateTicket, right: CandidateTicket) -> float:
    main_overlap = len(set(left.main_numbers).intersection(right.main_numbers))
    star_overlap = len(set(left.star_numbers).intersection(right.star_numbers))
    return 0.35 * main_overlap + 0.65 * star_overlap


def _two_stage_overlap_penalty(left: CandidateTicket, right: CandidateTicket) -> float:
    main_overlap = len(set(left.main_numbers).intersection(right.main_numbers))
    star_overlap = len(set(left.star_numbers).intersection(right.star_numbers))
    return 0.55 * main_overlap + 0.25 * star_overlap


def _top_probability_values(probabilities: dict[int, float], pool_size: int) -> tuple[int, ...]:
    top_values = [
        value
        for value, _ in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:pool_size]
    ]
    return tuple(sorted(top_values))


def _deterministic_candidates(
    main_probabilities: dict[int, float],
    star_probabilities: dict[int, float],
    spec: GameSpec,
    main_pair_bonus: np.ndarray,
    star_pair_bonus: np.ndarray,
    shape_priors: ShapePriors,
) -> dict[tuple[tuple[int, ...], tuple[int, ...]], float]:
    main_pool = _top_probability_values(main_probabilities, DETERMINISTIC_MAIN_POOL)
    star_pool = _top_probability_values(star_probabilities, min(DETERMINISTIC_STAR_POOL, spec.star_pool_size))

    main_combos = list(combinations(main_pool, spec.main_pick_count))
    star_combos = list(combinations(star_pool, spec.star_pick_count))
    if not main_combos or not star_combos:
        return {}

    unique_candidates: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
    for main_numbers in main_combos:
        for star_numbers in star_combos:
            score = _ticket_score(
                main_numbers,
                star_numbers,
                main_probabilities,
                star_probabilities,
                main_pair_bonus,
                star_pair_bonus,
                shape_priors,
            )
            unique_candidates[(main_numbers, star_numbers)] = score
    return unique_candidates


def _build_global_candidate_scores(
    main_probabilities: dict[int, float],
    star_probabilities: dict[int, float],
    spec: GameSpec,
    train_records: list[DrawRecord],
    sample_count: int,
    random_state: int,
) -> dict[tuple[tuple[int, ...], tuple[int, ...]], float]:
    rng = np.random.default_rng(random_state)
    main_pair_bonus = build_pair_bonus(train_records, "main", spec.main_pool_size)
    star_pair_bonus = build_pair_bonus(train_records, "star", spec.star_pool_size)
    shape_priors = build_shape_priors(train_records, spec)

    main_values = np.array(sorted(main_probabilities), dtype=int)
    main_weights = np.array([main_probabilities[value] for value in main_values], dtype=float)
    star_values = np.array(sorted(star_probabilities), dtype=int)
    star_weights = np.array([star_probabilities[value] for value in star_values], dtype=float)

    unique_candidates = _deterministic_candidates(
        main_probabilities=main_probabilities,
        star_probabilities=star_probabilities,
        spec=spec,
        main_pair_bonus=main_pair_bonus,
        star_pair_bonus=star_pair_bonus,
        shape_priors=shape_priors,
    )

    for _ in range(sample_count):
        main_numbers = _sample_without_replacement(main_weights, main_values, spec.main_pick_count, rng)
        star_numbers = _sample_without_replacement(star_weights, star_values, spec.star_pick_count, rng)
        score = _ticket_score(
            main_numbers,
            star_numbers,
            main_probabilities,
            star_probabilities,
            main_pair_bonus,
            star_pair_bonus,
            shape_priors,
        )
        key = (main_numbers, star_numbers)
        if key not in unique_candidates or score > unique_candidates[key]:
            unique_candidates[key] = score

    return unique_candidates


def _select_ranked_candidates(
    ranked_candidates: list[tuple[tuple[tuple[int, ...], tuple[int, ...]], float]],
    top_k: int,
    overlap_penalty_fn,
) -> list[CandidateTicket]:
    if not ranked_candidates:
        return []

    raw_candidates: list[CandidateTicket] = []
    min_score = ranked_candidates[-1][1]
    max_score = ranked_candidates[0][1]
    score_span = max(max_score - min_score, 1e-9)
    for (main_numbers, star_numbers), score in ranked_candidates:
        relative = (score - min_score) / score_span
        confidence = max(1, min(10, int(round(1 + 9 * relative))))
        raw_candidates.append(
            CandidateTicket(
                main_numbers=main_numbers,
                star_numbers=star_numbers,
                score=score,
                confidence=confidence,
            )
        )

    selected: list[CandidateTicket] = []
    for candidate in raw_candidates:
        adjusted_score = candidate.score
        if selected:
            adjusted_score -= max(overlap_penalty_fn(candidate, prior) for prior in selected)
        adjusted_candidate = CandidateTicket(
            main_numbers=candidate.main_numbers,
            star_numbers=candidate.star_numbers,
            score=adjusted_score,
            confidence=candidate.confidence,
        )
        if len(selected) < top_k:
            selected.append(adjusted_candidate)
            selected.sort(key=lambda item: item.score, reverse=True)
            continue

        if adjusted_candidate.score > selected[-1].score:
            selected[-1] = adjusted_candidate
            selected.sort(key=lambda item: item.score, reverse=True)

    return selected[:top_k]


def _resolve_rerank_sample_count(sample_count: int, rerank_pool_size: int | None = None) -> int:
    pool_floor = CONDITIONAL_RERANK_POOL if rerank_pool_size is None else rerank_pool_size
    return max(sample_count, pool_floor)


def _partial_main_combo_score(
    main_numbers: tuple[int, ...],
    main_probabilities: dict[int, float],
    main_pair_bonus: np.ndarray,
) -> float:
    score = sum(np.log(main_probabilities[number]) for number in main_numbers)
    score += 0.15 * sum(main_pair_bonus[left, right] for left, right in combinations(main_numbers, 2))
    return float(score)


def _beam_main_candidates(
    main_probabilities: dict[int, float],
    spec: GameSpec,
    main_pair_bonus: np.ndarray,
) -> list[tuple[tuple[int, ...], float]]:
    main_pool = _top_probability_values(main_probabilities, max(DETERMINISTIC_MAIN_POOL, BEAM_MAIN_POOL))
    if len(main_pool) < spec.main_pick_count:
        return []

    states = [
        ((number,), float(np.log(main_probabilities[number])))
        for number in main_pool
    ]
    index_map = {number: index for index, number in enumerate(main_pool)}

    for _ in range(1, spec.main_pick_count):
        expanded: dict[tuple[int, ...], float] = {}
        for combo, combo_score in states:
            start_index = index_map[combo[-1]] + 1
            for candidate in main_pool[start_index:]:
                next_combo = combo + (candidate,)
                next_score = combo_score + float(np.log(main_probabilities[candidate]))
                next_score += 0.15 * sum(main_pair_bonus[existing, candidate] for existing in combo)
                best_score = expanded.get(next_combo)
                if best_score is None or next_score > best_score:
                    expanded[next_combo] = next_score

        if not expanded:
            break

        states = sorted(expanded.items(), key=lambda item: item[1], reverse=True)[:BEAM_WIDTH]

    complete = [
        (combo, _partial_main_combo_score(combo, main_probabilities, main_pair_bonus))
        for combo, _ in states
        if len(combo) == spec.main_pick_count
    ]
    return sorted(complete, key=lambda item: item[1], reverse=True)


def _score_star_pair(
    star_numbers: tuple[int, int],
    star_probabilities: dict[int, float],
    star_pair_bonus: np.ndarray,
    shape_priors: ShapePriors,
) -> float:
    left, right = star_numbers
    star_sum = left + right
    star_odd = (left % 2) + (right % 2)
    score = np.log(star_probabilities[left]) + np.log(star_probabilities[right])
    score += 0.10 * star_pair_bonus[left, right]
    score += 0.08 * shape_priors.star_sum_log_probs[star_sum]
    score += 0.05 * shape_priors.star_odd_log_probs[star_odd]
    return float(score)


def _compatible_star_history(records: list[DrawRecord], spec: GameSpec) -> list[DrawRecord]:
    recent_records = records[-RECENT_PRIOR_DRAWS:] if len(records) > RECENT_PRIOR_DRAWS else records
    compatible_records = [record for record in recent_records if record.spec.star_pool_size == spec.star_pool_size]
    return compatible_records or recent_records


def _conditional_star_record_weights(
    records: list[DrawRecord],
    star_numbers: tuple[int, ...],
    spec: GameSpec,
) -> list[tuple[DrawRecord, float, int]]:
    compatible_records = _compatible_star_history(records, spec)
    weighted_records: list[tuple[DrawRecord, float, int]] = []
    target_stars = set(star_numbers)
    for age, record in enumerate(reversed(compatible_records), start=1):
        overlap = len(target_stars.intersection(record.star_numbers))
        if overlap == 0:
            continue
        overlap_weight = 1.0 if overlap == 1 else 2.6
        recency_weight = 0.45 + 0.55 * float(np.exp(-(age - 1) / 78.0))
        weighted_records.append((record, overlap_weight * recency_weight, overlap))
    return weighted_records


def _conditional_main_context(
    records: list[DrawRecord],
    spec: GameSpec,
    star_numbers: tuple[int, ...],
    base_main_probabilities: dict[int, float],
    global_main_pair_bonus: np.ndarray,
) -> tuple[dict[int, float], np.ndarray]:
    conditioned_probabilities, blended_pair_bonus, _ = _conditional_main_context_with_support(
        records=records,
        spec=spec,
        star_numbers=star_numbers,
        base_main_probabilities=base_main_probabilities,
        global_main_pair_bonus=global_main_pair_bonus,
    )
    return conditioned_probabilities, blended_pair_bonus


def _conditional_main_context_with_support(
    records: list[DrawRecord],
    spec: GameSpec,
    star_numbers: tuple[int, ...],
    base_main_probabilities: dict[int, float],
    global_main_pair_bonus: np.ndarray,
) -> tuple[dict[int, float], np.ndarray, float]:
    weighted_records = _conditional_star_record_weights(records, star_numbers, spec)
    if not weighted_records:
        return base_main_probabilities, global_main_pair_bonus, 0.0

    total_weight = sum(weight for _, weight, _ in weighted_records)
    exact_weight = sum(weight for _, weight, overlap in weighted_records if overlap == spec.star_pick_count)
    if total_weight <= 0.0:
        return base_main_probabilities, global_main_pair_bonus, 0.0

    conditional_counts = np.zeros(spec.main_pool_size + 1, dtype=float)
    pair_counts = np.ones((spec.main_pool_size + 1, spec.main_pool_size + 1), dtype=float)
    single_counts = np.ones(spec.main_pool_size + 1, dtype=float)

    for record, weight, _ in weighted_records:
        for value in record.main_numbers:
            conditional_counts[value] += weight
            single_counts[value] += weight
        for left, right in combinations(record.main_numbers, 2):
            pair_counts[left, right] += weight
            pair_counts[right, left] += weight

    support_scale = min(1.0, total_weight / CONDITIONAL_SUPPORT_DRAWS)
    exact_scale = min(1.0, exact_weight / EXACT_STAR_SUPPORT_DRAWS)
    prior_strength = CONDITIONAL_PRIOR_STRENGTH * (1.0 - 0.65 * support_scale)

    conditioned_probabilities: dict[int, float] = {}
    for candidate in range(1, spec.main_pool_size + 1):
        prior_probability = base_main_probabilities[candidate]
        conditioned_probability = (
            conditional_counts[candidate] + prior_strength * prior_probability
        ) / (total_weight + prior_strength)
        conditioned_probabilities[candidate] = float(np.clip(conditioned_probability, 1e-6, 1.0 - 1e-6))

    conditional_pair_bonus = np.zeros_like(global_main_pair_bonus)
    for left in range(1, spec.main_pool_size + 1):
        for right in range(left + 1, spec.main_pool_size + 1):
            empirical_pair_rate = pair_counts[left, right] / total_weight
            empirical_independence = (single_counts[left] / total_weight) * (single_counts[right] / total_weight)
            bonus = np.log(empirical_pair_rate / max(empirical_independence, 1e-9))
            conditional_pair_bonus[left, right] = bonus
            conditional_pair_bonus[right, left] = bonus

    pair_blend = 0.15 + 0.35 * support_scale + 0.25 * exact_scale
    blended_pair_bonus = global_main_pair_bonus + pair_blend * conditional_pair_bonus
    support = float(min(1.0, 0.65 * support_scale + 0.35 * exact_scale))
    return conditioned_probabilities, blended_pair_bonus, support


def _conditional_main_bonus(
    main_numbers: tuple[int, ...],
    base_main_probabilities: dict[int, float],
    global_main_pair_bonus: np.ndarray,
    conditioned_main_probabilities: dict[int, float],
    conditioned_pair_bonus: np.ndarray,
) -> float:
    log_diff = sum(
        np.log(conditioned_main_probabilities[number]) - np.log(base_main_probabilities[number])
        for number in main_numbers
    )
    pair_diff = 0.15 * sum(
        conditioned_pair_bonus[left, right] - global_main_pair_bonus[left, right]
        for left, right in combinations(main_numbers, 2)
    )
    return float(log_diff + pair_diff)


def generate_two_stage_star_tickets(
    main_probabilities: dict[int, float],
    star_probabilities: dict[int, float],
    spec: GameSpec,
    train_records: list[DrawRecord],
    top_k: int = 5,
    sample_count: int = 5000,
    random_state: int = 7,
) -> list[CandidateTicket]:
    rng = np.random.default_rng(random_state)
    main_pair_bonus = build_pair_bonus(train_records, "main", spec.main_pool_size)
    star_pair_bonus = build_pair_bonus(train_records, "star", spec.star_pool_size)
    shape_priors = build_shape_priors(train_records, spec)

    main_values = np.array(sorted(main_probabilities), dtype=int)
    main_weights = np.array([main_probabilities[value] for value in main_values], dtype=float)

    star_pair_scores = sorted(
        (
            (
                star_numbers,
                _score_star_pair(star_numbers, star_probabilities, star_pair_bonus, shape_priors),
            )
            for star_numbers in combinations(range(1, spec.star_pool_size + 1), spec.star_pick_count)
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    selected_star_pairs = star_pair_scores[: max(top_k, STAR_PAIR_POOL)]
    main_combos = _beam_main_candidates(main_probabilities, spec, main_pair_bonus)
    if not selected_star_pairs or not main_combos:
        return generate_candidate_tickets(
            main_probabilities=main_probabilities,
            star_probabilities=star_probabilities,
            spec=spec,
            train_records=train_records,
            top_k=top_k,
            sample_count=sample_count,
            random_state=random_state,
        )

    unique_candidates: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
    for star_numbers, _ in selected_star_pairs:
        for main_numbers, _ in main_combos[:BEAM_WIDTH]:
            score = _ticket_score(
                main_numbers,
                star_numbers,
                main_probabilities,
                star_probabilities,
                main_pair_bonus,
                star_pair_bonus,
                shape_priors,
            )
            unique_candidates[(main_numbers, star_numbers)] = score

        local_samples = max(1, sample_count // max(1, len(selected_star_pairs)))
        for _ in range(local_samples):
            main_numbers = _sample_without_replacement(main_weights, main_values, spec.main_pick_count, rng)
            score = _ticket_score(
                main_numbers,
                star_numbers,
                main_probabilities,
                star_probabilities,
                main_pair_bonus,
                star_pair_bonus,
                shape_priors,
            )
            key = (main_numbers, star_numbers)
            if key not in unique_candidates or score > unique_candidates[key]:
                unique_candidates[key] = score

    ranked = sorted(unique_candidates.items(), key=lambda item: item[1], reverse=True)
    return _select_ranked_candidates(ranked, top_k, _two_stage_overlap_penalty)


def generate_conditional_two_stage_tickets(
    main_probabilities: dict[int, float],
    star_probabilities: dict[int, float],
    spec: GameSpec,
    train_records: list[DrawRecord],
    top_k: int = 5,
    sample_count: int = 5000,
    random_state: int = 7,
) -> list[CandidateTicket]:
    rng = np.random.default_rng(random_state)
    global_main_pair_bonus = build_pair_bonus(train_records, "main", spec.main_pool_size)
    star_pair_bonus = build_pair_bonus(train_records, "star", spec.star_pool_size)
    shape_priors = build_shape_priors(train_records, spec)
    global_main_values = np.array(sorted(main_probabilities), dtype=int)
    global_main_weights = np.array([main_probabilities[value] for value in global_main_values], dtype=float)
    global_main_combos = _beam_main_candidates(main_probabilities, spec, global_main_pair_bonus)

    star_pair_scores = sorted(
        (
            (
                star_numbers,
                _score_star_pair(star_numbers, star_probabilities, star_pair_bonus, shape_priors),
            )
            for star_numbers in combinations(range(1, spec.star_pool_size + 1), spec.star_pick_count)
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    selected_star_pairs = star_pair_scores[: max(top_k, STAR_PAIR_POOL)]
    if not selected_star_pairs:
        return generate_candidate_tickets(
            main_probabilities=main_probabilities,
            star_probabilities=star_probabilities,
            spec=spec,
            train_records=train_records,
            top_k=top_k,
            sample_count=sample_count,
            random_state=random_state,
        )

    unique_candidates: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
    for star_numbers, _ in selected_star_pairs:
        conditioned_main_probabilities, conditioned_pair_bonus = _conditional_main_context(
            records=train_records,
            spec=spec,
            star_numbers=star_numbers,
            base_main_probabilities=main_probabilities,
            global_main_pair_bonus=global_main_pair_bonus,
        )

        conditioned_main_values = np.array(sorted(conditioned_main_probabilities), dtype=int)
        conditioned_main_weights = np.array(
            [conditioned_main_probabilities[value] for value in conditioned_main_values],
            dtype=float,
        )
        conditioned_main_combos = _beam_main_candidates(conditioned_main_probabilities, spec, conditioned_pair_bonus)
        hybrid_main_combos: dict[tuple[int, ...], None] = {}
        for main_numbers, _ in conditioned_main_combos[: BEAM_WIDTH // 2]:
            hybrid_main_combos[main_numbers] = None
        for main_numbers, _ in global_main_combos[: BEAM_WIDTH // 2]:
            hybrid_main_combos[main_numbers] = None

        for main_numbers in hybrid_main_combos:
            score = _ticket_score(
                main_numbers,
                star_numbers,
                conditioned_main_probabilities,
                star_probabilities,
                conditioned_pair_bonus,
                star_pair_bonus,
                shape_priors,
            )
            unique_candidates[(main_numbers, star_numbers)] = score

        local_samples = max(1, sample_count // max(1, len(selected_star_pairs)))
        conditioned_sample_count = max(1, local_samples // 2)
        global_sample_count = max(1, local_samples - conditioned_sample_count)

        for _ in range(conditioned_sample_count):
            main_numbers = _sample_without_replacement(
                conditioned_main_weights,
                conditioned_main_values,
                spec.main_pick_count,
                rng,
            )
            score = _ticket_score(
                main_numbers,
                star_numbers,
                conditioned_main_probabilities,
                star_probabilities,
                conditioned_pair_bonus,
                star_pair_bonus,
                shape_priors,
            )
            key = (main_numbers, star_numbers)
            if key not in unique_candidates or score > unique_candidates[key]:
                unique_candidates[key] = score

        for _ in range(global_sample_count):
            main_numbers = _sample_without_replacement(
                global_main_weights,
                global_main_values,
                spec.main_pick_count,
                rng,
            )
            score = _ticket_score(
                main_numbers,
                star_numbers,
                conditioned_main_probabilities,
                star_probabilities,
                conditioned_pair_bonus,
                star_pair_bonus,
                shape_priors,
            )
            key = (main_numbers, star_numbers)
            if key not in unique_candidates or score > unique_candidates[key]:
                unique_candidates[key] = score

    ranked = sorted(unique_candidates.items(), key=lambda item: item[1], reverse=True)
    return _select_ranked_candidates(ranked, top_k, _two_stage_overlap_penalty)


def generate_hybrid_conditional_tickets(
    main_probabilities: dict[int, float],
    star_probabilities: dict[int, float],
    spec: GameSpec,
    train_records: list[DrawRecord],
    top_k: int = 5,
    sample_count: int = 5000,
    random_state: int = 7,
) -> list[CandidateTicket]:
    global_scores = _build_global_candidate_scores(
        main_probabilities=main_probabilities,
        star_probabilities=star_probabilities,
        spec=spec,
        train_records=train_records,
        sample_count=sample_count,
        random_state=random_state,
    )

    conditioned_scores: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
    for ticket in generate_conditional_two_stage_tickets(
        main_probabilities=main_probabilities,
        star_probabilities=star_probabilities,
        spec=spec,
        train_records=train_records,
        top_k=max(top_k * 2, 10),
        sample_count=sample_count,
        random_state=random_state,
    ):
        conditioned_scores[(ticket.main_numbers, ticket.star_numbers)] = ticket.score

    combined_keys = set(global_scores).union(conditioned_scores)
    if not combined_keys:
        return []

    ranked_candidates: list[tuple[tuple[tuple[int, ...], tuple[int, ...]], float]] = []
    for key in combined_keys:
        global_score = global_scores.get(key)
        conditioned_score = conditioned_scores.get(key)
        if global_score is not None and conditioned_score is not None:
            score = 0.72 * global_score + 0.28 * conditioned_score + 0.18
        elif global_score is not None:
            score = global_score
        else:
            assert conditioned_score is not None
            score = 0.88 * conditioned_score
        ranked_candidates.append((key, float(score)))

    ranked_candidates.sort(key=lambda item: item[1], reverse=True)
    return _select_ranked_candidates(ranked_candidates, top_k, _overlap_penalty)


def generate_conditional_rerank_tickets(
    main_probabilities: dict[int, float],
    star_probabilities: dict[int, float],
    spec: GameSpec,
    train_records: list[DrawRecord],
    top_k: int = 5,
    sample_count: int = 5000,
    random_state: int = 7,
) -> list[CandidateTicket]:
    reranked = _build_conditional_reranked_candidates(
        main_probabilities=main_probabilities,
        star_probabilities=star_probabilities,
        spec=spec,
        train_records=train_records,
        sample_count=sample_count,
        random_state=random_state,
        rerank_weight=CONDITIONAL_RERANK_WEIGHT,
        rerank_pool_size=CONDITIONAL_RERANK_POOL,
    )
    return _select_ranked_candidates(reranked, top_k, _overlap_penalty)


def _build_conditional_reranked_candidates(
    main_probabilities: dict[int, float],
    star_probabilities: dict[int, float],
    spec: GameSpec,
    train_records: list[DrawRecord],
    sample_count: int,
    random_state: int,
    rerank_weight: float,
    rerank_pool_size: int | None = None,
) -> list[tuple[tuple[tuple[int, ...], tuple[int, ...]], float]]:
    rerank_sample_count = _resolve_rerank_sample_count(sample_count, rerank_pool_size)
    global_scores = _build_global_candidate_scores(
        main_probabilities=main_probabilities,
        star_probabilities=star_probabilities,
        spec=spec,
        train_records=train_records,
        sample_count=rerank_sample_count,
        random_state=random_state,
    )
    return _build_conditional_reranked_candidates_from_global_scores(
        global_scores=global_scores,
        main_probabilities=main_probabilities,
        spec=spec,
        train_records=train_records,
        rerank_weight=rerank_weight,
    )


def _build_conditional_reranked_candidates_from_global_scores(
    global_scores: dict[tuple[tuple[int, ...], tuple[int, ...]], float],
    main_probabilities: dict[int, float],
    spec: GameSpec,
    train_records: list[DrawRecord],
    rerank_weight: float,
) -> list[tuple[tuple[tuple[int, ...], tuple[int, ...]], float]]:
    if not global_scores:
        return []

    global_main_pair_bonus = build_pair_bonus(train_records, "main", spec.main_pool_size)
    star_context_cache: dict[tuple[int, ...], tuple[dict[int, float], np.ndarray, float]] = {}
    reranked: list[tuple[tuple[tuple[int, ...], tuple[int, ...]], float]] = []

    for (main_numbers, star_numbers), score in sorted(global_scores.items(), key=lambda item: item[1], reverse=True):
        context = star_context_cache.get(star_numbers)
        if context is None:
            context = _conditional_main_context_with_support(
                records=train_records,
                spec=spec,
                star_numbers=star_numbers,
                base_main_probabilities=main_probabilities,
                global_main_pair_bonus=global_main_pair_bonus,
            )
            star_context_cache[star_numbers] = context

        conditioned_main_probabilities, conditioned_pair_bonus, support = context
        bonus = _conditional_main_bonus(
            main_numbers=main_numbers,
            base_main_probabilities=main_probabilities,
            global_main_pair_bonus=global_main_pair_bonus,
            conditioned_main_probabilities=conditioned_main_probabilities,
            conditioned_pair_bonus=conditioned_pair_bonus,
        )
        adjusted_score = score + rerank_weight * support * bonus
        reranked.append(((main_numbers, star_numbers), float(adjusted_score)))

    reranked.sort(key=lambda item: item[1], reverse=True)
    return reranked


def _select_guarded_rerank_portfolio(
    guarded_candidates: list[tuple[tuple[tuple[int, ...], tuple[int, ...]], float]],
    reranked: list[tuple[tuple[tuple[int, ...], tuple[int, ...]], float]],
    top_k: int,
) -> list[CandidateTicket]:
    seed_candidates = [
        CandidateTicket(
            main_numbers=main_numbers,
            star_numbers=star_numbers,
            score=score,
            confidence=10,
        )
        for (main_numbers, star_numbers), score in guarded_candidates
    ]

    selected: list[CandidateTicket] = seed_candidates[:]
    if not reranked:
        return selected[:top_k]

    min_score = reranked[-1][1]
    max_score = reranked[0][1]
    score_span = max(max_score - min_score, 1e-9)
    for (main_numbers, star_numbers), score in reranked:
        if any(ticket.main_numbers == main_numbers and ticket.star_numbers == star_numbers for ticket in selected):
            continue
        relative = (score - min_score) / score_span
        confidence = max(1, min(10, int(round(1 + 9 * relative))))
        candidate = CandidateTicket(
            main_numbers=main_numbers,
            star_numbers=star_numbers,
            score=score,
            confidence=confidence,
        )
        adjusted_score = candidate.score
        if selected:
            adjusted_score -= max(_overlap_penalty(candidate, prior) for prior in selected)
        adjusted_candidate = CandidateTicket(
            main_numbers=candidate.main_numbers,
            star_numbers=candidate.star_numbers,
            score=adjusted_score,
            confidence=candidate.confidence,
        )
        if len(selected) < top_k:
            selected.append(adjusted_candidate)
            selected.sort(key=lambda item: item.score, reverse=True)
            continue
        if adjusted_candidate.score > selected[-1].score:
            selected[-1] = adjusted_candidate
            selected.sort(key=lambda item: item.score, reverse=True)

    return selected[:top_k]


def _top_distinct_star_pair_candidates(
    candidate_scores: dict[tuple[tuple[int, ...], tuple[int, ...]], float],
    count: int,
) -> list[tuple[tuple[tuple[int, ...], tuple[int, ...]], float]]:
    selected: list[tuple[tuple[tuple[int, ...], tuple[int, ...]], float]] = []
    seen_star_pairs: set[tuple[int, ...]] = set()
    for key, score in sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True):
        _, star_numbers = key
        if star_numbers in seen_star_pairs:
            continue
        selected.append((key, score))
        seen_star_pairs.add(star_numbers)
        if len(selected) >= count:
            break
    return selected


def _guarded_star_support(
    candidate_scores: dict[tuple[tuple[int, ...], tuple[int, ...]], float],
    spec: GameSpec,
    train_records: list[DrawRecord],
    base_main_probabilities: dict[int, float],
) -> float:
    guarded = _top_distinct_star_pair_candidates(candidate_scores, 1)
    if not guarded:
        return 0.0

    (_, star_numbers), _ = guarded[0]
    _, _, support = _conditional_main_context_with_support(
        records=train_records,
        spec=spec,
        star_numbers=star_numbers,
        base_main_probabilities=base_main_probabilities,
        global_main_pair_bonus=build_pair_bonus(train_records, "main", spec.main_pool_size),
    )
    return support


def generate_star_guard_rerank_tickets(
    main_probabilities: dict[int, float],
    star_probabilities: dict[int, float],
    spec: GameSpec,
    train_records: list[DrawRecord],
    top_k: int = 5,
    sample_count: int = 5000,
    random_state: int = 7,
    ) -> list[CandidateTicket]:
    return _generate_star_guard_rerank_tickets(
        main_probabilities=main_probabilities,
        star_probabilities=star_probabilities,
        spec=spec,
        train_records=train_records,
        top_k=top_k,
        sample_count=sample_count,
        random_state=random_state,
        guard_count=STAR_GUARD_COUNT,
        rerank_weight=CONDITIONAL_RERANK_WEIGHT,
        rerank_pool_size=CONDITIONAL_RERANK_POOL,
    )


def _generate_star_guard_rerank_tickets(
    main_probabilities: dict[int, float],
    star_probabilities: dict[int, float],
    spec: GameSpec,
    train_records: list[DrawRecord],
    top_k: int,
    sample_count: int,
    random_state: int,
    guard_count: int,
    rerank_weight: float,
    rerank_pool_size: int,
) -> list[CandidateTicket]:
    rerank_sample_count = _resolve_rerank_sample_count(sample_count, rerank_pool_size)
    global_scores = _build_global_candidate_scores(
        main_probabilities=main_probabilities,
        star_probabilities=star_probabilities,
        spec=spec,
        train_records=train_records,
        sample_count=rerank_sample_count,
        random_state=random_state,
    )
    if not global_scores:
        return []

    guarded = _top_distinct_star_pair_candidates(global_scores, min(guard_count, top_k))
    reranked = _build_conditional_reranked_candidates_from_global_scores(
        global_scores=global_scores,
        main_probabilities=main_probabilities,
        spec=spec,
        train_records=train_records,
        rerank_weight=rerank_weight,
    )
    return _select_guarded_rerank_portfolio(guarded, reranked, top_k)


def generate_soft_star_guard_rerank_tickets(
    main_probabilities: dict[int, float],
    star_probabilities: dict[int, float],
    spec: GameSpec,
    train_records: list[DrawRecord],
    top_k: int = 5,
    sample_count: int = 5000,
    random_state: int = 7,
) -> list[CandidateTicket]:
    return _generate_soft_star_guard_rerank_tickets(
        main_probabilities=main_probabilities,
        star_probabilities=star_probabilities,
        spec=spec,
        train_records=train_records,
        top_k=top_k,
        sample_count=sample_count,
        random_state=random_state,
        rerank_pool_size=CONDITIONAL_RERANK_POOL,
    )


def _generate_soft_star_guard_rerank_tickets(
    main_probabilities: dict[int, float],
    star_probabilities: dict[int, float],
    spec: GameSpec,
    train_records: list[DrawRecord],
    top_k: int,
    sample_count: int,
    random_state: int,
    rerank_pool_size: int,
) -> list[CandidateTicket]:
    rerank_sample_count = _resolve_rerank_sample_count(sample_count, rerank_pool_size)
    global_scores = _build_global_candidate_scores(
        main_probabilities=main_probabilities,
        star_probabilities=star_probabilities,
        spec=spec,
        train_records=train_records,
        sample_count=rerank_sample_count,
        random_state=random_state,
    )
    if not global_scores:
        return []

    guarded = _top_distinct_star_pair_candidates(global_scores, min(STAR_GUARD_COUNT, top_k))
    reranked = _build_conditional_reranked_candidates(
        main_probabilities=main_probabilities,
        star_probabilities=star_probabilities,
        spec=spec,
        train_records=train_records,
        sample_count=sample_count,
        random_state=random_state,
        rerank_weight=CONDITIONAL_RERANK_WEIGHT,
        rerank_pool_size=rerank_pool_size,
    )
    guard_bonus_by_key = {
        key: 0.30 - 0.04 * index
        for index, (key, _) in enumerate(guarded)
    }
    boosted = [
        (key, float(score + guard_bonus_by_key.get(key, 0.0)))
        for key, score in reranked
    ]
    boosted.sort(key=lambda item: item[1], reverse=True)
    return _select_ranked_candidates(boosted, top_k, _overlap_penalty)


def generate_star_guard1_rerank_tickets(
    main_probabilities: dict[int, float],
    star_probabilities: dict[int, float],
    spec: GameSpec,
    train_records: list[DrawRecord],
    top_k: int = 5,
    sample_count: int = 5000,
    random_state: int = 7,
) -> list[CandidateTicket]:
    return _generate_star_guard_rerank_tickets(
        main_probabilities=main_probabilities,
        star_probabilities=star_probabilities,
        spec=spec,
        train_records=train_records,
        top_k=top_k,
        sample_count=sample_count,
        random_state=random_state,
        guard_count=1,
        rerank_weight=CONDITIONAL_RERANK_WEIGHT,
        rerank_pool_size=CONDITIONAL_RERANK_POOL,
    )


def generate_star_guard1_soft_rerank_tickets(
    main_probabilities: dict[int, float],
    star_probabilities: dict[int, float],
    spec: GameSpec,
    train_records: list[DrawRecord],
    top_k: int = 5,
    sample_count: int = 5000,
    random_state: int = 7,
) -> list[CandidateTicket]:
    return _generate_star_guard_rerank_tickets(
        main_probabilities=main_probabilities,
        star_probabilities=star_probabilities,
        spec=spec,
        train_records=train_records,
        top_k=top_k,
        sample_count=sample_count,
        random_state=random_state,
        guard_count=1,
        rerank_weight=0.30,
        rerank_pool_size=CONDITIONAL_RERANK_POOL,
    )


def generate_star_guard1_soft_screen_tickets(
    main_probabilities: dict[int, float],
    star_probabilities: dict[int, float],
    spec: GameSpec,
    train_records: list[DrawRecord],
    top_k: int = 5,
    sample_count: int = 5000,
    random_state: int = 7,
) -> list[CandidateTicket]:
    rerank_sample_count = _resolve_rerank_sample_count(sample_count, EXPERIMENTAL_RERANK_POOL)
    global_scores = _build_global_candidate_scores(
        main_probabilities=main_probabilities,
        star_probabilities=star_probabilities,
        spec=spec,
        train_records=train_records,
        sample_count=rerank_sample_count,
        random_state=random_state,
    )
    if not global_scores:
        return []

    guarded = _top_distinct_star_pair_candidates(global_scores, 1)
    reranked = _build_conditional_reranked_candidates_from_global_scores(
        global_scores=global_scores,
        main_probabilities=main_probabilities,
        spec=spec,
        train_records=train_records,
        rerank_weight=0.30,
    )
    return _select_guarded_rerank_portfolio(guarded, reranked, top_k)


def generate_soft_star_guard_screen_tickets(
    main_probabilities: dict[int, float],
    star_probabilities: dict[int, float],
    spec: GameSpec,
    train_records: list[DrawRecord],
    top_k: int = 5,
    sample_count: int = 5000,
    random_state: int = 7,
) -> list[CandidateTicket]:
    return _generate_soft_star_guard_rerank_tickets(
        main_probabilities=main_probabilities,
        star_probabilities=star_probabilities,
        spec=spec,
        train_records=train_records,
        top_k=top_k,
        sample_count=sample_count,
        random_state=random_state,
        rerank_pool_size=EXPERIMENTAL_RERANK_POOL,
    )


def _generate_support_gated_star_guard_screen_tickets(
    main_probabilities: dict[int, float],
    star_probabilities: dict[int, float],
    spec: GameSpec,
    train_records: list[DrawRecord],
    top_k: int,
    sample_count: int,
    random_state: int,
    support_threshold: float,
) -> list[CandidateTicket]:
    rerank_sample_count = _resolve_rerank_sample_count(sample_count, EXPERIMENTAL_RERANK_POOL)
    global_scores = _build_global_candidate_scores(
        main_probabilities=main_probabilities,
        star_probabilities=star_probabilities,
        spec=spec,
        train_records=train_records,
        sample_count=rerank_sample_count,
        random_state=random_state,
    )
    if not global_scores:
        return []

    guarded_support = _guarded_star_support(
        candidate_scores=global_scores,
        spec=spec,
        train_records=train_records,
        base_main_probabilities=main_probabilities,
    )
    if guarded_support < support_threshold:
        return generate_candidate_tickets(
            main_probabilities=main_probabilities,
            star_probabilities=star_probabilities,
            spec=spec,
            train_records=train_records,
            top_k=top_k,
            sample_count=sample_count,
            random_state=random_state,
        )

    guarded = _top_distinct_star_pair_candidates(global_scores, 1)
    reranked = _build_conditional_reranked_candidates_from_global_scores(
        global_scores=global_scores,
        main_probabilities=main_probabilities,
        spec=spec,
        train_records=train_records,
        rerank_weight=0.30,
    )
    return _select_guarded_rerank_portfolio(guarded, reranked, top_k)


def generate_support_gated_star_guard_screen_050_tickets(
    main_probabilities: dict[int, float],
    star_probabilities: dict[int, float],
    spec: GameSpec,
    train_records: list[DrawRecord],
    top_k: int = 5,
    sample_count: int = 5000,
    random_state: int = 7,
) -> list[CandidateTicket]:
    return _generate_support_gated_star_guard_screen_tickets(
        main_probabilities=main_probabilities,
        star_probabilities=star_probabilities,
        spec=spec,
        train_records=train_records,
        top_k=top_k,
        sample_count=sample_count,
        random_state=random_state,
        support_threshold=SUPPORT_GATE_THRESHOLD_050,
    )


def generate_support_gated_star_guard_screen_060_tickets(
    main_probabilities: dict[int, float],
    star_probabilities: dict[int, float],
    spec: GameSpec,
    train_records: list[DrawRecord],
    top_k: int = 5,
    sample_count: int = 5000,
    random_state: int = 7,
) -> list[CandidateTicket]:
    return _generate_support_gated_star_guard_screen_tickets(
        main_probabilities=main_probabilities,
        star_probabilities=star_probabilities,
        spec=spec,
        train_records=train_records,
        top_k=top_k,
        sample_count=sample_count,
        random_state=random_state,
        support_threshold=SUPPORT_GATE_THRESHOLD_060,
    )


def generate_candidate_tickets(
    main_probabilities: dict[int, float],
    star_probabilities: dict[int, float],
    spec: GameSpec,
    train_records: list[DrawRecord],
    top_k: int = 5,
    sample_count: int = 5000,
    random_state: int = 7,
) -> list[CandidateTicket]:
    unique_candidates = _build_global_candidate_scores(
        main_probabilities=main_probabilities,
        star_probabilities=star_probabilities,
        spec=spec,
        train_records=train_records,
        sample_count=sample_count,
        random_state=random_state,
    )
    ranked = sorted(unique_candidates.items(), key=lambda item: item[1], reverse=True)
    return _select_ranked_candidates(ranked, top_k, _overlap_penalty)


def evaluate_ticket(ticket: CandidateTicket, actual_record: DrawRecord) -> dict[str, int]:
    main_hits = len(set(ticket.main_numbers).intersection(actual_record.main_numbers))
    star_hits = len(set(ticket.star_numbers).intersection(actual_record.star_numbers))
    return {
        "main_hits": main_hits,
        "star_hits": star_hits,
        "weighted_hits": main_hits + 2 * star_hits,
    }
