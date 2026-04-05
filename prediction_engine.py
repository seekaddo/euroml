from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime
from pathlib import Path

from engine.cache import DEFAULT_CACHE_DIR, load_cached_object, save_cached_object
from engine.backtest import BacktestConfig, run_backtest
from engine.calibration import build_confidence_calibration, calibrate_confidence
from engine.data import DEFAULT_DATASET_DIR, load_draw_records, next_draw_date
from engine.features import build_feature_tables, build_prediction_rows, feature_signature
from engine.memory import (
    calibration_matches_dataset,
    dataset_fingerprint,
    load_memory,
    save_memory,
    select_confidence_calibration,
)
from engine.spec import spec_for_draw_date
from engine.strategies import available_strategy_names, get_strategy
from engine.tickets import CandidateTicket, ticket_generator_signature
from tabulate import tabulate

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_MEMORY_FILE = Path(".engine.mem.json")


def _engine_signature(strategy_name: str) -> str:
    strategy = get_strategy(strategy_name)
    return f"{strategy.signature}:{feature_signature()}:{ticket_generator_signature()}"


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rule-aware EuroMillions prediction and backtesting engine.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Dataset directory containing yearly *_eml.json files. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--memory-file",
        type=Path,
        default=DEFAULT_MEMORY_FILE,
        help=f"Structured engine state file. Default: {DEFAULT_MEMORY_FILE}",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Deterministic feature cache directory. Default: {DEFAULT_CACHE_DIR}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest_parser = subparsers.add_parser("backtest", help="Run a chronological held-out backtest.")
    backtest_parser.add_argument("--train-end", type=parse_iso_date, required=True, help="Final training draw date in YYYY-MM-DD.")
    backtest_parser.add_argument("--test-start", type=parse_iso_date, required=True, help="Backtest start date in YYYY-MM-DD.")
    backtest_parser.add_argument("--test-end", type=parse_iso_date, required=True, help="Backtest end date in YYYY-MM-DD.")
    backtest_parser.add_argument("--mode", choices=["frozen", "rolling"], default="frozen")
    backtest_parser.add_argument("--strategy", choices=available_strategy_names(), default="baseline")
    backtest_parser.add_argument("--top-k", type=int, default=5)
    backtest_parser.add_argument("--samples", type=int, default=5000)
    backtest_parser.add_argument("--seed", type=int, default=7)
    backtest_parser.add_argument("--output", type=Path, help="Optional JSON output path for the detailed backtest report.")

    compare_parser = subparsers.add_parser("compare-strategies", help="Compare multiple strategies on the same held-out backtest window.")
    compare_parser.add_argument("--train-end", type=parse_iso_date, required=True, help="Final training draw date in YYYY-MM-DD.")
    compare_parser.add_argument("--test-start", type=parse_iso_date, required=True, help="Backtest start date in YYYY-MM-DD.")
    compare_parser.add_argument("--test-end", type=parse_iso_date, required=True, help="Backtest end date in YYYY-MM-DD.")
    compare_parser.add_argument("--mode", choices=["frozen", "rolling"], default="frozen")
    compare_parser.add_argument("--strategies", nargs="+", choices=available_strategy_names(), default=available_strategy_names())
    compare_parser.add_argument("--top-k", type=int, default=5)
    compare_parser.add_argument("--samples", type=int, default=5000)
    compare_parser.add_argument("--seed", type=int, default=7)
    compare_parser.add_argument("--output", type=Path, help="Optional JSON output path for the comparison summary.")

    predict_parser = subparsers.add_parser("predict-next", help="Predict the next unseen EuroMillions draw.")
    predict_parser.add_argument("--train-end", type=parse_iso_date, help="Final training draw date in YYYY-MM-DD. Defaults to the latest draw.")
    predict_parser.add_argument("--next-draw-date", type=parse_iso_date, help="Explicit next draw date in YYYY-MM-DD. Defaults to the inferred next Tuesday/Friday draw.")
    predict_parser.add_argument("--strategy", choices=available_strategy_names(), default="baseline")
    predict_parser.add_argument("--top-k", type=int, default=5)
    predict_parser.add_argument("--samples", type=int, default=5000)
    predict_parser.add_argument("--seed", type=int, default=7)
    predict_parser.add_argument("--output", type=Path, help="Optional JSON output path for the prediction report.")

    return parser.parse_args()


def _ticket_payload(rank: int, ticket: CandidateTicket) -> dict[str, object]:
    return {
        "rank": rank,
        "main_numbers": list(ticket.main_numbers),
        "star_numbers": list(ticket.star_numbers),
        "score": round(ticket.score, 6),
        "confidence": ticket.confidence,
    }


def _format_ticket(main_numbers: list[int] | tuple[int, ...], star_numbers: list[int] | tuple[int, ...]) -> str:
    main_text = " ".join(f"{number:02d}" for number in main_numbers)
    star_text = " ".join(f"{number:02d}" for number in star_numbers)
    return f"{main_text} | {star_text}"


def _format_hit(main_hits: int, star_hits: int) -> str:
    if main_hits == 0 and star_hits == 0:
        return "miss"
    return f"{main_hits}+{star_hits}"


def _print_backtest_tables(report: dict[str, object]) -> None:
    summary = report["summary"]
    summary_table = [
        {
            "strategy": summary.get("strategy", "baseline"),
            "mode": summary["mode"],
            "draws": summary["draw_count"],
            "top1_w": summary["top1_avg_weighted_hits"],
            "rand_top1_w": summary["random_top1_avg_weighted_hits"],
            "lift_top1": summary["lift_top1_weighted_hits"],
            "best5_w": summary["best5_avg_weighted_hits"],
            "rand_best5_w": summary["random_best5_avg_weighted_hits"],
            "lift_best5": summary["lift_best5_weighted_hits"],
            "exact_top1_5+2": summary["top1_exact_5_2_hits"],
            "exact_best5_5+2": summary["best5_exact_5_2_hits"],
        }
    ]
    print("\nBacktest Summary")
    print(tabulate(summary_table, headers="keys", tablefmt="github", floatfmt=".4f"))

    milestones = report.get("milestones")
    if milestones:
        milestone_rows = []
        for label in ("top1", "best5", "random_top1", "random_best5"):
            row = milestones.get(label, {})
            rates = row.get("rate", {})
            milestone_rows.append(
                {
                    "portfolio": label,
                    "useful_rate": row.get("useful_hit_rate", 0.0),
                    "0+2": rates.get("0+2", 0.0),
                    "1+2": rates.get("1+2", 0.0),
                    "2+2": rates.get("2+2", 0.0),
                    "3+1": rates.get("3+1", 0.0),
                    "3+2": rates.get("3+2", 0.0),
                    "4+1": rates.get("4+1", 0.0),
                    "4+2": rates.get("4+2", 0.0),
                    "5+2": rates.get("5+2", 0.0),
                }
            )
        print("\nMilestone Rates")
        print(tabulate(milestone_rows, headers="keys", tablefmt="github", floatfmt=".4f"))

    release_gate = report.get("release_gate")
    if release_gate:
        release_rows = []
        for portfolio in ("top1", "best5"):
            row = release_gate.get(portfolio, {})
            patterns = row.get("patterns", {})
            release_rows.append(
                {
                    "portfolio": portfolio,
                    "3+2": patterns.get("3+2", {}).get("count", 0),
                    "rand_3+2": patterns.get("3+2", {}).get("random_count", 0),
                    "4+2": patterns.get("4+2", {}).get("count", 0),
                    "rand_4+2": patterns.get("4+2", {}).get("random_count", 0),
                    "window_pass": row.get("window_pass", False),
                }
            )
        print("\nRelease Gate")
        print(tabulate(release_rows, headers="keys", tablefmt="github"))

    progression_rows = []
    cumulative_weighted_hits = 0
    for index, draw in enumerate(report["draws"], start=1):
        predictions = draw["predictions"]
        if not predictions:
            continue
        best_prediction = max(
            predictions,
            key=lambda item: (item["weighted_hits"], item["main_hits"], item["star_hits"], -item["rank"]),
        )
        cumulative_weighted_hits += best_prediction["weighted_hits"]
        cumulative_success_rate = 100.0 * cumulative_weighted_hits / (index * 9)
        progression_rows.append(
            {
                "draw_date": draw["draw_date"],
                "best_rank": best_prediction["rank"],
                "prediction": _format_ticket(best_prediction["main_numbers"], best_prediction["star_numbers"]),
                "actual": _format_ticket(draw["actual_main_numbers"], draw["actual_star_numbers"]),
                "hit": _format_hit(best_prediction["main_hits"], best_prediction["star_hits"]),
                "confidence": best_prediction["confidence"],
                "success_rate": f"{cumulative_success_rate:.2f}%",
            }
        )

    print("\nBacktest Progression")
    print(tabulate(progression_rows, headers="keys", tablefmt="github"))

    model_diagnostics = report.get("model_diagnostics")
    if model_diagnostics:
        diagnostics_rows = []
        for component, diagnostics in model_diagnostics.items():
            weights = diagnostics.get("weights", {})
            expert_weights = diagnostics.get("expert_weights", {})
            expert_mix = ", ".join(f"{label}={weight}" for label, weight in expert_weights.items())
            diagnostics_rows.append(
                {
                    "component": component,
                    "model_type": diagnostics.get("model_type", "blend"),
                    "validation_draws": diagnostics.get("validation_draw_count", 0),
                    "blend_brier": diagnostics.get("blend_brier", "n/a"),
                    "prior_w": round(float(weights.get("prior", 0.0)), 4),
                    "linear_w": round(float(weights.get("linear", 0.0)), 4),
                    "tree_w": round(float(weights.get("tree", 0.0)), 4),
                    "expert_mix": expert_mix or "",
                }
            )
        print("\nModel Diagnostics")
        print(tabulate(diagnostics_rows, headers="keys", tablefmt="github"))

    calibration = report.get("confidence_calibration")
    if calibration:
        calibration_rows = [
            {
                "score_min": round(item["score_min"], 4),
                "score_max": round(item["score_max"], 4),
                "count": item["count"],
                "avg_w_hits": round(item["avg_weighted_hits"], 4),
                "hit_2plus": round(item["hit_two_plus_rate"], 4),
                "hit_4plus": round(item["hit_four_plus_rate"], 4),
                "confidence": item["confidence"],
            }
            for item in calibration["bins"]
        ]
        print("\nConfidence Calibration")
        print(tabulate(calibration_rows, headers="keys", tablefmt="github"))


def _print_prediction_tables(report: dict[str, object]) -> None:
    model_diagnostics = report.get("model_diagnostics")
    if model_diagnostics:
        diagnostics_rows = []
        for component, diagnostics in model_diagnostics.items():
            weights = diagnostics.get("weights", {})
            expert_weights = diagnostics.get("expert_weights", {})
            expert_mix = ", ".join(f"{label}={weight}" for label, weight in expert_weights.items())
            diagnostics_rows.append(
                {
                    "component": component,
                    "model_type": diagnostics.get("model_type", "blend"),
                    "validation_draws": diagnostics.get("validation_draw_count", 0),
                    "blend_brier": diagnostics.get("blend_brier", "n/a"),
                    "prior_w": round(float(weights.get("prior", 0.0)), 4),
                    "linear_w": round(float(weights.get("linear", 0.0)), 4),
                    "tree_w": round(float(weights.get("tree", 0.0)), 4),
                    "expert_mix": expert_mix or "",
                }
            )
        print("\nModel Diagnostics")
        print(tabulate(diagnostics_rows, headers="keys", tablefmt="github"))

    print("\nTop Main Number Probabilities")
    print(tabulate(report["top_main_probabilities"], headers="keys", tablefmt="github", floatfmt=".6f"))

    print("\nTop Lucky Star Probabilities")
    print(tabulate(report["top_star_probabilities"], headers="keys", tablefmt="github", floatfmt=".6f"))

    predictions_table = [
        {
            "rank": prediction["rank"],
            "prediction": _format_ticket(prediction["main_numbers"], prediction["star_numbers"]),
            "confidence": prediction["confidence"],
            "score": prediction["score"],
        }
        for prediction in report["predictions"]
    ]
    print(f"\nNext Draw Predictions For {report['next_draw_date']}")
    print(tabulate(predictions_table, headers="keys", tablefmt="github", floatfmt=".6f"))

    calibration_meta = report.get("confidence_calibration_source")
    if calibration_meta:
        meta_table = [calibration_meta]
        print("\nConfidence Source")
        print(tabulate(meta_table, headers="keys", tablefmt="github"))


def _save_optional_output(path: Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logging.info("Wrote output report to %s", path)


def _update_memory(path: Path, payload: dict[str, object]) -> None:
    memory = load_memory(path)
    memory.update(payload)
    save_memory(path, memory)
    logging.info("Updated engine memory at %s", path)


def _load_feature_tables_for_records(
    records: list,
    cache_dir: Path,
    cache_key: str,
    fingerprint: str,
    build_reason: str,
) -> dict[str, object]:
    feature_tables = load_cached_object(cache_dir, cache_key, fingerprint)
    if feature_tables is None:
        logging.info("Building feature tables for %s", build_reason)
        feature_tables = build_feature_tables(records)
        save_cached_object(cache_dir, cache_key, fingerprint, feature_tables)
        logging.info("Saved feature cache to %s", cache_dir)
    else:
        logging.info("Loaded feature tables from cache %s", cache_dir)
    return feature_tables


def _comparison_row(report: dict[str, object]) -> dict[str, object]:
    summary = report["summary"]
    milestones = report.get("milestones", {})
    best5_row = milestones.get("best5", {})
    best5_rates = best5_row.get("rate", {})
    release_gate = report.get("release_gate", {}).get("best5", {})
    release_patterns = release_gate.get("patterns", {})
    return {
        "strategy": summary.get("strategy", "baseline"),
        "mode": summary["mode"],
        "top1_w": summary["top1_avg_weighted_hits"],
        "best5_w": summary["best5_avg_weighted_hits"],
        "lift_best5": summary["lift_best5_weighted_hits"],
        "2+2_rate": best5_rates.get("2+2", 0.0),
        "3+2_count": release_patterns.get("3+2", {}).get("count", 0),
        "4+2_count": release_patterns.get("4+2", {}).get("count", 0),
        "release_pass": release_gate.get("window_pass", False),
    }


def run_backtest_command(args: argparse.Namespace) -> None:
    logging.info("Loading historical draw records from %s", args.dataset_dir)
    records = load_draw_records(args.dataset_dir)
    logging.info("Loaded %s draws", len(records))
    fingerprint = dataset_fingerprint(records)
    feature_cache_key = f"feature_tables_{feature_signature()}"
    feature_tables = _load_feature_tables_for_records(
        records=records,
        cache_dir=args.cache_dir,
        cache_key=feature_cache_key,
        fingerprint=fingerprint,
        build_reason="backtest",
    )
    strategy = get_strategy(args.strategy)
    engine_signature = _engine_signature(args.strategy)
    logging.info(
        "Running %s strategy with %s backtest from %s to %s and training cutoff %s",
        strategy.name,
        args.mode,
        args.test_start.isoformat(),
        args.test_end.isoformat(),
        args.train_end.isoformat(),
    )
    report = run_backtest(
        records=records,
        feature_tables=feature_tables,
        config=BacktestConfig(
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
            mode=args.mode,
            top_k=args.top_k,
            sample_count=args.samples,
            random_state=args.seed,
        ),
        strategy=strategy,
    )
    calibration = build_confidence_calibration(report)
    if calibration is not None:
        calibration["dataset_fingerprint"] = fingerprint
        calibration["engine_signature"] = engine_signature
        report["confidence_calibration"] = calibration

    _print_backtest_tables(report)
    _save_optional_output(args.output, report)
    existing_memory = load_memory(args.memory_file)
    stored_calibration = select_confidence_calibration(
        existing_memory.get("confidence_calibration"),
        calibration,
    )
    _update_memory(
        args.memory_file,
        {
            "last_backtest_dataset_fingerprint": fingerprint,
            "engine_signature": engine_signature,
            "last_backtest": report["summary"],
            "confidence_calibration": stored_calibration,
        },
    )


def run_compare_strategies_command(args: argparse.Namespace) -> None:
    logging.info("Loading historical draw records from %s", args.dataset_dir)
    records = load_draw_records(args.dataset_dir)
    logging.info("Loaded %s draws", len(records))
    fingerprint = dataset_fingerprint(records)
    feature_cache_key = f"feature_tables_{feature_signature()}"
    feature_tables = _load_feature_tables_for_records(
        records=records,
        cache_dir=args.cache_dir,
        cache_key=feature_cache_key,
        fingerprint=fingerprint,
        build_reason="strategy comparison",
    )

    comparison_reports: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    for strategy_name in args.strategies:
        strategy = get_strategy(strategy_name)
        logging.info("Comparing strategy %s", strategy.name)
        report = run_backtest(
            records=records,
            feature_tables=feature_tables,
            config=BacktestConfig(
                train_end=args.train_end,
                test_start=args.test_start,
                test_end=args.test_end,
                mode=args.mode,
                top_k=args.top_k,
                sample_count=args.samples,
                random_state=args.seed,
            ),
            strategy=strategy,
        )
        comparison_reports.append(report)
        comparison_rows.append(_comparison_row(report))

    print("\nStrategy Comparison")
    print(tabulate(comparison_rows, headers="keys", tablefmt="github", floatfmt=".4f"))

    if args.output:
        payload = {
            "train_end": args.train_end.isoformat(),
            "test_start": args.test_start.isoformat(),
            "test_end": args.test_end.isoformat(),
            "mode": args.mode,
            "strategies": args.strategies,
            "results": comparison_rows,
        }
        _save_optional_output(args.output, payload)


def run_predict_command(args: argparse.Namespace) -> None:
    logging.info("Loading historical draw records from %s", args.dataset_dir)
    records = load_draw_records(args.dataset_dir)
    fingerprint = dataset_fingerprint(records)
    memory = load_memory(args.memory_file)
    strategy = get_strategy(args.strategy)
    engine_signature = _engine_signature(args.strategy)
    train_end = args.train_end or max(record.draw_date for record in records)
    history_records = [record for record in records if record.draw_date <= train_end]
    if not history_records:
        raise ValueError("No history records are available for the requested training cutoff")

    target_date = args.next_draw_date or next_draw_date(history_records)
    logging.info("Loaded %s training draws through %s", len(history_records), train_end.isoformat())
    history_fingerprint = dataset_fingerprint(history_records)
    feature_cache_key = f"feature_tables_{feature_signature()}_{train_end.isoformat()}"
    feature_tables = _load_feature_tables_for_records(
        records=history_records,
        cache_dir=args.cache_dir,
        cache_key=feature_cache_key,
        fingerprint=history_fingerprint,
        build_reason="next-draw prediction",
    )
    prediction_rows = build_prediction_rows(history_records, target_date)

    logging.info("Training %s strategy for next draw %s", strategy.name, target_date.isoformat())
    main_model = strategy.create_main_model(random_state=args.seed).fit(
        feature_tables["main"]
    )
    star_model = strategy.create_star_model(random_state=args.seed).fit(
        feature_tables["star"]
    )

    main_rows = prediction_rows["main"]
    star_rows = prediction_rows["star"]
    main_probabilities = dict(
        zip(main_rows["candidate"].astype(int), main_model.predict_proba(main_rows), strict=True)
    )
    star_probabilities = dict(
        zip(star_rows["candidate"].astype(int), star_model.predict_proba(star_rows), strict=True)
    )

    tickets = strategy.ticket_generator(
        main_probabilities=main_probabilities,
        star_probabilities=star_probabilities,
        spec=spec_for_draw_date(target_date),
        train_records=history_records,
        top_k=args.top_k,
        sample_count=args.samples,
        random_state=args.seed,
    )
    calibration = memory.get("confidence_calibration")
    if calibration and not calibration_matches_dataset(calibration, fingerprint, engine_signature):
        logging.info("Ignoring stale confidence calibration for an incompatible dataset or engine signature")
        calibration = None
    if calibration:
        logging.info("Applying confidence calibration from memory %s", args.memory_file)
        calibrated_tickets: list[CandidateTicket] = []
        for ticket in tickets:
            calibrated_confidence = calibrate_confidence(ticket.score, calibration)
            calibrated_tickets.append(
                CandidateTicket(
                    main_numbers=ticket.main_numbers,
                    star_numbers=ticket.star_numbers,
                    score=ticket.score,
                    confidence=calibrated_confidence if calibrated_confidence is not None else ticket.confidence,
                )
            )
        tickets = calibrated_tickets
    logging.info("Generated %s ranked candidate tickets for %s", len(tickets), target_date.isoformat())

    top_main_probabilities = [
        {"number": int(number), "probability": round(float(probability), 6)}
        for number, probability in sorted(main_probabilities.items(), key=lambda item: item[1], reverse=True)[:10]
    ]
    top_star_probabilities = [
        {"number": int(number), "probability": round(float(probability), 6)}
        for number, probability in sorted(star_probabilities.items(), key=lambda item: item[1], reverse=True)[:6]
    ]

    report = {
        "train_end": train_end.isoformat(),
        "next_draw_date": target_date.isoformat(),
        "model_diagnostics": {
            "main": main_model.diagnostics,
            "star": star_model.diagnostics,
        },
        "top_main_probabilities": top_main_probabilities,
        "top_star_probabilities": top_star_probabilities,
        "predictions": [_ticket_payload(rank, ticket) for rank, ticket in enumerate(tickets, start=1)],
    }
    if calibration:
        source_summary = calibration.get("source_summary", {})
        report["confidence_calibration_source"] = {
            "mode": source_summary.get("mode", "unknown"),
            "test_start": source_summary.get("test_start", "unknown"),
            "test_end": source_summary.get("test_end", "unknown"),
            "observation_count": calibration.get("observation_count", 0),
        }
    _print_prediction_tables(report)
    _save_optional_output(args.output, report)
    _update_memory(
        args.memory_file,
        {
            "last_prediction_dataset_fingerprint": fingerprint,
            "engine_signature": engine_signature,
            "last_prediction": report,
        },
    )


def main() -> None:
    args = parse_args()
    if args.command == "backtest":
        run_backtest_command(args)
        return

    if args.command == "compare-strategies":
        run_compare_strategies_command(args)
        return

    if args.command == "predict-next":
        run_predict_command(args)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
