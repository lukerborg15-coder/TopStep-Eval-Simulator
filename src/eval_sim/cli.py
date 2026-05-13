from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime
from dataclasses import replace
from pathlib import Path

from eval_sim.config import INSTRUMENTS, DATA_SPLIT, SEARCH, TOPSTEP_50K
from eval_sim.continuous_eval import run_continuous_eval
from eval_sim.data import load_ohlcv, DataLoadError
from eval_sim.evaluator import evaluate_window
from eval_sim.monte_carlo import run_monte_carlo
from eval_sim.search import random_search, SearchConfig
from eval_sim.sensitivity import run_sensitivity
from eval_sim.sizing import run_sizing_optimizer
from eval_sim.strategy import load_strategy, StrategyLoadError
from eval_sim.verdict import compute_verdict, VerdictThresholds, DEFAULT_THRESHOLDS
from eval_sim.windows import compute_windows


RISK_GRID = [150.0, 200.0, 300.0, 400.0, 500.0, 650.0, 800.0, 1000.0, 1200.0, 1500.0]
FAST_RISK_GRID = [500.0]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TopStep Eval Sim — continuous Combine evaluation pipeline")
    p.add_argument("--strategy", required=True, help="Path to strategy .py file")
    p.add_argument("--instrument", choices=list(INSTRUMENTS.keys()), default="mnq")
    p.add_argument("--timeframe", default="5min")
    p.add_argument("--fast", action="store_true", help="Skip WF search, use param range midpoints")
    p.add_argument("--search-n", type=int, default=SEARCH.n_candidates, help="Random search candidates")
    p.add_argument("--fixed-risk", type=float, default=None, help="Fixed risk dollars for sizing comparison")
    p.add_argument("--max-contracts", type=int, default=10)
    p.add_argument("--mc-n", type=int, default=500, help="Monte Carlo permutations")
    p.add_argument("--mc-block-size", type=int, default=5)
    p.add_argument("--data-dir", default="Data")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--reject-pass-rate", type=float, default=0.30)
    p.add_argument("--reject-max-dd", type=float, default=1800.0)
    p.add_argument("--ready-pass-rate", type=float, default=0.60)
    p.add_argument("--ready-max-dd", type=float, default=1200.0)
    return p


def _header(text: str) -> None:
    print(f"\n=== {text} ===")


def main(argv: list[str] | None = None) -> int:
    # Required on Windows when using multiprocessing with spawn context.
    # Must be called before any other multiprocessing code runs.
    import multiprocessing
    multiprocessing.freeze_support()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    args = build_parser().parse_args(argv)

    # Load strategy
    try:
        strategy = load_strategy(args.strategy)
    except StrategyLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Load data
    try:
        bars = load_ohlcv(args.instrument, args.timeframe, data_dir=args.data_dir)
    except DataLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    instrument = INSTRUMENTS[args.instrument]
    data_start, data_end = bars.index[0], bars.index[-1]

    # Compute windows
    try:
        split_config = replace(DATA_SPLIT, warmup_bars=strategy.warmup_bars)
        windows = compute_windows(data_start, data_end, bars_index=bars.index, config=split_config)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    thresholds = VerdictThresholds(
        reject_pass_rate=args.reject_pass_rate,
        reject_max_dd=args.reject_max_dd,
        ready_pass_rate=args.ready_pass_rate,
        ready_max_dd=args.ready_max_dd,
    )

    # Stage 1: Search
    _header("Stage 1 — Param Search")
    search_result = random_search(
        bars=bars,
        strategy_fn=strategy.generate_signals,
        param_ranges=strategy.param_ranges,
        wf_windows=windows,
        instrument=instrument,
        risk_dollars=500.0,
        max_contracts=args.max_contracts,
        config=SearchConfig(n_candidates=args.search_n, seed=SEARCH.seed),
        fast_mode=args.fast,
        strategy_path=args.strategy,
    )
    print(f"{'FAST MODE' if args.fast else f'Evaluated {search_result.n_evaluated} candidates'}")
    print(f"Selected params: {search_result.best_params}")
    print(f"WF OOS score: {search_result.best_score:.3f}")

    selected_params = search_result.best_params

    # Stage 2: Sensitivity
    _header("Stage 2 — Sensitivity Sweep")
    # Per-window re-evaluation of selected params for reporting
    # (search only stored aggregate score; this gives per-window stats)
    _header("Stage 2 — Per-Window Evaluation (selected params)")
    from eval_sim.evaluator import Window
    wf_window_results = []
    for idx, w in enumerate(windows.wf_windows, 1):
        scoring_window = Window(start=w.start, end=w.end)
        w_trades = evaluate_window(
            bars, strategy.generate_signals, selected_params,
            scoring_window, instrument, 500.0, args.max_contracts,
            warmup_start=w.warmup_start,
        )
        w_result = run_continuous_eval(w_trades)
        n_trades = len(w_trades)
        wf_window_results.append((idx, w, w_result, n_trades))
        if w_result.is_empty:
            status = f"EMPTY  n_trades={n_trades}"
        else:
            status = (
                f"pass={w_result.pass_rate:.3f} mean_days={w_result.mean_days_per_attempt:.1f} "
                f"median_days={w_result.median_days_per_attempt:.1f}  n_trades={n_trades}"
            )
        print(f"  Window {idx} ({w.start.date()} → {w.end.date()}): {status}")

    # Sensitivity sweep uses the last WF window with warmup
    _header("Stage 3 — Sensitivity Sweep")
    last_wf = windows.wf_windows[-1] if windows.wf_windows else None
    wf_dev_window = Window(start=last_wf.start, end=last_wf.end) if last_wf else Window(start=windows.holdout.start, end=windows.holdout.end)
    wf_dev_warmup = last_wf.warmup_start if last_wf else windows.holdout.warmup_start
    sensitivity = run_sensitivity(
        bars=bars,
        strategy_fn=strategy.generate_signals,
        selected_params=selected_params,
        param_ranges=strategy.param_ranges,
        window=wf_dev_window,
        instrument=instrument,
        risk_dollars=500.0,
        max_contracts=args.max_contracts,
        warmup_start=wf_dev_warmup,
    )
    cliff_label = "CLIFF DETECTED" if sensitivity.is_cliff else "flat"
    print(f"Sensitivity: {cliff_label}  selected_pass_rate={sensitivity.selected_pass_rate:.3f}")
    if sensitivity.is_cliff:
        print(f"  cliff params: {', '.join(sensitivity.cliff_params)}")

    # Stage 3: Continuous Eval Sim (holdout)
    _header("Stage 3 — Continuous Eval Sim (holdout)")
    from eval_sim.evaluator import Window
    holdout_window = Window(start=windows.holdout.start, end=windows.holdout.end)
    holdout_trades = evaluate_window(
        bars, strategy.generate_signals, selected_params,
        holdout_window, instrument, 500.0, args.max_contracts,
        warmup_start=windows.holdout.warmup_start,
    )
    eval_result = run_continuous_eval(holdout_trades)
    holdout_n_trades = len(holdout_trades)
    print(
        f"pass_rate={eval_result.pass_rate:.3f}  passes={eval_result.passes}  "
        f"attempts={eval_result.attempts}  n_trades={holdout_n_trades}  "
        f"worst_dd=${eval_result.worst_attempt_drawdown:.0f}"
    )

    # Stage 4: Monte Carlo
    _header("Stage 4 — Monte Carlo")
    mc_result = run_monte_carlo(holdout_trades, n=args.mc_n, block_size=args.mc_block_size)
    print(
        f"pass_rate_p05={mc_result.pass_rate_p05:.3f}  "
        f"median={mc_result.pass_rate_median:.3f}  "
        f"p95={mc_result.pass_rate_p95:.3f}"
    )

    # Stage 5: Sizing
    _header("Stage 5 — Sizing Optimizer")
    sizing = run_sizing_optimizer(
        bars=bars,
        strategy_fn=strategy.generate_signals,
        selected_params=selected_params,
        wf_windows=windows,
        instrument=instrument,
        max_contracts=args.max_contracts,
        fixed_risk_dollars=args.fixed_risk,
        risk_grid=FAST_RISK_GRID if args.fast else RISK_GRID,
    )
    print(f"Optimal risk: ${sizing.optimal_risk_dollars:.0f}  pass rate: {sizing.optimal_pass_rate:.3f}  expected days/success: {sizing.optimal_expected_days:.1f}")
    print("Risk grid (holdout):")
    for lvl in sizing.all_levels:
        marker = " <-- optimal" if lvl.risk_dollars == sizing.optimal_risk_dollars else ""
        print(f"  ${lvl.risk_dollars:.0f}  pass={lvl.pass_rate:.3f}  days/success={lvl.expected_days_per_success:.1f}{marker}")
    if sizing.fixed_risk_result is not None:
        fr = sizing.fixed_risk_result
        print(f"Fixed ${fr.risk_dollars:.0f}: pass={fr.pass_rate:.3f}  days/success={fr.expected_days_per_success:.1f}")

    # Stage 6: Verdict
    _header("Stage 6 — Verdict")
    verdict = compute_verdict(
        pass_rate=eval_result.pass_rate,
        mc_pass_rate_p05=mc_result.pass_rate_p05,
        worst_drawdown=eval_result.worst_attempt_drawdown,
        sensitivity_is_cliff=sensitivity.is_cliff,
        thresholds=thresholds,
    )
    print(f"VERDICT: {verdict.verdict}")
    for r in verdict.reject_reasons:
        print(f"  REJECT: {r}")
    for w in verdict.warn_reasons:
        print(f"  WARN: {w}")

    # Write JSON output (timestamped so reruns do not clobber prior results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_file = output_dir / f"{strategy.name}_{args.instrument}_{args.timeframe}_result_{stamp}.json"

    holdout_payload = {
        "start": str(windows.holdout.start.date()),
        "end": str(windows.holdout.end.date()),
        "is_empty": eval_result.is_empty,
        "pass_rate": eval_result.pass_rate,
        "passes": eval_result.passes,
        "attempts": eval_result.attempts,
        "mean_days_per_attempt": eval_result.mean_days_per_attempt,
        "median_days_per_attempt": eval_result.median_days_per_attempt,
        "worst_attempt_drawdown": eval_result.worst_attempt_drawdown,
        "n_trades": holdout_n_trades,
    }

    result_bundle = {
        "strategy": strategy.name,
        "instrument": args.instrument,
        "timeframe": args.timeframe,
        "fast_mode": args.fast,
        "topstep_rules": {
            "account_size": TOPSTEP_50K.account_size,
            "profit_target": TOPSTEP_50K.profit_target,
            "max_drawdown": TOPSTEP_50K.max_drawdown,
            "daily_loss_limit": TOPSTEP_50K.daily_loss_limit,
            "consistency_pct": TOPSTEP_50K.consistency_pct,
            "max_trading_days": TOPSTEP_50K.max_trading_days,
            "drawdown_type": "EOD_trailing",
        },
        "data_range": {"start": str(data_start), "end": str(data_end)},
        "best_params": search_result.best_params,
        "search": {
            "n_evaluated": search_result.n_evaluated,
            "wf_oos_score": search_result.best_score,
        },
        "wf_windows": [
            {
                "window": idx,
                "start": str(w.start.date()),
                "end": str(w.end.date()),
                "is_empty": res.is_empty,
                "pass_rate": res.pass_rate,
                "passes": res.passes,
                "attempts": res.attempts,
                "mean_days_per_attempt": res.mean_days_per_attempt,
                "median_days_per_attempt": res.median_days_per_attempt,
                "worst_drawdown": res.worst_attempt_drawdown,
                "n_trades": n_trades,
            }
            for idx, w, res, n_trades in wf_window_results
        ],
        "wf_aggregate": {
            "mean_pass_rate": (
                sum(r.pass_rate for _, _, r, _ in wf_window_results if not r.is_empty)
                / max(1, sum(1 for _, _, r, _ in wf_window_results if not r.is_empty))
            ),
            "empty_windows": sum(1 for _, _, r, _ in wf_window_results if r.is_empty),
        },
        "sensitivity": {
            "is_cliff": sensitivity.is_cliff,
            "cliff_params": list(sensitivity.cliff_params),
            "selected_pass_rate": sensitivity.selected_pass_rate,
            "param_results": {
                param: {
                    "selected_value": ps.selected_value,
                    "neighbors": {str(k): v for k, v in ps.neighbor_results.items()},
                }
                for param, ps in sensitivity.param_results.items()
            },
        },
        "holdout": holdout_payload,
        "continuous_eval": holdout_payload,
        "monte_carlo": {
            "n_permutations": mc_result.n_permutations,
            "pass_rate_p05": mc_result.pass_rate_p05,
            "pass_rate_median": mc_result.pass_rate_median,
            "pass_rate_p95": mc_result.pass_rate_p95,
            "worst_drawdown_median": mc_result.worst_drawdown_median,
        },
        "sizing": {
            "selection_source": sizing.selection_source,
            "optimal_risk_dollars": sizing.optimal_risk_dollars,
            "optimal_pass_rate": sizing.optimal_pass_rate,
            "optimal_expected_days_per_success": sizing.optimal_expected_days,
            "optimal_training": (
                {
                    "risk_dollars": sizing.optimal_training_result.risk_dollars,
                    "pass_rate": sizing.optimal_training_result.pass_rate,
                    "mean_days_per_attempt": sizing.optimal_training_result.mean_days_per_attempt,
                    "expected_days_per_success": sizing.optimal_training_result.expected_days_per_success,
                }
                if sizing.optimal_training_result is not None else None
            ),
            "optimal_holdout": (
                {
                    "risk_dollars": sizing.optimal_holdout_result.risk_dollars,
                    "pass_rate": sizing.optimal_holdout_result.pass_rate,
                    "mean_days_per_attempt": sizing.optimal_holdout_result.mean_days_per_attempt,
                    "expected_days_per_success": sizing.optimal_holdout_result.expected_days_per_success,
                }
                if sizing.optimal_holdout_result is not None else None
            ),
            "training_levels": [
                {
                    "risk_dollars": l.risk_dollars,
                    "pass_rate": l.pass_rate,
                    "mean_days_per_attempt": l.mean_days_per_attempt,
                    "expected_days_per_success": l.expected_days_per_success,
                }
                for l in sizing.training_levels
            ],
            "all_levels": [
                {
                    "risk_dollars": l.risk_dollars,
                    "pass_rate": l.pass_rate,
                    "mean_days_per_attempt": l.mean_days_per_attempt,
                    "expected_days_per_success": l.expected_days_per_success,
                }
                for l in sizing.all_levels
            ],
            "fixed_risk": (
                {
                    "risk_dollars": sizing.fixed_risk_result.risk_dollars,
                    "pass_rate": sizing.fixed_risk_result.pass_rate,
                    "mean_days_per_attempt": sizing.fixed_risk_result.mean_days_per_attempt,
                    "expected_days_per_success": sizing.fixed_risk_result.expected_days_per_success,
                }
                if sizing.fixed_risk_result is not None else None
            ),
        },
        "verdict": {
            "verdict": verdict.verdict,
            "reject_reasons": list(verdict.reject_reasons),
            "warn_reasons": list(verdict.warn_reasons),
        },
    }

    out_file.write_text(json.dumps(result_bundle, indent=2, default=str))
    print(f"\nResult written to: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
