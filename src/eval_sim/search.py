from __future__ import annotations
import multiprocessing
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd

from eval_sim.config import Instrument, SearchConfig, SEARCH, TOPSTEP_50K
from eval_sim.evaluator import evaluate_window
from eval_sim.topstep import simulate_seq_evals
from eval_sim.windows import WFWindows
from eval_sim.trades import TradeResult


@dataclass(frozen=True)
class SearchResult:
    best_params: dict
    best_score: float   # mean seq eval pass rate across OOS folds
    n_evaluated: int


def sample_params(param_ranges: dict[str, list], rng: np.random.Generator) -> dict:
    """Sample one random combination from param_ranges."""
    return {key: rng.choice(values).item() for key, values in param_ranges.items()}


def _midpoint_params(param_ranges: dict[str, list]) -> dict:
    """Return midpoint value for each param range (used in --fast mode)."""
    return {key: values[len(values) // 2] for key, values in param_ranges.items()}


def _score_candidate(
    bars: pd.DataFrame,
    strategy_fn: Callable,
    params: dict,
    wf_windows: WFWindows,
    instrument: Instrument,
    risk_dollars: float,
    max_contracts: int,
) -> float:
    """Score one param set by mean seq eval pass rate across all WF OOS windows."""
    window_scores: list[float] = []
    for w in wf_windows.wf_windows:
        from eval_sim.evaluator import Window
        scoring_window = Window(start=w.start, end=w.end)
        trades = evaluate_window(
            bars, strategy_fn, params, scoring_window,
            instrument, risk_dollars, max_contracts,
            warmup_start=w.warmup_start,
        )
        seq = simulate_seq_evals(trades, TOPSTEP_50K)
        window_scores.append(seq.pass_rate)
    return float(np.mean(window_scores)) if window_scores else 0.0


def _worker(args: tuple) -> tuple[dict, float]:
    """Multiprocessing worker: (bars, fn, params, windows, instrument, risk, max_c) -> (params, score)"""
    bars, strategy_fn, params, wf_windows, instrument, risk_dollars, max_contracts = args
    score = _score_candidate(bars, strategy_fn, params, wf_windows, instrument, risk_dollars, max_contracts)
    return params, score


def random_search(
    bars: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame, dict], pd.DataFrame],
    param_ranges: dict[str, list],
    wf_windows: WFWindows,
    instrument: Instrument,
    risk_dollars: float,
    max_contracts: int,
    config: SearchConfig = SEARCH,
    n_workers: int | None = None,
    fast_mode: bool = False,
) -> SearchResult:
    """
    Randomly sample n_candidates param combinations and score each across WF OOS folds.
    Returns the best params by mean sequential Combine eval pass rate.

    fast_mode=True: skip search, return midpoint params with score=0 and n_evaluated=0.
    """
    if fast_mode:
        return SearchResult(
            best_params=_midpoint_params(param_ranges),
            best_score=0.0,
            n_evaluated=0,
        )

    rng = np.random.default_rng(config.seed)
    candidates = [sample_params(param_ranges, rng) for _ in range(config.n_candidates)]

    worker_args = [
        (bars, strategy_fn, params, wf_windows, instrument, risk_dollars, max_contracts)
        for params in candidates
    ]

    if n_workers == 1:
        results = [_worker(args) for args in worker_args]
    else:
        # On Windows, multiprocessing uses "spawn" (not "fork"), which requires
        # the entry point to be protected by `if __name__ == "__main__"`.
        # We use get_context("spawn") explicitly so behaviour is consistent
        # across platforms and freeze_support() in cli.py covers the frozen case.
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(_worker, worker_args)

    best_params, best_score = max(results, key=lambda x: x[1])
    return SearchResult(
        best_params=best_params,
        best_score=best_score,
        n_evaluated=len(candidates),
    )
