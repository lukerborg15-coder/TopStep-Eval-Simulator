from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import pandas as pd

from eval_sim.config import Instrument, TopstepRules, TOPSTEP_50K
from eval_sim.continuous_eval import run_continuous_eval
from eval_sim.evaluator import evaluate_window
from eval_sim.windows import WFWindows


HOLDOUT_RISK_GRID = [float(r) for r in range(100, 1100, 100)]  # $100–$1000 in $100 steps


@dataclass(frozen=True)
class RiskLevelResult:
    risk_dollars: float
    pass_rate: float
    mean_days_per_attempt: float
    expected_days_per_success: float  # mean_days_per_attempt / pass_rate; lower = faster funded


@dataclass(frozen=True)
class SizingResult:
    optimal_risk_dollars: float
    optimal_expected_days: float      # expected days per successful pass at optimal risk
    optimal_pass_rate: float
    all_levels: tuple[RiskLevelResult, ...]
    fixed_risk_result: RiskLevelResult | None


def _eval_risk_on_holdout(
    bars: pd.DataFrame,
    strategy_fn: Callable,
    params: dict,
    wf_windows: WFWindows,
    instrument: Instrument,
    risk_dollars: float,
    max_contracts: int,
    rules: TopstepRules,
) -> RiskLevelResult:
    from eval_sim.evaluator import Window
    holdout_window = Window(start=wf_windows.holdout.start, end=wf_windows.holdout.end)
    trades = evaluate_window(
        bars, strategy_fn, params, holdout_window,
        instrument, risk_dollars, max_contracts,
        warmup_start=wf_windows.holdout.warmup_start,
    )
    result = run_continuous_eval(trades, rules)
    expected = (
        result.mean_days_per_attempt / result.pass_rate
        if result.pass_rate > 0 else float("inf")
    )
    return RiskLevelResult(
        risk_dollars=risk_dollars,
        pass_rate=result.pass_rate,
        mean_days_per_attempt=result.mean_days_per_attempt,
        expected_days_per_success=expected,
    )


def run_sizing_optimizer(
    bars: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame, dict], pd.DataFrame],
    selected_params: dict,
    wf_windows: WFWindows,
    instrument: Instrument,
    max_contracts: int,
    fixed_risk_dollars: float | None,
    risk_grid: list[float] = HOLDOUT_RISK_GRID,
    min_pass_rate: float = 0.40,
    rules: TopstepRules = TOPSTEP_50K,
) -> SizingResult:
    """
    Test each risk level in risk_grid ($100–$1000 in $100 steps) on the holdout.

    Optimal = lowest expected_days_per_success (mean_days_per_attempt / pass_rate)
    among levels where pass_rate >= min_pass_rate. If none meet the floor, fall
    back to the highest pass_rate level.

    expected_days_per_success is the right metric here: it captures how quickly
    you'd expect to get funded, penalising both low pass rates and long attempts
    in a single number.
    """
    all_levels: list[RiskLevelResult] = []
    for risk in risk_grid:
        level = _eval_risk_on_holdout(
            bars, strategy_fn, selected_params, wf_windows,
            instrument, risk, max_contracts, rules,
        )
        all_levels.append(level)

    viable = [l for l in all_levels if l.pass_rate >= min_pass_rate]
    best = (
        min(viable, key=lambda l: l.expected_days_per_success)
        if viable
        else max(all_levels, key=lambda l: l.pass_rate)
    )

    fixed_result: RiskLevelResult | None = None
    if fixed_risk_dollars is not None:
        fixed_result = _eval_risk_on_holdout(
            bars, strategy_fn, selected_params, wf_windows,
            instrument, fixed_risk_dollars, max_contracts, rules,
        )

    return SizingResult(
        optimal_risk_dollars=best.risk_dollars,
        optimal_expected_days=best.expected_days_per_success,
        optimal_pass_rate=best.pass_rate,
        all_levels=tuple(all_levels),
        fixed_risk_result=fixed_result,
    )
