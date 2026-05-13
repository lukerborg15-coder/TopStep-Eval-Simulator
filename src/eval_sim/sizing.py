from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import pandas as pd

from eval_sim.config import Instrument, TopstepRules, TOPSTEP_50K
from eval_sim.continuous_eval import run_continuous_eval
from eval_sim.evaluator import evaluate_window
from eval_sim.windows import WFWindows


HOLDOUT_RISK_GRID = [float(r) for r in range(100, 1100, 100)]  # $100-$1000 in $100 steps


@dataclass(frozen=True)
class RiskLevelResult:
    risk_dollars: float
    pass_rate: float
    mean_days_per_attempt: float
    expected_days_per_success: float  # mean_days_per_attempt / pass_rate; lower = faster funded


@dataclass(frozen=True)
class SizingResult:
    optimal_risk_dollars: float
    optimal_expected_days: float      # holdout expected days per successful pass at selected risk
    optimal_pass_rate: float          # holdout pass rate at selected risk
    all_levels: tuple[RiskLevelResult, ...]
    fixed_risk_result: RiskLevelResult | None
    training_levels: tuple[RiskLevelResult, ...] = ()
    optimal_training_result: RiskLevelResult | None = None
    optimal_holdout_result: RiskLevelResult | None = None
    selection_source: str = "wf"


def _expected_days(result) -> float:
    return (
        result.mean_days_per_attempt / result.pass_rate
        if result.pass_rate > 0 else float("inf")
    )


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
    return RiskLevelResult(
        risk_dollars=risk_dollars,
        pass_rate=result.pass_rate,
        mean_days_per_attempt=result.mean_days_per_attempt,
        expected_days_per_success=_expected_days(result),
    )


def _eval_risk_on_wf(
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

    window_results = []
    for w in wf_windows.wf_windows:
        scoring_window = Window(start=w.start, end=w.end)
        trades = evaluate_window(
            bars, strategy_fn, params, scoring_window,
            instrument, risk_dollars, max_contracts,
            warmup_start=w.warmup_start,
        )
        window_results.append(run_continuous_eval(trades, rules))

    if not window_results:
        return RiskLevelResult(
            risk_dollars=risk_dollars,
            pass_rate=0.0,
            mean_days_per_attempt=0.0,
            expected_days_per_success=float("inf"),
        )

    pass_rate = sum(r.pass_rate for r in window_results) / len(window_results)
    mean_days = sum(r.mean_days_per_attempt for r in window_results) / len(window_results)
    expected = mean_days / pass_rate if pass_rate > 0 else float("inf")
    return RiskLevelResult(
        risk_dollars=risk_dollars,
        pass_rate=pass_rate,
        mean_days_per_attempt=mean_days,
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
    Select risk using WF windows, then validate/report each risk level on holdout.

    Optimal = lowest expected_days_per_success among WF levels where pass_rate
    >= min_pass_rate. If none meet the floor, fall back to the highest WF
    pass_rate level. Holdout metrics are reported after selection and are not
    used to choose the risk.
    """
    training_levels: list[RiskLevelResult] = []
    for risk in risk_grid:
        level = _eval_risk_on_wf(
            bars, strategy_fn, selected_params, wf_windows,
            instrument, risk, max_contracts, rules,
        )
        training_levels.append(level)

    viable = [l for l in training_levels if l.pass_rate >= min_pass_rate]
    best_training = (
        min(viable, key=lambda l: l.expected_days_per_success)
        if viable
        else max(training_levels, key=lambda l: l.pass_rate)
    )

    all_levels: list[RiskLevelResult] = []
    for risk in risk_grid:
        level = _eval_risk_on_holdout(
            bars, strategy_fn, selected_params, wf_windows,
            instrument, risk, max_contracts, rules,
        )
        all_levels.append(level)

    best_holdout = next(
        level for level in all_levels
        if level.risk_dollars == best_training.risk_dollars
    )

    fixed_result: RiskLevelResult | None = None
    if fixed_risk_dollars is not None:
        fixed_result = _eval_risk_on_holdout(
            bars, strategy_fn, selected_params, wf_windows,
            instrument, fixed_risk_dollars, max_contracts, rules,
        )

    return SizingResult(
        optimal_risk_dollars=best_training.risk_dollars,
        optimal_expected_days=best_holdout.expected_days_per_success,
        optimal_pass_rate=best_holdout.pass_rate,
        all_levels=tuple(all_levels),
        fixed_risk_result=fixed_result,
        training_levels=tuple(training_levels),
        optimal_training_result=best_training,
        optimal_holdout_result=best_holdout,
        selection_source="wf",
    )
