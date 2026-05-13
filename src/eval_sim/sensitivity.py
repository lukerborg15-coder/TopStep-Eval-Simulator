from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import pandas as pd

from eval_sim.config import Instrument, TopstepRules, TOPSTEP_50K
from eval_sim.evaluator import Window, evaluate_window
from eval_sim.topstep import simulate_seq_evals
from eval_sim.trades import TradeResult


@dataclass(frozen=True)
class ParamSensitivity:
    selected_value: object
    neighbor_results: dict  # {value: pass_rate}


@dataclass(frozen=True)
class SensitivityResult:
    selected_pass_rate: float
    param_results: dict[str, ParamSensitivity]
    is_cliff: bool
    cliff_params: tuple[str, ...]


def _pass_rate_for_params(
    bars: pd.DataFrame,
    strategy_fn: Callable,
    params: dict,
    window: Window,
    instrument: Instrument,
    risk_dollars: float,
    max_contracts: int,
    rules: TopstepRules,
    warmup_start: pd.Timestamp | None,
) -> float:
    trades = evaluate_window(
        bars, strategy_fn, params, window, instrument, risk_dollars, max_contracts,
        warmup_start=warmup_start,
    )
    seq = simulate_seq_evals(trades, rules)
    return seq.pass_rate


def run_sensitivity(
    bars: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame, dict], pd.DataFrame],
    selected_params: dict,
    param_ranges: dict[str, list],
    window: Window,
    instrument: Instrument,
    risk_dollars: float,
    max_contracts: int,
    rules: TopstepRules = TOPSTEP_50K,
    cliff_threshold_pct: float = 20.0,
    warmup_start: pd.Timestamp | None = None,
) -> SensitivityResult:
    """
    For each param, evaluate pass rate at neighboring values (one step up/down
    from selected). Flag as cliff if any neighbor drops >cliff_threshold_pct
    percentage points below selected.
    """
    selected_rate = _pass_rate_for_params(
        bars, strategy_fn, selected_params, window, instrument, risk_dollars, max_contracts, rules, warmup_start
    )

    param_results: dict[str, ParamSensitivity] = {}
    cliff_params: list[str] = []

    for param_key, all_values in param_ranges.items():
        selected_val = selected_params[param_key]
        if selected_val not in all_values:
            all_values = sorted(all_values + [selected_val])

        idx = all_values.index(selected_val)
        neighbor_values = []
        if idx > 0:
            neighbor_values.append(all_values[idx - 1])
        if idx < len(all_values) - 1:
            neighbor_values.append(all_values[idx + 1])

        neighbor_results: dict = {}
        is_cliff_for_param = False

        for neighbor_val in neighbor_values:
            neighbor_params = {**selected_params, param_key: neighbor_val}
            rate = _pass_rate_for_params(
                bars, strategy_fn, neighbor_params, window, instrument, risk_dollars, max_contracts, rules, warmup_start
            )
            neighbor_results[neighbor_val] = rate
            drop_pct = (selected_rate - rate) * 100
            if drop_pct > cliff_threshold_pct:
                is_cliff_for_param = True

        param_results[param_key] = ParamSensitivity(
            selected_value=selected_val,
            neighbor_results=neighbor_results,
        )
        if is_cliff_for_param:
            cliff_params.append(param_key)

    return SensitivityResult(
        selected_pass_rate=selected_rate,
        param_results=param_results,
        is_cliff=len(cliff_params) > 0,
        cliff_params=tuple(cliff_params),
    )
