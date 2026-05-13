import pandas as pd
import pytest
from tests.conftest import make_bars
from eval_sim.sensitivity import run_sensitivity, SensitivityResult
from eval_sim.evaluator import Window
from eval_sim.config import MNQ


PARAM_RANGES = {"stop_dist": [5.0, 10.0, 15.0, 20.0], "rr": [1.5, 2.0, 2.5]}
SELECTED = {"stop_dist": 10.0, "rr": 2.0}


def _strategy(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    import pandas as _pd
    signals = []
    for ts, row in list(bars.iterrows())[:5]:
        signals.append({
            "entry_time": ts,
            "direction": "long",
            "entry": row["close"],
            "stop": row["close"] - params["stop_dist"],
            "target": row["close"] + params["stop_dist"] * params["rr"],
        })
    return _pd.DataFrame(signals)


def test_run_sensitivity_returns_result(bars):
    window = Window(start=bars.index[0], end=bars.index[-1])
    result = run_sensitivity(bars, _strategy, SELECTED, PARAM_RANGES, window, MNQ,
                             risk_dollars=500.0, max_contracts=3)
    assert isinstance(result, SensitivityResult)


def test_run_sensitivity_has_per_param_results(bars):
    window = Window(start=bars.index[0], end=bars.index[-1])
    result = run_sensitivity(bars, _strategy, SELECTED, PARAM_RANGES, window, MNQ,
                             risk_dollars=500.0, max_contracts=3)
    assert "stop_dist" in result.param_results
    assert "rr" in result.param_results


def test_run_sensitivity_is_cliff_is_bool(bars):
    window = Window(start=bars.index[0], end=bars.index[-1])
    result = run_sensitivity(bars, _strategy, SELECTED, PARAM_RANGES, window, MNQ,
                             risk_dollars=500.0, max_contracts=3)
    assert isinstance(result.is_cliff, bool)


def test_run_sensitivity_single_value_param(bars):
    window = Window(start=bars.index[0], end=bars.index[-1])
    # Param with only one value — no neighbors, no cliff possible
    single_ranges = {"stop_dist": [10.0], "rr": [2.0]}
    result = run_sensitivity(bars, _strategy, SELECTED, single_ranges, window, MNQ,
                             risk_dollars=500.0, max_contracts=3)
    assert result.is_cliff is False
