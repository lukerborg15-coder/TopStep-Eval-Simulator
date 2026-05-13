import pandas as pd
import pytest
from tests.conftest import make_bars
from eval_sim.sizing import run_sizing_optimizer, SizingResult
from eval_sim.windows import compute_windows
from eval_sim.config import MNQ, DataSplitConfig


RISK_GRID = [200.0, 350.0, 500.0, 750.0, 1000.0]
SELECTED_PARAMS = {"stop_dist": 10.0, "rr": 2.0}


def _strategy(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    signals = []
    for ts, row in list(bars.iterrows())[:3]:
        signals.append({
            "entry_time": ts, "direction": "long",
            "entry": row["close"],
            "stop": row["close"] - params["stop_dist"],
            "target": row["close"] + params["stop_dist"] * params["rr"],
        })
    import pandas as _pd
    return _pd.DataFrame(signals)


def _windows():
    start = pd.Timestamp("2020-01-01", tz="US/Eastern")
    end = pd.Timestamp("2023-07-01", tz="US/Eastern")
    return compute_windows(
        start, end,
        bars_index=None,
        config=DataSplitConfig(wf_years=2.5, holdout_months=6, n_wf_windows=2, warmup_bars=0),
    )


RISK_GRID = [100.0, 200.0, 300.0, 400.0, 500.0]  # subset for test speed


def test_run_sizing_optimizer_returns_result(bars):
    windows = _windows()
    result = run_sizing_optimizer(
        bars=bars, strategy_fn=_strategy, selected_params=SELECTED_PARAMS,
        wf_windows=windows, instrument=MNQ, max_contracts=5,
        fixed_risk_dollars=None, risk_grid=RISK_GRID,
    )
    assert isinstance(result, SizingResult)


def test_optimal_risk_in_grid(bars):
    windows = _windows()
    result = run_sizing_optimizer(
        bars=bars, strategy_fn=_strategy, selected_params=SELECTED_PARAMS,
        wf_windows=windows, instrument=MNQ, max_contracts=5,
        fixed_risk_dollars=None, risk_grid=RISK_GRID,
    )
    assert result.optimal_risk_dollars in RISK_GRID


def test_all_levels_covers_full_grid(bars):
    windows = _windows()
    result = run_sizing_optimizer(
        bars=bars, strategy_fn=_strategy, selected_params=SELECTED_PARAMS,
        wf_windows=windows, instrument=MNQ, max_contracts=5,
        fixed_risk_dollars=None, risk_grid=RISK_GRID,
    )
    assert len(result.all_levels) == len(RISK_GRID)
    tested_risks = {l.risk_dollars for l in result.all_levels}
    assert tested_risks == set(RISK_GRID)


def test_fixed_risk_comparison_included(bars):
    windows = _windows()
    result = run_sizing_optimizer(
        bars=bars, strategy_fn=_strategy, selected_params=SELECTED_PARAMS,
        wf_windows=windows, instrument=MNQ, max_contracts=5,
        fixed_risk_dollars=500.0, risk_grid=RISK_GRID,
    )
    assert result.fixed_risk_result is not None
    assert result.fixed_risk_result.risk_dollars == 500.0
    assert 0.0 <= result.fixed_risk_result.pass_rate <= 1.0


def test_no_fixed_risk_result_is_none(bars):
    windows = _windows()
    result = run_sizing_optimizer(
        bars=bars, strategy_fn=_strategy, selected_params=SELECTED_PARAMS,
        wf_windows=windows, instrument=MNQ, max_contracts=5,
        fixed_risk_dollars=None, risk_grid=RISK_GRID,
    )
    assert result.fixed_risk_result is None
