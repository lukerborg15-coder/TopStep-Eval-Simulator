import pandas as pd
import pytest
from tests.conftest import make_bars
from eval_sim.continuous_eval import ContinuousEvalResult
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


def test_optimal_risk_selected_from_wf_not_holdout(monkeypatch, bars):
    windows = _windows()
    calls = []

    def fake_evaluate_window(
        bars,
        strategy_fn,
        params,
        window,
        instrument,
        risk_dollars,
        max_contracts,
        warmup_start=None,
    ):
        phase = "holdout" if window.start == windows.holdout.start else "wf"
        calls.append((phase, risk_dollars, warmup_start))
        return [(phase, risk_dollars)]

    def fake_run_continuous_eval(trades, rules):
        phase, risk_dollars = trades[0]
        pass_rates = {
            ("wf", 100.0): 0.80,
            ("wf", 200.0): 0.45,
            ("holdout", 100.0): 0.40,
            ("holdout", 200.0): 0.95,
        }
        pass_rate = pass_rates[(phase, risk_dollars)]
        return ContinuousEvalResult(
            passes=int(pass_rate * 100),
            attempts=100,
            pass_rate=pass_rate,
            worst_attempt_drawdown=0.0,
            mean_days_per_attempt=10.0,
            median_days_per_attempt=10.0,
            total_trading_days=100,
            is_empty=False,
        )

    monkeypatch.setattr("eval_sim.sizing.evaluate_window", fake_evaluate_window)
    monkeypatch.setattr("eval_sim.sizing.run_continuous_eval", fake_run_continuous_eval)

    result = run_sizing_optimizer(
        bars=bars,
        strategy_fn=_strategy,
        selected_params=SELECTED_PARAMS,
        wf_windows=windows,
        instrument=MNQ,
        max_contracts=5,
        fixed_risk_dollars=None,
        risk_grid=[100.0, 200.0],
        min_pass_rate=0.40,
    )

    assert result.optimal_risk_dollars == 100.0
    assert result.optimal_training_result.risk_dollars == 100.0
    assert result.optimal_training_result.pass_rate == pytest.approx(0.80)
    assert result.optimal_holdout_result.risk_dollars == 100.0
    assert result.optimal_holdout_result.pass_rate == pytest.approx(0.40)
    assert {level.risk_dollars for level in result.training_levels} == {100.0, 200.0}
    assert {level.risk_dollars for level in result.all_levels} == {100.0, 200.0}
    assert max(result.all_levels, key=lambda level: level.pass_rate).risk_dollars == 200.0
    assert any(phase == "wf" for phase, _, _ in calls)
    assert any(phase == "holdout" for phase, _, _ in calls)
