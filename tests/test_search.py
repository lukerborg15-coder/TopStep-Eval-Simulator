import pandas as pd
import pytest
from tests.conftest import make_bars
from eval_sim.search import random_search, sample_params, SearchResult
from eval_sim.windows import compute_windows, WFWindows
from eval_sim.config import MNQ, DataSplitConfig, SearchConfig


PARAM_RANGES = {"stop_dist": [5.0, 10.0, 15.0], "rr": [1.5, 2.0]}


def _simple_strategy(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    signals = []
    for ts, row in list(bars.iterrows())[:3]:
        signals.append({
            "entry_time": ts,
            "direction": "long",
            "entry": row["close"],
            "stop": row["close"] - params["stop_dist"],
            "target": row["close"] + params["stop_dist"] * params["rr"],
        })
    return pd.DataFrame(signals)


def _make_windows() -> WFWindows:
    # Span wide enough to satisfy the 6-month WF minimum check.
    # warmup_bars=0 so windows start exactly at scoring dates (no bar index needed).
    start = pd.Timestamp("2018-01-01", tz="US/Eastern")
    end = pd.Timestamp("2022-07-01", tz="US/Eastern")
    return compute_windows(
        start, end,
        bars_index=None,
        config=DataSplitConfig(wf_years=3.5, holdout_months=6, n_wf_windows=2, warmup_bars=0),
    )


def test_sample_params_returns_dict_from_ranges():
    import numpy as np
    rng = np.random.default_rng(42)
    params = sample_params(PARAM_RANGES, rng)
    assert set(params.keys()) == set(PARAM_RANGES.keys())
    assert params["stop_dist"] in PARAM_RANGES["stop_dist"]
    assert params["rr"] in PARAM_RANGES["rr"]


def test_sample_params_n_candidates_unique_enough():
    import numpy as np
    rng = np.random.default_rng(42)
    samples = [sample_params(PARAM_RANGES, rng) for _ in range(20)]
    unique = {tuple(sorted(s.items())) for s in samples}
    assert len(unique) >= 2  # should get more than one unique combo


def test_random_search_returns_search_result(bars):
    windows = _make_windows()
    result = random_search(
        bars=bars,
        strategy_fn=_simple_strategy,
        param_ranges=PARAM_RANGES,
        wf_windows=windows,
        instrument=MNQ,
        risk_dollars=500.0,
        max_contracts=3,
        config=SearchConfig(n_candidates=6, seed=42),
        n_workers=1,
    )
    assert isinstance(result, SearchResult)
    assert set(result.best_params.keys()) == set(PARAM_RANGES.keys())
    assert 0.0 <= result.best_score <= 1.0
    assert result.n_evaluated >= 1


def test_random_search_fast_mode_skips_search(bars):
    windows = _make_windows()
    result = random_search(
        bars=bars,
        strategy_fn=_simple_strategy,
        param_ranges=PARAM_RANGES,
        wf_windows=windows,
        instrument=MNQ,
        risk_dollars=500.0,
        max_contracts=3,
        config=SearchConfig(n_candidates=6, seed=42),
        n_workers=1,
        fast_mode=True,
    )
    assert result.n_evaluated == 0
    # Params should be midpoints of each range
    assert result.best_params["stop_dist"] == 10.0
    assert result.best_params["rr"] == 2.0


def test_random_search_best_params_in_ranges(bars):
    windows = _make_windows()
    result = random_search(
        bars=bars,
        strategy_fn=_simple_strategy,
        param_ranges=PARAM_RANGES,
        wf_windows=windows,
        instrument=MNQ,
        risk_dollars=500.0,
        max_contracts=3,
        config=SearchConfig(n_candidates=10, seed=42),
        n_workers=1,
    )
    for key, value in result.best_params.items():
        assert value in PARAM_RANGES[key], f"{key}={value} not in {PARAM_RANGES[key]}"
