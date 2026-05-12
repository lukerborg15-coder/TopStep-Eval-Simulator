import pandas as pd
import pytest
from eval_sim.windows import compute_windows, WFWindows, OOSWindow
from eval_sim.config import DataSplitConfig


_SPLIT = DataSplitConfig(
    wf_years=3.5,
    holdout_months=18,
    n_wf_windows=4,
    warmup_bars=500,
)


def test_compute_windows_returns_wfwindows():
    start = pd.Timestamp("2019-01-01", tz="US/Eastern")
    end = pd.Timestamp("2024-07-01", tz="US/Eastern")
    result = compute_windows(start, end, bars_index=None, config=_SPLIT)
    assert isinstance(result, WFWindows)


def test_holdout_is_last_18_months():
    start = pd.Timestamp("2019-01-01", tz="US/Eastern")
    end = pd.Timestamp("2024-07-01", tz="US/Eastern")
    result = compute_windows(start, end, bars_index=None, config=_SPLIT)
    holdout_months = (result.holdout.end - result.holdout.start).days / 30.44
    assert 16 <= holdout_months <= 20


def test_wf_window_count_matches_config():
    start = pd.Timestamp("2019-01-01", tz="US/Eastern")
    end = pd.Timestamp("2024-07-01", tz="US/Eastern")
    result = compute_windows(start, end, bars_index=None, config=_SPLIT)
    assert len(result.wf_windows) == _SPLIT.n_wf_windows


def test_wf_windows_cover_full_wf_period():
    start = pd.Timestamp("2019-01-01", tz="US/Eastern")
    end = pd.Timestamp("2024-07-01", tz="US/Eastern")
    result = compute_windows(start, end, bars_index=None, config=_SPLIT)
    assert result.wf_windows[0].start >= result.wf_start
    assert result.wf_windows[-1].end <= result.wf_end


def test_wf_windows_do_not_overlap():
    start = pd.Timestamp("2019-01-01", tz="US/Eastern")
    end = pd.Timestamp("2024-07-01", tz="US/Eastern")
    result = compute_windows(start, end, bars_index=None, config=_SPLIT)
    for i in range(len(result.wf_windows) - 1):
        assert result.wf_windows[i].end <= result.wf_windows[i + 1].start


def test_warmup_start_precedes_scoring_start():
    start = pd.Timestamp("2019-01-01", tz="US/Eastern")
    end = pd.Timestamp("2024-07-01", tz="US/Eastern")
    # Need a real bars_index so warmup_start can be computed in bar-count terms
    dates = pd.date_range(start, end, freq="5min", tz="US/Eastern")
    result = compute_windows(start, end, bars_index=dates, config=_SPLIT)
    for w in result.wf_windows:
        assert w.warmup_start <= w.start
    assert result.holdout.warmup_start <= result.holdout.start


def test_insufficient_data_raises():
    start = pd.Timestamp("2023-01-01", tz="US/Eastern")
    end = pd.Timestamp("2023-06-01", tz="US/Eastern")
    with pytest.raises(ValueError, match="insufficient"):
        compute_windows(start, end, bars_index=None, config=_SPLIT)
