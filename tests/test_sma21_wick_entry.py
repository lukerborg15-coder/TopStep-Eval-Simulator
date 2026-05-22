"""Tests for sma21_wick_entry strategy."""
import json
from pathlib import Path

import pandas as pd

from eval_sim.strategy import load_strategy
from tests.conftest import make_bars

STRAT_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "eval_sim"
    / "strategies"
    / "sma21_wick_entry.py"
)


def test_sma21_wick_entry_loads():
    contract = load_strategy(STRAT_PATH)
    assert contract.name == "sma21_wick_entry"
    assert "flat_only" in contract.param_ranges
    assert contract.warmup_bars == 80


def test_sma21_wick_entry_smoke():
    contract = load_strategy(STRAT_PATH)
    bars = make_bars(n=2500)
    out = contract.generate_signals(bars, {"flat_only": True})
    assert isinstance(out, pd.DataFrame)
    for col in (
        "entry_time", "direction", "entry", "stop", "target",
        "stop_mode", "be_on_tp1", "partial_targets", "partial_fracs",
    ):
        assert col in out.columns


def test_sma21_wick_entry_signal_properties():
    """Every signal: entry==prev_sma, be_on_tp1==True, fracs==[0.5,0.25,0.25], TPs at 1R/2R/4R."""
    from eval_sim.strategies import sma21_wick_entry as sm

    bars = make_bars(n=3000)
    sig_df = sm.generate_signals(bars, {"flat_only": False})
    if sig_df.empty:
        return

    hl2 = (bars["high"] + bars["low"]) / 2.0
    sma = hl2.rolling(21, min_periods=21).mean()

    for _, row in sig_df.iterrows():
        entry_ts = pd.Timestamp(row["entry_time"])
        loc = bars.index.get_loc(entry_ts)
        prev_sma = float(sma.iloc[loc - 1])

        assert abs(row["entry"] - prev_sma) < 1e-6, "entry must equal prev bar SMA HL/2"
        assert bool(row["be_on_tp1"]) is True

        targets = json.loads(row["partial_targets"])
        fracs = json.loads(row["partial_fracs"])
        assert len(targets) == 3
        assert fracs == [0.50, 0.25, 0.25]

        r = abs(row["entry"] - row["stop"])
        assert r > 0
        if row["direction"] == "long":
            assert row["stop"] < row["entry"]
            assert abs(targets[0] - (row["entry"] + 1.0 * r)) < 1e-6
            assert abs(targets[1] - (row["entry"] + 2.0 * r)) < 1e-6
            assert abs(targets[2] - (row["entry"] + 4.0 * r)) < 1e-6
        else:
            assert row["stop"] > row["entry"]
            assert abs(targets[0] - (row["entry"] - 1.0 * r)) < 1e-6
            assert abs(targets[1] - (row["entry"] - 2.0 * r)) < 1e-6
            assert abs(targets[2] - (row["entry"] - 4.0 * r)) < 1e-6


def test_sma21_wick_entry_all_signals_pass_no_touch():
    """Every signal's prior 10 bars had no wick touch of the SMA."""
    from eval_sim.strategies import sma21_wick_entry as sm

    bars = make_bars(n=3000, freq="3min")
    sig_df = sm.generate_signals(bars, {"flat_only": False})
    if sig_df.empty:
        return

    hl2 = (bars["high"] + bars["low"]) / 2.0
    sma = hl2.rolling(21, min_periods=21).mean()

    for _, row in sig_df.iterrows():
        entry_ts = pd.Timestamp(row["entry_time"])
        loc = bars.index.get_loc(entry_ts)
        window = slice(loc - 10, loc)
        sma_w = sma.iloc[window]

        if row["direction"] == "long":
            low_w = bars["low"].iloc[window]
            assert (low_w > sma_w).all(), (
                f"Long signal at {entry_ts}: a prior bar's low touched the SMA"
            )
        else:
            high_w = bars["high"].iloc[window]
            assert (high_w < sma_w).all(), (
                f"Short signal at {entry_ts}: a prior bar's high touched the SMA"
            )
