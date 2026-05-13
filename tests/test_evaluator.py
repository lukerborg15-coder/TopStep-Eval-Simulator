import pandas as pd
import pytest
from tests.conftest import make_bars, make_trade
from eval_sim.evaluator import evaluate_window, Window
from eval_sim.config import MNQ
from eval_sim.trades import TradeResult


def _always_long_strategy(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Strategy that signals long on every bar."""
    stop_dist = params.get("stop_dist", 10.0)
    signals = []
    for ts, row in bars.iterrows():
        signals.append({
            "entry_time": ts,
            "direction": "long",
            "entry": row["close"],
            "stop": row["close"] - stop_dist,
            "target": row["close"] + stop_dist * 2,
        })
    return pd.DataFrame(signals)


def _no_signal_strategy(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    return pd.DataFrame(columns=["entry_time", "direction", "entry", "stop", "target"])


def test_evaluate_window_returns_trades(bars):
    window = Window(start=bars.index[0], end=bars.index[-1])
    trades = evaluate_window(
        bars, _always_long_strategy, params={"stop_dist": 10.0},
        window=window, instrument=MNQ, risk_dollars=500.0, max_contracts=5,
    )
    assert isinstance(trades, list)
    assert len(trades) > 0
    assert all(isinstance(t, TradeResult) for t in trades)


def test_evaluate_window_no_signals(bars):
    window = Window(start=bars.index[0], end=bars.index[-1])
    trades = evaluate_window(
        bars, _no_signal_strategy, params={},
        window=window, instrument=MNQ, risk_dollars=500.0, max_contracts=5,
    )
    assert trades == []


def test_evaluate_window_filters_to_window(bars):
    mid = bars.index[len(bars) // 2]
    window = Window(start=mid, end=bars.index[-1])
    trades = evaluate_window(
        bars, _always_long_strategy, params={"stop_dist": 10.0},
        window=window, instrument=MNQ, risk_dollars=500.0, max_contracts=5,
    )
    for t in trades:
        assert t.entry_time >= mid


def test_evaluate_window_warmup_bars_not_scored(bars):
    # warmup_start is earlier than window.start — signals before window.start are discarded
    mid = bars.index[len(bars) // 2]
    early = bars.index[10]
    window = Window(start=mid, end=bars.index[-1])
    trades = evaluate_window(
        bars, _always_long_strategy, params={"stop_dist": 10.0},
        window=window, instrument=MNQ, risk_dollars=500.0, max_contracts=5,
        warmup_start=early,
    )
    # All scored trades must start at or after window.start
    for t in trades:
        assert t.entry_time >= mid


def test_evaluate_window_position_sizing(bars):
    window = Window(start=bars.index[0], end=bars.index[-1])
    # stop_dist=10 ticks * 2.0 point_value * contracts = risk
    # risk_dollars=200, stop_dist=10 -> 10*2 = $20/contract -> 10 contracts capped at 5
    trades = evaluate_window(
        bars, _always_long_strategy, params={"stop_dist": 10.0},
        window=window, instrument=MNQ, risk_dollars=200.0, max_contracts=5,
    )
    assert all(1 <= t.contracts <= 5 for t in trades)


def test_evaluate_window_net_pnl_includes_commission(bars):
    window = Window(start=bars.index[0], end=bars.index[-1])
    trades = evaluate_window(
        bars, _always_long_strategy, params={"stop_dist": 10.0},
        window=window, instrument=MNQ, risk_dollars=500.0, max_contracts=1,
    )
    for t in trades:
        assert abs(t.net_pnl - (t.gross_pnl - t.commission)) < 1e-6
        assert t.commission > 0


def test_evaluate_window_trades_do_not_span_sessions():
    # Bars spanning two calendar days — trades entered on day 1 must exit by day 1.
    multi_day_bars = make_bars(n=400, freq="5min", start="2020-01-02 09:30")
    # 400 bars × 5min = 2000min ≈ 33h, covering Jan 2 and Jan 3
    window = Window(start=multi_day_bars.index[0], end=multi_day_bars.index[-1])
    trades = evaluate_window(
        multi_day_bars, _always_long_strategy, params={"stop_dist": 10.0},
        window=window, instrument=MNQ, risk_dollars=500.0, max_contracts=5,
    )
    for t in trades:
        entry_date = t.entry_time.tz_convert("US/Eastern").date()
        exit_date = t.exit_time.tz_convert("US/Eastern").date()
        assert entry_date == exit_date, (
            f"Trade spanned sessions: entry {t.entry_time}, exit {t.exit_time}"
        )


def _session_17h_strategy(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    ts = pd.Timestamp("2020-01-02 10:00", tz="US/Eastern")
    if ts not in bars.index:
        ts = bars.index[10]
    return pd.DataFrame(
        [
            {
                "entry_time": ts,
                "direction": "long",
                "entry": 100.0,
                "stop": 50.0,
                "target": 200.0,
                "session_end": "17:00",
            }
        ]
    )


def test_session_end_17_00_exits_before_eth():
    idx = pd.date_range("2020-01-02 09:30", periods=100, freq="5min", tz="US/Eastern")
    df = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 500.0,
        },
        index=idx,
    )
    window = Window(start=df.index[0], end=df.index[-1])
    trades = evaluate_window(
        df,
        _session_17h_strategy,
        params={},
        window=window,
        instrument=MNQ,
        risk_dollars=10_000.0,
        max_contracts=1,
    )
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "eod"
    assert t.exit_time <= pd.Timestamp("2020-01-02 16:55", tz="US/Eastern")
    assert t.exit_time >= pd.Timestamp("2020-01-02 10:00", tz="US/Eastern")


def _close_stop_strategy(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    ts = bars.index[1]
    return pd.DataFrame(
        [
            {
                "entry_time": ts,
                "direction": "long",
                "entry": 100.0,
                "stop": 95.0,
                "target": 120.0,
                "stop_mode": "close",
            }
        ]
    )


def test_close_stop_not_triggered_by_wick_only():
    idx = pd.date_range("2020-01-02 10:00", periods=6, freq="5min", tz="US/Eastern")
    df = pd.DataFrame(
        {
            "open": 100.0,
            "high": [100.0, 100.0, 101.0, 121.0, 100.0, 100.0],
            "low": [100.0, 100.0, 92.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 99.0, 100.0, 100.0, 100.0],
            "volume": [500.0] * 6,
        },
        index=idx,
    )
    window = Window(start=df.index[0], end=df.index[-1])
    trades = evaluate_window(
        df,
        _close_stop_strategy,
        params={},
        window=window,
        instrument=MNQ,
        risk_dollars=10_000.0,
        max_contracts=1,
    )
    assert len(trades) == 1
    assert trades[0].exit_reason == "target"


def _partial_ladder_strategy(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    ts = bars.index[0]
    return pd.DataFrame(
        [
            {
                "entry_time": ts,
                "direction": "long",
                "entry": 100.0,
                "stop": 90.0,
                "target": 108.0,
                "stop_mode": "intrabar",
                "partial_targets": [108.0, 118.0],
                "partial_fracs": [0.5, 0.5],
            }
        ]
    )


def test_partial_ladder_two_legs_gross_and_commission():
    idx = pd.date_range("2020-01-02 10:00", periods=4, freq="5min", tz="US/Eastern")
    df = pd.DataFrame(
        {
            "open": 100.0,
            "high": [100.0, 120.0, 120.0, 100.0],
            "low": 99.0,
            "close": [100.0, 115.0, 115.0, 100.0],
            "volume": [500.0] * 4,
        },
        index=idx,
    )
    window = Window(start=df.index[0], end=df.index[-1])
    trades = evaluate_window(
        df,
        _partial_ladder_strategy,
        params={},
        window=window,
        instrument=MNQ,
        risk_dollars=10_000.0,
        max_contracts=10,
    )
    assert len(trades) == 1
    t = trades[0]
    slip = MNQ.slippage_per_side()
    entry_f = 100.0 + slip
    leg1_px = 108.0 - slip
    leg2_px = 118.0 - slip
    n1, n2 = 5, 5
    g1 = (leg1_px - entry_f) * n1 * MNQ.point_value
    g2 = (leg2_px - entry_f) * n2 * MNQ.point_value
    exp_gross = g1 + g2
    exp_comm = MNQ.commission_round_turn * (n1 + n2)
    assert abs(t.gross_pnl - exp_gross) < 1e-6
    assert abs(t.commission - exp_comm) < 1e-6
    assert abs(t.net_pnl - (exp_gross - exp_comm)) < 1e-6
    exp_vwap = (leg1_px * n1 + leg2_px * n2) / (n1 + n2)
    assert abs(t.exit - exp_vwap) < 1e-6


def _partial_close_stop_conflict_strategy(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    ts = bars.index[0]
    return pd.DataFrame(
        [
            {
                "entry_time": ts,
                "direction": "long",
                "entry": 100.0,
                "stop": 99.0,
                "target": 110.0,
                "stop_mode": "close",
                "partial_targets": [110.0],
                "partial_fracs": [1.0],
            }
        ]
    )


def test_partial_ladder_close_stop_wins_before_target_same_bar():
    idx = pd.date_range("2020-01-02 10:00", periods=3, freq="5min", tz="US/Eastern")
    df = pd.DataFrame(
        {
            "open": 100.0,
            "high": [100.0, 115.0, 100.0],
            "low": [100.0, 90.0, 100.0],
            "close": [100.0, 98.0, 100.0],
            "volume": [500.0] * 3,
        },
        index=idx,
    )
    window = Window(start=df.index[0], end=df.index[-1])
    trades = evaluate_window(
        df,
        _partial_close_stop_conflict_strategy,
        params={},
        window=window,
        instrument=MNQ,
        risk_dollars=10_000.0,
        max_contracts=2,
    )
    assert len(trades) == 1
    assert trades[0].exit_reason == "stop"


def _two_quick_signals_strategy(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "entry_time": bars.index[0],
                "direction": "long",
                "entry": 100.0,
                "stop": 99.0,
                "target": 101.0,
            },
            {
                "entry_time": bars.index[1],
                "direction": "long",
                "entry": 100.0,
                "stop": 99.0,
                "target": 101.0,
            },
            {
                "entry_time": bars.index[3],
                "direction": "long",
                "entry": 100.0,
                "stop": 99.0,
                "target": 101.0,
            },
        ]
    )


def test_flat_only_skips_overlapping_entry():
    idx = pd.date_range("2020-01-02 10:00", periods=8, freq="5min", tz="US/Eastern")
    highs = [100.0, 100.0, 102.0, 100.0, 100.0, 102.0, 102.0, 102.0]
    df = pd.DataFrame(
        {
            "open": 100.0,
            "high": highs,
            "low": 99.0,
            "close": [100.0, 100.0, 100.5, 100.5, 100.5, 100.5, 100.5, 100.5],
            "volume": [500.0] * 8,
        },
        index=idx,
    )
    window = Window(start=df.index[0], end=df.index[-1])
    trades_flat = evaluate_window(
        df,
        _two_quick_signals_strategy,
        params={"flat_only": True},
        window=window,
        instrument=MNQ,
        risk_dollars=10_000.0,
        max_contracts=5,
    )
    trades_all = evaluate_window(
        df,
        _two_quick_signals_strategy,
        params={"flat_only": False},
        window=window,
        instrument=MNQ,
        risk_dollars=10_000.0,
        max_contracts=5,
    )
    assert len(trades_all) == 3
    assert len(trades_flat) == 2
