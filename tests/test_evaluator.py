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
