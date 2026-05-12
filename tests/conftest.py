import pandas as pd
import numpy as np
import pytest


def make_bars(n: int = 200, freq: str = "5min", start: str = "2020-01-02 09:30") -> pd.DataFrame:
    """Synthetic OHLCV bars with realistic price action."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(start, periods=n, freq=freq, tz="US/Eastern")
    closes = 16000.0 + np.cumsum(rng.normal(0, 2, n))
    highs = closes + rng.uniform(1, 5, n)
    lows = closes - rng.uniform(1, 5, n)
    opens = closes - rng.normal(0, 1, n)
    volume = rng.integers(100, 1000, n).astype(float)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
        index=dates,
    )


def make_trade(net_pnl: float = 100.0, entry_time: str = "2020-01-02 09:35"):
    from eval_sim.trades import TradeResult
    ts = pd.Timestamp(entry_time, tz="US/Eastern")
    return TradeResult(
        entry_time=ts,
        exit_time=ts + pd.Timedelta("5min"),
        direction="long",
        entry=16000.0,
        stop=15990.0,
        target=16020.0,
        exit=16020.0,
        contracts=1,
        gross_pnl=net_pnl + 0.62,
        commission=0.62,
        net_pnl=net_pnl,
        r_multiple=2.0,
        exit_reason="target",
    )


@pytest.fixture
def bars():
    return make_bars()


@pytest.fixture
def winning_trade():
    return make_trade(net_pnl=40.0)


@pytest.fixture
def losing_trade():
    return make_trade(net_pnl=-20.62)
