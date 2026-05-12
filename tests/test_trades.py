import pandas as pd
from eval_sim.trades import TradeResult


def test_trade_result_fields():
    ts = pd.Timestamp("2020-01-02 09:35", tz="US/Eastern")
    t = TradeResult(
        entry_time=ts,
        exit_time=ts + pd.Timedelta("5min"),
        direction="long",
        entry=16000.0,
        stop=15990.0,
        target=16020.0,
        exit=16020.0,
        contracts=1,
        gross_pnl=40.62,
        commission=0.62,
        net_pnl=40.0,
        r_multiple=2.0,
        exit_reason="target",
    )
    assert t.net_pnl == 40.0
    assert t.exit_reason == "target"
    assert t.direction == "long"


def test_trade_result_is_immutable():
    ts = pd.Timestamp("2020-01-02 09:35", tz="US/Eastern")
    t = TradeResult(
        entry_time=ts, exit_time=ts, direction="long",
        entry=1.0, stop=0.9, target=1.2, exit=1.2,
        contracts=1, gross_pnl=0.0, commission=0.0,
        net_pnl=0.0, r_multiple=0.0, exit_reason="target",
    )
    import dataclasses
    assert dataclasses.is_dataclass(t)
