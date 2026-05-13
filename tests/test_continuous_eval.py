import pandas as pd
import pytest
from tests.conftest import make_trade
from eval_sim.continuous_eval import run_continuous_eval, ContinuousEvalResult
from eval_sim.config import TOPSTEP_50K


def _day_offset(i: int) -> str:
    return (pd.Timestamp("2020-01-02") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")


def _trade(pnl: float, day: str) -> object:
    return make_trade(net_pnl=pnl, entry_time=f"{day} 09:35")


def test_run_continuous_eval_returns_result():
    trades = [_trade(150.0, _day_offset(i)) for i in range(40)]
    result = run_continuous_eval(trades, TOPSTEP_50K)
    assert isinstance(result, ContinuousEvalResult)


def test_run_continuous_eval_empty_trades():
    result = run_continuous_eval([], TOPSTEP_50K)
    assert result.attempts == 0
    assert result.passes == 0
    assert result.pass_rate == 0.0


def test_run_continuous_eval_losing_trades():
    # All losing trades — no passes
    trades = [_trade(-300.0, f"2020-01-{2 + i:02d}") for i in range(20)]
    result = run_continuous_eval(trades, TOPSTEP_50K)
    assert result.passes == 0
    assert result.attempts >= 1


def test_run_continuous_eval_winning_trades():
    # 31 winning trades of $100 = $3100 profit > $3000 target, should pass
    trades = [_trade(100.0, _day_offset(i)) for i in range(31)]
    result = run_continuous_eval(trades, TOPSTEP_50K)
    assert result.passes >= 1
    assert result.pass_rate > 0.0


def test_run_continuous_eval_pass_rate_in_range():
    trades = [_trade(80.0, _day_offset(i)) for i in range(50)]
    result = run_continuous_eval(trades, TOPSTEP_50K)
    assert 0.0 <= result.pass_rate <= 1.0


def test_run_continuous_eval_has_worst_drawdown():
    trades = [_trade(-200.0, f"2020-01-{2 + i:02d}") for i in range(10)]
    result = run_continuous_eval(trades, TOPSTEP_50K)
    assert result.worst_attempt_drawdown >= 0.0
