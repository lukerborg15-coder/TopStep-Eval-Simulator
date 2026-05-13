import pandas as pd
import pytest
from tests.conftest import make_trade
from eval_sim.topstep import simulate_topstep, simulate_seq_evals, TopstepResult
from eval_sim.config import TOPSTEP_50K, TopstepRules


def _day_offset(i: int) -> str:
    return (pd.Timestamp("2020-01-02") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")


def _trade(pnl: float, day: str = "2020-01-02") -> object:
    return make_trade(net_pnl=pnl, entry_time=f"{day} 09:35")


def test_simulate_topstep_pass_consistency_satisfied():
    # Spread profit across multiple days so best day is well under 50%
    # 31 days × $100 = $3,100. Best day = $100, total = $3,100 → 3.2% < 50%
    trades = [_trade(100.0, _day_offset(i)) for i in range(31)]
    result = simulate_topstep(trades, TOPSTEP_50K)
    assert result.passed is True


def test_simulate_topstep_consistency_blocks_pass():
    # Day 1: $3,100 profit (all profit in one day = 100% > 50% → blocked)
    # Day 2: $100 more → total $3,200, best day still $3,100 → 96.9% → still blocked
    # With only 2 days of trades, consistency can never be satisfied here
    trades = [_trade(3100.0, "2020-01-02"), _trade(100.0, "2020-01-03")]
    result = simulate_topstep(trades, TOPSTEP_50K)
    assert result.passed is False
    assert result.fail_reason == "end_of_trades"  # blocked by consistency, not breached


def test_simulate_topstep_consistency_eventually_satisfied():
    # Day 1: $1,800 profit. Day 2: $1,800 more = $3,600 total.
    # best_day=$1,800 / total=$3,600 = 50% → exactly at limit, passes
    trades = [_trade(1800.0, "2020-01-02"), _trade(1800.0, "2020-01-03")]
    result = simulate_topstep(trades, TOPSTEP_50K)
    assert result.passed is True


def test_simulate_topstep_drawdown_breach():
    trades = [_trade(-2500.0, "2020-01-02")]
    result = simulate_topstep(trades, TOPSTEP_50K)
    assert result.passed is False
    assert result.fail_reason == "max_drawdown"


def test_simulate_topstep_eod_trailing_floor_raises():
    # Day 1: +$1,500 → EOD balance $51,500 → floor becomes $49,500
    # Day 2: -$900 → balance $50,600 (stays under $1,000 daily-loss lockout) → EOD floor stays $49,500
    # Day 3: -$1,200 single trade → balance $49,400 ≤ $49,500 floor → breach (floor check fires before daily-loss break)
    trades = [
        _trade(1500.0, "2020-01-02"),   # EOD floor now $49,500
        _trade(-900.0, "2020-01-03"),   # balance $50,600, no lockout
        _trade(-1200.0, "2020-01-06"),  # balance $49,400 ≤ $49,500 → breach
    ]
    result = simulate_topstep(trades, TOPSTEP_50K)
    assert result.passed is False
    assert result.fail_reason == "max_drawdown"


def test_simulate_topstep_reports_drawdown_consumed_from_active_trailing_floor():
    # Day 1 EOD raises the trailing floor to $49,500, implying active high-water
    # protection at $51,500. Day 2 ends near that floor, so consumed trailing
    # drawdown is $1,900 even though the account is only $400 below start.
    trades = [
        _trade(1500.0, "2020-01-02"),
        _trade(-1900.0, "2020-01-03"),
    ]
    result = simulate_topstep(trades, TOPSTEP_50K)
    assert result.fail_reason != "max_drawdown"
    assert result.max_drawdown_seen == 1900.0


def test_simulate_topstep_eod_floor_does_not_trail_intraday():
    # Intraday spike does NOT raise the floor — only EOD balance does.
    # Day 1 closes at $50,000 (started $50,000, made and lost $1,500 intraday)
    # Floor stays at $48,000 (intraday high doesn't count)
    # Day 2: -$1,900 → balance $48,100 → above $48,000 → no breach
    trades = [
        _trade(1500.0, "2020-01-02"),   # intraday +1500
        _trade(-1500.0, "2020-01-02"),  # intraday -1500; EOD balance = $50,000 → floor stays $48,000
        _trade(-1900.0, "2020-01-03"),  # balance $48,100 → above $48,000 floor → no breach
    ]
    result = simulate_topstep(trades, TOPSTEP_50K)
    assert result.fail_reason != "max_drawdown"


def test_simulate_topstep_intraday_pass_detected():
    # Profit crosses target mid-day (two trades on day 2 each +$750).
    # After 4th trade: balance=$53,000. Consistency: best_day=$1,500 / total=$3,000 = 50% → passes.
    # The 5th trade (-$5,000) must never execute — pass was already returned.
    trades = [
        _trade(750.0, "2020-01-02"),
        _trade(750.0, "2020-01-02"),
        _trade(750.0, "2020-01-03"),
        _trade(750.0, "2020-01-03"),  # crosses target, consistency satisfied → pass
        _trade(-5000.0, "2020-01-03"),  # must not execute — already passed
    ]
    result = simulate_topstep(trades, TOPSTEP_50K)
    assert result.passed is True
    assert result.trading_days == 2
    assert result.final_balance == 53_000.0


def test_simulate_topstep_daily_loss_locks_out_day_not_attempt():
    # After -$1,100 on day 1 (exceeds $1,000 limit), day is locked out.
    # Day 2 still proceeds — attempt is not failed.
    # Trades: day 1 -$600, -$600 (second trade breaches limit → lockout);
    #         day 2 +$100 (still allowed).
    trades = [
        _trade(-600.0, "2020-01-02"),
        _trade(-600.0, "2020-01-02"),  # cumulative -$1,200 → lockout; second -$600 still applied
        _trade(100.0, "2020-01-03"),
    ]
    result = simulate_topstep(trades, TOPSTEP_50K)
    assert result.fail_reason != "daily_loss"  # daily loss never fails the attempt
    assert result.trading_days == 2             # both days processed


def test_simulate_topstep_daily_loss_does_not_fail_attempt():
    # Exceed daily loss on day 1, then recover over many subsequent days.
    # The attempt should complete normally (end_of_trades or max_days) not "daily_loss".
    trades = (
        [_trade(-1100.0, "2020-01-02")]  # day 1 locked out after this trade
        + [_trade(100.0, f"2020-01-{3 + i:02d}") for i in range(10)]
    )
    result = simulate_topstep(trades, TOPSTEP_50K)
    assert result.fail_reason in ("end_of_trades", "max_days", None)
    assert result.fail_reason != "daily_loss"


def test_simulate_topstep_no_trades():
    result = simulate_topstep([], TOPSTEP_50K)
    assert result.passed is False
    assert result.fail_reason == "no_trades"
    assert result.trading_days == 0


def test_simulate_seq_evals_returns_attempt_days():
    trades = [_trade(100.0, _day_offset(i)) for i in range(60)]
    result = simulate_seq_evals(trades, TOPSTEP_50K)
    assert isinstance(result.attempt_days, tuple)
    assert len(result.attempt_days) == result.attempts
    assert all(d > 0 for d in result.attempt_days)


def test_simulate_seq_evals_zero_trades():
    result = simulate_seq_evals([], TOPSTEP_50K)
    assert result.attempts == 0
    assert result.passes == 0
    assert result.pass_rate == 0.0
    assert result.attempt_days == ()
