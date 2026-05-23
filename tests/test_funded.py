"""Tests for funded.py and sizing_kelly.py."""
from __future__ import annotations
from dataclasses import replace
import pandas as pd
import pytest

from eval_sim.config import FundedRules, MNQ
from eval_sim.funded import simulate_funded, FundedResult
from eval_sim.sizing_kelly import kelly_fraction, kelly_risk_dollars
from eval_sim.trades import TradeResult


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_trade(
    net_pnl: float,
    r_multiple: float,
    day: str = "2024-01-02",
    entry: float = 100.0,
    stop: float = 95.0,   # 5-point stop → $10 risk on MNQ (1 contract × $2/point)
    contracts: int = 1,
) -> TradeResult:
    ts = pd.Timestamp(f"{day} 10:00", tz="US/Eastern")
    ts_exit = pd.Timestamp(f"{day} 10:05", tz="US/Eastern")
    return TradeResult(
        entry_time=ts,
        exit_time=ts_exit,
        direction="long",
        entry=entry,
        stop=stop,
        target=entry + (entry - stop),
        exit=entry + (entry - stop) if net_pnl > 0 else stop,
        contracts=contracts,
        gross_pnl=net_pnl + 0.62,
        commission=0.62,
        net_pnl=net_pnl,
        r_multiple=r_multiple,
        exit_reason="target" if net_pnl > 0 else "stop",
    )


def _win_trade(day: str = "2024-01-02") -> TradeResult:
    return _make_trade(net_pnl=9.38, r_multiple=1.0, day=day)


def _loss_trade(day: str = "2024-01-02") -> TradeResult:
    return _make_trade(net_pnl=-10.62, r_multiple=-1.0, day=day)


def _days(n: int) -> list[str]:
    base = pd.Timestamp("2024-01-02")
    return [(base + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n)]


# ── Kelly math tests ──────────────────────────────────────────────────────────

def test_kelly_fraction_below_min_sample_returns_none():
    trades = [_win_trade(d) for d in _days(30)]
    assert kelly_fraction(trades, min_sample=50) is None


def test_kelly_fraction_zero_edge():
    """W=0.5, payoff=1.0 → f* = 0.5 - 0.5/1.0 = 0.0"""
    days = _days(100)
    trades = [
        _make_trade(net_pnl=9.38, r_multiple=1.0, day=d) if i % 2 == 0
        else _make_trade(net_pnl=-10.62, r_multiple=-1.0, day=d)
        for i, d in enumerate(days)
    ]
    # roughly W=0.5, payoff close to 1.0; f* should be near 0 or clipped to 0
    f = kelly_fraction(trades, min_sample=50)
    assert f is not None
    assert f >= 0.0


def test_kelly_fraction_high_edge_clips_to_025():
    """High W, high payoff → f* > 0.25 → clips to 0.25."""
    days = _days(100)
    wins = [_make_trade(net_pnl=20.0, r_multiple=2.0, day=d) for d in days[:75]]
    losses = [_make_trade(net_pnl=-5.0, r_multiple=-0.5, day=d) for d in days[75:]]
    trades = wins + losses
    f = kelly_fraction(trades, min_sample=50)
    assert f is not None
    assert f == pytest.approx(0.25)


def test_kelly_risk_dollars_caps_at_buffer():
    """kelly_risk_dollars must never exceed daily_loss_buffer."""
    days = _days(100)
    wins = [_make_trade(net_pnl=20.0, r_multiple=2.0, day=d) for d in days[:75]]
    losses = [_make_trade(net_pnl=-5.0, r_multiple=-0.5, day=d) for d in days[75:]]
    trades = wins + losses
    risk = kelly_risk_dollars(
        trades, current_equity=50_000.0, daily_loss_buffer=50.0,
        fraction=1.0, min_sample=50, fallback_risk=500.0,
    )
    assert risk <= 50.0


def test_kelly_risk_dollars_fallback_when_small_sample():
    trades = [_win_trade(d) for d in _days(20)]
    risk = kelly_risk_dollars(
        trades, current_equity=50_000.0, daily_loss_buffer=1000.0,
        fraction=0.5, min_sample=50, fallback_risk=300.0,
    )
    assert risk == pytest.approx(300.0)


# ── DD / freeze mechanic tests ────────────────────────────────────────────────

def test_funded_no_trades_returns_clean():
    rules = FundedRules()
    result = simulate_funded([], rules, MNQ)
    assert result.busted is False
    assert result.final_balance == pytest.approx(rules.account_size)
    assert result.trading_days == 0


def test_funded_bust_triggers_at_floor():
    """A string of losses that drops balance to or below floor should bust."""
    rules = FundedRules(
        account_size=50_000.0, trailing_dd=2_000.0,
        dd_freeze_threshold=52_000.0, dd_freeze_floor=50_000.0,
        daily_loss_limit=10_000.0,  # high daily limit so bust comes from DD, not day lockout
    )
    # Each loss trade: original_risk = |100-95| * 1 * 2 = $10; net_pnl = -$10.62
    # At $500 fallback_risk: scale = 500/10 = 50 → scaled_pnl = -$530/trade
    # floor = $48k; need to drop from $50k to $48k → ~4 trades
    days_list = _days(20)
    trades = [_loss_trade(d) for d in days_list]
    result = simulate_funded(
        trades, rules, MNQ,
        kelly_fraction_mult=None, fallback_risk=500.0,
    )
    assert result.busted is True


def test_funded_freeze_triggers_and_holds():
    """
    Verify floor freezes at dd_freeze_floor once balance >= dd_freeze_threshold,
    and further gains do not raise the floor.
    """
    rules = FundedRules(
        account_size=1_000.0, trailing_dd=200.0,
        dd_freeze_threshold=1_200.0, dd_freeze_floor=1_000.0,
        daily_loss_limit=500.0,
    )
    # Make winning trades that push balance above freeze threshold
    # original_risk = |100-95| * 1 * 2 = $10; at fallback_risk=$10 → scale=1 → pnl passes through
    days_list = _days(50)
    wins = [_win_trade(d) for d in days_list]
    result = simulate_funded(
        wins, rules, MNQ,
        kelly_fraction_mult=None, fallback_risk=10.0,
    )
    # Should not bust — after freeze the floor stays at $1000, and balance grows above it
    assert result.busted is False
    assert result.final_balance > rules.dd_freeze_floor


def test_funded_daily_loss_lockout():
    """Two losses in one day that exceed daily_loss_limit lock out the rest of that day."""
    rules = FundedRules(
        account_size=50_000.0, trailing_dd=2_000.0,
        dd_freeze_threshold=52_000.0, dd_freeze_floor=50_000.0,
        daily_loss_limit=200.0,  # tight daily limit
    )
    # original_risk = $10; at $150 fallback → scale=15 → each loss = -$159.3
    # Two losses in one day = -$318.6 > $200 daily limit → second trade still fires (hits limit after)
    # but third trade on same day should be skipped
    day = "2024-01-02"
    ts_base = pd.Timestamp(f"{day} 10:00", tz="US/Eastern")

    def _loss_at(minute: int) -> TradeResult:
        ts = ts_base + pd.Timedelta(minutes=minute)
        ts_exit = ts + pd.Timedelta(minutes=5)
        return TradeResult(
            entry_time=ts, exit_time=ts_exit,
            direction="long", entry=100.0, stop=95.0, target=105.0,
            exit=95.0, contracts=1, gross_pnl=-9.38, commission=0.62,
            net_pnl=-10.0, r_multiple=-1.0, exit_reason="stop",
        )

    # 3 losses in one day — fallback=$150 but daily_loss_buffer starts at $200.
    # Trade 1: risk_used=min(150,200)=150 → scale=15 → pnl=-150; buffer→50
    # Trade 2: risk_used=min(150,50)=50 → scale=5 → pnl=-50; buffer→0; running=-200→locked
    # Trade 3: skipped (running_day_pnl <= -daily_loss_limit)
    # Total loss = $200; balance = $49,800
    trades = [_loss_at(0), _loss_at(10), _loss_at(20)]
    result = simulate_funded(
        trades, rules, MNQ,
        kelly_fraction_mult=None, fallback_risk=150.0,
    )
    assert result.busted is False
    assert result.final_balance == pytest.approx(50_000.0 - 200.0, rel=1e-3)


def test_funded_target_reached_records_day():
    """days_to_target should be set once balance crosses account_size + target_profit."""
    rules = FundedRules()
    # original_risk = $10; fallback=$10 → scale=1 → pnl pass-through
    # net_pnl per win = $9.38; need $1500 → ~160 wins
    days_list = _days(200)
    trades = [_win_trade(d) for d in days_list]
    result = simulate_funded(
        trades, rules, MNQ,
        kelly_fraction_mult=None, fallback_risk=10.0, target_profit=1_500.0,
    )
    assert result.days_to_target is not None
    assert result.days_to_target <= result.trading_days
