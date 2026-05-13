from __future__ import annotations
from dataclasses import dataclass
import statistics
import pandas as pd
from eval_sim.config import TopstepRules
from eval_sim.trades import TradeResult


@dataclass(frozen=True)
class TopstepResult:
    passed: bool
    fail_reason: str | None  # "max_drawdown"|"max_days"|"end_of_trades"|"no_trades"|None
    final_balance: float
    max_drawdown_seen: float
    trading_days: int


@dataclass(frozen=True)
class SeqEvalResult:
    passes: int
    attempts: int
    pass_rate: float
    attempt_days: tuple[int, ...]   # trading days consumed per attempt — used for median calculation
    worst_drawdown: float = 0.0     # max drawdown seen across all attempts


def _group_by_day(trades: list[TradeResult]) -> dict[str, list[TradeResult]]:
    """Group trades by exit day as YYYY-MM-DD string (US/Eastern). No calendar gaps."""
    groups: dict[str, list[TradeResult]] = {}
    for t in trades:
        day = t.exit_time.tz_convert("US/Eastern").date().isoformat()
        groups.setdefault(day, []).append(t)
    return groups


def simulate_topstep(
    trades: list[TradeResult],
    rules: TopstepRules,
) -> TopstepResult:
    """
    Simulate one Combine evaluation attempt with correct Topstep 50K rules:

    DRAWDOWN — trails from end-of-day closing balance (updates at 16:00 Chicago /
    US/Eastern close, NOT intraday). floor = max(previous floors, eod_balance - max_drawdown).
    Intraday balance is checked against the current floor after every trade.

    CONSISTENCY — when balance crosses profit_target, check:
        best_profitable_day_pnl / total_profit <= consistency_pct (50%)
    Only profitable days count for the numerator. If consistency fails, trading
    continues — the attempt is not failed, just not yet passed. Naturally requires
    at least 2 trading days (can't satisfy 50% with a single day of profit).

    DAILY LOSS — if cumulative net PnL for the day reaches -daily_loss_limit at any
    point intraday, remaining trades for that day are skipped (account locked out).
    The attempt is NOT failed — trading resumes the next session.

    NO MINIMUM DAYS — but consistency rule naturally enforces >= 2 profitable days.
    """
    if not trades:
        return TopstepResult(
            passed=False, fail_reason="no_trades",
            final_balance=rules.account_size,
            max_drawdown_seen=0.0, trading_days=0,
        )

    balance = rules.account_size
    floor = rules.account_size - rules.max_drawdown   # starts static, trails EOD
    target_balance = rules.account_size + rules.profit_target
    max_drawdown_seen = 0.0
    trading_days = 0
    daily_pnls: dict[str, float] = {}  # day → net PnL for that day

    by_day = _group_by_day(trades)

    for day, day_trades in sorted(by_day.items()):
        trading_days += 1
        if trading_days > rules.max_trading_days:
            return TopstepResult(
                passed=False, fail_reason="max_days",
                final_balance=balance,
                max_drawdown_seen=max_drawdown_seen,
                trading_days=trading_days,
            )

        day_start_balance = balance
        running_day_pnl = 0.0
        for trade in day_trades:
            balance += trade.net_pnl
            running_day_pnl += trade.net_pnl
            active_high_water = floor + rules.max_drawdown
            trailing_dd_consumed = max(0.0, active_high_water - balance)
            max_drawdown_seen = max(max_drawdown_seen, trailing_dd_consumed)

            # Intraday drawdown check against current floor
            if balance <= floor:
                return TopstepResult(
                    passed=False, fail_reason="max_drawdown",
                    final_balance=balance,
                    max_drawdown_seen=max_drawdown_seen,
                    trading_days=trading_days,
                )

            # Intraday pass check — Topstep evaluates continuously, not just at EOD.
            # Include the current day's running PnL (not yet in daily_pnls) for consistency.
            if balance >= target_balance:
                temp_pnls = {**daily_pnls, day: running_day_pnl}
                total_profit = balance - rules.account_size
                profitable_days = [p for p in temp_pnls.values() if p > 0]
                best_day = max(profitable_days, default=0.0)
                if total_profit > 0 and best_day / total_profit <= rules.consistency_pct:
                    return TopstepResult(
                        passed=True, fail_reason=None,
                        final_balance=balance,
                        max_drawdown_seen=max_drawdown_seen,
                        trading_days=trading_days,
                    )
                # else: consistency not yet satisfied — keep trading

            # Daily loss limit: lock out remaining trades for today.
            # The attempt is NOT failed — trading resumes next session.
            if running_day_pnl <= -rules.daily_loss_limit:
                break

        # End-of-day processing
        daily_pnls[day] = running_day_pnl

        # EOD trailing drawdown: floor advances to eod_balance - max_drawdown
        # but never goes below the starting floor
        floor = max(floor, balance - rules.max_drawdown)

    return TopstepResult(
        passed=False, fail_reason="end_of_trades",
        final_balance=balance,
        max_drawdown_seen=max_drawdown_seen,
        trading_days=trading_days,
    )


def simulate_seq_evals(
    trades: list[TradeResult],
    rules: TopstepRules,
) -> SeqEvalResult:
    """
    Chain sequential Combine evaluation attempts across a trade list.
    After each attempt (pass or fail), resume from the first trading day
    not consumed by that attempt.

    Uses YYYY-MM-DD date strings throughout — _group_by_day only includes
    days with actual trades, so weekends and holidays create no gaps.
    attempt_days records trading_days consumed per attempt for median reporting.
    """
    if not trades:
        return SeqEvalResult(passes=0, attempts=0, pass_rate=0.0, attempt_days=())

    remaining = list(trades)
    passes = 0
    attempts = 0
    attempt_days: list[int] = []
    worst_drawdown = 0.0
    MAX_ATTEMPTS = 10_000  # safety valve against infinite loops from buggy strategies

    while remaining and attempts < MAX_ATTEMPTS:
        result = simulate_topstep(remaining, rules)

        if result.trading_days == 0:
            break  # no progress — stop

        attempts += 1
        attempt_days.append(result.trading_days)
        worst_drawdown = max(worst_drawdown, result.max_drawdown_seen)
        if result.passed:
            passes += 1

        by_day = _group_by_day(remaining)
        sorted_days = sorted(by_day.keys())

        if result.trading_days >= len(sorted_days):
            break  # all trading days consumed

        next_day = sorted_days[result.trading_days]
        remaining = [
            t for t in remaining
            if t.exit_time.tz_convert("US/Eastern").date().isoformat() >= next_day
        ]

    pass_rate = passes / attempts if attempts > 0 else 0.0
    return SeqEvalResult(
        passes=passes,
        attempts=attempts,
        pass_rate=pass_rate,
        attempt_days=tuple(attempt_days),
        worst_drawdown=worst_drawdown,
    )
