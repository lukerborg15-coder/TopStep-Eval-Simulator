from __future__ import annotations
from dataclasses import dataclass
from eval_sim.config import TopstepRules, TOPSTEP_50K
from eval_sim.topstep import simulate_seq_evals, simulate_topstep, _group_by_day
from eval_sim.trades import TradeResult


@dataclass(frozen=True)
class ContinuousEvalResult:
    passes: int
    attempts: int
    pass_rate: float               # 0.0–1.0
    worst_attempt_drawdown: float
    mean_days_per_attempt: float
    median_days_per_attempt: float  # median of per-attempt trading day counts
    total_trading_days: int
    is_empty: bool                  # True if no trades — flag in output, don't score


def run_continuous_eval(
    trades: list[TradeResult],
    rules: TopstepRules = TOPSTEP_50K,
) -> ContinuousEvalResult:
    """
    Chain sequential Combine evaluation attempts across all trades.
    Primary output of the pipeline — measures how reliably the strategy
    passes evaluations when run continuously.

    is_empty=True means the strategy generated no signals for this window.
    This should be flagged in output, not treated as a pass_rate=0 score.
    """
    if not trades:
        return ContinuousEvalResult(
            passes=0,
            attempts=0,
            pass_rate=0.0,
            worst_attempt_drawdown=0.0,
            mean_days_per_attempt=0.0,
            median_days_per_attempt=0.0,
            total_trading_days=0,
            is_empty=True,
        )

    seq = simulate_seq_evals(trades, rules)

    # Worst drawdown: re-run each attempt to collect per-attempt drawdown.
    # simulate_seq_evals already consumed the trades cleanly; we replay here
    # only for reporting purposes.
    worst_dd = 0.0
    remaining = list(trades)
    for days_consumed in seq.attempt_days:
        result = simulate_topstep(remaining, rules)
        worst_dd = max(worst_dd, result.max_drawdown_seen)

        by_day = _group_by_day(remaining)
        sorted_days = sorted(by_day.keys())
        if days_consumed >= len(sorted_days):
            break
        next_day = sorted_days[days_consumed]
        remaining = [
            t for t in remaining
            if t.exit_time.tz_convert("US/Eastern").date().isoformat() >= next_day
        ]

    days = list(seq.attempt_days)
    mean_days = sum(days) / len(days) if days else 0.0
    median_days = float(sorted(days)[len(days) // 2]) if days else 0.0
    total_days = sum(days)

    return ContinuousEvalResult(
        passes=seq.passes,
        attempts=seq.attempts,
        pass_rate=seq.pass_rate,
        worst_attempt_drawdown=worst_dd,
        mean_days_per_attempt=mean_days,
        median_days_per_attempt=median_days,
        total_trading_days=total_days,
        is_empty=False,
    )
