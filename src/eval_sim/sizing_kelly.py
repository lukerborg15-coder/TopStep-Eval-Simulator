from __future__ import annotations
from eval_sim.trades import TradeResult


def kelly_fraction(trades: list[TradeResult], min_sample: int = 50) -> float | None:
    """
    Compute full Kelly fraction f* from trade history.
    Returns None if sample is too small to trust.
    Clips result to [0.0, 0.25] to prevent over-sizing.
    """
    if len(trades) < min_sample:
        return None
    winners = [t.r_multiple for t in trades if t.r_multiple > 0]
    losers = [t.r_multiple for t in trades if t.r_multiple < 0]
    if not winners:
        return 0.0
    W = len(winners) / len(trades)
    mean_win = sum(winners) / len(winners)
    if losers:
        mean_loss = abs(sum(losers) / len(losers))
        payoff = mean_win / mean_loss if mean_loss > 0 else 10.0
    else:
        payoff = 10.0  # no losses recorded — cap payoff conservatively
    f_star = W - (1.0 - W) / payoff
    return max(0.0, min(0.25, f_star))


def kelly_risk_dollars(
    trades: list[TradeResult],
    current_equity: float,
    daily_loss_buffer: float,
    fraction: float = 0.5,
    min_sample: int = 50,
    fallback_risk: float = 500.0,
) -> float:
    """
    Return risk_dollars for the next trade under fractional Kelly sizing.
    Falls back to fallback_risk when sample is below min_sample.
    Always capped by daily_loss_buffer.
    """
    f = kelly_fraction(trades, min_sample)
    if f is None:
        risk = fallback_risk
    else:
        risk = fraction * f * current_equity
    return min(risk, max(0.0, daily_loss_buffer))
