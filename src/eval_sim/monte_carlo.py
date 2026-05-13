from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from eval_sim.config import TopstepRules, TOPSTEP_50K
from eval_sim.continuous_eval import run_continuous_eval
from eval_sim.topstep import _group_by_day
from eval_sim.trades import TradeResult


@dataclass(frozen=True)
class MCResult:
    n_permutations: int
    pass_rate_median: float
    pass_rate_p05: float
    pass_rate_p95: float
    worst_drawdown_median: float


def _block_bootstrap(
    trades: list[TradeResult],
    block_size: int,
    rng: np.random.Generator,
) -> list[TradeResult]:
    """Resample trade-days in blocks, preserving within-day order."""
    by_day = _group_by_day(trades)
    days = sorted(by_day.keys())
    if not days:
        return []

    n_blocks = max(1, len(days) // block_size)
    sampled_days: list[str] = []
    for _ in range(n_blocks):
        start_idx = rng.integers(0, max(1, len(days) - block_size + 1))
        sampled_days.extend(days[start_idx: start_idx + block_size])

    resampled: list[TradeResult] = []
    for day in sampled_days:
        resampled.extend(by_day.get(day, []))
    return resampled


def run_monte_carlo(
    trades: list[TradeResult],
    rules: TopstepRules = TOPSTEP_50K,
    n: int = 500,
    block_size: int = 5,
    seed: int = 42,
    ci_pct: float = 5.0,
) -> MCResult:
    """
    Block-bootstrap Monte Carlo on holdout trades.
    Resamples trade order N times and reports confidence intervals on pass rate.
    """
    if not trades:
        return MCResult(
            n_permutations=n,
            pass_rate_median=0.0,
            pass_rate_p05=0.0,
            pass_rate_p95=0.0,
            worst_drawdown_median=0.0,
        )

    rng = np.random.default_rng(seed)
    pass_rates: list[float] = []
    worst_drawdowns: list[float] = []

    for _ in range(n):
        resampled = _block_bootstrap(trades, block_size, rng)
        result = run_continuous_eval(resampled, rules)
        pass_rates.append(result.pass_rate)
        worst_drawdowns.append(result.worst_attempt_drawdown)

    return MCResult(
        n_permutations=n,
        pass_rate_median=float(np.median(pass_rates)),
        pass_rate_p05=float(np.percentile(pass_rates, ci_pct)),
        pass_rate_p95=float(np.percentile(pass_rates, 100 - ci_pct)),
        worst_drawdown_median=float(np.median(worst_drawdowns)),
    )
