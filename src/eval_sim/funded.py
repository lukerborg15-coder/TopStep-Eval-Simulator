from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from eval_sim.config import FundedRules, Instrument
from eval_sim.monte_carlo import _block_bootstrap
from eval_sim.sizing_kelly import kelly_risk_dollars
from eval_sim.topstep import _group_by_day
from eval_sim.trades import TradeResult


@dataclass(frozen=True)
class FundedResult:
    busted: bool
    final_balance: float
    peak_balance: float
    max_drawdown_seen: float
    trading_days: int
    days_to_target: int | None   # None if target not reached or busted


@dataclass(frozen=True)
class FundedVariantResult:
    name: str
    ruin_prob: float
    median_days_to_target: float | None
    p10_days_to_target: float | None
    p90_days_to_target: float | None


@dataclass(frozen=True)
class FundedMCResult:
    n_permutations: int
    variants: tuple[FundedVariantResult, ...]


# (name, kelly_fraction_multiplier) — None fraction = fixed-risk mode
VARIANT_SPECS: list[tuple[str, float | None]] = [
    ("kelly_full", 1.0),
    ("kelly_half", 0.5),
    ("kelly_quarter", 0.25),
    ("fixed", None),
]


def _trade_initial_risk(trade: TradeResult, instrument: Instrument) -> float:
    return abs(trade.entry - trade.stop) * trade.contracts * instrument.point_value


def simulate_funded(
    trades: list[TradeResult],
    rules: FundedRules,
    instrument: Instrument,
    *,
    kelly_fraction_mult: float | None = 0.5,
    fallback_risk: float = 500.0,
    target_profit: float = 1500.0,
    min_kelly_sample: int = 50,
) -> FundedResult:
    """
    Simulate one funded-account run under Topstep PA rules.

    Trailing DD: floor trails EOD balance. Once balance reaches dd_freeze_threshold,
    floor freezes permanently at dd_freeze_floor (trail-then-freeze mechanic).

    kelly_fraction_mult=None uses fallback_risk as a fixed per-trade risk dollar amount.
    kelly_fraction_mult=float uses fractional Kelly, falling back to fallback_risk
    until min_kelly_sample trades have been seen.

    PnL is re-scaled from the holdout trade's original risk to the Kelly-sized risk,
    so contract count changes are implicitly modelled without re-running the engine.
    """
    if not trades:
        return FundedResult(
            busted=False, final_balance=rules.account_size,
            peak_balance=rules.account_size, max_drawdown_seen=0.0,
            trading_days=0, days_to_target=None,
        )

    balance = rules.account_size
    floor = rules.account_size - rules.trailing_dd
    freeze_triggered = False
    peak_balance = rules.account_size
    max_drawdown_seen = 0.0
    trading_days = 0
    target_balance = rules.account_size + target_profit
    days_to_target: int | None = None
    seen_trades: list[TradeResult] = []

    by_day = _group_by_day(trades)

    for day, day_trades in sorted(by_day.items()):
        trading_days += 1
        running_day_pnl = 0.0
        daily_loss_buffer = rules.daily_loss_limit

        for trade in day_trades:
            if running_day_pnl <= -rules.daily_loss_limit:
                break

            original_risk = _trade_initial_risk(trade, instrument)
            if original_risk <= 0:
                seen_trades.append(trade)
                continue

            if kelly_fraction_mult is None:
                risk_used = min(fallback_risk, max(0.0, daily_loss_buffer))
            else:
                risk_used = kelly_risk_dollars(
                    seen_trades, balance, daily_loss_buffer,
                    fraction=kelly_fraction_mult,
                    min_sample=min_kelly_sample,
                    fallback_risk=fallback_risk,
                )

            scale = risk_used / original_risk
            scaled_pnl = trade.net_pnl * scale

            balance += scaled_pnl
            running_day_pnl += scaled_pnl
            daily_loss_buffer = max(0.0, daily_loss_buffer - max(0.0, -scaled_pnl))
            peak_balance = max(peak_balance, balance)
            max_drawdown_seen = max(max_drawdown_seen, peak_balance - balance)
            seen_trades.append(trade)

            if balance <= floor:
                return FundedResult(
                    busted=True, final_balance=balance,
                    peak_balance=peak_balance, max_drawdown_seen=max_drawdown_seen,
                    trading_days=trading_days, days_to_target=None,
                )

            if days_to_target is None and balance >= target_balance:
                days_to_target = trading_days

        # EOD trailing DD update — Topstep trail-then-freeze mechanic
        if not freeze_triggered:
            floor = max(floor, balance - rules.trailing_dd)
            if balance >= rules.dd_freeze_threshold:
                freeze_triggered = True
                floor = rules.dd_freeze_floor

    return FundedResult(
        busted=False, final_balance=balance,
        peak_balance=peak_balance, max_drawdown_seen=max_drawdown_seen,
        trading_days=trading_days, days_to_target=days_to_target,
    )


def run_funded_mc(
    holdout_trades: list[TradeResult],
    rules: FundedRules,
    instrument: Instrument,
    *,
    variant_names: list[str] | None = None,
    fallback_risk: float = 500.0,
    target_profit: float = 1500.0,
    n: int = 1000,
    block_size: int = 5,
    seed: int = 42,
) -> FundedMCResult:
    """
    Block-bootstrap Monte Carlo over holdout trades for funded account simulation.
    Runs each Kelly variant in parallel across all bootstrap draws.
    variant_names=None runs all variants.
    """
    specs = [
        (name, frac) for name, frac in VARIANT_SPECS
        if variant_names is None or name in variant_names
    ]

    if not holdout_trades:
        return FundedMCResult(
            n_permutations=n,
            variants=tuple(
                FundedVariantResult(
                    name=name, ruin_prob=0.0,
                    median_days_to_target=None,
                    p10_days_to_target=None,
                    p90_days_to_target=None,
                )
                for name, _ in specs
            ),
        )

    rng = np.random.default_rng(seed)
    per_variant_busts: dict[str, list[bool]] = {name: [] for name, _ in specs}
    per_variant_days: dict[str, list[float]] = {name: [] for name, _ in specs}

    for _ in range(n):
        resampled = _block_bootstrap(holdout_trades, block_size, rng)
        for name, frac in specs:
            result = simulate_funded(
                resampled, rules, instrument,
                kelly_fraction_mult=frac,
                fallback_risk=fallback_risk,
                target_profit=target_profit,
            )
            per_variant_busts[name].append(result.busted)
            if not result.busted and result.days_to_target is not None:
                per_variant_days[name].append(float(result.days_to_target))

    variant_results: list[FundedVariantResult] = []
    for name, _ in specs:
        busts = per_variant_busts[name]
        ruin_prob = sum(busts) / len(busts) if busts else 0.0
        days = per_variant_days[name]
        variant_results.append(FundedVariantResult(
            name=name,
            ruin_prob=ruin_prob,
            median_days_to_target=float(np.median(days)) if days else None,
            p10_days_to_target=float(np.percentile(days, 10)) if days else None,
            p90_days_to_target=float(np.percentile(days, 90)) if days else None,
        ))

    return FundedMCResult(n_permutations=n, variants=tuple(variant_results))
