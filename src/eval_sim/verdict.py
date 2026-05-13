from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class VerdictThresholds:
    reject_pass_rate: float = 0.30      # reject if pass_rate below this
    reject_max_dd: float = 1800.0       # reject if worst drawdown above this
    ready_pass_rate: float = 0.60       # combine-ready if pass_rate at or above this
    ready_max_dd: float = 1200.0        # combine-ready if worst drawdown at or below this


DEFAULT_THRESHOLDS = VerdictThresholds()


@dataclass(frozen=True)
class VerdictResult:
    verdict: str                        # "COMBINE-READY" | "MARGINAL" | "REJECT"
    reject_reasons: tuple[str, ...]
    warn_reasons: tuple[str, ...]


def compute_verdict(
    pass_rate: float,
    mc_pass_rate_p05: float,
    worst_drawdown: float,
    sensitivity_is_cliff: bool,
    thresholds: VerdictThresholds = DEFAULT_THRESHOLDS,
) -> VerdictResult:
    reject_reasons: list[str] = []
    warn_reasons: list[str] = []

    if pass_rate < thresholds.reject_pass_rate:
        reject_reasons.append(f"pass_rate {pass_rate:.1%} below reject floor {thresholds.reject_pass_rate:.1%}")
    if worst_drawdown > thresholds.reject_max_dd:
        reject_reasons.append(f"drawdown ${worst_drawdown:.0f} exceeds reject limit ${thresholds.reject_max_dd:.0f}")

    if sensitivity_is_cliff:
        warn_reasons.append("sensitivity cliff detected — params are on edge of performance region")

    if reject_reasons:
        verdict = "REJECT"
    elif pass_rate >= thresholds.ready_pass_rate and worst_drawdown <= thresholds.ready_max_dd:
        verdict = "COMBINE-READY"
    else:
        verdict = "MARGINAL"

    return VerdictResult(
        verdict=verdict,
        reject_reasons=tuple(reject_reasons),
        warn_reasons=tuple(warn_reasons),
    )
