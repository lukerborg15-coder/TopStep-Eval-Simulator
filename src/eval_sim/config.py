from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    symbol: str
    point_value: float
    tick_size: float
    commission_round_turn: float
    slippage_ticks_per_side: int = 1

    def slippage_per_side(self) -> float:
        return self.slippage_ticks_per_side * self.tick_size * self.point_value


@dataclass(frozen=True)
class TopstepRules:
    account_size: float = 50_000.0
    profit_target: float = 3_000.0
    max_drawdown: float = 2_000.0        # trailing from EOD balance, not intraday peak
    daily_loss_limit: float = 1_000.0   # locks out trading for the rest of the day; does NOT fail the attempt
    consistency_pct: float = 0.50       # best single profitable day / total profit must be <= 50% at pass
    max_trading_days: int = 60


@dataclass(frozen=True)
class DataSplitConfig:
    wf_years: float = 3.5
    holdout_months: int = 18
    n_wf_windows: int = 4     # equal OOS scoring windows across the WF period
    warmup_bars: int = 200    # default bars prepended for warmup; strategies override via WARMUP_BARS


@dataclass(frozen=True)
class SearchConfig:
    n_candidates: int = 400
    seed: int = 42


MNQ = Instrument(
    symbol="MNQ",
    point_value=2.0,
    tick_size=0.25,
    commission_round_turn=0.62,
    slippage_ticks_per_side=1,
)

MES = Instrument(
    symbol="MES",
    point_value=5.0,
    tick_size=0.25,
    commission_round_turn=0.85,
    slippage_ticks_per_side=1,
)

TOPSTEP_50K = TopstepRules()
DATA_SPLIT = DataSplitConfig()
SEARCH = SearchConfig()

INSTRUMENTS: dict[str, Instrument] = {"mnq": MNQ, "mes": MES}
