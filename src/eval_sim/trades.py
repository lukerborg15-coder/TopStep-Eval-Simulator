from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class TradeResult:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str          # "long" | "short"
    entry: float
    stop: float
    target: float
    exit: float
    contracts: int
    gross_pnl: float
    commission: float
    net_pnl: float
    r_multiple: float
    exit_reason: str        # "target" | "stop" | "eod"
