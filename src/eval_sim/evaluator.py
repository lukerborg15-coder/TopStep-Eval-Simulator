from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from eval_sim.config import Instrument
from eval_sim.trades import TradeResult


@dataclass(frozen=True)
class Window:
    start: pd.Timestamp
    end: pd.Timestamp


def _size_position(
    entry: float,
    stop: float,
    instrument: Instrument,
    risk_dollars: float,
    max_contracts: int,
) -> int:
    stop_distance = abs(entry - stop)
    if stop_distance == 0:
        return 0
    value_per_contract = stop_distance * instrument.point_value
    contracts = int(risk_dollars / value_per_contract)
    return max(0, min(contracts, max_contracts))


def _simulate_trade(
    bars: pd.DataFrame,
    entry_time: pd.Timestamp,
    direction: str,
    entry: float,
    stop: float,
    target: float,
    contracts: int,
    instrument: Instrument,
) -> TradeResult:
    slippage = instrument.slippage_per_side()
    if direction == "long":
        actual_entry = entry + slippage
        actual_stop = stop - slippage
        actual_target = target - slippage
    else:
        actual_entry = entry - slippage
        actual_stop = stop + slippage
        actual_target = target + slippage

    # Limit to same calendar session (US/Eastern date) — Topstep requires EOD close.
    # Prevents stop/target being checked against next-day bars for late-day entries.
    entry_local = entry_time.tz_convert("US/Eastern")
    session_end = entry_local.normalize() + pd.Timedelta("1D")
    future_bars = bars[(bars.index > entry_time) & (bars.index < session_end)]
    exit_price = actual_entry
    exit_time = entry_time
    exit_reason = "eod"

    for ts, bar in future_bars.iterrows():
        if direction == "long":
            if bar["low"] <= actual_stop:
                exit_price = actual_stop
                exit_time = ts
                exit_reason = "stop"
                break
            if bar["high"] >= actual_target:
                exit_price = actual_target
                exit_time = ts
                exit_reason = "target"
                break
        else:
            if bar["high"] >= actual_stop:
                exit_price = actual_stop
                exit_time = ts
                exit_reason = "stop"
                break
            if bar["low"] <= actual_target:
                exit_price = actual_target
                exit_time = ts
                exit_reason = "target"
                break
    else:
        if len(future_bars) > 0:
            exit_time = future_bars.index[-1]
            exit_price = future_bars.iloc[-1]["close"]

    gross = (exit_price - actual_entry) * contracts * instrument.point_value
    if direction == "short":
        gross = -gross
    commission = instrument.commission_round_turn * contracts
    net = gross - commission

    r_dist = abs(actual_entry - actual_stop)
    if direction == "long":
        r_mult = (exit_price - actual_entry) / r_dist if r_dist > 0 else 0.0
    else:
        r_mult = (actual_entry - exit_price) / r_dist if r_dist > 0 else 0.0

    return TradeResult(
        entry_time=entry_time,
        exit_time=exit_time,
        direction=direction,
        entry=actual_entry,
        stop=actual_stop,
        target=actual_target,
        exit=exit_price,
        contracts=contracts,
        gross_pnl=gross,
        commission=commission,
        net_pnl=net,
        r_multiple=r_mult,
        exit_reason=exit_reason,
    )


def evaluate_window(
    bars: pd.DataFrame,
    strategy_fn: callable,
    params: dict,
    window: Window,
    instrument: Instrument,
    risk_dollars: float,
    max_contracts: int,
    warmup_start: pd.Timestamp | None = None,
) -> list[TradeResult]:
    """
    Run strategy_fn on bars and simulate each trade within the scoring window.

    warmup_start: if provided, feed bars from warmup_start onward to the strategy
    so indicators can stabilize. Only trades with entry_time >= window.start are
    returned (warmup trades are discarded). If None, context starts at window.start.
    """
    context_start = warmup_start if warmup_start is not None else window.start
    context_bars = bars[(bars.index >= context_start) & (bars.index <= window.end)]
    if context_bars.empty:
        return []

    signals_df = strategy_fn(context_bars, params)
    if signals_df is None or signals_df.empty:
        return []

    required = {"entry_time", "direction", "entry", "stop", "target"}
    if not required.issubset(set(signals_df.columns)):
        raise ValueError(f"Strategy must return columns: {required}")

    # Coerce entry_time to timezone-aware US/Eastern. Strategies may return
    # naive timestamps (e.g. from bars.index.tz_localize(None) or plain strings).
    entry_times = pd.to_datetime(signals_df["entry_time"])
    if entry_times.dt.tz is None:
        signals_df = signals_df.copy()
        signals_df["entry_time"] = entry_times.dt.tz_localize("US/Eastern")
    else:
        signals_df = signals_df.copy()
        signals_df["entry_time"] = entry_times.dt.tz_convert("US/Eastern")

    trades: list[TradeResult] = []
    for _, sig in signals_df.iterrows():
        contracts = _size_position(
            float(sig["entry"]),
            float(sig["stop"]),
            instrument,
            risk_dollars,
            max_contracts,
        )
        if contracts == 0:
            continue
        entry_ts = pd.Timestamp(sig["entry_time"])
        if entry_ts < window.start:
            continue  # warmup signal — discard
        trade = _simulate_trade(
            bars,
            entry_ts,
            str(sig["direction"]),
            float(sig["entry"]),
            float(sig["stop"]),
            float(sig["target"]),
            contracts,
            instrument,
        )
        trades.append(trade)

    return trades
