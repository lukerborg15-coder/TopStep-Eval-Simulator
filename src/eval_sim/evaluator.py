from __future__ import annotations
from dataclasses import dataclass
import json
from typing import Any

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


def _session_cutoff_exclusive(entry_time: pd.Timestamp, session_end: Any) -> pd.Timestamp:
    """
    Exclusive upper bound for bar timestamps in the trade session (US/Eastern calendar day).
    session_end None: through end of calendar day (legacy, exclusive next midnight).
    session_end str like '17:00' / '17:00:00': last bar strictly before this clock on entry date.
    """
    entry_local = entry_time.tz_convert("US/Eastern")
    day_midnight = entry_local.normalize()
    if session_end is None or pd.isna(session_end):
        return day_midnight + pd.Timedelta(days=1)

    if isinstance(session_end, pd.Timestamp):
        se = session_end
        if se.tzinfo is None:
            se = se.tz_localize("US/Eastern")
        else:
            se = se.tz_convert("US/Eastern")
        return pd.Timestamp(
            year=entry_local.year,
            month=entry_local.month,
            day=entry_local.day,
            hour=se.hour,
            minute=se.minute,
            second=se.second,
            microsecond=se.microsecond,
            tz="US/Eastern",
        )

    s = str(session_end).strip()
    if not s:
        return day_midnight + pd.Timedelta(days=1)

    parsed = pd.to_datetime(s, errors="coerce")
    if pd.isna(parsed):
        return day_midnight + pd.Timedelta(days=1)
    t = parsed.time()
    return pd.Timestamp(
        year=entry_local.year,
        month=entry_local.month,
        day=entry_local.day,
        hour=t.hour,
        minute=t.minute,
        second=t.second,
        microsecond=t.microsecond,
        tz="US/Eastern",
    )


def _parse_stop_mode(raw: Any) -> str:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return "intrabar"
    s = str(raw).strip().lower()
    if s in ("close", "on_close", "close_only"):
        return "close"
    return "intrabar"


def _parse_partial_ladder(sig: pd.Series) -> tuple[list[float], list[float]] | None:
    """Return (targets, fracs) as lists of initial-contract fractions, or None for single-target mode."""
    if "partial_targets" not in sig.index and "partial_fracs" not in sig.index:
        return None
    pt_raw = sig.get("partial_targets")
    pf_raw = sig.get("partial_fracs")
    if pt_raw is None or pf_raw is None:
        return None
    if isinstance(pt_raw, float) and pd.isna(pt_raw):
        return None
    if isinstance(pf_raw, float) and pd.isna(pf_raw):
        return None

    if isinstance(pt_raw, str):
        try:
            targets = json.loads(pt_raw)
        except json.JSONDecodeError:
            return None
    else:
        targets = list(pt_raw)

    if isinstance(pf_raw, str):
        try:
            fracs = json.loads(pf_raw)
        except json.JSONDecodeError:
            return None
    else:
        fracs = list(pf_raw)

    if not targets or not fracs or len(targets) != len(fracs):
        return None
    targets_f = [float(x) for x in targets]
    fracs_f = [float(x) for x in fracs]
    if any(f < 0 for f in fracs_f) or sum(fracs_f) > 1.0001:
        return None
    return targets_f, fracs_f


def _simulate_trade(
    bars: pd.DataFrame,
    entry_time: pd.Timestamp,
    direction: str,
    entry: float,
    stop: float,
    target: float,
    contracts: int,
    instrument: Instrument,
    *,
    session_end: Any = None,
    stop_mode: str = "intrabar",
    partial_ladder: tuple[list[float], list[float]] | None = None,
    be_on_tp1: bool = False,
) -> TradeResult:
    """
    Simulate one trade from entry bar forward.

    stop_mode 'intrabar': stop hit when low/high touches stop (legacy).
    stop_mode 'close': stop hit when close crosses stop (full remaining exits at stop).

    Partial ladder: partial_ladder = (target_prices, fractions_of_initial_contracts).
    Targets fill on bar wick (high >= target for long). Same-bar priority: stop before targets
    (if close violates stop, full exit at stop and no partials that bar). Intraday stop (legacy)
    is checked before targets on each bar.

    r_multiple: net_pnl / initial_risk_dollars where initial_risk_dollars is
    abs(actual_entry - actual_stop) * initial_contracts * point_value.

    exit: VWAP of all exit prices weighted by contracts exited.
    """
    slippage = instrument.slippage_per_side()
    if direction == "long":
        actual_entry = entry + slippage
        actual_stop = stop - slippage
        actual_target = target - slippage
    else:
        actual_entry = entry - slippage
        actual_stop = stop + slippage
        actual_target = target + slippage

    cutoff = _session_cutoff_exclusive(entry_time, session_end)
    future_bars = bars[(bars.index > entry_time) & (bars.index < cutoff)]

    initial_contracts = contracts
    initial_risk_dollars = abs(actual_entry - actual_stop) * initial_contracts * instrument.point_value

    if partial_ladder is not None and len(partial_ladder[0]) > 0:
        return _simulate_partial_ladder(
            future_bars,
            entry_time,
            direction,
            actual_entry,
            actual_stop,
            instrument,
            initial_contracts,
            initial_risk_dollars,
            partial_ladder,
            stop_mode,
            stop,
            target,
            be_on_tp1=be_on_tp1,
        )

    return _simulate_single_target(
        future_bars,
        entry_time,
        direction,
        actual_entry,
        actual_stop,
        actual_target,
        initial_contracts,
        instrument,
        stop_mode,
        initial_risk_dollars,
    )


def _simulate_single_target(
    future_bars: pd.DataFrame,
    entry_time: pd.Timestamp,
    direction: str,
    actual_entry: float,
    actual_stop: float,
    actual_target: float,
    contracts: int,
    instrument: Instrument,
    stop_mode: str,
    initial_risk_dollars: float,
) -> TradeResult:
    exit_price = actual_entry
    exit_time = entry_time
    exit_reason = "eod"

    for ts, bar in future_bars.iterrows():
        if direction == "long":
            if stop_mode == "close":
                if bar["close"] <= actual_stop:
                    exit_price = actual_stop
                    exit_time = ts
                    exit_reason = "stop"
                    break
            else:
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
            if stop_mode == "close":
                if bar["close"] >= actual_stop:
                    exit_price = actual_stop
                    exit_time = ts
                    exit_reason = "stop"
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

    r_mult = net / initial_risk_dollars if initial_risk_dollars > 0 else 0.0

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


def _simulate_partial_ladder(
    future_bars: pd.DataFrame,
    entry_time: pd.Timestamp,
    direction: str,
    actual_entry: float,
    actual_stop: float,
    instrument: Instrument,
    initial_contracts: int,
    initial_risk_dollars: float,
    partial_ladder: tuple[list[float], list[float]],
    stop_mode: str,
    raw_stop: float,
    raw_target: float,
    *,
    be_on_tp1: bool = False,
) -> TradeResult:
    """Wick-based partial targets; close-based or intrabar full stop on remaining."""
    targets, fracs = partial_ladder
    slippage = instrument.slippage_per_side()
    if direction == "long":
        actual_targets = [t - slippage for t in targets]
    else:
        actual_targets = [t + slippage for t in targets]

    remaining = initial_contracts
    legs_filled = [False] * len(actual_targets)
    first_partial_filled = False
    gross = 0.0
    commission = 0.0
    weighted_exit_sum = 0.0
    contracts_exited = 0
    exit_time = entry_time
    exit_reason = "eod"

    def add_leg(price: float, n: int, ts: pd.Timestamp, reason: str) -> None:
        nonlocal gross, commission, weighted_exit_sum, contracts_exited, exit_time, exit_reason
        if n <= 0:
            return
        px = price
        g = (px - actual_entry) * n * instrument.point_value
        if direction == "short":
            g = -g
        gross += g
        commission += instrument.commission_round_turn * n
        weighted_exit_sum += px * n
        contracts_exited += n
        exit_time = ts
        exit_reason = reason

    for ts, bar in future_bars.iterrows():
        if remaining <= 0:
            break

        # 1) Close-based stop (full remaining)
        if stop_mode == "close":
            if direction == "long" and bar["close"] <= actual_stop:
                add_leg(actual_stop, remaining, ts, "stop")
                remaining = 0
                break
            if direction == "short" and bar["close"] >= actual_stop:
                add_leg(actual_stop, remaining, ts, "stop")
                remaining = 0
                break
        else:
            if direction == "long" and bar["low"] <= actual_stop:
                add_leg(actual_stop, remaining, ts, "stop")
                remaining = 0
                break
            if direction == "short" and bar["high"] >= actual_stop:
                add_leg(actual_stop, remaining, ts, "stop")
                remaining = 0
                break

        # 2) Partial targets (wick), in order
        for i, (at, frac) in enumerate(zip(actual_targets, fracs, strict=True)):
            if remaining <= 0 or legs_filled[i]:
                continue
            want = int(round(initial_contracts * frac))
            leg_size = max(0, min(remaining, want))
            if leg_size <= 0:
                legs_filled[i] = True
                continue
            touched = (
                bar["high"] >= at
                if direction == "long"
                else bar["low"] <= at
            )
            if touched:
                add_leg(at, leg_size, ts, "target")
                remaining -= leg_size
                legs_filled[i] = True
                if be_on_tp1 and not first_partial_filled:
                    actual_stop = actual_entry
                    first_partial_filled = True

    if remaining > 0:
        if len(future_bars) > 0:
            ts = future_bars.index[-1]
            px = future_bars.iloc[-1]["close"]
            add_leg(px, remaining, ts, "eod")
            remaining = 0
        else:
            add_leg(actual_entry, remaining, entry_time, "eod")

    net = gross - commission
    vwap_exit = weighted_exit_sum / contracts_exited if contracts_exited > 0 else actual_entry
    r_mult = net / initial_risk_dollars if initial_risk_dollars > 0 else 0.0

    slippage = instrument.slippage_per_side()
    if direction == "long":
        display_target = raw_target - slippage
    else:
        display_target = raw_target + slippage

    return TradeResult(
        entry_time=entry_time,
        exit_time=exit_time,
        direction=direction,
        entry=actual_entry,
        stop=actual_stop,
        target=display_target,
        exit=vwap_exit,
        contracts=initial_contracts,
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

    Optional per-signal columns (see eval_sim.strategy.StrategyContract docs):
      session_end, stop_mode, partial_targets, partial_fracs

    params['flat_only'] == True: skip signals that open before the prior simulated trade exits.
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

    entry_times = pd.to_datetime(signals_df["entry_time"])
    if entry_times.dt.tz is None:
        signals_df = signals_df.copy()
        signals_df["entry_time"] = entry_times.dt.tz_localize("US/Eastern")
    else:
        signals_df = signals_df.copy()
        signals_df["entry_time"] = entry_times.dt.tz_convert("US/Eastern")

    signals_df = signals_df.sort_values("entry_time").reset_index(drop=True)
    flat_only = bool(params.get("flat_only", False))
    last_exit_time: pd.Timestamp | None = None

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
            continue
        if flat_only and last_exit_time is not None and entry_ts < last_exit_time:
            continue

        session_end = None
        if "session_end" in sig.index:
            v = sig["session_end"]
            if pd.notna(v):
                session_end = v
        sm = _parse_stop_mode(sig.get("stop_mode") if "stop_mode" in sig.index else None)
        ladder = _parse_partial_ladder(sig)

        be_tp1 = False
        if "be_on_tp1" in sig.index:
            v = sig["be_on_tp1"]
            if pd.notna(v):
                be_tp1 = bool(v)

        trade = _simulate_trade(
            bars,
            entry_ts,
            str(sig["direction"]),
            float(sig["entry"]),
            float(sig["stop"]),
            float(sig["target"]),
            contracts,
            instrument,
            session_end=session_end,
            stop_mode=sm,
            partial_ladder=ladder,
            be_on_tp1=be_tp1,
        )
        trades.append(trade)
        if flat_only:
            last_exit_time = trade.exit_time

    return trades
