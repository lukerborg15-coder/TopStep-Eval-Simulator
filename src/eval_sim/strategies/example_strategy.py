"""
Example strategy for topstep_eval_sim.

To create a new strategy:
1. Copy this file to src/eval_sim/strategies/your_strategy_name.py
2. Define PARAM_RANGES with the parameters you want to search
3. Implement generate_signals() to return a DataFrame of trade signals

generate_signals receives:
  bars   — pd.DataFrame with columns [open, high, low, close, volume],
            timezone-aware DatetimeIndex (US/Eastern), filtered to the eval window
  params — dict with one value per key in PARAM_RANGES (the specific combo being tested)

generate_signals must return a pd.DataFrame with columns:
  entry_time  — pd.Timestamp (must match a bar's index value)
  direction   — "long" or "short"
  entry       — float (limit/market entry price)
  stop        — float (stop loss price)
  target      — float (profit target price)

Return an empty DataFrame (with these columns) if there are no signals.
"""
import pandas as pd


# WARMUP_BARS tells the pipeline how many bars to prepend before the scoring
# window so your indicators have time to stabilize. The default is 200.
#
# If your strategy uses a higher timeframe indicator, calculate carefully:
#   e.g. 200-period 1h EMA on 5min bars → 200 × 12 = 2400 warmup bars
#   e.g. 50-period daily EMA on 5min bars → 50 × 78 = 3900 warmup bars
#
# Setting this too low means your first N signals have unstable indicator
# values and will pollute the OOS score. Setting it too high wastes OOS data.
WARMUP_BARS = 300  # 300 five-min bars ≈ 5 trading sessions

PARAM_RANGES = {
    "fast_ema": [5, 8, 13, 21],
    "slow_ema": [34, 55, 89],
    "atr_mult_stop": [1.0, 1.5, 2.0],
    "rr_ratio": [1.5, 2.0, 2.5],
}


def generate_signals(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    fast = int(params["fast_ema"])
    slow = int(params["slow_ema"])
    atr_mult = float(params["atr_mult_stop"])
    rr = float(params["rr_ratio"])

    if len(bars) < slow + 1:
        return pd.DataFrame(columns=["entry_time", "direction", "entry", "stop", "target"])

    fast_ema = bars["close"].ewm(span=fast, adjust=False).mean()
    slow_ema = bars["close"].ewm(span=slow, adjust=False).mean()

    # ATR for stop sizing
    tr = pd.concat([
        bars["high"] - bars["low"],
        (bars["high"] - bars["close"].shift()).abs(),
        (bars["low"] - bars["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()

    signals = []
    prev_fast = fast_ema.shift(1)
    prev_slow = slow_ema.shift(1)

    # Bullish crossover: fast crosses above slow
    crossed_up = (fast_ema > slow_ema) & (prev_fast <= prev_slow)
    # Bearish crossover: fast crosses below slow
    crossed_down = (fast_ema < slow_ema) & (prev_fast >= prev_slow)

    for ts in bars.index[slow:]:
        price = bars.loc[ts, "close"]
        atr_val = atr.loc[ts]
        if atr_val <= 0:
            continue

        if crossed_up.loc[ts]:
            stop = price - atr_val * atr_mult
            target = price + (price - stop) * rr
            signals.append({
                "entry_time": ts,
                "direction": "long",
                "entry": price,
                "stop": stop,
                "target": target,
            })
        elif crossed_down.loc[ts]:
            stop = price + atr_val * atr_mult
            target = price - (stop - price) * rr
            signals.append({
                "entry_time": ts,
                "direction": "short",
                "entry": price,
                "stop": stop,
                "target": target,
            })

    return pd.DataFrame(signals) if signals else pd.DataFrame(
        columns=["entry_time", "direction", "entry", "stop", "target"]
    )
