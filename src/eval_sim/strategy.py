from __future__ import annotations
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import pandas as pd


class StrategyLoadError(Exception):
    pass


@dataclass(frozen=True)
class StrategyContract:
    name: str
    param_ranges: dict[str, list]
    generate_signals: Callable[[pd.DataFrame, dict], pd.DataFrame]
    warmup_bars: int = 200    # bars needed before first scored signal; overrides config default


def load_strategy(path: str | Path) -> StrategyContract:
    """
    Load a strategy from a Python file.
    The file must define:
      PARAM_RANGES: dict[str, list]
      generate_signals(bars: pd.DataFrame, params: dict) -> pd.DataFrame

    generate_signals must return at least columns:
      entry_time, direction, entry, stop, target

    Optional columns (trade simulation in evaluate_window):
      session_end — str clock on entry date (US/Eastern), e.g. '17:00'; omit for legacy midnight session end.
      stop_mode — 'intrabar' (default) or 'close' (stop when bar close crosses stop).
      partial_targets, partial_fracs — same-length lists or JSON strings; fractions are shares of
      initial contracts per ladder rung; wick-based fills. Omit for single target.

    Optional params key for evaluate_window (include in PARAM_RANGES if search should vary it):
      flat_only — if True, skip signals whose entry_time is before the prior trade's exit_time.
    """
    path = Path(path)
    if not path.exists():
        raise StrategyLoadError(f"Strategy file not found: {path}")

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise StrategyLoadError(f"Error loading {path}: {exc}") from exc

    if not hasattr(module, "PARAM_RANGES"):
        raise StrategyLoadError(f"Strategy {path.name} must define PARAM_RANGES dict")
    if not hasattr(module, "generate_signals"):
        raise StrategyLoadError(f"Strategy {path.name} must define generate_signals function")
    if not callable(module.generate_signals):
        raise StrategyLoadError(f"generate_signals in {path.name} must be callable")

    # WARMUP_BARS is optional — strategies that use HTF indicators or long-period
    # indicators should declare this explicitly.
    # Example: a strategy using a 200-period 1h EMA on 5min bars needs
    # 200 * 12 = 2400 warmup bars (200 hourly bars × 12 five-min bars/hour).
    warmup_bars = getattr(module, "WARMUP_BARS", 200)
    if not isinstance(warmup_bars, int) or warmup_bars < 0:
        raise StrategyLoadError(f"WARMUP_BARS in {path.name} must be a non-negative int")

    return StrategyContract(
        name=path.stem,
        param_ranges=module.PARAM_RANGES,
        generate_signals=module.generate_signals,
        warmup_bars=warmup_bars,
    )
