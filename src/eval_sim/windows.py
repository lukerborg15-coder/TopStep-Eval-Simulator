from __future__ import annotations
from dataclasses import dataclass
from dateutil.relativedelta import relativedelta
import pandas as pd
from eval_sim.config import DataSplitConfig, DATA_SPLIT


@dataclass(frozen=True)
class OOSWindow:
    warmup_start: pd.Timestamp   # feed bars from here to strategy (indicator warmup)
    start: pd.Timestamp          # score trades from here
    end: pd.Timestamp            # end of window


@dataclass(frozen=True)
class WFWindows:
    wf_start: pd.Timestamp
    wf_end: pd.Timestamp
    holdout: OOSWindow
    wf_windows: tuple[OOSWindow, ...]


def _resolve_warmup_start(
    scoring_start: pd.Timestamp,
    bars_index: pd.DatetimeIndex | None,
    warmup_bars: int,
) -> pd.Timestamp:
    """
    Walk back warmup_bars bars from scoring_start using the actual bar index.
    If no index is provided (e.g., in tests), returns scoring_start unchanged.
    """
    if bars_index is None or warmup_bars == 0:
        return scoring_start
    prior = bars_index[bars_index < scoring_start]
    if len(prior) == 0:
        return scoring_start
    idx = max(0, len(prior) - warmup_bars)
    return prior[idx]


def compute_windows(
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    bars_index: pd.DatetimeIndex | None,
    config: DataSplitConfig = DATA_SPLIT,
) -> WFWindows:
    """
    Divide data into: [WF period (3.5 years)] [Holdout (18 months)].

    The WF period is split into n_wf_windows equal non-overlapping OOS scoring
    windows. Each window has a warmup_start set warmup_bars bars before its
    scoring start (using the actual bar index). All WF data is scored — no
    train/test split.
    """
    holdout_start = data_end - relativedelta(months=config.holdout_months)
    wf_start = data_start
    wf_end = holdout_start

    total_months = (data_end - data_start).days / 30.44
    min_required = config.holdout_months + 6  # at least 6 months of WF data
    if total_months < min_required:
        raise ValueError(
            f"Data span ({total_months:.1f}mo) insufficient — need at least "
            f"{min_required}mo ({config.holdout_months}mo holdout + 6mo WF minimum)"
        )

    wf_duration = wf_end - wf_start
    window_duration = wf_duration / config.n_wf_windows

    wf_windows: list[OOSWindow] = []
    for i in range(config.n_wf_windows):
        scoring_start = wf_start + window_duration * i
        scoring_end = wf_start + window_duration * (i + 1)
        if i == config.n_wf_windows - 1:
            scoring_end = wf_end  # ensure last window reaches exactly wf_end
        warmup_start = _resolve_warmup_start(scoring_start, bars_index, config.warmup_bars)
        wf_windows.append(OOSWindow(
            warmup_start=warmup_start,
            start=scoring_start,
            end=scoring_end,
        ))

    holdout_warmup = _resolve_warmup_start(holdout_start, bars_index, config.warmup_bars)

    return WFWindows(
        wf_start=wf_start,
        wf_end=wf_end,
        holdout=OOSWindow(
            warmup_start=holdout_warmup,
            start=holdout_start,
            end=data_end,
        ),
        wf_windows=tuple(wf_windows),
    )
