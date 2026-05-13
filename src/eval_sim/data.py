from __future__ import annotations
from pathlib import Path
import pandas as pd


class DataLoadError(Exception):
    pass


REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


def load_ohlcv(
    instrument: str,
    timeframe: str,
    data_dir: str | Path = "Data",
) -> pd.DataFrame:
    """Load OHLCV CSV and return a timezone-aware, sorted DataFrame."""
    data_dir = Path(data_dir)
    filename = f"{instrument.lower()}_{timeframe}_databento.csv"
    path = data_dir / filename

    if not path.exists():
        raise DataLoadError(f"Data file not found: {path}")

    header_cols = pd.read_csv(path, nrows=0).columns
    if "timestamp" in header_cols:
        index_col = "timestamp"
    elif "datetime" in header_cols:
        index_col = "datetime"
    else:
        raise DataLoadError(f"No timestamp/datetime column in {path}")

    df = pd.read_csv(path, index_col=index_col, parse_dates=True)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataLoadError(f"Missing columns {missing} in {path}")

    normalized_index = []
    for value in df.index:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None or ts.utcoffset() is None:
            normalized_index.append(ts.tz_localize("US/Eastern"))
        else:
            normalized_index.append(ts.tz_convert("US/Eastern"))
    df.index = pd.DatetimeIndex(normalized_index)

    df = df.sort_index()
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df
