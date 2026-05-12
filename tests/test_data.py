import pandas as pd
import pytest
from pathlib import Path
from eval_sim.data import load_ohlcv, DataLoadError


def test_load_ohlcv_returns_dataframe(tmp_path):
    csv = tmp_path / "mnq_5min_databento.csv"
    csv.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2020-01-02 09:30:00-05:00,16000,16010,15990,16005,500\n"
        "2020-01-02 09:35:00-05:00,16005,16015,15995,16010,400\n"
    )
    df = load_ohlcv("mnq", "5min", data_dir=tmp_path)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2


def test_load_ohlcv_index_is_datetime_eastern(tmp_path):
    csv = tmp_path / "mnq_5min_databento.csv"
    csv.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2020-01-02 09:30:00-05:00,16000,16010,15990,16005,500\n"
    )
    df = load_ohlcv("mnq", "5min", data_dir=tmp_path)
    assert df.index.tz is not None
    assert str(df.index.tz) in ("US/Eastern", "America/New_York", "EST", "EDT")


def test_load_ohlcv_sorted_ascending(tmp_path):
    csv = tmp_path / "mnq_5min_databento.csv"
    csv.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2020-01-02 09:35:00-05:00,16005,16015,15995,16010,400\n"
        "2020-01-02 09:30:00-05:00,16000,16010,15990,16005,500\n"
    )
    df = load_ohlcv("mnq", "5min", data_dir=tmp_path)
    assert df.index.is_monotonic_increasing


def test_load_ohlcv_missing_file_raises(tmp_path):
    with pytest.raises(DataLoadError, match="not found"):
        load_ohlcv("mnq", "5min", data_dir=tmp_path)


def test_load_ohlcv_missing_columns_raises(tmp_path):
    csv = tmp_path / "mnq_5min_databento.csv"
    csv.write_text("timestamp,open,high,low\n2020-01-02 09:30:00-05:00,1,2,3\n")
    with pytest.raises(DataLoadError, match="columns"):
        load_ohlcv("mnq", "5min", data_dir=tmp_path)


def test_load_ohlcv_accepts_datetime_column(tmp_path):
    csv = tmp_path / "mnq_5min_databento.csv"
    csv.write_text(
        "datetime,open,high,low,close,volume\n"
        "2020-01-02 09:30:00-05:00,16000,16010,15990,16005,500\n"
    )
    df = load_ohlcv("mnq", "5min", data_dir=tmp_path)
    assert len(df) == 1
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
