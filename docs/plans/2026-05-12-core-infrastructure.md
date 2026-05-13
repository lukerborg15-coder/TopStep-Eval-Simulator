# TopStep Eval Sim — Plan 1: Core Infrastructure

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the standalone core of `topstep_eval_sim` — project setup, data loading, trade simulation, Topstep rules, WF window computation, strategy interface, and random param search with multiprocessing. Zero imports from the Topstep pipeline project.

**Architecture:** Standalone Python package at `C:\Users\Luker\projects\TopStep Eval Sim\`. Reads OHLCV CSVs from `Data/`. Core flow: load data → auto-compute windows → discover strategy → random-search params → score across WF OOS folds. Plan 2 builds the pipeline stages on top of these primitives.

**Tech Stack:** Python 3.11+, pandas, numpy, multiprocessing (stdlib), pytest

---

## File Map

| File | Responsibility |
|------|----------------|
| `pyproject.toml` | Package config, entry point, dependencies |
| `src/eval_sim/__init__.py` | Empty package marker |
| `src/eval_sim/config.py` | Topstep constants, instrument specs, data-split defaults, search defaults |
| `src/eval_sim/data.py` | Load OHLCV CSV → validated DataFrame |
| `src/eval_sim/trades.py` | `TradeResult` dataclass |
| `src/eval_sim/topstep.py` | `simulate_topstep`, `simulate_seq_evals` |
| `src/eval_sim/evaluator.py` | `evaluate_window(bars, strategy_fn, params, window, instrument, risk_dollars, max_contracts) → list[TradeResult]` |
| `src/eval_sim/windows.py` | `compute_windows(data_start, data_end) → WFWindows` |
| `src/eval_sim/strategy.py` | Strategy contract + auto-discovery from `strategies/` |
| `src/eval_sim/search.py` | `random_search(bars, strategy_fn, param_ranges, wf_windows, ...) → dict` |
| `src/eval_sim/strategies/__init__.py` | Empty |
| `src/eval_sim/strategies/example_strategy.py` | Working reference strategy |
| `tests/conftest.py` | Shared fixtures (synthetic OHLCV bars, sample trades) |
| `tests/test_config.py` | Instrument and rules sanity checks |
| `tests/test_data.py` | CSV loading, column validation, bad-file handling |
| `tests/test_trades.py` | TradeResult construction |
| `tests/test_topstep.py` | simulate_topstep pass/fail/drawdown breach, simulate_seq_evals |
| `tests/test_evaluator.py` | Signal → trade conversion, slippage, sizing, window filtering |
| `tests/test_windows.py` | Auto-computed splits, edge cases |
| `tests/test_strategy.py` | Auto-discovery, missing-file handling, bad contract error |
| `tests/test_search.py` | Random sampling, multiprocessing, best-params selection |

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `src/eval_sim/__init__.py`
- Create: `src/eval_sim/strategies/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create directory structure**

```
C:\Users\Luker\projects\TopStep Eval Sim\
  src\eval_sim\__init__.py
  src\eval_sim\strategies\__init__.py
  tests\__init__.py
```

Run in project root (`C:\Users\Luker\projects\TopStep Eval Sim`):
```bash
mkdir -p src/eval_sim/strategies tests
touch src/eval_sim/__init__.py src/eval_sim/strategies/__init__.py tests/__init__.py
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "topstep-eval-sim"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0",
    "numpy>=1.26",
    "python-dateutil>=2.8",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov"]

[project.scripts]
eval-sim = "eval_sim.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
```

- [ ] **Step 3: Install package in editable mode**

```bash
pip install -e ".[dev]"
```

Expected: `Successfully installed topstep-eval-sim-0.1.0`

- [ ] **Step 4: Create `tests/conftest.py` with shared fixtures**

```python
import pandas as pd
import numpy as np
import pytest
from eval_sim.trades import TradeResult


def make_bars(n: int = 200, freq: str = "5min", start: str = "2020-01-02 09:30") -> pd.DataFrame:
    """Synthetic OHLCV bars with realistic price action."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(start, periods=n, freq=freq, tz="US/Eastern")
    closes = 16000.0 + np.cumsum(rng.normal(0, 2, n))
    highs = closes + rng.uniform(1, 5, n)
    lows = closes - rng.uniform(1, 5, n)
    opens = closes - rng.normal(0, 1, n)
    volume = rng.integers(100, 1000, n).astype(float)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
        index=dates,
    )


def make_trade(net_pnl: float = 100.0, entry_time: str = "2020-01-02 09:35") -> TradeResult:
    ts = pd.Timestamp(entry_time, tz="US/Eastern")
    return TradeResult(
        entry_time=ts,
        exit_time=ts + pd.Timedelta("5min"),
        direction="long",
        entry=16000.0,
        stop=15990.0,
        target=16020.0,
        exit=16020.0,
        contracts=1,
        gross_pnl=net_pnl + 0.62,
        commission=0.62,
        net_pnl=net_pnl,
        r_multiple=2.0,
        exit_reason="target",
    )


@pytest.fixture
def bars():
    return make_bars()


@pytest.fixture
def winning_trade():
    return make_trade(net_pnl=40.0)


@pytest.fixture
def losing_trade():
    return make_trade(net_pnl=-20.62)
```

- [ ] **Step 5: Verify pytest runs (no tests yet)**

```bash
pytest -v
```

Expected: `no tests ran` (exit 0 or 5 depending on pytest version — both fine)

- [ ] **Step 6: Commit**

```bash
git init
git add pyproject.toml src/ tests/
git commit -m "feat: project setup, package structure, shared fixtures"
```

---

## Task 2: Config

**Files:**
- Create: `src/eval_sim/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from eval_sim.config import MNQ, MES, TOPSTEP_50K, DATA_SPLIT, SEARCH


def test_mnq_instrument():
    assert MNQ.symbol == "MNQ"
    assert MNQ.point_value == 2.0
    assert MNQ.tick_size == 0.25
    assert MNQ.commission_round_turn == 0.62
    assert MNQ.slippage_ticks_per_side == 1


def test_mes_instrument():
    assert MES.symbol == "MES"
    assert MES.point_value == 5.0
    assert MES.tick_size == 0.25
    assert MES.commission_round_turn == 0.85


def test_topstep_50k_rules():
    assert TOPSTEP_50K.account_size == 50_000.0
    assert TOPSTEP_50K.profit_target == 3_000.0
    assert TOPSTEP_50K.max_drawdown == 2_000.0
    assert TOPSTEP_50K.daily_loss_limit == 1_000.0
    assert TOPSTEP_50K.consistency_pct == 0.50
    assert TOPSTEP_50K.max_trading_days == 60


def test_data_split_defaults():
    assert DATA_SPLIT.wf_years == 3.5
    assert DATA_SPLIT.holdout_months == 18
    assert DATA_SPLIT.n_wf_windows == 4
    assert DATA_SPLIT.warmup_bars == 200


def test_search_defaults():
    assert SEARCH.n_candidates == 400
    assert SEARCH.seed == 42
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'eval_sim.config'`

- [ ] **Step 3: Create `src/eval_sim/config.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_config.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval_sim/config.py tests/test_config.py
git commit -m "feat: config — instruments, topstep rules, data split and search defaults"
```

---

## Task 3: Data Loader

**Files:**
- Create: `src/eval_sim/data.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_data.py
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
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_data.py -v
```

Expected: `ModuleNotFoundError: No module named 'eval_sim.data'`

- [ ] **Step 3: Create `src/eval_sim/data.py`**

```python
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

    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataLoadError(f"Missing columns {missing} in {path}")

    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")

    df = df.sort_index()
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_data.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval_sim/data.py tests/test_data.py
git commit -m "feat: data loader — CSV reader with column validation and timezone handling"
```

---

## Task 4: Trade Types

**Files:**
- Create: `src/eval_sim/trades.py`
- Create: `tests/test_trades.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_trades.py
import pandas as pd
from eval_sim.trades import TradeResult


def test_trade_result_fields():
    ts = pd.Timestamp("2020-01-02 09:35", tz="US/Eastern")
    t = TradeResult(
        entry_time=ts,
        exit_time=ts + pd.Timedelta("5min"),
        direction="long",
        entry=16000.0,
        stop=15990.0,
        target=16020.0,
        exit=16020.0,
        contracts=1,
        gross_pnl=40.62,
        commission=0.62,
        net_pnl=40.0,
        r_multiple=2.0,
        exit_reason="target",
    )
    assert t.net_pnl == 40.0
    assert t.exit_reason == "target"
    assert t.direction == "long"


def test_trade_result_is_immutable():
    ts = pd.Timestamp("2020-01-02 09:35", tz="US/Eastern")
    t = TradeResult(
        entry_time=ts, exit_time=ts, direction="long",
        entry=1.0, stop=0.9, target=1.2, exit=1.2,
        contracts=1, gross_pnl=0.0, commission=0.0,
        net_pnl=0.0, r_multiple=0.0, exit_reason="target",
    )
    import dataclasses
    assert dataclasses.is_dataclass(t)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_trades.py -v
```

Expected: `ModuleNotFoundError: No module named 'eval_sim.trades'`

- [ ] **Step 3: Create `src/eval_sim/trades.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_trades.py -v
```

Expected: all 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval_sim/trades.py tests/test_trades.py
git commit -m "feat: TradeResult dataclass"
```

---

## Task 5: Topstep Simulation Core

**Files:**
- Create: `src/eval_sim/topstep.py`
- Create: `tests/test_topstep.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_topstep.py
import pandas as pd
import pytest
from tests.conftest import make_trade
from eval_sim.topstep import simulate_topstep, simulate_seq_evals, TopstepResult
from eval_sim.config import TOPSTEP_50K, TopstepRules


def _trade(pnl: float, day: str = "2020-01-02") -> object:
    return make_trade(net_pnl=pnl, entry_time=f"{day} 09:35")


def test_simulate_topstep_pass_consistency_satisfied():
    # Spread profit across multiple days so best day is well under 50%
    # 31 days × $100 = $3,100. Best day = $100, total = $3,100 → 3.2% < 50%
    trades = [_trade(100.0, f"2020-01-{2 + i:02d}") for i in range(31)]
    result = simulate_topstep(trades, TOPSTEP_50K)
    assert result.passed is True


def test_simulate_topstep_consistency_blocks_pass():
    # Day 1: $3,100 profit (all profit in one day = 100% > 50% → blocked)
    # Day 2: $100 more → total $3,200, best day still $3,100 → 96.9% → still blocked
    # With only 2 days of trades, consistency can never be satisfied here
    trades = [_trade(3100.0, "2020-01-02"), _trade(100.0, "2020-01-03")]
    result = simulate_topstep(trades, TOPSTEP_50K)
    assert result.passed is False
    assert result.fail_reason == "end_of_trades"  # blocked by consistency, not breached


def test_simulate_topstep_consistency_eventually_satisfied():
    # Day 1: $1,800 profit. Day 2: $1,800 more = $3,600 total.
    # best_day=$1,800 / total=$3,600 = 50% → exactly at limit, passes
    trades = [_trade(1800.0, "2020-01-02"), _trade(1800.0, "2020-01-03")]
    result = simulate_topstep(trades, TOPSTEP_50K)
    assert result.passed is True


def test_simulate_topstep_drawdown_breach():
    trades = [_trade(-2500.0, "2020-01-02")]
    result = simulate_topstep(trades, TOPSTEP_50K)
    assert result.passed is False
    assert result.fail_reason == "max_drawdown"


def test_simulate_topstep_eod_trailing_floor_raises():
    # Day 1: +$1,500 → EOD balance $51,500 → floor becomes $49,500
    # Day 2: -$1,600 → balance $49,900 → still above new floor $49,500 → OK
    # Day 2: then -$500 more → balance $49,400 → below $49,500 floor → breach
    trades = [
        _trade(1500.0, "2020-01-02"),   # EOD floor now $49,500
        _trade(-1600.0, "2020-01-03"),  # balance $49,900, above floor
        _trade(-500.0, "2020-01-03"),   # balance $49,400, below $49,500 floor → breach
    ]
    result = simulate_topstep(trades, TOPSTEP_50K)
    assert result.passed is False
    assert result.fail_reason == "max_drawdown"


def test_simulate_topstep_eod_floor_does_not_trail_intraday():
    # Intraday spike does NOT raise the floor — only EOD balance does.
    # Day 1 closes at $50,000 (started $50,000, made and lost $1,500 intraday)
    # Floor stays at $48,000 (intraday high doesn't count)
    # Day 2: -$1,900 → balance $48,100 → above $48,000 → no breach
    trades = [
        _trade(1500.0, "2020-01-02"),   # intraday +1500
        _trade(-1500.0, "2020-01-02"),  # intraday -1500; EOD balance = $50,000 → floor stays $48,000
        _trade(-1900.0, "2020-01-03"),  # balance $48,100 → above $48,000 floor → no breach
    ]
    result = simulate_topstep(trades, TOPSTEP_50K)
    assert result.fail_reason != "max_drawdown"


def test_simulate_topstep_intraday_pass_detected():
    # Profit crosses target mid-day (two trades on day 2 each +$750).
    # After 4th trade: balance=$53,000. Consistency: best_day=$1,500 / total=$3,000 = 50% → passes.
    # The 5th trade (-$5,000) must never execute — pass was already returned.
    trades = [
        _trade(750.0, "2020-01-02"),
        _trade(750.0, "2020-01-02"),
        _trade(750.0, "2020-01-03"),
        _trade(750.0, "2020-01-03"),  # crosses target, consistency satisfied → pass
        _trade(-5000.0, "2020-01-03"),  # must not execute — already passed
    ]
    result = simulate_topstep(trades, TOPSTEP_50K)
    assert result.passed is True
    assert result.trading_days == 2
    assert result.final_balance == 53_000.0


def test_simulate_topstep_daily_loss_locks_out_day_not_attempt():
    # After -$1,100 on day 1 (exceeds $1,000 limit), day is locked out.
    # Day 2 still proceeds — attempt is not failed.
    # Trades: day 1 -$600, -$600 (second trade breaches limit → lockout);
    #         day 2 +$100 (still allowed).
    trades = [
        _trade(-600.0, "2020-01-02"),
        _trade(-600.0, "2020-01-02"),  # cumulative -$1,200 → lockout; second -$600 still applied
        _trade(100.0, "2020-01-03"),
    ]
    result = simulate_topstep(trades, TOPSTEP_50K)
    assert result.fail_reason != "daily_loss"  # daily loss never fails the attempt
    assert result.trading_days == 2             # both days processed


def test_simulate_topstep_daily_loss_does_not_fail_attempt():
    # Exceed daily loss on day 1, then recover over many subsequent days.
    # The attempt should complete normally (end_of_trades or max_days) not "daily_loss".
    trades = (
        [_trade(-1100.0, "2020-01-02")]  # day 1 locked out after this trade
        + [_trade(100.0, f"2020-01-{3 + i:02d}") for i in range(10)]
    )
    result = simulate_topstep(trades, TOPSTEP_50K)
    assert result.fail_reason in ("end_of_trades", "max_days", None)
    assert result.fail_reason != "daily_loss"


def test_simulate_topstep_no_trades():
    result = simulate_topstep([], TOPSTEP_50K)
    assert result.passed is False
    assert result.fail_reason == "no_trades"
    assert result.trading_days == 0


def test_simulate_seq_evals_returns_attempt_days():
    trades = [_trade(100.0, f"2020-01-{2 + i:02d}") for i in range(60)]
    result = simulate_seq_evals(trades, TOPSTEP_50K)
    assert isinstance(result.attempt_days, tuple)
    assert len(result.attempt_days) == result.attempts
    assert all(d > 0 for d in result.attempt_days)


def test_simulate_seq_evals_zero_trades():
    result = simulate_seq_evals([], TOPSTEP_50K)
    assert result.attempts == 0
    assert result.passes == 0
    assert result.pass_rate == 0.0
    assert result.attempt_days == ()
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_topstep.py -v
```

Expected: `ModuleNotFoundError: No module named 'eval_sim.topstep'`

- [ ] **Step 3: Create `src/eval_sim/topstep.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
import statistics
import pandas as pd
from eval_sim.config import TopstepRules
from eval_sim.trades import TradeResult


@dataclass(frozen=True)
class TopstepResult:
    passed: bool
    fail_reason: str | None  # "max_drawdown"|"max_days"|"end_of_trades"|"no_trades"|None
    final_balance: float
    max_drawdown_seen: float
    trading_days: int


@dataclass(frozen=True)
class SeqEvalResult:
    passes: int
    attempts: int
    pass_rate: float
    attempt_days: tuple[int, ...]   # trading days consumed per attempt — used for median calculation


def _group_by_day(trades: list[TradeResult]) -> dict[str, list[TradeResult]]:
    """Group trades by exit day as YYYY-MM-DD string (US/Eastern). No calendar gaps."""
    groups: dict[str, list[TradeResult]] = {}
    for t in trades:
        day = t.exit_time.tz_convert("US/Eastern").date().isoformat()
        groups.setdefault(day, []).append(t)
    return groups


def simulate_topstep(
    trades: list[TradeResult],
    rules: TopstepRules,
) -> TopstepResult:
    """
    Simulate one Combine evaluation attempt with correct Topstep 50K rules:

    DRAWDOWN — trails from end-of-day closing balance (updates at 16:00 Chicago /
    US/Eastern close, NOT intraday). floor = max(previous floors, eod_balance - max_drawdown).
    Intraday balance is checked against the current floor after every trade.

    CONSISTENCY — when balance crosses profit_target, check:
        best_profitable_day_pnl / total_profit <= consistency_pct (50%)
    Only profitable days count for the numerator. If consistency fails, trading
    continues — the attempt is not failed, just not yet passed. Naturally requires
    at least 2 trading days (can't satisfy 50% with a single day of profit).

    DAILY LOSS — if cumulative net PnL for the day reaches -daily_loss_limit at any
    point intraday, remaining trades for that day are skipped (account locked out).
    The attempt is NOT failed — trading resumes the next session.

    NO MINIMUM DAYS — but consistency rule naturally enforces >= 2 profitable days.
    """
    if not trades:
        return TopstepResult(
            passed=False, fail_reason="no_trades",
            final_balance=rules.account_size,
            max_drawdown_seen=0.0, trading_days=0,
        )

    balance = rules.account_size
    floor = rules.account_size - rules.max_drawdown   # starts static, trails EOD
    target_balance = rules.account_size + rules.profit_target
    max_drawdown_seen = 0.0
    trading_days = 0
    daily_pnls: dict[str, float] = {}  # day → net PnL for that day

    by_day = _group_by_day(trades)

    for day, day_trades in sorted(by_day.items()):
        trading_days += 1
        if trading_days > rules.max_trading_days:
            return TopstepResult(
                passed=False, fail_reason="max_days",
                final_balance=balance,
                max_drawdown_seen=max_drawdown_seen,
                trading_days=trading_days,
            )

        day_start_balance = balance
        running_day_pnl = 0.0
        for trade in day_trades:
            balance += trade.net_pnl
            running_day_pnl += trade.net_pnl
            dd = balance - rules.account_size  # negative = drawdown from start
            max_drawdown_seen = max(max_drawdown_seen, -dd)

            # Intraday drawdown check against current floor
            if balance <= floor:
                return TopstepResult(
                    passed=False, fail_reason="max_drawdown",
                    final_balance=balance,
                    max_drawdown_seen=max_drawdown_seen,
                    trading_days=trading_days,
                )

            # Intraday pass check — Topstep evaluates continuously, not just at EOD.
            # Include the current day's running PnL (not yet in daily_pnls) for consistency.
            if balance >= target_balance:
                temp_pnls = {**daily_pnls, day: running_day_pnl}
                total_profit = balance - rules.account_size
                profitable_days = [p for p in temp_pnls.values() if p > 0]
                best_day = max(profitable_days, default=0.0)
                if total_profit > 0 and best_day / total_profit <= rules.consistency_pct:
                    return TopstepResult(
                        passed=True, fail_reason=None,
                        final_balance=balance,
                        max_drawdown_seen=max_drawdown_seen,
                        trading_days=trading_days,
                    )
                # else: consistency not yet satisfied — keep trading

            # Daily loss limit: lock out remaining trades for today.
            # The attempt is NOT failed — trading resumes next session.
            if running_day_pnl <= -rules.daily_loss_limit:
                break

        # End-of-day processing
        daily_pnls[day] = running_day_pnl

        # EOD trailing drawdown: floor advances to eod_balance - max_drawdown
        # but never goes below the starting floor
        floor = max(floor, balance - rules.max_drawdown)

    return TopstepResult(
        passed=False, fail_reason="end_of_trades",
        final_balance=balance,
        max_drawdown_seen=max_drawdown_seen,
        trading_days=trading_days,
    )


def simulate_seq_evals(
    trades: list[TradeResult],
    rules: TopstepRules,
) -> SeqEvalResult:
    """
    Chain sequential Combine evaluation attempts across a trade list.
    After each attempt (pass or fail), resume from the first trading day
    not consumed by that attempt.

    Uses YYYY-MM-DD date strings throughout — _group_by_day only includes
    days with actual trades, so weekends and holidays create no gaps.
    attempt_days records trading_days consumed per attempt for median reporting.
    """
    if not trades:
        return SeqEvalResult(passes=0, attempts=0, pass_rate=0.0, attempt_days=())

    remaining = list(trades)
    passes = 0
    attempts = 0
    attempt_days: list[int] = []
    MAX_ATTEMPTS = 10_000  # safety valve against infinite loops from buggy strategies

    while remaining and attempts < MAX_ATTEMPTS:
        result = simulate_topstep(remaining, rules)

        if result.trading_days == 0:
            break  # no progress — stop

        attempts += 1
        attempt_days.append(result.trading_days)
        if result.passed:
            passes += 1

        by_day = _group_by_day(remaining)
        sorted_days = sorted(by_day.keys())

        if result.trading_days >= len(sorted_days):
            break  # all trading days consumed

        next_day = sorted_days[result.trading_days]
        remaining = [
            t for t in remaining
            if t.exit_time.tz_convert("US/Eastern").date().isoformat() >= next_day
        ]

    pass_rate = passes / attempts if attempts > 0 else 0.0
    return SeqEvalResult(
        passes=passes,
        attempts=attempts,
        pass_rate=pass_rate,
        attempt_days=tuple(attempt_days),
    )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_topstep.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval_sim/topstep.py tests/test_topstep.py
git commit -m "feat: topstep simulation — simulate_topstep, simulate_seq_evals"
```

---

## Task 6: Trade Evaluator

**Files:**
- Create: `src/eval_sim/evaluator.py`
- Create: `tests/test_evaluator.py`

The evaluator converts raw signals from a strategy into `TradeResult` objects. It applies position sizing (risk dollars / stop distance / point value) and slippage.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_evaluator.py
import pandas as pd
import pytest
from tests.conftest import make_bars, make_trade
from eval_sim.evaluator import evaluate_window, Window
from eval_sim.config import MNQ
from eval_sim.trades import TradeResult


def _always_long_strategy(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Strategy that signals long on every bar."""
    stop_dist = params.get("stop_dist", 10.0)
    signals = []
    for ts, row in bars.iterrows():
        signals.append({
            "entry_time": ts,
            "direction": "long",
            "entry": row["close"],
            "stop": row["close"] - stop_dist,
            "target": row["close"] + stop_dist * 2,
        })
    return pd.DataFrame(signals)


def _no_signal_strategy(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    return pd.DataFrame(columns=["entry_time", "direction", "entry", "stop", "target"])


def test_evaluate_window_returns_trades(bars):
    window = Window(start=bars.index[0], end=bars.index[-1])
    trades = evaluate_window(
        bars, _always_long_strategy, params={"stop_dist": 10.0},
        window=window, instrument=MNQ, risk_dollars=500.0, max_contracts=5,
    )
    assert isinstance(trades, list)
    assert len(trades) > 0
    assert all(isinstance(t, TradeResult) for t in trades)


def test_evaluate_window_no_signals(bars):
    window = Window(start=bars.index[0], end=bars.index[-1])
    trades = evaluate_window(
        bars, _no_signal_strategy, params={},
        window=window, instrument=MNQ, risk_dollars=500.0, max_contracts=5,
    )
    assert trades == []


def test_evaluate_window_filters_to_window(bars):
    mid = bars.index[len(bars) // 2]
    window = Window(start=mid, end=bars.index[-1])
    trades = evaluate_window(
        bars, _always_long_strategy, params={"stop_dist": 10.0},
        window=window, instrument=MNQ, risk_dollars=500.0, max_contracts=5,
    )
    for t in trades:
        assert t.entry_time >= mid


def test_evaluate_window_warmup_bars_not_scored(bars):
    # warmup_start is earlier than window.start — signals before window.start are discarded
    mid = bars.index[len(bars) // 2]
    early = bars.index[10]
    window = Window(start=mid, end=bars.index[-1])
    trades = evaluate_window(
        bars, _always_long_strategy, params={"stop_dist": 10.0},
        window=window, instrument=MNQ, risk_dollars=500.0, max_contracts=5,
        warmup_start=early,
    )
    # All scored trades must start at or after window.start
    for t in trades:
        assert t.entry_time >= mid


def test_evaluate_window_position_sizing(bars):
    window = Window(start=bars.index[0], end=bars.index[-1])
    # stop_dist=10 ticks * 2.0 point_value * contracts = risk
    # risk_dollars=200, stop_dist=10 -> 10*2 = $20/contract -> 10 contracts capped at 5
    trades = evaluate_window(
        bars, _always_long_strategy, params={"stop_dist": 10.0},
        window=window, instrument=MNQ, risk_dollars=200.0, max_contracts=5,
    )
    assert all(1 <= t.contracts <= 5 for t in trades)


def test_evaluate_window_net_pnl_includes_commission(bars):
    window = Window(start=bars.index[0], end=bars.index[-1])
    trades = evaluate_window(
        bars, _always_long_strategy, params={"stop_dist": 10.0},
        window=window, instrument=MNQ, risk_dollars=500.0, max_contracts=1,
    )
    for t in trades:
        assert abs(t.net_pnl - (t.gross_pnl - t.commission)) < 1e-6
        assert t.commission > 0


def test_evaluate_window_trades_do_not_span_sessions():
    # Bars spanning two calendar days — trades entered on day 1 must exit by day 1.
    multi_day_bars = make_bars(n=400, freq="5min", start="2020-01-02 09:30")
    # 400 bars × 5min = 2000min ≈ 33h, covering Jan 2 and Jan 3
    window = Window(start=multi_day_bars.index[0], end=multi_day_bars.index[-1])
    trades = evaluate_window(
        multi_day_bars, _always_long_strategy, params={"stop_dist": 10.0},
        window=window, instrument=MNQ, risk_dollars=500.0, max_contracts=5,
    )
    for t in trades:
        entry_date = t.entry_time.tz_convert("US/Eastern").date()
        exit_date = t.exit_time.tz_convert("US/Eastern").date()
        assert entry_date == exit_date, (
            f"Trade spanned sessions: entry {t.entry_time}, exit {t.exit_time}"
        )
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_evaluator.py -v
```

Expected: `ModuleNotFoundError: No module named 'eval_sim.evaluator'`

- [ ] **Step 3: Create `src/eval_sim/evaluator.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_evaluator.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Run full test suite to confirm no regressions**

```bash
pytest -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/eval_sim/evaluator.py tests/test_evaluator.py
git commit -m "feat: trade evaluator — signal to TradeResult with sizing and slippage"
```

---

## Task 7: WF Window Auto-Computation

**Files:**
- Create: `src/eval_sim/windows.py`
- Create: `tests/test_windows.py`

No train/test split. The full 3.5-year WF period is divided into `n_wf_windows` equal OOS scoring windows. Each window gets a `warmup_start` that is `warmup_bars` bars before its scoring start — the evaluator feeds those bars to the strategy for indicator initialization but discards any signals before `scoring_start`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_windows.py
import pandas as pd
import pytest
from eval_sim.windows import compute_windows, WFWindows, OOSWindow
from eval_sim.config import DataSplitConfig


_SPLIT = DataSplitConfig(
    wf_years=3.5,
    holdout_months=18,
    n_wf_windows=4,
    warmup_bars=500,
)


def test_compute_windows_returns_wfwindows():
    start = pd.Timestamp("2019-01-01", tz="US/Eastern")
    end = pd.Timestamp("2024-07-01", tz="US/Eastern")
    result = compute_windows(start, end, bars_index=None, config=_SPLIT)
    assert isinstance(result, WFWindows)


def test_holdout_is_last_18_months():
    start = pd.Timestamp("2019-01-01", tz="US/Eastern")
    end = pd.Timestamp("2024-07-01", tz="US/Eastern")
    result = compute_windows(start, end, bars_index=None, config=_SPLIT)
    holdout_months = (result.holdout.end - result.holdout.start).days / 30.44
    assert 16 <= holdout_months <= 20


def test_wf_window_count_matches_config():
    start = pd.Timestamp("2019-01-01", tz="US/Eastern")
    end = pd.Timestamp("2024-07-01", tz="US/Eastern")
    result = compute_windows(start, end, bars_index=None, config=_SPLIT)
    assert len(result.wf_windows) == _SPLIT.n_wf_windows


def test_wf_windows_cover_full_wf_period():
    start = pd.Timestamp("2019-01-01", tz="US/Eastern")
    end = pd.Timestamp("2024-07-01", tz="US/Eastern")
    result = compute_windows(start, end, bars_index=None, config=_SPLIT)
    assert result.wf_windows[0].start >= result.wf_start
    assert result.wf_windows[-1].end <= result.wf_end


def test_wf_windows_do_not_overlap():
    start = pd.Timestamp("2019-01-01", tz="US/Eastern")
    end = pd.Timestamp("2024-07-01", tz="US/Eastern")
    result = compute_windows(start, end, bars_index=None, config=_SPLIT)
    for i in range(len(result.wf_windows) - 1):
        assert result.wf_windows[i].end <= result.wf_windows[i + 1].start


def test_warmup_start_precedes_scoring_start():
    start = pd.Timestamp("2019-01-01", tz="US/Eastern")
    end = pd.Timestamp("2024-07-01", tz="US/Eastern")
    # Need a real bars_index so warmup_start can be computed in bar-count terms
    dates = pd.date_range(start, end, freq="5min", tz="US/Eastern")
    result = compute_windows(start, end, bars_index=dates, config=_SPLIT)
    for w in result.wf_windows:
        assert w.warmup_start <= w.start
    assert result.holdout.warmup_start <= result.holdout.start


def test_insufficient_data_raises():
    start = pd.Timestamp("2023-01-01", tz="US/Eastern")
    end = pd.Timestamp("2023-06-01", tz="US/Eastern")
    with pytest.raises(ValueError, match="insufficient"):
        compute_windows(start, end, bars_index=None, config=_SPLIT)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_windows.py -v
```

Expected: `ModuleNotFoundError: No module named 'eval_sim.windows'`

- [ ] **Step 3: Create `src/eval_sim/windows.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_windows.py -v
```

Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval_sim/windows.py tests/test_windows.py
git commit -m "feat: window computation — 4 equal OOS windows with warmup buffer, no train split"
```

---

## Task 8: Strategy Interface and Auto-Discovery

**Files:**
- Create: `src/eval_sim/strategy.py`
- Create: `src/eval_sim/strategies/example_strategy.py`
- Create: `tests/test_strategy.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_strategy.py
import pandas as pd
import pytest
from pathlib import Path
from eval_sim.strategy import load_strategy, StrategyLoadError, StrategyContract


def test_load_strategy_from_file(tmp_path):
    strat_file = tmp_path / "test_strat.py"
    strat_file.write_text(
        "import pandas as pd\n"
        "PARAM_RANGES = {'period': [5, 10, 20]}\n"
        "def generate_signals(bars, params):\n"
        "    return pd.DataFrame(columns=['entry_time','direction','entry','stop','target'])\n"
    )
    contract = load_strategy(strat_file)
    assert isinstance(contract, StrategyContract)
    assert "period" in contract.param_ranges
    assert callable(contract.generate_signals)


def test_load_strategy_missing_param_ranges(tmp_path):
    strat_file = tmp_path / "bad_strat.py"
    strat_file.write_text(
        "import pandas as pd\n"
        "def generate_signals(bars, params):\n"
        "    return pd.DataFrame()\n"
    )
    with pytest.raises(StrategyLoadError, match="PARAM_RANGES"):
        load_strategy(strat_file)


def test_load_strategy_missing_generate_signals(tmp_path):
    strat_file = tmp_path / "bad_strat2.py"
    strat_file.write_text("PARAM_RANGES = {'period': [5, 10]}\n")
    with pytest.raises(StrategyLoadError, match="generate_signals"):
        load_strategy(strat_file)


def test_load_strategy_file_not_found():
    with pytest.raises(StrategyLoadError, match="not found"):
        load_strategy(Path("/nonexistent/strategy.py"))


def test_strategy_contract_generate_signals_callable(tmp_path):
    strat_file = tmp_path / "test_strat.py"
    strat_file.write_text(
        "import pandas as pd\n"
        "PARAM_RANGES = {'k': [1, 2]}\n"
        "def generate_signals(bars, params):\n"
        "    return pd.DataFrame([{'entry_time': bars.index[0], 'direction': 'long', "
        "                         'entry': 100.0, 'stop': 90.0, 'target': 120.0}])\n"
    )
    from tests.conftest import make_bars
    contract = load_strategy(strat_file)
    bars = make_bars(10)
    result = contract.generate_signals(bars, {"k": 1})
    assert isinstance(result, pd.DataFrame)
    assert "entry_time" in result.columns
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_strategy.py -v
```

Expected: `ModuleNotFoundError: No module named 'eval_sim.strategy'`

- [ ] **Step 3: Create `src/eval_sim/strategy.py`**

```python
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
```

- [ ] **Step 4: Create the example strategy `src/eval_sim/strategies/example_strategy.py`**

This is the reference file for AI-written strategies. Keep it simple and well-commented.

```python
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
```

- [ ] **Step 5: Run all tests**

```bash
pytest tests/test_strategy.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/eval_sim/strategy.py src/eval_sim/strategies/example_strategy.py tests/test_strategy.py
git commit -m "feat: strategy interface — load_strategy, StrategyContract, example strategy"
```

---

## Task 9: Random Param Search

**Files:**
- Create: `src/eval_sim/search.py`
- Create: `tests/test_search.py`

Randomly samples `n_candidates` param combinations from `PARAM_RANGES`, evaluates each across WF OOS folds in parallel using `multiprocessing.Pool`, and returns the best params by mean sequential Combine eval pass rate.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_search.py
import pandas as pd
import pytest
from tests.conftest import make_bars
from eval_sim.search import random_search, sample_params, SearchResult
from eval_sim.windows import compute_windows, WFWindows
from eval_sim.config import MNQ, DataSplitConfig, SearchConfig


PARAM_RANGES = {"stop_dist": [5.0, 10.0, 15.0], "rr": [1.5, 2.0]}


def _simple_strategy(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    signals = []
    for ts, row in list(bars.iterrows())[:3]:
        signals.append({
            "entry_time": ts,
            "direction": "long",
            "entry": row["close"],
            "stop": row["close"] - params["stop_dist"],
            "target": row["close"] + params["stop_dist"] * params["rr"],
        })
    return pd.DataFrame(signals)


def _make_windows() -> WFWindows:
    # Span wide enough to satisfy the 6-month WF minimum check.
    # warmup_bars=0 so windows start exactly at scoring dates (no bar index needed).
    start = pd.Timestamp("2018-01-01", tz="US/Eastern")
    end = pd.Timestamp("2022-07-01", tz="US/Eastern")
    return compute_windows(
        start, end,
        bars_index=None,
        config=DataSplitConfig(wf_years=3.5, holdout_months=6, n_wf_windows=2, warmup_bars=0),
    )


def test_sample_params_returns_dict_from_ranges():
    import numpy as np
    rng = np.random.default_rng(42)
    params = sample_params(PARAM_RANGES, rng)
    assert set(params.keys()) == set(PARAM_RANGES.keys())
    assert params["stop_dist"] in PARAM_RANGES["stop_dist"]
    assert params["rr"] in PARAM_RANGES["rr"]


def test_sample_params_n_candidates_unique_enough():
    import numpy as np
    rng = np.random.default_rng(42)
    samples = [sample_params(PARAM_RANGES, rng) for _ in range(20)]
    unique = {tuple(sorted(s.items())) for s in samples}
    assert len(unique) >= 2  # should get more than one unique combo


def test_random_search_returns_search_result(bars):
    windows = _make_windows()
    result = random_search(
        bars=bars,
        strategy_fn=_simple_strategy,
        param_ranges=PARAM_RANGES,
        wf_windows=windows,
        instrument=MNQ,
        risk_dollars=500.0,
        max_contracts=3,
        config=SearchConfig(n_candidates=6, seed=42),
        n_workers=1,
    )
    assert isinstance(result, SearchResult)
    assert set(result.best_params.keys()) == set(PARAM_RANGES.keys())
    assert 0.0 <= result.best_score <= 1.0
    assert result.n_evaluated >= 1


def test_random_search_fast_mode_skips_search(bars):
    windows = _make_windows()
    result = random_search(
        bars=bars,
        strategy_fn=_simple_strategy,
        param_ranges=PARAM_RANGES,
        wf_windows=windows,
        instrument=MNQ,
        risk_dollars=500.0,
        max_contracts=3,
        config=SearchConfig(n_candidates=6, seed=42),
        n_workers=1,
        fast_mode=True,
    )
    assert result.n_evaluated == 0
    # Params should be midpoints of each range
    assert result.best_params["stop_dist"] == 10.0
    assert result.best_params["rr"] == 2.0


def test_random_search_best_params_in_ranges(bars):
    windows = _make_windows()
    result = random_search(
        bars=bars,
        strategy_fn=_simple_strategy,
        param_ranges=PARAM_RANGES,
        wf_windows=windows,
        instrument=MNQ,
        risk_dollars=500.0,
        max_contracts=3,
        config=SearchConfig(n_candidates=10, seed=42),
        n_workers=1,
    )
    for key, value in result.best_params.items():
        assert value in PARAM_RANGES[key], f"{key}={value} not in {PARAM_RANGES[key]}"
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_search.py -v
```

Expected: `ModuleNotFoundError: No module named 'eval_sim.search'`

- [ ] **Step 3: Create `src/eval_sim/search.py`**

```python
from __future__ import annotations
import multiprocessing
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd

from eval_sim.config import Instrument, SearchConfig, SEARCH, TOPSTEP_50K
from eval_sim.evaluator import evaluate_window
from eval_sim.topstep import simulate_seq_evals
from eval_sim.windows import WFWindows
from eval_sim.trades import TradeResult


@dataclass(frozen=True)
class SearchResult:
    best_params: dict
    best_score: float   # mean seq eval pass rate across OOS folds
    n_evaluated: int


def sample_params(param_ranges: dict[str, list], rng: np.random.Generator) -> dict:
    """Sample one random combination from param_ranges."""
    return {key: rng.choice(values).item() for key, values in param_ranges.items()}


def _midpoint_params(param_ranges: dict[str, list]) -> dict:
    """Return midpoint value for each param range (used in --fast mode)."""
    return {key: values[len(values) // 2] for key, values in param_ranges.items()}


def _score_candidate(
    bars: pd.DataFrame,
    strategy_fn: Callable,
    params: dict,
    wf_windows: WFWindows,
    instrument: Instrument,
    risk_dollars: float,
    max_contracts: int,
) -> float:
    """Score one param set by mean seq eval pass rate across all WF OOS windows."""
    window_scores: list[float] = []
    for w in wf_windows.wf_windows:
        from eval_sim.evaluator import Window
        scoring_window = Window(start=w.start, end=w.end)
        trades = evaluate_window(
            bars, strategy_fn, params, scoring_window,
            instrument, risk_dollars, max_contracts,
            warmup_start=w.warmup_start,
        )
        seq = simulate_seq_evals(trades, TOPSTEP_50K)
        window_scores.append(seq.pass_rate)
    return float(np.mean(window_scores)) if window_scores else 0.0


def _worker(args: tuple) -> tuple[dict, float]:
    """Multiprocessing worker: (bars, fn, params, windows, instrument, risk, max_c) -> (params, score)"""
    bars, strategy_fn, params, wf_windows, instrument, risk_dollars, max_contracts = args
    score = _score_candidate(bars, strategy_fn, params, wf_windows, instrument, risk_dollars, max_contracts)
    return params, score


def random_search(
    bars: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame, dict], pd.DataFrame],
    param_ranges: dict[str, list],
    wf_windows: WFWindows,
    instrument: Instrument,
    risk_dollars: float,
    max_contracts: int,
    config: SearchConfig = SEARCH,
    n_workers: int | None = None,
    fast_mode: bool = False,
) -> SearchResult:
    """
    Randomly sample n_candidates param combinations and score each across WF OOS folds.
    Returns the best params by mean sequential Combine eval pass rate.

    fast_mode=True: skip search, return midpoint params with score=0 and n_evaluated=0.
    """
    if fast_mode:
        return SearchResult(
            best_params=_midpoint_params(param_ranges),
            best_score=0.0,
            n_evaluated=0,
        )

    rng = np.random.default_rng(config.seed)
    candidates = [sample_params(param_ranges, rng) for _ in range(config.n_candidates)]

    worker_args = [
        (bars, strategy_fn, params, wf_windows, instrument, risk_dollars, max_contracts)
        for params in candidates
    ]

    if n_workers == 1:
        results = [_worker(args) for args in worker_args]
    else:
        # On Windows, multiprocessing uses "spawn" (not "fork"), which requires
        # the entry point to be protected by `if __name__ == "__main__"`.
        # We use get_context("spawn") explicitly so behaviour is consistent
        # across platforms and freeze_support() in cli.py covers the frozen case.
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(_worker, worker_args)

    best_params, best_score = max(results, key=lambda x: x[1])
    return SearchResult(
        best_params=best_params,
        best_score=best_score,
        n_evaluated=len(candidates),
    )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_search.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Run full test suite**

```bash
pytest -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/eval_sim/search.py tests/test_search.py
git commit -m "feat: random param search with multiprocessing and fast mode"
```

---

## Final: Smoke Test Plan 1

- [ ] **Run full test suite one last time**

```bash
pytest -v --tb=short
```

Expected: all tests PASS with output like:
```
tests/test_config.py .....          5 passed
tests/test_data.py .....            5 passed
tests/test_evaluator.py ......      6 passed
tests/test_search.py .....          5 passed
tests/test_strategy.py .....        5 passed
tests/test_topstep.py ..........   10 passed
tests/test_trades.py ..             2 passed
tests/test_windows.py .......       7 passed
```

- [ ] **Verify example strategy loads cleanly**

```bash
python -c "
from pathlib import Path
from eval_sim.strategy import load_strategy
c = load_strategy(Path('src/eval_sim/strategies/example_strategy.py'))
print('Strategy loaded:', c.name, '| Param keys:', list(c.param_ranges.keys()))
"
```

Expected: prints strategy name and param keys, no errors.

- [ ] **Final commit**

```bash
git add -A
git commit -m "feat: Plan 1 complete — core infrastructure ready for Plan 2 pipeline stages"
```

---

*Plan 2 builds on these primitives: continuous eval sim, sensitivity sweep, Monte Carlo, sizing optimizer, verdict, and CLI.*
