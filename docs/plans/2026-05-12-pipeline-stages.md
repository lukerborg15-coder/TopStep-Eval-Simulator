# TopStep Eval Sim — Plan 2: Pipeline Stages

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pipeline stages on top of Plan 1's core: continuous eval simulation, sensitivity sweep, Monte Carlo, sizing optimizer, verdict, and the CLI. Produces a complete, runnable `eval-sim` command.

**Architecture:** Each stage is a focused module that consumes `list[TradeResult]` or `WFWindows` from Plan 1 primitives and produces a typed result dataclass. The CLI orchestrates stages in order, writes JSON output. No imports from the Topstep pipeline project.

**Tech Stack:** Python 3.11+, pandas, numpy, multiprocessing (stdlib), pytest

**Prerequisite:** Plan 1 complete — all tests passing.

---

## File Map

| File | Responsibility |
|------|----------------|
| `src/eval_sim/continuous_eval.py` | `run_continuous_eval(trades, rules) -> ContinuousEvalResult` — primary pipeline output |
| `src/eval_sim/monte_carlo.py` | Block-bootstrap MC on eval sim trades → confidence intervals |
| `src/eval_sim/sensitivity.py` | Perturb params around selected set → cliff detection |
| `src/eval_sim/sizing.py` | WF-trained speed sizing optimizer → holdout validation + fixed-risk comparison |
| `src/eval_sim/verdict.py` | `compute_verdict(...)  -> VerdictResult` |
| `src/eval_sim/cli.py` | Argparse entry point — orchestrates all stages, writes JSON |
| `tests/test_continuous_eval.py` | Pass rate, attempt count, edge cases |
| `tests/test_monte_carlo.py` | Block size, CI bounds, seed reproducibility |
| `tests/test_sensitivity.py` | Cliff detection, flat detection, empty param ranges |
| `tests/test_sizing.py` | WF sizing, holdout validation, fixed-risk comparison |
| `tests/test_verdict.py` | Pass/reject/warn thresholds |
| `tests/test_cli.py` | End-to-end smoke test with example strategy |

---

## Task 1: Continuous Eval Simulation

**Files:**
- Create: `src/eval_sim/continuous_eval.py`
- Create: `tests/test_continuous_eval.py`

This is the primary output of the pipeline. It wraps `simulate_seq_evals` from Plan 1 and adds reporting metrics.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_continuous_eval.py
import pandas as pd
import pytest
from tests.conftest import make_trade
from eval_sim.continuous_eval import run_continuous_eval, ContinuousEvalResult
from eval_sim.config import TOPSTEP_50K


def _trade(pnl: float, day: str) -> object:
    return make_trade(net_pnl=pnl, entry_time=f"{day} 09:35")


def test_run_continuous_eval_returns_result():
    trades = [_trade(150.0, f"2020-01-{2 + i:02d}") for i in range(40)]
    result = run_continuous_eval(trades, TOPSTEP_50K)
    assert isinstance(result, ContinuousEvalResult)


def test_run_continuous_eval_empty_trades():
    result = run_continuous_eval([], TOPSTEP_50K)
    assert result.attempts == 0
    assert result.passes == 0
    assert result.pass_rate == 0.0


def test_run_continuous_eval_losing_trades():
    # All losing trades — no passes
    trades = [_trade(-300.0, f"2020-01-{2 + i:02d}") for i in range(20)]
    result = run_continuous_eval(trades, TOPSTEP_50K)
    assert result.passes == 0
    assert result.attempts >= 1


def test_run_continuous_eval_winning_trades():
    # 31 winning trades of $100 = $3100 profit > $3000 target, should pass
    trades = [_trade(100.0, f"2020-01-{2 + i:02d}") for i in range(31)]
    result = run_continuous_eval(trades, TOPSTEP_50K)
    assert result.passes >= 1
    assert result.pass_rate > 0.0


def test_run_continuous_eval_pass_rate_in_range():
    trades = [_trade(80.0, f"2020-01-{2 + i:02d}") for i in range(50)]
    result = run_continuous_eval(trades, TOPSTEP_50K)
    assert 0.0 <= result.pass_rate <= 1.0


def test_run_continuous_eval_has_worst_drawdown():
    trades = [_trade(-200.0, f"2020-01-{2 + i:02d}") for i in range(10)]
    result = run_continuous_eval(trades, TOPSTEP_50K)
    assert result.worst_attempt_drawdown >= 0.0
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_continuous_eval.py -v
```

Expected: `ModuleNotFoundError: No module named 'eval_sim.continuous_eval'`

- [ ] **Step 3: Create `src/eval_sim/continuous_eval.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from eval_sim.config import TopstepRules, TOPSTEP_50K
from eval_sim.topstep import simulate_seq_evals, simulate_topstep, _group_by_day
from eval_sim.trades import TradeResult


@dataclass(frozen=True)
class ContinuousEvalResult:
    passes: int
    attempts: int
    pass_rate: float               # 0.0–1.0
    worst_attempt_drawdown: float
    mean_days_per_attempt: float
    median_days_per_attempt: float  # median of per-attempt trading day counts
    total_trading_days: int
    is_empty: bool                  # True if no trades — flag in output, don't score


def run_continuous_eval(
    trades: list[TradeResult],
    rules: TopstepRules = TOPSTEP_50K,
) -> ContinuousEvalResult:
    """
    Chain sequential Combine evaluation attempts across all trades.
    Primary output of the pipeline — measures how reliably the strategy
    passes evaluations when run continuously.

    is_empty=True means the strategy generated no signals for this window.
    This should be flagged in output, not treated as a pass_rate=0 score.
    """
    if not trades:
        return ContinuousEvalResult(
            passes=0,
            attempts=0,
            pass_rate=0.0,
            worst_attempt_drawdown=0.0,
            mean_days_per_attempt=0.0,
            median_days_per_attempt=0.0,
            total_trading_days=0,
            is_empty=True,
        )

    seq = simulate_seq_evals(trades, rules)

    # Worst drawdown: re-run each attempt to collect per-attempt drawdown.
    # simulate_seq_evals already consumed the trades cleanly; we replay here
    # only for reporting purposes.
    worst_dd = 0.0
    remaining = list(trades)
    for days_consumed in seq.attempt_days:
        result = simulate_topstep(remaining, rules)
        worst_dd = max(worst_dd, result.max_drawdown_seen)

        by_day = _group_by_day(remaining)
        sorted_days = sorted(by_day.keys())
        if days_consumed >= len(sorted_days):
            break
        next_day = sorted_days[days_consumed]
        remaining = [
            t for t in remaining
            if t.exit_time.tz_convert("US/Eastern").date().isoformat() >= next_day
        ]

    days = list(seq.attempt_days)
    mean_days = sum(days) / len(days) if days else 0.0
    median_days = float(sorted(days)[len(days) // 2]) if days else 0.0
    total_days = sum(days)

    return ContinuousEvalResult(
        passes=seq.passes,
        attempts=seq.attempts,
        pass_rate=seq.pass_rate,
        worst_attempt_drawdown=worst_dd,
        mean_days_per_attempt=mean_days,
        median_days_per_attempt=median_days,
        total_trading_days=total_days,
        is_empty=False,
    )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_continuous_eval.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval_sim/continuous_eval.py tests/test_continuous_eval.py
git commit -m "feat: continuous eval simulation — primary pipeline output"
```

---

## Task 2: Monte Carlo

**Files:**
- Create: `src/eval_sim/monte_carlo.py`
- Create: `tests/test_monte_carlo.py`

Block-bootstrap on holdout trades: resample trade order N times, run `run_continuous_eval` on each permutation, compute confidence intervals on pass rate.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_monte_carlo.py
import pytest
from tests.conftest import make_trade
from eval_sim.monte_carlo import run_monte_carlo, MCResult
from eval_sim.config import TOPSTEP_50K


def _trades(n: int = 40) -> list:
    return [make_trade(net_pnl=80.0, entry_time=f"2020-01-{2 + (i % 28):02d} 09:35") for i in range(n)]


def test_run_monte_carlo_returns_result():
    result = run_monte_carlo(_trades(), TOPSTEP_50K, n=50, block_size=5, seed=42)
    assert isinstance(result, MCResult)


def test_run_monte_carlo_pass_rate_in_range():
    result = run_monte_carlo(_trades(), TOPSTEP_50K, n=50, block_size=5, seed=42)
    assert 0.0 <= result.pass_rate_p05 <= result.pass_rate_median <= 1.0


def test_run_monte_carlo_seed_reproducible():
    r1 = run_monte_carlo(_trades(), TOPSTEP_50K, n=50, block_size=5, seed=7)
    r2 = run_monte_carlo(_trades(), TOPSTEP_50K, n=50, block_size=5, seed=7)
    assert r1.pass_rate_median == r2.pass_rate_median
    assert r1.pass_rate_p05 == r2.pass_rate_p05


def test_run_monte_carlo_different_seeds_differ():
    r1 = run_monte_carlo(_trades(60), TOPSTEP_50K, n=100, block_size=5, seed=1)
    r2 = run_monte_carlo(_trades(60), TOPSTEP_50K, n=100, block_size=5, seed=2)
    # Not guaranteed but should differ with different seeds and enough permutations
    assert r1.pass_rate_median != r2.pass_rate_median or r1.pass_rate_p05 != r2.pass_rate_p05


def test_run_monte_carlo_empty_trades():
    result = run_monte_carlo([], TOPSTEP_50K, n=20, block_size=5, seed=42)
    assert result.pass_rate_median == 0.0
    assert result.pass_rate_p05 == 0.0
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_monte_carlo.py -v
```

Expected: `ModuleNotFoundError: No module named 'eval_sim.monte_carlo'`

- [ ] **Step 3: Create `src/eval_sim/monte_carlo.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from eval_sim.config import TopstepRules, TOPSTEP_50K
from eval_sim.continuous_eval import run_continuous_eval
from eval_sim.topstep import _group_by_day
from eval_sim.trades import TradeResult


@dataclass(frozen=True)
class MCResult:
    n_permutations: int
    pass_rate_median: float
    pass_rate_p05: float
    pass_rate_p95: float
    worst_drawdown_median: float


def _block_bootstrap(
    trades: list[TradeResult],
    block_size: int,
    rng: np.random.Generator,
) -> list[TradeResult]:
    """Resample trade-days in blocks, preserving within-day order."""
    by_day = _group_by_day(trades)
    days = sorted(by_day.keys())
    if not days:
        return []

    n_blocks = max(1, len(days) // block_size)
    sampled_days: list[str] = []
    for _ in range(n_blocks):
        start_idx = rng.integers(0, max(1, len(days) - block_size + 1))
        sampled_days.extend(days[start_idx: start_idx + block_size])

    resampled: list[TradeResult] = []
    for day in sampled_days:
        resampled.extend(by_day.get(day, []))
    return resampled


def run_monte_carlo(
    trades: list[TradeResult],
    rules: TopstepRules = TOPSTEP_50K,
    n: int = 500,
    block_size: int = 5,
    seed: int = 42,
    ci_pct: float = 5.0,
) -> MCResult:
    """
    Block-bootstrap Monte Carlo on holdout trades.
    Resamples trade order N times and reports confidence intervals on pass rate.
    """
    if not trades:
        return MCResult(
            n_permutations=n,
            pass_rate_median=0.0,
            pass_rate_p05=0.0,
            pass_rate_p95=0.0,
            worst_drawdown_median=0.0,
        )

    rng = np.random.default_rng(seed)
    pass_rates: list[float] = []
    worst_drawdowns: list[float] = []

    for _ in range(n):
        resampled = _block_bootstrap(trades, block_size, rng)
        result = run_continuous_eval(resampled, rules)
        pass_rates.append(result.pass_rate)
        worst_drawdowns.append(result.worst_attempt_drawdown)

    return MCResult(
        n_permutations=n,
        pass_rate_median=float(np.median(pass_rates)),
        pass_rate_p05=float(np.percentile(pass_rates, ci_pct)),
        pass_rate_p95=float(np.percentile(pass_rates, 100 - ci_pct)),
        worst_drawdown_median=float(np.median(worst_drawdowns)),
    )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_monte_carlo.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval_sim/monte_carlo.py tests/test_monte_carlo.py
git commit -m "feat: block-bootstrap Monte Carlo for holdout confidence intervals"
```

---

## Task 3: Sensitivity Sweep

**Files:**
- Create: `src/eval_sim/sensitivity.py`
- Create: `tests/test_sensitivity.py`

For each param in `PARAM_RANGES`, step one value up and one value down from the selected value, re-evaluate on the WF development window, report pass rate at each neighbor. If any neighbor drops more than `cliff_threshold_pct` below the selected params' pass rate, flag as a cliff.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sensitivity.py
import pandas as pd
import pytest
from tests.conftest import make_bars
from eval_sim.sensitivity import run_sensitivity, SensitivityResult
from eval_sim.evaluator import Window
from eval_sim.config import MNQ


PARAM_RANGES = {"stop_dist": [5.0, 10.0, 15.0, 20.0], "rr": [1.5, 2.0, 2.5]}
SELECTED = {"stop_dist": 10.0, "rr": 2.0}


def _strategy(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    import pandas as _pd
    signals = []
    for ts, row in list(bars.iterrows())[:5]:
        signals.append({
            "entry_time": ts,
            "direction": "long",
            "entry": row["close"],
            "stop": row["close"] - params["stop_dist"],
            "target": row["close"] + params["stop_dist"] * params["rr"],
        })
    return _pd.DataFrame(signals)


def test_run_sensitivity_returns_result(bars):
    window = Window(start=bars.index[0], end=bars.index[-1])
    result = run_sensitivity(bars, _strategy, SELECTED, PARAM_RANGES, window, MNQ,
                             risk_dollars=500.0, max_contracts=3)
    assert isinstance(result, SensitivityResult)


def test_run_sensitivity_has_per_param_results(bars):
    window = Window(start=bars.index[0], end=bars.index[-1])
    result = run_sensitivity(bars, _strategy, SELECTED, PARAM_RANGES, window, MNQ,
                             risk_dollars=500.0, max_contracts=3)
    assert "stop_dist" in result.param_results
    assert "rr" in result.param_results


def test_run_sensitivity_is_cliff_is_bool(bars):
    window = Window(start=bars.index[0], end=bars.index[-1])
    result = run_sensitivity(bars, _strategy, SELECTED, PARAM_RANGES, window, MNQ,
                             risk_dollars=500.0, max_contracts=3)
    assert isinstance(result.is_cliff, bool)


def test_run_sensitivity_single_value_param(bars):
    window = Window(start=bars.index[0], end=bars.index[-1])
    # Param with only one value — no neighbors, no cliff possible
    single_ranges = {"stop_dist": [10.0], "rr": [2.0]}
    result = run_sensitivity(bars, _strategy, SELECTED, single_ranges, window, MNQ,
                             risk_dollars=500.0, max_contracts=3)
    assert result.is_cliff is False
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_sensitivity.py -v
```

Expected: `ModuleNotFoundError: No module named 'eval_sim.sensitivity'`

- [ ] **Step 3: Create `src/eval_sim/sensitivity.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import pandas as pd

from eval_sim.config import Instrument, TopstepRules, TOPSTEP_50K
from eval_sim.evaluator import Window, evaluate_window
from eval_sim.topstep import simulate_seq_evals
from eval_sim.trades import TradeResult


@dataclass(frozen=True)
class ParamSensitivity:
    selected_value: object
    neighbor_results: dict  # {value: pass_rate}


@dataclass(frozen=True)
class SensitivityResult:
    selected_pass_rate: float
    param_results: dict[str, ParamSensitivity]
    is_cliff: bool
    cliff_params: tuple[str, ...]


def _pass_rate_for_params(
    bars: pd.DataFrame,
    strategy_fn: Callable,
    params: dict,
    window: Window,
    instrument: Instrument,
    risk_dollars: float,
    max_contracts: int,
    rules: TopstepRules,
    warmup_start: pd.Timestamp | None,
) -> float:
    trades = evaluate_window(
        bars, strategy_fn, params, window, instrument, risk_dollars, max_contracts,
        warmup_start=warmup_start,
    )
    seq = simulate_seq_evals(trades, rules)
    return seq.pass_rate


def run_sensitivity(
    bars: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame, dict], pd.DataFrame],
    selected_params: dict,
    param_ranges: dict[str, list],
    window: Window,
    instrument: Instrument,
    risk_dollars: float,
    max_contracts: int,
    rules: TopstepRules = TOPSTEP_50K,
    cliff_threshold_pct: float = 20.0,
    warmup_start: pd.Timestamp | None = None,
) -> SensitivityResult:
    """
    For each param, evaluate pass rate at neighboring values (one step up/down
    from selected). Flag as cliff if any neighbor drops >cliff_threshold_pct
    percentage points below selected.
    """
    selected_rate = _pass_rate_for_params(
        bars, strategy_fn, selected_params, window, instrument, risk_dollars, max_contracts, rules, warmup_start
    )

    param_results: dict[str, ParamSensitivity] = {}
    cliff_params: list[str] = []

    for param_key, all_values in param_ranges.items():
        selected_val = selected_params[param_key]
        if selected_val not in all_values:
            all_values = sorted(all_values + [selected_val])

        idx = all_values.index(selected_val)
        neighbor_values = []
        if idx > 0:
            neighbor_values.append(all_values[idx - 1])
        if idx < len(all_values) - 1:
            neighbor_values.append(all_values[idx + 1])

        neighbor_results: dict = {}
        is_cliff_for_param = False

        for neighbor_val in neighbor_values:
            neighbor_params = {**selected_params, param_key: neighbor_val}
            rate = _pass_rate_for_params(
                bars, strategy_fn, neighbor_params, window, instrument, risk_dollars, max_contracts, rules, warmup_start
            )
            neighbor_results[neighbor_val] = rate
            drop_pct = (selected_rate - rate) * 100
            if drop_pct > cliff_threshold_pct:
                is_cliff_for_param = True

        param_results[param_key] = ParamSensitivity(
            selected_value=selected_val,
            neighbor_results=neighbor_results,
        )
        if is_cliff_for_param:
            cliff_params.append(param_key)

    return SensitivityResult(
        selected_pass_rate=selected_rate,
        param_results=param_results,
        is_cliff=len(cliff_params) > 0,
        cliff_params=tuple(cliff_params),
    )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_sensitivity.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval_sim/sensitivity.py tests/test_sensitivity.py
git commit -m "feat: sensitivity sweep — cliff detection around selected params"
```

---

## Task 4: Sizing Optimizer

**Files:**
- Create: `src/eval_sim/sizing.py`
- Create: `tests/test_sizing.py`

Trains on WF fold train/OOS pairs to find the risk level with the best pass rate. Validates that optimal sizing on the holdout. Compares against user-defined fixed risk.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sizing.py
import pandas as pd
import pytest
from tests.conftest import make_bars
from eval_sim.sizing import run_sizing_optimizer, SizingResult
from eval_sim.windows import compute_windows
from eval_sim.config import MNQ, DataSplitConfig


RISK_GRID = [200.0, 350.0, 500.0, 750.0, 1000.0]
SELECTED_PARAMS = {"stop_dist": 10.0, "rr": 2.0}


def _strategy(bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    signals = []
    for ts, row in list(bars.iterrows())[:3]:
        signals.append({
            "entry_time": ts, "direction": "long",
            "entry": row["close"],
            "stop": row["close"] - params["stop_dist"],
            "target": row["close"] + params["stop_dist"] * params["rr"],
        })
    import pandas as _pd
    return _pd.DataFrame(signals)


def _windows():
    start = pd.Timestamp("2020-01-01", tz="US/Eastern")
    end = pd.Timestamp("2023-07-01", tz="US/Eastern")
    return compute_windows(
        start, end,
        config=DataSplitConfig(wf_years=2.5, holdout_months=6, min_wf_folds=2, max_wf_folds=2, target_oos_months=6),
    )


RISK_GRID = [100.0, 200.0, 300.0, 400.0, 500.0]  # subset for test speed


def test_run_sizing_optimizer_returns_result(bars):
    windows = _windows()
    result = run_sizing_optimizer(
        bars=bars, strategy_fn=_strategy, selected_params=SELECTED_PARAMS,
        wf_windows=windows, instrument=MNQ, max_contracts=5,
        fixed_risk_dollars=None, risk_grid=RISK_GRID,
    )
    assert isinstance(result, SizingResult)


def test_optimal_risk_in_grid(bars):
    windows = _windows()
    result = run_sizing_optimizer(
        bars=bars, strategy_fn=_strategy, selected_params=SELECTED_PARAMS,
        wf_windows=windows, instrument=MNQ, max_contracts=5,
        fixed_risk_dollars=None, risk_grid=RISK_GRID,
    )
    assert result.optimal_risk_dollars in RISK_GRID


def test_all_levels_covers_full_grid(bars):
    windows = _windows()
    result = run_sizing_optimizer(
        bars=bars, strategy_fn=_strategy, selected_params=SELECTED_PARAMS,
        wf_windows=windows, instrument=MNQ, max_contracts=5,
        fixed_risk_dollars=None, risk_grid=RISK_GRID,
    )
    assert len(result.all_levels) == len(RISK_GRID)
    tested_risks = {l.risk_dollars for l in result.all_levels}
    assert tested_risks == set(RISK_GRID)


def test_fixed_risk_comparison_included(bars):
    windows = _windows()
    result = run_sizing_optimizer(
        bars=bars, strategy_fn=_strategy, selected_params=SELECTED_PARAMS,
        wf_windows=windows, instrument=MNQ, max_contracts=5,
        fixed_risk_dollars=500.0, risk_grid=RISK_GRID,
    )
    assert result.fixed_risk_result is not None
    assert result.fixed_risk_result.risk_dollars == 500.0
    assert 0.0 <= result.fixed_risk_result.pass_rate <= 1.0


def test_no_fixed_risk_result_is_none(bars):
    windows = _windows()
    result = run_sizing_optimizer(
        bars=bars, strategy_fn=_strategy, selected_params=SELECTED_PARAMS,
        wf_windows=windows, instrument=MNQ, max_contracts=5,
        fixed_risk_dollars=None, risk_grid=RISK_GRID,
    )
    assert result.fixed_risk_result is None
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_sizing.py -v
```

Expected: `ModuleNotFoundError: No module named 'eval_sim.sizing'`

- [ ] **Step 3: Create `src/eval_sim/sizing.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import pandas as pd

from eval_sim.config import Instrument, TopstepRules, TOPSTEP_50K
from eval_sim.continuous_eval import run_continuous_eval
from eval_sim.evaluator import evaluate_window
from eval_sim.windows import WFWindows


HOLDOUT_RISK_GRID = [float(r) for r in range(100, 1100, 100)]  # $100–$1000 in $100 steps


@dataclass(frozen=True)
class RiskLevelResult:
    risk_dollars: float
    pass_rate: float
    mean_days_per_attempt: float
    expected_days_per_success: float  # mean_days_per_attempt / pass_rate; lower = faster funded


@dataclass(frozen=True)
class SizingResult:
    optimal_risk_dollars: float
    optimal_expected_days: float      # expected days per successful pass at optimal risk
    optimal_pass_rate: float
    all_levels: tuple[RiskLevelResult, ...]
    fixed_risk_result: RiskLevelResult | None


def _eval_risk_on_holdout(
    bars: pd.DataFrame,
    strategy_fn: Callable,
    params: dict,
    wf_windows: WFWindows,
    instrument: Instrument,
    risk_dollars: float,
    max_contracts: int,
    rules: TopstepRules,
) -> RiskLevelResult:
    from eval_sim.evaluator import Window
    holdout_window = Window(start=wf_windows.holdout.start, end=wf_windows.holdout.end)
    trades = evaluate_window(
        bars, strategy_fn, params, holdout_window,
        instrument, risk_dollars, max_contracts,
        warmup_start=wf_windows.holdout.warmup_start,
    )
    result = run_continuous_eval(trades, rules)
    expected = (
        result.mean_days_per_attempt / result.pass_rate
        if result.pass_rate > 0 else float("inf")
    )
    return RiskLevelResult(
        risk_dollars=risk_dollars,
        pass_rate=result.pass_rate,
        mean_days_per_attempt=result.mean_days_per_attempt,
        expected_days_per_success=expected,
    )


def run_sizing_optimizer(
    bars: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame, dict], pd.DataFrame],
    selected_params: dict,
    wf_windows: WFWindows,
    instrument: Instrument,
    max_contracts: int,
    fixed_risk_dollars: float | None,
    risk_grid: list[float] = HOLDOUT_RISK_GRID,
    min_pass_rate: float = 0.40,
    rules: TopstepRules = TOPSTEP_50K,
) -> SizingResult:
    """
    Test each risk level in risk_grid ($100–$1000 in $100 steps) on the holdout.

    Optimal = lowest expected_days_per_success (mean_days_per_attempt / pass_rate)
    among levels where pass_rate >= min_pass_rate. If none meet the floor, fall
    back to the highest pass_rate level.

    expected_days_per_success is the right metric here: it captures how quickly
    you'd expect to get funded, penalising both low pass rates and long attempts
    in a single number.
    """
    all_levels: list[RiskLevelResult] = []
    for risk in risk_grid:
        level = _eval_risk_on_holdout(
            bars, strategy_fn, selected_params, wf_windows,
            instrument, risk, max_contracts, rules,
        )
        all_levels.append(level)

    viable = [l for l in all_levels if l.pass_rate >= min_pass_rate]
    best = (
        min(viable, key=lambda l: l.expected_days_per_success)
        if viable
        else max(all_levels, key=lambda l: l.pass_rate)
    )

    fixed_result: RiskLevelResult | None = None
    if fixed_risk_dollars is not None:
        fixed_result = _eval_risk_on_holdout(
            bars, strategy_fn, selected_params, wf_windows,
            instrument, fixed_risk_dollars, max_contracts, rules,
        )

    return SizingResult(
        optimal_risk_dollars=best.risk_dollars,
        optimal_expected_days=best.expected_days_per_success,
        optimal_pass_rate=best.pass_rate,
        all_levels=tuple(all_levels),
        fixed_risk_result=fixed_result,
    )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_sizing.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval_sim/sizing.py tests/test_sizing.py
git commit -m "feat: sizing optimizer — WF-trained, holdout-validated, fixed-risk comparison"
```

---

## Task 5: Verdict

**Files:**
- Create: `src/eval_sim/verdict.py`
- Create: `tests/test_verdict.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_verdict.py
from eval_sim.verdict import compute_verdict, VerdictResult, VerdictThresholds


_T = VerdictThresholds(
    reject_pass_rate=0.30,
    reject_max_dd=1800.0,
    ready_pass_rate=0.60,
    ready_max_dd=1200.0,
)


def test_verdict_reject_low_pass_rate():
    result = compute_verdict(
        pass_rate=0.20, mc_pass_rate_p05=0.15, worst_drawdown=500.0,
        sensitivity_is_cliff=False, thresholds=_T,
    )
    assert result.verdict == "REJECT"
    assert any("pass_rate" in r for r in result.reject_reasons)


def test_verdict_reject_high_drawdown():
    result = compute_verdict(
        pass_rate=0.70, mc_pass_rate_p05=0.55, worst_drawdown=1900.0,
        sensitivity_is_cliff=False, thresholds=_T,
    )
    assert result.verdict == "REJECT"
    assert any("drawdown" in r for r in result.reject_reasons)


def test_verdict_combine_ready():
    result = compute_verdict(
        pass_rate=0.70, mc_pass_rate_p05=0.60, worst_drawdown=900.0,
        sensitivity_is_cliff=False, thresholds=_T,
    )
    assert result.verdict == "COMBINE-READY"
    assert not result.reject_reasons


def test_verdict_marginal_with_cliff_warn():
    result = compute_verdict(
        pass_rate=0.65, mc_pass_rate_p05=0.50, worst_drawdown=800.0,
        sensitivity_is_cliff=True, thresholds=_T,
    )
    assert any("cliff" in w for w in result.warn_reasons)


def test_verdict_marginal_between_thresholds():
    result = compute_verdict(
        pass_rate=0.45, mc_pass_rate_p05=0.35, worst_drawdown=700.0,
        sensitivity_is_cliff=False, thresholds=_T,
    )
    assert result.verdict == "MARGINAL"
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_verdict.py -v
```

Expected: `ModuleNotFoundError: No module named 'eval_sim.verdict'`

- [ ] **Step 3: Create `src/eval_sim/verdict.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class VerdictThresholds:
    reject_pass_rate: float = 0.30      # reject if pass_rate below this
    reject_max_dd: float = 1800.0       # reject if worst drawdown above this
    ready_pass_rate: float = 0.60       # combine-ready if pass_rate at or above this
    ready_max_dd: float = 1200.0        # combine-ready if worst drawdown at or below this


DEFAULT_THRESHOLDS = VerdictThresholds()


@dataclass(frozen=True)
class VerdictResult:
    verdict: str                        # "COMBINE-READY" | "MARGINAL" | "REJECT"
    reject_reasons: tuple[str, ...]
    warn_reasons: tuple[str, ...]


def compute_verdict(
    pass_rate: float,
    mc_pass_rate_p05: float,
    worst_drawdown: float,
    sensitivity_is_cliff: bool,
    thresholds: VerdictThresholds = DEFAULT_THRESHOLDS,
) -> VerdictResult:
    reject_reasons: list[str] = []
    warn_reasons: list[str] = []

    if pass_rate < thresholds.reject_pass_rate:
        reject_reasons.append(f"pass_rate {pass_rate:.1%} below reject floor {thresholds.reject_pass_rate:.1%}")
    if worst_drawdown > thresholds.reject_max_dd:
        reject_reasons.append(f"drawdown ${worst_drawdown:.0f} exceeds reject limit ${thresholds.reject_max_dd:.0f}")

    if sensitivity_is_cliff:
        warn_reasons.append("sensitivity cliff detected — params are on edge of performance region")

    if reject_reasons:
        verdict = "REJECT"
    elif pass_rate >= thresholds.ready_pass_rate and worst_drawdown <= thresholds.ready_max_dd:
        verdict = "COMBINE-READY"
    else:
        verdict = "MARGINAL"

    return VerdictResult(
        verdict=verdict,
        reject_reasons=tuple(reject_reasons),
        warn_reasons=tuple(warn_reasons),
    )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_verdict.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval_sim/verdict.py tests/test_verdict.py
git commit -m "feat: verdict — COMBINE-READY / MARGINAL / REJECT with threshold config"
```

---

## Task 6: CLI

**Files:**
- Create: `src/eval_sim/cli.py`
- Create: `tests/test_cli.py`

Orchestrates all stages. Writes a JSON result file. Prints a summary table.

- [ ] **Step 1: Write failing end-to-end test**

```python
# tests/test_cli.py
import subprocess
import sys
import json
from pathlib import Path
import pytest


DATA_DIR = Path(__file__).parent.parent / "Data"
STRATEGY = Path(__file__).parent.parent / "src" / "eval_sim" / "strategies" / "example_strategy.py"


@pytest.mark.skipif(not DATA_DIR.exists(), reason="Data/ directory not present")
def test_cli_fast_mode_runs_end_to_end(tmp_path):
    result = subprocess.run(
        [
            sys.executable, "-m", "eval_sim.cli",
            "--strategy", str(STRATEGY),
            "--instrument", "mnq",
            "--timeframe", "5min",
            "--fast",
            "--output-dir", str(tmp_path),
            "--data-dir", str(DATA_DIR),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"CLI failed:\n{result.stdout}\n{result.stderr}"
    out_files = list(tmp_path.glob("*.json"))
    assert len(out_files) == 1
    data = json.loads(out_files[0].read_text())
    assert "verdict" in data
    assert "continuous_eval" in data
    assert data["verdict"]["verdict"] in ("COMBINE-READY", "MARGINAL", "REJECT")


def test_cli_missing_strategy_exits_nonzero(tmp_path):
    result = subprocess.run(
        [
            sys.executable, "-m", "eval_sim.cli",
            "--strategy", "/nonexistent/strategy.py",
            "--fast",
            "--output-dir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_cli.py -v
```

Expected: `ModuleNotFoundError: No module named 'eval_sim.cli'`

- [ ] **Step 3: Create `src/eval_sim/cli.py`**

```python
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from eval_sim.config import INSTRUMENTS, DATA_SPLIT, SEARCH, DEFAULT_THRESHOLDS
from eval_sim.continuous_eval import run_continuous_eval
from eval_sim.data import load_ohlcv, DataLoadError
from eval_sim.evaluator import evaluate_window
from eval_sim.monte_carlo import run_monte_carlo
from eval_sim.search import random_search, SearchConfig
from eval_sim.sensitivity import run_sensitivity
from eval_sim.sizing import run_sizing_optimizer
from eval_sim.strategy import load_strategy, StrategyLoadError
from eval_sim.verdict import compute_verdict, VerdictThresholds, DEFAULT_THRESHOLDS
from eval_sim.windows import compute_windows


RISK_GRID = [150.0, 200.0, 300.0, 400.0, 500.0, 650.0, 800.0, 1000.0, 1200.0, 1500.0]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TopStep Eval Sim — continuous Combine evaluation pipeline")
    p.add_argument("--strategy", required=True, help="Path to strategy .py file")
    p.add_argument("--instrument", choices=list(INSTRUMENTS.keys()), default="mnq")
    p.add_argument("--timeframe", default="5min")
    p.add_argument("--fast", action="store_true", help="Skip WF search, use param range midpoints")
    p.add_argument("--search-n", type=int, default=SEARCH.n_candidates, help="Random search candidates")
    p.add_argument("--fixed-risk", type=float, default=None, help="Fixed risk dollars for sizing comparison")
    p.add_argument("--max-contracts", type=int, default=10)
    p.add_argument("--mc-n", type=int, default=500, help="Monte Carlo permutations")
    p.add_argument("--mc-block-size", type=int, default=5)
    p.add_argument("--data-dir", default="Data")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--reject-pass-rate", type=float, default=0.30)
    p.add_argument("--reject-max-dd", type=float, default=1800.0)
    p.add_argument("--ready-pass-rate", type=float, default=0.60)
    p.add_argument("--ready-max-dd", type=float, default=1200.0)
    return p


def _header(text: str) -> None:
    print(f"\n=== {text} ===")


def main(argv: list[str] | None = None) -> int:
    # Required on Windows when using multiprocessing with spawn context.
    # Must be called before any other multiprocessing code runs.
    import multiprocessing
    multiprocessing.freeze_support()

    args = build_parser().parse_args(argv)

    # Load strategy
    try:
        strategy = load_strategy(args.strategy)
    except StrategyLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Load data
    try:
        bars = load_ohlcv(args.instrument, args.timeframe, data_dir=args.data_dir)
    except DataLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    instrument = INSTRUMENTS[args.instrument]
    data_start, data_end = bars.index[0], bars.index[-1]

    # Compute windows
    try:
        windows = compute_windows(data_start, data_end, bars_index=bars.index, config=DATA_SPLIT)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    thresholds = VerdictThresholds(
        reject_pass_rate=args.reject_pass_rate,
        reject_max_dd=args.reject_max_dd,
        ready_pass_rate=args.ready_pass_rate,
        ready_max_dd=args.ready_max_dd,
    )

    # Stage 1: Search
    _header("Stage 1 — Param Search")
    search_result = random_search(
        bars=bars,
        strategy_fn=strategy.generate_signals,
        param_ranges=strategy.param_ranges,
        wf_windows=windows,
        instrument=instrument,
        risk_dollars=500.0,
        max_contracts=args.max_contracts,
        config=SearchConfig(n_candidates=args.search_n, seed=SEARCH.seed),
        fast_mode=args.fast,
    )
    print(f"{'FAST MODE' if args.fast else f'Evaluated {search_result.n_evaluated} candidates'}")
    print(f"Selected params: {search_result.best_params}")
    print(f"WF OOS score: {search_result.best_score:.3f}")

    selected_params = search_result.best_params

    # Stage 2: Sensitivity
    _header("Stage 2 — Sensitivity Sweep")
    # Per-window re-evaluation of selected params for reporting
    # (search only stored aggregate score; this gives per-window stats)
    _header("Stage 2 — Per-Window Evaluation (selected params)")
    from eval_sim.evaluator import Window
    wf_window_results = []
    for idx, w in enumerate(windows.wf_windows, 1):
        scoring_window = Window(start=w.start, end=w.end)
        w_trades = evaluate_window(
            bars, strategy.generate_signals, selected_params,
            scoring_window, instrument, 500.0, args.max_contracts,
            warmup_start=w.warmup_start,
        )
        w_result = run_continuous_eval(w_trades)
        wf_window_results.append((idx, w, w_result))
        status = "EMPTY" if w_result.is_empty else f"pass={w_result.pass_rate:.3f} mean_days={w_result.mean_days_per_attempt:.1f} median_days={w_result.median_days_per_attempt:.1f}"
        print(f"  Window {idx} ({w.start.date()} → {w.end.date()}): {status}")

    # Sensitivity sweep uses the last WF window with warmup
    _header("Stage 3 — Sensitivity Sweep")
    last_wf = windows.wf_windows[-1] if windows.wf_windows else None
    wf_dev_window = Window(start=last_wf.start, end=last_wf.end) if last_wf else Window(start=windows.holdout.start, end=windows.holdout.end)
    wf_dev_warmup = last_wf.warmup_start if last_wf else windows.holdout.warmup_start
    sensitivity = run_sensitivity(
        bars=bars,
        strategy_fn=strategy.generate_signals,
        selected_params=selected_params,
        param_ranges=strategy.param_ranges,
        window=wf_dev_window,
        instrument=instrument,
        risk_dollars=500.0,
        max_contracts=args.max_contracts,
        warmup_start=wf_dev_warmup,
    )
    cliff_label = "CLIFF DETECTED" if sensitivity.is_cliff else "flat"
    print(f"Sensitivity: {cliff_label}  selected_pass_rate={sensitivity.selected_pass_rate:.3f}")
    if sensitivity.is_cliff:
        print(f"  cliff params: {', '.join(sensitivity.cliff_params)}")

    # Stage 3: Continuous Eval Sim (holdout)
    _header("Stage 3 — Continuous Eval Sim (holdout)")
    from eval_sim.evaluator import Window
    holdout_window = Window(start=windows.holdout.start, end=windows.holdout.end)
    holdout_trades = evaluate_window(
        bars, strategy.generate_signals, selected_params,
        holdout_window, instrument, 500.0, args.max_contracts,
        warmup_start=windows.holdout.warmup_start,
    )
    eval_result = run_continuous_eval(holdout_trades)
    print(
        f"pass_rate={eval_result.pass_rate:.3f}  passes={eval_result.passes}  "
        f"attempts={eval_result.attempts}  worst_dd=${eval_result.worst_attempt_drawdown:.0f}"
    )

    # Stage 4: Monte Carlo
    _header("Stage 4 — Monte Carlo")
    mc_result = run_monte_carlo(holdout_trades, n=args.mc_n, block_size=args.mc_block_size)
    print(
        f"pass_rate_p05={mc_result.pass_rate_p05:.3f}  "
        f"median={mc_result.pass_rate_median:.3f}  "
        f"p95={mc_result.pass_rate_p95:.3f}"
    )

    # Stage 5: Sizing
    _header("Stage 5 — Sizing Optimizer")
    sizing = run_sizing_optimizer(
        bars=bars,
        strategy_fn=strategy.generate_signals,
        selected_params=selected_params,
        wf_windows=windows,
        instrument=instrument,
        max_contracts=args.max_contracts,
        fixed_risk_dollars=args.fixed_risk,
    )
    print(f"Optimal risk: ${sizing.optimal_risk_dollars:.0f}  pass rate: {sizing.optimal_pass_rate:.3f}  expected days/success: {sizing.optimal_expected_days:.1f}")
    print("Risk grid (holdout):")
    for lvl in sizing.all_levels:
        marker = " <-- optimal" if lvl.risk_dollars == sizing.optimal_risk_dollars else ""
        print(f"  ${lvl.risk_dollars:.0f}  pass={lvl.pass_rate:.3f}  days/success={lvl.expected_days_per_success:.1f}{marker}")
    if sizing.fixed_risk_result is not None:
        fr = sizing.fixed_risk_result
        print(f"Fixed ${fr.risk_dollars:.0f}: pass={fr.pass_rate:.3f}  days/success={fr.expected_days_per_success:.1f}")

    # Stage 6: Verdict
    _header("Stage 6 — Verdict")
    verdict = compute_verdict(
        pass_rate=eval_result.pass_rate,
        mc_pass_rate_p05=mc_result.pass_rate_p05,
        worst_drawdown=eval_result.worst_attempt_drawdown,
        sensitivity_is_cliff=sensitivity.is_cliff,
        thresholds=thresholds,
    )
    print(f"VERDICT: {verdict.verdict}")
    for r in verdict.reject_reasons:
        print(f"  REJECT: {r}")
    for w in verdict.warn_reasons:
        print(f"  WARN: {w}")

    # Write JSON output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{strategy.name}_{args.instrument}_{args.timeframe}_result.json"

    result_bundle = {
        "strategy": strategy.name,
        "instrument": args.instrument,
        "timeframe": args.timeframe,
        "fast_mode": args.fast,
        "topstep_rules": {
            "account_size": TOPSTEP_50K.account_size,
            "profit_target": TOPSTEP_50K.profit_target,
            "max_drawdown": TOPSTEP_50K.max_drawdown,
            "daily_loss_limit": TOPSTEP_50K.daily_loss_limit,
            "consistency_pct": TOPSTEP_50K.consistency_pct,
            "max_trading_days": TOPSTEP_50K.max_trading_days,
            "drawdown_type": "EOD_trailing",
        },
        "data_range": {"start": str(data_start), "end": str(data_end)},
        "best_params": search_result.best_params,
        "search": {
            "n_evaluated": search_result.n_evaluated,
            "wf_oos_score": search_result.best_score,
        },
        "wf_windows": [
            {
                "window": idx,
                "start": str(w.start.date()),
                "end": str(w.end.date()),
                "is_empty": res.is_empty,
                "pass_rate": res.pass_rate,
                "passes": res.passes,
                "attempts": res.attempts,
                "mean_days_per_attempt": res.mean_days_per_attempt,
                "median_days_per_attempt": res.median_days_per_attempt,
                "worst_drawdown": res.worst_attempt_drawdown,
            }
            for idx, w, res in wf_window_results
        ],
        "wf_aggregate": {
            "mean_pass_rate": (
                sum(r.pass_rate for _, _, r in wf_window_results if not r.is_empty)
                / max(1, sum(1 for _, _, r in wf_window_results if not r.is_empty))
            ),
            "empty_windows": sum(1 for _, _, r in wf_window_results if r.is_empty),
        },
        "sensitivity": {
            "is_cliff": sensitivity.is_cliff,
            "cliff_params": list(sensitivity.cliff_params),
            "selected_pass_rate": sensitivity.selected_pass_rate,
            "param_results": {
                param: {
                    "selected_value": ps.selected_value,
                    "neighbors": {str(k): v for k, v in ps.neighbor_results.items()},
                }
                for param, ps in sensitivity.param_results.items()
            },
        },
        "holdout": {
            "start": str(windows.holdout.start.date()),
            "end": str(windows.holdout.end.date()),
            "is_empty": eval_result.is_empty,
            "pass_rate": eval_result.pass_rate,
            "passes": eval_result.passes,
            "attempts": eval_result.attempts,
            "mean_days_per_attempt": eval_result.mean_days_per_attempt,
            "median_days_per_attempt": eval_result.median_days_per_attempt,
            "worst_attempt_drawdown": eval_result.worst_attempt_drawdown,
        },
        "monte_carlo": {
            "n_permutations": mc_result.n_permutations,
            "pass_rate_p05": mc_result.pass_rate_p05,
            "pass_rate_median": mc_result.pass_rate_median,
            "pass_rate_p95": mc_result.pass_rate_p95,
            "worst_drawdown_median": mc_result.worst_drawdown_median,
        },
        "sizing": {
            "optimal_risk_dollars": sizing.optimal_risk_dollars,
            "optimal_pass_rate": sizing.optimal_pass_rate,
            "optimal_expected_days_per_success": sizing.optimal_expected_days,
            "all_levels": [
                {
                    "risk_dollars": l.risk_dollars,
                    "pass_rate": l.pass_rate,
                    "mean_days_per_attempt": l.mean_days_per_attempt,
                    "expected_days_per_success": l.expected_days_per_success,
                }
                for l in sizing.all_levels
            ],
            "fixed_risk": (
                {
                    "risk_dollars": sizing.fixed_risk_result.risk_dollars,
                    "pass_rate": sizing.fixed_risk_result.pass_rate,
                    "mean_days_per_attempt": sizing.fixed_risk_result.mean_days_per_attempt,
                    "expected_days_per_success": sizing.fixed_risk_result.expected_days_per_success,
                }
                if sizing.fixed_risk_result is not None else None
            ),
        },
        "verdict": {
            "verdict": verdict.verdict,
            "reject_reasons": list(verdict.reject_reasons),
            "warn_reasons": list(verdict.warn_reasons),
        },
    }

    out_file.write_text(json.dumps(result_bundle, indent=2, default=str))
    print(f"\nResult written to: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run all tests**

```bash
pytest -v
```

Expected: all tests PASS (CLI test may be skipped if `Data/` not present — that's fine)

- [ ] **Step 5: Run the CLI manually with example strategy in fast mode**

```bash
python -m eval_sim.cli \
  --strategy src/eval_sim/strategies/example_strategy.py \
  --instrument mnq \
  --timeframe 5min \
  --fast \
  --output-dir output
```

Expected: pipeline runs through all 6 stages and prints a verdict. Check `output/` for JSON file.

- [ ] **Step 6: Commit**

```bash
git add src/eval_sim/cli.py tests/test_cli.py
git commit -m "feat: CLI — full pipeline orchestration with JSON output"
```

---

## Final: End-to-End Verification

- [ ] **Run full test suite**

```bash
pytest -v --tb=short
```

Expected: all tests PASS.

- [ ] **Run CLI with real data, normal mode (not fast)**

```bash
python -m eval_sim.cli \
  --strategy src/eval_sim/strategies/example_strategy.py \
  --instrument mnq \
  --timeframe 5min \
  --search-n 50 \
  --output-dir output
```

Verify:
- Completes without error
- Output JSON contains `verdict`, `continuous_eval`, `monte_carlo`, `sizing`, `sensitivity`
- Runtime is well under 10 minutes for 50 candidates

- [ ] **Verify no imports from Topstep pipeline project**

```bash
grep -r "Topstep pipeline" src/
grep -r "from v3" src/
grep -r "import v3" src/
```

Expected: no matches.

- [ ] **Final commit**

```bash
git add -A
git commit -m "feat: Plan 2 complete — full eval-sim pipeline operational"
```
