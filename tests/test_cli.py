import subprocess
import sys
import json
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import pytest
import shutil
from eval_sim.config import DATA_SPLIT
from eval_sim.continuous_eval import ContinuousEvalResult
from eval_sim.monte_carlo import MCResult
from eval_sim.search import SearchResult
from eval_sim.sensitivity import SensitivityResult
from eval_sim.sizing import RiskLevelResult, SizingResult
from eval_sim.verdict import VerdictResult
from eval_sim.windows import WFWindows, OOSWindow


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
            "--mc-n", "5",
            "--mc-block-size", "5",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"CLI failed:\n{result.stdout}\n{result.stderr}"
    out_files = list(tmp_path.glob("*.json"))
    assert len(out_files) == 1
    data = json.loads(out_files[0].read_text())
    assert "verdict" in data
    assert "holdout" in data
    assert "n_trades" in data["holdout"]
    assert isinstance(data["holdout"]["n_trades"], int)
    assert data["verdict"]["verdict"] in ("COMBINE-READY", "MARGINAL", "REJECT")
    for w in data.get("wf_windows", []):
        assert "n_trades" in w
        assert isinstance(w["n_trades"], int)


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


@dataclass(frozen=True)
class _FakeStrategy:
    name: str = "fake_strategy"
    param_ranges: dict = None
    warmup_bars: int = 37

    def __post_init__(self):
        object.__setattr__(self, "param_ranges", {"x": [1]})

    def generate_signals(self, bars, params):
        return pd.DataFrame()


def _fake_eval_result(pass_rate: float = 0.5) -> ContinuousEvalResult:
    return ContinuousEvalResult(
        passes=1,
        attempts=2,
        pass_rate=pass_rate,
        worst_attempt_drawdown=100.0,
        mean_days_per_attempt=5.0,
        median_days_per_attempt=5.0,
        total_trading_days=10,
        is_empty=False,
    )


@pytest.mark.parametrize(
    ("fast_args", "expected_grid"),
    [(["--fast"], [500.0]), ([], None)],
)
def test_cli_uses_strategy_warmup_and_emits_continuous_eval(monkeypatch, fast_args, expected_grid):
    import eval_sim.cli as cli

    captured = {}
    output_dir = Path("tmp_test_cli_output") / "warmup_contract"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="US/Eastern")
    bars = pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100.0},
        index=idx,
    )
    wf = OOSWindow(warmup_start=idx[0], start=idx[10], end=idx[100])
    holdout = OOSWindow(warmup_start=idx[200], start=idx[250], end=idx[-1])
    windows = WFWindows(wf_start=idx[0], wf_end=idx[250], holdout=holdout, wf_windows=(wf,))

    def fake_compute_windows(data_start, data_end, bars_index, config):
        captured["warmup_bars"] = config.warmup_bars
        captured["base_config_unchanged"] = DATA_SPLIT.warmup_bars
        return windows

    def fake_sizing(**kwargs):
        captured["risk_grid"] = kwargs["risk_grid"]
        level = RiskLevelResult(150.0, 0.5, 5.0, 10.0)
        return SizingResult(
            optimal_risk_dollars=150.0,
            optimal_expected_days=10.0,
            optimal_pass_rate=0.5,
            all_levels=(level,),
            fixed_risk_result=None,
            training_levels=(level,),
            optimal_training_result=level,
            optimal_holdout_result=level,
        )

    monkeypatch.setattr(cli, "load_strategy", lambda path: _FakeStrategy())
    monkeypatch.setattr(cli, "load_ohlcv", lambda instrument, timeframe, data_dir: bars)
    monkeypatch.setattr(cli, "compute_windows", fake_compute_windows)
    monkeypatch.setattr(cli, "random_search", lambda **kwargs: SearchResult({"x": 1}, 0.25, 1))
    monkeypatch.setattr(cli, "evaluate_window", lambda *args, **kwargs: [])
    monkeypatch.setattr(cli, "run_continuous_eval", lambda trades: _fake_eval_result())
    monkeypatch.setattr(
        cli,
        "run_sensitivity",
        lambda **kwargs: SensitivityResult(0.5, {}, False, ()),
    )
    monkeypatch.setattr(cli, "run_monte_carlo", lambda *args, **kwargs: MCResult(3, 0.5, 0.4, 0.6, 100.0))
    monkeypatch.setattr(cli, "run_sizing_optimizer", fake_sizing)
    monkeypatch.setattr(cli, "compute_verdict", lambda **kwargs: VerdictResult("MARGINAL", (), ()))

    rc = cli.main(
        [
            "--strategy",
            "unused.py",
            "--output-dir",
            str(output_dir),
            "--data-dir",
            "unused",
            "--mc-n",
            "3",
        ]
        + fast_args
    )

    assert rc == 0
    assert captured["warmup_bars"] == 37
    assert captured["base_config_unchanged"] == DATA_SPLIT.warmup_bars
    assert captured["risk_grid"] == (expected_grid if expected_grid is not None else cli.RISK_GRID)

    out_files = list(output_dir.glob("*.json"))
    assert len(out_files) == 1
    data = json.loads(out_files[0].read_text())
    assert "holdout" in data
    assert data["holdout"]["n_trades"] == 0
    assert "continuous_eval" in data
    assert data["continuous_eval"] == data["holdout"]
    shutil.rmtree(output_dir.parent)
