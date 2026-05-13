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
