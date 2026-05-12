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
