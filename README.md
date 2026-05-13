# TopStep Eval Simulator

Standalone Python toolkit to load OHLCV bars, run strategy signals through a Topstep-style Combine simulator, walk-forward search, holdout evaluation, Monte Carlo, sizing, and a verdict — **no dependency on any other repo**.

**Repository:** [github.com/lukerborg15-coder/TopStep-Eval-Simulator](https://github.com/lukerborg15-coder/TopStep-Eval-Simulator)

## Requirements

- Python **3.11+**
- Windows, macOS, or Linux (multiprocessing uses `spawn` for Windows compatibility)

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -e ".[dev]"
```

## Data layout (MNQ **and** MES)

Place Databento-style OHLCV CSVs under `Data/` (or pass `--data-dir`). The loader expects:

| CLI `--instrument` | Filename pattern |
|--------------------|------------------|
| `mnq` | `mnq_{timeframe}_databento.csv` |
| `mes` | `mes_{timeframe}_databento.csv` |

Examples: `mnq_5min_databento.csv`, `mes_1h_databento.csv`.

**Columns:** `datetime` or `timestamp`, plus `open`, `high`, `low`, `close`, `volume`. Index is normalized to **US/Eastern**.

The `Data/` folder is **gitignored** (files are large). Copy your own `mnq_*` and `mes_*` exports into `Data/` locally.

## Instruments

Configured in `eval_sim.config`:

- **MNQ** — micro Nasdaq futures (default CLI instrument)
- **MES** — micro S&P futures (`--instrument mes`)

Both use the same pipeline; point value, tick size, commission, and slippage differ per instrument.

## CLI

Entry point: `eval-sim` (from `[project.scripts]`) or `python -m eval_sim.cli`.

```bash
# MNQ, fast smoke (skips full random search)
eval-sim --strategy src/eval_sim/strategies/example_strategy.py --instrument mnq --timeframe 5min --fast --data-dir Data --output-dir output

# MES — same flags, different instrument and CSV
eval-sim --strategy src/eval_sim/strategies/example_strategy.py --instrument mes --timeframe 5min --fast --data-dir Data --output-dir output
```

Useful flags: `--search-n`, `--mc-n`, `--mc-block-size`, `--fixed-risk`, `--max-contracts`, `--reject-pass-rate`, `--data-dir`, `--output-dir`.

## Tests

```bash
pytest -v
```

## Project layout

- `src/eval_sim/` — package (config, data, evaluator, Topstep sim, windows, search, pipeline stages, CLI)
- `src/eval_sim/strategies/` — strategies (`PARAM_RANGES`, `generate_signals`, optional `WARMUP_BARS`)
- `tests/` — pytest suite
- `docs/plans/` — implementation plans (reference)

## Design notes

- **Standalone:** only `src/eval_sim/` and tests; no imports from other local projects.
- **Timestamps:** timezone-aware **US/Eastern** after load.
- **Multiprocessing:** `multiprocessing.get_context("spawn")` where pools are used (Windows-safe).

## License

Add a `LICENSE` file if you want an explicit license; none is bundled by default.
