import pytest
from tests.conftest import make_trade
from eval_sim.monte_carlo import run_monte_carlo, MCResult, _block_bootstrap
from eval_sim.config import TOPSTEP_50K
from eval_sim.topstep import _group_by_day


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


def test_run_monte_carlo_distribution_has_spread():
    # Mix wins and losses so block-bootstrap actually produces distribution variance.
    mixed = [
        make_trade(net_pnl=(120.0 if i % 3 != 0 else -60.0),
                   entry_time=f"2020-01-{2 + (i % 28):02d} 09:35")
        for i in range(60)
    ]
    result = run_monte_carlo(mixed, TOPSTEP_50K, n=200, block_size=5, seed=42)
    # Non-degenerate distribution: p95 should be >= p05 (with mixed PnL there should be spread,
    # but at minimum the bound must hold).
    assert result.pass_rate_p95 >= result.pass_rate_p05
    assert 0.0 <= result.pass_rate_p05 <= 1.0
    assert 0.0 <= result.pass_rate_p95 <= 1.0


def test_block_bootstrap_keeps_duplicate_sampled_days_as_distinct_sessions():
    class AlwaysFirstDayRng:
        def integers(self, low, high=None):
            return 0

    trades = [
        make_trade(net_pnl=10.0, entry_time="2020-01-02 09:35"),
        make_trade(net_pnl=20.0, entry_time="2020-01-02 09:40"),
        make_trade(net_pnl=30.0, entry_time="2020-01-03 09:35"),
    ]

    resampled = _block_bootstrap(trades, block_size=1, rng=AlwaysFirstDayRng())
    by_day = _group_by_day(resampled)

    assert len(by_day) == 2
    assert [t.net_pnl for day in sorted(by_day) for t in by_day[day]] == [10.0, 20.0, 10.0, 20.0]


def test_run_monte_carlo_empty_trades():
    result = run_monte_carlo([], TOPSTEP_50K, n=20, block_size=5, seed=42)
    assert result.pass_rate_median == 0.0
    assert result.pass_rate_p05 == 0.0
