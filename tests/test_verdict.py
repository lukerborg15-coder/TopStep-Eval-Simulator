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


def test_verdict_reject_low_mc_p05():
    result = compute_verdict(
        pass_rate=0.70, mc_pass_rate_p05=0.20, worst_drawdown=500.0,
        sensitivity_is_cliff=False, thresholds=_T,
    )
    assert result.verdict == "REJECT"
    assert any("mc_pass_rate_p05" in r for r in result.reject_reasons)


def test_verdict_combine_ready():
    result = compute_verdict(
        pass_rate=0.70, mc_pass_rate_p05=0.60, worst_drawdown=900.0,
        sensitivity_is_cliff=False, thresholds=_T,
    )
    assert result.verdict == "COMBINE-READY"
    assert not result.reject_reasons


def test_verdict_requires_mc_p05_for_combine_ready():
    result = compute_verdict(
        pass_rate=0.70, mc_pass_rate_p05=0.50, worst_drawdown=900.0,
        sensitivity_is_cliff=False, thresholds=_T,
    )
    assert result.verdict == "MARGINAL"
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
