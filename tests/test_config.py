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
