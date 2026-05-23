# Kelly Sizing for Funded Account Stage (Post-Holdout)

## STATUS: BLOCKED — answer the 3 questions at the bottom before coding starts.

## Context

User asked: "would Kelly sizing make sense to add for the holdout?"

After discussion, refined the question to: **Kelly fits the funded account, not the eval/holdout. Should we add a funded-account simulation stage after holdout, with Kelly-based sizing?**

**Why Kelly is wrong for eval / holdout:**
- Eval has finite-horizon hard constraints: 60-day cap, $3k profit target, 50% consistency rule. Kelly maximizes long-run geometric growth — irrelevant when the game ends at $3k profit.
- The 50% consistency rule actively *punishes* the "scale up on edge" behavior Kelly produces.
- `src/eval_sim/sizing.py` already does the right thing for eval: sweeps risk_dollars through a real Combine MC and picks the level that minimizes `expected_days_per_success` above a 40% pass-rate gate. That is a domain-specific sizer using the actual loss function — better than Kelly for this stage.
- Each WF window has small trade samples → noisy W and R → Kelly oversizes. Don't trust it here.

**Why Kelly fits funded:**
- No profit target, no day cap, no consistency rule. Goal shifts from "pass a test" to "compound the account without busting." This is the classic log-utility-with-absorbing-barrier problem Kelly was built for.
- By the time strategy reaches funded simulation it has WF + holdout trade history → larger sample, less Kelly variance.
- Trailing DD + daily loss limit still apply, so we use **fractional Kelly** (¼ or ½), not full Kelly, with a hard cap from the daily-loss buffer.

**Intended outcome:** A new `funded` simulation stage that runs *only after* holdout passes the Verdict gate. Reports expected time-to-payout, ruin probability, equity-curve distribution under Kelly / fractional-Kelly / fixed-risk sizing — three head-to-head variants — using MC over the holdout trade distribution.

## Scope

In scope:
- `FundedRules` dataclass — funded-account constraints (no profit target, no day cap, no consistency).
- `simulate_funded()` — reuse the trade-application + trailing-DD + daily-loss machinery from `simulate_topstep`; strip the pass/fail criteria that don't apply.
- Kelly sizer with three modes: `kelly_full`, `kelly_half`, `kelly_quarter`. Inputs: rolling W and R from prior trades (estimated from WF + holdout). Output: per-trade `risk_dollars` = fraction × current_equity, capped by daily-loss buffer remaining.
- New CLI stage (`--funded-sim`), gated on holdout Verdict = PASS or READY.
- MC over funded trade sequence — report ruin prob, expected days to $1.5k profit (typical first-payout threshold on Topstep 50k), equity-curve percentiles, max DD distribution.
- Head-to-head comparison: Kelly variants vs fixed-risk-from-eval (the level chosen by `sizing.py`).

Out of scope:
- Live payout mechanics (5-winning-day rule, withdrawal cadence) — backtest only models equity dynamics, not when you can actually withdraw.
- Multi-account scaling, copy trading.
- Touching eval / holdout sizing — that stays as-is.

## Files to Add / Modify

**New:**
- `src/eval_sim/funded.py` — `FundedRules`, `simulate_funded()`, MC wrapper. Mirror structure of `topstep.py`.
- `src/eval_sim/sizing_kelly.py` — Kelly fraction computation from trade history; per-trade sizing function.
- `tests/test_funded.py` — covers trailing-DD behavior, daily-loss lockout, Kelly fraction math against hand-computed cases.

**Modify:**
- `src/eval_sim/config.py` — add `FundedRules` dataclass and `TOPSTEP_50K_FUNDED` instance (account_size $50k, trailing_dd $2k, daily_loss $1k, no profit target / day cap / consistency).
- `src/eval_sim/cli.py` — add `--funded-sim` flag, `--kelly-fraction {full,half,quarter,none}` (default `half`), `--funded-mc-n` (default 1000), `--funded-target-profit` (default $1500). Gate new stage on `verdict in {"PASS","READY"}`. Add Stage 6 console + JSON output.
- `docs/plans/` — drop a short reference doc explaining the funded-stage model and how Kelly inputs are estimated.

## Functions / Patterns to Reuse

- `src/eval_sim/topstep.py:36` — `simulate_topstep` rule-application loop (trailing DD update, daily-loss lockout, intraday balance checks). Strip profit target + consistency + day cap; keep the rest.
- `src/eval_sim/monte_carlo.py:38` — block-bootstrap of trade ordering; reuse for funded MC over the holdout trade list.
- `src/eval_sim/trades.py:6` — `TradeResult.r_multiple` and `net_pnl` already there; Kelly needs `win_rate = mean(r_multiple > 0)` and `payoff_ratio = mean_win_r / abs(mean_loss_r)`. No new trade-level fields required.
- `src/eval_sim/evaluator.py:18` — `_size_position` signature; Kelly sizer produces a `risk_dollars` value that feeds straight into it. No engine changes needed.
- `src/eval_sim/cli.py:202` — existing risk-grid JSON output pattern (`sizing.all_levels[]`); mirror it for `funded.kelly_variants[]`.

## Kelly Computation Detail

Per trade in the funded sim:
1. Estimate W and R from the *cumulative* trade history seen so far (WF + holdout + funded trades up to now). Lock in minimum sample of 50 trades before activating Kelly; below that, fall back to fixed risk from `sizing.py` optimum.
2. `f_kelly = W - (1 - W) / R` clipped to [0, 0.25] before fractioning.
3. `risk_dollars = fraction × f_kelly × current_equity`, then `min(risk_dollars, daily_loss_buffer_remaining)`.
4. Pass to `_size_position` as usual.

Three variants run in parallel MC and reported side-by-side; user picks based on ruin prob vs growth tradeoff.

## Verification

1. **Unit tests** (`tests/test_funded.py`):
   - `simulate_funded` matches `simulate_topstep` trade application when only trailing DD + daily loss are active (set profit_target=∞, day_cap=∞, consistency=1.0 → behavior should be identical until DD bust).
   - Kelly fraction calc against hand-computed cases (e.g. W=0.55, R=1.5 → f*=0.25; W=0.5, R=1.0 → f*=0).
   - Daily-loss cap correctly clamps Kelly-sized risk.
2. **Pipeline run end-to-end**: pick an existing passing strategy (e.g. `example_strategy.py` with known-good params), run full pipeline with `--funded-sim`, confirm Stage 6 output appears, JSON has `funded` key, no regressions in Stages 1–5.
3. **Sanity vs theory**: run funded MC with synthetic trades from a known distribution (W=0.6, R=1.5, fixed $100 stops) → confirm half-Kelly ruin prob over 250 trades < 5%, full-Kelly > 20%. If not, math is wrong.
4. **Caveman check** the JSON output and console summary read clean to user.

## Open Questions to Resolve Before Implementation

**BLOCKING — do not start coding until these are answered. Drop your answers inline under each question.**

### Q1. Funded account size

Start at $50k matching eval, or model the larger Topstep PA size ($53k after first profit pull)?

- **Option A — $50k start.** Cleanest comparison vs eval. Pessimistic on payout timing.
- **Option B — $53k start with trailing DD frozen at $50k floor.** Models the real PA account once activated. More realistic.
- **Recommended:** B if we want the report to reflect real funded behavior; A if we just want a quick growth-vs-ruin study.

**Answer:**

---

### Q2. Trailing DD model

- **Option A — true trailing from peak balance.** Aggressive; DD floor follows every new high. Punishes Kelly heavily after a winning streak.
- **Option B — Topstep "trail then freeze at start + buffer".** Real Topstep behavior. DD floor trails until balance hits start + $2k (i.e. $52k on a 50k), then freezes there permanently. Once you're past that threshold, you have a fixed $2k cushion below the freeze line forever.
- **Recommended:** B. Matches reality. Materially changes Kelly's optimal fraction post-threshold.

**Answer:**

---

### Q3. MC source for funded sim trade distribution

- **Option A — holdout trades only.** ~18 months, smaller sample, but cleanest OOS.
- **Option B — WF OOS + holdout combined.** More trades → more stable Kelly W/R inputs, but bias risk if WF and holdout regimes differ.
- **Option C — holdout only for sizing inputs, but block-bootstrap the holdout for the MC sequence.** Best of both: OOS purity + sequence resampling for confidence intervals.
- **Recommended:** C.

**Answer:**

---

When all three answered, remove the BLOCKED status banner at the top and start implementation.
