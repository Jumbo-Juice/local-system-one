# Registration: boosted trees alone on the 20 FINAL windows (read-out)

Revision Stage 1, fallback branch. Registered 2026-11-19. This file and `fraud/eval_gbdt_readout.py` are committed together
before any FINAL row is scored by this setup.

## Why this is a read-out, not a test

The Stage 0 audit (`fraud/results/window_audit.json`, `REVISION_LOG.md`) found that all 118 IEEE-CIS test windows had already
been scored. Only 108 tail rows are unscored, so no clean held-out test remains. The FINAL windows were read once, in Phase 23
(`fraud/results/eval_final/20261104-090649.json`). The setup below was never scored on them, but its windows are not fresh.
Every number this produces is labelled **`read-out`**, and the paper says that no clean data remained for a new test.

Owner decision (Juji, 2026-11-19, `REVISION_LOG.md` → paper-driver verdicts): the 800-tree `tab-all`, from one inference pass.
It is not the 400-tree score cached in `signal.npz`.

## Setup: `final-tab-all-amt`

- **Model.**
  - Phase 20's `tab-all`, loaded from `fraud/models/tabular_tab-all.pkl` (sha256
    `92640f230ce032c4faa6bf8589a2bd524f8a73487c89f83af884504a0e50adb1`). The script refuses any other file.
  - `fraud/gbdt.py`, all 800 trees (`Params(trees=800, depth=6, lr=0.05, row_sample=0.5, col_sample=0.3, seed=0)`), fitted on
    fit rows only.
  - No retraining.
- **Scoring.**
  - `et.score(m, f, t, windows)`, which is `np.clip(m.predict(f(t, rows)), 1e-6, 1 - 1e-6)` with `trees=None`, so all 800
    trees are used.
  - `f` is `fs.FeatureSet((), "", cache)`, the 583 features of `tab-all`.
  - Features are binned in float32 (`gbdt.to_bins`).
  - This is deterministic.
- **Policy, fixed from dev rows before any FINAL label is read.** This is Phase 21's recipe for `tab-all`
  (`fraud/eval_lm_nuc.py:197-200`):
  - p from `tb.fit_policy(q, t, dev30)`, a logistic fit of the score's logit on the 30 dev windows;
  - λ from `et.lambda_on(pol, q, t, dev30 + 88 held-out dev chains)`.
  - The decisions use Phase 14's cost rule only (`am.decide`). There are no fixed lines and no customer line.
- **Dev reproduction check, run before FINAL.** The script stops before scoring any FINAL row unless all of the following
  match Phase 21 (`fraud/results/eval_lm_nuc/20261104-090217.json`):
  - λ = 5;
  - the two lines, as a fit check;
  - the mean rule cost on the 88 held-out dev chains, 1,830.15 USD/window (±0.5);
  - the AUC on those chains, 0.9139 (±0.0001).
- **Costs.** As every IEEE-CIS phase since 14:
  - friction 10% of the amount, at least 1 USD;
  - review 10 USD;
  - at most 5% of a window reviewed.

## Windows

The 20 FINAL windows, 1,000 contiguous rows each, from `fraud/final_test.py:final_windows`. The script checks them against
`eval_final/20261104-090649.json`.

| id | rows | id | rows |
|---|---|---|---|
| final0 | 472752–473751 | final10 | 534565–535564 |
| final1 | 482973–483972 | final11 | 539368–540367 |
| final2 | 489742–490741 | final12 | 542838–543837 |
| final3 | 492355–493354 | final13 | 548871–549870 |
| final4 | 498948–499947 | final14 | 552509–553508 |
| final5 | 502141–503140 | final15 | 556970–557969 |
| final6 | 509607–510606 | final16 | 566087–567086 |
| final7 | 514028–515027 | final17 | 567087–568086 |
| final8 | 523003–524002 | final18 | 581157–582156 |
| final9 | 526783–527782 | final19 | 584622–585621 |

20 windows, 20,000 rows.

## Comparisons

The other setups' per-window costs come from their saved Phase 23 traces (`fraud/runs/20261104-090649_eval_<setup>_final<k>.jsonl`).
Nothing is re-scored.

- **Primary** (the revision prompt's primary): LM + tree score (`final-signal-amt`) against GBDT alone (`final-tab-all-amt`).
- **Secondary:** GBDT alone (`final-tab-all-amt`) against LR (`final-logreg-amt`).
- **Reported, not a question:** LM alone (`final-rich-amt`) against GBDT alone.
- The other setups in the prompt's held-out list are not part of this fallback and get no FINAL number:
  - LR with two fixed cut-offs;
  - LR + rule + customer line.
- No correction for multiple comparisons.

## Decision rule

The 95% paired window-bootstrap percentile interval of the mean per-window cost difference uses 10,000 resamples of the 20
windows, seed 20261102 (`fraud/eval_ieee.py:boot_ci` through `compare`). A difference is distinguishable only if the interval
excludes zero. Every result is reported in customers as well as dollars: good customers declined and frauds stopped.

## Minimum detectable effect

The MDE is 2.8 × SD / √20: 80% power, two-sided 5%, normal approximation. SD is the paired per-window SD on the 88 held-out
dev chains of Phase 21 (dev):

| comparison | dev mean difference | dev paired SD | MDE on 20 windows |
|---|---:|---:|---:|
| LM + tree − GBDT | +89.8 | 399.3 | **250 USD/window** |
| GBDT − LR | −294.9 | 569.0 | **356 USD/window** |
| LM alone − GBDT | +281.8 | 656.5 | 411 USD/window |

The dev reads of the LM setups and of `tab-all` are flattered, because λ was picked with these chains in.

## Run once

- `python -m fraud.eval_gbdt_readout` runs once and refuses a second run if traces of `final-tab-all-amt` exist.
- Traces go to `fraud/runs/<eval id>_eval_final-tab-all-amt_final<k>.jsonl`.
- Results go to `fraud/results/eval_gbdt_readout/<eval id>.json` and `.md`. The JSON carries `"label": "read-out"`, and each
  setup row carries its label: `read-out` for this setup, `held-out (pre-registered)` for the Phase 23 setups it is compared
  with.

## Stated before running (Inferred)

- GBDT alone is cheaper than LR on FINAL, by less than its dev gap of 295, and not distinguishable: the MDE is 356.
- LM + tree and GBDT alone are within about ±250 of each other, not distinguishable.
- Whatever comes out is reported as is.
