# Registration: LR with λ picked on all 118 dev windows, on the 20 FINAL windows (read-out)

Referee revision, 2026-11-21. Owner: "Register and run it". This file and `fraud/eval_lr_lambda118.py` are committed together
before the script computes anything on FINAL.

## Why

The rule's review price λ was picked on different dev windows for different setups (`REVISION_LOG.md`, Stage 0 §3):

- LR, the baseline: on the 30 dev windows only (`dev30`, Phase 14's procedure).
- The language-model setups and GBDT: on all 118 dev windows (`dev30` + the 88 held-out dev chains).

The referee asked which claims this asymmetry affects. On dev, LR's λ moves from 10 (`dev30`) to 5 (118 windows, Appendix E
of the paper). The effect on the final test has not been measured. This read-out measures it, from saved outputs only.

## Why this is a read-out, not a test

The FINAL windows were read in Phase 23 and again for the GBDT read-out. No unscored window is left
(`fraud/results/window_audit.json`). Every number this script produces is labelled **`read-out`**.

## What runs

No model is trained or run on FINAL. Nothing is re-scored.

- **FINAL scores.** LR's per-row scores `q` come from its saved Phase 23 traces,
  `fraud/runs/20261104-090649_eval_final-logreg-amt_final<k>.jsonl`.
- **Dev scores.** LR is refitted on fit rows exactly as Phase 23 did (`eval_lm_nuc.references` → `tab-logreg`, through
  `eval_readouts.dev_scores`).
- **p.** A logistic fit of the score's logit on `dev30` (`tabular.fit_policy`). This is unchanged from the baseline.
- **λ_30.** The baseline's λ, from `fit_policy` on `dev30`.
- **λ_118.** `am.pick_lambda` on all 118 dev windows, with p unchanged. This is the same call `eval_tabular.lambda_on` makes for
  the other setups, with the same grid and the same costs.
- **Decisions.** `am.decide(p(q), amount, λ, COSTS)` on each FINAL window, scored by `costs.score`, as in
  `eval_readouts.section_grid`.
- **Costs.** As every IEEE-CIS phase since 14:
  - friction 10% of the amount, at least 1 USD;
  - review 10 USD;
  - at most 5% of a window reviewed.

## Checks (the script stops if one fails)

1. λ_30 equals the λ in the saved LR trace header (10).
2. With λ_30, the recomputed per-window FINAL costs equal the saved LR costs (`tabular.window_costs`), within 1e-6 USD.

## Comparisons

The other setups' per-window costs come from their saved traces: Phase 23, and the GBDT read-out `20261119-231754`.

- **Primary:** LR with λ_118 against LR with λ_30. How much does the asymmetry move the baseline on FINAL?
- **Reported:** each setup against LR with λ_118:
  - LM + tree (`final-signal-amt`);
  - LM + tree with the customer line (`final-signal-cust`);
  - LM alone (`final-rich-amt`);
  - GBDT alone (`final-tab-all-amt`, itself a read-out).

  For each, the verdict is set beside the one against the baseline as registered.

## Decision rule

The 95% paired window-bootstrap percentile interval of the mean per-window difference uses 10,000 resamples, seed 20261102
(`eval_ieee.boot_ci` of the per-window differences a − b, as `eval_readouts` does). A difference is distinguishable only if the interval excludes zero. The script also reports the paired
SD and the post-hoc MDE, 2.8 × SD / √20, both labelled read-out. There is no correction for multiple comparisons.

## Stated before running (Inferred)

- λ_118 = 5, as on dev in Appendix E (`eval_parity` → `tab-logreg`).
- LR with λ_118 differs from LR with λ_30 by less than 100 USD per window, which is not distinguishable. On dev the change was
  −8, in-sample.
- No verdict against LR changes. Every setup stays not distinguishable from LR.
- Whatever comes out is reported as is, in the paper's Section VII-C and in `paper/REFEREE-RESPONSE.md`.

## Output

- `python -m fraud.eval_lr_lambda118` writes `fraud/results/eval_lr_lambda118/<eval id>.json` and `.md`.
- The JSON has `"label": "read-out"`.
- No traces are written, because no new scoring is done.

## Amendment 1 (2026-11-21), before any run

The first attempt stopped at load time, before anything was computed on FINAL:
`fraud/runs/20261104-090649_eval_final-logreg-amt_final0.jsonl` does not exist on this machine.

- Observed: this machine's `fraud/runs/` holds no run after `20261103-164830`. The FINAL traces and the GBDT read-out traces
  were written on the machine that ran the final test.
- The results JSONs are committed, and they hold every setup's per-window FINAL costs:
  - `eval_final/20261104-090649.json` → `setups.<setup>.windows`;
  - `eval_gbdt_readout/20261119-231754.json`, the same for GBDT.

**The change.** The design above is unchanged except where LR's FINAL scores come from.

- **Scores.** LR is refitted on fit rows exactly as Phase 23 did. `eval_tabular.fit("tab-logreg", …)` uses IRLS on the
  583 features, has no randomness and loads no saved model. It scores the dev windows and the 20 FINAL windows (`et.score`).
- **Labels.** No FINAL label enters any fit. Labels are used only to cost decisions, as in every read-out.
- **Check 2 (stronger).** With λ_30, the 20 recomputed LR window costs must equal Phase 23's saved per-window LR costs within
  1e-6 USD. That holds only if these scores make every decision Phase 23 made. If it fails, the script stops, and nothing is
  reported but the failure.
- **Check 1.** It now compares λ_30 with Phase 21's recorded `tab-logreg` `lambda_on_30`, which is 10.
- **Other setups.** Their per-window costs are read from the two results JSONs above, not from traces.
