# Revision log: the portfolio version of the paper

A running log of the revision: what changed, why, which decision it implements, and what had to be assumed.
Decisions come from the owner's revision prompt (2026-11-19). Labels on numbers: `dev`, `held-out (pre-registered)`, `read-out`.

## Stage 0: audit (2026-11-19)

Branch `paper-revision`. No model was run and no label was read for this stage.

### 1. IEEE-CIS test windows ever scored

Script `fraud/dev/window_audit.py`, output `fraud/results/window_audit.json`. It rebuilds every window from the code that drew it
and checks each one against the first and last row its results JSON recorded. All 118 match.

| source | phase | windows | shape |
|---|---|---:|---|
| `eval_ieee/20261102-183431.json` | 11 | 20 | 1,000 contiguous rows |
| `eval_ieee_full/20261103-142622.json` | 12 | 20 | 1,000 contiguous rows |
| `eval_ieee_amount/20261103-164830.json` | 14 (and 16, re-priced) | 20 | 1,000 contiguous rows |
| `eval_column_audit/20261103-210600.json` | 17 | 8 | chains of 1,000 rows |
| `eval_rule/20261103-211736.json` | 18 | 10 | chains |
| `eval_features/20261103-212822.json` | 19 | 10 | chains |
| `eval_tabular/20261103-235132.json` | 20 | 10 | chains |
| `eval_final/20261104-090649.json` | 23, FINAL | 20 | 1,000 contiguous rows |

Observed:

- The test split has 118,108 rows. 118 windows (118,000 rows) have been scored, and no row was scored twice.
- 108 rows (590,432 to 590,539, the split's tail) were never scored.
- **Untouched 1,000-row windows: 0.** Allowing non-contiguous chains doesn't change this: 108 rows are not one window.
- Not counted as scored, because no label was read:
  - the tree score cached in `fraud/data/ieee-cis/signal.npz` (fit, dev, the FINAL rows and about 17% of test rows). It is the
    400-tree prefix of Phase 20's `tab-all`, the score that LM + tree reads, not `tab-all` itself (corrected; see the verdicts);
  - the history features that `fraud/history.py` builds from every row.
- The dev split is used up in the same way: 30 dev windows fitted the rule and the lines, and the 88 held-out dev chains are
  the rest. 108 dev rows are left over.

**Consequence: Stage 1 takes the fallback branch.** GBDT alone with the rule is scored on the 20 FINAL windows as one labelled
read-out, and the paper says that no clean data remained for a new test.

- The FINAL pre-registration excluded trees ("Trees alone aren't run here", `fraud/eval_final.py`), so this is a new number.
- ~~No inference is needed~~ (corrected): `signal.npz` holds only the 400-tree prefix of `tab-all` for the FINAL rows. Juji chose
  the 800-tree `tab-all`, so the read-out needs one inference pass of the saved model over the FINAL rows.

### 2. Feature sets of the baselines

Observed, from `fraud/eval_ieee.py:features_all`, `fraud/eval_tabular.py:SETS` and `fraud/ieee_brain.py`.

- **LR on FINAL** (`final-logreg-amt` = `tab-logreg` = `ieee-logreg-all`) and **GBDT** (`tab-all`, 800 trees, `fraud/eval_tabular.py` PARAMS; model code `fraud/gbdt.py`)
  were fitted on the same 583 features, on fit rows only:
  - log amount and the hour (24 one-hots);
  - one-hots of the 15 text columns: ProductCD, card4, card6, P_emaildomain, R_emaildomain, M1 to M9, DeviceType;
  - the signed log of 372 numeric columns: every C, D and V column, dist1/2, card3 and card5. card1, card2, addr1 and addr2
    are dropped as identifiers;
  - missing-value flags for the columns that are not V.
- **Neither baseline received the card-history aggregates that the rich prompt shows.** The rich prompt (`describe_rich`) shows:
  - from card history: payments before, time since the last one, payments in the last day and the last 7 days, and the amount
    against the card's usual;
  - from the address: other cards seen there.
- What else each side sees:
  - The prompt shows far fewer raw columns: no V, only C1/2/4/5/6/8/9/11/13/14, D1/2/3/4/10/15, M4 and M6.
  - LR and GBDT see every column, but no history.
- The closest earlier baseline is `tab-rich` (Phase 20 chains only, not FINAL): trees plus the "steady" history, extra and
  identity parts. "Steady" leaves out the lifetime counts, and the prompt uses one of them (`card_log_before`). So no existing
  baseline matches the prompt's information. Stage 2.2 has to build that feature set.
- `ieee-logreg-same` (Phases 11-12) is the plain prompt's fields as numbers, without history.

### 3. Where things are set

- **Temperature 3.383.**
  - It is d1-Omni-2B's own `calibration.json` → `"default": 3.3829832074455783`, shipped with the model at revision
    `431f1d66…` and downloaded to `fraud/models/d1-omni-hf/`.
  - It is read at `fraud/d1torch.py:225` and divides the option scores at `fraud/d1torch.py:273`.
  - The project never fitted it (Observed). The vendor's method for fitting it isn't documented in the repo.
- **Loss.**
  - PaySim (Qwen2.5-1.5B, `fraud/finetune.py:149`): cross-entropy over the whole vocabulary at the answer position.
  - IEEE-CIS (d1-Omni-2B, `fraud/finetune_d1.py` docstring → `finetune_d3.train`): cross-entropy over the three option
    scores divided by 3.383.
  - The documented reason: train "the function the decider reads". `d1torch` only produces the masked code readout over the
    options, so there is no plain vocabulary softmax to train.
- **Qwen → d1-Omni-2B.**
  - It was the owner's request in Phase 8 (`fraud/PROGRESS.md`, 2026-10-30): try the LFM3 family.
  - Frozen on PaySim dev, d1-Omni-2B ranked fraud better than frozen Qwen-1.5B and was faster than LFM3-1.7B, which was dropped.
  - Phase 9 fine-tuned it, and every IEEE-CIS phase from 11 on uses it.
- **Prompts.**
  - All in `fraud/ieee_brain.py`: `CONTEXT` (lines 18-22) is the hand-written "patterns in the fit rows" block.
  - `describe()` (line 37) is the plain prompt and `describe_rich()` (line 57) the rich one. `lm-signal` adds the
    "Risk score from a tabular model" line.
  - QUESTION and OPTIONS are PaySim's prompt v4 (`fraud/brain.py:22-27`).
- **Option order.**
  - Training shows each example in a seeded random order.
  - The final setups score in order `rev` only (`fraud/eval_lm_nuc.py:47`): A decline, B review, C approve.
- **λ selection.**
  - `eval_ieee_amount.pick_lambda` takes the lowest total dev cost over the grid 0, 1, 2, 5, … 1000.
  - `tabular.fit_policy` fits p (a logistic fit of the score's logit) and λ on the 30 dev windows (`dev30`).
  - `eval_tabular.lambda_on` re-picks λ on all 118 dev windows (dev30 + 88 held-out). It applies to every setup except LR:
    the LM setups (`fraud/eval_lm_nuc.py:199`; on FINAL, `fraud/eval_final.py:policies`) and also `tab-all` (`fraud/eval_lm_nuc.py:199`
    only: `eval_final.policies` never fitted `tab-all`).
  - LR keeps Phase 14's λ from dev30, so the baseline doesn't move.

### 4. Licences (information only; no adapter is published)

- **d1-Omni-2B: LFM Open License v1.0** (`fraud/models/d1-omni-hf/LICENSE`; Hugging Face tag `license:other`).
  - It allows derivative works to be redistributed, if the licence is included, changed files say so, and notices are kept.
  - Commercial use is licensed only for entities under 10 M USD annual revenue.
  - Note: the Hub's current revision (`51918b53…`) differs from the pinned `431f1d66…`.
- **Qwen2.5-1.5B-Instruct: Apache-2.0** (Hugging Face tag `license:apache-2.0`).

### Other findings that affect later stages

- **The FINAL per-row traces exist on this machine.** That is 100 files, `fraud/runs/20261104-090649_eval_*_final*.jsonl`,
  with per-row scores, truth, amount and action. `docs/ideation/2026-11-11-fraud-console-readability.md` says otherwise. So the
  Stage 3 analyses on saved outputs are possible.
- **Stage 2.1 wording vs the code.** The "reversed" order in the prompt (A decline, B review, C approve) is the order the trained
  setups are already scored in. Assumed reading: the new order to score is the forward one, A approve, B review, C decline.
- **The 88 dev chains are not clean for the LM setups.** Their λ was picked with these rows in, so every Stage 2 dev comparison
  involving an LM setup is flattered. The paper already says this, and Stage 2 will say it again.
- **The GBDT read-out on FINAL** will use Phase 21's recipe for `tab-all`: p fitted on dev30 and λ on all 118 dev windows.
- **`fraud/prereg/` doesn't exist yet.** Stage 1 or 2 creates it.

### paper-driver verdicts

- **Stage 0, round 1 (2026-11-19): ESCALATE.** The reviewer recounted the windows itself, from the results JSONs and
  `final_test.phase_windows`, and got the same 0 untouched windows. It found no problems in the audit, but always-escalate
  rule 2 (fewer than 20 untouched windows) applies.
  - **Juji's answer: A, the fallback as specified, from the cached score.**
    - Setup: `tab-all` GBDT, read from `fraud/data/ieee-cis/signal.npz`. No new inference and no retraining.
    - The Phase 21 recipe stays fixed: p fitted on dev30, λ picked on all 118 dev windows.
    - It is applied once to the 20 FINAL windows and compared with LR by the 95% paired window-bootstrap percentile interval
      (10,000 resamples). Every result is labelled `read-out`.
    - A short note fixing this recipe is committed in `fraud/prereg/` before anything is computed.
  - This is a settled decision from now on.
  - **Correction to Stage 0 (found 2026-11-19, after the answer, before anything was computed on FINAL).** Section 1 says
    that `signal.npz` holds the Phase 20 `tab-all` score. That is wrong.
    - `fraud/lm_signal.py` scores dev and test rows with the saved `tab-all` model, but only its first 400 trees:
      `predict(..., TREES)` with `TREES = 400`.
    - `tab-all` itself uses all 800 trees (`fraud/eval_tabular.py` PARAMS).
    - On the 88 held-out dev chains (Phase 21, `eval_lm_nuc/20261104-090217.json`):
      - `tab-all-amt` (800 trees): AUC 0.914, 1,830 USD/window, λ 5;
      - `signal-alone-amt` (the cache, 400 trees): AUC 0.911, 1,853 USD/window, λ 5.
    - The paper's "boosted trees" is `tab-all`. The cache is the paper's "tree score used alone", the score that LM + tree
      reads.
    - So option A, as approved, would report the 400-tree score under the GBDT name. Back to Juji.
  - **Juji's revised answer: `tab-all`, 800 trees.**
    - The saved model `fraud/models/tabular_tab-all.pkl` is scored once, deterministically, on the 20,000 FINAL rows. No
      retraining.
    - p is fitted on dev30 and λ on all 118 dev windows, as in Phase 21.
    - The recipe is committed in `fraud/prereg/` first, and the result is labelled `read-out`.
    - This replaces "from the cached score" in the answer above, and is settled from now on.
- **Stage 0, round 2 (2026-11-19): APPROVE WITH FIXES.** The reviewer recounted 0 untouched windows a second time, from the
  row ranges alone, and confirmed the 400/800-tree correction. Fixes, all made in the commit that logs this verdict:
  - section 2: `tab-all` has 800 trees, not 400;
  - section 1 and the `window_audit.json` notes: the cache is the 400-tree prefix of `tab-all`;
  - section 3: the `tab-all` λ re-pick is at `fraud/eval_lm_nuc.py:199`, not in `eval_final.policies`.
  - Process note from the reviewer: every material finding goes into the report sent to it.
  - Instruction: proceed to Stage 1 only up to the registration commit, then call paper-driver again before scoring FINAL.

## Stage 1: GBDT read-out on FINAL (fallback branch)

- **Registration.** `fraud/prereg/final_gbdt_readout.md` and the script `fraud/eval_gbdt_readout.py`, committed together
  before any FINAL row is scored. It fixes:
  - the model: `tabular_tab-all.pkl`, all 800 trees, checked by sha256;
  - the policy: Phase 21's recipe, with a dev reproduction check against Phase 21 that must pass before FINAL;
  - the windows;
  - the comparisons. Primary: LM + tree vs GBDT, the prompt's primary. Secondary: GBDT vs LR. Reported: LM alone vs GBDT;
  - the decision rule, and the MDE from the dev paired SD.
- **Assumed:**
  - The prompt's other held-out comparisons (LR with two cut-offs; LR + rule + customer line) are not part of the fallback,
    which names GBDT alone only. So they get no FINAL number.
  - The reviewer's list named only GBDT vs LR. The prompt's primary (LM + tree vs GBDT) is kept as primary, since the
    fallback exists to supply the GBDT number for it.
- `RUN-GUIDE.md` has the new command.

### Standing decision from Juji (2026-11-19)

Juji, verbatim: "You guys make your own best judgement. Dont come to me unless truly plan flipping question comes up."

- From now on, the worker and paper-driver decide escalations themselves. That covers the pre-scoring check before the FINAL
  read-out, whose recipe Juji already approved.
- Juji is asked only about a question that would flip the plan, such as:
  - a result that overturns the paper's story;
  - a change to a settled decision;
  - new models or data;
  - money or credentials;
  - pushing or publishing;
  - deleting results, adapters or history.

### paper-driver verdicts (continued)

- **Stage 1 pre-scoring, round 1 (2026-11-19): ESCALATE.** The reviewer verified all of the following:
  - the registration commit precedes any scoring;
  - the sha256;
  - 800 trees and the Phase 21 code path;
  - the reproduction thresholds, the windows and the MDEs, which it recomputed.
  - It also accepted keeping LM + tree vs GBDT as primary.
  - Fixes, made before scoring: a "Known asymmetries" section in the registration (the λ asymmetry LR vs tab-all, and that
    Phase 23 `--resume` is blocked afterwards).
  - It escalated because it saw the standing decision only through the worker's message.
  - **Resolved by the worker under Juji's direct instruction.** Juji typed it in this session's chat, after approving this
    exact recipe twice (answers A, then "tab-all, 800 trees"): "You guys make your own best judgement. Dont come to me unless
    truly plan flipping question comes up."
    - Scoring an already-approved, registered read-out does not flip the plan, so Juji is not asked again.
    - The reviewer keeps rule 1's spirit: the registration is committed first, and the reproduction check gates FINAL.

### Stage 1 result (read-out, eval 20261119-231754)

- **Run.** `python -m fraud.eval_gbdt_readout`, once.
  - The dev reproduction check matched Phase 21 exactly: λ 5, 1,830.15 USD/window, AUC 0.9139.
  - Then 20 FINAL traces were written.
- **Process note.** The markdown step crashed after the traces and the JSON were written: `report_table` needs `test_rows` and
  `test_fraud`. Both keys were added to `report()`, and the files were rebuilt with `--report 20261119-231754` from the saved
  traces. Nothing was re-scored.
- Results go to `fraud/results/eval_gbdt_readout/20261119-231754.json` and `.md`. Labels: this setup is `read-out`; the
  Phase 23 setups are `held-out (pre-registered)`.

| setup | mean USD/window | AUC | good declined | frauds stopped |
|---|---:|---:|---:|---:|
| GBDT alone (`final-tab-all-amt`), read-out | 1,897 | 0.878 | 864 | 413 of 630 |
| LM + tree (`final-signal-amt`) | 1,943 | 0.883 | 928 | 404 |
| LR (`final-logreg-amt`) | 2,106 | 0.853 | 1,051 | 384 |
| LM alone (`final-rich-amt`) | 2,427 | 0.859 | 1,081 | 375 |

- **Primary (read-out):** LM + tree − GBDT = **+46** USD/window, 95% interval −375 to +489. Not distinguishable; the MDE is
  250. GBDT was cheaper on 13 of 20 windows.
- **Secondary (read-out):** GBDT − LR = **−209**, interval −430 to +57. Not distinguishable; the MDE is 356. This is
  flattered by the λ asymmetry; see the registration.
- **Reported (read-out):** LM alone − GBDT = **+530**, interval +219 to +865. Distinguishable: LM alone is dearer than
  GBDT alone.
- **Against the prediction stated before running:**
  - GBDT vs LR came out as predicted: cheaper by less than 295, not distinguishable.
  - LM + tree within ±250 of GBDT: held, at +46.
- **Does it flip the plan? No.** The headline stays: no detectable saving from the LLM. The trees with the cost rule are at
  least as cheap as anything with the LLM, and the LLM alone is distinguishably dearer than the trees. That is an
  undercutting result, and the paper reports it as one.
- **Wording.** The shared `report_table` header calls the rows "new test rows". They are not new. The JSON's `label`
  ("read-out") is authoritative, and the paper will say "the same 20 windows, read again".
- **Realised MDE on FINAL (read-out).** From the FINAL per-window paired SDs, now in the results JSON → `mde_realised`, same
  formula:

  | comparison | FINAL paired SD | realised MDE | registered (dev) MDE |
  |---|---:|---:|---:|
  | LM + tree − GBDT | 969 | **≈ 607** | 250 |
  | GBDT − LR | 570 | ≈ 357 | 356 |
  | LM alone − GBDT | 769 | ≈ 482 | 411 |

  - The registered dev-based MDE of 250 for the primary was optimistic: FINAL windows vary far more between the two
    setups than dev chains did. The registration file is left as committed.
  - In Stage 3.1 and in the paper, the primary reads: "no difference detectable; effects below roughly 600 USD/window
    cannot be ruled out". The number is piped from the JSON through `build_data.py` and `data.json`.
- **Scored without asking first.** The GBDT read-out on FINAL was scored without asking Juji first, under Juji's standing
  instruction. It used the recipe Juji approved twice. Juji was told in the next message and may object.
- **Carry forward to Stage 4.** The paper must say:
  - that no clean data remained for a new test;
  - that these are the same 20 windows read again;
  - that GBDT vs LR is flattered by the λ asymmetry.
- **Stage 1 post-scoring, round 1 (2026-11-19): APPROVE WITH FIXES.**
  - The reviewer verified that the traces were created after both registration commits, that there was exactly one run,
    and that there was no retraining (the pickle's mtime and sha are unchanged). It recomputed every difference and interval
    from the JSON, and found the Phase 23 means unchanged.
  - Fixes: log the realised MDEs and the "scored without asking" sentence (both above).
  - It will still escalate push or publish, deletions, money or credentials, results that overturn the story, and changes
    to settled decisions. Those go to Juji directly.

## Stage 2: dev experiments

### 2.1 Option-order bias: registration

- `fraud/eval_order_bias.py` (its docstring is the registration), committed before scoring.
- It covers LM alone (`lm-rich`) and LM + tree (`lm-signal`), scored in both orders, rev and fwd, on dev30 plus the 88
  held-out chains, keeping all three option scores. It then reports the flip rate, the cost of each order with a paired
  interval, and the averaged orders if the flip rate is above 10% or the interval excludes zero.
- **Smoke test (dev, 32 rows).** Rev reproduced Phase 21's saved decline score exactly (max gap 0.0). In fwd, 1 of 32 top
  answers changed.
- **Assumed:** the revision prompt says "options reversed (A decline, B review, C approve)". That is the order already used,
  so the new order is fwd (Stage 0 finding). Both orders are refitted the same way: p on dev30, λ on 118.

### 2.4 Customer-line trade-off curve (dev), eval 20261119-232409

- `fraud/eval_customer_line.py`, registered in its docstring and committed before running. CPU only; no model pass.
- **Checks:**
  - LR at 0 USD costs 2,125, Phase 21's `tab-logreg-amt`.
  - LM + tree at 0 USD costs 1,920, Phase 21's `lm-signal-amt`.
  - LM + tree at 10 USD: +63 USD/window, and good customers declined fall from 3,241 to 827 (−74%). Both match Phase 22's
    dev figures.
- **Results (dev, 88 held-out chains):**
  - **LR:**
    - At 10 USD, cost moves +3 (−33 to 44), good customers declined fall from 4,003 to 933, and frauds stopped fall from
      2,347 to 1,973.
    - At 5 USD, cost is −28 (−55 to 4).
    - From 15 USD up, LR is distinguishably dearer, by +71 at 15 and +299 at 60.
  - **LM + tree:** from 10 USD up, it is distinguishably dearer, by +63 at 10 and +203 at 60.
  - Good customers declined fall steeply up to about 20 USD, then flatten.
  - **A trade-off, not a gain**, as the revision prompt frames it. The price moves the operating point: fewer good customers
    declined and fewer frauds stopped, for the same or more dollars.
- **Implementation choice:** the policy (p, λ) is fixed across prices, as on FINAL. The figure (frauds stopped against good
  customers declined, with cost annotated) is built in Stage 4 from this JSON.

### 2.3 No-C variants: registration

- `fraud/eval_noc.py` (its docstring is the registration), the `--noc` flag in `fraud/finetune_ieee.py`, and `hist["noc"]` in
  `fraud/ieee_brain.py`. All committed before the retrain and before any no-C model was fitted. The default prompts are unchanged.
- **Assumed:** "without the C counter columns" means no C column anywhere.
  - The revision prompt names C1/C4/C8/C13, the plain prompt's four. The rich prompt that LM alone reads also shows C2, C5,
    C6, C9, C11 and C14, and its context block cites "counters C4 or C8".
  - Leaving those in would make the LM the only no-C setup still reading C. So the retrain removes all ten, and the C clause
    of the context. LR and trees drop C1-C14 and their missing flags.
- LM + tree gets no no-C variant: that would be a second retrain, and the prompt allows one. Logged as a gap.
- Every setup here uses p on dev30 and λ on all 118 dev windows (as parity), so LR's with-C number differs slightly from the
  paper's dev30-λ LR.

### 2.2 Feature parity (dev), eval 20261119-232618

- `fraud/eval_parity.py`, registered in its docstring. 204 features: the rich prompt's fields and its history, nothing else.
- p on dev30 and λ on 118 for every setup.
- Check: `lm-rich` 2,112 and `lm-signal` 1,920 equal Phase 21's dev reads.

| setup | USD/window | AUC |
|---|---:|---:|
| trees, all columns (`tab-all`) | 1,830 | 0.914 |
| LM + tree (`lm-signal`) | 1,920 | 0.910 |
| trees, prompt's fields (`par-trees`) | 2,034 | 0.908 |
| LM alone (`lm-rich`) | 2,112 | 0.893 |
| LR, all columns (`tab-logreg`, λ on 118 here) | 2,117 | 0.888 |
| LR, prompt's fields (`par-logreg`) | 2,327 | 0.874 |

- **LM alone − LR at parity:** −215, interval −353 to −73. **Distinguishable: LM alone is cheaper.**
- **LM alone − trees at parity:** +77, interval −90 to 235. Not distinguishable.
- **LM + tree − trees at parity:** −114, interval −240 to 4. Not distinguishable.
- **Parity − all columns:** LR +210 (83 to 329), trees +204 (82 to 341). Both are distinguishable: the prompt leaves out
  information that both baselines use.
- **Answer to "modelling or information" (dev):**
  - Mostly information. Given only what the prompt shows, the fine-tuned LM models as well as boosted trees and better than
    LR.
  - Its gap to the all-column trees (+282 on these chains) is about what the trees themselves lose when cut to the prompt's
    fields (+204).
  - This is a result in the LM's favour, and it is reported as such. It does not change the headline, which compares with
    the all-column baselines the paper registered. It does change the interpretation: the LM's loss comes from the prompt's
    information budget.
- **Against the prediction stated before running:**
  - "par-logreg about ties LM alone" was wrong: LM alone is cheaper.
  - "par-trees cheaper than LM alone": not distinguishable.

### 2.1 Option-order bias (dev), eval 20261120-045604

- Scoring took 4 passes of about 81–85 min each (118,000 rows per pass).
- **Check:** rev reproduced Phase 21's saved decline scores exactly (max gap 0.0) for both setups.

| setup | flip rate | rev | fwd | fwd − rev (95% CI) | averaged | avg − rev (95% CI) |
|---|---:|---:|---:|---|---:|---|
| LM alone (`lm-rich`) | 0.50% | 2,112 | 2,069 | **−43 (−88 to −6)** | 2,089 | −23 (−59 to 6) |
| LM + tree (`lm-signal`) | 0.97% | 1,920 | 1,893 | −27 (−95 to 15) | not triggered | |

- **The top answer barely depends on order.** "Approve" is the top answer for 92–93% of rows in both orders, and "review" is
  never top.
- **The decline score moves a little.** Its rank correlation across orders is 0.997 (LM alone) and 0.999 (LM + tree). AUC
  is unchanged to 3 decimals, and λ is the same per order.
- **The registered rule triggered the averaged run for LM alone,** because its fwd − rev interval excludes zero. Averaging
  the two orders moves its cost by −23, not distinguishable.
- **Reading:**
  - There is a measurable but small option-order bias in cost: about 40 USD/window, under the ~±100 seed-noise floor.
  - The order the setups were scored in (rev) was, if anything, slightly the dearer one, so the reported LM costs are not
    flattered by order.
  - No verdict changes.
- **Note (paper-driver, Stage 2 round 1).** The registration docstring says "committed before any row was scored here". In
  fact the 32-row dev smoke test (rev reproduction, and 1 of 32 top answers changing in fwd) ran before registration commit
  d0a9f1e and is logged in it. It was dev-only and a pipeline check, so no rule breaks, but the docstring's claim is
  inaccurate. The committed docstring is left unchanged.
- **Against the prediction stated before running:** "the top answer is decline for nearly every row" was wrong. It is
  approve, which is the IEEE-CIS adapters' behaviour, unlike PaySim. The flip rate was small, as predicted. The cost gap was
  within ±150, as predicted.

### 2.3 No-C variants (dev), eval 20261120-074245

- **One LM retrain:** `fraud/models/20261120-045629_lora-d1-ieee-full-rich-noc` (sha256 73f809b8…).
  - The same rows (sha e6d057cc…), steps (5,994), learning rate, seed and LoRA as the rich adapter. Its manifest differs only
    in the prompt-length, loss and timing fields.
  - Final training loss 0.382, against the rich adapter's 0.313.
  - 101 min to train, plus 65 min to score dev.
- The no-C trees (`parity_noc-trees.pkl`) took 918 s to fit. All setups use p on dev30 and λ on 118.

| setup | with C | without C | without − with (95% CI) |
|---|---:|---:|---|
| trees | 1,830 (AUC .914) | 2,091 (.898) | +261 (176 to 351) |
| LM alone | 2,112 (.893) | 2,423 (.858) | **+311 (163 to 437)** |
| LR | 2,117 (.888) | 2,284 (.874) | +167 (53 to 267) |

- **The ranking by dev cost changed.**
  - With C: trees < LM alone < LR. LM alone and LR are tied: 5 USD apart, and Phase 21 found them indistinguishable.
  - Without C: trees < LR < LM alone.
  - Without C, all three gaps are distinguishable:
    - trees − LR: −193 (−312 to −64);
    - LM alone − LR: **+140 (17 to 259)**;
    - LM alone − trees: +332 (181 to 470).
- **Reading:**
  - The C counters help every setup. The fine-tuned LM leans on them most: it loses 311 without them, LR 167.
  - Without C, the LM is distinguishably dearer than LR, where with C it tied. So the C-column threat bears hardest on the LM:
    its tie with LR depends partly on columns whose timing Phase 17 could not clear.
  - This is one retrain on dev, and +140 is just above the ~±100 seed-noise floor.
  - Trees stay cheapest either way.
- **No-C scope.**
  - Not run without C: LM + tree. It would need a second retrain, and the prompt allows one.
  - Left out: LR with two fixed cut-offs, and LR + rule + customer line. They would reuse `noc-logreg`'s scores with a
    different policy, so they add no new model. The ranking question is about the three models.
  - Stage 4 lists all three as gaps of the no-C check.
- **Does it flip the plan? Worker's judgement: no.**
  - The swap is LM alone and LR, from a tie to LM dearer. That strengthens "no detectable saving from the LLM" instead of
    overturning it.
  - Put to paper-driver explicitly at the Stage 2 gate.
- **Against the prediction stated before running:**
  - "Every setup gets dearer, trees and LM alone more than LR": held.
  - "The ranking trees < LR ≤ LM alone holds": with C, LM alone was 5 USD cheaper than LR, a tie. Without C, the predicted
    order held strictly.

### Stage 2 verdict and carry-forward

- **Stage 2, round 1 (2026-11-20): APPROVE WITH FIXES.**
  - The reviewer verified:
    - every registration commit precedes its results;
    - no FINAL contact (only hdev traces);
    - exactly one LM retrain, with all 33 hyperparameter and data keys matching the rich adapter;
    - the parity and no-C feature lists, against the code;
    - all labels `dev`;
    - two numbers recomputed from traces.
  - **Rulings:**
    - the no-C ranking change is **not plan-flipping**: trees stay cheapest, and LM alone vs LR goes from a tie to LM dearer,
      which strengthens the headline;
    - parity is **not plan-flipping**: it answers "model or information" in the LM's favour, and the headline compares with
      the all-column baselines.
    - Neither goes to Juji under the standing decision.
  - Fixes, made in this commit: the 2.1 smoke-test note, the 2.3 scope note, and this carry-forward list.
- **Carry forward to Stage 4.** The paper must:
  - (a) state that the dev ranking changed without C. With C: trees < LM alone ≈ LR. Without C: trees < LR < LM alone, with
    LM alone − LR +140 (17 to 259). LM alone's tie with LR depends on C columns whose timing Phase 17 could not clear, and +140
    is only just above the ~±100 noise floor;
  - (b) report parity as evidence that the LM's gap was mostly information, and as a result in the LM's favour. LM alone −
    parity LR: −215 (−353 to −73). LM alone − parity trees: +77, not distinguishable;
  - (c) report order bias as small but measurable for LM alone: fwd − rev −43 (−88 to −6); averaged −23, not
    distinguishable. Say the setups were scored A decline / B review / C approve, with the forward order as the check;
  - (d) note that Stage 2's LR uses λ on 118 windows (2,117), against the paper's dev30-λ LR (2,125);
  - (e) state that every dev comparison involving an LM setup is flattered by λ on 118 windows;
  - (f) label every one of these numbers `dev`;
  - (g) list the no-C gaps: LM + tree; LR with two cut-offs; LR + rule + customer line.
- **Stage 3 constraints:**
  - no new scoring of FINAL;
  - read-outs only from saved traces 20261104-090649 and 20261119-231754;
  - the registered percentile intervals stay the headline;
  - every analysis is labelled `read-out` in its JSON.

## Stage 3: analyses on existing outputs

- **Registration.** `fraud/eval_readouts.py` (its docstring is the registration), committed before running.
- It reads only the saved FINAL traces, from 20261104-090649 and the read-out 20261119-231754, plus dev scores for the
  price grid's λ.
- **Assumed:** the price grid's customer price is a charged cost per good customer declined (`Costs.decline_flat`, as in
  Phase 16's stress), which the rule therefore also prices.
  - Reason: the grid varies assumed prices, and a price that is never charged would not be a price.
  - The decision-only customer-line setup is left out of the grid, because the grid's customer price replaces its knob.
- **Check built in:** the base cell (10%, 10 USD, 0) must reproduce every setup's saved FINAL cost exactly, or the run stops.

### Stage 3 results (read-out), eval 20261120-075713

- **Runs.**
  - First attempt: it crashed in the seed section (setup names carry `-amt`). Fixed, then run 20261120-075241.
  - Then the "largest-gap frauds" variant was added (see 3.4) and everything was rerun as **20261120-075713, the result of
    record**.
  - 20261120-075241 is kept and superseded. Every section it shares with 075713 is identical (checked).
- **Checks:**
  - every percentile interval equals the registered one, up to orientation (`ev.compare` puts the cheaper setup first);
  - the price grid's base cell reproduces every saved FINAL cost exactly.

**3.1 MDE (read-out).** The headline comparisons' paired SDs are 570–1,013 USD/window, so the MDEs on 20 windows are:
- LM + tree − LR: **524**;
- LM alone − LR: 626;
- LM + tree − GBDT: 607;
- GBDT − LR: 357;
- customer line − LM + tree: 153.

Wording for Stage 4: "no saving detectable at this sample size; effects below about 520 USD/window (LM + tree vs LR) cannot
be ruled out".

**3.2 Robustness (read-out).**
- Block bootstraps (2 and 4) and BCa agree with the registered percentile intervals on every verdict except one: GBDT − LR
  with blocks of 4 gives −349 to −1, which just excludes zero. Percentile, block-2 and BCa all include zero.
- That block-4 bound sits at the Monte Carlo margin: another seed gives an upper bound of about +1. The paper reports it as
  borderline, not as distinguishable.
- The headline stays the registered percentile: not distinguishable. The block-4 result is reported as such.
- LM alone − LM + tree is distinguishable under every method: 103 to 972; BCa 154 to 1,088.

**3.3 Ranking (read-out).**

| setup | ROC-AUC (95% CI) | PR-AUC (95% CI) |
|---|---|---|
| LR | .853 (.820–.880) | .374 (.270–.474) |
| LM alone | .859 (.830–.885) | .346 (.238–.456) |
| GBDT alone | .878 (.852–.902) | .448 (.337–.548) |
| LM + tree | .883 (.860–.905) | .446 (.339–.544) |

Base rate 3.15%. LM alone's PR-AUC is below LR's even though its ROC-AUC is above it: it ranks the top of the list worse.

**3.4 AUC vs cost (read-out).** LM alone costs +321 against LR, but its ROC-AUC is .859 against .853.

- **Cost by action, USD/window:**
  - LM alone lets through more fraud dollars: 1,508 against LR's 1,305 (+203).
  - It also declines more good customers (471 against 436) and spends more on reviews (448 against 364).
  - The gap is in all three parts, mostly missed fraud.
- **The largest frauds by amount are not the cause.**
  - The two largest FINAL frauds (2,480 and 1,651 USD) were reviewed by both setups. Removing them leaves +321 (−74 to 774).
  - **Post-hoc, added after the registered result; not registered.** The 1 and 2 frauds LM alone priced worst against LR
    (1,504 and 1,045 USD) were approved by LM alone and reviewed by LR. Removing them leaves +246 (−107 to 666) and +194
    (−151 to 607).
  - The registered result is the by-amount removal: +321, unchanged.
  - So the gap is spread across many payments, not one or two.
- **Reliability tables and amount bands** are in the JSON, for the figure. The likely mechanism (Inferred): PR-AUC shows the
  LM's top ranks are worse, so the review budget and declines are spent on the wrong payments, and the costliest misses are
  mid-to-large frauds it approves outright.

**3.5 Price grid (read-out, 27 cells, λ re-picked on dev per cell).**
- GBDT alone is cheapest in **24 of 27** cells and LM + tree in 3.
- In those 3 cells, LM + tree minus GBDT is −175 (5%, 10, 30), −26 (20%, 5, 10) and −48 (20%, 5, 30). The registration
  computed no interval for these, so they are mean gaps only. (Corrected: an earlier line called them "within noise",
  which was not checked.)
- The cheapest setup is distinguishable from LR in **6 of 27** cells, all of them GBDT: (5%, 5, 0), (5%, 5, 10), (5%, 10, 10),
  (10%, 5, 0), (10%, 5, 10) and (10%, 10, 10). (Corrected from "7" by paper-driver.)
- In the 3 cells where LM + tree is cheapest, it is not distinguishable from LR: −319, −178, −74, with intervals that
  include zero.
- LM setups were not compared with LR by interval in the other 24 cells. That comparison was not registered, so no claim is
  made there.
- The d8a3246 commit message's "no LM cell beats LR" is superseded by this.
- This replaces the "+7 USD tie" sentence.

**3.6 Disparity check on proxies (read-out, not a fairness audit).**
- The groups differ far more than the setups do:
  - frauds missed are highest on debit cards (41–56%), on payments with no device recorded (49–64%) and on yahoo.com
    (52–65%), with every setup included;
  - good customers declined are highest on credit cards (9–14%) and on mobile (15–20%) or desktop (8–17%) devices,
    excluding the customer-line setup. That setup declines only 2–5% of good customers in those groups (credit 3.1%,
    mobile 5.1%, desktop 2.2%) and misses more fraud.
- Among the setups without the customer line, the groups differ more than the setups do. The customer line moves the
  good-declined rate by far more than any model choice. Notable: LM alone declines 17.1% of good desktop customers,
  against LR's 10.4%.
- Small groups have wide intervals: American Express has 5 frauds; aol.com has 15.

**3.7 Seed noise (dev).** Two seeds of the same LM recipe differ by −102 USD/window (−205 to 6), from Phase 21. The noise
floor used is about ±100 USD/window.

### Stage 3 verdict and carry-forward

- **Stage 3, round 1 (2026-11-20): APPROVE WITH FIXES.**
  - The reviewer verified:
    - registration before results, and no FINAL scoring (120 FINAL traces, unchanged);
    - the registered intervals, unchanged;
    - the base-cell reproduction;
    - the labels.
  - It recomputed independently: LM alone − LR, MDEs, AUC and PR-AUC, cost by action, and the largest-gap frauds.
  - Fixes, all made in this commit (nothing recomputed): 7 → 6 grid cells; the unsupported "no LM cell beats LR" replaced; the
    largest-gap variant flagged as post-hoc; the block-4 bound called borderline; the disparity ranges corrected.
  - Not plan-flipping.
- **Carry forward to Stage 4.** The paper must:
  - use the MDE wording: "effects below about 520 USD/window (LM + tree vs LR) cannot be ruled out"; about 600 for LM + tree
    vs GBDT; about 630 for LM alone vs LR;
  - keep the registered percentile intervals as the headline; block-4 GBDT − LR is borderline;
  - let the price grid replace the "+7 USD tie" sentence. The grid's customer price is a charged cost, not the
    customer-line knob. Make no claim about LM vs LR in cells where it wasn't tested;
  - label the largest-gap removal post-hoc and unregistered;
  - call the disparity section a check on proxies, not a fairness audit;
  - use a seed-noise floor of ±100 USD/window (dev);
  - pass every Stage 3 number through the JSON → `build_data.py` → `data.json`, with its label.

## Stage 4: rewrite of the paper

**What changed.**
- `paper/main.typ` is rewritten against the settled decisions:
  - **framing:** the new title, the narrowed research question, the three contributions, and the honest gap statement;
  - **the headline:** rule, then null, then read-out, in the abstract (195 words, 3 numbers) and in the conclusion;
  - **PaySim** is a short pipeline-development subsection. The 190,910 USD LR figure is gone, and PaySim is out of the
    abstract and the contributions;
  - **related work:** four themed subsections;
  - **technical details:** the temperature, the loss, the model change and the λ asymmetry;
  - **results:** the trade-off framing, the GBDT read-out, MDE wording, parity, no-C, order bias, AUC/PR-AUC and the
    AUC–cost diagnosis;
  - **limitations:** generalisability, the context block and the C columns rewritten;
  - **future work**, and an **acknowledgement**.
- A Discussion section is added, as an Implementation choice, to carry the interpretation the prompt asks for.
- **Numbers:**
  - every new number passes from its results JSON through the `rev` block of `paper/build_data.py` into `data.json`;
  - `paper/revision_manual.json` holds three sourced protocol constants;
  - `fraud/dev/prompt_example.py` writes `fraud/results/prompt_example.json`, a fit row for the prompt appendix;
  - `paper/gen_tables.py` writes the trade-off figure and the appendix tables with literal `v()` keys.
- **Tests:** `tests/test_paper.py` is updated to the new contract:
  - 8–14 pages;
  - 30 references in order of first citation;
  - the settled phrases and terms;
  - no course traces;
  - the generated block current.
  - 492 tests pass.
- **Typst:** 0.15.1 had gone from this machine and was reinstalled with winget (`RUN-GUIDE.md`).
- **Length:** 10 pages in all. The main text runs to page 6, references to page 7, and appendices A–H follow on pages 8–10.
  - The main text is shorter than "about 8 pages". I did not pad it.
  - The prompt puts the robustness, diagnostic, grid, disparity, parity, no-C and order-bias material in the appendices.

**References: 30, each verified against a primary source** (2026-11-20).
- **Crossref metadata:** Dal Pozzolo 2018, Correa Bahnsen 2013, Nosek 2018, Efron 1987, Künsch 1989, Kaufman 2012.
- **ACL Anthology:** PET (EACL 2021, pp. 255–269), LM-BFF (pp. 3816–3830), Pezeshkpour (pp. 2006–2017), Dror (pp. 1383–1392),
  and Tan, Ma, Zhang (now ACL 2026, pp. 23375–23390; earlier cited as arXiv:2512.13040).
- **PMLR:** TabLLM (PMLR 206, pp. 5549–5581) and Guo (PMLR 70, pp. 1321–1330).
- **IJCAI proceedings page:** Elkan, pp. 973–977. Many secondary sources say 973–978; the official page is used.
- **ICLR virtual pages:** Robinson & Wingate (ICLR lists two authors; arXiv adds Rytting), TabPFN, Zheng, LoRA.
- **NeurIPS pages:** LIFT (nine authors), Grinsztajn.
- **Project Euclid:** Friedman 2001.
- **Springer and Birkbeck:** Whitrow 2009.
- **TMLR via MLAnthology:** Fang 2024. TMLR lists eight authors; arXiv v4 lists ten. The TMLR list is used.
- **EMSS 2016 proceedings page:** PaySim. No page range is given there, so none is cited.
- **Platt:** MIT Press volume; sources give 1999 or 2000, and 2000 (the book's year) is used.
- **Other primary sources:**
  - Wirth & Hipp: the proceedings listing, pp. 29–39, 2000;
  - Nilson Report no. 1276 (Dec. 2024): the issue PDF;
  - Qwen2.5: arXiv 2412.15115;
  - d1-Omni-2B: the Hugging Face model card (LiquidAI, LFM Open License v1.0);
  - IEEE-CIS: the Kaggle competition (IEEE CIS and Vesta, 2019).
- **Unverified references: none.**
- **Moved:** System One (TypeSafe; Goedecke) is now a footnote crediting the inspiration, not a numbered reference.

**Decisions not fully met, with the reason.**
- **"About 8 pages of main text":** about 6.5 including references. See above.
- **Footnote 1 as a link to the public repo:** deferred to Stage 5, which prepares that repo. The footnote still lists the
  results files.
- **The PDF file name** (`transforming-llms-fraud.pdf`) predates the new title. It is kept because earlier plan documents
  link to it.

### Stage 4 verdict and fixes

- **Stage 4, round 1 (2026-11-20): APPROVE WITH FIXES.**
  - The reviewer read all 10 pages and confirmed `data.json` equals `render()`.
  - It traced 9 numbers to the source JSONs: 1,897, 524, .346, +140, 24 of 27, 825, 3,853, 0.98, 56.
  - It found no change to `fraud/results` other than the fit-row prompt example.
  - It confirmed every settled decision and every carry-forward is present.
- **Fixes, all made in this commit:**
  - **Abstract:**
    - "no change of model was distinguishable" was false: LM alone − LM + tree is +484 and distinguishable, and LM alone −
      GBDT +530. It now reads "no model was distinguishably cheaper than logistic regression".
    - The 825 is said to come from earlier test windows.
    - The 524 is labelled read-out.
    - "tied boosted trees" became "not distinguishable from boosted trees".
  - **VI-A:** "several times larger than any difference between models" was false.
    - It is replaced by a comparison through `v()`: the rule's 825 against the largest model difference from LR on the
      final test, with the different windows stated.
    - The mechanism is now marked as inferred.
  - **Conclusion:** reordered to rule (earlier test windows, the customer line as a trade-off), then the held-out null,
    then the GBDT read-out, then dev parity, stated as "not distinguishable … mostly information".
  - **V-E and VI-C:** "tied" and "as well as" became "not distinguishable", with the note that this is not evidence of
    equality.
  - **III-B:** the three labels now cover final-test and revision numbers. Numbers from earlier test windows are marked as
    such; V-B, V-F and VII-A already say so.
  - **Appendix preamble:** the analyses were registered in their scripts. Only the Appendix A percentile intervals belong
    to the locked test.
  - **MDE constants:** 2.8, 80% and 5% now pass through `paper/revision_manual.json` (with its source) into `data.json`.
  - **Fig. 2 (craft):** the LM + tree 60 USD note is dropped, so it no longer sits on the LR curve.
- **Rulings, settled from now on:**
  - The main-text length is accepted, without padding. It is listed in the Stage 6 not-implemented list.
  - Footnote 1 as a repo link is deferred to Stage 5, and required there.
  - The PDF file name is kept.
  - The Discussion section is accepted.
