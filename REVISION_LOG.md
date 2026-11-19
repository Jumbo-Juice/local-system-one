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
