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
  - the Phase 20 tree score cached in `fraud/data/ieee-cis/signal.npz` (fit, dev, the FINAL rows and about 17% of test rows);
  - the history features that `fraud/history.py` builds from every row.
- The dev split is used up in the same way: 30 dev windows fitted the rule and the lines, and the 88 held-out dev chains are
  the rest. 108 dev rows are left over.

**Consequence: Stage 1 takes the fallback branch.** GBDT alone with the rule is scored on the 20 FINAL windows as one labelled
read-out, and the paper says that no clean data remained for a new test.

- The FINAL pre-registration excluded trees ("Trees alone aren't run here", `fraud/eval_final.py`), so this is a new number.
- No inference is needed: `signal.npz` already holds the Phase 20 `tab-all` score for every FINAL row (Observed: 100% non-missing).

### 2. Feature sets of the baselines

Observed, from `fraud/eval_ieee.py:features_all`, `fraud/eval_tabular.py:SETS` and `fraud/ieee_brain.py`.

- **LR on FINAL** (`final-logreg-amt` = `tab-logreg` = `ieee-logreg-all`) and **GBDT** (`tab-all`, 400 trees, `fraud/gbdt.py`)
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
    the LM setups and also `tab-all` (`fraud/eval_lm_nuc.py:199`, `fraud/eval_final.py:policies`).
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
