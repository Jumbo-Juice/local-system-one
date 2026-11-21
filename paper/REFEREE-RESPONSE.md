# Referee revision: change log (2026-11-21)

A writing and claim-scoping pass on `paper/main.typ`, from an independent review. No experiment was run, no model trained,
and no result changed. Every number is still read through `v()` from `paper/data.json`. New numbers come from committed
results or from `paper/revision_manual.json` → `claims`, and each one has its source there.

Line numbers refer to `paper/main.typ` after the revision. Sections use the PDF numbering. Commits: `468365f` (data keys),
`bc322bf` (text), `000cc7b` (title), then the abstract trim, the PDF and this log.

Before changing anything, I checked each point against the repo. "Referee right" means the source confirmed the finding.

## Major

### 1. Test-set integrity. Referee right; the text overstated it.

- **What the repo shows.** Sources: `fraud/results/window_audit.json`, `fraud/final_test.py` docstring, `REVISION_LOG.md` Stage 0.
  - Before the final test, 98 test windows had been scored:
    - 60 windows of 1,000 consecutive payments, in three evaluations of 20;
    - then 38 *chains*: 1,000 untouched rows in arrival order, skipping rows that were used before.
  - The cost rule's result (−825) comes from 20 of the consecutive windows.
  - GBDT against LR (−97, n.d.) comes from 10 of the chains.
  - The 20 final windows were reserved on 2026-11-03, after the first 60 windows and before any chain. The draw:
    - the untouched rows were cut into 1,000-row slots;
    - the slots, in time order, were cut into 20 equal blocks;
    - one slot was drawn from each block with seed 20261105.
  - The audit found 0 test rows scored twice, so the final windows are disjoint from every earlier one.
  - Earlier test results did steer what went forward: the rule was adopted after its test result, and the prompts and
    setups followed the earlier phases.
- **Changed.**
  - III-A (l. 522): "the test part is never tuned on" now reads "no model or decision parameter is fitted on the test
    part".
  - III-A (l. 526): a new paragraph, *Which test windows were used when*, with the counts, the draw and the disjointness.
    It says plainly that what was tested was chosen partly on earlier test results.
  - Abstract (l. 32): "Every choice was made on development data" is replaced with the same facts in short.
  - VII-C (l. 629): this selection is listed as a source of optimism.
- **No justification was added** beyond what the code and the log record.

### 2a. "The decision rule matters more than the model". Referee right.

- **The baseline is now defined** (IV-A, l. 544).
  - It is two fixed cut-offs on the score: review above one, decline above a higher one.
  - Both were picked to minimize total cost on 10 dev windows (`fraud/eval_ieee.py` `pick_thresholds`, `DEV_WINDOWS = 10`).
  - Reviews past the budget are approved.
- **V-B is retitled** "The decision rule on earlier test windows" (l. 576). It now states:
  - the −825 is for LR's scores only;
  - with the scores of the language model of that time (the Phase 12 adapter), the rule came out at −397 (−937 to 14),
    not distinguishable. Source: `eval_ieee_amount` verdict "second";
  - re-picking the cut-offs on the rule's own 30 dev windows changed their cost by −7 (n.d.), so fewer dev windows don't
    explain the saving;
  - it depends on prices;
  - it comes from windows other than the final test, and the two were never compared on the same windows.
- **The headline wording is scoped** in the Abstract, Contributions (ii) (l. 499), "In short" (l. 501), VI-A (l. 609)
  and the Conclusion (l. 657).
- **A related error is corrected in VII-A** (l. 623). The old text said the rule's saving "stopped being distinguishable
  from 30 USD per customer". `fraud/RESULTS-COST-STRESS.md` shows that holds only when the rule is *not told* the price.
  Refitted to the price, it stays distinguishable up to 200 USD, the highest price tested. The text now says both.

### 2b. "Its deficit is mostly information". Referee right. The claim is withdrawn.

- The new wording, in the Abstract, V-E (l. 596), VI-C (l. 615) and the Conclusion: on dev, with LR and GBDT limited to
  the prompt's fields, the language model was cheaper than LR (−215, distinguishable) and not distinguishable from GBDT
  (+77). So missing information *may explain part* of its dev deficit.
- V-E adds why this can't explain the held-out gap:
  - with every column, LM alone ran level with LR on dev (−13);
  - yet it cost +321 more on the final test;
  - the information each side saw was the same on dev and on test.

### 3. Baseline asymmetry. Referee right on the substance; one detail corrected.

- **The detail.** The constant picked on 30 windows against 118 is **λ**, the price of a review in the rule. It is not
  a regularization constant.
  - LR's L2 penalty is fixed at 1e-3 and never tuned (`fraud/baselines.py`).
  - The paper now says so (IV-B, l. 549) without typing the value.
- **Disclosed in three places:**
  - Method IV-A (l. 544): which setups used 30 windows and which used 118, and why. λ picked on 30 windows proved
    unstable for trees (`fraud/eval_tabular.py` docstring).
  - IV-G (l. 566).
  - VII-C (l. 629), which lists the claims it affects:
    - the held-out LM + tree and LM alone comparisons;
    - the GBDT read-out;
    - the price grid;
    - every dev comparison.
- **The size on dev.** VII-C gives it: picking LR's λ on 118 windows moves it from 10 to 5, and LR's cost from 2,125 to
  2,117 USD per window, in-sample. The effect on the final test was **not measured**; see the open questions below.

### 4. Statistics. Referee right.

- **Seed variance.** III-D (l. 537), V-D (l. 588), VII-G (l. 641), the Abstract, Fig. 3, and the Table III and
  Appendix A captions now say that intervals resample windows only and include no training-seed variance. Two dev seeds
  differed by −102, and each setup was trained once.
- **The MDE.**
  - It is now described as post hoc, computed from the paired SD observed on the final test. It is not a planned power
    (III-D; V-D; Appendix A caption).
  - "Effects below about 524 USD cannot be ruled out" was also *wrong* as written: the registered interval already rules
    out savings above 462 USD.
  - V-D now gives the interval's own bounds (462 / 238) and reads the MDE as "a test of this size would often miss true
    differences smaller than these".

### 5. Open methodological questions

| question | answer in the repo | where in the paper |
|---|---|---|
| Were the GBDT scores shown to the LM in training out-of-fold? | **Yes.** The fit rows were cut into 4 time-ordered blocks, and each block was scored by trees fitted on the other three, so no training row saw its own label. A block's trees did see *later* rows (`fraud/lm_signal.py` docstring). On dev and test rows, the score comes from the trees fitted on all fit rows. The score is the first 400 of GBDT's 800 trees. | IV-D, l. 557 |
| How is "same card" defined? | A key: the combination of card1, card2, card3, card5, card4 and card6, with a missing value counted as a value (`fraud/history.py`). An address is addr1 + addr2. "Other cards at the address" counts distinct card1 values. Only earlier rows count. IEEE-CIS has no card id, so different real cards can share a key. | IV-D, l. 557; VII-J, l. 650 |
| How were the IEEE-CIS labels built? | **Not documented in the repo.** `fraud/DATA-IEEE.md` only says the `isFraud` column is used as given. | New limitation VII-J, "What the data leave open" (l. 650). No description was added. |

## Minor

| point | what changed | where |
|---|---|---|
| Units in cost tables | Table II: mean, interval and differences are USD per window; good customers declined and frauds stopped are totals over the 20 windows. Fig. 2: cost per window, axes are totals. Appendices F and G: USD per window. Table III and Appendix B already said so. | l. 196; `gen_tables.py` → Fig. 2, App. F, G |
| 1/1.07 shown as 0.94 | The worked example now shows 1 / 1.066 ≈ 0.938, with three digits from the raw values (`calc.round`), and says it appears as 0.94 elsewhere. 0.94 is correct for the unrounded sum; the 1.07 was the rounded one. | IV-C, l. 552 |
| Fig. 3 omits a setup in Table II | "LM alone, customer line (read-out)" is added as a second open dot, and the caption names both read-outs. | l. 163, 168 |
| The LM's fraud score for Platt scaling | Defined: the softmax score of the decline option. For LR and GBDT, the predicted fraud probability. | IV-A, l. 542 |
| How d1-Omni-2B option scores are obtained | The last hidden state at the answer position, dotted with each letter's token embedding (the larger of "A" and " A" when the latter is one token), divided by the shipped temperature 3.383, then a softmax over the three (`fraud/d1torch.py` `logits`). | IV-C, l. 554 |
| "Frozen check" without numbers | Now given (dev): ROC-AUC .985 for d1-Omni-2B against .903 for frozen Qwen on 2,194 PaySim dev rows. Also added: below LR's .992 (the gate it failed); 91 ms p90; LFM3-1.7B took 454–1,061 ms in a smoke test. | IV-F, l. 563 |
| Claims without numbers | "On dev did no better than the score alone" now reads 1,920 vs 1,853 USD per window, not distinguishable (VII-F, l. 639). "Against far less for LR" (latency, never measured comparably) was **removed** (VIII, l. 654). | |
| The two LR baselines, 2,125 and 2,117 | Same LR and the same 88 dev windows. λ = 10 picked on 30 windows (the paper's baseline) against λ = 5 picked on 118 (Appendices E–F). Explained in the Appendix E caption, the Fig. 2 caption and VII-C. | |
| Acronyms at first use | LoRA, LIFT and LM-BFF are expanded in Related Work, CRISP-DM in III-A, SD in III-D, ROC-AUC in IV-F, PR-AUC in V-E, LM in the Abstract. The Appendix A and B captions are expanded too. | |
| File-path footnote | Moved to an unnumbered *Data and Code Availability* section before the Acknowledgement, at 8 pt and unjustified. The System One footnote stays, because it credits an idea and holds no path. `tests/test_paper.py` checks the new place. | l. 659 |
| "LLM" in the title, "small language model" in the body | The title is now "Can a Small Fine-Tuned **Language Model** Make Fraud Decisions?". README, REPRODUCE, RUN-GUIDE and the paper test follow. Commit `000cc7b` holds this change alone. | l. 24 |
| American spelling | serializes, minimizes, neighboring, toward, labeled, anonymized, modeling, Labeling, Generalizability, Gray. The reference title "CRISP-DM: Towards…" is kept as published. | |

Other changes:

- The abstract was trimmed to 249 words, within IEEE's limit.
- In V-E, "with C columns LM alone and LR tie" now reads "are not distinguishable", so a null is not read as a tie.

## Checks

- The PDF compiles with no errors. The only warnings are the 8 known font warnings (`paper/README.md`, Fonts).
- The PDF has 11 pages; it had 10. The main text runs to page 7.
- There are no `??` in the PDF text. Every `@`-reference resolves; Typst fails the build otherwise.
- `pytest -m "not model"`: 492 passed. This includes:
  - `data.json == render()`;
  - every key resolves;
  - the generated block matches `gen_tables.py`;
  - the paper's wording rules.
- **Number trace.** No number in the prose is typed. Each is a `v()` key from a committed results JSON, or from a sourced
  manual entry. The only typed numerals left:
  - "90th percentile", a label for the p90 values that come from data;
  - the 2026 dates in the System One footnote.

## Numbers I could not trace to a results JSON (traced to another file instead)

All of these are in `paper/revision_manual.json` → `claims`, with their sources.

| number | where it comes from |
|---|---|
| 454–1,061 ms (LFM3-1.7B smoke latency) | `fraud/RESULTS-LFM.md` only; not in `eval_lfm/*.json` |
| 200 USD (the highest price at which the refitted rule stays distinguishable) | `fraud/RESULTS-COST-STRESS.md` only (its results JSON was not wired into the paper build) |
| 10 cut-off dev windows | code constant `fraud/eval_ieee.py` `DEV_WINDOWS` |
| 4 out-of-fold blocks | code constant `fraud/lm_signal.py` `FOLDS` |
| 400 trees in the tree score | code constant `fraud/lm_signal.py` `TREES` |

## For the author to decide

1. **The title.** It was a settled decision of the earlier revision. Revert `000cc7b` if "LLM" should stay. If it does,
   the body needs to say "LLM" for a 2B-parameter model, which the referee would query.
2. **IEEE-CIS label construction.** The repo doesn't document how the provider assigned `isFraud`. If you have a source
   you can cite, add it to VII-J. I added none, because I couldn't verify one from the repo.
3. **Currency of amounts.** `fraud/DATA-IEEE.md` records "USD" as unverified, from memory of the competition page.
   VII-A now says so.
4. **The λ asymmetry on the final test.** Its effect on the held-out comparisons was not measured. LR's λ refitted on 118
   dev windows could be applied to the saved final-test scores. That needs no model run, but it is a new, unregistered
   read-out, so I didn't run it under this brief's no-rerun rule. Say if you want it registered and run.
5. **Seed variance on held-out data.** Quantifying it needs retraining, so it was not done. The paper now states the
   omission.
6. **Length.** The PDF grew from 10 to 11 pages, and the main text from about 6.5 to about 7 pages.
7. **References.** No citation was added or changed. The 30 references were verified in the earlier revision
   (`REVISION_LOG.md`, Stage 4), and I didn't re-verify them online.
