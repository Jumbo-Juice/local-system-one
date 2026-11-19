# Revision prompt — "Transforming LLMs into Classifiers for Fraud Detection"

Paste everything below the line into Claude Code, run from the repo root.

---

You are revising a research paper and its experiment repo. The repo has
`fraud/` (experiments and results), `paper/` (Typst source; every number
comes from `paper/data.json`, which `paper/build_data.py` builds) and
`slides/`. The paper is a 7-page IEEE-style evaluation of a LoRA-tuned
small LLM (d1-Omni-2B on IEEE-CIS; Qwen2.5-1.5B on PaySim) used as a
single-pass three-way fraud decider (approve / review / decline), scored
by dollar cost per 1,000-payment window against logistic regression (LR)
and boosted trees (GBDT), on a pre-registered, locked 20-window final test.

The revision is for a **portfolio version** of the paper. All decisions
below are settled. Do not re-open them. Where something is unclear, pick
the reading most consistent with these decisions, note it in
`REVISION_LOG.md`, and carry on.

## Hard rules

1. **Never re-score or tune on the 20 final-test windows** except for the
   read-outs explicitly listed below, which use already-saved outputs or
   score a fixed, untuned setup. Label every such result *read-out*.
2. **Label every new number** as one of: `dev`, `held-out (pre-registered)`,
   or `read-out`. The label must appear in the results JSON and next to the
   number in the paper.
3. **`data.json` stays the single source of numbers.** Every new number goes
   into a results JSON, then into `build_data.py`, then into `data.json`.
   Nothing is hard-coded in `.typ` files.
4. **Pre-registration means a git commit before scoring.** The commit must
   contain the registration file, and its timestamp must precede the
   results file's.
5. **Report what you find, including nulls and results that undercut the
   paper.** Never soften a result to fit the story.
6. Commit after each stage with a clear message. Keep a running
   `REVISION_LOG.md`: what changed, why, which decision it implements, and
   anything you had to assume.
7. **Do not add new model families or training methods** (no adversarial
   training, no GANs, no synthetic data, no classification-head ablation).

## Stage 0 — Audit (report back and STOP before Stage 1)

1. Read every results JSON referenced in footnote 1 of the paper and in
   `slides/build_data.py`. Build the exact list of IEEE-CIS test-part
   windows (1,000 payments each, time order) that **any** run has ever
   scored: the final test, earlier tests, the column audit, the rule tests,
   the tabular tests and any others you find. List the windows never scored.
   Save this to `fraud/results/window_audit.json` and report the count.
2. Confirm the feature sets: which columns LR and GBDT were trained on, and
   whether they received the card-history aggregates the rich prompt uses.
3. Locate: the code that sets the 3.383 temperature, the plain and rich
   prompt templates (including the hand-written "patterns in the fit rows"
   context block), and the λ selection code for each setup (the LM setups
   used more dev windows than LR).
4. Check the d1-Omni-2B and Qwen2.5 licences for publishing LoRA adapters
   (for information only; adapters are not being published).

Report the audit results, then wait for confirmation.

## Stage 1 — Held-out test (criterion 4)

**If Stage 0 found ≥ 20 untouched windows**, use **all** of them:

1. Write `fraud/prereg/heldout_v2.md` and commit it **before** scoring
   anything. It must state the exact window list and count; the setups; the
   comparisons; the decision rule (a difference is distinguishable only if
   its 95% paired window-bootstrap percentile interval, 10,000 resamples,
   excludes zero); and the minimum detectable effect from the dev paired SD.
   - Setups, all with the cost rule (λ re-picked on dev exactly as before):
     GBDT alone; LM + tree score; LM alone; LR; LR with two fixed cut-offs;
     LR + rule + customer line.
   - **Primary:** LM + tree score vs GBDT alone.
   - **Secondary:** LM alone vs LR; GBDT alone vs LR; LR + cost rule vs LR +
     two fixed cut-offs; LR + rule vs LR + rule + customer line.
   - Use the existing trained adapters and models. No retraining.
2. Score each setup once. Save to `fraud/results/eval_heldout_v2/`.

**If fewer than 20 untouched windows:** score GBDT alone + rule on the 20
final-test windows as a single labelled read-out, and record in the paper
that no clean data remained for a new test.

## Stage 2 — Dev experiments (criteria 3, 4, 6)

All on the 88 IEEE-CIS dev windows, with paired window-bootstrap intervals
against the matching existing setups.

1. **Option-order bias.** Re-score the trained LM-alone and LM + tree
   setups with the options reversed (A decline, B review, C approve). Report
   the top-answer flip rate and the cost of each order. **If the flip rate is
   > 10% or the cost-gap interval excludes zero**, also score with both
   orders averaged and report whether averaging changes cost.
2. **Feature parity.** Train LR and GBDT on exactly the fields the rich
   prompt shows, plus the same card-history aggregates. Compare them with
   LM alone and LM + tree on dev. The question is whether the LLM lost on
   modelling or on information.
3. **No-C-column variants.** Run every setup without the C counter columns:
   retrain LR and GBDT, and do **one** LM-alone LoRA retrain with C1/C4/C8/C13
   removed from the prompt (same hyperparameters). Report whether the
   ranking of setups by dev cost changes.
4. **Customer-line trade-off curve.** For LR and LM + tree, sweep the
   customer price from 0 to 60 USD (step 5) using existing dev scores.
   Plot frauds stopped against good customers declined, with total cost
   annotated.

## Stage 3 — Analyses on existing outputs (criteria 4, 5, 6)

No new scoring of the final test. Use saved final-test outputs. Label
everything here *read-out*, except intervals that recompute registered
results.

1. **Minimum detectable effect** for every final-test comparison, from the
   paired SD. Replace wording like "no saving" with "no saving detectable at
   this sample size; effects below X USD/window cannot be ruled out."
2. **Bootstrap robustness table.** The registered percentile intervals stay
   the headline. Add a block bootstrap (blocks of 2 and 4 windows) and BCa
   intervals for every comparison.
3. **Ranking metrics.** Window-bootstrap 95% intervals for ROC-AUC and
   PR-AUC for every setup.
4. **AUC-vs-cost diagnostic.** Why does LM alone rank like LR (AUC .859 vs
   .853) but cost +321 USD/window? Produce a reliability plot per setup,
   cost broken down by action and by amount band, and the LM-alone vs LR
   difference with the 1 and 2 largest frauds removed.
5. **Price grid.** Vary the decline rate r ∈ {5, 10, 20%}, review fee
   c_r ∈ {5, 10, 25} USD and customer price ∈ {0, 10, 30} USD. For each
   combination, re-pick λ on dev, apply it to the saved final-test scores,
   and record which setup is cheapest and whether it is distinguishable from
   LR. Show it as a heatmap. This replaces the standalone "+7 USD tie"
   sentence.
6. **Proxy-group disparity check.** Rate of good customers declined and of
   frauds missed, with intervals, by card type, device type and the top
   email domains (others pooled), for every setup. Call it a disparity check
   on proxies, not a fairness audit.
7. **Seed noise.** Bring the existing two-seed dev result (−102 USD/window,
   −205 to 6) into Results and use about ±100 USD/window as the stated noise
   floor when interpreting differences.

## Stage 4 — Rewrite the paper

### Framing (criterion 1)
- **New title**, evaluation-style, e.g. *"Can a Small Fine-Tuned LLM Make
  Fraud Decisions? A Locked, Cost-Based Evaluation."*
- **Research question** narrowed to decision cost under assumed prices:
  does a small fine-tuned LM, used as a single-pass three-way classifier,
  lower decision cost against logistic regression and boosted trees, and
  where does any difference come from: the model, the information it sees,
  or the decision rule? Remove "is it worth it in dollars"; compute and
  running cost go to Future Work.
- **Contributions:** (i) a locked, pre-registered, dollar-costed evaluation
  protocol and its results; (ii) evidence that the decision rule changes
  outcomes more than the model; (iii) an honest account of biases and
  limitations. Letter scoring + LoRA is presented as an **existing
  technique applied**, not a contribution.
- Delete "This is the part that is new" and the stacked gap sentence ending
  Related Work. Replace the latter with an honest gap: no locked,
  dollar-costed, pre-registered comparison of fine-tuned small LMs against
  linear and tree baselines on real payments.
- **Headline in the abstract and conclusion:** per-payment expected-cost
  pricing changed outcomes more than any model; the fine-tuned LLM showed
  no detectable saving over logistic regression. Then the held-out result.
- **PaySim** becomes a short "pipeline development" subsection: training
  works (trained model 830 vs hand-written rules 808 USD/window, same action
  on 3,853 of 3,862 payments) and the option-order bias demonstration.
  **Drop the 190,910 USD PaySim LR figure.** Remove PaySim from the
  contributions and the abstract.
- Classification-head ablation: one line in Limitations/Future Work.

### Related Work (criterion 2), about 25–30 references
Themed subsections:
1. **LLMs on tabular data:** TabLLM (zero-shot and fine-tuned), Tan et al.,
   plus one or two sentences naming untested alternatives: TabPFN, LLM
   embeddings fed into GBDT, large zero-shot LLMs.
2. **Prompt-based fine-tuning and verbalizers:** PET (Schick & Schütze
   2021), LM-BFF (Gao, Fisch & Chen 2021), LIFT (Dinh et al. 2022),
   Robinson & Wingate 2023, Pezeshkpour & Hruschka 2024.
3. **Cost-sensitive fraud evaluation:** Dal Pozzolo et al. 2018; Elkan
   2001; Correa Bahnsen et al. 2013 (Bayes minimum risk for card fraud).
   Present the cost rule as a standard Bayes minimum-risk rule with a review
   budget.
4. **Tree ensembles on tables:** Grinsztajn et al. 2022.

Move ref [13] (System One / TypeSafe / Goedecke) to a footnote saying it
inspired the project. Credit the technique to Robinson & Wingate and PET.
In Section III add 2–3 citations for pre-registration (e.g. Nosek et al.
2018) and the paired bootstrap / significance testing (Dror et al. 2018;
Efron 1987 for BCa; Künsch 1989 for the block bootstrap).
**Verify every reference** (authors, title, venue, year, pages) against a
primary source. Mark anything you cannot verify in `REVISION_LOG.md`
rather than guessing.

### Technical details to document (criterion 3)
State where the 3.383 temperature came from, why IEEE-CIS used a
three-letter loss and PaySim a full-vocabulary loss, why the model changed
from Qwen to d1-Omni-2B, and that λ for the LM setups used more dev windows
than LR's. Note the hand-written context block as an untested human
feature-engineering step in Limitations.

### Results and interpretation (criterion 5)
- The customer line is a **trade-off, not a gain**: "a cheap knob that moves
  the operating point". Use the Stage 2 trade-off figure.
- Report the held-out test (or the read-out), feature parity, order bias,
  no-C results, AUC/PR-AUC intervals and the AUC-vs-cost diagnosis.

### Limitations and reproducibility (criterion 6)
- Generalisability: one merchant, one period, one region; no second dataset.
- The C-column threat, rewritten using the no-C dev results.
- Footnote 1 becomes a link to the public repo (Stage 5).

### Writing and presentation (criterion 7)
- **Voice:** keep the plain, worked-example style, but name the standard
  term the first time each idea appears: Platt scaling, Bayes minimum-risk
  rule, operating point, post-hoc analysis, option-order bias, verbalizer.
  Rename informal headings ("Two flips" → e.g. "Accuracy vs cost; dev vs
  held-out").
- **Course traces:** keep author names and "Siam University" only. Remove
  student IDs, "Group 13, Frog Watch", "the proposal", "the deck". Add an
  acknowledgement that the work began as a course project.
- **Length:** about 8 IEEE pages of main text. Main text holds the
  headline, the held-out test and the trade-off figure. Appendices hold the
  bootstrap robustness table, the AUC diagnostic, the price-grid heatmap,
  the disparity check, feature parity, no-C, order bias and the verbatim
  prompts.
- **Prompt appendix:** the plain and rich IEEE-CIS prompts verbatim,
  including the context block, with one filled example row.
- **Abstract:** about 200 words, at most 4 numbers, leading with the rule
  finding, then the null result, then the held-out result.
- **Future Work:** compute and running cost per decision, a classification-
  head ablation, new real datasets, measured (not assumed) costs.

## Stage 5 — Public repo

Prepare (do not push) a public version: code, result JSONs, pre-registration
files with their commit history, a README with IEEE-CIS Kaggle download
steps (the data is **not** redistributed), an environment lockfile, and
GPU/CUDA/driver versions. Scan for secrets, absolute local paths and
personal data before the user publishes.

## Stage 6 — Verify and report

1. Rebuild `data.json` and the PDF and confirm both compile.
2. Check every number in the PDF traces to `data.json`, and every
   `held-out` / `read-out` / `dev` label is correct.
3. Re-read the paper against each decision above and list any not
   implemented, with the reason.
4. Final report: the audit result, held-out (or read-out) results table,
   every new dev and read-out result, the page count, and open issues.
