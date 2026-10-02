# PRD: System One fraud showcase

## Problem
The game demos show that the local System One engine decides fast and that its choices can be
inspected. They do not show it on a serious task. This app asks a concrete question: on a public
fraud dataset, with leak-free inputs, does a small local LLM add anything to hand-written rules?
How fast does it decide, and can you see what it decided and why?

## Audience
A showcase, run by the repo owner on the Lenovo (Arc 140V iGPU). Not production: no API, no
hardening, no real customer data. Non-commercial licences are acceptable (Qwen2.5-3B Research
licence, PaySim CC BY-SA 4.0).

## What v1 does
1. Loads PaySim (Kaggle `ealaxi/paysim1`, CC BY-SA 4.0) into a local cache without the four
   balance columns.
2. Builds time-ordered **windows** of ~500 transactions from a held-out test period, with fraud
   enriched to ~5%, reproducible from a seed.
3. Decides **approve / review / decline** for every transaction in a window under one of these
   setups:
   - **hybrid**: a rule filter auto-approves the types with no fraud in training; Qwen2.5-1.5B
     decides the rest (order-debiased);
   - ~~hybrid-3b~~: dropped on 2026-10-03 (owner): the 3B does not fit in the iGPU memory free on the
     Lenovo (Observed, `fraud/PROGRESS.md`);
   - **rules-only**: a pre-registered rule table decides every transaction;
   - **filter + logistic regression** (numpy, same signals as the prompt);
   - **filter + random**; **approve-all** (the cost of catching nothing).
4. Scores each window with a money cost model under a review budget, and records per-decision
   latency.
5. Captures runs as traces and replays them in an **analyst console** at the recorded speed (1×):
   a transaction feed, a card per decision (signals, option scores, latency, truth), running cost
   and confusion matrix, a latency timeline, "About this run" and an auto-demo.
6. Shows an **eval page**: 20 test windows × setups against the success bar.

## Success (pre-registered in `fraud/eval.py`)
- **Bar 1:** hybrid (1.5B) total cost < rules-only total cost in **≥ 16 of 20** test windows.
- **Bar 2:** hybrid (1.5B) per-decision latency **p90 ≤ 250 ms** on the Arc iGPU, order debias on,
  measured over the model decisions of the 20 test windows.
- Reported whatever they show: logistic regression, random, approve-all, ECE and
  `outside_mass` of the model scores, and costs reweighted to the natural fraud rate.
- **Done for the owner:** the console runs and replays a recorded run at 1× recorded speed.

## Non-goals
- No balance columns, not even as a comparison. (Documented leak; the dataset authors forbid it.)
- No live inference in the viewer. (Replay only; it keeps the console shareable and model-free.)
- No per-account or slow tier. (In PaySim origin accounts rarely repeat; to be confirmed in Phase 0.)
- No calibrated fraud probability (Platt, isotonic, temperature). Scores are measured and labelled
  uncalibrated.
- No gradient boosting and no new packages. (Logistic regression in numpy is the classic baseline;
  the docs say it is the weaker kind.)
- No fraud-type classification and no analyst workflow beyond the review decision.
- No integration with the game viewer, its pools or its styling.

## Open questions
None at spec time. Values that depend on the data (the train/test step cut, window size and fraud
share, cost constants, the review budget, the rule table) are set in Phase 1 from training data
only, written into `fraud/eval.py` and logged in `PROGRESS.md` before any test window runs.
