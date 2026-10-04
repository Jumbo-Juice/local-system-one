# ROADMAP: the fraud app

The owner asked for autonomous work (2026-10-03): each phase is self-verified, committed and tagged
(`fraud-phase-N`), and the next one starts without a stop. The owner checks the end result: the
console replaying a run at 1×.

## Phase 0: specs and data look
- [x] Spec files (`fraud/CLAUDE.md`, PRD, ARCHITECTURE, ROADMAP, PROGRESS); root CLAUDE.md "two applications"
- [x] Download PaySim; `fraud/data.py` builds the cache without balance columns (tested)
- [x] `fraud/docs/data.md`: Observed counts: rows, fraud by type, steps, origin/destination repeats,
      the fraud pattern, `isFlaggedFraud`

**Accept:** `python -m fraud.data` builds the cache; a test proves no balance column is in it;
`data.md` answers which types are fraud-free and whether origin accounts repeat.

## Phase 1: signals, windows, costs, rules, baselines
- [x] `signals.py` (no look-ahead, tested on hand-made rows)
- [x] `windows.py`: train/test step cut, seeded enriched windows, natural-rate weights (tested)
- [x] `costs.py`: cost model + review budget (tested)
- [x] `rules.py`: type filter + rules-only table, written from training steps only
- [x] `baselines.py`: numpy logistic regression trained on training steps; random; approve-all
- [x] Constants and the rule table logged in PROGRESS before Phase 3

**Accept:** tests green; on dev windows the non-model setups score without errors and the
numbers are logged.

## Phase 2: model brain and capture
- [x] `brain.py`: prompt (≤ ~250 tokens), approve/review/decline, order debias, timing
- [x] `runs.py`, `capture.py` → `fraud/runs/` (mock backend for tests)
- [x] Prompt development on dev windows only; latency checked against the 250 ms bar

**Accept:** a mock capture and a 1.5B capture of a dev window each write a finished trace;
p50/p90 latency logged.

## Phase 3: the pre-registered eval
- [x] `eval.py`: test window seeds, setups, bars and the decision rule written and committed before running
- [x] Run all setups on the 20 test windows → `fraud/results/eval/`
- [x] `fraud/docs/results.md`: bars pass/fail, baselines, ECE/outside_mass, labelled claims

**Accept:** results committed; both bars reported pass or fail with numbers; negative results stated.

## Phase 4: the analyst console
- [x] `fraud/viewer/`: start page (runs, auto-demo, eval link), replay console at 1× (feed,
      decision card, running cost and confusion, latency timeline, About this run), eval page
- [x] A pinned example run committed; RUN-GUIDE and README updated
- [x] Checked: start page, full replay, auto-demo, eval page, 375 px with no sideways scroll, no
      console errors

**Accept:** `python -m fraud.viewer` opens, and a recorded run replays at 1× with the decisions
appearing at their recorded latency.

## Phase 5: the fine-tuned brain (owner, 2026-10-05)
- [x] Spike: LoRA trains on the Arc 140V with `peft` in the repo `.venv` (PROGRESS 2026-10-05)
- [x] `finetune.py` (training rows, examples, encoding; tested) and `eval_ft.py` (new windows,
      thresholds, AUC; tested); `brain.ThresholdDecider`, `brain.load_adapter`
- [x] Pre-registration in the `eval_ft.py` docstring, committed before training
- [ ] Train the adapter once (`python -m fraud.finetune`); manifest logged in PROGRESS
- [ ] `python -m fraud.eval_ft`: dev gate, thresholds, then the 20 new windows (if the gate passes)
- [ ] `fraud/docs/results-ft.md`: bars A/B, dev AUCs, paired counts, labelled claims, negatives
- [ ] Console: `hybrid-ft` label and eval-chart series; the new eval page renders

**Accept:** results committed with both bars reported pass or fail (or the gate's failure),
whatever they show.
