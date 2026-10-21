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
- [x] Train the adapter once (`python -m fraud.finetune`); manifest logged in PROGRESS
- [x] `python -m fraud.eval_ft`: dev gate, thresholds, then the 20 new windows (if the gate passes)
- [x] `fraud/docs/results-ft.md`: bars A/B, dev AUCs, paired counts, labelled claims, negatives
- [x] Console: `hybrid-ft` label and eval-chart series; the new eval page renders

**Accept:** results committed with both bars reported pass or fail (or the gate's failure),
whatever they show.

## Phase 6: the Decision 3.0 models (owner, 2026-10-21)
- [x] llama.cpp `llama-server` (winget) + d3-nano / d3-mini Q8_0 GGUFs, sha256 checked
- [x] `d3.py` (server, `D3Decider`; tested with a fake server) and `eval_d3.py` (pre-registration
      in its docstring, committed with the PROGRESS pre-statement before any D3 run)
- [x] Dev: both models on dev0–dev9, both option orders; debias rule; gate vs logreg dev AUC
- [x] Test windows test100–test119 for models that pass the gate (bar: pooled AUC > hybrid-1.5b):
      not run, no model passed the gate
- [x] `fraud/docs/results-d3.md`: dev AUCs, debias decision, gate, bar (if run), labelled claims

**Accept:** results committed with the gate (and bar, if reached) reported pass or fail,
whatever they show.

## Phase 7: the fine-tuned d3-nano (owner, 2026-10-21)
- [x] `d3torch.py` (d3-nano's backbone, non-causal softmax layers and readout in our own torch
      code; tested on a tiny random Qwen3.5) and `finetune_d3.py` (cap rule, hard stop; tested)
- [x] `eval_d3ft.py` with the pre-registration in its docstring, committed with the PROGRESS
      pre-statement before training
- [x] Reproduction check: the frozen model in torch matches the Phase 6 llama-server traces (R1–R3)
      → R3 FAIL (0.0207/0.0203 vs 0.02; Q8_0 diagnosis); the owner chose to proceed (deviation)
- [ ] Smoke run (s/step, memory), then one training run of at most 2 h (`--plan`)
- [ ] `python -m fraud.eval_d3ft`: dev, debias rule, gate vs logreg, then the 20 Phase 5 windows
      (if the gate passes) with bars A, B and C
- [ ] `fraud/docs/results-d3ft.md`: training, gate, bars, labelled claims, negatives

**Accept:** results committed with the gate and bars reported pass or fail (or the reproduction
check's or the cap rule's failure), whatever they show.
