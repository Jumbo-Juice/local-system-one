# ROADMAP: the fraud app

The owner asked for autonomous work (2026-10-03): each phase is self-verified, committed and tagged
(`fraud-phase-N`), and the next one starts without a stop. The owner checks the end result: the
console replaying a run at 1×.

## Phase 0: specs and data look
- [x] Spec files (`fraud/CLAUDE.md`, PRD, ARCHITECTURE, ROADMAP, PROGRESS); root CLAUDE.md "two applications"
- [x] Download PaySim; `fraud/data.py` builds the cache without balance columns (tested)
- [x] `fraud/DATA.md`: Observed counts: rows, fraud by type, steps, origin/destination repeats,
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
- [x] `fraud/RESULTS.md`: bars pass/fail, baselines, ECE/outside_mass, labelled claims

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
- [x] `fraud/RESULTS-FT.md`: bars A/B, dev AUCs, paired counts, labelled claims, negatives
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
- [x] `fraud/RESULTS-D3.md`: dev AUCs, debias decision, gate, bar (if run), labelled claims

**Accept:** results committed with the gate (and bar, if reached) reported pass or fail,
whatever they show.

## Phase 7: the fine-tuned d3-nano (owner, 2026-10-21)
- [x] `d3torch.py` (d3-nano's backbone, non-causal softmax layers and readout in our own torch
      code; tested on a tiny random Qwen3.5) and `finetune_d3.py` (cap rule, hard stop; tested)
- [x] `eval_d3ft.py` with the pre-registration in its docstring, committed with the PROGRESS
      pre-statement before training
- [x] Reproduction check: the frozen model in torch matches the Phase 6 llama-server traces (R1–R3)
      → R3 FAIL (0.0207/0.0203 vs 0.02; Q8_0 diagnosis); the owner chose to proceed (deviation)
- [x] Smoke run (s/step, memory), then one training run of at most 2 h (`--plan`)
- [x] `python -m fraud.eval_d3ft`: dev, debias rule, gate vs logreg, then the 20 Phase 5 windows
      (if the gate passes) with bars A, B and C
- [x] `fraud/RESULTS-D3FT.md`: training, gate, bars, labelled claims, negatives
- [x] Console (owner, after the eval): `hybrid-d3ft` label and eval-chart series, the eval page reads
      `fraud/results/eval_d3ft/` (training card, dev gate by option order, bar C), D3 traces replay

**Accept:** results committed with the gate and bars reported pass or fail (or the reproduction
check's or the cap rule's failure), whatever they show.

## Phase 8: the LFM3 family, frozen (owner, 2026-10-30)
- [x] llama.cpp b11960+ (owner: winget upgraded to b11977) + d1-Omni-2B and LFM3-1.7B Q8_0 GGUFs,
      sha256 checked
- [x] `eval_lfm.py` (pre-registration in its docstring, committed with the PROGRESS pre-statement
      before any LFM run): d1-Omni-2B on `/v1/systemone` as in Phase 6, LFM3-1.7B through the engine
      (llama.cpp backend) as the 1.5B
- [x] Dev: both models on dev0–dev9, both option orders; debias rule; vs the frozen 1.5B; gate vs logreg
      → d1-Omni-2B 0.9847 (frozen 1.5B 0.9028), gate FAIL; LFM3-1.7B dropped by the owner (too slow)
- [x] Test windows test100–test119 for models that pass the gate (bar: pooled AUC > hybrid-1.5b):
      not run, no model passed the gate
- [x] `fraud/RESULTS-LFM.md`: dev AUCs, vs the frozen 1.5B, gate, bar (if run), labelled claims

**Accept:** results committed with the comparison, the gate (and bar, if reached) reported pass or
fail, whatever they show.

## Phase 9: d1-Omni-2B fine-tuned for one hour (owner, 2026-11-02)
- [x] `d1torch.py` (the LFM3 text layers, "set" masks and code readout in our own torch code; tested on
      a tiny random model; the repo's remote code is not run) and `finetune_d1.py` (the hour's cap)
- [x] `eval_d1ft.py` with the pre-registration in its docstring, committed with the PROGRESS
      pre-statement before training
- [x] Reproduction check: the frozen model in torch matches the Phase 8 llama-server traces (R1–R3): PASS
- [x] Smoke run (s/step, memory), then one training run inside one hour (`--plan`): a full epoch in 56 min
- [x] `python -m fraud.eval_d1ft`: dev, debias rule, gate vs logreg, then the 20 Phase 5 windows
      (if the gate passes) with bars A, B and C
- [x] `fraud/RESULTS-D1FT.md`: training, gate, bars, labelled claims, negatives

**Accept:** results committed with the gate and bars reported pass or fail (or the reproduction
check's failure), whatever they show.

## Phase 10: no review, a tightened cash-out rule, by F1 (owner, 2026-11-02)
- [x] `eval_chain.py` with the pre-registration in its docstring, committed with the PROGRESS pre-statement before it ran
      on any test-split hour
- [x] `python -m fraud.eval_chain`, once, on the untouched test-split hours: bar 1 PASS, bar 2 FAIL, row F1 reported
- [x] `fraud/RESULTS-CHAIN.md`: bars, row and chain F1, labelled claims, limits

**Accept:** results committed with both bars reported pass or fail, whatever they show.


## Phase 11: first real data, IEEE-CIS (owner, 2026-11-02)
- [x] `ieee_data.py` (train files → cache, time split) and `DATA-IEEE.md`
- [x] `ieee_brain.py` (one new prompt, prompt v4's question and labels) and `eval_ieee.py` with the pre-registration in its
      docstring, committed with the PROGRESS pre-statement before any model scored a row
- [x] Owner's amendment, committed before any eval output was read: retraining treated as guaranteed, review 10 by default, train on IEEE-CIS
- [x] `python -m fraud.eval_ieee`: the frozen and PaySim fine-tuned models as they are, dev, thresholds, 20 test windows; `--reprice` at review 10
- [x] `python -m fraud.finetune_ieee --train`: d1-Omni-2B on IEEE fit rows, the PaySim adapter's size and configuration
- [x] The IEEE fine-tune on the same windows; `fraud/RESULTS-IEEE.md`: IEEE vs PaySim fine-tune, frozen vs PaySim, baselines, negatives

**Accept:** results committed with the answers and bars reported, whatever they show.

## Phase 12: IEEE-CIS fine-tune on every fit fraud (owner, 2026-11-03)
- [x] `eval_ieee_full.py` with the pre-registration in its docstring and 20 new test windows, committed before training
- [x] `python -m fraud.finetune_ieee --train --full`: 11,988 fraud + 35,964 legit fit rows, 5,994 batches
- [x] `python -m fraud.eval_ieee_full --adapter <dir>`: new adapter vs Phase 11's and logistic regression on the new windows; results in `fraud/RESULTS-IEEE-FULL.md`

**Accept:** results committed with the answers and bars reported, whatever they show.

## Phase 13: PaySim's older results at a 10 USD review (owner, 2026-11-03)
- [x] `reprice_paysim.py` with the pre-registration in its docstring, and its gate passing at 50 USD, committed before anything is priced at 10
- [x] `python -m fraud.reprice_paysim`: same decisions and re-picked thresholds, from the recorded scores; results in `fraud/RESULTS-REPRICE.md`

**Accept:** results committed beside the originals, which stay as they are, with both views and the bars reported, whatever they show.

## Phase 14: decisions that weigh the amount, IEEE-CIS (owner, 2026-11-03)
- [x] `eval_ieee_amount.py` with the pre-registration in its docstring, 20 new test windows and 20 more dev windows, committed before any of them was scored
- [ ] `python -m fraud.eval_ieee_amount --adapter <dir>`: amount-aware rule, more dev windows and a blend, from logreg's and Phase 12's frozen scores; results in `fraud/RESULTS-IEEE-AMOUNT.md`

**Accept:** results committed with the answers reported, whatever they show.
