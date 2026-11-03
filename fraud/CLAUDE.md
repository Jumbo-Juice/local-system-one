# CLAUDE.md: the fraud app

A serious showcase of the local System One engine (`system_one/`) on fraud detection: every
PaySim transaction gets **approve / review / decline**. Hand-written rules filter out the
transaction types that had no fraud in training, and the model judges the rest. The app is measured
against pre-registered baselines and replayed in its own analyst console. It shares **only** the
engine, the `.venv` and the pytest run with the game demos. Keep the two apart.

## Read these first, in order
1. `fraud/PRD.md`: what and why, the success bar, non-goals
2. `fraud/ARCHITECTURE.md`: data, signals, cost model, setups, trace format, viewer
3. `fraud/ROADMAP.md`: phases with acceptance lines; the checkboxes say where we are
4. `fraud/PROGRESS.md`: what happened and why (append-only)
5. `fraud/DATA.md`: Observed facts about PaySim (written in Phase 0); `fraud/DATA-IEEE.md`: the same for IEEE-CIS

## Hard rules
- **Separate from the games.** Nothing in `demo/`, `viewer/`, `runs/` or `bench/` imports from or
  writes for `fraud/`, and the reverse. No game styling, no canvas stage, no game wording.
- **The balance columns are never used.** `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest` and
  `newbalanceDest` are dropped when the CSV is read and never reach the cache, the signals, the
  prompt or a baseline. The PaySim authors say they "must not be used" (fraud is cancelled).
  A test enforces this.
- **No look-ahead.** A signal for row *i* uses only rows before *i* in file order (file order is
  treated as arrival order).
- **Train/test by time.** Prompt and rule development use training steps and dev windows only.
  The 20 test windows are pre-registered in `fraud/eval.py` and run once per setup. The fine-tune
  (Phase 5) trains on steps 1–300 minus the dev hours and is scored on 20 new windows that share
  no hour with them (`fraud/eval_ft.py`).
- **Adapters** go to `fraud/models/<time>_lora/` (git-ignored). The eval results record their sha256.
- **Runs** go to `fraud/runs/<time>_<label>.jsonl` only, through `fraud/runs.py`. They are
  git-ignored except `*.pinned.jsonl`, each listed with a reason in `fraud/runs/README.md`.
- **Eval results** go to `fraud/results/<script>/` and are committed.
- **Model scores are not probabilities.** Show them as "model scores (uncalibrated)". Never
  present a softmax as a fraud probability.
- Label every claim in docs as Documented (cite) / Observed / Inferred / Implementation choice.
  Report negative results.
- Raw data lives in `fraud/data/` (git-ignored). Never commit PaySim rows.

## Commands
```bash
.venv/Scripts/python -m pytest -m "not model"                 # all tests (games and fraud)
.venv/Scripts/python -m fraud.data                            # build fraud/data/paysim.npz from the CSV
.venv/Scripts/python -m fraud.capture --window dev0           # one run into fraud/runs/
.venv/Scripts/python -m fraud.eval                            # the pre-registered eval
.venv/Scripts/python -m fraud.finetune                        # Phase 5: LoRA adapter → fraud/models/
.venv/Scripts/python -m fraud.eval_ft --adapter <dir>         # Phase 5: its pre-registered eval
.venv/Scripts/python -m fraud.eval_d3                         # Phase 6: Decision 3.0 via llama-server
.venv/Scripts/python -m fraud.eval_d3ft --repro               # Phase 7: frozen d3-nano in torch vs Phase 6
.venv/Scripts/python -m fraud.finetune_d3 --plan <smoke dir>  # Phase 7: LoRA on d3-nano (2 h cap)
.venv/Scripts/python -m fraud.eval_d3ft --adapter <dir>       # Phase 7: its pre-registered eval
.venv/Scripts/python -m fraud.eval_lfm                        # Phase 8: LFM3-1.7B + d1-Omni-2B, frozen
.venv/Scripts/python -m fraud.eval_d1ft --repro               # Phase 9: frozen d1-Omni in torch vs Phase 8
.venv/Scripts/python -m fraud.finetune_d1 --plan <smoke dir>  # Phase 9: LoRA on d1-Omni-2B (1 h cap)
.venv/Scripts/python -m fraud.eval_d1ft --adapter <dir>       # Phase 9: its pre-registered eval
.venv/Scripts/python -m fraud.reviewer_check                 # exploratory: could a model replace review?
.venv/Scripts/python -m fraud.eval_chain                     # Phase 10: no review, tightened cash-out rule
.venv/Scripts/python -m fraud.ieee_data                       # Phase 11: build the IEEE-CIS cache
.venv/Scripts/python -m fraud.finetune_ieee --train           # Phase 11: LoRA on d1-Omni-2B, IEEE fit rows
.venv/Scripts/python -m fraud.eval_ieee --ieee-adapter <dir>  # Phase 11: frozen, PaySim and IEEE fine-tunes on IEEE-CIS
.venv/Scripts/python -m fraud.finetune_ieee --train --full    # Phase 12: the same on every fit fraud
.venv/Scripts/python -m fraud.eval_ieee_full --adapter <dir>  # Phase 12: its pre-registered eval on new test windows
.venv/Scripts/python -m fraud.reprice_paysim                  # Phase 13: older PaySim results at a 10 USD review, no model
.venv/Scripts/python -m fraud.eval_ieee_amount --adapter <dir> # Phase 14: decisions that weigh the amount, on new test windows
.venv/Scripts/python -m fraud.viewer                          # the analyst console
```
Every command and flag is in the repo's `RUN-GUIDE.md` → Fraud app. Keep it current.

## What NOT to do
- Don't add the fraud app to the game viewer, its filter or its pools, and don't reuse
  `viewer/shell.js`. The console is its own page.
- Don't add scikit-learn, pandas or other packages. The baselines are numpy. Ask first if a new
  dependency seems needed. (`peft` + `accelerate` were approved by the owner on 2026-10-05 for the
  Phase 5 fine-tune only; `peft` is imported only by `finetune.py` and `brain.load_adapter`, and
  in Phase 7, owner-approved, by `finetune_d3.py` and `d3torch.D3Torch.merge_adapter`.
  Phase 9 (owner: train d1-Omni) uses it the same way: through `finetune_d3.train` and `d1torch.D1Torch.merge_adapter`.
  `flash-linear-attention` was approved on 2026-10-21 but removed again: it does not run on this
  XPU, see PROGRESS 2026-10-21.)
- Don't tune prompts, rules or thresholds on test windows. If a test result prompts a change, the
  change gets new, untouched windows.
- Don't run inference in the viewer. It only replays recorded runs.
- Don't add a per-account (slow) tier. One decision per transaction.
- Don't use the balance columns, even "just to compare".

## When resuming
Check the `fraud/ROADMAP.md` checkboxes, `fraud/PROGRESS.md` and `git log --oneline -15`. Run the
tests for a green baseline, then continue with the first unticked phase.
