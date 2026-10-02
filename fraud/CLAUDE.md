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
5. `fraud/docs/data.md`: Observed facts about PaySim (written in Phase 0)

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
  The 20 test windows are pre-registered in `fraud/eval.py` and run once per setup.
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
.venv/Scripts/python -m fraud.viewer                          # the analyst console
```
Every command and flag is in the repo's `RUN-GUIDE.md` → Fraud app. Keep it current.

## What NOT to do
- Don't add the fraud app to the game viewer, its filter or its pools, and don't reuse
  `viewer/shell.js`. The console is its own page.
- Don't add scikit-learn, pandas or other packages. The baselines are numpy. Ask first if a new
  dependency seems needed.
- Don't tune prompts, rules or thresholds on test windows. If a test result prompts a change, the
  change gets new, untouched windows.
- Don't run inference in the viewer. It only replays recorded runs.
- Don't add a per-account (slow) tier. One decision per transaction.
- Don't use the balance columns, even "just to compare".

## When resuming
Check the `fraud/ROADMAP.md` checkboxes, `fraud/PROGRESS.md` and `git log --oneline -15`. Run the
tests for a green baseline, then continue with the first unticked phase.
