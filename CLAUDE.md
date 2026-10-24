# Working in this repo

A local System One-style decision engine (`system_one/`) and two applications that exercise it.
Start with the "Where things are" table in `README.md`; every command is in `RUN-GUIDE.md`.

## Two applications, kept apart

| app | where | its rules |
|---|---|---|
| **Game demos** (playful): grid, dungeon, shooter | `demo/`, `viewer/`, `runs/`, `bench/` | the Rules below |
| **Fraud app** (serious): approve / review / decline on PaySim | `fraud/` (its own runs, results, viewer; write-ups at its root) | `fraud/CLAUDE.md` |

They share only `system_one/`, `config/`, `.venv` and the pytest run. Neither imports from or writes
for the other. The game rules below do not apply to `fraud/`, and the fraud app never appears in the
game viewer.

## Rules (game demos)

- **Every captured run goes into its game's pool: `runs/<game>/`, one flat `.jsonl` per run.**
  Write runs only through `demo/runs.py` (`write_run` / `trace_path`), never to another folder,
  and never write a per-run HTML page. Names are `<time>_<label>.jsonl`; eval runs are
  `<eval id>_eval..._<setup>_seed<N>.jsonl` and resume with `--resume <eval id>`. Runs are
  git-ignored; only `*.pinned.jsonl` is committed (`runs/.gitignore`), and each pinned run is
  listed with the reason in `runs/README.md`.
- **Every game demo gets its own viewer.** A new demo (for example a platformer) means:
  - a package `demo/<game>/` whose trace header says `"scenario": "<game>"`;
  - a pool `runs/<game>/`;
  - a renderer `viewer/games/<game>.js`, loaded in `viewer/index.html`, following the contract in
    `viewer/README.md`.
  Do not make stand-alone viewer pages. The shell (`viewer/shell.js`) stays game-agnostic.
  `tests/test_viewer.py` checks that every game viewer is loaded and every pinned pool has one.
- **"About this run" layout** (every game): an outcome banner, facts as titled cards with one fact
  per row (short value; explanations go in the hint under it, never `a · b (c)` run-ons), and notes
  as titled cards of short bullets. Hint text uses `--muted`. No sideways scroll at 375 px.
- **Keep `RUN-GUIDE.md` current** with every command, flag or output-path change.
- Benchmark and eval results go to `bench/results/<script>/` and are committed.
- Label claims in docs as Documented (cite) / Observed (measured) / Inferred / Implementation
  choice. Pre-register seeds, setups and decision rules in the eval script before running.
  Report negative results.

## Checking changes

```bash
.venv/Scripts/python -m pytest -m "not model"
```

For viewer changes, run `python -m viewer` (or the `viewer` entry of `.claude/launch.json`) and
check the start page, a full run view and auto-demo. The desktop app's browser pane does not
animate while hidden; drive the page's `frame(ts)` by hand or render with headless Edge:
`msedge --headless=new --window-size=1600,1000 --virtual-time-budget=4000 --screenshot=<png> <url>`.

## Agent skills

### Issue tracker

Issues live in GitHub Issues for Jumbo-Juice/local-system-one, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The seven default labels, unchanged (bug, enhancement, needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one root `GLOSSARY.md` and `docs/adr/`, both created when first needed. See `docs/agents/domain.md`.

### External peers

host-only: `code-review` and `jury` start no external peer CLI. See `docs/agents/cross-model.json`.
