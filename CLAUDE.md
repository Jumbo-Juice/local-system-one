# Working in this repo

A local System One-style decision engine (`system_one/`) and game demos that exercise it (`demo/`).
Start with the "Where things are" table in `README.md`; every command is in `RUN-GUIDE.md`.

## Rules

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
