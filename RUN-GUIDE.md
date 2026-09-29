# Run guide

Every command, in one place. Run them from the repo root. Background and results are in
[README.md](README.md) and the demo write-ups in `docs/`. This file is updated whenever a command, flag or output path changes.

Configs: `config/default.toml` (Qwen2.5-1.5B), `config/lenovo-3b.toml` (Qwen2.5-3B),
`config/mock.toml` (no model, random choices), `config/nuc.example.toml` (copy to `config/nuc.toml`
for bigger hardware). Pass one with `--config`, or set `SYSTEM_ONE_CONFIG` for the whole shell.

## Setup (once)

```bash
py -3.13 -m venv .venv
```

Install the PyTorch build for your hardware first (index URLs in `requirements.txt`), then:

```bash
.venv/Scripts/python -m pip install -r requirements.txt
```

Check the model and backend work (the first run downloads the model):

```bash
.venv/Scripts/python -m system_one check
```

One decision by hand:

```bash
.venv/Scripts/python -m system_one decide --question "Which colour is the sky on a clear day?" --options red blue yellow
```

## Tests

All tests (the `model` tests skip if Qwen2.5-0.5B-Instruct is not cached):

```bash
.venv/Scripts/python -m pytest
```

Without the model tests:

```bash
.venv/Scripts/python -m pytest -m "not model"
```

## Watch runs (Master Viewer)

Start page: load a trace, a random run, or auto-demo (two panes, newest to oldest, looping). Opens a
browser tab; Ctrl+C stops it:

```bash
.venv/Scripts/python -m viewer
```

Open one run:

```bash
.venv/Scripts/python -m viewer runs/shooter/<run>.jsonl
```

Reachable from other devices on the network (default: this machine only):

```bash
.venv/Scripts/python -m viewer --host 0.0.0.0
```

One self-contained page with one run, to share (opens from disk, no server; `-o` picks the file):

```bash
.venv/Scripts/python -m viewer --bundle runs/shooter/<run>.jsonl
```

Flags: `--port N` (default 8765; the next free one if taken), `--no-browser`, `--verbose`. In the
page: Space play/pause, ←/→ one tick, Home restart; `#run=<path>&tick=N` in the URL opens a tick.

## Shooter demo

Capture one run with the 1.5B (writes `runs/shooter/<time>_qwen2.5-1.5b_seed<N>.jsonl`):

```bash
.venv/Scripts/python -m demo.shooter.capture --config config/default.toml --seed 0
```

Same with the 3B:

```bash
.venv/Scripts/python -m demo.shooter.capture --config config/lenovo-3b.toml --seed 0
```

Without a model:

```bash
.venv/Scripts/python -m demo.shooter.capture --config config/mock.toml --seed 0
```

Ammo is on by default (6-bullet magazine, reloads, ammo boxes; tick limit 600). The game without
ammo, as evaluated on seeds 0-39 (the run's name starts `classic_`):

```bash
.venv/Scripts/python -m demo.shooter.capture --classic --config config/default.toml --seed 0
```

Useful flags: `--out <file>` (write the trace there instead of the pool), `--max-ticks N`,
`--plan-budget N` (-1 = unlimited), `--order-debias` / `--no-order-debias`, `--fire-head` /
`--no-fire-head` (one "shoot" option plus an aim head that picks the enemy; default from
`[shooter] fire_head`: on for the 1.5B, off for the 3B, always off with `--classic`), `--quiet`.

### Shooter evaluation

Full closed-loop evaluation (setups `1.5b`, `3b`, `1.5b-listed`, `3b-listed`, `1.5b-onehead`
(the 1.5B without the fire head), `random`, `bot-reference`, `bot-nododge`, `bot-noammo`; seeds 0-9
by default; with ammo). Traces go to the shooter pool as `<eval id>_eval-<game>_<setup>_seed<N>.jsonl`
(bots write none); results to `bench/results/shooter_eval/`:

```bash
.venv/Scripts/python -m bench.shooter_eval
```

The same without ammo (the game every result before 2026-09-29 was measured on):

```bash
.venv/Scripts/python -m bench.shooter_eval --game classic
```

Only some setups and seeds:

```bash
.venv/Scripts/python -m bench.shooter_eval --setups 1.5b 3b --seeds 10 11 12
```

Resume an interrupted evaluation (reuses its finished runs; the eval id is the time at the start of
its trace names; pass the same `--game`):

```bash
.venv/Scripts/python -m bench.shooter_eval --resume <eval id>
```

A long evaluation run from a Claude session stops if the desktop app restarts; run it from your own
terminal to be safe.

Rule calibration with bots only (no model):

```bash
.venv/Scripts/python -m bench.shooter_calibration
```

Ammo calibration (bots only; the grid that chose the ammo rules):

```bash
.venv/Scripts/python -m bench.shooter_calibration --ammo
```

## Dungeon demo

Capture one run (writes `runs/dungeon/<time>_<model>_seed<N>.jsonl`):

```bash
.venv/Scripts/python -m demo.dungeon.capture --config config/lenovo-3b.toml --seed 0
```

Extra flags: `--label-style closer|steps`, `--enemy-aware` (both show up in the run's name), plus
the shooter's `--out`, `--max-ticks`, `--plan-budget`, `--quiet`.

Evaluation (setups `3b-closer`, `3b-steps`, `1.5b-closer`, `random`, `3b-enemy-aware`; traces go to
the dungeon pool as `<eval id>_eval_<setup>_seed<N>.jsonl`, results to `bench/results/dungeon_eval/`):

```bash
.venv/Scripts/python -m bench.dungeon_eval
```

Resume an interrupted evaluation:

```bash
.venv/Scripts/python -m bench.dungeon_eval --resume <eval id>
```

## 2D grid demo (live window)

```bash
.venv/Scripts/python -m demo.grid
```

Without a model:

```bash
.venv/Scripts/python -m demo.grid --config config/mock.toml
```

Headless, 100 ticks:

```bash
.venv/Scripts/python -m demo.grid --headless --ticks 100
```

Most responsive setup (3B, one agent, ~5 ticks/s on the reference machine):

```bash
.venv/Scripts/python -m demo.grid --config config/lenovo-3b.toml --agents 1
```

Flags: `--agents N`, `--seed N`, `--gems N`, `--food N`, `--plan-budget N`, `--no-goals`,
`--json <file>` (headless summary).

## Benchmarks

Each writes to `bench/results/<script>/`:

```bash
.venv/Scripts/python -m bench.bench --gen-baseline
```

```bash
.venv/Scripts/python -m bench.model_eval --models Qwen/Qwen2.5-1.5B-Instruct
```

```bash
.venv/Scripts/python -m bench.tournament_compare
```

```bash
.venv/Scripts/python -m bench.demo_compare
```

## Where things land

| what | where |
|---|---|
| every captured run (captures and evals) | `runs/shooter/`, `runs/dungeon/`: one `<time>_<label>.jsonl` per run, git-ignored |
| pinned runs (committed) | `runs/<game>/*.pinned.jsonl`, listed in `runs/README.md`; pin a run by renaming it |
| benchmark and eval results | `bench/results/<script>/` (committed) |
| lessons from past test and eval runs | `docs/lessons-learned.md` |
| the viewer of each game demo | `viewer/games/<game>.js` (a new demo adds one; see `viewer/README.md`) |
