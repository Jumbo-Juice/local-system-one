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

Optional: `--state "<text>"` and `--context "<text>"` add a state and context to the prompt.

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

The default game has ammo (6-bullet magazine, reloads, ammo boxes; tick limit 600) and the dash (up
to 3 cells in one tick, recharges in 8 ticks). `--game` picks an older one (the run's name starts
with its name): `ammo` is ammo without the dash (as evaluated on seeds 40-49), `classic` neither
ammo nor dash (as evaluated on seeds 0-39; `--classic` is short for it):

```bash
.venv/Scripts/python -m demo.shooter.capture --game ammo --config config/default.toml --seed 0
```

```bash
.venv/Scripts/python -m demo.shooter.capture --classic --config config/default.toml --seed 0
```

Useful flags: `--out <file>` (write the trace there instead of the pool), `--max-ticks N`,
`--plan-budget N` (-1 = unlimited), `--group-size N` (tournament group size, default 8), `--order-debias` / `--no-order-debias`, `--fire-head` /
`--no-fire-head` (one "shoot" option plus an aim head that picks the enemy; default from
`[shooter] fire_head`: on for the 1.5B, off for the 3B, always off with `--classic`), `--quiet`.

### Shooter evaluation

Full closed-loop evaluation (setups `1.5b`, `3b`, `1.5b-listed`, `3b-listed`, `1.5b-onehead`
(the 1.5B without the fire head), `random`, `bot-reference`, `bot-nododge`, `bot-noammo`; seeds 0-9
by default; the dash game). Traces go to the shooter pool as `<eval id>_eval-<game>_<setup>_seed<N>.jsonl`
(bots write none); results to `bench/results/shooter_eval/`:

```bash
.venv/Scripts/python -m bench.shooter_eval
```

An older game: `--game ammo` (no dash; the results of 2026-09-29) or `--game classic` (no ammo, no
dash; every result before 2026-09-29):

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

Rule calibration with bots only (no model; default: dev seeds 1000-1059, pick others with
`--seeds`):

```bash
.venv/Scripts/python -m bench.shooter_calibration
```

Ammo calibration (bots only; the grid that chose the ammo rules):

```bash
.venv/Scripts/python -m bench.shooter_calibration --ammo
```

## Dungeon demo

The dungeon was rebuilt on 2026-09-30 (wide corridors, two or more doorways per room, ghouls, the
dash); the first dungeon is gone. Capture one run (writes `runs/dungeon/<time>_<model>_seed<N>.jsonl`):

```bash
.venv/Scripts/python -m demo.dungeon.capture --config config/default.toml --seed 0
```

Same with the 3B:

```bash
.venv/Scripts/python -m demo.dungeon.capture --config config/lenovo-3b.toml --seed 0
```

Without a model:

```bash
.venv/Scripts/python -m demo.dungeon.capture --config config/mock.toml --seed 0
```

Flags: `--out <file>`, `--max-ticks N` (default 400), `--plan-budget N` (-1 = unlimited),
`--group-size N`, `--order-debias` / `--no-order-debias` (default: `[dungeon] order_debias` in the
config), `--quiet`.

Evaluation (setups `1.5b`, `3b`, `random`, `bot-reference`, `bot-careless`, `bot-random`; seeds 0-9
by default; traces go to the dungeon pool as `<eval id>_eval_<setup>_seed<N>.jsonl`, results to
`bench/results/dungeon_eval/`):

```bash
.venv/Scripts/python -m bench.dungeon_eval
```

Only some setups and seeds:

```bash
.venv/Scripts/python -m bench.dungeon_eval --setups 1.5b 3b --seeds 1000 1001 1002
```

Resume an interrupted evaluation:

```bash
.venv/Scripts/python -m bench.dungeon_eval --resume <eval id>
```

How the ghoul rules were chosen (bots only, dev seeds 1000-1059; writes
`bench/results/dungeon_calibration/`):

```bash
.venv/Scripts/python -m bench.dungeon_calibration
```

## Escape check (both demos, 20 runs)

Each model plays each game on seeds 0-4 (20 runs); done when at least 16 escape. Traces go to each
game's pool as `<check id>_check_<model>_seed<N>.jsonl`, results to `bench/results/escape_check/`:

```bash
.venv/Scripts/python -m bench.escape_check
```

Other seeds, only some models or games, or resume an interrupted check:

```bash
.venv/Scripts/python -m bench.escape_check --seeds 50 51 52 53 54
```

```bash
.venv/Scripts/python -m bench.escape_check --resume <check id>
```

Flags: `--models 1.5b 3b`, `--games dungeon shooter`. About 40 minutes on the Lenovo (Arc 140V); it
stops if the desktop app restarts, so a long check is safer from your own terminal.

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

Flags: `--agents N`, `--seed N`, `--gems N`, `--food N`, `--plan-budget N`, `--group-size N`,
`--no-goals`, `--json <file>` (headless summary), `--quiet`, `--close-after S` (close the window
after S seconds).

## Benchmarks

Each writes to `bench/results/<script>/`. All but `model_eval` (which takes `--models`) take
`--config`.

Decision latency, batched vs sequential vs text generation:

```bash
.venv/Scripts/python -m bench.bench --gen-baseline
```

Accuracy of models and answer templates on `bench/eval_set.py`:

```bash
.venv/Scripts/python -m bench.model_eval --models Qwen/Qwen2.5-1.5B-Instruct
```

Tournament sampling vs one full decision on large choice sets:

```bash
.venv/Scripts/python -m bench.tournament_compare
```

Grid demo, tiered goals vs flat control (`--ticks`, `--seeds`):

```bash
.venv/Scripts/python -m bench.demo_compare
```

Grid demo survival, old vs "aware" strategy prompts (`--ticks`, `--seeds`, `--plan-budget`):

```bash
.venv/Scripts/python -m bench.survival_compare --config config/lenovo-3b.toml
```

Does the strategy tier react to low energy:

```bash
.venv/Scripts/python -m bench.strategy_probe --config config/lenovo-3b.toml
```

Do move decisions approach the target (`--n` decisions, default 60):

```bash
.venv/Scripts/python -m bench.action_probe --config config/lenovo-3b.toml
```

## Fraud app

A separate application on PaySim (`fraud/`, its own rules in `fraud/CLAUDE.md`). Its runs go to
`fraud/runs/`, its results to `fraud/results/`; nothing here touches the game pools.

Build the data cache (needs the PaySim CSV in `fraud/data/raw/`; the balance columns are dropped):

```bash
.venv/Scripts/python -m fraud.data
```

One window under one setup, written to `fraud/runs/<time>_<label>.jsonl`:

```bash
.venv/Scripts/python -m fraud.capture --setup hybrid --window dev0
```

```bash
.venv/Scripts/python -m fraud.capture --setup rules --window dev3
```

Flags: `--setup hybrid|rules|logreg|random|approve-all`, `--window dev<seed>` (test windows need
`--test-ok`; they belong to the eval), `--config <toml>` (the hybrid's model, e.g.
`config/lenovo-3b.toml` or `config/mock.toml`), `--no-order-debias`, `--head-dtype model|float32`,
`--out <file>`, `--quiet`.

Prompt development on dev windows (prints AUC, actions and latency per prompt variant; defined in
`fraud/dev/promptdev.py`). Arguments: dev window seeds, then variants; `HEAD=float32` sets the LM
head dtype (default the model dtype):

```bash
HEAD=float32 .venv/Scripts/python -m fraud.dev.promptdev 1,2 v1,v2
```

The pre-registered eval (20 test windows × 6 setups → `fraud/results/eval/<eval id>.json|.md`):

```bash
.venv/Scripts/python -m fraud.eval
```

Flags: `--setups rules logreg ...` (a subset), `--resume <eval id>`, `--report <eval id>` (rebuild
the results files from the traces).

Phase 5, the fine-tuned brain (needs `peft`, see `requirements.txt`). Train a LoRA adapter on the
training steps (about 1 h on the Lenovo) into `fraud/models/<time>_lora/` (git-ignored):

```bash
.venv/Scripts/python -m fraud.finetune
```

Flags: `--dry-run` (build the training set, print counts and the row hash, no training),
`--steps N` (a smoke run of N batches, saved as `<time>_lora-smoke`).

Its pre-registered eval (dev gate and thresholds on dev0–dev9, then 20 new test windows × 6 setups
→ `fraud/results/eval/<eval id>.json|.md`; about 45 min):

```bash
.venv/Scripts/python -m fraud.eval_ft --adapter fraud/models/<time>_lora
```

Flags: `--windows` (print the new test windows, no model), `--setups hybrid-ft logreg ...`,
`--resume <eval id>` (with `--adapter`), `--report <eval id>`.

The analyst console (replay only; start page, a run at 1×, auto-demo, the eval page), on
http://127.0.0.1:8766:

```bash
.venv/Scripts/python -m fraud.viewer
```

Open one run directly:

```bash
.venv/Scripts/python -m fraud.viewer fraud/runs/20261003-052030_hybrid_qwen2.5-1.5b_dev0.pinned.jsonl
```

Flags: `--port N` (default 8766), `--host <addr>` (default 127.0.0.1), `--no-browser`, `--verbose`.
Pages: `#/` runs, `#/run/<file>`, `#/eval` (latest) or `#/eval/<eval id>`, `#/demo`. Pin a run by
renaming it to `*.pinned.jsonl` and listing it in `fraud/runs/README.md`.

## Where things land

| what | where |
|---|---|
| every captured run (captures, evals, escape checks) | `runs/shooter/`, `runs/dungeon/`: one `<time>_<label>.jsonl` per run, git-ignored |
| pinned runs (committed) | `runs/<game>/*.pinned.jsonl`, listed in `runs/README.md`; pin a run by renaming it |
| benchmark and eval results | `bench/results/<script>/` (committed) |
| lessons from past test and eval runs | `docs/lessons-learned.md` |
| the viewer of each game demo | `viewer/games/<game>.js` (a new demo adds one; see `viewer/README.md`) |
| fraud app runs and results | `fraud/runs/` (git-ignored but `*.pinned.jsonl`), `fraud/results/<script>/` (committed) |
