# Run guide

Every command, in one place. Run them from the repo root. Background and results are in
[README.md](README.md). This file is updated whenever a command, flag or output path changes.

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

## Shooter demo

Capture one run with the 1.5B (writes `demo/output/shooter/<model>_seed<N>/trace.jsonl` + `replay.html`):

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

Rebuild a replay page after editing `demo/shooter/viewer.html` (same trace, new page):

```bash
.venv/Scripts/python -m demo.shooter.capture --rebuild demo/output/shooter/<run>/trace.jsonl
```

Useful flags: `--out <dir>`, `--max-ticks N`, `--plan-budget N` (-1 = unlimited),
`--order-debias` / `--no-order-debias`, `--quiet`.

### Four-run comparison page

Rebuild `docs/shooter_quad.html` (two random 1.5B + two random 3B runs; pool = every
`demo/output/shooter/*/trace.jsonl` + `docs/shooter_replay_*.html`). Run it after adding runs or
editing `demo/shooter/quad.html`:

```bash
.venv/Scripts/python -m demo.shooter.quad
```

Build from chosen traces instead, to another file (needs at least two runs per model):

```bash
.venv/Scripts/python -m demo.shooter.quad --out demo/output/my_quad.html demo/output/shooter/eval_1.5b_seed12/trace.jsonl docs/shooter_replay_1.5b_seed0.html demo/output/shooter/eval_3b_seed5/trace.jsonl docs/shooter_replay_3b_seed0.html
```

Open it (no server needed). In the page: Space play/pause, ←/→ one tick, S shuffle.
`#runs=a,b,c,d&t=120` in the URL pins four runs and a tick.

```bash
start docs/shooter_quad.html
```

### Shooter evaluation

Full closed-loop evaluation (setups `1.5b`, `3b`, `1.5b-listed`, `3b-listed`, `random`,
`bot-reference`, `bot-nododge`; seeds 0-9 by default):

```bash
.venv/Scripts/python -m bench.shooter_eval
```

Only some setups and seeds:

```bash
.venv/Scripts/python -m bench.shooter_eval --setups 1.5b 3b --seeds 10 11 12
```

Resume an interrupted evaluation (reuses finished runs in that folder):

```bash
.venv/Scripts/python -m bench.shooter_eval --traces demo/output/shooter/eval_<time>
```

Rule calibration with bots only (no model):

```bash
.venv/Scripts/python -m bench.shooter_calibration
```

## Dungeon demo

Capture one run (writes `demo/output/dungeon/<model>_seed<N>/`):

```bash
.venv/Scripts/python -m demo.dungeon.capture --config config/lenovo-3b.toml --seed 0
```

Rebuild its replay page:

```bash
.venv/Scripts/python -m demo.dungeon.capture --rebuild demo/output/dungeon/<run>/trace.jsonl
```

Extra flags: `--label-style closer|steps`, `--enemy-aware`, plus the shooter's `--out`,
`--max-ticks`, `--plan-budget`, `--quiet`.

Evaluation (setups `3b-closer`, `3b-steps`, `1.5b-closer`, `random`, `3b-enemy-aware`; resume with `--traces <dir>`):

```bash
.venv/Scripts/python -m bench.dungeon_eval
```

## 2D grid demo (live window)

```bash
.venv/Scripts/python -m demo.sim
```

Without a model:

```bash
.venv/Scripts/python -m demo.sim --config config/mock.toml
```

Headless, 100 ticks:

```bash
.venv/Scripts/python -m demo.sim --headless --ticks 100
```

Most responsive setup (3B, one agent, ~5 ticks/s on the reference machine):

```bash
.venv/Scripts/python -m demo.sim --config config/lenovo-3b.toml --agents 1
```

Flags: `--agents N`, `--seed N`, `--gems N`, `--food N`, `--plan-budget N`, `--no-goals`,
`--json <file>` (headless summary).

## Benchmarks

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
| captured runs | `demo/output/<demo>/<run>/` (git-ignored) |
| kept runs overview | `demo/output/README.md` |
| committed replay pages | `docs/shooter_replay_*.html`, `docs/dungeon_replay_seed0.html`, `docs/shooter_quad.html` |
| benchmark results | `bench/results/` |
