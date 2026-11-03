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

Build the data cache (needs `Synthetic_Financial_datasets_log.csv` from Kaggle
`sriharshaeedala/financial-fraud-detection-dataset` in `fraud/data/raw/`; the balance columns are dropped):

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

Phase 6, the Decision 3.0 models (d3-nano, d3-mini) through llama.cpp's `/v1/systemone`. Once:
install llama.cpp (build ≥ 11572; `fraud/d3.py` also finds the winget copy when it is not on PATH)
and download the two Q8_0 GGUFs into `fraud/models/d3/` (git-ignored; sha256 pinned in `fraud/d3.py`
and checked at start). If `LLAMA_API_KEY` is set, llama-server requires it and `fraud/d3.py` sends it.

```bash
winget install llama.cpp
```

```bash
.venv/Scripts/python -c "from huggingface_hub import hf_hub_download as d; [d(f'ggml-org/vllm-d3-{m}-GGUF', f'vllm-d3-{m}-Q8_0.gguf', local_dir='fraud/models/d3') for m in ('nano', 'mini')]"
```

A few dev rows through the endpoint (both orders, each order alone, timing; writes nothing):

```bash
.venv/Scripts/python -m fraud.eval_d3 --smoke d3-nano
```

The pre-registered eval (dev runs, debias rule, gate vs logreg, then test100–test119 for models
that pass → `fraud/results/eval_d3/<eval id>.json|.md`):

```bash
.venv/Scripts/python -m fraud.eval_d3
```

Flags: `--models d3-nano d3-mini`, `--resume <eval id>`, `--report <eval id>`, `--smoke <model>`.

Phase 7, the fine-tuned d3-nano (torch on the XPU, no llama.cpp; d3-mini was dropped after Phase 6
and its GGUF deleted). Once: download the checkpoint (4.4 GB, BF16) at the pinned revision into
`fraud/models/d3-nano-hf/` (git-ignored; sha256 pinned in `fraud/d3torch.py` and checked on load):

```bash
.venv/Scripts/python -c "from huggingface_hub import snapshot_download as d; d('vllm-sr/d3-nano', revision='462efc366229ea268cf4eed3738b3fc4415e4817', local_dir='fraud/models/d3-nano-hf', allow_patterns=['*.json', '*.jinja', '*.txt', '*.safetensors'])"
```

The reproduction check (the frozen model through `fraud/d3torch.py` vs the Phase 6 llama-server
traces on dev0–dev9 → `fraud/results/eval_d3ft/<id>_repro.json|.md`):

```bash
.venv/Scripts/python -u -m fraud.eval_d3ft --repro
```

Train: a smoke run first (s/step and peak memory, saved as `<time>_lora-d3-smoke`), then the
training run, whose batch count comes from the smoke run's s/step and which stops itself at 2 h
(`fraud/models/<time>_lora-d3/`):

```bash
.venv/Scripts/python -u -m fraud.finetune_d3 --steps 20
```

```bash
.venv/Scripts/python -u -m fraud.finetune_d3 --plan fraud/models/<time>_lora-d3-smoke
```

Flags: `--dry-run` (the training set only: counts and the row hash). `--plan` exits with code 2 and
trains nothing when the rule gives fewer than 755 batches.

Its pre-registered eval (dev, debias rule, gate vs logreg, then the 20 Phase 5 windows →
`fraud/results/eval_d3ft/<eval id>.json|.md`):

```bash
.venv/Scripts/python -u -m fraud.eval_d3ft --adapter fraud/models/<time>_lora-d3
```

Flags: `--resume <eval id>` (with `--adapter` or `--repro`), `--report <eval id>` (with `--adapter`
to refresh the training facts).

Phase 8, the LFM3 family frozen (d1-Omni-2B on `/v1/systemone`, LFM3-1.7B through the engine).
Both need llama.cpp b11960 or newer (the owner upgraded the winget install to b11977 on 2026-10-30;
Phase 6 ran on b11748). Once: upgrade llama.cpp, and download the two Q8_0 GGUFs into `fraud/models/lfm/`
(git-ignored; sha256 pinned in `fraud/d3.py` / `fraud/eval_lfm.py`):

```bash
winget upgrade --id ggml.llamacpp
```

```bash
.venv/Scripts/python -c "from huggingface_hub import hf_hub_download as d; [d(f'LiquidAI/{m}-GGUF', f'{m}-Q8_0.gguf', local_dir='fraud/models/lfm') for m in ('d1-Omni-2B', 'LFM3-1.7B')]"
```

A few dev rows (both orders, the forward order alone, timing; writes nothing):

```bash
.venv/Scripts/python -m fraud.eval_lfm --smoke lfm3-1.7b
```

The pre-registered eval (dev runs, debias rule, vs the frozen 1.5B, gate vs logreg, then
test100–test119 for models that pass → `fraud/results/eval_lfm/<eval id>.json|.md`):

```bash
.venv/Scripts/python -u -m fraud.eval_lfm
```

Flags: `--models d1-omni-2b lfm3-1.7b`, `--resume <eval id>`, `--report <eval id>`, `--smoke <model>`.

Phase 9, d1-Omni-2B fine-tuned for one hour (torch on the XPU, no llama.cpp, no remote code). Once: the
checkpoint's weights and tokenizer at the pinned revision into `fraud/models/d1-omni-hf/` (git-ignored; 4.3 GB;
sha256 pinned in `fraud/d1torch.py` and checked on load):

```bash
.venv/Scripts/python -c "from huggingface_hub import snapshot_download as d; d('LiquidAI/d1-Omni-2B', revision='431f1d667de5afcc9b0c58b2d9c170a7b38b358b', local_dir='fraud/models/d1-omni-hf', allow_patterns=['model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'config.json', 'calibration.json', 'LICENSE'])"
```

The reproduction check (frozen model in torch vs the Phase 8 llama-server traces on dev0–dev9 →
`fraud/results/eval_d1ft/<id>_repro.json|.md`):

```bash
.venv/Scripts/python -u -m fraud.eval_d1ft --repro
```

Train: a smoke run first (s/step and peak memory, saved as `<time>_lora-d1-smoke`), then the training run, whose
batch count comes from the smoke run and which stops itself inside one hour (`fraud/models/<time>_lora-d1/`):

```bash
.venv/Scripts/python -u -m fraud.finetune_d1 --steps 20
```

```bash
.venv/Scripts/python -u -m fraud.finetune_d1 --plan fraud/models/<time>_lora-d1-smoke
```

Its pre-registered eval (dev, debias rule, gate vs logreg, then the 20 Phase 5 windows →
`fraud/results/eval_d1ft/<eval id>.json|.md`):

```bash
.venv/Scripts/python -u -m fraud.eval_d1ft --adapter fraud/models/<time>_lora-d1
```

Flags: `--resume <eval id>` (with `--adapter` or `--repro`), `--report <eval id>` (with `--adapter` to refresh the
training facts).

The review-tier check (exploratory; recorded dev scores, no inference, no test window →
`fraud/results/reviewer_check/<id>.json|.md`; `--probe` reads the queue on the training steps → `<id>_probe.json|.md`):

```bash
.venv/Scripts/python -m fraud.reviewer_check
```

Flags: `--d1ft <eval id>` (adds that eval's fine-tuned d1-Omni-2B), `--probe`.

Phase 10, no human review with a tightened cash-out rule, scored by F1 (no model, CPU, seconds). `--hours` counts the
untouched test-split hours and reads no outcome, `--train` reproduces the in-sample numbers on steps 1–300 and writes nothing:

```bash
.venv/Scripts/python -m fraud.eval_chain --train
```

The pre-registered test, once (→ `fraud/results/eval_chain/<id>.json|.md`):

```bash
.venv/Scripts/python -m fraud.eval_chain
```

After it, `--diagnose <eval id>` splits the tight setup's wrong declines by rule line (→ `<eval id>_diagnosis.json`).

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
Eval pages read `fraud/results/eval/`, `fraud/results/eval_d3ft/` and `fraud/results/eval_d1ft/` (the fine-tune pages merge in the
Phase 5 setups of the same windows); Phase 6's `fraud/results/eval_d3/` ran no test windows and has no page.
Pages: `#/` runs, `#/run/<file>`, `#/eval` (latest) or `#/eval/<eval id>`, `#/demo`. Pin a run by
renaming it to `*.pinned.jsonl` and listing it in `fraud/runs/README.md`.

### IEEE-CIS (real data, Phase 11)

Put the unzipped Kaggle CSVs in `fraud/data/ieee-cis/` (git-ignored), then build the cache (`fraud/data/ieee-cis/ieee.npz`):

```bash
.venv/Scripts/python -m fraud.ieee_data
```

The IEEE fine-tune (`fraud/models/<time>_lora-d1-ieee/`): `--dry-run` prints the rows, `--steps N` is a smoke, `--train` is the run (65 min here, safety stop at 3 h):

```bash
.venv/Scripts/python -m fraud.finetune_ieee --dry-run
.venv/Scripts/python -m fraud.finetune_ieee --steps 20
.venv/Scripts/python -m fraud.finetune_ieee --train
```

The eval: `--draw` prints the seeded draw and windows; `--smoke N` measures latency on N fit rows; `--ieee-adapter <dir>` runs it (about 3.5 h);
add `--resume <eval id>` to finish one; `--reprice <eval id>` re-decides its traces at the current costs from the recorded scores, with no model;
`--report <eval id>` rebuilds `fraud/results/eval_ieee/<eval id>.json|.md`. Traces: `fraud/runs/<eval id>_{dev,eval}_ieee-*.jsonl`. Write-up: `fraud/RESULTS-IEEE.md`.
Keep the machine awake: a row can stall for as long as Windows is in standby (`fraud/RESULTS-IEEE.md`, Limits).

Training on every fit fraud (Phase 12): add `--full` to `--dry-run`, `--steps N` or `--train`. Output `fraud/models/<time>_lora-d1-ieee-full/`, about 5.25 h, safety stop at 10 h.
Every 250 batches the adapter so far goes to `checkpoint/` in that folder; it can't be resumed. The eval compares it with Phase 11's adapter on 20 new test windows (about 1.75 h):
`--windows` prints them, `--adapter <dir>` runs it, add `--resume <eval id>` to finish one, `--report <eval id>` rebuilds `fraud/results/eval_ieee_full/<eval id>.json|.md`.
Write-up: `fraud/RESULTS-IEEE-FULL.md`. Keep the screen on and pause Windows updates for the whole run: standby and an update restart killed the first attempt.

```bash
.venv/Scripts/python -m fraud.finetune_ieee --dry-run --full
.venv/Scripts/python -m fraud.finetune_ieee --steps 20 --full
.venv/Scripts/python -m fraud.finetune_ieee --train --full
.venv/Scripts/python -m fraud.eval_ieee_full --windows
.venv/Scripts/python -m fraud.eval_ieee_full --adapter fraud/models/<time>_lora-d1-ieee-full
.venv/Scripts/python -m fraud.eval_ieee_full --adapter fraud/models/<time>_lora-d1-ieee-full --resume <eval id>
.venv/Scripts/python -m fraud.eval_ieee_full --report <eval id>
```

```bash
.venv/Scripts/python -m fraud.eval_ieee --draw
.venv/Scripts/python -m fraud.eval_ieee --smoke 30
.venv/Scripts/python -m fraud.eval_ieee --ieee-adapter fraud/models/<time>_lora-d1-ieee
.venv/Scripts/python -m fraud.eval_ieee --ieee-adapter fraud/models/<time>_lora-d1-ieee --resume <eval id>
.venv/Scripts/python -m fraud.eval_ieee --reprice <eval id>
.venv/Scripts/python -m fraud.eval_ieee --report <eval id>
```

PaySim's older results at a 10 USD review (Phase 13): `--gate` only checks that the committed 50 USD numbers reproduce; with no flag it then prices both views and writes
`fraud/results/reprice_paysim/<id>.json|.md`. It needs the older traces in `fraud/runs/`, runs no model and changes no trace or older result. About a minute.

```bash
.venv/Scripts/python -m fraud.reprice_paysim --gate
.venv/Scripts/python -m fraud.reprice_paysim
```

Decisions that weigh the amount (Phase 14): no training. `--windows` prints the 20 new dev and 20 new test windows; `--adapter <dir>`, Phase 12's folder, scores them with logistic regression and the adapter (about 80 min), derives the other setups
from those scores and writes `fraud/results/eval_ieee_amount/<eval id>.json|.md`. Add `--resume <eval id>` to finish one; `--report <eval id>` rebuilds the results. Keep the screen on for the run.

```bash
.venv/Scripts/python -m fraud.eval_ieee_amount --windows
.venv/Scripts/python -m fraud.eval_ieee_amount --adapter fraud/models/<time>_lora-d1-ieee-full
.venv/Scripts/python -m fraud.eval_ieee_amount --adapter fraud/models/<time>_lora-d1-ieee-full --resume <eval id>
.venv/Scripts/python -m fraud.eval_ieee_amount --report <eval id>
```

On a new machine (Phase 15): install the PyTorch build for its hardware (`fraud/device.py` picks xpu, then cuda, then cpu), then run the gate. It writes `fraud/results/gate_nuc/gate.json`. `--score` needs the adapter and base model in `fraud/models/`.

```bash
.venv/Scripts/python -m fraud.dev.gate_nuc --logreg
.venv/Scripts/python -m fraud.dev.gate_nuc --sha
.venv/Scripts/python -m fraud.dev.gate_nuc --score
```

The FINAL windows and what is left for later phases, no model and no labels. Only the last phase may score a FINAL row:

```bash
.venv/Scripts/python -m fraud.final_test
```

Cost-model stress (Phase 16): Phase 14's recorded scores under other friction rates, minimums, flat prices, review fees and shares. No model and no new window, about 15 min. Writes `fraud/results/eval_cost_stress/<run id>.json|.md`; `fraud/dev/verify_cost_stress.py` checks it apart from the report code. Write-up: `fraud/RESULTS-COST-STRESS.md`.

```bash
.venv/Scripts/python -m fraud.eval_cost_stress
.venv/Scripts/python -m fraud.dev.verify_cost_stress <run id>
```

Column audit (Phase 17): logreg by column group, 8 new windows, about 5 min. `--leak` looks at whether columns lean on later rows, no labels. Write-up: `fraud/RESULTS-COLUMN-AUDIT.md`. Any tabular eval is checked apart from its report code by `fraud/dev/verify_tabular.py`.

```bash
.venv/Scripts/python -m fraud.eval_column_audit --leak
.venv/Scripts/python -m fraud.eval_column_audit
.venv/Scripts/python -m fraud.eval_column_audit --report <eval id>
.venv/Scripts/python -m fraud.dev.verify_tabular fraud/results/eval_column_audit/<eval id>.json
```

Better use of the rule (Phase 18): calibration, a paced review price and a customer line on 10 new windows, under a minute. `fraud/dev/explore_rule.py` and `explore_rule2.py` are the dev-only looks. Write-up: `fraud/RESULTS-RULE.md`.

```bash
.venv/Scripts/python -m fraud.eval_rule
.venv/Scripts/python -m fraud.eval_rule --report <eval id>
.venv/Scripts/python -m fraud.dev.verify_tabular fraud/results/eval_rule/<eval id>.json
```

History and category features for logreg (Phase 19): `fraud.history` builds the cache from the CSVs (about 12 s, no labels), then the eval runs on 10 new windows in about 3 min. `fraud/dev/explore_features.py` is the dev-only look. Write-up: `fraud/RESULTS-FEATURES.md`.

```bash
.venv/Scripts/python -m fraud.history
.venv/Scripts/python -m fraud.eval_features
.venv/Scripts/python -m fraud.eval_features --report <eval id>
.venv/Scripts/python -m fraud.dev.verify_tabular fraud/results/eval_features/<eval id>.json
```

## Where things land

| what | where |
|---|---|
| every captured run (captures, evals, escape checks) | `runs/shooter/`, `runs/dungeon/`: one `<time>_<label>.jsonl` per run, git-ignored |
| pinned runs (committed) | `runs/<game>/*.pinned.jsonl`, listed in `runs/README.md`; pin a run by renaming it |
| benchmark and eval results | `bench/results/<script>/` (committed) |
| lessons from past test and eval runs | `docs/lessons-learned.md` |
| the viewer of each game demo | `viewer/games/<game>.js` (a new demo adds one; see `viewer/README.md`) |
| fraud app runs and results | `fraud/runs/` (git-ignored but `*.pinned.jsonl`), `fraud/results/<script>/` (committed) |
