# local-system-one

A local proof of concept of a **System One-style decision engine**, inspired by Jev, running on an
ordinary open-weight LLM.

**This is not Jev and does not reproduce Jev.** Jev's model, architecture and training are not
public. This project reproduces only the publicly described *inference pattern*: answer
multiple-choice decisions with one forward pass, reading the next-token logits of the option
labels. All inference runs locally. No remote inference API is called.

Sources, and which claims are documented, observed, inferred or our own choices:
[`docs/research.md`](docs/research.md). Machine inspection and runtime choice:
[`docs/machine.md`](docs/machine.md).

## Where things are

| I want to… | Go to |
|---|---|
| run a demo | [`docs/shooter.md`](docs/shooter.md), [`docs/dungeon.md`](docs/dungeon.md), [`docs/grid.md`](docs/grid.md) |
| watch recorded runs | `python -m viewer`: the [Master Viewer](viewer/README.md) |
| find a run | [`runs/`](runs/README.md): one folder per game, one `.jsonl` per run |
| copy a command | [`RUN-GUIDE.md`](RUN-GUIDE.md): every command in one place |
| see the evidence | [`docs/research.md`](docs/research.md) (sources and Observed results), `bench/results/<benchmark>/` (raw data) |
| learn from past runs | [`docs/lessons-learned.md`](docs/lessons-learned.md) |
| read or change the engine | `system_one/` (see How it works below) |

## How it works

1. A decision is `(state, instruction, options)`. The engine renders one chat prompt that lists
   the options with labels `A`, `B`, `C`, … (two-letter labels above 26). It ends with the
   assistant prefill `{"choice": "`.
2. It checks at runtime that each label is **exactly one distinct token at that position** by
   tokenising `prompt + label + '"}'` and requiring the prompt tokens as an exact prefix. A label
   that merges with the prefill, or two options sharing a token, raise `TokenMappingError`.
3. **One forward pass** (no generation) gives the next-token log-probs. Only the label tokens are
   read. The result has the chosen option, a softmax over the allowed tokens, and the
   probability mass *outside* the allowed tokens (a diagnostic).
4. Many independent decisions share one batched forward pass (left padding, attention mask,
   mask-derived `position_ids`, `logits_to_keep=1`).
5. `answer="text"` makes the model answer with the option text instead. If the options' first
   tokens differ, one token decides. Otherwise a slower **multi-token fallback** scores each full
   option sequence.
6. **Tiered goals** (`system_one/goals.py`): slow tiers choose goals from fixed lists, and faster
   tiers see those goals as context. **Tournament sampling** (`system_one/tournament.py`): large
   option sets are split into groups, and the winners are regrouped until one is left. Each
   round is one batch.

The probabilities are softmax values from an ordinary LLM. **They are not calibrated.** Measured:
all tested models are overconfident (see Known limitations).

```python
from system_one import Decision, load_config, make_engine

engine = make_engine(load_config("config/default.toml"))
r = engine.decide(Decision(
    instruction="Which team should handle this ticket?",
    options=("returns and exchanges", "shipping and delivery", "billing and payments"),
    state="My shoes arrived in the wrong size. Can I exchange them?",
))
print(r.choice, r.probs, r.outside_mass)          # one forward pass
results = engine.decide_batch([...])               # many decisions, one forward pass
```

## System requirements

- Python 3.11+ (the config loader uses `tomllib`).
- No GPU is required. CPU works (`device = "cpu"` in the config); a GPU just makes it faster.
- Model size is the main hardware constraint. Pick a model that fits your GPU/shared memory (or
  RAM, for CPU), then start from the matching config. Weight size (bfloat16) is the floor — the
  KV-cache and activations add more on top, growing with `max_batch` and prompt length.

| Available GPU / shared memory | Recommended model | Config to start from |
|---|---|---|
| CPU only | Qwen2.5-0.5B-Instruct (1.0 GB weights) | `config/default.toml`, set `device = "cpu"`, lower `max_batch` |
| ~4–6 GB | Qwen2.5-0.5B or 1.5B-Instruct | `config/default.toml` (lower `max_batch` if it runs out of memory) |
| ~8 GB (e.g. a laptop iGPU) | Qwen2.5-1.5B-Instruct (3.1 GB weights) — the default | `config/default.toml` |
| ~8 GB, trading speed for accuracy | Qwen2.5-3B-Instruct (6.2 GB weights; Qwen Research licence, non-commercial) | `config/lenovo-3b.toml` (lower `max_batch` if it runs out of memory) |
| ≥16 GB (discrete GPU) | Qwen2.5-7B-Instruct (~15 GB weights) or larger | `config/nuc.example.toml` → copy to `config/nuc.toml` |

Any decoder-only Hugging Face chat model works — `system_one check` verifies at runtime that its
option labels tokenise to single tokens after the prefill and fails with a clear error if not.
This table only covers the models actually tried; see [`docs/research.md`](docs/research.md) →
Observed → Model selection for how the default was chosen. Concrete numbers for one specific
machine are in [Benchmarks](#benchmarks) below.

## Setup

All commands without the explanations: [RUN-GUIDE.md](RUN-GUIDE.md).

```bash
py -3.13 -m venv .venv
```

Install the PyTorch build for your hardware first (see `requirements.txt` for the Intel XPU /
NVIDIA CUDA / CPU-only index URLs), then the rest of the dependencies:

```bash
.venv/Scripts/python -m pip install -r requirements.txt
```

The first real run downloads `Qwen/Qwen2.5-1.5B-Instruct` (3.1 GB) from Hugging Face into the
standard HF cache. After that everything runs offline (set `HF_HUB_OFFLINE=1` to be sure).

```bash
.venv/Scripts/python -m system_one check
```

```bash
.venv/Scripts/python -m system_one decide --question "Which colour is the sky on a clear day?" --options red blue yellow
```

Tests. The `model` tests load Qwen2.5-0.5B-Instruct (1 GB) and skip when it is not in the HF
cache. Download it once with `.venv/Scripts/python -c "from huggingface_hub import snapshot_download;
snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"`, or point them at another cached model with
`SYSTEM_ONE_TEST_MODEL=Qwen/Qwen2.5-1.5B-Instruct`.

```bash
.venv/Scripts/python -m pytest
```

```bash
.venv/Scripts/python -m pytest -m "not model"
```

## Demos

Each demo turns a small game into decisions for the engine. The model makes every decision, and
failed runs are shown as they happened. Headline results (Observed, on the Lenovo's Arc 140V
unless noted; full tables in each write-up):

| demo | what the agent does | headline result | write-up |
|---|---|---|---|
| **Shooter** | one agent clears locked rooms of gunners and brutes, takes the key, leaves by the exit | classic game, seeds 0–9: the 1.5B (order averaging) and the 3B (listed order) each escape **7 of 10**; with ammo, the 1.5B's fire head escapes **6 of 10** vs 0 of 10 without it | [`docs/shooter.md`](docs/shooter.md) |
| **Dungeon** (replaced by the shooter) | one agent looks for the key and the exit among chasing enemies, gems and food | no run escaped (**0 of 48** model runs); the agent walks into enemies it is warned about | [`docs/dungeon.md`](docs/dungeon.md) |
| **2D grid** (live window) | four agents collect gems, eat food and avoid hazards, all in one batch per tick | tiered goals: 2.3× the gems and 5.4× the food of flat control, which was no better than random | [`docs/grid.md`](docs/grid.md) |

```bash
.venv/Scripts/python -m demo.shooter.capture --config config/default.toml --seed 0
```

Every captured run (single captures and eval runs alike) lands in the game's pool, `runs/shooter/`
or `runs/dungeon/`, as one `<time>_<label>.jsonl`. Each future game demo gets its own pool
(`runs/<game>/`) and its own viewer (`viewer/games/<game>.js`).

## Watching runs: the Master Viewer

```bash
.venv/Scripts/python -m viewer
```

One page for every demo's runs; it replays recorded decisions and never runs the model. The start
page offers three ways to watch:

- **Load a trace**: pick or drop any `.jsonl`.
- **Random run**: one run from the pool, in the full view (map, decision cards, latency timeline,
  the prompts of each tick, and "About this run").
- **Auto-demo**: two panes play the pool from newest to oldest. When a pane's run ends, it takes
  the next older run; after the oldest, both start over at the newest.

A filter (All / Shooter / Dungeon) limits random and auto-demo to one game.

- `python -m viewer <run>.jsonl` opens one run.
- `python -m viewer --bundle <run>.jsonl` writes one self-contained `.html` with that run, to
  share.

The committed showcase runs are listed in [`runs/README.md`](runs/README.md). How the viewer is
built, and how to add a viewer for a new demo: [`viewer/README.md`](viewer/README.md).

## Benchmarks

Machine: Lenovo 83HM, Intel Core Ultra 7 256V, Intel Arc 140V iGPU (8 GB shared), 15.6 GB RAM,
Windows 11. Runtime: PyTorch 2.14.0+xpu, transformers 5.17.0. Model: Qwen2.5-1.5B-Instruct,
bfloat16, no quantisation. Median of 5 runs after warm-up. Raw data:
`bench/results/bench/bench_Qwen2.5-1.5B-Instruct_bfloat16_20260926_050240.{json,md}`.

| prompt tokens | batch | sequential ms/decision | batched ms/decision | speed-up | batched decisions/s |
|---:|---:|---:|---:|---:|---:|
| ~165 | 1 | 64.2 | 66.1 | 1.0× | 15.1 |
| ~165 | 8 | 65.0 | 39.8 | 1.6× | 25.1 |
| ~165 | 32 | 65.2 | 36.3 | 1.8× | 27.5 |
| ~350 | 8 | 101.5 | 88.3 | 1.15× | 11.3 |
| ~350 | 32 | 101.0 | 77.0 | 1.3× | 13.0 |
| ~1120 | 8 | 272.1 | 270.2 | 1.0× | 3.7 |
| ~1120 | 32 | 267.7 | 297.9 | 0.9× | 3.4 |

Generating the answer as text instead (16 decisions, ~350-token prompts):

| method | batch | ms/decision |
|---|---:|---:|
| single-token read | 1 | 99.8 |
| generate `{"choice": "X"}` (7 tokens) | 1 | 358.0 |
| single-token read | 8 | 88.3 |
| generate `{"choice": "X"}` (7 tokens) | 8 | 127.4 |

On this iGPU, prompt processing is compute-bound. Batching therefore helps much less than on a
data-centre GPU: at most 1.8× here, and nothing for long prompts. The single-token read's
advantage over generation is 3.6× at batch 1 but only 1.4× at batch 8. Details and caveats:
[`docs/research.md`](docs/research.md) → Observed.

Tournament vs one full decision (1.5B; 30 problems each; one correct option):

| options | one full decision | tournament, groups ≤10 |
|---:|---:|---:|
| 10 | 93% | 93% |
| 26 | 87% | 93% |
| 40 | 47% | 100% |
| 80 | 17% | 80% |

Reproduce:

```bash
.venv/Scripts/python -m bench.bench --gen-baseline
```

```bash
.venv/Scripts/python -m bench.model_eval --models Qwen/Qwen2.5-1.5B-Instruct
```

```bash
.venv/Scripts/python -m bench.tournament_compare
```

## Switch model or runtime

Only the config changes. There are no code changes.

1. Install the PyTorch build for your hardware (see `requirements.txt`: the `xpu` index for
   Intel Arc/Iris, a CUDA index such as `cu128` for NVIDIA (check pytorch.org for the current
   one), `cpu` otherwise), then `pip install -r requirements.txt`.
2. Copy `config/nuc.example.toml` (a larger-hardware example) to `config/nuc.toml`. Set `model`
   (any decoder-only Hugging Face chat model — see [System requirements](#system-requirements)),
   `device` (`auto`, `cuda`, `xpu` or `cpu`), `dtype` and `max_batch`.
3. Select it per command with `--config config/nuc.toml`, or for the whole shell with the
   environment variable `SYSTEM_ONE_CONFIG=config/nuc.toml`.
4. Verify it: `python -m system_one --config config/nuc.toml check`. This checks full-vocabulary
   logits, single-token labels for the new tokenizer, and batched vs sequential agreement. It
   also prints how many of 6 easy direction questions the model gets right. That score is
   informational (the 1.5B gets 4/6; it mixes up east and west) and does not affect OK/FAILED.

Adding a different runtime (for example llama.cpp or OpenVINO) means one class that implements
`system_one.backends.base.Backend` (tokenize, chat template, `next_token_logits` for a batch)
and one entry in `BACKENDS` in `system_one/config.py`. The engine, goals, tournaments and demo
do not change.

## Layout

```
system_one/            the engine: engine.py (decisions), goals.py (tiers), tournament.py, config.py
system_one/backends/   base.py (interface), hf.py (PyTorch + transformers), llamacpp.py (local
                        llama-server, GGUF; untested), mock.py (no model)
demo/                  the game demos; runs.py (where captured runs go), common.py (shared helpers)
demo/grid/             the live 2D window: world.py, brain.py, sim.py (python -m demo.grid)
demo/dungeon/          world.py (rules), brain.py (tiers, text, runner), capture.py (a run into runs/dungeon/)
demo/shooter/          world.py (rules), bots.py (non-model reference bots), brain.py (tiers, heads,
                        runner), capture.py (a run into runs/shooter/)
viewer/                the Master Viewer (python -m viewer): index.html, shell.js, core.js, paint.js,
                        server.py; games/ holds one renderer per game demo
runs/                  captured runs, one flat pool per game (shooter/, dungeon/); only *.pinned.jsonl
                        is committed
bench/                 evaluations and benchmarks, one script each; results/<script>/ holds their output
config/                default.toml (1.5B, default), lenovo-3b.toml (3B alternative),
                        nuc.example.toml (larger-hardware example, 7B), mock.toml (no model)
docs/                  shooter.md, dungeon.md, grid.md (the demos), research.md (sources + Observed
                        results), lessons-learned.md, machine.md, deep-research-report.md (JEV background)
tests/                 pytest suite (mock tests always; `model` tests when weights are cached)
```

## Known limitations

- **Not calibrated.** Probabilities are softmax values renormalised over the allowed tokens.
  On a 58-item check, Qwen3-1.7B averaged 0.99 confidence at 57% accuracy, and the chosen 1.5B
  0.77 at 66%. Do not branch on them as if they were calibrated.
- **Small models make poor goal decisions.** No model tried (see System requirements) met our
  accuracy rule. All fail rule-based goal selection (30–40%). The 1.5B strongly prefers "find food" and
  is also sensitive to option order. S1 describes this limitation: one forward pass cannot
  derive goals. Tiered goals structure the problem; they do not make the model smarter.
- **Accuracy falls fast with many options in one decision** (1.5B: 47% at 40 options, 17% at 80).
  Use tournaments; they cost about 2× latency.
- **The prefill is not neutral.** Reading the next token after `{"choice": "` agreed with the
  model's own unprefilled bare-label answer only ~50% of the time on ambiguous decisions.
- **Probability resolution**: with a bf16 LM head, close options can tie exactly (logits are
  rounded to ~0.125 steps). The default config computes the head in float32 (`head_dtype`,
  +0.9 GB); the benchmark tables above were measured with the bf16 head.
- **Batching gains are small on the reference iGPU** (≤1.8×; see Benchmarks). Long prompts in
  large batches were slower than sequential. bf16 batching changes probabilities by up to ~0.07
  versus sequential and can flip near-ties. float32 is exact but ~7× slower there.
- **Latency is not constant**: it grows with total prompt tokens in the batch. A demo tick that
  also re-plans (long target lists) takes 2–3× longer than an action-only tick. The first forward
  pass in a new process takes 1.5–4 s on the XPU (kernel compilation). The demo and benchmarks
  warm up first; a one-off `decide` from the command line pays it.
- **Prefix caching gains are small** (~9–10% on demo moves) with the prompt order that keeps
  accuracy. Orders that put the question or options first reuse more but lost accuracy
  (3B navigation 94% → 62–81%). The cache needs full-attention models; it turns itself off
  otherwise.
- Not implemented: Score and yes/no question types, quantised backends, training or calibration
  of any kind.
- The eval sets are small (58 items; 30 problems per tournament cell). Treat the accuracy
  numbers as smoke tests.
- **Averages hide traps.** Single decisions were 97–100% correct in a move probe. But a
  deterministic model repeats a mistake every time it returns to the same state, so an agent can
  loop forever. Test agents in closed loop, not only per decision. Hazard avoidance by the
  strategy tier is still weak (11–44% in the probe).
- **The first dungeon was not solved** (0 of 48 model runs escaped). Its shooter redo is beatable
  (1.5B: 7 of 10 escaped), but the models still step into bullets they are warned about: nearly
  every hit was a move labelled `BULLET` chosen while a safe move was offered.
- **Option order sways small models.** In the shooter the 1.5B's decision to fire followed the
  option order; order averaging (`Engine(order_debias=True)`) fixes that at the cost of twice the
  rows per forward pass. It is off by default outside the shooter.
- The demo state text gives relative offsets and names conditions in words, because the model
  cannot do the arithmetic in one pass. That is part of the demo design, not of the engine.

## Sources

- Sean Goedecke, [Two techniques for working with System One models](https://www.seangoedecke.com/two-techniques-for-working-with-system-one-models/) (primary)
- Sean Goedecke, [Jev means structured output is interesting again](https://www.seangoedecke.com/jev-means-structured-output-is-interesting-again/)
- Sean Goedecke, [System One models like Jev can train their own replacements](https://www.seangoedecke.com/system-one-models-can-train-their-own-replacements/)
- [sgoedecke/system-one](https://github.com/sgoedecke/system-one) (reference implementation linked from the primary source)
- TypeSafe, [Introducing System One models and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
