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

### The 2D demo

```bash
.venv/Scripts/python -m demo.sim
```

```bash
.venv/Scripts/python -m demo.sim --config config/mock.toml
```

```bash
.venv/Scripts/python -m demo.sim --headless --ticks 100
```

Four agents collect gems, eat food and avoid roaming hazards. Every tick, **all agents' due
decisions go through the engine in one batched forward pass**. Each agent has three tiers:

| tier | every | options |
|---|---|---|
| strategy | 12 ticks | collect gems / find food / avoid hazards / explore |
| target | 6 ticks, or when the target is reached or gone | every gem (24) or food item (12) on the map, safe spots, or regions. Sets larger than 8 use a **tournament**, one round per tick |
| action | every tick | move north / south / east / west / stay, each labelled with its outcome, e.g. `move west (target: 2 steps)`, `move north (wall)` |

The agent is **aware of its condition**. The strategy state spells out consequences ("about 20
ticks until starving"). A change in condition (an energy band, low health, an adjacent hazard)
makes the strategy re-decide on the next tick. Lower tiers see the target without its stale
distance. In a 4-seed × 250-tick comparison with the 3B, this took starvation from 2 of 4 runs to
0 of 4, halved "stuck" time and raised gems by 22%. `docs/research.md` → Observed → Agent
awareness has the diagnosis.

The window shows each agent's strategy and target with their probabilities, the tournament
progress, the action probabilities as bars, the outside-token mass, and per tick the batch size
and forward-pass latency. Agents glide between cells over about one tick; this is rendering
only. Space pauses, Esc quits. `--config config/mock.toml` runs without a model (random
decisions). `--no-goals` gives flat control (action tier only) for comparison.

`--plan-budget N` (default 1) lets each agent make at most N planning decisions (strategy,
target, tournament groups) per tick, next to its move. The rest wait for later ticks. This keeps
tick latency nearly constant. `-1` runs a whole tournament round per tick, which is how the
comparison below was measured.

The most responsive setup measured (see [Benchmarks](#benchmarks) for the hardware), one agent
with the 3B model:

```bash
.venv/Scripts/python -m demo.sim --config config/lenovo-3b.toml --agents 1
```

That runs at ~5 ticks/s: ~137 ms for a move-only tick and ~230 ms when a planning decision
rides along. Over 2 × 80 ticks the median was 194–195 ms, p90 236–238 ms and max 265–266 ms.
Without the budget, planning ticks took 550–880 ms.

`prefix_cache = true` (on by default in the bundled configs) reuses the keys/values of prompt
prefixes that repeat across ticks, which saves ~9–10% here. A 3B pass costs ~80 ms even for 8
tokens on the reference GPU (see Benchmarks), so reuse cannot go much further. Prompt orders that
reuse more (`prompt_order = "question_first"` or `"options_first"`) were up to 21% faster but hurt
accuracy, and the agent played badly. Details: `docs/research.md` → Observed → Prefix caching.

### The shooter demo: the dungeon redone, one agent, recorded and replayed

The dungeon below was never escaped. This is its redo as a room-clearing shooter in the spirit of
*Enter the Gungeon*: the agent shoots, rooms lock it in until their enemies are dead, and
doorways are three cells wide, so no enemy can trap it in one. One agent per run, in a seeded
dungeon of nine rooms with a key, an exit, gunners that aim for a tick and then fire slow
bullets, and brutes that walk up and hit. Its standing order, as the model reads it: *"Find the
key, then leave through the exit alive. Rooms lock you in until their enemies are dead: shoot
them and keep out of their bullets."* The model makes every decision, and failed runs are shown
as they happened.

Two variants, one per model:

```bash
.venv/Scripts/python -m demo.shooter.capture --config config/default.toml --seed 0
```

```bash
.venv/Scripts/python -m demo.shooter.capture --config config/lenovo-3b.toml --seed 0
```

The first runs Qwen2.5-1.5B, the second Qwen2.5-3B. Each writes `trace.jsonl` and `replay.html`
to `demo/output/shooter/<model>_seed0/`; open `replay.html` in a browser (no server, no network).
`--config config/mock.toml` runs without a model, `--rebuild <trace.jsonl>` rebuilds the page.

Recorded runs, the pre-registered showcase (seed 0, whatever the outcome): [`docs/shooter_replay_1.5b_seed0.html`](docs/shooter_replay_1.5b_seed0.html)
(the 1.5B picks up the key and dies to a gunner at tick 98) and
[`docs/shooter_replay_3b_seed0.html`](docs/shooter_replay_3b_seed0.html) (the 3B dies to a gunner at
tick 48, with order averaging, its pre-registered setting). Also included:
[`docs/shooter_replay_3b_listed_seed0.html`](docs/shooter_replay_3b_listed_seed0.html), seed 0 with the
3B's current setting (it escapes at tick 163), and
[`docs/shooter_replay_1.5b_seed1_escaped.html`](docs/shooter_replay_1.5b_seed1_escaped.html), the
1.5B's first escaped seed, chosen after the results. Download them and open locally.

[`docs/shooter_quad.html`](docs/shooter_quad.html) plays four runs side by side, two 1.5B and two 3B
picked at random on each load (Shuffle picks again), with each run's decisions drawn as a colour-coded
graph: one colour per strategic goal, a marker each time the goal was re-checked or changed. The pool
is the kept runs in `demo/output/shooter/` (the best run per version, so not a random sample of all
runs) plus the four replays above; `python -m demo.shooter.quad` rebuilds it.

Each tick is one batched forward pass with up to three decisions (strategy or target, move, shoot):

| tier | every | options |
|---|---|---|
| strategy | 12 ticks, or at once when the situation changes | the goals possible now: explore, fight the enemies here, get the key, go to the exit, drink a health potion |
| target | 6 ticks, or when reached, gone or unsafe | unexplored rooms, the key, the exit, potions, or up to 5 firing spots (a clear line to an enemy, away from enemies, out of every bullet's path) |
| move (control head) | every tick | open moves and stay, labelled with outcomes: `move west (safe; closer: 4 steps to the target)`, `stay (BULLET: -15 health)` |
| shoot (control head) | every tick | `shoot gunner #4, 3 cells east (AIMING at you; 3 hits to kill; clear line)` or `hold fire`; committed without a model call while the gun reloads |

**Order averaging (implementation choice, per model).** Small models often pick an option for its
position. On a development seed the 1.5B's decision to fire followed the option order in 34 of 34
recorded states (hold fire listed last: never fired; listed first: always fired). With order
averaging (`Engine(order_debias=True)`), every decision is read twice in the same batch, with the
options as listed and reversed, and the two readings are averaged; it doubles the rows per pass,
and the replay shows both readings as ticks on each probability bar. It is on for the 1.5B and,
after the evaluation below, off for the 3B (`[shooter] order_debias` in each config; `--order-debias`
overrides).

The replay page is the dungeon viewer with more contrast (lighter floors, darker walls, bright
enemies and bullets): the map with fog, sealed doorways (red bars), aim telegraphs (red dashed
lines), bullets in flight and enemy health pips; one card per tier and head; the latency
timeline with hits and kills; a prompt inspector; `#tick-N` deep links; WebM recording.

Closed-loop evaluation (`bench/shooter_eval.py`; seeds, setups, metrics and showcase fixed and
committed before the first run on those seeds):

```bash
.venv/Scripts/python -m bench.shooter_eval
```

Results on the reference machine (Observed; seeds 0–9; `bench/results/shooter_eval_20260927_002315.json`):

| setup | escaped | died | out of time | key picked up | rooms cleared | kills | hits taken | forward ms/tick (median / p90) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **1.5B, order averaging (the 1.5B demo)** | **7 / 10** | 2 | 1 | 9 | 4.6 | 8.5 | 3.0 | 245 / 410 |
| 3B, order averaging (pre-registered default) | 2 / 10 | 8 | 0 | 10 | 4.8 | 9.1 | 6.7 | 338 / 803 |
| 1.5B, listed order only | 0 / 10 | 8 | 2 | 5 | 0.7 | 0.7 | 6.1 | 176 / 317 |
| **3B, listed order (the 3B demo since the replication)** | **7 / 10** | 2 | 1 | 9 | 6.5 | 11.4 | 5.4 | 195 / 387 |
| random decisions (mock) | 0 / 10 | 0 | 10 | 0 | 0 | 0 | 0 | 0.8 / 1.9 |
| reference bot (hand-written, not a model): the ceiling | 10 / 10 | 0 | 0 | 10 | 6.1 | 11.2 | 0.4 | – |
| the same bot, never dodging | 5 / 10 | 5 | 0 | 9 | 4.7 | 9.1 | 8.1 | – |

- **Beatable.** The rules were tuned against the bots before any model ran (reference bot 60/60 on
  dev seeds), and the 1.5B escaped 7 of 10 evaluation seeds. The first dungeon: 0 of 32.
- **Order averaging is what makes the 1.5B play**: without it, it held fire in 98% of its shoot
  decisions and never escaped.
- **The 3B did worse with averaging than without** (2 vs 7 of 10, p = 0.07). A post-hoc
  replication on new seeds 10–19, with its decision rule committed before the run, gave 6 vs 7:
  by that rule the 3B demo now uses the listed order. Over all 20 seeds: 3B listed 14/20,
  3B averaged 8/20 (p = 0.11), 1.5B averaged 14/20. Both demos escape about 7 runs in 10.
- **Nearly every hit was a chosen risk**: 29 of 30 (1.5B) and 66 of 67 (3B) hits came from a
  move labelled `BULLET` or `next to a brute` while a safe move was offered. The 3B often steps
  toward the gunner it is fighting.

Details, development probes and the rule calibration: `docs/research.md` → Observed → Shooter demo.

### The first dungeon demo (replaced by the shooter): one agent, recorded and replayed

Kept with its results: the shooter above is its redo.

One agent in a seeded dungeon: nine rooms joined by corridors, a key that opens the exit, two
enemies that chase, and gems, food and potions. The agent knows only the rooms it has seen.
Its standing order, as the model reads it: *"Find the key, then leave through the exit alive.
Gems are a bonus. Eat and heal when needed."* The model makes every decision; nothing is
scripted, and failed runs are shown as they happened.

Capture a run with the real model, then open the replay:

```bash
.venv/Scripts/python -m demo.dungeon.capture --config config/lenovo-3b.toml --seed 0
```

This writes `trace.jsonl` and `replay.html` to `demo/output/dungeon/<model>_seed0/`. Open
`replay.html` in a browser. It needs no server and no network. `--config config/mock.toml` runs
without a model (random decisions). `--rebuild <trace.jsonl>` rebuilds the page from an existing
trace.

A recorded run is in [`docs/dungeon_replay_seed0.html`](docs/dungeon_replay_seed0.html) (download
it and open it locally): the pre-registered showcase, seed 0 with the 3B. It collects 8 gems and
explores 5 of 9 rooms, never finds the key, and dies to an enemy at tick 143.

The replay page shows:

- The map. Dimmed rooms are rooms the agent has not seen yet (you see the whole map; the agent
  does not). The dashed line and ring mark its current target; dots show its last 15 cells, so
  loops are visible.
- One card per tier: the question, the options with the label the model answered with (`A`,
  `B`, …), the probabilities, the chosen answer, and the latency of that tick's batched forward
  pass. Each card also shows the tier's state: new answer, same answer, held since tick N,
  deciding (tournament in progress, or queued behind the one-planning-decision-per-tick
  budget), or only option (committed without a model call).
- A timeline of forward-pass latency per tick, with events (key, hits, gems, food, rooms) and
  "no progress for 6+ ticks" stretches.
- Below the stage: play/pause (Space), step (← →), speed (1× is the recorded speed), scrubbing,
  and *What the model saw*, which shows every prompt of the current tick. Add `#tick-120` to the
  URL to open that tick. **Record WebM** saves the stage as a video (Chrome or Edge, file opened
  locally). Convert it with `ffmpeg -i run.webm -c:v libx264 -pix_fmt yuv420p -crf 18 run.mp4`.

| tier | every | options |
|---|---|---|
| strategy | 12 ticks, or at once when health, energy, visible enemies, the key or the rooms seen change | the goals possible now, from: explore, get the key, go to the exit, collect gems, eat food, drink a health potion, flee the enemy |
| target | 6 ticks, or when the target is reached or gone | unexplored rooms behind known doors, the key, the exit, each known gem/food/potion, or safe spots. More than 8 options run as a tournament |
| action | every tick | 5 moves labelled with outcomes, e.g. `move west (closer: 2 steps to the target)`, `move north (ENEMY: -30 health)` |

**Why record and replay (implementation choice).** The Doom demo in sgoedecke/system-one captures
decisions to JSONL and renders a video from them [S4]. We do the same, but render in the browser:

- One self-contained HTML file (canvas and plain JavaScript). There are no dependencies, no
  server and no build step, and the page opens from disk.
- The replay shows the exact recorded decisions and latencies. It can play at the recorded speed
  (~5 ticks/s with the 3B on the reference iGPU; see Benchmarks) or faster, pause on any
  decision, and scrub.
- The video export records the same canvas, so there is no second renderer to keep in sync.
- Rejected: a live browser mode (it needs a server; the tkinter demo remains for live runs), and
  rendering MP4 frames in Python (a second renderer, and an image library we do not have).

Closed-loop evaluation (`bench/dungeon_eval.py`; setups, seeds and metrics were fixed before the
first run):

```bash
.venv/Scripts/python -m bench.dungeon_eval
```

Results on the reference machine (Observed; see [Benchmarks](#benchmarks) for the hardware;
seeds 0–7; `bench/results/dungeon_eval_20260926_163151.json`):

| setup | escaped | died: enemy | died: starvation | key picked up | rooms seen | stuck ticks | forward ms/tick (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3B, `closer/farther` wording (default) | 0 / 8 | 7 | 1 | 2 | 3.6 | 5.8 | 165 |
| 3B, first demo's wording (`target: N steps`) | 0 / 8 | 3 | 5 | 1 | 2.9 | 63.4 | 153 |
| 1.5B, `closer/farther` wording | 0 / 8 | 7 | 1 | 1 | 3.0 | 19.4 | 82 |
| random decisions (mock) | 0 / 8 | 1 | 7 | 0 | 1.5 | 54.4 | 0.4 |

**No agent escaped, and none even saw the exit room.** The `closer/farther` wording removed the
loops (stuck ticks 63 → 6), and the agent then walked into enemies: in 21 of 29 hits it chose the
move labelled `ENEMY`. A post-hoc change that spelled out enemy consequences and safer flee
targets made no difference on seeds 0–7 or on 8 new seeds, so it is off by default
(`--enemy-aware`). The move tier picks a move toward its target 97–98% of the time, whatever the
danger labels say. Tournaments never ran: no agent knew more than 8 gems at once. Details:
`docs/research.md` → Observed → Dungeon demo.

## Benchmarks

Machine: Lenovo 83HM, Intel Core Ultra 7 256V, Intel Arc 140V iGPU (8 GB shared), 15.6 GB RAM,
Windows 11. Runtime: PyTorch 2.14.0+xpu, transformers 5.17.0. Model: Qwen2.5-1.5B-Instruct,
bfloat16, no quantisation. Median of 5 runs after warm-up. Raw data:
`bench/results/bench_Qwen2.5-1.5B-Instruct_bfloat16_20260926_050240.{json,md}`.

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

### Demo results

![demo window with Qwen2.5-1.5B on the Arc 140V](docs/demo.png)

With the real model on the Lenovo, the demo runs at ~3 ticks/s: 300–450 ms per tick for 6–9
decisions in one batch. Tiered goals vs flat control, 100 ticks, 4 agents, mean of 3 seeds
(`bench/results/demo_compare_20260926_051317.json`):

| setup | gems | food eaten | hazard hits | forward ms/tick (median) |
|---|---:|---:|---:|---:|
| model, tiered goals | 36.0 | 59.3 | 10.7 | 312 |
| model, flat (action tier only) | 15.7 | 11.0 | 5.7 | 198 |
| random decisions (mock), tiered goals | 17.7 | 10.7 | 8.3 | 3 |

Flat control was no better than random. Tiered goals gave 2.3× the gems and 5.4× the food, but
also more hazard hits. Tournament vs one full decision (1.5B; 30 problems each; one correct option):

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

```bash
.venv/Scripts/python -m bench.demo_compare
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
system_one/            engine.py (decisions), goals.py (tiers), tournament.py, config.py
system_one/backends/   base.py (interface), hf.py (PyTorch + transformers), mock.py (no model)
demo/                  world.py (grid world), brain.py (tiers + state text), sim.py (window / headless)
demo/dungeon/          world.py (rules), brain.py (tiers, text, runner), capture.py (trace + replay), viewer.html
demo/shooter/          world.py (rules), bots.py (non-model reference bots), brain.py (tiers, heads, runner),
                        capture.py (trace + replay), viewer.html
bench/                 bench.py, model_eval.py, tournament_compare.py, demo_compare.py, dungeon_eval.py,
                        shooter_calibration.py, shooter_eval.py, results/
config/                default.toml (1.5B, default), lenovo-3b.toml (3B alternative),
                        nuc.example.toml (larger-hardware example, 7B), mock.toml (no model)
docs/                  research.md (sources + Observed results), machine.md
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
