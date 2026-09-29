# Dungeon demo (the first dungeon; replaced by the shooter)

Kept with its results: the [shooter](shooter.md) is its redo.

One agent in a seeded dungeon:

- nine rooms joined by corridors;
- a key that opens the exit;
- two enemies that chase;
- gems, food and potions.

The agent knows only the rooms it has seen. Its standing order, as the model reads it: *"Find the
key, then leave through the exit alive. Gems are a bonus. Eat and heal when needed."* The model
makes every decision; nothing is scripted, and failed runs are shown as they happened. Code:
`demo/dungeon/`. Viewer: `viewer/games/dungeon.js`.

## Run it

```bash
.venv/Scripts/python -m demo.dungeon.capture --config config/lenovo-3b.toml --seed 0
```

This writes one trace to the dungeon pool, `runs/dungeon/<time>_<model>_seed0.jsonl`.
`--config config/mock.toml` runs without a model (random decisions). Watch it in the Master Viewer:

```bash
.venv/Scripts/python -m viewer
```

The pinned run, `runs/dungeon/20260926-162605_qwen2.5-3b_seed0.pinned.jsonl`, is the pre-registered
showcase: seed 0 with the 3B. It collects 8 gems and explores 5 of 9 rooms. It never finds the
key, and dies to an enemy at tick 143.

## Decisions

| tier | every | options |
|---|---|---|
| strategy | 12 ticks, or at once when health, energy, visible enemies, the key or the rooms seen change | the goals possible now, from: explore, get the key, go to the exit, collect gems, eat food, drink a health potion, flee the enemy |
| target | 6 ticks, or when the target is reached or gone | unexplored rooms behind known doors, the key, the exit, each known gem/food/potion, or safe spots. More than 8 options run as a tournament |
| action | every tick | 5 moves labelled with outcomes, e.g. `move west (closer: 2 steps to the target)`, `move north (ENEMY: -30 health)` |

## What the viewer shows

- **The map.** Dimmed rooms are rooms the agent has not seen yet (you see the whole map; the agent
  does not). The dashed line and ring mark its current target. Dots show its last 15 cells, so
  loops are visible.
- **One card per tier.** Each card shows:
  - the question, and the options with the label the model answered with (`A`, `B`, …);
  - the probabilities and the chosen answer;
  - the latency of that tick's batched forward pass;
  - the tier's state: new answer, same answer, held since tick N, deciding (a tournament in
    progress, or queued behind the one-planning-decision-per-tick budget), or only option
    (committed without a model call).
- **A timeline** of forward-pass latency per tick, with events (key, hits, gems, food, rooms) and
  "no progress for 6+ ticks" stretches.
- **Below the stage:**
  - play/pause (Space), step (← →), speed (1× is the recorded speed) and scrubbing;
  - *What the model saw*, every prompt of the current tick;
  - `&tick=120` at the end of a `#run=` link opens that tick;
  - **Record WebM**, which saves the stage as a video (Chrome or Edge).

**Why record and replay (implementation choice).** The Doom demo in sgoedecke/system-one captures
decisions to JSONL and renders a video from them ([S4](research.md)). We do the same, but render in the browser:

- The page is plain JavaScript and canvas, with no dependencies and no build step.
  - `python -m viewer` serves it together with the run pools, because a page opened from disk
    cannot list a folder.
  - `python -m viewer --bundle <run>` still writes one self-contained file that opens from disk.
- The replay shows the exact recorded decisions and latencies. It can play at the recorded speed
  (~5 ticks/s with the 3B on the reference iGPU) or faster, pause on any decision, and scrub.
- The video export records the same canvas, so there is no second renderer to keep in sync.
- Rejected:
  - a live browser mode (the tkinter [grid demo](grid.md) remains for live runs);
  - rendering MP4 frames in Python (a second renderer, and an image library we do not have).

## Results

Closed-loop evaluation (`bench/dungeon_eval.py`). Setups, seeds and metrics were fixed before the
first run:

```bash
.venv/Scripts/python -m bench.dungeon_eval
```

Lenovo (Arc 140V), seeds 0–7 (Observed; `bench/results/dungeon_eval/dungeon_eval_20260926_163151.json`):

| setup | escaped | died: enemy | died: starvation | key picked up | rooms seen | stuck ticks | forward ms/tick (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3B, `closer/farther` wording (default) | 0 / 8 | 7 | 1 | 2 | 3.6 | 5.8 | 165 |
| 3B, first demo's wording (`target: N steps`) | 0 / 8 | 3 | 5 | 1 | 2.9 | 63.4 | 153 |
| 1.5B, `closer/farther` wording | 0 / 8 | 7 | 1 | 1 | 3.0 | 19.4 | 82 |
| random decisions (mock) | 0 / 8 | 1 | 7 | 0 | 1.5 | 54.4 | 0.4 |

**No agent escaped, and none even saw the exit room.**

- The `closer/farther` wording removed the loops (stuck ticks 63 → 6), and the agent then walked
  into enemies: in 21 of 29 hits it chose the move labelled `ENEMY`.
- A post-hoc change that spelled out enemy consequences and safer flee targets made no difference,
  on seeds 0–7 or on 8 new seeds, so it is off by default (`--enemy-aware`).
- The move tier picks a move toward its target 97–98% of the time, whatever the danger labels say.
- Tournaments never ran: no agent knew more than 8 gems at once.

Details: [`research.md`](research.md) → Observed → Dungeon demo.
