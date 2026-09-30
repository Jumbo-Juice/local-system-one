# Dungeon demo

One agent without a weapon sneaks through a seeded dungeon: find the key, then leave through the
exit. Rebuilt on 2026-09-30. The first dungeon (one-cell corridors, enemies that chased anywhere,
hunger) was never escaped in 48 model runs; its code and runs are gone (they are in git history up
to commit `52bb540`), and its results stay in [`research.md`](research.md) → Observed → Dungeon demo.

- Nine rooms in a 3×3 layout. Corridors and doorways are **three cells wide**, and **every room has
  at least two doorways**, so no room is a dead end and the agent can always leave another way.
- **Ghouls** guard the rooms and cannot be fought. A ghoul sleeps until the agent steps into its
  room, then chases it inside the room (never outside), hits for 20 when next to it, backs off,
  and tires after 12 ticks of chasing. Two guard the key.
- **Dash**: instead of a step, the agent can move up to 3 cells in one tick; it recharges in 8
  ticks. The agent always outruns a ghoul; with the dash, by far.
- Gems are a bonus; potions heal 40. No hunger. Tick limit 400.

Its standing order, as the model reads it: *"Find the key, then leave through the exit alive.
Ghouls guard the rooms and cannot be fought: keep out of their reach. Gems are a bonus."*

The model makes every decision, and failed runs are shown as they happened. Code: `demo/dungeon/`
(`world.py` rules, `bots.py` non-model reference bots, `brain.py` tiers and texts, `capture.py`).
Viewer: `viewer/games/dungeon.js`.

## Run it

```bash
.venv/Scripts/python -m demo.dungeon.capture --config config/default.toml --seed 0
```

```bash
.venv/Scripts/python -m demo.dungeon.capture --config config/lenovo-3b.toml --seed 0
```

The first runs Qwen2.5-1.5B, the second Qwen2.5-3B. Each writes one trace to the dungeon pool,
`runs/dungeon/<time>_<model>_seed0.jsonl`; `--config config/mock.toml` runs without a model. Watch
it with every other run in the Master Viewer:

```bash
.venv/Scripts/python -m viewer
```

## Decisions

Each tick is one batched forward pass (two rows per decision with order averaging, on for the 1.5B):

| tier | when | options |
|---|---|---|
| strategy | when the situation changes (health, key, rooms seen, a chasing ghoul), or the agent stalls | the goals possible now: explore, get the key, go to the exit, collect gems, drink a health potion |
| target | when the goal changes, or the target is reached, gone or stalls | unexplored rooms behind known doorways, the key, the exit, gems, potions |
| move (control head) | every tick | steps and (when ready) dashes, labelled with outcomes: `dash east (3 cells; safe; closer: 4 steps to the target)` |

What the brain does for the model (implementation choices; the evidence is in
[`research.md`](research.md) → Dash and loops, and [`lessons-learned.md`](lessons-learned.md)):

- **"closer" is measured along routes that keep out of every ghoul's reach**, so the move the
  model likes best is also the safe one.
- **No trap options.** Walls, ghouls and one-cell dashes are not offered. `stay` is left out while
  a safe move gets closer. A move into a ghoul's reach is offered only when nothing is safe.
- **Targets are held** until reached, gone or stalled, not re-decided every few ticks. After 6
  ticks without getting closer the target is re-decided; after 30 ticks without a new cell or any
  progress, the strategy too.
- **With the key in hand and the exit known**, exploring and gems are no longer offered.

## How the rules were chosen

With bots only, before any model ran (`bench/dungeon_calibration.py`, dev seeds 1000–1059):

| rules | reference bot | careless bot (ignores ghouls) | random moves |
|---|---:|---:|---:|
| first draft: ghouls never tire | 28 / 60 | 50 / 60 | 0 / 60 |
| **ghouls tire after 12 ticks of chasing (frozen)** | **59 / 60** | 52 / 60 | 0 / 60 |

In the first draft a ghoul could shadow the careful bot along a doorway from inside its room
forever. Tiring ghouls give way. Not pre-registered: 12 was the first value tried that passed; 8
and 20 behave alike (`bench/results/dungeon_calibration/`).

## What the viewer shows

- The map with fog, ghouls (grey with a `z`: asleep; glowing: chasing; dim: tired or backing off),
  gems, potions, the key and the exit. A white streak is a dash.
- The HUD: health, the dash's charge bar, gems, the key, rooms seen.
- One card per tier, the latency timeline with hits, the key and dashes, and two warning strips:
  red for no progress toward the target (6+ ticks), orange for going in circles (30+ ticks without
  a new cell or any progress).

## Results

Observed on the Lenovo (Arc 140V). Raw data in `bench/results/`.

| run set | 1.5B | 3B |
|---|---:|---:|
| development, dev seeds 1000–1004 (`dungeon_eval/dungeon_eval_20260930_094943.json`) | 5 / 5 | 5 / 5 |
| escape check, seeds 0–4 (`escape_check/escape_check_20260930-095727.json`) | 5 / 5 | 5 / 5 |
| confirmation, seeds 50–54, never run before (`escape_check/escape_check_20260930-101725.json`) | 5 / 5 | 5 / 5 |

- Hits taken: 0 in every 1.5B run, one hit in one 3B run. Loop ticks: 0 everywhere.
- The dash was used 5–27 times per run.
- **These runs escape because the brain keeps the agent safe, not because the model judges danger
  well.** Moves into a ghoul's reach are offered only when nothing is safe, and "closer" is measured
  along safe routes. What the runs show is that the models plan (goal, target) and navigate
  (closer moves, dashes) well enough to finish, in 45–229 ticks (47–203 on seeds 50–54).
- For comparison, the first dungeon: 0 of 48 model runs escaped.

The 20-run check across both games is described in [`research.md`](research.md) → Dash and loops.
