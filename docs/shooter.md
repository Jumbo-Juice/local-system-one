# Shooter demo

The first dungeon ([`dungeon.md`](dungeon.md)) was never escaped. This is its redo as a
room-clearing shooter in the spirit of *Enter the Gungeon*:

- The agent shoots. Rooms lock it in until their enemies are dead.
- Doorways are three cells wide, so no enemy can trap it in one.
- One agent per run, in a seeded dungeon of nine rooms with a key and an exit.
- Gunners aim for a tick, then fire slow bullets. Brutes walk up and hit.

Its standing order, as the model reads it: *"Find the key, then leave through the exit alive.
Rooms lock you in until their enemies are dead: shoot them and keep out of their bullets."*

The model makes every decision, and failed runs are shown as they happened. Code:
`demo/shooter/` (`world.py` rules, `bots.py` non-model reference bots, `brain.py` tiers and texts,
`capture.py`). Viewer: `viewer/games/shooter.js`.

## Run it

One run with each model:

```bash
.venv/Scripts/python -m demo.shooter.capture --config config/default.toml --seed 0
```

```bash
.venv/Scripts/python -m demo.shooter.capture --config config/lenovo-3b.toml --seed 0
```

The first runs Qwen2.5-1.5B, the second Qwen2.5-3B. Each writes one trace to the shooter pool,
`runs/shooter/<time>_<model>_seed0.jsonl`. `--config config/mock.toml` runs without a model.
`--classic` plays the game without ammo, the version evaluated on seeds 0–39.

Watch it, with every other run, in the Master Viewer. It can load a trace, pick a random run, or
auto-demo the pool:

```bash
.venv/Scripts/python -m viewer
```

The pinned runs are committed in `runs/shooter/` and listed in [`runs/README.md`](../runs/README.md):

- **Pre-registered showcase** (seed 0, whatever the outcome):
  - the 1.5B picks up the key and dies to a gunner at tick 98;
  - the 3B, with order averaging (its pre-registered setting), dies to a gunner at tick 48.
- **Chosen after the results:**
  - the 3B's current setting on seed 0 (escapes);
  - the 1.5B's first escaped seed;
  - the best run per model (escaped, then kills, rooms cleared, minimum health): 1.5B seed 12 and
    3B seed 5.

`python -m viewer --bundle <run>` turns any of them into one self-contained page to share.

## Decisions

Each tick is one batched forward pass with up to three decisions (strategy or target, move, shoot):

| tier | every | options |
|---|---|---|
| strategy | 12 ticks, or at once when the situation changes | the goals possible now: explore, fight the enemies here, get the key, go to the exit, drink a health potion (and pick up ammo, with ammo) |
| target | 6 ticks, or when reached, gone or unsafe | unexplored rooms, the key, the exit, potions, or up to 5 firing spots (a clear line to an enemy, away from enemies, out of every bullet's path) |
| move (control head) | every tick | open moves and stay, labelled with outcomes: `move west (safe; closer: 4 steps to the target)`, `stay (BULLET: -15 health)` |
| shoot (control head) | every tick | `shoot gunner #4, 3 cells east (AIMING at you; 3 hits to kill; clear line)` or `hold fire`; committed without a model call while the gun reloads |

**Order averaging (implementation choice, per model).** Small models often pick an option for its
position. On a development seed the 1.5B's decision to fire followed the option order in all 34
recorded states: with hold fire listed last it never fired; listed first, it always fired.

With order averaging (`Engine(order_debias=True)`), every decision is read twice in the same
batch, once with the options as listed and once reversed, and the two readings are averaged. This
doubles the rows per pass; the viewer shows both readings as ticks on each probability bar. It is
on for the 1.5B and, after the evaluation below, off for the 3B (`[shooter] order_debias` in each
config; `--order-debias` overrides).

**Ammo (on by default).** The rules:

- The gun holds 6 bullets. A reload takes 3 ticks, and the gun cannot fire meanwhile.
- The agent starts with 36 bullets (30 in reserve, the cap). 5 ammo boxes of 10 lie in rooms.
- Runs are longer, so the tick limit is 600.

What the model sees:

- The shoot head offers `reload` while the magazine is not full. An empty gun reloads without a
  model call.
- The strategy tier offers `pick up ammo`.
- The state texts count bullets against the hits the known enemies still take.

The rules were chosen with bots only, on dev seeds 1000–1059, by a rule written before the runs
(`bench/shooter_calibration.py --ammo`). The reference bot escapes 60/60 and never runs dry. A bot
that ignores ammo (reloads only an empty gun, never walks to a box) escapes 31/60. `--classic`
(capture) and `--game classic` (`bench/shooter_eval.py`) reproduce the game without ammo exactly.

**Fire head for the 1.5B (post hoc, checked on new seeds).** In the game with ammo, the 1.5B held
fire in all 53 shoot decisions of seed 0: each enemy in sight was its own option, and `hold fire`
won every time. With `fire_head`:

- The shoot head offers a single `shoot (N enemies in sight, clear line)` option.
- An aim head in the same batch picks the enemy (no model call with one enemy in sight).
- It is on for the 1.5B (`[shooter] fire_head`), off for the 3B, and always off in the classic
  game.

## What the viewer shows

The stage has more contrast than the dungeon's: lighter floors, darker walls, bright enemies and
bullets. It shows:

- the map with fog, sealed doorways (red bars), aim telegraphs (red dashed lines), bullets in
  flight and enemy health pips;
- one card per tier and head, and the latency timeline with hits and kills;
- below the stage, a prompt inspector;
- in auto-demo, each pane shows the map, the HUD and one chip per tier.

## Results

Closed-loop evaluation (`bench/shooter_eval.py`). Seeds, setups, metrics and showcase were fixed
and committed before the first run on those seeds:

```bash
.venv/Scripts/python -m bench.shooter_eval
```

Classic game (no ammo), Lenovo (Arc 140V), seeds 0–9 (Observed;
`bench/results/shooter_eval/shooter_eval_20260927_002315.json`):

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
- **Order averaging is what makes the 1.5B play.** Without it, the 1.5B held fire in 98% of its
  shoot decisions and never escaped.
- **The 3B did worse with averaging than without** (2 vs 7 of 10, p = 0.07).
  - A post-hoc replication on new seeds 10–19, with its decision rule committed before the run,
    gave 6 vs 7. By that rule the 3B demo now uses the listed order.
  - Over all 20 seeds: 3B listed 14/20, 3B averaged 8/20 (p = 0.11), 1.5B averaged 14/20. Both
    demos escape about 7 runs in 10.
- **Nearly every hit was a chosen risk.** 29 of 30 (1.5B) and 66 of 67 (3B) hits came from a move
  labelled `BULLET` or `next to a brute` while a safe move was offered. The 3B often steps toward
  the gunner it is fighting.

**Pre-registered check on seeds 30–39.** This tested the fixes of commit `21f6358`, run at that
commit on an RTX 3060 Ti with CUDA, so on different hardware than the runs above
(`bench/results/shooter_eval/shooter_eval_20260928_123222.json`):

- 1.5B 5/10 escaped, 3B listed order 5/10, reference bot 10/10, never-dodging bot 3/10.
- No hit came after a move labelled safe (the fixed defect).
- Both models timed out on seed 37. Not fixed yet:
  - the 1.5B because a sleeping brute next to the key left every move labelled "no safe route to
    the target";
  - the 3B by staying put on an explore target with the key in hand.

**Fire head, pre-registered check** (Observed; seeds 40–49, game with ammo, Lenovo;
`bench/results/shooter_eval/shooter_eval_ammo_20260929_134450.json`):

| setup (seeds 40–49, with ammo) | escaped | died | out of time | kills | held fire | hits taken | forward ms, median / p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **1.5B with the fire head (the 1.5B demo)** | **6 / 10** | 3 | 1 | 8.8 | 0% | 4.1 | 324 / 812 |
| 1.5B, one shoot option per enemy | 0 / 10 | 7 | 3 | 0.6 | 98% | 5.4 | 476 / 931 |
| reference bot (not a model) | 10 / 10 | 0 | 0 | 11.8 | – | 0.2 | – |

6 vs 0 of 10 (two-sided Fisher p = 0.011), so by the rule fixed before the run the fire head stays
on. The 3B with ammo is not evaluated yet.

Details, development probes and the rule calibration: [`research.md`](research.md) → Observed →
Shooter demo. What the runs taught us: [`lessons-learned.md`](lessons-learned.md).
