# Lessons learned from the test and eval runs

Distilled from every test and eval run so far (most of their traces were pruned on 2026-09-29).
Every run is deterministic, so any pruned run can be captured again with the command in
[`RUN-GUIDE.md`](../RUN-GUIDE.md). The numbers are in `bench/results/` and in
[`research.md`](research.md), the full write-up. Labels: **Observed** = measured in those runs,
**Inferred** = our reading of them, not tested.

Runs live in the two pools, `runs/shooter/` and `runs/dungeon/` (see [`runs/README.md`](../runs/README.md),
which also lists the pinned runs). Watch them with `python -m viewer`.

## How to run experiments here

- **Pre-register, then run.** Writing the seeds, setups and decision rule into the eval script and
  committing it before the run kept every post-hoc change honest. Three times, a change that looked
  good on development seeds was checked on seeds nothing had played (10-19, 30-39, 40-49).
- **Keep track of which seeds anything has touched, including bots.** Bot runs on seeds 20-29
  while developing a fix meant the model check had to move to seeds 30-39.
- **Ten seeds decide little.** 2 vs 7 escapes (p = 0.07) became 6 vs 7 on a replication. Only big
  gaps held up: 0 vs 7 (p = 0.003) and 0 vs 6 (p = 0.011).
- **Read the traces, not only the totals.** Every real bug below came from classifying single
  decisions: which move was chosen when a hit happened, and what the options said at that moment.
- **Averages hide traps.** A 97-100% single-decision accuracy coexisted with one state that
  repeated for 200+ ticks, because nothing changes in an empty corridor.
- **A deterministic mock is not a random floor.** The mock backend picks a fixed function of the
  prompt, so in a static state it repeats forever and never leaves the start room.
- **Don't render or run other GPU work during a latency run** (shared iGPU power budget).

## What the small models do (Qwen2.5-1.5B and 3B)

- **They follow option position (Observed).** The 1.5B's decision to fire followed the option
  order in 34 of 34 states. Reading every decision in both orders and averaging
  (`order_debias`) made the 1.5B play (0 → 7 of 10 escapes); for the 3B it did not help (8 vs 14
  of 20), so the 3B reads options in the listed order.
- **They follow the first half of a question when two halves conflict (Observed).** "Get closer
  to the target, but never step into a bullet's path": nearly every hit was a move labelled
  `BULLET` or `next to a brute`, chosen while a safe move was offered (29 of 30 hits for the
  1.5B, 66 of 67 for the 3B). In the first dungeon the move tier took the move toward the target
  97-98% of the time, into an enemy when that was the only way (13 of 13).
- **Spelling out consequences did not change that (Observed).** `ENEMY: you stay here and lose 30
  health` did no better than `ENEMY: -30 health` on new seeds. Safety has to come from the
  planning tiers choosing safe targets (Inferred).
- **Wording matters more than structure (Observed).** For the 1.5B's shot: one option per enemy,
  held fire 108/189; a yes/no "Do you shoot this tick?", 161/189 (worse); the enemies merged into
  one `shoot (N enemies in sight, clear line)` option, 0/189. The merged option (the fire head)
  took the 1.5B from 0 to 6 escapes of 10 in the game with ammo.
- **With several similar options the probability spreads across them (Observed)**, and a single
  "do nothing" option can win without being the model's preferred kind of action.
- **Don't offer trap options.** Offering walls as moves let the 3B choose `move west (wall)` at
  p = 0.71 for 200+ ticks; a compass bearing to a target behind a bend did the same. Only offering
  open moves and giving the route's waypoint fixed it.
- **A small wording change can decide whether a model moves at all.** The 3B chose `stay (safe;
  no closer)` in 235 of 362 states with one move question and the safe closer move 128 of 130
  times with another.

## Bugs the traces found (all fixed)

- **Labels that lie cost more than any prompt tweak.** A risky stay read "reach the target"; a
  move next to a sleeping brute read `safe` although entering the room woke it (all 4 hits after a
  "safe" move). Metric added to catch this: `hits_after_safe_move` (0 since the fix).
- **A goal can be missing from the menu.** Brutes guarding a doorway from inside their room left
  the 1.5B in the corridor for 389 ticks: fights were offered only inside the room. Fix: fight a
  seen room from its doorways and corridors, decided by position (line of sight made the bot
  flip-flop between two cells).
- **Open:** that sleeping-brute fix left the 1.5B with "no safe route" on seed 37 (key next to a
  sleeping brute, timeout).

## Game design

- **Prove the level beatable with bots before any model runs.** The first dungeon was never
  escaped (0 of 48 model runs); the shooter was tuned with a reference bot first (60/60 on dev
  seeds) and the models then escaped about 7 in 10.
- **One-cell doorways trap agents**; three-cell doorways and sealed rooms fixed that.
- **An agent that moves with purpose meets more enemies.** Fixing loops in the first dungeon cut
  stuck ticks from 63 to 6 but raised enemy deaths from 3 to 7.

## Engineering

- **Batching helped far less on this iGPU than the sources suggest**; a forward pass has a floor
  (3B, batch 1: 83 ms at 8 tokens).
- **Prompt order decides how much of the prompt cache is reused**, and it changes accuracy.
- **Tournaments fix long option lists** (100% at 40 options with groups), but no evaluated run
  ever needed one.
