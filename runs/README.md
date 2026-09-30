# Runs

Every captured run lands here as one `.jsonl` trace, in one flat folder per game:

| folder | written by |
|---|---|
| `shooter/` | `python -m demo.shooter.capture`, `python -m bench.shooter_eval`, `python -m bench.escape_check` |
| `dungeon/` | `python -m demo.dungeon.capture`, `python -m bench.dungeon_eval`, `python -m bench.escape_check` |

Names start with the time the run (or its eval) started, so sorting by name sorts by time:

```
20260929-133306_qwen2.5-3b_seed0.jsonl          one capture (model, seed)
20260929-134450_eval-ammo_1.5b_seed3.jsonl      one run of an eval (eval id, game, setup, seed)
20260930-095727_check_3b_seed1.jsonl            one run of the escape check (check id, model, seed)
```

Watch them with the Master Viewer (load one, a random one, or auto-demo through all of them):

```bash
.venv/Scripts/python -m viewer
```

Runs are git-ignored, except the pinned ones below (`*.pinned.jsonl`, see `.gitignore`). Every run
is deterministic, so a pruned run can be captured again with the same command and seed.

## Pinned runs

Shooter. The first six are the classic game (no ammo, no dash). Ticks are the last tick of the run, as the viewer shows them:

| file | why it is pinned | outcome |
|---|---|---|
| `shooter/20260927-002350_classic_qwen2.5-1.5b_seed0.pinned.jsonl` | pre-registered showcase: seed 0, whatever the outcome | died to a gunner at tick 98; key picked up, 7 kills |
| `shooter/20260927-003620_classic_qwen2.5-3b_seed0.pinned.jsonl` | pre-registered showcase (3B with order averaging, its pre-registered setting) | died to a gunner at tick 48; 4 kills |
| `shooter/20260927-004642_classic_qwen2.5-3b-listed_seed0.pinned.jsonl` | seed 0 with the 3B's current setting (options in listed order) | escaped at tick 164, 13 kills |
| `shooter/20260927-002415_classic_qwen2.5-1.5b_seed1.pinned.jsonl` | the 1.5B's first escaped seed, chosen after the results | escaped at tick 146, 12 kills |
| `shooter/20260927-005931_classic_qwen2.5-1.5b_seed12.pinned.jsonl` | best 1.5B run, chosen after the results (escaped, then kills, rooms cleared, minimum health) | escaped at tick 261, 14 kills |
| `shooter/20260927-004052_classic_qwen2.5-3b_seed5.pinned.jsonl` | best 3B run, chosen the same way | escaped at tick 287, 15 kills |

| `shooter/20260930-095727_check_3b_seed1.pinned.jsonl` | the 3B's seed 1 in the escape check (dash game): the seed that looped for 570 ticks before the 2026-09-30 fixes | escaped at tick 145, key at tick 7 |

Dungeon (the rebuilt dungeon of 2026-09-30; the first dungeon's pinned run was removed with it):

| file | why it is pinned | outcome |
|---|---|---|
| `dungeon/20260930-095727_check_1.5b_seed0.pinned.jsonl` | seed 0 of the escape check with the 1.5B, whatever the outcome | escaped at tick 189, 22 dashes, no hits |
| `dungeon/20260930-095727_check_3b_seed0.pinned.jsonl` | seed 0 of the escape check with the 3B, whatever the outcome | escaped at tick 45, no hits |

A pinned escape-check run no longer counts for `--resume` of its check (the name changed); the
check's results are in `bench/results/escape_check/`.
