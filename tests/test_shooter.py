"""The shooter demo: rules, bots, brain texts, runner and trace, replay page (mock backend only)."""

import json
import random

import pytest

from demo.shooter import bots
from demo.shooter.world import Bullet, Dungeon, Enemy, Rules


def quiet(seed=0, **rules):
    """A dungeon without enemies or potions unless asked for, so single rules can be tested."""
    base = dict(room_enemies=(0, 0), key_room_enemies=0, exit_room_enemies=0, potions=0)
    base.update(rules)
    return Dungeon(seed, Rules(**base))


def walk_to(d, cell):
    """Put the agent next to ``cell`` and return the move onto it."""
    for move, (dx, dy) in (("move east", (-1, 0)), ("move west", (1, 0)), ("move south", (0, -1)), ("move north", (0, 1))):
        here = (cell[0] + dx, cell[1] + dy)
        if d.passable(here) and d.enemy_at(here) is None:
            d.pos = here
            return move
    raise AssertionError("no open neighbour")


def place_enemy(d, kind="gunner", room=None, awake=True, hp=None):
    """Replace the enemies with one enemy in ``room`` (default: a room next to the start)."""
    room = d.rooms[room if room is not None else d.corridors[0].rooms[1]]
    e = Enemy(0, kind, room.centre, room.id, hp or (d.rules.gunner_hp if kind == "gunner" else d.rules.brute_hp),
              awake=awake, timer=99 if kind == "gunner" else 0)
    d.enemies = [e]
    return e, room


def test_layout_is_deterministic_connected_and_consistent():
    for seed in range(25):
        d, again = Dungeon(seed), Dungeon(seed)
        assert d.layout() == again.layout() and d.snapshot() == again.snapshot()
        assert set(d.room_distances(d.start_room)) == set(range(9))
        assert len(d.corridors) == 9  # spanning tree (8) + one extra link
        reach = d.distances(d.pos)
        items = [d.exit, d.key, *d.potions, *(e.pos for e in d.enemies)]
        assert all(c in reach for c in items)
        assert len(set(items + [d.pos])) == len(items) + 1
        assert all(d.inside(c) is not None for c in items)
        assert not d.living(d.start_room) and len(d.living(d.key_room)) == d.rules.key_room_enemies
        assert all(len(d.living(r)) >= 1 for r in range(9) if r != d.start_room)


def test_doorways_and_corridors_are_three_cells_wide():
    for seed in range(25):
        d = Dungeon(seed)
        for cor in d.corridors:
            for door in cor.doors:
                assert len(door) == 3
                xs, ys = {c[0] for c in door}, {c[1] for c in door}
                assert (len(xs) == 1 and len(ys) == 3) or (len(xs) == 3 and len(ys) == 1)
            assert len(cor.cells) % 3 == 0


def test_rooms_open_only_through_their_doorways():
    d = Dungeon(4)
    for room in d.rooms:
        border = set()
        for x, y in room.cells:
            for c in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if c not in room and d.passable(c):
                    border.add(c)
        assert border and border <= d.doors[room.id]


def test_exit_is_locked_without_the_key_and_opens_with_it():
    d = quiet()
    ev = d.step(walk_to(d, d.exit))
    assert d.pos == d.exit and d.outcome is None and {"kind": "exit_locked"} in ev
    d.step(walk_to(d, d.key))
    assert d.has_key and d.key is None
    ev = d.step(walk_to(d, d.exit))
    assert d.outcome == "escaped" and ev[-1] == {"kind": "escaped"}
    with pytest.raises(RuntimeError):
        d.step("stay")


def test_potion_heals_up_to_the_cap():
    d = quiet(potions=1)
    d.health = 80
    ev = d.step(walk_to(d, d.potions[0]))
    assert d.health == 100 and any(e["kind"] == "potion" and e["gain"] == 20 for e in ev)


def test_tick_limit():
    d = quiet(max_ticks=3)
    for _ in range(3):
        d.step("stay")
    assert d.outcome == "timeout"


def test_entering_a_room_with_enemies_seals_it_until_they_die():
    d = quiet()
    e, room = place_enemy(d, awake=False)
    d.pos = next(c for c in room.cells if abs(c[0] - e.pos[0]) + abs(c[1] - e.pos[1]) >= 3)
    ev = d.step("stay")
    assert e.awake and d.sealed == room.id and {"kind": "room_sealed", "room": room.id} in ev
    assert all(not d.passable(c) for c in d.doors[room.id])
    e.hp = 0
    ev = d.step("stay")
    assert d.sealed is None and room.id in d.cleared and {"kind": "room_cleared", "room": room.id} in ev
    assert all(d.passable(c) for c in d.doors[room.id])


def test_agent_bullet_flies_and_kills():
    d = quiet(cooldown=2)
    e, room = place_enemy(d, hp=2)
    d.pos = (e.pos[0] - 3, e.pos[1]) if (e.pos[0] - 3, e.pos[1]) in room else (e.pos[0] + 3, e.pos[1])
    d.seen.add(room.id)
    e.pos, frozen = e.pos, e.pos
    ev = d.step("stay", shoot=e.id)
    kinds = [x["kind"] for x in ev]
    assert "shot" in kinds and "enemy_hit" in kinds and e.hp == 1 and d.shots == 1
    assert d.cooldown == 1  # counts down at the end of the tick
    ev = d.step("stay", shoot=e.id)
    assert {"kind": "dry_fire", "enemy": e.id} in ev  # still cooling down
    e.pos = frozen
    ev = d.step("stay", shoot=e.id)
    assert not e.alive and d.kills == 1 and any(x["kind"] == "enemy_killed" for x in ev)


def test_walls_stop_bullets():
    d = quiet()
    x, y = d.pos
    wall = next((x + k, y) for k in range(1, 20) if not d.passable((x + k, y)))
    d.bullets = [Bullet(1, "agent", float(wall[0] - 1), float(y), 3.0, 0.0, 1)]
    ev = d.step("stay")
    assert not d.bullets and any(e["kind"] == "impact" for e in ev)


def test_gunner_aims_for_one_tick_then_fires_a_dodgeable_bullet():
    d = quiet()
    e, room = place_enemy(d)
    d.pos = next(c for c in room.cells if c[1] == e.pos[1] and abs(c[0] - e.pos[0]) >= 3)
    d.seen.add(room.id)
    e.timer = 1
    d.step("stay")
    assert e.aiming and not d.bullets  # the telegraph tick
    ev = d.step("stay")
    assert not e.aiming and len(d.bullets) == 1 and any(x["kind"] == "enemy_shot" for x in ev)
    path = d.bullet_path(d.bullets[0])
    assert path and d.danger() == {c: d.rules.enemy_shot_damage for c in path}
    # keep standing in the line: the bullet arrives and hurts
    for _ in range(10):
        ev = d.step("stay")
        if any(x["kind"] == "hit" for x in ev):
            break
    assert d.health == 100 - d.rules.enemy_shot_damage


def test_stepping_out_of_the_line_dodges():
    d = quiet()
    e, room = place_enemy(d)
    y = e.pos[1]
    d.pos = next(c for c in room.cells if c[1] == y and abs(c[0] - e.pos[0]) >= 3)
    side = "move north" if (d.pos[0], y - 1) in room else "move south"
    d.bullets = [Bullet(1, "enemy", float(e.pos[0]), float(y), 1.0 if d.pos[0] > e.pos[0] else -1.0, 0.0, 15)]
    e.timer = 99
    d.step(side)
    for _ in range(12):
        d.step("stay")
    assert d.health == 100


def test_brute_hits_when_adjacent_then_rests():
    d = quiet()
    e, room = place_enemy(d, kind="brute")
    d.pos = (e.pos[0] + 1, e.pos[1]) if (e.pos[0] + 1, e.pos[1]) in room else (e.pos[0] - 1, e.pos[1])
    ev = d.step("stay")
    assert d.health == 100 - d.rules.brute_damage and any(x["kind"] == "hit" and x["by"] == "brute" for x in ev)
    assert e.timer == d.rules.brute_rest
    ev = d.step("stay")
    assert not any(x["kind"] == "hit" for x in ev)


def test_death_records_the_cause():
    d = quiet()
    e, room = place_enemy(d, kind="brute")
    d.health = 10
    d.pos = (e.pos[0] + 1, e.pos[1]) if (e.pos[0] + 1, e.pos[1]) in room else (e.pos[0] - 1, e.pos[1])
    d.step("stay")
    assert d.outcome == "died" and d.cause == "brute" and d.health == 0


def test_fog_reveals_rooms_through_their_doorways():
    d = quiet(3)
    known = d.known_cells()
    assert set(d.rooms[d.start_room].cells) <= known
    unseen = [r for r in d.rooms if r.id not in d.seen]
    assert all(not (set(r.cells) & known) for r in unseen)


def test_reference_bot_wins_and_random_does_not_on_dev_seeds():
    """The calibration claim in docs/research.md, on a few of the dev seeds (1000+)."""
    ref = [bots.play(Dungeon(s), bots.reference).outcome for s in range(1000, 1006)]
    rnd = [bots.play(Dungeon(s), bots.random_policy(random.Random(s))).outcome for s in range(1000, 1006)]
    assert ref.count("escaped") == 6 and rnd.count("escaped") == 0


def test_snapshot_is_json():
    d = Dungeon(1)
    for _ in range(30):
        d.step(*bots.reference(d))
    json.dumps(d.layout())
    json.dumps(d.snapshot())
