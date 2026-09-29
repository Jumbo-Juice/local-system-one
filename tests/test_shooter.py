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


# -- engine and goal-stack additions ----------------------------------------------------------

from system_one import Decision, Engine  # noqa: E402
from system_one.backends.mock import MockBackend  # noqa: E402
from system_one.goals import GoalStack, Tier  # noqa: E402


def test_order_debias_averages_the_two_orders_in_one_pass():
    be = MockBackend()
    eng = Engine(be)
    ds = [Decision("q?", ("a", "b", "c"), state="s1"), Decision("q?", ("only",), state="s2")]
    fwd = eng.decide_batch([ds[0]])[0]
    rev = eng.decide_batch([Decision("q?", ("c", "b", "a"), state="s1")])[0]
    eng.order_debias = True
    before = be.forward_calls
    out = eng.decide_batch(ds)
    assert be.forward_calls - before == 1 and eng.last_stats["decisions"] == 3  # one mirror row
    avg = [(a + b) / 2 for a, b in zip(fwd.probs, rev.probs[::-1])]
    assert out[0].probs == pytest.approx(avg) and out[0].index == avg.index(max(avg))
    assert out[0].orders == [pytest.approx(fwd.probs), pytest.approx(rev.probs[::-1])]
    assert out[0].decision is ds[0] and out[1].orders is None


def test_context_from_limits_what_a_tier_sees():
    stack = GoalStack([Tier("plan", "p?", ("x", "y"), every=5, title="Plan"),
                       Tier("move", "m?", ("l", "r")), Tier("shoot", "s?", ("f", "h"), context_from=("plan",)),
                       Tier("other", "o?", ("a", "b"))])
    for t, c in zip(stack.tiers, ("x", "l", "f", "a")):
        stack.apply(t, c, 0)
    assert stack.context(stack.tier("shoot")) == "Plan: x"
    assert stack.context(stack.tier("other")).splitlines() == ["Plan: x", "move: l", "shoot: f"]


# -- brain, runner, trace ---------------------------------------------------------------------

from demo.shooter.brain import GOALS, HOLD, Runner, ShooterBrain, cell_of, enemy_of  # noqa: E402
from demo.shooter.capture import VIEWER, build_replay, to_jsonl  # noqa: E402


def fight(kind="gunner"):
    """The agent inside a room with one awake enemy three cells away on the same row."""
    d = quiet()
    e, room = place_enemy(d, kind=kind)
    d.pos = next(c for c in room.cells if c[1] == e.pos[1] and abs(c[0] - e.pos[0]) == 3)
    d.seen.add(room.id)
    d._known = None
    d.step("stay")  # seals the room
    return d, e, room


def test_goals_offered_only_when_possible():
    d = quiet()
    b = ShooterBrain(d)
    assert b.goal_options() == ["explore"]
    d.seen = set(range(9))
    d._known = None
    b.refresh()
    assert "get the key" in b.goal_options() and "explore" not in b.goal_options()
    d, e, room = fight()
    b = ShooterBrain(d)
    assert d.sealed == room.id and b.goal_options()[0] == "fight the enemies here"
    assert "explore" not in b.goal_options()  # the doorways are sealed
    assert all(g in GOALS for g in b.goal_options())
    assert "sealed until its 1 enemy is dead" in b.strategy_state()


def test_firing_spots_have_a_clear_line_and_keep_distance():
    d, e, room = fight()
    b = ShooterBrain(d)
    spots = b.target_options("fight the enemies here")
    assert spots and all(s.startswith("firing spot at (") for s in spots)
    for s in spots:
        c = cell_of(s)
        assert c in room and d.clear_line(c, e.pos) and ((c[0] - e.pos[0]) ** 2 + (c[1] - e.pos[1]) ** 2) >= 9
    assert any("where you stand" in s for s in spots)  # the agent is 3 cells from the enemy


def test_shoot_options_hold_fire_last_or_only_option():
    d, e, room = fight()
    b = ShooterBrain(d)
    d.cooldown = 0
    b.refresh()
    opts = b.shoot_options()
    assert opts[-1] == HOLD and opts[0].startswith(f"shoot gunner #{e.id}, ") and " cells " in opts[0]
    assert "3 hits to kill; clear line" in opts[0] and enemy_of(opts[0]) == e.id and enemy_of(HOLD) is None
    d.cooldown = 1
    assert b.shoot_options() == ["hold fire (next shot ready in 1 tick)"]
    d.rules = CLASSIC  # the evaluated wording without ammo
    assert b.shoot_options() == ["hold fire (the gun is reloading: ready in 1 tick)"]


def test_moves_are_labelled_and_walls_are_not_offered():
    d, e, room = fight()
    b = ShooterBrain(d)
    x, y = d.pos
    d.bullets = [__import__("demo.shooter.world", fromlist=["Bullet"]).Bullet(9, "enemy", float(x), float(y - 2), 0.0, 1.5, 15)]
    b.refresh()
    opts = b.move_options()
    names = [o.split(" (")[0] for o in opts]
    assert names[-1] == "stay" and all(d.passable((x + dx, y + dy)) for dx, dy in
                                        ((0, -1) if n == "move north" else (0, 1) if n == "move south" else
                                         (1, 0) if n == "move east" else (-1, 0) if n == "move west" else (0, 0)
                                         for n in names))
    stay = opts[-1]
    assert "BULLET: -15 health" in stay and "safe" not in stay
    assert any(o.startswith("move") and "(safe" in o for o in opts)


def test_brute_neighbourhood_is_labelled():
    d, e, room = fight("brute")
    d.pos = next(c for c in room.cells if c[1] == e.pos[1] and abs(c[0] - e.pos[0]) == 2)
    b = ShooterBrain(d)
    e.timer = 0
    b.refresh()
    toward = "move east" if e.pos[0] > d.pos[0] else "move west"
    opt = next(o for o in b.move_options() if o.startswith(toward))
    assert "next to a brute: -20 health" in opt


def doorway(d):
    """(room, its middle doorway cell m, the step d into the room) of the first corridor."""
    cor = d.corridors[0]
    room, m = d.rooms[cor.rooms[1]], cor.doors[1][1]
    step = next((dx, dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)) if (m[0] + dx, m[1] + dy) in room)
    return room, m, step


def test_sleeping_brute_counts_when_the_move_enters_its_room():
    """Stepping in wakes the room before enemies act, so the brute hits at once (the four hits taken
    after a move labelled safe in the evaluation)."""
    d = quiet()
    room, m, (dx, dy) = doorway(d)
    e, _ = place_enemy(d, kind="brute", room=room.id, awake=False)
    e.pos = (m[0] + 2 * dx, m[1] + 2 * dy)  # two cells in from the doorway
    d.pos = m
    d.seen.add(room.id)
    d._known = None
    b = ShooterBrain(d)
    move = next(n for n, (sx, sy) in (("move east", (1, 0)), ("move west", (-1, 0)), ("move south", (0, 1)),
                                       ("move north", (0, -1))) if (sx, sy) == (dx, dy))
    opt = next(o for o in b.move_options() if o.startswith(move))
    assert "next to a brute: -20 health" in opt and "safe" not in opt
    ev = d.step(move)
    assert any(x["kind"] == "hit" and x["by"] == "brute" for x in ev)  # the label was right
    d2 = quiet()
    room2, m2, _ = doorway(d2)
    e2, _ = place_enemy(d2, kind="brute", room=room2.id, awake=False)
    e2.pos = (m2[0] + 2 * dx, m2[1] + 2 * dy)
    assert m2 not in bots.threat_cells(d2)  # from the doorway a sleeping brute cannot reach


def test_stay_on_the_target_in_a_bullet_path_reads_on_the_target():
    d, e, room = fight()
    b = ShooterBrain(d)
    x, y = d.pos
    d.bullets = [Bullet(9, "enemy", float(x), float(y - 2), 0.0, 1.5, 15)]
    b.refresh()
    b.stack.apply(b.stack.tier("strategy"), "fight the enemies here", 0)
    b.stack.apply(b.stack.tier("target"), f"firing spot at ({x},{y}), where you stand: clear shot at 1 of 1 enemy", 0)
    stay = b.move_options()[-1]
    assert stay == "stay (BULLET: -15 health; on the target)"


def test_brutes_guarding_a_doorway_can_be_fought_from_outside():
    """Found after the evaluation (1.5B, seed 3): awake brutes just inside a doorway, the agent in
    the corridor, and no fight on offer for 389 ticks."""
    d = quiet()
    room, m, (dx, dy) = doorway(d)
    e, _ = place_enemy(d, kind="brute", room=room.id, awake=True)
    e.pos = (m[0] + 2 * dx, m[1] + 2 * dy)
    d.pos = (m[0] - dx, m[1] - dy)  # just outside the doorway, three cells from the brute
    assert d.inside(d.pos) is None or d.inside(d.pos).id != room.id
    assert bots.fighting(d) == []  # a room the agent has not seen
    d.seen.add(room.id)
    d._known = None
    assert bots.fighting(d) == [e] and d.pos in bots.fight_cells(d, room.id)
    b = ShooterBrain(d)
    assert "fight the enemies here" in b.goal_options()
    spots = b.target_options("fight the enemies here")
    assert any("where you stand" in s for s in spots)
    assert f"Enemies in the {room.name}: brute #{e.id}" in b.move_state()


def test_waypoint_follows_the_route_not_the_bearing():
    d = quiet()
    b = ShooterBrain(d)
    b.stack.apply(b.stack.tier("strategy"), "explore", 0)
    b.stack.apply(b.stack.tier("target"), b.target_options("explore")[0], 0)
    b.refresh()
    way = b.waypoint()
    assert way is not None and d.clear_line(d.pos, way[0])
    s = b.move_state()
    assert "Your target is" in s and "North is up" in s


def test_runner_one_forward_pass_per_tick_and_json_records():
    be = MockBackend()
    r = Runner(Dungeon(2), Engine(be), group_size=8, plan_budget=1)
    header = r.header()
    for _ in range(80):
        before = be.forward_calls
        rec = r.tick()
        assert be.forward_calls - before <= 1
        model = [x for x in rec["decisions"] if x["method"] != "only_option"]
        assert rec["batch"]["decisions"] == len(model)
        assert sum(1 for x in model if x["kind"] == "plan") <= 1
        assert sum(1 for x in rec["decisions"] if x["tier"] == "move") == 1
        assert sum(1 for x in rec["decisions"] if x["tier"] == "shoot") == 1
        for x in model:
            assert abs(sum(x["probs"]) - 1) < 1e-3 and len(x["orders"]) == 2
        if r.d.outcome:
            break
    assert not r.engine.order_debias  # restored after each tick
    end = r.end()
    lines = [json.loads(x) for x in to_jsonl([header, *r.records, end]).splitlines()]
    assert lines[0]["scenario"] == "shooter" and lines[0]["order_debias"] is True and lines[-1]["type"] == "end"
    assert lines[0]["map"]["width"] == 33 and end["summary"]["ticks"] == len(r.records)


def test_shots_chosen_by_the_head_are_fired():
    d, e, room = fight()
    r = Runner(d, Engine(MockBackend()))
    for _ in range(30):
        rec = r.tick()
        if rec["shoot"] is not None:
            assert any(x["kind"] == "shot" and x["enemy"] == rec["shoot"] for x in rec["events"])
            break
    else:
        pytest.skip("the mock never chose to shoot in 30 ticks")


def two_gunners():
    """fight() with a second awake gunner in the room, also in the agent's line of fire."""
    d, e, room = fight()
    c = next(c for c in sorted(room.cells) if c not in (e.pos, d.pos) and d.clear_line(d.pos, c)
             and ((c[0] - d.pos[0]) ** 2 + (c[1] - d.pos[1]) ** 2) >= 4)
    d.enemies.append(Enemy(1, "gunner", c, room.id, d.rules.gunner_hp, awake=True, timer=99))
    d.cooldown = 0
    return d, room


def test_fire_head_offers_one_shoot_option_and_an_aim_head():
    d, room = two_gunners()
    one = ShooterBrain(d)
    assert [o for o in one.shoot_options() if o.startswith("shoot")][0].startswith("shoot gunner #")
    assert "aim" not in [t.name for t in one.stack.tiers]
    b = ShooterBrain(d, fire_head=True)
    opts = b.shoot_options()
    assert opts[0] == "shoot (2 enemies in sight, clear line)" and opts[-1] == HOLD and enemy_of(opts[0]) is None
    aims = b.aim_options()
    assert len(aims) == 2 and all(a.startswith("shoot gunner #") for a in aims)
    assert {enemy_of(a) for a in aims} == {0, 1}
    d.cooldown = 1
    b.refresh()
    assert b.aim_options() == []  # no shot on offer, nothing to aim


def test_fire_head_fires_at_the_enemy_the_aim_head_chose():
    d, room = two_gunners()
    r = Runner(d, Engine(MockBackend()), fire_head=True)
    assert r.header()["fire_head"] is True
    fired = held = 0
    for _ in range(40):
        rec = r.tick()
        shoot = next((x for x in rec["decisions"] if x["tier"] == "shoot"), None)
        aim = next((x for x in rec["decisions"] if x["tier"] == "aim"), None)
        chose_fire = shoot is not None and shoot["options"][shoot["choice"]].startswith("shoot (")
        if chose_fire:
            assert aim is not None and rec["shoot"] == enemy_of(aim["options"][aim["choice"]])
            assert any(x["kind"] == "shot" and x["enemy"] == rec["shoot"] for x in rec["events"])
            fired += 1
        else:
            assert rec["shoot"] is None
            held += 1
        if d.outcome is not None or not d.living(room.id):
            break
    if not fired:
        pytest.skip("the mock never chose to shoot")


def test_fire_head_default_comes_from_the_config_and_is_off_in_classic():
    from demo.shooter.capture import fire_head
    on = {"shooter": {"fire_head": True}}
    assert fire_head({}) is False and fire_head(on) is True
    assert fire_head(on, classic=True) is False  # the classic game was evaluated without it
    assert fire_head(on, True, classic=True) is True and fire_head(on, False) is False  # the flag wins


def test_viewer_has_a_trace_slot(tmp_path):
    text = to_jsonl([{"type": "header", "note": "</script>"}])
    out = build_replay(text, tmp_path / "replay.html", VIEWER)
    assert out.read_text(encoding="utf-8").count("application/x-ndjson") == 1


def test_order_averaging_default_comes_from_the_config():
    from demo.shooter.capture import order_debias
    assert order_debias({}) is True and order_debias({"shooter": {"order_debias": False}}) is False
    assert order_debias({"shooter": {"order_debias": False}}, True) is True  # the flag wins


# -- ammo --------------------------------------------------------------------------------------

from demo.shooter.brain import RELOAD, is_reload  # noqa: E402
from demo.shooter.world import CLASSIC  # noqa: E402


def test_ammo_boxes_leave_the_rest_of_the_dungeon_unchanged():
    for seed in range(12):
        d, c = Dungeon(seed), Dungeon(seed, CLASSIC)
        for k in ("rooms", "corridors", "exit", "start_room", "key_room", "exit_room"):
            assert d.layout()[k] == c.layout()[k]
        assert [(e.kind, e.pos, e.hp, e.timer) for e in d.enemies] == [(e.kind, e.pos, e.hp, e.timer) for e in c.enemies]
        assert d.potions == c.potions and d.key == c.key and d.pos == c.pos and c.ammo == []
        rooms = [d.area[b] for b in d.ammo]
        assert len(d.ammo) == d.rules.ammo_boxes and len(set(rooms)) == len(rooms) and d.start_room not in rooms
        assert all(d.inside(b) is not None for b in d.ammo)
        assert not set(d.ammo) & ({d.pos, d.exit, d.key, *d.potions} | {e.pos for e in d.enemies})
        assert "ammo" not in c.snapshot()["items"] and "loaded" not in c.snapshot()["agent"]


def shooting_range(**rules):
    """The agent three cells from one awake gunner that never fires, with a clear line."""
    d = quiet(cooldown=1, **rules)
    e, room = place_enemy(d, hp=99)  # timer 99: it never aims (the tests reset it each tick)
    d.pos = next(c for c in room.cells if c[1] == e.pos[1] and abs(c[0] - e.pos[0]) == 3)
    d.seen.add(room.id)
    d._known = None
    return d, e


def test_each_shot_uses_a_bullet_and_an_empty_gun_cannot_fire():
    d, e = shooting_range()
    for k in range(d.rules.magazine):
        e.timer = 99
        ev = d.step("stay", shoot=e.id)
        assert any(x["kind"] == "shot" for x in ev) and d.loaded == d.rules.magazine - 1 - k
    assert not d.can_fire() and d.reserve == d.rules.reserve
    ev = d.step("stay", shoot=e.id)
    assert {"kind": "dry_fire", "enemy": e.id} in ev and d.shots == d.rules.magazine


def test_a_reload_takes_its_ticks_then_refills_from_the_reserve():
    d, e = shooting_range(reserve=4)
    d.loaded = 1
    ev = d.step("stay", reload=True)
    assert any(x["kind"] == "reload" for x in ev) and d.reloading == d.rules.reload_ticks - 1
    for _ in range(d.rules.reload_ticks - 1):
        assert not d.can_fire()
        e.timer = 99
        ev = d.step("stay", shoot=e.id)
        assert {"kind": "dry_fire", "enemy": e.id} in ev  # no shooting while reloading
    assert any(x["kind"] == "reloaded" for x in ev)
    assert (d.loaded, d.reserve) == (5, 0)  # the loaded bullet is kept; the reserve ran short
    assert not d.can_reload()  # nothing left to load
    ev = d.step("stay", reload=True)
    assert {"kind": "reload_refused"} in ev


def test_ammo_boxes_fill_the_reserve_up_to_its_cap():
    d = quiet(ammo_boxes=1)
    box = d.ammo[0]
    d.reserve = d.rules.max_reserve  # full: the box stays where it lies
    d.step(walk_to(d, box))
    assert d.pos == box and d.ammo == [box]
    d.reserve = d.rules.max_reserve - 4
    ev = d.step("stay")  # picked up from under the agent once there is room
    assert d.ammo == [] and d.reserve == d.rules.max_reserve and {"kind": "ammo", "cell": list(box), "gain": 4} in ev


def test_classic_rules_have_unlimited_bullets():
    d, e = shooting_range(ammo=False)
    for _ in range(20):
        e.timer = 99
        d.step("stay", shoot=e.id)
    assert d.shots == 20 and d.loaded is None and d.can_fire() and not d.can_reload()


def test_shoot_head_offers_reload_and_commits_it_when_empty():
    d, e, room = fight()
    b = ShooterBrain(d)
    assert b.stack.tier("shoot").instruction == "What do you do with your gun this tick?"
    d.cooldown, d.loaded = 0, 3
    b.refresh()
    opts = b.shoot_options()
    assert opts[0].startswith("shoot gunner") and is_reload(opts[-2]) and opts[-1] == HOLD
    assert opts[-2] == f"{RELOAD} (3 of 6 bullets loaded; {d.reserve} in reserve; takes 3 ticks, no shooting meanwhile)"
    d.loaded = 0
    assert [o.split(" (")[0] for o in b.shoot_options()] == [RELOAD]  # the only option: no model call
    d.reserve = 0
    assert b.shoot_options() == ["hold fire (out of bullets)"]
    d.loaded, d.reserve, d.reloading = 2, 10, 2
    assert b.shoot_options() == ["hold fire (reloading: ready in 2 ticks)"]
    d.reloading, d.cooldown = 0, 1
    assert b.shoot_options() == ["hold fire (next shot ready in 1 tick)"]
    d.cooldown = 0
    e.hp = 0  # no enemy in sight: reloading is a choice
    b.refresh()
    assert [o.split(" (")[0] for o in b.shoot_options()] == [RELOAD, HOLD]
    d.loaded = d.rules.magazine
    assert b.shoot_options() == ["hold fire (no enemy in sight)"]


def test_ammo_goal_targets_labels_and_state():
    d = quiet(ammo_boxes=5)
    d.seen = set(range(9))
    d._known = None
    b = ShooterBrain(d)
    d.reserve = 12
    b.refresh()
    assert "pick up ammo" in b.goal_options()
    targets = b.target_options("pick up ammo")
    assert len(targets) == 5 and all(t.startswith("ammo box at (") for t in targets)
    s = b.strategy_state()
    assert "Ammo: fine (18 bullets: 6 loaded, 12 in reserve, at most 30 in reserve); no living enemy you know of." in s
    assert "Ammo boxes: 5 known, nearest" in s
    d.reserve = d.rules.max_reserve
    b.refresh()
    assert "pick up ammo" not in b.goal_options() and "your reserve is full" in b.strategy_state()
    d.reserve = 25
    move = walk_to(d, d.ammo[0])
    b.refresh()
    assert "ammo box: +5 bullets" in next(o for o in b.move_options() if o.startswith(move))
    c = ShooterBrain(quiet(ammo=False))
    assert "Ammo" not in c.strategy_state() and "pick up ammo" not in c.goal_options()
    assert c.stack.tier("shoot").instruction == "Which shot do you take this tick?"


def test_ammo_word_counts_bullets_against_the_hits_needed():
    d, e, room = fight()  # one gunner with 3 health
    b = ShooterBrain(d)
    d.loaded, d.reserve = 2, 0
    b.refresh()
    assert b.ammo_word() == "LOW" and "the 1 living enemy you know of takes 3 hits" in b.ammo_status()
    d.loaded = 0
    assert b.ammo_status().startswith("Ammo: OUT")
    before = b._situation_now()
    d.loaded, d.reserve = 6, 30
    assert b._situation_now() != before  # an ammo change re-plans the strategy


def test_runner_passes_a_reload_to_the_world():
    d, e, room = fight()
    d.loaded = 0
    r = Runner(d, Engine(MockBackend()))
    rec = r.tick()
    shoot = next(x for x in rec["decisions"] if x["tier"] == "shoot")
    assert shoot["method"] == "only_option" and is_reload(shoot["options"][0])
    assert rec["reload"] is True and any(x["kind"] == "reload" for x in rec["events"])
    for _ in range(3):
        r.tick()
    s = r.end()["summary"]
    assert s["reloads"] == 1 and s["reloads_chosen"] == 0 and "ticks_out_of_ammo" in s


def test_reference_bot_manages_ammo_on_dev_seeds():
    """The calibration claim (docs/research.md): with ammo the reference bot escapes and never runs dry."""
    for s in range(1000, 1004):
        d, dry = Dungeon(s), 0
        while d.outcome is None:
            d.step(*bots.reference(d))
            dry += d.ammo_total() == 0
        assert d.outcome == "escaped" and dry == 0 and d.reloads > 0
