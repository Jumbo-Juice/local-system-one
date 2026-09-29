"""Turns the shooter into text decisions for one agent, and runs it tick by tick.

Tiers (tiered goals, system_one/goals.py), all due decisions in ONE batched forward pass per tick:
  strategy  every 12 ticks, or at once when the agent's situation changes: a goal from GOALS.
            Only goals that are possible now are offered (no "get the key" before it has been
            seen; nothing outside a sealed room while the agent is locked in).
  target    every 6 ticks, or when the target is reached, gone or unsafe: a concrete cell.
            Fighting offers firing spots: cells with a clear line to an enemy, away from enemies
            and out of every bullet's path.
  move      control head, every tick: north/south/east/west/stay, each labelled with its outcome.
  shoot     control head, every tick: hold fire, or shoot one of the enemies in sight. It is
            committed without a model call when holding fire is the only option (gun reloading
            or no enemy in sight).

With ammo (Rules.ammo) the shoot head also offers "reload" while the magazine is not full and the
reserve is not empty (an empty magazine reloads without a model call, the only sensible choice),
the strategy tier offers "pick up ammo" when an ammo box is within reach and the reserve has room,
and the state texts count the bullets against the hits the known enemies still take. Without
ammo every text is exactly as it was evaluated on seeds 0-39.

The two control heads run side by side in the same batch, like the Doom demo's control heads
in sgoedecke/system-one [S4]; neither sees the other's answer. The model makes every choice.
The code only computes what the text says: walking distances over the cells the agent knows,
which cells bullets will cross next tick, and what a shot can hit. A tier with a single possible
option is committed without a model call.
"""

from __future__ import annotations

import math
import re
import statistics
import time
from collections import Counter

from system_one.goals import ONLY_OPTION, GoalStack, Tier, step_all

from ..brain import _steps, base_move
from .bots import brute_reach, fight_cells, fighting, frontier, threat_cells
from .world import DIRS, MOVES, STEP, Cell, Dungeon

STANDING_ORDER = ("Find the key, then leave through the exit alive. Rooms lock you in until their enemies are dead: "
                  "shoot them and keep out of their bullets.")
GOALS = ("explore", "fight the enemies here", "get the key", "go to the exit", "drink a health potion", "pick up ammo")
HOLD = "hold fire"
RELOAD = "reload"
SHOOT_QUESTION = "Which shot do you take this tick?"
GUN_QUESTION = "What do you do with your gun this tick?"  # with ammo: shoot, reload or hold fire
AIM_QUESTION = "Which enemy do you shoot?"
_CELL = re.compile(r"\((\d+),(\d+)\)")
_ENEMY = re.compile(r"#(\d+)")


def _offset(dx: int, dy_north: int) -> str:
    """'3 cells east and 1 cell north' (relative position; the model cannot do the arithmetic)."""
    parts = [f"{abs(v)} {'cell' if abs(v) == 1 else 'cells'} {pos if v > 0 else neg}"
             for v, pos, neg in ((dx, "east", "west"), (dy_north, "north", "south")) if v]
    return " and ".join(parts) or "here"


def cell_of(text: str | None) -> Cell | None:
    m = _CELL.search(text or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def enemy_of(text: str | None) -> int | None:
    """'shoot gunner #3 ...' -> 3; 'hold fire' -> None."""
    m = _ENEMY.search(text or "")
    return int(m.group(1)) if m and text and text.startswith("shoot") else None


def is_reload(text: str | None) -> bool:
    return bool(text) and text.startswith(RELOAD)


def _ticks(n: int) -> str:
    return f"{n} tick" if n == 1 else f"{n} ticks"


def _bullets(n: int) -> str:
    return f"{n} bullet" if n == 1 else f"{n} bullets"


def drop_distance(option: str) -> str:
    """How lower tiers see a target: the part before the first comma (distances go stale)."""
    return option.split(", ")[0]


def _hits(n: int) -> str:
    return f"{n} hit" if n == 1 else f"{n} hits"


class ShooterBrain:
    """Goal stack and state texts of the one agent."""

    def __init__(self, dungeon: Dungeon, strategy_every: int = 12, target_every: int = 6, fire_head: bool = False):
        """``fire_head``: the shoot head offers ONE "shoot" option instead of one per enemy, and an
        aim head in the same batch picks the enemy. Observed (1.5B, seed 0 with ammo): with one
        option per enemy it held fire in 53 of 53 decisions; asked again with the enemies merged
        into one option, 0 of 53 (and 0 of 136 recorded states of dev seeds 1000-1001)."""
        self.d = dungeon
        self.fire_head = fire_head
        aim = [Tier("aim", AIM_QUESTION, lambda goals: self.aim_options(), every=1, context_from=("strategy",))]
        self.stack = GoalStack([
            Tier("strategy", "Which goal should the agent pursue now?", lambda goals: self.goal_options(),
                 every=strategy_every, title="Strategic goal"),
            Tier("target", "Which target best serves the strategic goal?",
                 lambda goals: self.target_options(goals.get("strategy", "explore")),
                 every=target_every, title="Current target", describe=drop_distance),
            Tier("move", "Which move is best? Get closer to your target, but never step into a bullet's path.",
                 lambda goals: self.move_options(), every=1, context_from=("strategy", "target")),
            Tier("shoot", GUN_QUESTION if dungeon.rules.ammo else SHOOT_QUESTION, lambda goals: self.shoot_options(),
                 every=1, context_from=("strategy",)),
        ] + (aim if fire_head else []))
        self._situation: tuple | None = None
        self.refresh()

    # -- per-tick bookkeeping ------------------------------------------------------------

    def refresh(self) -> None:
        """Recompute what the agent knows. Re-plan the strategy when the situation changes or its
        goal became impossible; re-plan the target when it was reached, is gone, or became unsafe."""
        d = self.d
        self._known = d.known_cells()
        self._dist = d.distances(d.pos, within=self._known)
        self._visible = d.visible_enemies()
        self._foes = fighting(d)
        self._danger = d.danger()
        self._threat = threat_cells(d)
        situation = self._situation_now()
        if self._situation is not None and situation != self._situation:
            self.stack.invalidate("strategy")
        self._situation = situation
        strategy = self.stack.current.get("strategy")
        if strategy is not None and strategy not in self.goal_options():
            self.stack.invalidate("strategy")
        target = self.stack.current.get("target")
        if target is not None and self._target_done(target):
            self.stack.invalidate("target")

    def _situation_now(self) -> tuple:
        d = self.d
        hits = min(3, math.ceil(d.health / d.rules.enemy_shot_damage))
        out = hits, bool(self._foes), d.sealed, d.has_key, len(d.seen), d.key is not None and d.key in self._dist
        return out + (self.ammo_word(),) if d.rules.ammo else out

    def _target_done(self, target: str) -> bool:
        d, cell = self.d, cell_of(target)
        if cell is None:
            return True
        if target.startswith("firing spot"):
            return not self._foes or cell in self._threat or not self._firing_spot(cell)
        if cell == d.pos:
            return True
        if target.startswith("health potion"):
            return cell not in d.potions
        if target.startswith("the key"):
            return d.key != cell
        if target.startswith("ammo box"):
            return cell not in d.ammo
        if target.startswith("unexplored room"):
            return d.area.get(cell) in d.seen
        return False

    def target_cell(self) -> Cell | None:
        return cell_of(self.stack.current.get("target"))

    # -- options -----------------------------------------------------------------------

    def _reachable(self, cells) -> list[Cell]:
        return sorted((c for c in cells if c in self._dist), key=lambda c: (c[1], c[0]))  # map order

    def goal_options(self) -> list[str]:
        """The goals that are possible now, in the fixed order of GOALS."""
        d = self.d
        ok = {
            "explore": bool(frontier(d, self._dist)),
            "fight the enemies here": bool(self._foes),
            "get the key": d.key is not None and d.key in self._dist,
            "go to the exit": d.has_key and d.exit in self._dist,
            "drink a health potion": bool(self._reachable(d.potions)) and d.health < d.rules.health,
            "pick up ammo": d.rules.ammo and bool(self._reachable(d.ammo)) and d.box_gain() > 0,
        }
        return [g for g in GOALS if ok[g]]

    def place(self, c: Cell) -> str:
        room = self.d.room_at(c)
        if room is None:
            return "a corridor"
        return f"the {room.name} doorway" if self.d.is_door(c) else f"the {room.name}"

    def _firing_spot(self, c: Cell) -> tuple[int, float] | None:
        """(enemies with a clear line from c, distance to the nearest enemy) if c is a firing spot."""
        foes = self._foes
        if not foes or c in self._threat or self.d.enemy_at(c) is not None:
            return None
        near = min(math.dist(c, e.pos) for e in foes)
        lines = sum(self.d.clear_line(c, e.pos) for e in foes)
        return (lines, near) if lines and near >= 3 else None

    def firing_spots(self, n: int = 5) -> list[tuple[Cell, int, int, float]]:
        """Up to ``n`` firing spots in the fight's room, its doorways or the corridors out of it, the
        quickest to reach first: (cell, steps, enemies in line, nearest enemy distance)."""
        cand = []
        for c in sorted(fight_cells(self.d, self._foes[0].home)):
            spot = self._firing_spot(c) if c in self._dist else None
            if spot:
                cand.append((self._dist[c], -spot[1], c, spot))
        cand.sort()
        return [(c, s, spot[0], spot[1]) for s, _, c, spot in cand[:n]]

    def target_options(self, strategy: str) -> list[str]:
        """Candidate targets for a goal, in map order (not sorted by distance)."""
        d, dist = self.d, self._dist
        if strategy == "explore":
            return [f"unexplored room behind the doorway at ({x},{y}), {_steps(dist[(x, y)])} away"
                    for x, y in sorted(frontier(d, dist), key=lambda c: (c[1], c[0]))]
        if strategy == "fight the enemies here" and self._foes:
            out = []
            for (x, y), s, lines, near in sorted(self.firing_spots(), key=lambda t: (t[0][1], t[0][0])):
                where = "where you stand" if s == 0 else f"{_steps(s)} away"
                out.append(f"firing spot at ({x},{y}), {where}: clear shot at {lines} of {len(self._foes)} "
                           f"{'enemy' if len(self._foes) == 1 else 'enemies'}, nearest enemy {round(near)} cells from it")
            return out
        if strategy == "get the key" and d.key in dist:
            (x, y) = d.key
            return [f"the key at ({x},{y}) in {self.place(d.key)}, {_steps(dist[d.key])} away"]
        if strategy == "go to the exit" and d.exit in dist:
            (x, y) = d.exit
            return [f"the exit at ({x},{y}) in {self.place(d.exit)}, {_steps(dist[d.exit])} away"]
        if strategy == "drink a health potion":
            return [f"health potion at ({x},{y}) in {self.place((x, y))}, {_steps(dist[(x, y)])} away"
                    for x, y in self._reachable(d.potions)]
        if strategy == "pick up ammo" and d.rules.ammo:
            return [f"ammo box at ({x},{y}) in {self.place((x, y))}, {_steps(dist[(x, y)])} away"
                    for x, y in self._reachable(d.ammo)]
        return []

    def _brute_next_to(self, c: Cell) -> bool:
        """A brute the agent has seen would hit it on ``c`` this tick (a sleeping one too, if ``c``
        is inside its room: stepping in wakes it)."""
        return any(c in brute_reach(self.d, e) for e in self.d.living() if e.home in self.d.seen)

    def move_options(self) -> list[str]:
        """Each move labelled with its outcome. The model still chooses.

        "closer/farther" is measured along routes that avoid the cells an enemy can hurt next tick
        (bullet paths, next to a brute). Observed on dev seed 1000 with plain walking distance: the
        1.5B chose "next to a brute; closer" at p = 0.78 when the only short route passed the brute.
        A move into danger is labelled with the damage only."""
        d, target = self.d, self.target_cell()
        tdist = self._route_dist()
        here = tdist.get(d.pos)
        out = []
        for move in MOVES:
            dx, dy = STEP[move]
            c = (d.pos[0] + dx, d.pos[1] + dy)
            if move != "stay" and (not d.passable(c) or d.enemy_at(c) is not None):
                continue  # walls, sealed doorways and enemies are not offered (they would be trap options)
            notes = []
            if c in self._danger:
                notes.append(f"BULLET: -{self._danger[c]} health")
            if self._brute_next_to(c):
                notes.append(f"next to a brute: -{d.rules.brute_damage} health")
            if move != "stay":
                if c == d.exit:
                    notes.append("EXIT: you escape" if d.has_key else "the exit, locked without the key")
                elif c == d.key:
                    notes.append("pick up the key")
                elif c in d.potions:
                    notes.append(f"potion: +{d.rules.potion_health} health")
            if d.rules.ammo and c in d.ammo:
                gain = d.box_gain()
                notes.append(f"ammo box: +{gain} bullets" if gain else "ammo box: your reserve is full")
            risky = c in self._danger or self._brute_next_to(c)
            if target is not None and not risky:
                if c == target:
                    notes.append("on the target" if move == "stay" else "reach the target")
                elif c in tdist and here is not None:
                    n = tdist[c]
                    rel = "closer" if n < here else "farther" if n > here else "no closer"
                    notes.append(f"{rel}: {_steps(n)} to the target")
                elif c in tdist:
                    notes.append(f"target: {_steps(tdist[c])}")
                else:
                    notes.append("no safe route to the target")
            elif target is not None and c == target:
                notes.append("on the target" if move == "stay" else "reach the target")
            if not risky:
                notes.insert(0, "safe")
            out.append(f"{move} ({'; '.join(notes)})")
        return out

    def _route_dist(self) -> dict[Cell, int]:
        """Walking distance to the target over known cells that no enemy can hurt next tick."""
        d, target = self.d, self.target_cell()
        if target is None:
            return {}
        return d.distances(target, within=(self._known - self._threat) | {target, d.pos})

    def waypoint(self) -> tuple[Cell, int] | None:
        """(the farthest cell of the safe route that lies in a straight line from the agent, route
        length). Precedent: the Doom demo in sgoedecke/system-one gave its model an A*-computed
        waypoint bearing [S4]. Observed on dev seeds 1000-1001: with the target's own bearing the 3B
        walked into a wall or away from a route that bends (e.g. out of a doorway)."""
        d, dist = self.d, self._route_dist()
        here = dist.get(d.pos)
        if here is None or here == 0:
            return None
        path, c = [], d.pos
        while dist[c] > 0:  # follow the gradient; ties in DIRS order, so it is deterministic
            c = next(n for n in ((c[0] + dx, c[1] + dy) for dx, dy in DIRS.values()) if dist.get(n) == dist[c] - 1)
            path.append(c)
        far = next((p for p in reversed(path) if d.clear_line(d.pos, p)), path[0])
        return far, here

    def _enemy_name(self, e) -> str:
        return f"{e.kind} #{e.id}"

    def _reload_option(self) -> str:
        d, R = self.d, self.d.rules
        now = "magazine empty" if d.loaded == 0 else f"{d.loaded} of {R.magazine} bullets loaded"
        return f"{RELOAD} ({now}; {d.reserve} in reserve; takes {_ticks(R.reload_ticks)}, no shooting meanwhile)"

    def shoot_options(self) -> list[str]:
        d = self.d
        if d.rules.ammo:
            if d.reloading:
                return [f"{HOLD} (reloading: ready in {_ticks(d.reloading)})"]
            if d.loaded == 0:  # reloading is the only sensible choice; out of bullets, nothing is
                return [self._reload_option()] if d.reserve > 0 else [f"{HOLD} (out of bullets)"]
            if d.cooldown > 0:
                return [f"{HOLD} (next shot ready in {_ticks(d.cooldown)})"]
        elif d.cooldown > 0:
            return [f"{HOLD} (the gun is reloading: ready in {d.cooldown} {'tick' if d.cooldown == 1 else 'ticks'})"]
        reload = [self._reload_option()] if d.can_reload() else []
        if not self._visible:
            return reload + [f"{HOLD} (no enemy in sight)"]
        if self.fire_head:
            n = len(self._visible)
            return [f"shoot ({n} {'enemy' if n == 1 else 'enemies'} in sight, clear line)"] + reload + [HOLD]
        return self._shots() + reload + [HOLD]

    def _shots(self) -> list[str]:
        d, (x, y) = self.d, self.d.pos
        out = []
        for e in self._visible:
            hp = e.hp // d.rules.shot_damage + (e.hp % d.rules.shot_damage > 0)
            what = "aiming at you" if e.aiming else "awake" if e.awake else "asleep"
            out.append(f"shoot {self._enemy_name(e)}, {_offset(e.pos[0] - x, y - e.pos[1])} ({what}; "
                       f"{_hits(hp)} to kill; clear line)")
        return out

    def aim_options(self) -> list[str]:
        """With the fire head: the enemies the shoot head's "shoot" would fire at (none when it
        offers no shot). One enemy in sight is committed without a model call."""
        return self._shots() if any(o.startswith("shoot (") for o in self.shoot_options()) else []

    # -- world -> text -----------------------------------------------------------------

    def status(self) -> str:
        d, h = self.d, self.d.health
        hits = math.ceil(h / d.rules.enemy_shot_damage)
        word = "CRITICAL" if hits <= 2 else "hurt" if h < 60 else "fine"
        return f"Health: {word} ({h}/{d.rules.health}): {hits} more bullet {'hit' if hits == 1 else 'hits'} would kill you."

    def known_foes(self) -> list:
        """Living enemies of the rooms the agent has seen."""
        return [e for e in self.d.living() if e.home in self.d.seen]

    def hits_needed(self) -> int:
        R = self.d.rules
        return sum(e.hp // R.shot_damage + (e.hp % R.shot_damage > 0) for e in self.known_foes())

    def ammo_word(self) -> str:
        d = self.d
        total = d.ammo_total()
        return "OUT" if total == 0 else "LOW" if total < max(d.rules.magazine, self.hits_needed()) else "fine"

    def ammo_status(self) -> str:
        d, R = self.d, self.d.rules
        n, need = len(self.known_foes()), self.hits_needed()
        foes = (f"the {n} living {'enemy' if n == 1 else 'enemies'} you know of {'takes' if n == 1 else 'take'} "
                f"{_hits(need)}" if n else "no living enemy you know of")
        word = self.ammo_word()
        if word == "OUT":
            return f"Ammo: OUT: no bullets left; {foes}."
        return (f"Ammo: {word} ({_bullets(d.ammo_total())}: {d.loaded} loaded, {d.reserve} in reserve, at most "
                f"{R.max_reserve} in reserve); {foes}.")

    def _boxes_line(self) -> str:
        d = self.d
        p = self._nearest(d.ammo)
        if not p:
            return "Ammo boxes: none within reach."
        full = "; your reserve is full" if d.box_gain() == 0 else ""
        return (f"Ammo boxes: {p[0]} known, nearest {_steps(p[1])} away; each gives up to "
                f"+{d.rules.ammo_box} bullets{full}.")

    def _nearest(self, cells) -> tuple[int, int] | None:
        steps = [self._dist[c] for c in cells if c in self._dist]
        return (len(steps), min(steps)) if steps else None

    def _enemies_line(self, limit: int = 4) -> str:
        d = self.d
        x, y = d.pos
        foes = self._foes or self._visible
        if not foes:
            return "Enemies: none in sight."
        parts = []
        for e in sorted(foes, key=lambda e: (math.dist(d.pos, e.pos), e.id))[:limit]:
            state = ("AIMING at you: fires this tick" if e.aiming else
                     "resting" if e.kind == "brute" and e.timer else "awake" if e.awake else "asleep")
            parts.append(f"{self._enemy_name(e)} {_offset(e.pos[0] - x, y - e.pos[1])} ({e.hp} health, {state})")
        where = " in sight"
        if self._foes:
            home = self._foes[0].home
            here = d.room_at(d.pos)
            where = " in this room" if (here is not None and here.id == home) or d.sealed == home else \
                f" in the {d.rooms[home].name}"
        return f"Enemies{where}: " + "; ".join(parts) + "."

    def _bullets_line(self) -> str:
        d = self.d
        x, y = d.pos
        incoming = [b for b in d.bullets if b.owner == "enemy"]
        if not incoming:
            return "Enemy bullets: none flying."
        parts = []
        for b in sorted(incoming, key=lambda b: math.dist((b.x, b.y), d.pos))[:3]:
            bx, by = round(b.x), round(b.y)
            parts.append(f"one {_offset(bx - x, y - by)} of you")
        return f"Enemy bullets flying: {len(incoming)} ({'; '.join(parts)})."

    def strategy_state(self) -> str:
        d = self.d
        lines = [f"Standing order: {STANDING_ORDER}",
                 f"You are in {self.place(d.pos)}. Rooms explored: {len(d.seen)} of {len(d.rooms)}; "
                 f"rooms cleared: {len(d.cleared)}.", self.status()]
        if d.rules.ammo:
            lines.append(self.ammo_status())
        if d.sealed is not None:
            lines.append(f"The doorways of the {d.rooms[d.sealed].name} are sealed until its "
                         f"{len(d.living(d.sealed))} {'enemy is' if len(d.living(d.sealed)) == 1 else 'enemies are'} dead.")
        if d.has_key:
            lines.append("The key: you carry it, so the exit will open for you.")
        elif d.key in self._dist:
            lines.append(f"The key: seen in {self.place(d.key)}, {_steps(self._dist[d.key])} away. "
                         "The exit stays locked until you carry it.")
        elif d.key is not None and d.key in self._known:
            lines.append(f"The key: seen in {self.place(d.key)}, out of reach while the doorways are sealed.")
        else:
            lines.append("The key: not found yet. The exit stays locked until you carry it.")
        if d.exit in self._dist:
            lines.append(f"The exit: in {self.place(d.exit)}, {_steps(self._dist[d.exit])} away"
                         + ("." if d.has_key else ", locked."))
        elif d.exit in self._known:
            lines.append(f"The exit: in {self.place(d.exit)}, out of reach while the doorways are sealed.")
        else:
            lines.append("The exit: not found yet.")
        p = self._nearest(d.potions)
        lines.append(f"Health potions: {p[0]} known, nearest {_steps(p[1])} away; each gives "
                     f"+{d.rules.potion_health} health." if p else "Health potions: none within reach.")
        if d.rules.ammo:
            lines.append(self._boxes_line())
        lines.append(self._enemies_line())
        fr = frontier(d, self._dist)
        if fr:
            lines.append(f"Unexplored rooms: {len(fr)} reachable, nearest doorway {_steps(min(self._dist[c] for c in fr))} away.")
        else:
            lines.append("Unexplored rooms: none reachable now." if len(d.seen) < len(d.rooms) else "Unexplored rooms: none left.")
        return " ".join(lines)

    def move_state(self) -> str:
        d = self.d
        x, y = d.pos
        cell, way = self.target_cell(), self.waypoint()
        if cell is None:
            where = "You have no target yet."
        elif cell == d.pos:
            where = "You are on your target."
        elif way is None:
            where = f"Your target is {_offset(cell[0] - x, y - cell[1])} of you; no safe route to it now."
        elif way[0] == cell:
            where = f"Your target is {_offset(cell[0] - x, y - cell[1])} of you ({_steps(way[1])} away)."
        else:
            w = way[0]
            where = (f"Your target is {_steps(way[1])} away along the safe route. The route's next waypoint is "
                     f"{_offset(w[0] - x, y - w[1])} of you.")
        nb = []
        for name, (dx, dy) in DIRS.items():
            c = (x + dx, y + dy)
            nb.append(f"{name} {'wall' if not d.passable(c) else 'enemy' if d.enemy_at(c) else 'bullet path' if c in self._danger else 'free'}")
        return " ".join([f"You are at ({x},{y}) in {self.place(d.pos)}. North is up.", where,
                         "Neighbouring cells: " + ", ".join(nb) + ".", self._bullets_line(), self._enemies_line(3),
                         self.status()])

    def shoot_state(self) -> str:
        d, R = self.d, self.d.rules
        if R.ammo:
            now = (f"reloading, ready in {_ticks(d.reloading)}" if d.reloading else "empty" if d.loaded == 0
                   else "ready" if d.cooldown == 0 else f"next shot in {_ticks(d.cooldown)}")
            return " ".join([
                f"You are in {self.place(d.pos)}. Your gun ({now}): {d.loaded} of {R.magazine} bullets loaded, "
                f"{d.reserve} in reserve. It fires one bullet every {R.cooldown} ticks, {R.shot_speed:g} cells per tick, "
                f"flying to where the enemy stands now. A reload takes {_ticks(R.reload_ticks)}, and the gun cannot "
                "fire meanwhile.", self.ammo_status(), self._enemies_line(), self.status()])
        gun = "ready" if d.cooldown == 0 else f"reloading ({d.cooldown} {'tick' if d.cooldown == 1 else 'ticks'})"
        return " ".join([f"You are in {self.place(d.pos)}. Your gun is {gun}: one bullet every {d.rules.cooldown} ticks, "
                         f"{d.rules.shot_speed:g} cells per tick, flying to where the enemy stands now.",
                         self._enemies_line(), self.status()])

    def state_for(self, tier: Tier) -> str:
        return {"move": self.move_state, "shoot": self.shoot_state, "aim": self.shoot_state}.get(
            tier.name, self.strategy_state)()


class Runner:
    """Runs one episode. Each tick is ONE batched engine call for every due decision, and
    returns a trace record: the world the decisions saw, every decision with its question,
    options and probabilities, the batch latency, and what happened."""

    STUCK = 6  # ticks without getting closer to an unchanged target that count as stuck (as in the dungeon)

    def __init__(self, dungeon: Dungeon, engine, group_size: int = 8, plan_budget: int | None = 1,
                 order_debias: bool = True, **brain_options):
        """``order_debias``: every decision is read in two option orders and averaged (Engine)."""
        self.d, self.engine, self.group_size, self.plan_budget = dungeon, engine, group_size, plan_budget
        self.order_debias = order_debias
        self.brain = ShooterBrain(dungeon, **brain_options)
        self.records: list[dict] = []
        self._t0: float | None = None
        self._best: int | None = None
        self._streak = 0

    def header(self) -> dict:
        eng = self.engine
        example = eng.render(self.brain.stack.decision(self.brain.stack.tier("move"), "<state>"), ["A", "B", "C", "D", "E"])
        return {
            "type": "header", "version": 1, "scenario": "shooter", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "standing_order": STANDING_ORDER, "goals": list(GOALS), "map": self.d.layout(),
            "tiers": [{"name": t.name, "question": t.instruction, "every": t.every, "title": t.title,
                       "kind": "control" if t.every == 1 else "plan"} for t in self.brain.stack.tiers],
            "group_size": self.group_size, "plan_budget": self.plan_budget, "order_debias": self.order_debias,
            "fire_head": self.brain.fire_head,
            "backend": eng.backend.info(),
            "brain": {"strategy_every": self.brain.stack.tier("strategy").every,
                      "target_every": self.brain.stack.tier("target").every},
            "engine": {"answer": eng.answer, "prompt_order": eng.prompt_order, "system_prompt": eng.system_prompt,
                       "prefill": eng.prefill, "suffix": eng.suffix},
            "example_prompt": example,
        }

    def tick(self) -> dict:
        d, stack = self.d, self.brain.stack
        start = time.perf_counter()
        if self._t0 is None:
            self._t0 = start
        self.brain.refresh()
        world, tick = d.snapshot(), d.tick
        target_before = self.brain.target_cell()
        before, self.engine.order_debias = getattr(self.engine, "order_debias", False), self.order_debias
        try:
            updates = step_all(self.engine, [(stack, self.brain.state_for)], tick, group_size=self.group_size,
                               plan_budget=self.plan_budget, skip_single=True)
        finally:
            self.engine.order_debias = before
        model = [u for u in updates if u[2].method != ONLY_OPTION]
        stats = dict(self.engine.last_stats) if model else {}
        move = base_move(stack.current.get("move", "stay"))
        shoot = enemy_of(stack.current.get("shoot"))
        if self.brain.fire_head and (stack.current.get("shoot") or "").startswith("shoot ("):
            aimed = [r for _, tier, r, _ in updates if tier.name == "aim"]  # decided in this tick's batch
            shoot = enemy_of(aimed[-1].choice) if aimed else None
        reload = is_reload(stack.current.get("shoot"))
        target = self.brain.target_cell()
        dist_before = self._dist_to(target_before)
        events = d.step(move, shoot, reload)
        self._track_progress(target_before, dist_before, target)
        rec = {
            "type": "tick", "tick": tick, "t": round(start - self._t0, 4), "move": move, "shoot": shoot, "world": world,
            **({"reload": reload} if d.rules.ammo else {}),
            "target": list(target) if target else None, "events": events, "stuck_streak": self._streak,
            "batch": {"decisions": len(model), "forward_ms": round(1000 * stats.get("forward_s", 0.0), 1),
                      "prompt_tokens": stats.get("prompt_tokens", 0),
                      "tick_ms": round(1000 * (time.perf_counter() - start), 1)},
            "decisions": [self._decision(tier, r, tour) for _, tier, r, tour in updates],
            "goals": {t.name: {"choice": stack.current.get(t.name), "p": round(stack.probability.get(t.name, 0.0), 5),
                               "since": stack.decided_at(t.name), "waiting": stack.waiting(t.name)}
                      for t in stack.tiers},
            "tournaments": {name: dict(zip(("round", "resolved", "contested"), t.progress()),
                                       alive=len(t.alive), options=len(t.options))
                            for name, t in stack.tournaments.items()},
        }
        self.records.append(rec)
        return rec

    def _dist_to(self, cell: Cell | None) -> int | None:
        if cell is None:
            return None
        return self.d.distances(self.d.pos, within=self.d.known_cells() | {cell}).get(cell)

    def _track_progress(self, cell: Cell | None, before: int | None, target_now: Cell | None) -> None:
        """Stuck = ticks in streaks of STUCK+ ticks without getting closer to an unchanged target.
        Standing on a firing spot is not stuck: the target is reached."""
        if before is None or cell != target_now or before == 0:
            self._best, self._streak = None, 0
            return
        now = self._dist_to(cell)
        now = before if now is None else now
        if self._best is None or now < self._best:
            self._best, self._streak = now, 0
        else:
            self._streak += 1

    @staticmethod
    def _decision(tier, r, tour) -> dict:
        dec = r.decision
        out = {
            "tier": tier.name, "kind": "control" if tier.every == 1 else "plan", "question": dec.instruction,
            "state": dec.state, "context": dec.context, "options": list(dec.options), "labels": list(r.labels),
            "probs": [round(p, 5) for p in r.probs], "choice": r.index, "outside": round(r.outside_mass, 5),
            "method": r.method, "top_token": r.top_token, "prompt_tokens": r.prompt_tokens,
        }
        if getattr(r, "orders", None):
            out["orders"] = [[round(p, 5) for p in ps] for ps in r.orders]
        if tour is not None:
            out["tournament"] = {"options": len(tour.options), "alive": len(tour.alive), "done": tour.done,
                                 "winner": tour.winner if tour.done else None}
        return out

    def run(self, max_ticks: int | None = None, on_tick=None) -> dict:
        while self.d.outcome is None and (max_ticks is None or self.d.tick < max_ticks):
            rec = self.tick()
            if on_tick:
                on_tick(rec)
        return self.end()

    def end(self) -> dict:
        return {"type": "end", "tick": self.d.tick, "outcome": self.d.outcome, "cause": self.d.cause,
                "world": self.d.snapshot(), "summary": summarise(self.records, self.d)}


def _risky(option: str) -> bool:
    return "BULLET" in option or "next to a brute" in option


def summarise(records: list[dict], d: Dungeon) -> dict:
    """Closed-loop metrics of one run."""
    events = [(r["tick"], e) for r in records for e in r["events"]]

    def first(kind, **match):
        return next((t for t, e in events if e["kind"] == kind and all(e.get(k) == v for k, v in match.items())), None)

    fw = [r["batch"]["forward_ms"] for r in records if r["batch"]["decisions"]]
    ticks = [r["batch"]["tick_ms"] for r in records]
    stuck = 0
    for r in records:
        s = r["stuck_streak"]
        stuck += 0 if s < Runner.STUCK else Runner.STUCK if s == Runner.STUCK else 1
    decisions = [x for r in records for x in r["decisions"]]
    moves = [x for x in decisions if x["tier"] == "move" and x["method"] != ONLY_OPTION]
    # a move into a bullet's path or next to a brute when a safe open move existed
    avoidable = [x for x in moves if _risky(x["options"][x["choice"]])
                 and any(not _risky(o) and "wall" not in o and "sealed" not in o and "blocked" not in o for o in x["options"])]
    shoots = [x for x in decisions if x["tier"] == "shoot" and x["method"] != ONLY_OPTION]
    shots = sum(1 for _, e in events if e["kind"] == "shot")
    hits_scored = sum(1 for _, e in events if e["kind"] in ("enemy_hit",))
    hits = [e for _, e in events if e["kind"] == "hit"]
    # hits on a tick whose chosen move was labelled safe: each one is a label that was wrong
    safe_hits = 0
    for r in records:
        mv = next((x for x in r["decisions"] if x["tier"] == "move"), None)
        if mv and not _risky(mv["options"][mv["choice"]]):
            safe_hits += sum(1 for e in r["events"] if e["kind"] == "hit")
    ammo = {}
    if d.rules.ammo:
        chosen = [x for x in shoots if is_reload(x["options"][x["choice"]])]
        offered = [x for x in shoots if any(is_reload(o) for o in x["options"])]
        dry = [r for r in records if r["world"]["agent"]["loaded"] + r["world"]["agent"]["reserve"] == 0]
        ammo = {
            "reloads": d.reloads,
            "reloads_chosen": len(chosen),  # by the model; the rest were an empty magazine (only option)
            "reloads_chosen_in_fight": sum(1 for x in chosen if any(o.startswith("shoot") for o in x["options"])),
            "reload_offered": len(offered),
            "ammo_boxes_picked": sum(1 for _, e in events if e["kind"] == "ammo"),
            "ammo_picked": d.ammo_picked,
            "ticks_out_of_ammo": len(dry), "first_out_of_ammo_tick": dry[0]["tick"] if dry else None,
            "ammo_left": d.ammo_total(),
            "min_ammo": min([r["world"]["agent"]["loaded"] + r["world"]["agent"]["reserve"] for r in records]
                            + [d.ammo_total()]),
            "ammo_goal_ticks": sum(1 for r in records if r["goals"]["strategy"]["choice"] == "pick up ammo"),
        }
    return {
        "outcome": d.outcome, "cause": d.cause, "ticks": len(records),
        "hits": len(hits), "hits_gunner": sum(e["by"] == "gunner" for e in hits), "hits_brute": sum(e["by"] == "brute" for e in hits),
        "hits_after_safe_move": safe_hits,
        "shots": shots, "shots_on_target": hits_scored, "kills": d.kills,
        "shoot_decisions": len(shoots), "held_fire": sum(1 for x in shoots if x["options"][x["choice"]] == HOLD),
        "move_decisions": len(moves), "avoidable_risky_moves": len(avoidable),
        "potions_drunk": sum(1 for _, e in events if e["kind"] == "potion"),
        "bumps": sum(1 for _, e in events if e["kind"] == "bump"),
        "rooms_seen": len(d.seen), "rooms_cleared": len(d.cleared),
        "key_seen_tick": first("room_seen", room=d.key_room), "key_tick": first("key"),
        "exit_seen_tick": first("room_seen", room=d.exit_room), "escape_tick": first("escaped"),
        "min_health": min([r["world"]["agent"]["health"] for r in records] + [d.health]),
        "stuck_ticks": stuck,
        "strategy_ticks": dict(Counter(r["goals"]["strategy"]["choice"] or "-" for r in records)),
        "model_decisions": sum(1 for x in decisions if x["method"] != ONLY_OPTION),
        "planning_decisions": sum(1 for x in decisions if x["kind"] == "plan" and x["method"] != ONLY_OPTION),
        "only_option_commits": sum(1 for x in decisions if x["method"] == ONLY_OPTION),
        "tournament_decisions": sum(1 for x in decisions if "tournament" in x),
        "forward_ms_median": round(statistics.median(fw), 1) if fw else 0.0,
        "forward_ms_p90": round(statistics.quantiles(fw, n=10)[-1], 1) if len(fw) >= 10 else max(fw, default=0.0),
        "forward_ms_max": max(fw, default=0.0),
        "tick_ms_median": round(statistics.median(ticks), 1) if ticks else 0.0,
        **ammo,
    }
