"""A small labelled decision set used to choose a model and an answer template.

Categories mirror what the demo needs (moves, goals, targets) plus general classification.
It is small (~50 items); treat its accuracy numbers as a smoke test, not a benchmark.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from system_one import Decision

MOVES = ("move north", "move south", "move east", "move west", "stay")
GOALS = ("collect gems", "find food", "avoid hazards", "explore")


@dataclass(frozen=True)
class Item:
    category: str
    decision: Decision
    correct: frozenset[str]


def _general() -> list[Item]:
    rows = [
        ("What colour is the sky on a clear day?", "", ("red", "blue", "yellow"), {"blue"}),
        ("What is the capital of Japan?", "", ("Kyoto", "Osaka", "Tokyo", "Seoul"), {"Tokyo"}),
        ("What is the sentiment of the review?", "I absolutely love this phone, the battery lasts for days.",
         ("negative", "neutral", "positive"), {"positive"}),
        ("What is the sentiment of the review?", "It broke after two days and support never answered.",
         ("positive", "negative", "neutral"), {"negative"}),
        ("Which team should handle this ticket?", "My shoes arrived in the wrong size. Can I exchange them?",
         ("billing and payments", "shipping and delivery", "returns and exchanges"), {"returns and exchanges"}),
        ("Which team should handle this ticket?", "I was charged twice for the same order this month.",
         ("returns and exchanges", "billing and payments", "shipping and delivery"), {"billing and payments"}),
        ("Which team should handle this ticket?", "My package was due last week and still has not arrived.",
         ("billing and payments", "returns and exchanges", "shipping and delivery"), {"shipping and delivery"}),
        ("Is this message spam?", "WIN a FREE iPhone!!! Click this link now to claim your prize.",
         ("not spam", "spam"), {"spam"}),
        ("Which language is this text in?", "Bonjour, comment ça va aujourd'hui ?",
         ("Spanish", "German", "Italian", "French"), {"French"}),
        ("Which of these animals is a mammal?", "", ("shark", "salmon", "dolphin", "octopus"), {"dolphin"}),
        ("What is 7 + 5?", "", ("10", "11", "13", "12"), {"12"}),
        ("What is the opposite of hot?", "", ("warm", "cold", "wet", "loud"), {"cold"}),
        ("How urgent is this ticket?", "The whole website is down and no customer can log in.",
         ("low", "medium", "high"), {"high"}),
        ("Which planet is the largest?", "", ("Earth", "Mars", "Jupiter", "Venus"), {"Jupiter"}),
    ]
    return [Item("general", Decision(q, o, state=s), frozenset(c)) for q, s, o, c in rows]


def _offset_text(dx: int, dy: int) -> str:
    ew = f"{abs(dx)} cells {'east' if dx >= 0 else 'west'}"
    ns = f"{abs(dy)} cells {'north' if dy >= 0 else 'south'}"
    return f"{ew} and {ns}"


def _navigation(rng: random.Random) -> list[Item]:
    items = []
    # Straight-line targets, one per direction, then diagonal ones and blocked ones.
    cases = [(3, 0), (-4, 0), (0, 2), (0, -5), (5, 0), (0, 3), (-2, 0), (0, -1)]
    cases += [(2, 3), (-3, 1), (4, -2), (-1, -4)]
    for dx, dy in cases:
        ok = set()
        if dx > 0: ok.add("move east")
        if dx < 0: ok.add("move west")
        if dy > 0: ok.add("move north")
        if dy < 0: ok.add("move south")
        state = (f"You are at (5,5). North is up. Your target is {_offset_text(dx, dy)} of you. "
                 "Neighbouring cells: north free, south free, east free, west free.")
        items.append(Item("navigation", Decision("Which move brings you closer to your target?", MOVES, state=state),
                          frozenset(ok)))
    # Blocked straight line: the only good moves go around.
    for dx, dy, blocked, ok in [(3, 0, "east", {"move north", "move south"}),
                                (0, 2, "north", {"move east", "move west"}),
                                (2, 2, "east", {"move north"}),
                                (-2, -3, "south", {"move west"})]:
        free = {d: "free" for d in ("north", "south", "east", "west")}
        free[blocked] = "wall"
        cells = ", ".join(f"{d} {v}" for d, v in free.items())
        state = (f"You are at (5,5). North is up. Your target is {_offset_text(dx, dy)} of you. "
                 f"Neighbouring cells: {cells}. You cannot move into a wall.")
        items.append(Item("navigation", Decision("Which move brings you closer to your target?", MOVES, state=state),
                          frozenset(ok)))
    return items


def _goals(rng: random.Random) -> list[Item]:
    rows = [
        (12, 90, 3, 2, 8, "find food"),
        (80, 90, 6, 2, 9, "collect gems"),
        (70, 60, 5, 3, 1, "avoid hazards"),
        (20, 70, 2, 5, 10, "find food"),
        (95, 100, 9, 1, 12, "collect gems"),
        (60, 80, 4, None, 9, "explore"),
        (85, 40, 7, 4, 1, "avoid hazards"),
        (8, 100, 4, 3, 6, "find food"),
        (75, 90, 8, None, 11, "explore"),
        (90, 85, 3, 1, 7, "collect gems"),
    ]
    items = []
    for energy, health, food, gem, hazard, goal in rows:
        gem_text = f"Nearest gem {gem} cells away." if gem is not None else "No gems are visible."
        state = (f"Energy {energy}/100. Health {health}/100. Nearest food {food} cells away. {gem_text} "
                 f"Nearest hazard {hazard} cells away.")
        rules = ("Rules: if energy is below 30, find food. If a hazard is 1 cell away, avoid hazards. "
                 "If no gems are visible, explore. Otherwise collect gems.")
        items.append(Item("goal", Decision("Which goal should the agent pursue now?", GOALS,
                                           state=state, context=rules), frozenset({goal})))
    return items


def _goals_in_words() -> list[Item]:
    """Same situations as _goals, but the state names the condition instead of listing rules.

    Reported separately (not part of the model-selection rule): it tests whether the state
    encoding, rather than the model, causes goal-selection failures.
    """
    rows = [
        ("LOW", "fine", "3 cells away", "2 cells away", "far", "find food"),
        ("high", "fine", "6 cells away", "2 cells away", "far", "collect gems"),
        ("high", "fine", "5 cells away", "3 cells away", "ADJACENT", "avoid hazards"),
        ("LOW", "fine", "2 cells away", "5 cells away", "far", "find food"),
        ("high", "fine", "9 cells away", "1 cell away", "far", "collect gems"),
        ("high", "fine", "4 cells away", "none visible", "far", "explore"),
        ("high", "low", "7 cells away", "4 cells away", "ADJACENT", "avoid hazards"),
        ("LOW", "fine", "4 cells away", "3 cells away", "far", "find food"),
        ("high", "fine", "8 cells away", "none visible", "far", "explore"),
        ("high", "fine", "3 cells away", "1 cell away", "far", "collect gems"),
    ]
    items = []
    for energy, health, food, gem, hazard, goal in rows:
        state = f"Energy: {energy}. Health: {health}. Food: {food}. Gems: {gem}. Hazard: {hazard}."
        items.append(Item("goal_words", Decision("Which goal should the agent pursue now?", GOALS,
                                                 state=state), frozenset({goal})))
    return items


def _targets(rng: random.Random) -> list[Item]:
    items = []
    for n in (4, 6, 8, 10, 12, 6, 8, 10):
        dists = rng.sample(range(2, 30), n)
        cells = [(rng.randrange(0, 20), rng.randrange(0, 15)) for _ in range(n)]
        opts = tuple(f"gem at ({x},{y}), {d} steps away" for (x, y), d in zip(cells, dists))
        best = opts[dists.index(min(dists))]
        items.append(Item("target", Decision("Which gem is closest to you?", opts,
                                             state="You are collecting gems. Pick the nearest one."),
                          frozenset({best})))
    return items


def eval_items(seed: int = 7) -> list[Item]:
    rng = random.Random(seed)
    return _general() + _navigation(rng) + _goals(rng) + _targets(rng) + _goals_in_words()


#: Categories used by the pre-registered model-selection rule (>= 80% in each).
SELECTION_CATEGORIES = ("general", "navigation", "goal", "target")
