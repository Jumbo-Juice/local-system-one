"""Build the four-run comparison page: two random 1.5B runs and two random 3B runs, side by side.

    python -m demo.shooter.quad                      # pool: demo/output/shooter/*/trace.jsonl + docs/shooter_replay_*.html
    python -m demo.shooter.quad --out some.html a/trace.jsonl b/replay.html ...

Every run in the pool is slimmed (prompts and option texts dropped; world state, events and which
decision fired on each tick kept) and embedded in demo/shooter/quad.html, so the page opens from disk.
The page picks two runs per model at random on load and on Shuffle; it replays recorded decisions only.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).with_name("quad.html")
_SLOT = re.compile(r'<script id="runs" type="application/json">.*?</script>', re.S)
_TRACE = re.compile(r'<script id="trace" type="application/x-ndjson">(.*?)</script>', re.S)

GOALS = ["explore", "fight the enemies here", "get the key", "go to the exit", "drink a health potion"]
# A target is coloured by the goal it serves: "firing spot" belongs to fight, "the key" to get the key, ...
TARGET_KIND = [("unexplored", 0), ("firing spot", 1), ("the key", 2), ("the exit", 3), ("health potion", 4)]


def read_trace(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".html":
        m = _TRACE.search(text)
        if not m:
            raise ValueError(f"{path} has no embedded trace")
        text = m.group(1)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def run_name(path: Path) -> str:
    if path.name == "trace.jsonl":
        return path.parent.name
    return path.stem.removeprefix("shooter_replay_")


def model_size(header: dict) -> str | None:
    model = str((header.get("backend") or {}).get("model", ""))
    for size in ("1.5B", "3B"):
        if f"-{size}" in model:
            return size
    return None


def target_kind(choice: str | None) -> int:
    for prefix, k in TARGET_KIND:
        if choice and choice.startswith(prefix):
            return k
    return -1


def world(w: dict) -> list:
    """[ax, ay, health, has_key, kills, enemies, bullets, potions, key, seen, cleared, sealed]
    enemies: flat [x, y, hp, flags] per enemy (flags: 1 awake, 2 aiming, 4 resting)
    bullets: flat [x, y, vx, vy, mine] per bullet"""
    a = w["agent"]
    en = []
    for e in w["enemies"]:
        en += [*e["pos"], e["hp"], int(e["awake"]) | int(e.get("aiming", False)) << 1 | int(e.get("resting", False)) << 2]
    bl = []
    for b in w["bullets"]:
        bl += [round(b["pos"][0], 2), round(b["pos"][1], 2), round(b["vel"][0], 2), round(b["vel"][1], 2),
               int(b["owner"] == "agent")]
    items = w["items"]
    return [*a["pos"], a["health"], int(a["has_key"]), a["kills"], en, bl,
            [c for p in items["potions"] for c in p], items["key"], w["seen"], w["cleared"], w["sealed"]]


def events(evs: list[dict]) -> list:
    out = []
    for e in evs:
        k = e["kind"]
        if k == "hit":
            out.append(["hit", e["damage"]])
        elif k == "enemy_killed":
            out.append(["kill", *e["cell"]])
        elif k == "shot":
            out.append(["shot", *e["from"], *e["at"]])
        elif k in ("key", "potion"):
            out.append([k, *e["cell"]])
        elif k in ("room_cleared", "room_sealed", "room_seen"):
            out.append([k.removeprefix("room_"), e["room"]])
    return out


def decided(tick: dict, tier: str) -> int:
    """0 no decision this tick, 1 a model call, 2 committed without one (a single possible option)."""
    ds = [d for d in tick["decisions"] if d["tier"] == tier]
    if not ds:
        return 0
    return 2 if ds[-1]["method"] == "only_option" else 1


def slim(path: Path) -> dict | None:
    records = read_trace(path)
    header = records[0]
    ticks = [r for r in records if r.get("type") == "tick"]
    end = next((r for r in records if r.get("type") == "end"), None)
    size = model_size(header)
    if header.get("scenario") != "shooter" or not ticks or size is None:
        return None
    if list(header.get("goals", [])) != GOALS:
        raise ValueError(f"{path}: goal list differs from {GOALS}")
    m = header["map"]
    rows = []
    for t in ticks:
        g = t["goals"]
        s = (g.get("strategy") or {}).get("choice")
        rows.append({
            "w": world(t["world"]),
            "s": GOALS.index(s) if s in GOALS else -1,
            "sp": round((g.get("strategy") or {}).get("p") or 0, 3),
            "sd": decided(t, "strategy"),
            "tk": target_kind((g.get("target") or {}).get("choice")),
            "td": decided(t, "target"),
            "tg": t["target"],
            "mp": round((g.get("move") or {}).get("p") or 0, 3),
            "mv": t["move"],
            "st": t["stuck_streak"],
            "ev": events(t["events"]),
        })
    last = end["world"] if end else ticks[-1]["world"]
    s = (end or {}).get("summary") or {}
    return {
        "name": run_name(path),
        "model": size,
        "seed": m["seed"],
        "debias": bool(header.get("order_debias")),
        "created": header.get("created"),
        "map": {k: m[k] for k in ("width", "height", "rooms", "corridors", "exit")},
        "rules": {k: m["rules"][k] for k in ("health", "gunner_hp", "brute_hp")},
        "kinds": [int(e["kind"] == "brute") for e in ticks[0]["world"]["enemies"]],
        "ticks": rows,
        "final": world(last),
        "end": {"outcome": end["outcome"] if end else "unfinished", "cause": (end or {}).get("cause"),
                "tick": (end or {}).get("tick", ticks[-1]["tick"]), "kills": s.get("kills"), "hits": s.get("hits"),
                "cleared": s.get("rooms_cleared"), "seen": s.get("rooms_seen")},
    }


def default_pool() -> list[Path]:
    return sorted((ROOT / "demo/output/shooter").glob("*/trace.jsonl")) + sorted((ROOT / "docs").glob("shooter_replay_*.html"))


def build(paths: list[Path], out: Path) -> tuple[Path, dict[str, int]]:
    runs = [r for r in (slim(p) for p in paths) if r]
    counts = {size: sum(r["model"] == size for r in runs) for size in ("1.5B", "3B")}
    if min(counts.values()) < 2:
        raise SystemExit(f"need at least two runs per model, found {counts}")
    html = TEMPLATE.read_text(encoding="utf-8")
    if not _SLOT.search(html):
        raise ValueError(f"{TEMPLATE} has no runs slot")
    body = json.dumps(runs, separators=(",", ":")).replace("<", "\\u003c")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_SLOT.sub(lambda _: f'<script id="runs" type="application/json">{body}</script>', html, count=1),
                   encoding="utf-8")
    return out, counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*", help="trace.jsonl files or replay .html pages (default: the pool above)")
    ap.add_argument("--out", default=str(ROOT / "docs" / "shooter_quad.html"))
    args = ap.parse_args()
    paths = [Path(p) for p in args.sources] or default_pool()
    out, counts = build(paths, Path(args.out))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB): " + ", ".join(f"{n} {k} runs" for k, n in counts.items()))


if __name__ == "__main__":
    main()
