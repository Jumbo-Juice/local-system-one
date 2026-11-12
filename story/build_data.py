"""Build story/data.js from committed results and pinned runs. Standard library only; no model runs.

    python story/build_data.py

The story binds to data.js, so a number on the page can't drift from its source. The fraud numbers
come through slides/build_data.py, the deck's own build, so the story, the deck and the console say
the same thing. Numbers that exist only in prose are in story/manual_data.json, each with its source.
tests/test_story.py rebuilds this and compares.

No IEEE-CIS row is read here: window-level results only. The replays are pinned game runs and the
committed PaySim trace.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BENCH = ROOT / "bench" / "results"
RUNS = ROOT / "runs"

SOURCES = {
    "speed": "bench/results/bench/bench_Qwen2.5-1.5B-Instruct_bfloat16_20260926_050240.json",
    "tournament_small": "bench/results/tournament_compare/tournament_compare_20260926_050656.json",
    "tournament_large": "bench/results/tournament_compare/tournament_compare_20260926_050609.json",
    "grid": "bench/results/demo_compare/demo_compare_20260926_051317.json",
    "escape_a": "bench/results/escape_check/escape_check_20260930-095727.json",
    "escape_b": "bench/results/escape_check/escape_check_20260930-101725.json",
    "flip": "runs/shooter/20260927-002415_classic_qwen2.5-1.5b_seed1.pinned.jsonl",
    "bullet": "runs/shooter/20260927-003620_classic_qwen2.5-3b_seed0.pinned.jsonl",
    "escape": "runs/dungeon/20260930-095727_check_3b_seed0.pinned.jsonl",
}


def read(name: str) -> dict:
    return json.loads((ROOT / SOURCES[name]).read_text(encoding="utf-8"))


def trace(name: str) -> list:
    return [json.loads(line) for line in (ROOT / SOURCES[name]).read_text(encoding="utf-8").splitlines()]


def num(x, digits: int = 4):
    return None if x is None else round(float(x), digits)


def deck():
    """The deck's build, loaded by path: slides/ is a folder of files, not a package."""
    spec = importlib.util.spec_from_file_location("deck_build_data", ROOT / "slides" / "build_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build()


# ---------------------------------------------------------------- the engine

def speed_block() -> dict:
    """One look against writing the answer out as {"choice": "X"}: one decision at a time, the same
    256-token state, on the laptop (the README's Benchmarks table)."""
    j = read("speed")
    look = next(r["per_decision_ms"] for r in j["rows"] if r["state_tokens"] == 256 and r["batch"] == 1 and r["mode"] == "batched")
    write = next(g["per_decision_ms"] for g in j["generation"] if g["style"] == "json" and g["batch"] == 1)
    return {"look_ms": num(look, 1), "write_ms": num(write, 1), "times": num(write / look, 1)}


def tournament_block() -> list:
    """Right answers out of 30 problems: every option in one question, against groups of at most 10."""
    out = []
    for name in ("tournament_small", "tournament_large"):
        for run in read(name)["runs"]:
            groups = [t for t in run["tournaments"] if t["group_size"] == 10]
            if groups:
                out.append({"options": run["n"], "at_once": num(run["full"]["accuracy"]), "tournament": num(groups[0]["accuracy"])})
    return sorted(out, key=lambda r: r["options"])


# ---------------------------------------------------------------- the games

def grid_block() -> dict:
    s = read("grid")["summary"]
    tiers, flat, rand = s["model, tiered goals"], s["model, flat"], s["random (mock), tiered goals"]
    pick = lambda r: {"gems": num(r["gems"], 1), "food": num(r["food_eaten"], 1)}
    return {"tiers": pick(tiers), "flat": pick(flat), "random": pick(rand),
            "gems_times": num(tiers["gems"] / flat["gems"], 1), "food_times": num(tiers["food_eaten"] / flat["food_eaten"], 1)}


def escape_block() -> dict:
    a, b = read("escape_a")["verdict"], read("escape_b")["verdict"]
    return {"escaped": a["escaped"] + b["escaped"], "runs": a["runs"] + b["runs"]}


def flip_block() -> dict:
    """One recorded shoot decision, read in both orders: the first with more than one option in the
    1.5B's first escaped run. `listed` and `reversed` are the model's scores in each order, given
    here in the listed order of the options; `averaged` is what the run acted on."""
    rows = trace("flip")
    for row in rows:
        if row["type"] != "tick":
            continue
        for d in row["decisions"]:
            if d["tier"] == "shoot" and len(d["options"]) > 1 and d.get("orders"):
                return {"tick": row["tick"], "question": d["question"], "state": d["state"], "options": d["options"],
                        "listed": [num(p) for p in d["orders"][0]], "reversed": [num(p) for p in d["orders"][1]],
                        "averaged": [num(p) for p in d["probs"]], "choice": d["choice"], "model": rows[0]["backend"]["model"]}
    raise LookupError("no two-order shoot decision in the trace")


def look_block() -> dict:
    """One recorded look for the opening figure: the first move question in the small model's
    escaped shooter run with four or five options where a single reading, in the listed order,
    leans clearly (0.6 to 0.95) to a safe move that gets closer."""
    for row in trace("flip"):
        if row["type"] != "tick":
            continue
        d = next((d for d in row["decisions"] if d["tier"] == "move"), None)
        if not d or not 4 <= len(d["options"]) <= 5 or not d.get("orders"):
            continue
        scores = d["orders"][0]
        top = scores.index(max(scores))
        if 0.6 <= scores[top] <= 0.95 and "safe; closer" in d["options"][top]:
            return {"tick": row["tick"], "question": d["question"], "options": d["options"], "labels": d["labels"],
                    "scores": [num(p) for p in scores], "choice": top}
    raise LookupError("no clear move decision in the trace")


def game_map(header: dict) -> dict:
    m = header["map"]
    return {"w": m["width"], "h": m["height"], "exit": m["exit"],
            "rooms": [[r["x0"], r["y0"], r["w"], r["h"]] for r in m["rooms"]],
            "names": [r["name"] for r in m["rooms"]],
            "halls": [cell for c in m["corridors"] for cell in c["cells"]],
            "doors": [cell for c in m["corridors"] for door in c["doors"] for cell in door]}


def move_of(tick: dict):
    d = next((d for d in tick["decisions"] if d["tier"] == "move"), None)
    return None if d is None else {"o": d["options"], "p": [num(p, 3) for p in d["probs"]], "c": d["choice"]}


def bullet_block() -> dict:
    """The larger model's pre-registered showcase run of the first shooter (seed 0, whatever the outcome):
    it dies to a gunner. Every tick, with the move question it answered."""
    rows = trace("bullet")
    ticks = [r for r in rows if r["type"] == "tick"]
    frames = []
    for i, t in enumerate(ticks):
        w = t["world"]
        after = ticks[i + 1]["world"] if i + 1 < len(ticks) else rows[-1]["world"]
        frames.append({
            "t": t["tick"], "a": w["agent"]["pos"], "hp": w["agent"]["health"], "hp_after": after["agent"]["health"], "key": w["agent"]["has_key"],
            "e": [[e["pos"][0], e["pos"][1], e["kind"][0], e["hp"], int(e["awake"])] for e in w["enemies"] if e["hp"] > 0],
            "b": [[b["pos"][0], b["pos"][1], b["owner"][0]] for b in w["bullets"]],
            "k": w["items"]["key"], "ev": [e["kind"] for e in t["events"]], "m": move_of(t)})
    warned = [f["t"] for f in frames if f["m"] and "BULLET" in f["m"]["o"][f["m"]["c"]] and any("safe" in o for o in f["m"]["o"])]
    # the moment the story stops on: the warned step the model was surest of
    at = max((f for f in frames if f["t"] in warned), key=lambda f: f["m"]["p"][f["m"]["c"]])
    safe = [o for o in at["m"]["o"] if "safe" in o]
    moment = {"tick": at["t"], "score": at["m"]["p"][at["m"]["c"]], "move": at["m"]["o"][at["m"]["c"]],
              "safe": next((o for o in safe if "on the target" in o or "closer" in o), safe[0])}
    end = rows[-1]
    return {"model": rows[0]["backend"]["model"], "seed": rows[0]["map"]["seed"], "map": game_map(rows[0]), "frames": frames,
            "outcome": end["outcome"], "cause": end["cause"], "end_tick": end["tick"], "kills": end["world"]["agent"]["kills"],
            "hits": sum(f["ev"].count("hit") for f in frames), "warned_ticks": warned, "moment": moment}


def escape_run_block() -> dict:
    """The larger model's seed 0 of the escape check in the rebuilt dungeon (pinned whatever the outcome)."""
    rows = trace("escape")
    ticks = [r for r in rows if r["type"] == "tick"]
    frames = []
    for t in ticks:
        w = t["world"]
        frames.append({"t": t["tick"], "a": w["agent"]["pos"], "hp": w["agent"]["health"], "key": w["agent"]["has_key"],
                       "e": [[g["pos"][0], g["pos"][1], "g", 1, int(g["awake"])] for g in w["ghouls"]],
                       "k": w["items"]["key"], "gems": w["items"]["gems"], "ev": [e["kind"] for e in t["events"]], "m": move_of(t)})
    end = rows[-1]
    return {"model": rows[0]["backend"]["model"], "seed": rows[0]["map"]["seed"], "map": game_map(rows[0]), "frames": frames,
            "outcome": end["outcome"], "end_tick": end["tick"], "hits": sum(f["ev"].count("hit") for f in frames),
            "dashes": sum(f["ev"].count("dash") for f in frames)}


# ---------------------------------------------------------------- fraud, through the deck's build

def fraud_block(d: dict) -> dict:
    f, dev, rule = d["final"], d["dev"], d["rule"]
    keep = lambda s: {k: s[k] for k in ("mean", "ci", "frauds_stopped", "good_declined", "fraud_let_through", "reviews")}
    return {
        "costs": d["costs"],
        "final": {"windows_n": f["windows_n"], "rows": f["rows"], "frauds": f["frauds"], "fraud_usd": f["fraud_usd"], "latency": f["latency"],
                  "setups": {k: {**keep(f["setups"][k]), "windows": f["setups"][k]["windows"]} for k in ("baseline", "rich_amt", "rich_cust", "approve_all")},
                  "diffs": {k: f["diffs"][k] for k in ("rich_amt_baseline", "rich_amt_approve", "rich_cust_baseline", "rich_cust_rich_amt")}},
        "dev": {"windows_n": dev["windows_n"], "rows": dev["rows"],
                "setups": {k: keep(dev["setups"][k]) for k in ("frozen", "phase12", "rich", "logreg")},
                "diffs": {k: dev["diffs"][k] for k in ("frozen_phase12", "rich_phase12", "rich_logreg", "seed1_phase12")}},
        "rule": {"lines": rule["phase14"]["lines"], "rule": rule["phase14"]["rule"], "windows_n": rule["phase14"]["windows_n"], "diff": rule["phase14"]["diff"]},
        "lambda": d["cost"]["lambda"], "rank_corr": f["rank_corr"],
        "split": {k: v for k, v in d["manual"]["split"].items() if not k.startswith("_")},
        "paysim": {**{k: d["paysim"][k] for k in ("model", "window", "n", "model_n", "model_declined", "decision")}, **{k: v for k, v in d["manual"]["paysim"].items() if not k.startswith("_")},
                   "differ": d["manual"]["paysim"]["decisions"] - d["manual"]["paysim"]["matched"]},
    }


def build() -> dict:
    manual = json.loads((HERE / "manual_data.json").read_text(encoding="utf-8"))
    d = deck()
    return {
        "_sources": SOURCES | {f"fraud_{k}": v for k, v in d["_sources"].items()} | {"manual": "story/manual_data.json"},
        "engine": {"speed": speed_block(), "tournament": tournament_block()},
        "games": {"grid": grid_block(), "escape": escape_block(), "flip": flip_block(), "look": look_block()},
        "replays": {"bullet": bullet_block(), "escape": escape_run_block()},
        "fraud": fraud_block(d),
        "manual": manual,
    }


def render() -> str:
    body = json.dumps(build(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"// Generated by story/build_data.py from committed results and pinned runs. Don't edit by hand.\nwindow.STORY = {body};\n"


if __name__ == "__main__":
    (HERE / "data.js").write_text(render(), encoding="utf-8", newline="\n")
    print(f"wrote {HERE / 'data.js'}")
