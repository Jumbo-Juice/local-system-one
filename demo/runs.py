"""Where every captured run goes: one flat pool per game, one .jsonl trace per run.

    runs/shooter/20260929-133306_qwen2.5-3b_seed0.jsonl         python -m demo.shooter.capture
    runs/shooter/20260929-134450_eval-ammo_1.5b_seed3.jsonl     python -m bench.shooter_eval
    runs/dungeon/20260926-163600_enemy-aware_qwen2.5-3b_seed7.jsonl

A name starts with the time the run (or its eval) started, so sorting by name sorts by time. The
Master Viewer (python -m viewer) plays these pools. Runs named *.pinned.jsonl are committed (see
runs/.gitignore); every other run stays local. A new game demo gets its own pool: runs/<game>/.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def stamp() -> str:
    """The current local time as a run-name prefix: 20260929-133306."""
    return time.strftime("%Y%m%d-%H%M%S")


def model_tag(info: dict) -> str:
    """A short model name for run names: 'Qwen/Qwen2.5-3B-Instruct' -> 'qwen2.5-3b', mock -> 'mock'."""
    if info.get("kind") == "mock":
        return "mock"
    name = str(info.get("model") or info.get("kind") or "model").replace("\\", "/").split("/")[-1].lower()
    name = re.sub(r"\.gguf$", "", name)
    name = re.sub(r"-instruct.*$", "", name)
    return re.sub(r"[^a-z0-9.]+", "-", name).strip("-") or "model"


def pool(game: str) -> Path:
    path = RUNS / game
    path.mkdir(parents=True, exist_ok=True)
    return path


def trace_path(game: str, label: str, when: str | None = None) -> Path:
    """runs/<game>/<when>_<label>.jsonl; ``when`` defaults to now."""
    return pool(game) / f"{when or stamp()}_{label}.jsonl"


def to_jsonl(records: list[dict]) -> str:
    return "".join(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n" for r in records)


def write_run(game: str, label: str, records: list[dict], when: str | None = None) -> Path:
    path = trace_path(game, label, when)
    path.write_text(to_jsonl(records), encoding="utf-8")
    return path


def finished(path: Path) -> dict | None:
    """The end record of a finished trace, else None (missing, empty or cut off)."""
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    end = json.loads(lines[-1]) if lines else {}
    return end if end.get("type") == "end" else None
