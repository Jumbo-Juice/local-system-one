"""Where fraud runs go: fraud/runs/<time>_<label>.jsonl, one trace per run.

    fraud/runs/20261003-120000_hybrid_qwen2.5-1.5b_dev0.jsonl         python -m fraud.capture
    fraud/runs/20261003-130000_eval_hybrid_qwen2.5-1.5b_test07.jsonl  python -m fraud.eval

A name starts with the time the run (or its eval) started, so sorting by name sorts by time. Only
runs named *.pinned.jsonl are committed (fraud/runs/.gitignore); each is listed in README.md there.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"


def stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def model_tag(info: dict) -> str:
    """'Qwen/Qwen2.5-3B-Instruct' -> 'qwen2.5-3b'; the mock backend -> 'mock'."""
    if info.get("kind") == "mock":
        return "mock"
    name = str(info.get("model") or info.get("kind") or "model").replace("\\", "/").split("/")[-1].lower()
    name = re.sub(r"-instruct.*$", "", re.sub(r"\.gguf$", "", name))
    return re.sub(r"[^a-z0-9.]+", "-", name).strip("-") or "model"


def to_jsonl(records: list[dict]) -> str:
    return "".join(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n" for r in records)


def trace_path(label: str, when: str | None = None, root: Path = RUNS) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{when or stamp()}_{label}.jsonl"


def write_run(label: str, records: list[dict], when: str | None = None, root: Path = RUNS) -> Path:
    path = trace_path(label, when, root)
    path.write_text(to_jsonl(records), encoding="utf-8")
    return path


def read_run(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def finished(path: Path) -> dict | None:
    """The end record of a finished trace, else None."""
    if not Path(path).exists():
        return None
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    end = json.loads(lines[-1]) if lines else {}
    return end if end.get("type") == "end" else None
