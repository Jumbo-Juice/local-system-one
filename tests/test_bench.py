"""Step 4: benchmark helpers (the benchmark itself is run by hand, see README)."""

import pytest

from bench.bench import _stats, make_decisions, summary_markdown
from system_one import Engine
from system_one.backends.mock import MockBackend


def test_make_decisions_are_distinct_and_long_enough():
    eng = Engine(MockBackend())
    ds = make_decisions(eng, 6, state_tokens=80)
    assert len({d.state for d in ds}) == 6
    assert all(len(eng.backend.encode(d.state)) >= 80 for d in ds)


def test_stats_per_decision_and_spread():
    s = _stats([0.2, 0.4, 0.3], n=4)
    assert s["median_s"] == 0.3 and s["min_s"] == 0.2 and s["max_s"] == 0.4
    assert s["per_decision_ms"] == pytest.approx(75.0)
    assert s["decisions_per_s"] == pytest.approx(4 / 0.3)


def test_summary_markdown_lists_every_row():
    info = {"host": {"cpu": "c", "ram_gb": 1, "os": "o"}, "backend": {"model": "m"}, "started": "t"}
    rows = [{"prompt_tokens_mean": 100, "batch": b, "mode": "batched", **_stats([0.1, 0.2], b)} for b in (1, 2)]
    md = summary_markdown(info, rows, [])
    assert md.count("| batched |") == 2
