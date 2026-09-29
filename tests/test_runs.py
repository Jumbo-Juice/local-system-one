"""The run pools: every capture and eval writes one trace to runs/<game>/."""

import json
import re

from demo import runs


def test_model_tag_is_short_and_safe():
    assert runs.model_tag({"kind": "hf", "model": "Qwen/Qwen2.5-3B-Instruct"}) == "qwen2.5-3b"
    assert runs.model_tag({"kind": "hf", "model": "Qwen/Qwen2.5-1.5B-Instruct"}) == "qwen2.5-1.5b"
    assert runs.model_tag({"kind": "llamacpp", "model": "C:\\models\\Qwen2.5-7B-Instruct-Q4_K_M.gguf"}) == "qwen2.5-7b"
    assert runs.model_tag({"kind": "mock", "model": "mock (seeded random logits)"}) == "mock"


def test_runs_go_flat_into_the_game_pool_named_by_time(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "RUNS", tmp_path)
    path = runs.write_run("shooter", "qwen2.5-3b_seed0", [{"type": "header"}, {"type": "end", "outcome": "died"}])
    assert path.parent == tmp_path / "shooter"
    assert re.fullmatch(r"\d{8}-\d{6}_qwen2\.5-3b_seed0\.jsonl", path.name)
    assert [json.loads(x)["type"] for x in path.read_text(encoding="utf-8").splitlines()] == ["header", "end"]
    assert runs.trace_path("dungeon", "eval_random_seed3", "20260926-163151").name == "20260926-163151_eval_random_seed3.jsonl"


def test_only_a_trace_with_an_end_record_counts_as_finished(tmp_path):
    done, cut = tmp_path / "done.jsonl", tmp_path / "cut.jsonl"
    done.write_text(runs.to_jsonl([{"type": "header"}, {"type": "tick"}, {"type": "end", "tick": 1}]), encoding="utf-8")
    cut.write_text(runs.to_jsonl([{"type": "header"}, {"type": "tick"}]), encoding="utf-8")
    assert runs.finished(done) == {"type": "end", "tick": 1}
    assert runs.finished(cut) is None and runs.finished(tmp_path / "missing.jsonl") is None


def test_every_pinned_run_is_a_finished_trace_of_its_pool():
    pinned = sorted(runs.RUNS.glob("*/*.pinned.jsonl"))
    assert pinned, "the pinned showcase runs are committed in runs/"
    for path in pinned:
        header = json.loads(path.open(encoding="utf-8").readline())
        assert header["type"] == "header" and header["scenario"] == path.parent.name, path
        assert runs.finished(path), path
