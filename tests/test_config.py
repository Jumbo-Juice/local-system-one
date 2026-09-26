"""Switching model/runtime is a config change: every shipped config must parse and resolve."""

from pathlib import Path

import pytest

from system_one import Decision, load_config, make_backend, make_engine
from system_one.config import BACKENDS, REPO_ROOT

CONFIGS = sorted((REPO_ROOT / "config").glob("*.toml"))


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_config_parses_and_names_a_known_backend(path: Path):
    cfg = load_config(path)
    assert cfg["backend"]["kind"] in BACKENDS
    assert set(cfg["engine"]) <= {"answer", "answer_template", "multi_token", "system_prompt", "prompt_order"}


def test_mock_config_builds_a_working_engine():
    eng = make_engine(load_config(REPO_ROOT / "config" / "mock.toml"))
    r = eng.decide(Decision("pick", ("a", "b")))
    assert r.choice in ("a", "b")


def test_env_var_selects_the_config(monkeypatch):
    monkeypatch.setenv("SYSTEM_ONE_CONFIG", str(REPO_ROOT / "config" / "mock.toml"))
    assert load_config()["backend"]["kind"] == "mock"


def test_unknown_backend_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown backend kind"):
        make_backend({"backend": {"kind": "nope"}, "engine": {}})
