"""Config files select the backend and engine settings. Switching machines = switching files.

Resolution order for the config path: explicit argument, ``SYSTEM_ONE_CONFIG`` env var,
then ``config/default.toml`` in the repo.
"""

from __future__ import annotations

import importlib
import os
import tomllib
from pathlib import Path

from .backends.base import Backend

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.toml"

# kind -> "module:Class". Backends are imported lazily so the mock needs no torch.
BACKENDS = {
    "hf": "system_one.backends.hf:HFBackend",
    "mock": "system_one.backends.mock:MockBackend",
}


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path or os.environ.get("SYSTEM_ONE_CONFIG") or DEFAULT_CONFIG)
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    cfg.setdefault("backend", {})
    cfg.setdefault("engine", {})
    cfg["_path"] = str(path)
    return cfg


def make_backend(cfg: dict) -> Backend:
    options = dict(cfg["backend"])
    kind = options.pop("kind", None)
    if kind not in BACKENDS:
        raise ValueError(f"unknown backend kind {kind!r}; expected one of {sorted(BACKENDS)}")
    module, cls = BACKENDS[kind].split(":")
    return getattr(importlib.import_module(module), cls)(**options)


def make_engine(cfg: dict, backend: Backend | None = None):
    from .engine import Engine

    return Engine(backend or make_backend(cfg), **cfg["engine"])
