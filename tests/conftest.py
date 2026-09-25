"""Shared fixtures.

Tests marked ``model`` load a real model through the HF backend. They skip when torch or
the weights are missing. Choose the model with SYSTEM_ONE_TEST_MODEL (default: the small
Qwen2.5-0.5B-Instruct) and the dtype with SYSTEM_ONE_TEST_DTYPE (default: backend auto).
Run only the fast tests with: pytest -m "not model"
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never reach the network from tests

TEST_MODEL = os.environ.get("SYSTEM_ONE_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
TEST_DTYPE = os.environ.get("SYSTEM_ONE_TEST_DTYPE", "auto")


@pytest.fixture(scope="session")
def hf_backend():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from system_one.backends.hf import HFBackend

    try:
        return HFBackend(TEST_MODEL, dtype=TEST_DTYPE)
    except OSError as exc:  # weights not in the local cache
        pytest.skip(f"model {TEST_MODEL} not available offline: {exc}")


@pytest.fixture(scope="session")
def logit_tolerance(hf_backend):
    """Max |logit| difference that counts as 'the same' for this dtype.

    bf16 kernels with different batch shapes drift by up to ~0.3 logits (Observed on Arc 140V);
    float32 drifts by ~1e-4.
    """
    return 1e-3 if hf_backend.dtype_name == "float32" else 0.75
