"""Step 3: many independent decisions in one forward pass vs a sequential baseline."""

import pytest

from system_one import Decision, Engine
from system_one.backends.mock import MockBackend

MOVES = ("move north", "move south", "move east", "move west", "stay")


def _decisions(n: int) -> list[Decision]:
    # Different state lengths, so the batch really needs padding.
    return [
        Decision("Which move brings you closer to the target?", MOVES,
                 state=f"Agent {i}. " + "The corridor continues. " * (i % 5) + f"Target is {i % 7} cells east.")
        for i in range(n)
    ]


def test_mock_batched_equals_sequential_and_uses_one_pass():
    be = MockBackend()
    eng = Engine(be)
    ds = _decisions(12)
    before = be.forward_calls
    batched = eng.decide_batch(ds)
    assert be.forward_calls - before == 1
    before = be.forward_calls
    sequential = eng.decide_sequential(ds)
    assert be.forward_calls - before == 12
    for b, s in zip(batched, sequential):
        assert b.choice == s.choice and b.probs == s.probs and b.outside_mass == s.outside_mass


def test_mock_chunking_by_max_batch():
    be = MockBackend(max_batch=5)
    eng = Engine(be)
    ds = _decisions(12)
    before = be.forward_calls
    batched = eng.decide_batch(ds)
    assert be.forward_calls - before == 3  # ceil(12 / 5)
    assert [r.choice for r in batched] == [r.choice for r in eng.decide_sequential(ds)]


def test_results_keep_input_order_with_mixed_methods():
    be = MockBackend()
    eng = Engine(be, answer="text")
    ds = [Decision("Sky?", ("blue", "red")), Decision("Move?", MOVES), Decision("Size?", ("big", "small"))]
    out = eng.decide_batch(ds)
    assert [r.decision for r in out] == ds
    assert [r.method for r in out] == ["single_token", "multi_token", "single_token"]


@pytest.mark.model
def test_hf_batched_matches_sequential(hf_backend):
    from bench.eval_set import eval_items

    eng = Engine(hf_backend)
    ds = [it.decision for it in eval_items()]
    batched, sequential = eng.decide_batch(ds), eng.decide_sequential(ds)
    exact = hf_backend.dtype_name == "float32"
    # Observed on Arc 140V: float32 max |dp| < 1e-4; bfloat16 max |dp| ~0.05-0.07 with flips
    # only on near-ties. See docs/research.md -> Observed -> Batching.
    tol, tie = (1e-3, 0.0) if exact else (0.12, 0.15)
    for b, s in zip(batched, sequential):
        assert max(abs(x - y) for x, y in zip(b.probs, s.probs)) < tol
        top2 = sorted(s.probs)[-2:]
        if top2[1] - top2[0] > tie:
            assert b.choice == s.choice
