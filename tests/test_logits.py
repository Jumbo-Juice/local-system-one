"""Step 1: one forward pass, next-token logits read directly (no text generation)."""

import numpy as np
import pytest

from system_one.backends.base import Backend
from system_one.backends.mock import MockBackend


def test_mock_logits_shape_and_determinism():
    be = MockBackend()
    batch = [be.encode("hello world"), be.encode("a b c")]
    a, b = be.next_token_logits(batch), be.next_token_logits(batch)
    assert a.shape == (2, be.vocab_size) and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert be.forward_calls == 2  # one batched pass per call


def test_mock_tokenizer_round_trip_and_specials():
    be = MockBackend()
    text = be.render_chat("sys", 'user says {"x": 1}') + '{"choice": "'
    assert be.decode(be.encode(text)) == text
    assert be.encode("<|assistant|>") == [3]


def test_default_scores_are_full_vocab_log_softmax():
    be = MockBackend()
    seq = be.encode("some prompt")
    row = be.next_token_logits([seq])[0].astype(np.float64)
    logz = np.log(np.exp(row - row.max()).sum()) + row.max()
    s = be.next_token_scores([seq], [[5, 9, 11]])[0]
    np.testing.assert_allclose(s.logprobs, row[[5, 9, 11]] - logz)
    assert s.top_id == int(row.argmax())


@pytest.mark.model
def test_hf_logits_are_full_vocab_and_finite(hf_backend):
    be = hf_backend
    batch = [be.encode(be.render_chat("Answer briefly.", "Capital of France? One word.")),
             be.encode("The quick brown fox jumps over the lazy")]
    logits = be.next_token_logits(batch)
    assert logits.shape == (2, be.vocab_size)
    assert np.isfinite(logits).all()
    assert be.decode([int(logits[0].argmax())]).strip() == "Paris"
    assert be.decode([int(logits[1].argmax())]).strip().lower() == "dog"


@pytest.mark.model
def test_hf_left_padding_matches_unpadded(hf_backend, logit_tolerance):
    be = hf_backend
    seqs = [be.encode("One two three four five six seven eight nine"), be.encode("Hi"), be.encode("x y z")]
    batched = be.next_token_logits(seqs)
    for i, s in enumerate(seqs):
        alone = be.next_token_logits([s])[0]
        assert np.abs(alone - batched[i]).max() < logit_tolerance
        assert alone.argmax() == batched[i].argmax()


@pytest.mark.model
def test_hf_scores_agree_with_logits(hf_backend):
    be = hf_backend
    seq = be.encode(be.render_chat("Be brief.", "Say yes or no."))
    row = be.next_token_logits([seq])[0].astype(np.float64)
    logz = np.log(np.exp(row - row.max()).sum()) + row.max()
    cands = [int(row.argmax()), 100, 2000]
    s = be.next_token_scores([seq], [cands])[0]
    np.testing.assert_allclose(s.logprobs, row[cands] - logz, atol=1e-3)
    assert s.top_id == cands[0]


@pytest.mark.model
def test_hf_continuation_matches_generic_implementation(hf_backend, logit_tolerance):
    be = hf_backend
    prefix = be.encode(be.render_chat("Be brief.", "Name a colour.")) + be.encode('{"choice": "')
    conts = [be.encode('blue"}'), be.encode('dark green"}')]
    fast = be.continuation_logprobs([prefix, prefix], conts)
    slow = Backend.continuation_logprobs(be, [prefix, prefix], conts)
    for f, s in zip(fast, slow):
        assert f.shape == s.shape
        assert np.abs(f - s).max() < logit_tolerance
