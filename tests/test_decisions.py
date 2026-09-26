"""Step 2: single-token choice decisions, token mapping validation, multi-token fallback."""

import math

import numpy as np
import pytest

from system_one import Decision, Engine, TokenMappingError
from system_one.backends.base import Backend
from system_one.backends.mock import MockBackend


def scripted(values: dict[str, float], base: float = 0.0) -> MockBackend:
    """Mock whose next-token logits are ``base`` everywhere except the given pieces."""
    be = MockBackend()

    def fn(ids):
        v = np.full(be.vocab_size, base, dtype=np.float32)
        for piece, val in values.items():
            v[be.token_id(piece)] = val
        return v

    be.logit_fn = fn
    return be


def test_probs_are_softmax_over_allowed_tokens_and_outside_mass():
    be = scripted({"A": 5.0, "B": 3.0, "C": 1.0, "Z": 9.0})
    r = Engine(be).decide(Decision("pick", ("x", "y", "z"), state="s"))
    expected = np.exp([5.0, 3.0, 1.0]) / np.exp([5.0, 3.0, 1.0]).sum()
    np.testing.assert_allclose(r.probs, expected, rtol=1e-6)
    assert r.choice == "x" and r.index == 0 and r.labels == ["A", "B", "C"]
    z = math.exp(5) + math.exp(3) + math.exp(1) + math.exp(9) + (be.vocab_size - 4)
    inside = (math.exp(5) + math.exp(3) + math.exp(1)) / z
    assert r.outside_mass == pytest.approx(1 - inside, rel=1e-6)
    assert r.top_token == "Z"  # the unconstrained argmax is outside the allowed set
    assert r.method == "single_token"


def test_choice_is_argmax_of_allowed_tokens():
    be = scripted({"A": 1.0, "B": 4.0, "C": 2.0})
    r = Engine(be).decide(Decision("pick", ("x", "y", "z")))
    assert r.choice == "y" and r.prob == max(r.probs)
    assert [o for o, _ in r.ranked()] == ["y", "z", "x"]


def test_prompt_lists_labels_and_ends_with_prefill():
    be = MockBackend()
    eng = Engine(be)
    d = Decision("Which way?", ("left", "right"), state="S", context="Goal: G")
    text = eng.render(d, eng.labels_for(2))
    assert "State:\nS" in text and "Context:\nGoal: G" in text
    assert "A: left\nB: right" in text
    assert text.endswith('<|assistant|>{"choice": "')


def test_more_than_26_options_use_two_letter_labels():
    be = MockBackend()
    eng = Engine(be)
    labels = eng.labels_for(30)
    assert labels[:3] == ["AA", "AB", "AC"] and len(set(labels)) == 30
    r = eng.decide(Decision("pick", tuple(f"o{i}" for i in range(30))))
    assert len(r.probs) == 30 and sum(r.probs) == pytest.approx(1.0)


def test_too_many_options_is_rejected():
    with pytest.raises(TokenMappingError, match="tournament"):
        Engine(MockBackend()).decide(Decision("pick", tuple(str(i) for i in range(256))))


def test_duplicate_options_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        Decision("pick", ("a", "a"))


def test_label_that_merges_with_prefill_fails_clearly():
    # The mock tokenizer joins letters into one piece, so "LabelA" is one token: invalid.
    eng = Engine(MockBackend(), answer_template="Label<label>")
    with pytest.raises(TokenMappingError, match="not one token"):
        eng.decide(Decision("pick", ("x", "y")))


def test_trailing_space_in_template_moves_into_the_label_token():
    be = scripted({" A": 2.0, " B": 1.0})
    eng = Engine(be, answer_template="Answer: <label>")
    r = eng.decide(Decision("pick", ("x", "y")))
    assert eng.render(r.decision, r.labels).endswith("Answer:")
    assert r.choice == "x" and r.probs[0] == pytest.approx(1 / (1 + math.exp(-1)))


def test_text_mode_uses_first_token_when_distinct():
    be = scripted({"blue": 3.0, "red": 1.0})
    r = Engine(be, answer="text").decide(Decision("Sky colour?", ("blue", "red", "yellow")))
    assert r.method == "single_token" and r.choice == "blue"


def test_shared_first_token_raises_when_fallback_disabled():
    eng = Engine(MockBackend(), answer="text", multi_token="never")
    with pytest.raises(TokenMappingError, match="'move north' and 'move south' both map"):
        eng.decide(Decision("Which move?", ("move north", "move south")))


def test_multi_token_fallback_scores_full_sequences():
    be = scripted({"move": 4.0, " north": 1.0, " south": 2.5, '"': 3.0, "}": 3.0})
    eng = Engine(be, answer="text")
    d = Decision("Which move?", ("move north", "move south"))
    r = eng.decide(d)
    assert r.method == "multi_token" and r.choice == "move south"
    # Recompute the sequence log-probs by hand with the generic base-class path.
    prep = eng.prepare(d)
    seq_lp = [Backend.continuation_logprobs(be, [prep.ids], [c])[0].sum() for c in prep.continuations]
    expected = np.exp(np.array(seq_lp) - max(seq_lp))
    np.testing.assert_allclose(r.probs, expected / expected.sum(), rtol=1e-6)
    assert r.outside_mass == pytest.approx(1 - np.exp(seq_lp).sum(), rel=1e-6)
    assert eng.last_stats["multi_token"] == 1


def test_validation_is_cached_per_prompt_tail():
    be = MockBackend()
    eng = Engine(be)
    eng.decide_batch([Decision("q1", ("a", "b")), Decision("q2", ("c", "d"), state="other")])
    assert len(eng._cache) == 1  # same template tail and labels -> validated once


@pytest.mark.model
def test_hf_labels_are_distinct_single_tokens(hf_backend):
    eng = Engine(hf_backend)
    d = Decision("pick", tuple(f"option {i}" for i in range(40)))
    prep = eng.prepare(d)
    assert len(set(prep.tokens)) == 40
    assert [hf_backend.decode([t]) for t in prep.tokens] == prep.labels


@pytest.mark.model
def test_hf_colon_prefill_merges_single_letters(hf_backend):
    # Observed with the Qwen2.5 tokenizer: "choice_label:A" tokenises as [..., ":A"].
    with pytest.raises(TokenMappingError):
        Engine(hf_backend, answer_template="choice_label:<label>").decide(Decision("pick", ("x", "y")))


@pytest.mark.model
def test_hf_simple_decision(hf_backend):
    r = Engine(hf_backend).decide(Decision("What colour is the sky on a clear day?", ("red", "blue", "yellow")))
    assert r.choice == "blue" and r.prob > 0.5 and 0 <= r.outside_mass < 0.5


@pytest.mark.model
def test_hf_text_mode_multi_token(hf_backend):
    # Checks the mechanism, not the model's judgement: this item is a near-tie for some models
    # (Observed: Qwen2.5-1.5B gives 0.501 / 0.499).
    d = Decision("Which move brings you closer to the target?", ("move north", "move south"),
                 state="The target is 4 cells north of you.")
    eng = Engine(hf_backend, answer="text")
    r = eng.decide(d)
    assert r.method == "multi_token"
    prep = eng.prepare(d)
    seq = [lp.sum() for lp in hf_backend.continuation_logprobs([prep.ids] * 2, prep.continuations)]
    expected = np.exp(np.array(seq) - max(seq))
    np.testing.assert_allclose(r.probs, expected / expected.sum(), atol=1e-6)
    assert r.index == int(np.argmax(seq))


def test_prefix_len_covers_only_the_static_part():
    be = MockBackend()
    d = Decision("Which way?", ("left", "right"), state="S1", context="goal")
    for order, must_include, must_exclude in (("state_first", "<|user|>", "State:"),
                                              ("question_first", "B: right", "State:"),
                                              ("options_first", "B: right", "Question:")):
        eng = Engine(be, prompt_order=order)
        prep = eng.prepare(d)
        prefix = be.decode(prep.ids[:prep.prefix_len])
        assert must_include in prefix and must_exclude not in prefix, order
        # the same static part gives the same prefix for a different state
        other = eng.prepare(Decision("Which way?", ("left", "right"), state="a much longer state S2"))
        assert other.ids[:other.prefix_len] == prep.ids[:prep.prefix_len]


def test_engine_passes_prefix_lens_to_the_backend():
    seen = {}

    class Recording(MockBackend):
        def next_token_scores(self, batch, candidates, prefix_lens=None):
            seen["prefix_lens"] = prefix_lens
            return super().next_token_scores(batch, candidates, prefix_lens)

    eng = Engine(Recording())
    preps = [eng.prepare(Decision("q", ("a", "b"), state=s)) for s in ("x", "yy")]
    eng.decide_batch([p.decision for p in preps])
    assert seen["prefix_lens"] == [p.prefix_len for p in preps] and all(n > 0 for n in seen["prefix_lens"])
