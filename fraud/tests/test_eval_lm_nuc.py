import math

import numpy as np

from fraud import eval_lm_nuc as lm, finetune_ieee as fi, history as h, ieee_brain
from fraud.brain import OPTIONS

from .test_history import identity, rows


def make():
    t, ident = rows(), identity()
    extra = tuple(k for k in (*ieee_brain.COUNTERS, *ieee_brain.MORE_COUNTERS, *ieee_brain.DAYS) if k not in t.num_names)
    t.num = np.c_[t.num, np.arange(len(t) * len(extra)).reshape(len(t), len(extra)).astype(np.float32) % 7]
    t.num_names = t.num_names + extra
    return t, h.build(t, ident)


def test_the_decline_option_is_found_in_either_order():
    assert lm.order_of("rev")[lm.decline_at("rev")] == OPTIONS[2] and lm.order_of("fwd")[lm.decline_at("fwd")] == OPTIONS[2]
    assert lm.decline_at("rev") == 0 and lm.decline_at("fwd") == 2


def test_the_plain_prompt_is_unchanged_without_history():
    t, _ = make()
    assert ieee_brain.decision(t, 5).state == ieee_brain.describe(t, 5)
    assert ieee_brain.decision(t, 5, None).context == ieee_brain.CONTEXT


def test_the_rich_prompt_adds_to_the_plain_one_and_shows_no_card_ids():
    t, cache = make()
    i = 150
    text = ieee_brain.describe_rich(t, i, cache)
    assert text.startswith(ieee_brain.describe(t, i)) and "Card history:" in text and "More counters:" in text
    assert "card1" not in text and "addr1" not in text and "isFraud" not in text


def test_the_rich_prompt_does_not_change_when_labels_do():
    t, cache = make()
    a = ieee_brain.describe_rich(t, 150, cache)
    t.is_fraud = 1 - t.is_fraud
    assert a == ieee_brain.describe_rich(t, 150, cache)


def test_a_first_time_card_says_so():
    t, cache = make()
    first = int(np.flatnonzero(cache["history"]["card_first"] == 1)[0])
    assert "no earlier payments" in ieee_brain.describe_rich(t, first, cache)


def test_the_reader_of_time_since_last():
    assert ieee_brain._ago(600) == "10 minutes" and ieee_brain._ago(7200) == "2 hours" and ieee_brain._ago(3 * 86400) == "3 days"


def test_seed_changes_the_rows_and_examples_take_history():
    t, cache = make()
    t.is_fraud = (np.arange(len(t)) % 4 == 0).astype(np.int8)
    a, b = fi.training_rows(t, seed=0, full=True), fi.training_rows(t, seed=1, full=True)
    assert not np.array_equal(a, b)
    plain = fi.examples(t, a[:3])
    rich = fi.examples(t, a[:3], 0, cache)
    assert plain[0][1] == rich[0][1] and "Card history:" in rich[0][0].state and "Card history:" not in plain[0][0].state
    assert math.isclose(len(a) / 4, int(t.is_fraud[:int(len(t) * 0.6)].sum()))
