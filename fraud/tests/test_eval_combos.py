import numpy as np

from fraud import eval_combos as ec


def test_mean_logit_of_one_score_is_that_score_and_keeps_order():
    q = np.array([0.01, 0.2, 0.9])
    assert np.allclose(ec.mean_logit([q]), q)
    a, b = np.array([0.1, 0.5]), np.array([0.3, 0.5])
    assert ec.mean_logit([a, b])[0] < ec.mean_logit([a, b])[1]


def test_nan_stays_nan_in_a_mean():
    assert np.isnan(ec.mean_logit([np.array([np.nan, 0.5])]))[0]


def test_the_heart_line():
    assert ec.is_language_model(None) and ec.is_language_model([0.5, 0.5]) and not ec.is_language_model([0.49, 0.51])


COSTS = {"a": 2000.0, "b": 2100.0, "c": 1900.0, "d": 1800.0}
LAT = {"a": 60.0, "b": 60.0, "c": 60.0, "d": 60.0}
SHARES = {"a": None, "b": None, "c": None, "d": [0.3, 0.7]}


def test_choice_takes_the_cheapest_its_customer_line_and_the_cheapest_of_the_other_kind():
    assert ec.choose(COSTS, LAT, SHARES, {"c", "d"}) == ["c-amt", "c-cust", "a-amt"]


def test_a_setup_that_leans_on_trees_or_is_too_slow_is_not_chosen():
    assert ec.choose(COSTS, LAT, SHARES, {"c", "d"})[0] != "d-amt"
    assert ec.choose(COSTS, {**LAT, "c": 300.0}, SHARES, {"c", "d"}) == ["a-amt", "a-cust"]


def test_nothing_eligible_chooses_nothing():
    assert ec.choose({"d": 1.0}, {"d": 60.0}, {"d": [0.1, 0.9]}) == []


def test_a_mean_adds_its_members_latency_and_the_signal_costs_none():
    info = {"lm-rich": {"latency_ms": {"p90": 60.0}}, "lm-signal": {"latency_ms": {"p90": 62.0}}}
    assert ec.latency_p90(ec.Combo("x", ("lm-rich", "lm-signal")), info) == 122.0
    assert ec.latency_p90(ec.Combo("y", ("lm-rich", "signal-alone"), True), info) == 60.0
