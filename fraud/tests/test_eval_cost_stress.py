import numpy as np

from fraud import costs, eval_cost_stress as st, eval_ieee as ev, eval_ieee_amount as am

C = ev.COSTS


def test_a_flat_price_adds_to_friction_only_when_a_good_customer_is_declined():
    c = costs.Costs(0.10, 1.0, 10.0, 0.05, 20.0)
    assert costs.unit_cost(c, "decline", False, 100.0) == 30.0
    assert costs.unit_cost(c, "decline", True, 100.0) == 0.0
    assert costs.unit_cost(c, "approve", False, 100.0) == 0.0 and costs.unit_cost(c, "review", False, 100.0) == 10.0


def test_older_results_keep_the_same_costs_dict():
    assert C.as_dict() == {"friction_rate": 0.1, "friction_min": 1.0, "review_fee": 10.0, "review_share": 0.05}
    assert costs.Costs(0.1, 1.0, 10.0, 0.05, 5.0).as_dict()["decline_flat"] == 5.0


def test_the_rule_counts_a_flat_price_in_its_expected_cost():
    p, a = np.array([0.2]), np.array([50.0])  # approve 10; decline 0.8 * 5 = 4, or 0.8 * 25 = 20 with a flat 20
    assert am.decide(p, a, 1000, C)[0] == ["decline"]
    assert am.decide(p, a, 1000, costs.Costs(0.1, 1.0, 10.0, 0.05, 20.0))[0] == ["approve"]


def test_every_family_holds_phase_14s_costs_once_where_it_can():
    by = {}
    for family, _, value, c in st.cost_sets():
        by.setdefault(family, []).append((value, c))
    for family in ("friction rate", "friction minimum", "review fee", "review share"):
        assert [c for v, c in by[family] if v == st.base_value(family)] == [C]
    assert [c for v, c in by["flat per good customer declined"] if v == 0.0] == [C]


def test_stops_reads_the_first_value_the_rule_no_longer_wins():
    row = lambda v, wins, dist: {"value": v, "x": {"rule_wins": wins, "distinguishable": dist}}
    rows = [row(0, True, True), row(10, True, False), row(20, False, False), row(30, True, True)]
    assert st.stops(rows, "x") == {"first_not_lower": 20, "first_not_distinguishably_lower": 10}
    assert st.stops(rows[:1], "x") == {"first_not_lower": None, "first_not_distinguishably_lower": None}


def test_line_marks_a_distinguishable_win_only_when_the_interval_excludes_zero():
    base = {"good_declined": 1, "frauds_stopped": 1, "fraud_usd_stopped": 1.0}
    low = {**base, "per": np.array([1.0, 2.0, 3.0, 4.0])}
    high = {**base, "per": np.array([11.0, 12.0, 14.0, 13.0])}
    x = st.line(low, high)
    assert x["rule_wins"] and x["distinguishable"] and x["diff"] == -10.0
    same = st.line(low, low)
    assert not same["rule_wins"] and not same["distinguishable"]
