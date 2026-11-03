import copy

import numpy as np

from fraud import costs, reprice_paysim as rp


def row(by, q, truth, amount, wanted="approve", fallback="approve", weight=2.0):
    scores = {"fraud": q} if by == "logreg" else {"approve": 1 - q, "review": 0.0, "decline": q} if by == "model" else None
    return {"type": "tx", "by": by, "scores": scores, "truth": truth, "tx": {"amount": amount}, "wanted": wanted, "fallback": fallback, "weight": weight}


def window(rows):
    return [{"type": "header"}, *rows, {"type": "end", "summary": {}}]


def test_the_new_price_changes_only_the_review_fee():
    assert costs.DEFAULT == costs.Costs(0.10, 10.0, 10.0, 0.05)
    assert costs.PHASES_1_10 == costs.Costs(0.10, 10.0, 50.0, 0.05)


def test_same_decisions_differ_by_40_per_review_done():
    recs = window([row("model", 0.5, 1, 900.0, "review", "decline")] + [row("filter", 0.0, 0, 50.0)] * 39)
    old, new = rp.redecide(recs, costs.PHASES_1_10), rp.redecide(recs, costs.DEFAULT)
    assert old.summary["reviews_done"] == new.summary["reviews_done"] == 2 - 1
    assert old.summary["cost"] - new.summary["cost"] == 40.0
    assert old.final == new.final


def test_thresholds_decide_model_rows_but_not_filter_or_rule_rows():
    recs = window([row("model", 0.9, 1, 100.0), row("logreg", 0.3, 0, 100.0), row("filter", 0.0, 0, 100.0), row("rules", 0.0, 0, 100.0, "decline", "decline")] * 5)
    s = rp.redecide(recs, costs.DEFAULT, (0.2, 0.8))
    assert s.final[:4] == ["decline", "review", "approve", "decline"]


def test_a_spent_review_budget_uses_the_recorded_fallback():
    recs = window([row("model", 0.5, 1, 100.0, fallback="decline")] * 20 + [row("model", 0.5, 0, 100.0, fallback="approve")])
    s = rp.redecide(recs, costs.DEFAULT, (0.2, 0.8))
    assert s.final[0] == "review" and s.final[1] == "decline" and s.final[-1] == "approve"


def test_redecide_leaves_the_trace_alone():
    recs = window([row("model", 0.5, 1, 100.0)] * 20)
    before = copy.deepcopy(recs)
    rp.redecide(recs, costs.DEFAULT, (0.2, 0.8))
    assert recs == before


def view(q, y, a):
    n = len(q)
    return {"scores": [np.array(q)], "truth": [np.array(y, bool)], "amount": [np.array(a, float)], "filt": [np.zeros(n, bool)], "fallbacks": [["approve"] * n]}


def test_pick_lowers_cost_and_keeps_the_first_of_equal_costs():
    v = view([0.9, 0.1], [1, 0], [1000.0, 100.0])
    assert rp.pick(v, (0.5, 0.6), (0.7, 0.8), costs.DEFAULT) == (0.5, 0.7, 0.0)


def test_pick_skips_a_decline_line_under_the_review_line():
    v = view([0.1] * 4, [0] * 4, [100.0] * 4)
    assert rp.pick(v, (0.9,), (0.5, 0.95), costs.DEFAULT)[:2] == (0.9, 0.95)


def test_a_cheaper_review_can_move_the_chosen_lines():
    # Forty rows, two reviews allowed. A review saves 30 on the good rows it would otherwise decline; at 50 it does not pay, at 10 it does.
    q = [0.6] * 2 + [0.0] * 38
    v = view(q, [0] * 40, [300.0] * 40)
    v["fallbacks"] = [["decline"] * 2 + ["approve"] * 38]
    assert rp.pick(v, (0.5,), (0.5, 0.9), costs.PHASES_1_10)[:2] == (0.5, 0.5)
    assert rp.pick(v, (0.5,), (0.5, 0.9), costs.Costs(0.10, 10.0, 10.0, 0.05))[:2] == (0.5, 0.9)


def test_split_adds_up_to_the_cost():
    recs = window([row("model", 0.9, 1, 500.0), row("model", 0.9, 0, 500.0), row("model", 0.5, 1, 500.0), row("model", 0.1, 1, 500.0)] * 10)
    s = rp.redecide(recs, costs.DEFAULT, (0.2, 0.8))
    m = rp.split(s, recs)
    assert abs(sum(m.values()) - s.summary["cost"]) < 1e-9
    assert m["review_fees"] == 2 * 10.0


def test_bars_count_ties_only_where_the_rule_does():
    by = {"t": {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([1.0, 3.0, 2.0])}}
    assert rp.bar_value(("t", "a", "b", 2, "strict"), by) == 1
    assert rp.bar_value(("t", "a", "b", 2, "at most"), by) == 2


def test_every_phase_is_named_and_results_stay_out_of_the_console():
    from fraud.viewer import server
    assert set(rp.RESULT_FILES) == {rp.PHASE4, rp.PHASE5, rp.PHASE6, rp.PHASE7, rp.PHASE8, rp.PHASE9}
    assert rp.RESULTS.name not in server.EVAL_DIRS
    assert all(b[0] in rp.TEST_SETS and {b[1], b[2]} <= set(rp.TEST_SETS[b[0]]["setups"]) for b in rp.BARS.values())
    assert all({a, b} <= set(rp.TEST_SETS[k]["setups"]) for k, pairs in rp.PAIRS.items() for a, b in pairs)
