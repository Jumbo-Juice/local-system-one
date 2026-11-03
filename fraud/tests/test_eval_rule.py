import numpy as np

from fraud import costs, eval_ieee as ev, eval_ieee_amount as am, eval_rule as er, final_test as ft, rule
from fraud.baselines import LogReg

N = 590540
C = ev.COSTS


def fit(x_dim=1):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, x_dim))
    y = (rng.uniform(size=2000) < 1 / (1 + np.exp(-(2 * x[:, 0] - 3)))).astype(float)
    return LogReg().fit(x, y)


def rows():
    rng = np.random.default_rng(1)
    return rng.uniform(0.001, 0.9, 1000), rng.lognormal(4, 1.5, 1000)


def test_a_plain_spec_is_exactly_phase_14s_rule():
    q, a = rows()
    m = fit()
    assert er.Spec(m, 10.0).decide(q, a) == am.decide(m.predict(am.logit(q)[:, None]), a, 10.0, C)


def test_the_paced_and_customer_setups_reach_decide_paced_with_their_knobs():
    q, a = rows()
    m = fit()
    p = m.predict(am.logit(q)[:, None])
    assert er.Spec(m, 5.0, gamma=1.0).decide(q, a) == rule.decide_paced(p, a, 5.0, C, 1.0, 0.0)
    assert er.Spec(m, 10.0, flat=10.0).decide(q, a) == rule.decide_paced(p, a, 10.0, C, 0.0, 10.0)


def test_the_amount_fit_reads_the_amount_and_the_header_says_so():
    q, a = rows()
    m = fit(2)
    s = er.Spec(m, 5.0, with_amount=True)
    assert np.allclose(s.p(q, a), m.predict(np.c_[am.logit(q), np.log1p(a)]))
    assert s.header()["fit_features"] == ["logit", "log_amount"]


def test_a_shift_lowers_the_level_and_keeps_the_ranking():
    q, a = rows()
    m = fit()
    base, moved = er.Spec(m, 10.0), er.Spec(m, 10.0, shift=(0.04, 0.03))
    assert moved.p(q, a).sum() < base.p(q, a).sum()
    assert np.array_equal(np.argsort(moved.p(q, a)), np.argsort(base.p(q, a)))
    assert moved.header()["shift"] == [0.04, 0.03] and "shift" not in base.header()


def test_a_flat_price_changes_the_decision_and_never_the_charge():
    a = np.array([50.0])
    w, f = er.Spec(fit(), 1000.0, flat=20.0).decide(np.array([0.9]), a)
    charged = costs.score(C, ["decline"], ["decline"], np.array([False]), a).summary["cost"]
    assert charged == 5.0


def test_the_rule_windows_are_new_and_leave_the_audits_alone():
    new, audit = ft.phase_windows(N, "rule"), ft.phase_windows(N, "audit")
    assert len(new) == 10 and [w.id for w in new[:2]] == ["rule0", "rule1"]
    rows_new = np.concatenate([w.rows for w in new])
    ft.guard(N, rows_new)
    assert not np.isin(rows_new, np.concatenate([w.rows for w in audit + ft.used_windows(N)])).any()
