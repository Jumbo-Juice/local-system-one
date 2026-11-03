import numpy as np

from fraud import costs, eval_ieee as ev, eval_ieee_amount as am, rule

C = ev.COSTS


def test_gamma_0_and_flat_0_is_exactly_phase_14s_rule():
    rng = np.random.default_rng(0)
    p, a = rng.uniform(0, 1, 500), rng.lognormal(4, 1.5, 500)
    assert rule.decide_paced(p, a, 10, C) == am.decide(p, a, 10, C)


def test_a_flat_price_only_makes_declines_rarer():
    rng = np.random.default_rng(1)
    p, a = rng.uniform(0, 1, 2000), rng.lognormal(4, 1.5, 2000)
    base, _ = rule.decide_paced(p, a, 10, C)
    dear, _ = rule.decide_paced(p, a, 10, C, flat=20.0)
    assert sum(x == "decline" for x in dear) < sum(x == "decline" for x in base)


def test_the_flat_price_does_not_reach_the_scorer():
    s = costs.score(C, ["decline"], ["decline"], np.array([False]), np.array([100.0]))
    assert s.summary["cost"] == 10.0


def test_pacing_spends_no_more_than_the_budget_and_lowers_the_price_when_reviews_are_plenty():
    p, a = np.full(1000, 0.3), np.full(1000, 300.0)  # approve 90, decline 21
    w, _ = rule.decide_paced(p, a, 5, C, gamma=2.0)
    assert sum(x == "review" for x in w) <= 50
    assert np.allclose([rule.pace(50, 1000, 0.05), rule.pace(50, 500, 0.05), rule.pace(10, 1000, 0.05)], [1.0, 2.0, 0.2])


def test_pacing_holds_reviews_back_for_later_rows():
    p, a = np.full(1000, 0.3), np.full(1000, 150.0)  # approve 45, decline 10.5: a review is worth it only if fee + price < 10.5
    y = np.zeros(1000, bool)
    final = lambda g: costs.score(C, *rule.decide_paced(p, a, 0.4, C, gamma=g), y, a).final
    early = lambda w: sum(x == "review" for x in w[:500])
    assert early(final(0.0)) == 50 > early(final(4.0))  # first come, first served spends all 50 in the first 50 rows


def test_recent_prior_recovers_a_lower_share_when_only_the_share_moved():
    rng = np.random.default_rng(2)
    # Well-calibrated p at 4%, then the same p shifted as a 2% population would give.
    y = rng.uniform(size=200000) < 0.04
    s = rng.normal(loc=y * 2.0, size=200000)
    p04 = 1 / (1 + np.exp(-(2 * s - 2) - np.log(0.04 / 0.96)))
    keep = np.r_[np.flatnonzero(y)[:: 2], np.flatnonzero(~y)]  # half the fraud: about 2%
    got = rule.recent_prior(p04[keep], 0.04)
    assert abs(got - y[keep].mean()) < 0.004


def test_shift_p_keeps_the_ranking_and_reaches_the_prior_limits():
    p = np.array([0.01, 0.1, 0.5, 0.9])
    q = rule.shift_p(p, 0.04, 0.02)
    assert np.all(np.diff(q) > 0) and np.all(q < p)
    assert np.allclose(rule.shift_p(p, 0.04, 0.04), p)


def test_isotonic_is_monotone_and_never_leaves_the_range():
    x = np.linspace(0, 1, 1000)
    y = (np.random.default_rng(3).uniform(size=1000) < x).astype(float)
    iso = rule.Isotonic().fit(x, y)
    out = iso.predict(np.linspace(-1, 2, 500))
    assert np.all(np.diff(out) >= 0) and out.min() >= 0 and out.max() <= 1
