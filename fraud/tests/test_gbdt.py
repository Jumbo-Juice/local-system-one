import numpy as np

from fraud import eval_ieee as ev, gbdt
from fraud.gbdt import GBDT, Params

SMALL = Params(trees=60, depth=3, lr=0.2, min_hessian=1.0, l2=1.0, row_sample=1.0, col_sample=1.0, threads=2)


def data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5))
    logit = 4 * ((x[:, 0] > 0) ^ (x[:, 1] > 0)) - 2  # an interaction: a linear model can't read it
    return x, (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(float)


def test_bins_keep_order_and_give_missing_its_own_bin():
    x = np.array([[1.0], [2.0], [np.nan], [3.0], [4.0], [5.0]])
    e = gbdt.quantile_edges(x, bins=4)
    b = gbdt.to_bins(x, e)
    assert b[2, 0] == 0 and (b[[0, 1, 3, 4, 5], 0] >= 1).all()
    assert np.all(np.diff(b[[0, 1, 3, 4, 5], 0].astype(int)) >= 0)


def test_it_learns_an_interaction_a_linear_model_cannot():
    x, y = data()
    xt, yt = data(seed=1)
    from fraud.baselines import LogReg

    trees = ev.auc(GBDT(SMALL).fit(x, y).predict(xt), yt.astype(int))
    line = ev.auc(LogReg().fit(x, y).predict(xt), yt.astype(int))
    assert trees > 0.8 and line < 0.6


def test_the_loss_falls_and_validation_is_tracked():
    x, y = data()
    m = GBDT(SMALL).fit(x, y, valid=data(seed=2), every=20)
    h = m.history
    assert h[-1]["train_logloss"] < h[0]["train_logloss"] < 0.7 and "valid_logloss" in h[-1] and len(h) == 3


def test_same_seed_same_model_and_thread_count_does_not_matter():
    x, y = data(800)
    p = Params(trees=10, depth=3, lr=0.2, min_hessian=1.0, l2=1.0, row_sample=0.7, col_sample=0.6, threads=1, seed=3)
    a = GBDT(p).fit(x, y).raw(x)
    b = GBDT(p).fit(x, y).raw(x)
    c = GBDT(Params(trees=10, depth=3, lr=0.2, min_hessian=1.0, l2=1.0, row_sample=0.7, col_sample=0.6, threads=4, seed=3)).fit(x, y).raw(x)
    assert np.array_equal(a, b) and np.array_equal(a, c)
    d = GBDT(Params(trees=10, depth=3, lr=0.2, min_hessian=1.0, l2=1.0, row_sample=0.7, col_sample=0.6, threads=1, seed=4)).fit(x, y).raw(x)
    assert not np.array_equal(a, d)


def test_missing_values_can_be_sent_either_way():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 1))
    x[:500, 0] = np.nan
    y = np.zeros(2000)
    y[:500] = 1.0  # all the missing rows are positive
    p = GBDT(Params(trees=10, depth=2, lr=0.3, min_hessian=1.0, l2=1.0, row_sample=1.0, col_sample=1.0, threads=1)).fit(x, y).predict(x)
    assert p[:500].min() > 0.8 and p[500:].max() < 0.2


def test_min_hessian_stops_tiny_leaves():
    x, y = data(300)
    m = GBDT(Params(trees=1, depth=6, lr=0.1, min_hessian=1000.0, l2=1.0, row_sample=1.0, col_sample=1.0, threads=1)).fit(x, y)
    assert (m.trees[0].feature >= 0).sum() == 0


def test_fewer_trees_can_be_read_back():
    x, y = data(600)
    m = GBDT(Params(trees=20, depth=3, lr=0.2, min_hessian=1.0, l2=1.0, row_sample=1.0, col_sample=1.0, threads=1)).fit(x, y)
    assert not np.array_equal(m.raw(x, 5), m.raw(x))
    assert np.array_equal(m.raw(x, 20), m.raw(x))
