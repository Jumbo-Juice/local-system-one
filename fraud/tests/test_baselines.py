import numpy as np

from fraud.baselines import LogReg


def test_logreg_learns_a_separable_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 3))
    y = (x[:, 1] + 0.1 * rng.normal(size=2000) > 0.5).astype(float)
    m = LogReg(l2=1e-4).fit(x, y)
    p = m.predict(x)
    assert ((p > 0.5) == (y > 0.5)).mean() > 0.95
    assert abs(m.w[2]) > 5 * max(abs(m.w[1]), abs(m.w[3]))  # weight on the signal column


def test_logreg_is_deterministic():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(300, 2)), rng.integers(0, 2, 300).astype(float)
    assert np.allclose(LogReg().fit(x, y).w, LogReg().fit(x, y).w)
