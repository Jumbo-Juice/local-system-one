import numpy as np

from fraud import eval_ieee as ev, eval_tabular as et, final_test as ft, tabular as tb
from fraud.baselines import LogReg

N = 590540


def test_the_last_chains_are_the_tabular_windows_and_not_final():
    new = ft.phase_windows(N, "tabular")
    assert len(new) == 10 and sum(ft.PHASE_WINDOWS.values()) == 38
    rows = np.concatenate([w.rows for w in new])
    ft.guard(N, rows)
    others = [w for k in ("audit", "rule", "features") for w in ft.phase_windows(N, k)] + ft.used_windows(N)
    assert not np.isin(rows, np.concatenate([w.rows for w in others])).any()


def test_setups_are_as_registered():
    assert et.SETS["tab-logreg"] == ("logreg", (), "") and et.SETS["tab-all"] == ("trees", (), "") and et.SETS["tab-nocdv"] == ("trees", et.RICH, "CDV")
    assert et.PARAMS.trees == 800 and et.PARAMS.depth == 6 and et.PARAMS.lr == 0.05


def test_lambda_is_picked_from_the_windows_given_and_p_is_left_alone():
    rng = np.random.default_rng(0)

    class T:
        amount = rng.lognormal(5, 1.2, 4000)
        is_fraud = (rng.uniform(size=4000) < 0.05).astype(np.int8)

    q = np.clip(0.03 + 0.5 * T.is_fraud * rng.uniform(size=4000) + 0.05 * rng.uniform(size=4000), 1e-6, 1 - 1e-6)
    wins = [ev.Win(f"w{k}", "dev", np.arange(k * 1000, (k + 1) * 1000)) for k in range(4)]
    x = np.concatenate([np.log(q[w.rows] / (1 - q[w.rows]))[:, None] for w in wins])
    pol = tb.Policy(LogReg().fit(x, T.is_fraud.astype(float)), 0.0, None, None)
    before = pol.m.w.copy()
    lam = et.lambda_on(pol, q, T, wins)
    assert lam in (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) and np.array_equal(before, pol.m.w)
