import numpy as np
import pytest

from fraud import data

HEADER = "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud\n"
ROWS = [
    "1,PAYMENT,9839.64,C1231006815,777777.77,160296.36,M1979787155,0.0,0.0,0,0\n",
    "1,TRANSFER,181.0,C1305486145,181.0,0.0,C553264065,0.0,0.0,1,0\n",
    "1,CASH_OUT,181.0,C840083671,181.0,0.0,C38997010,21182.0,0.0,1,0\n",
    "2,TRANSFER,250000.0,C1305486145,888888.88,0.0,C553264065,0.0,999999.99,0,1\n",
]


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "paysim.csv"
    p.write_text(HEADER + "".join(ROWS), encoding="utf-8")
    return p


def test_cache_has_no_balance_columns(csv_path, tmp_path):
    cache = data.build(csv_path, tmp_path / "paysim.npz")
    with np.load(cache) as z:
        assert not any("balance" in k.lower() for k in z.files)
        values = np.concatenate([z[k].astype(np.float64).ravel() for k in z.files if z[k].dtype.kind in "biuf"])
    for leak in (777777.77, 888888.88, 999999.99, 160296.36, 21182.0):
        assert not np.isclose(values, leak).any(), leak


def test_load_round_trip(csv_path, tmp_path):
    t = data.load(data.build(csv_path, tmp_path / "paysim.npz"))
    assert len(t) == 4
    assert t.step.tolist() == [1, 1, 1, 2]
    assert [data.TYPES[k] for k in t.type] == ["PAYMENT", "TRANSFER", "CASH_OUT", "TRANSFER"]
    assert t.amount.tolist() == pytest.approx([9839.64, 181.0, 181.0, 250000.0])
    assert t.is_fraud.tolist() == [0, 1, 1, 0]
    assert t.flagged.tolist() == [0, 0, 0, 1]
    assert t.dest_merchant.tolist() == [True, False, False, False]
    # the same account name maps to the same id, in both columns
    assert t.orig[1] == t.orig[3]
    assert t.dest[1] == t.dest[3]
    assert t.orig[1] != t.orig[2]


def test_unknown_column_layout_is_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("step,type,amount\n1,PAYMENT,1.0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        data.build(p, tmp_path / "x.npz")
