"""PaySim → a compact numpy cache, without the balance columns.

    python -m fraud.data                    # fraud/data/raw/Synthetic_Financial_datasets_log.csv → fraud/data/paysim.npz
    python -m fraud.data --csv <file.csv>   # another copy of the CSV

PaySim (Kaggle sriharshaeedala/financial-fraud-detection-dataset, CC BY-SA 4.0) documents that fraud-detected transactions are
cancelled, so oldbalanceOrg, newbalanceOrig, oldbalanceDest and newbalanceDest "must not be used"
for detection. They are never read into memory here: the reader keeps only the columns in KEEP.
Row order is kept: it is the arrival order the signals rely on.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CACHE = DATA / "paysim.npz"

TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
TYPE_ID = {t: i for i, t in enumerate(TYPES)}
EXPECTED = ("step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest",
            "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud")
KEEP = ("step", "type", "amount", "nameOrig", "nameDest", "isFraud", "isFlaggedFraud")


@dataclass
class Transactions:
    """Column arrays, one entry per PaySim row in file (arrival) order."""

    step: np.ndarray           # int16, hour 1..743
    type: np.ndarray           # int8, index into TYPES
    amount: np.ndarray         # float64
    orig: np.ndarray           # int32 account id (one id space for both columns)
    dest: np.ndarray           # int32 account id
    dest_merchant: np.ndarray  # bool, PaySim names starting with "M"
    is_fraud: np.ndarray       # int8, the truth
    flagged: np.ndarray        # int8, PaySim's own isFlaggedFraud (a transfer over 200,000)

    def __len__(self) -> int:
        return len(self.step)


CSV_NAME = "Synthetic_Financial_datasets_log.csv"


def find_csv() -> Path:
    raw = DATA / "raw"
    # Older downloads kept the original PaySim file name (PS_*.csv); the bytes are identical.
    found = sorted(raw.glob(CSV_NAME)) + sorted(raw.glob("PS_*.csv"))
    if not found:
        raise SystemExit("no PaySim CSV in fraud/data/raw/; see RUN-GUIDE.md → Fraud app → Data")
    return found[0]


def build(csv_path: Path, out: Path = CACHE) -> Path:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = tuple(next(reader))
        if header != EXPECTED:
            raise ValueError(f"unexpected PaySim columns: {header}")
        col = [header.index(k) for k in KEEP]
        ids: dict[str, int] = {}
        step, typ, amount, orig, dest, merchant, fraud, flagged = ([] for _ in range(8))
        for row in reader:
            s, t, a, o, d, y, g = (row[i] for i in col)
            step.append(int(s))
            typ.append(TYPE_ID[t])
            amount.append(float(a))
            orig.append(ids.setdefault(o, len(ids)))
            dest.append(ids.setdefault(d, len(ids)))
            merchant.append(d.startswith("M"))
            fraud.append(int(y))
            flagged.append(int(g))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, step=np.array(step, np.int16), type=np.array(typ, np.int8), amount=np.array(amount, np.float64),
             orig=np.array(orig, np.int32), dest=np.array(dest, np.int32), dest_merchant=np.array(merchant, bool),
             is_fraud=np.array(fraud, np.int8), flagged=np.array(flagged, np.int8))
    return out


def load(path: Path = CACHE) -> Transactions:
    if not Path(path).exists():
        raise SystemExit(f"no cache at {path}; run: python -m fraud.data")
    with np.load(path) as z:
        return Transactions(**{k: z[k] for k in z.files})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=CACHE)
    args = ap.parse_args()
    t0 = time.perf_counter()
    path = build(args.csv or find_csv(), args.out)
    t = load(path)
    print(f"wrote {path} ({len(t):,} rows, {int(t.is_fraud.sum()):,} fraud) in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
