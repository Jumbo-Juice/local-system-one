"""IEEE-CIS Fraud Detection (Vesta) → a compact numpy cache, plus its time split.

    python -m fraud.ieee_data                       # fraud/data/ieee-cis/train_*.csv → fraud/data/ieee-cis/ieee.npz

Only the two train files are read: the test files have no labels (Observed: no isFraud column), so a
held-out test is a time split inside the train file. Rows stay in file order, which is TransactionDT
order (build checks it). TransactionDT is seconds from an unstated reference point, not a timestamp.

Split (Implementation choice, by row count, before any label was looked at): the first 60% of rows fit
baselines, the next 20% are dev, the last 20% are test. Labels of test rows are read only by the
pre-registered eval (fraud/eval_ieee.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "ieee-cis"
CACHE = RAW / "ieee.npz"
TEXT = ("ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
        *[f"M{k}" for k in range(1, 10)], "DeviceType")
SPLIT_SHARES = (0.6, 0.8)  # fit | dev | test
MISSING = ""


@dataclass
class Ieee:
    """Column arrays in file (time) order. ``num``: float32 with NaN for missing, one column per ``num_names``.
    ``cat``: int16 codes per ``cat_names``, code 0 is missing, ``vocab[name][code]`` is the text."""

    ids: np.ndarray
    dt: np.ndarray
    amount: np.ndarray
    is_fraud: np.ndarray
    num: np.ndarray
    num_names: tuple
    cat: np.ndarray
    cat_names: tuple
    vocab: dict
    has_identity: np.ndarray

    def __len__(self) -> int:
        return len(self.dt)

    def col(self, name: str) -> np.ndarray:
        return self.num[:, self.num_names.index(name)]

    def col_at(self, name: str, i: int) -> float:
        return self.num[i, self.num_names.index(name)]

    def text(self, name: str, i: int) -> str:
        return self.vocab[name][int(self.cat[i, self.cat_names.index(name)])]


def split_bounds(n: int) -> tuple[int, int]:
    return int(n * SPLIT_SHARES[0]), int(n * SPLIT_SHARES[1])


def split_rows(n: int, split: str) -> tuple[int, int]:
    a, b = split_bounds(n)
    return {"fit": (0, a), "dev": (a, b), "test": (b, n)}[split]


def _number(s: str) -> float:
    return float(s) if s else np.nan


def build(raw: Path = RAW, out: Path = CACHE) -> Path:
    identity = {}
    with open(raw / "train_identity.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        head = next(reader)
        at = head.index("DeviceType")
        for row in reader:
            identity[int(row[0])] = row[at]
    vocab = {k: [MISSING] for k in TEXT}
    code = {k: {MISSING: 0} for k in TEXT}
    with open(raw / "train_transaction.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        head = next(reader)
        fixed = ("TransactionID", "isFraud", "TransactionDT", "TransactionAmt")
        names = tuple(k for k in head if k not in fixed and k not in TEXT)
        num_at = [head.index(k) for k in names]
        cat_at = [(k, head.index(k)) for k in TEXT if k != "DeviceType"]
        pos = {k: head.index(k) for k in fixed}
        ids, dt, amount, fraud, num, cat, has = [], [], [], [], [], [], []
        for row in reader:
            tid = int(row[pos["TransactionID"]])
            ids.append(tid)
            dt.append(int(row[pos["TransactionDT"]]))
            amount.append(float(row[pos["TransactionAmt"]]))
            fraud.append(int(row[pos["isFraud"]]))
            num.append(np.array([_number(row[j]) for j in num_at], np.float32))
            device = identity.get(tid, MISSING)
            has.append(tid in identity)
            codes = []
            for k, v in [*[(k, row[j]) for k, j in cat_at], ("DeviceType", device)]:
                if v not in code[k]:
                    code[k][v] = len(vocab[k])
                    vocab[k].append(v)
                codes.append(code[k][v])
            cat.append(codes)
    dt = np.array(dt, np.int64)
    if np.any(np.diff(dt) < 0):
        raise ValueError("train_transaction.csv is not in TransactionDT order")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, ids=np.array(ids, np.int64), dt=dt, amount=np.array(amount, np.float64),
             is_fraud=np.array(fraud, np.int8), num=np.stack(num), num_names=np.array(names),
             cat=np.array(cat, np.int16), cat_names=np.array(TEXT), vocab=np.array(json.dumps(vocab)),
             has_identity=np.array(has, bool))
    return out


def load(path: Path = CACHE) -> Ieee:
    if not Path(path).exists():
        raise SystemExit(f"no cache at {path}; run: python -m fraud.ieee_data")
    with np.load(path) as z:
        return Ieee(ids=z["ids"], dt=z["dt"], amount=z["amount"], is_fraud=z["is_fraud"], num=z["num"],
                    num_names=tuple(str(k) for k in z["num_names"]), cat=z["cat"],
                    cat_names=tuple(str(k) for k in z["cat_names"]), vocab=json.loads(str(z["vocab"])),
                    has_identity=z["has_identity"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=CACHE)
    args = ap.parse_args()
    t0 = time.perf_counter()
    path = build(args.raw, args.out)
    print(f"wrote {path} ({len(load(path)):,} rows) in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
