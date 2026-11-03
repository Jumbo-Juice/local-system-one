"""Per-card and per-address history for IEEE-CIS: the signal for row i reads only rows before i, and never a label (Phase 19).

    python -m fraud.history        # fraud/data/ieee-cis/history.npz (git-ignored) from the cache and train_identity.csv

File order is arrival order, as everywhere in the fraud app. Ties in TransactionDT count the earlier row in the file as earlier. isFraud is not an input to anything here: the module has no access to it
(fraud/tests/test_history.py checks that the features don't change when labels do).
A row's own values are its own to read: its amount, its device. What it may not read is any later row.

Keys (Implementation choice): card = card1 card2 card3 card5 card4 card6; card_addr = card + addr1; addr = addr1 addr2; card_email = card1 P_emaildomain; device = DeviceInfo.
uid = card1 addr1 and the day the D1 counter started, which leans on D1 being known at decision time, so it's built apart and marked (Phase 17 could not clear D1).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from . import ieee_data
from .ieee_data import Ieee

CACHE = ieee_data.RAW / "history.npz"
SECONDS = {"1h": 3600, "1d": 86400, "7d": 7 * 86400}
ID_NUMERIC = tuple(f"id_{k:02d}" for k in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 17, 18, 19, 20, 21, 22, 24, 25, 26, 32))
ID_TEXT = tuple(f"id_{k}" for k in (12, 15, 16, 23, 27, 28, 29, 34, 35, 36, 37, 38))
TOP_DEVICES, TOP_BROWSERS, TOP_OS = 30, 12, 8


def codes(*cols) -> np.ndarray:
    """One integer per distinct combination of these columns. A missing value counts as a value."""
    x = np.stack([np.nan_to_num(np.asarray(c, np.float64), nan=-1.0) for c in cols], 1)
    return np.unique(x, axis=0, return_inverse=True)[1].ravel().astype(np.int64)


def by_group(key: np.ndarray, dt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(order, first, position): rows sorted by group then file order; each sorted row's group start; its count of earlier rows in the group."""
    order = np.argsort(key, kind="stable")
    k = key[order]
    new = np.r_[True, k[1:] != k[:-1]]
    first = np.maximum.accumulate(np.where(new, np.arange(len(k)), 0))
    return order, first, np.arange(len(k)) - first


def count_recent(order: np.ndarray, first: np.ndarray, key: np.ndarray, dt: np.ndarray, seconds: int) -> np.ndarray:
    """Earlier rows of the group within these seconds, per sorted row. Vectorised: group code and time packed into one sorted key."""
    span = int(dt.max()) + seconds + 2
    packed = key[order].astype(np.int64) * span + dt[order]
    return np.arange(len(order)) - np.searchsorted(packed, packed - seconds, side="left")


def distinct_before(order: np.ndarray, first: np.ndarray, other: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Distinct values of other seen in the group before each sorted row, not counting this row."""
    pair = key[order].astype(np.int64) * (int(other.max()) + 2) + other[order]
    _, first_seen = np.unique(pair, return_index=True)
    new = np.zeros(len(order), np.int64)
    new[first_seen] = 1
    cum = np.cumsum(new)
    return cum - new - (cum[first] - new[first])


def key_features(prefix: str, key: np.ndarray, dt: np.ndarray, amount: np.ndarray, others: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """What earlier rows of this key say about row i, per row, in file order."""
    order, first, pos = by_group(key, dt)
    n = len(key)
    x, d = np.log1p(amount[order]), dt[order].astype(np.float64)
    since = np.r_[0.0, d[1:] - d[:-1]]
    since[pos == 0] = 0.0
    cs, cs2 = np.cumsum(x), np.cumsum(x * x)
    before = lambda c: (c - x if c is cs else c - x * x) - (c[first] - (x[first] if c is cs else x[first] * x[first]))
    safe = np.maximum(pos, 1)
    mean = before(cs) / safe
    var = np.maximum(before(cs2) / safe - mean * mean, 0.0)
    z = np.where(pos >= 2, (x - mean) / np.sqrt(var + 0.01), 0.0)
    same_key = key[order].astype(np.int64) * 1_000_003 + np.round(amount[order] * 100).astype(np.int64)
    order2, _, pos2 = by_group(same_key, d)
    same = np.empty(n, np.int64)
    same[order2] = pos2
    out = {f"{prefix}_log_before": np.log1p(pos), f"{prefix}_first": (pos == 0).astype(np.float64), f"{prefix}_log_since": np.log1p(since),
           f"{prefix}_amt_z": z, f"{prefix}_amt_vs_mean": np.where(pos > 0, x - mean, 0.0), f"{prefix}_log_same_amt": np.log1p(same)}
    for name, s in SECONDS.items():
        out[f"{prefix}_log_{name}"] = np.log1p(count_recent(order, first, key, dt, s))
    for name, other in (others or {}).items():
        out[f"{prefix}_log_distinct_{name}"] = np.log1p(distinct_before(order, first, other, key))
    back = np.empty(n, np.int64)
    back[order] = np.arange(n)
    return {k: v[back] for k, v in out.items()}


def read_identity(path: Path, ids: np.ndarray) -> dict[str, np.ndarray]:
    """train_identity.csv's text as lists aligned with ids; "" where a row has no identity."""
    pos = {int(k): j for j, k in enumerate(ids)}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        head = next(reader)
        cols = {k: np.empty(len(ids), dtype=object) for k in head[1:]}
        for v in cols.values():
            v[:] = ""
        for row in reader:
            j = pos.get(int(row[0]))
            if j is not None:
                for k, val in zip(head[1:], row[1:]):
                    cols[k][j] = val
    return cols


def _family(text: np.ndarray, words: tuple[str, ...]) -> np.ndarray:
    """A code per row: the first of these words its lower-cased text holds, 0 if none. Browser and system names."""
    low = np.char.lower(text.astype(str))
    out = np.zeros(len(text), np.int16)
    for j, w in enumerate(words, 1):
        hit = (out == 0) & (np.char.find(low, w) >= 0)
        out[hit] = j
    return out


def _top_codes(text: np.ndarray, fit: int, top: int) -> tuple[np.ndarray, list[str]]:
    """A code per row among the most common non-empty values in the fit rows; 0 is anything else. The list comes from fit rows only."""
    values, counts = np.unique(text[:fit].astype(str), return_counts=True)
    keep = [v for v in values[np.argsort(-counts)] if v][:top]
    at = {v: j for j, v in enumerate(keep, 1)}
    return np.array([at.get(str(v), 0) for v in text], np.int16), keep


BROWSERS = ("chrome", "safari", "firefox", "edge", "samsung", "ie ", "opera", "android", "google", "mobile", "other")
SYSTEMS = ("windows", "ios", "mac", "android", "linux")


def build(t: Ieee, identity: dict[str, np.ndarray]) -> dict:
    """Every history and category feature. Reads t's columns, never t.is_fraud."""
    fit = ieee_data.split_bounds(len(t))[0]
    a = t.amount
    cat = lambda k: t.cat[:, t.cat_names.index(k)].astype(np.int64)
    c1, c2, c3, c5, addr1, addr2 = (t.col(k) for k in ("card1", "card2", "card3", "card5", "addr1", "addr2"))
    card = codes(c1, c2, c3, c5, cat("card4"), cat("card6"))
    device_text = identity["DeviceInfo"]
    device_codes = np.unique(device_text.astype(str), return_inverse=True)[1].ravel().astype(np.int64)
    feats: dict[str, np.ndarray] = {}
    feats.update(key_features("card", card, t.dt, a, {"addr": codes(addr1), "email": cat("P_emaildomain"), "device": device_codes}))
    feats.update(key_features("card_addr", codes(c1, c2, c3, c5, cat("card4"), cat("card6"), addr1), t.dt, a))
    feats.update(key_features("addr", codes(addr1, addr2), t.dt, a, {"card": codes(c1)}))
    feats.update(key_features("card_email", codes(c1, cat("P_emaildomain")), t.dt, a))
    has = t.has_identity.astype(np.float64)
    for k, v in key_features("device", device_codes, t.dt, a, {"card": codes(c1)}).items():
        feats[k] = v * has  # no identity row, no device history
    day_began = np.floor(t.dt / 86400 - np.nan_to_num(t.col("D1"), nan=-1.0))
    uid = key_features("uid", codes(c1, addr1, day_began), t.dt, a, {"email": cat("P_emaildomain")})

    cents = np.round((a - np.floor(a)) * 1000).astype(np.int64)
    extra = {"amt_is_whole": (cents == 0).astype(np.float64), "amt_three_decimals": (cents % 10 != 0).astype(np.float64), "amt_cents_99": (cents == 990).astype(np.float64),
             "email_same": ((cat("P_emaildomain") == cat("R_emaildomain")) & (cat("P_emaildomain") > 0)).astype(np.float64),
             "day_of_week_sin": np.sin(2 * np.pi * (t.dt // 86400 % 7) / 7), "day_of_week_cos": np.cos(2 * np.pi * (t.dt // 86400 % 7) / 7),
             "hour_sin": np.sin(2 * np.pi * (t.dt % 86400) / 86400), "hour_cos": np.cos(2 * np.pi * (t.dt % 86400) / 86400)}
    numeric = {}
    for k in ID_NUMERIC:
        v = np.array([float(x) if x else np.nan for x in identity[k]], np.float64)
        numeric[k] = np.nan_to_num(np.sign(v) * np.log1p(np.abs(v)))
        numeric[f"{k}_missing"] = np.isnan(v).astype(np.float64)
    categories: dict[str, list[str]] = {}
    one_hot = {}
    for k in ID_TEXT:
        c, names = _top_codes(identity[k], fit, 8)
        one_hot[k], categories[k] = c, names
    one_hot["device_top"], categories["device_top"] = _top_codes(identity["DeviceInfo"], fit, TOP_DEVICES)
    one_hot["browser"], categories["browser"] = _family(identity["id_31"], BROWSERS), list(BROWSERS)
    one_hot["system"], categories["system"] = _family(identity["id_30"], SYSTEMS), list(SYSTEMS)
    return {"history": feats, "uid": uid, "extra": extra, "numeric": numeric, "one_hot": one_hot, "categories": categories}


def save(b: dict, path: Path = CACHE) -> None:
    out = {f"history/{k}": v.astype(np.float32) for k, v in b["history"].items()}
    out.update({f"uid/{k}": v.astype(np.float32) for k, v in b["uid"].items()})
    out.update({f"extra/{k}": v.astype(np.float32) for k, v in b["extra"].items()})
    out.update({f"numeric/{k}": v.astype(np.float32) for k, v in b["numeric"].items()})
    out.update({f"one_hot/{k}": v for k, v in b["one_hot"].items()})
    np.savez(path, categories=np.array(json.dumps(b["categories"])), **out)


def load(path: Path = CACHE) -> dict:
    if not path.exists():
        raise SystemExit(f"no cache at {path}; run: python -m fraud.history")
    out: dict = {"history": {}, "uid": {}, "extra": {}, "numeric": {}, "one_hot": {}}
    with np.load(path) as z:
        out["categories"] = json.loads(str(z["categories"]))
        for k in z.files:
            if "/" in k:
                g, name = k.split("/", 1)
                out[g][name] = z[k]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    t0 = time.perf_counter()
    t = ieee_data.load()
    b = build(t, read_identity(ieee_data.RAW / "train_identity.csv", t.ids))
    save(b)
    print(f"wrote {CACHE}: history {len(b['history'])}, uid {len(b['uid'])}, extra {len(b['extra'])}, numeric {len(b['numeric'])}, one-hot {len(b['one_hot'])}, in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
