"""Every setup without the C counter columns, on dev only → fraud/results/eval_noc/ (revision Stage 2.3).

    python -m fraud.finetune_ieee --train --full --rich --noc        # the one LM retrain (~2 h on the RTX 3060 Ti)
    python -m fraud.eval_noc --score <adapter dir>                    # its dev scores (~85 min) → fraud/data/ieee-cis/lm_scores/lm-rich-noc.npz
    python -m fraud.eval_noc                                          # fit LR and trees without C, compare with and without C

Models: fraud/models/parity_noc-*.pkl (git-ignored). Results: fraud/results/eval_noc/<eval id>.json + .md.

PRE-REGISTRATION (revision Stage 2.3; committed before the retrain started and before any no-C model was fitted).
- The threat: C1-C14 are counters whose timing Phase 17 could not clear; they may know more than a decider would have at the
  time. Does the ranking of setups by dev cost survive without them?
- What "without C" means here (Assumed, REVISION_LOG.md): no C column anywhere. LR and trees: all columns minus C1-C14 and their
  missing flags (fraud/eval_column_audit.features_without("C")). LM alone: the rich prompt without its four counters
  (C1, C4, C8, C13) and its six more counters (C2, C5, C6, C9, C11, C14), and the context block without "with counters C4 or C8
  above 0" (ieee_brain, hist["noc"]). The revision prompt names C1/C4/C8/C13, the plain prompt's four; the rich prompt also shows
  six more, and leaving them in would make the LM's no-C variant the only one still reading C.
- Setups without C: noc-logreg (LR, all columns minus C), noc-trees (fraud/gbdt.py at Phase 20's settings: 800 trees, depth 6,
  learning rate 0.05, row sample 0.5, column sample 0.3, seed 0), lm-rich-noc (exactly one LoRA retrain of LM alone:
  finetune_ieee --full --rich --noc, the same rows (sha256 e6d057cc...), seed, LoRA, schedule and loss as the lm-rich adapter;
  scored in order rev, batches of 16, as Phase 21).
- With C, the same rows: tab-logreg, tab-all, lm-rich (Phase 21's saved scores), and lm-signal (LM + tree). LM + tree has no no-C
  variant: it would need a second retrain, and the revision prompt allows one.
- Policy for every setup here: p fitted on dev30, lambda on all 118 dev windows (as fraud/eval_parity.py). Phase 14's rule. Read on
  the 88 held-out chains. Label: dev (flattered, lambda saw these chains).
- Reported: mean cost and AUC per setup; the ranking by dev cost of LR, trees and LM alone with C and without C, and whether it
  changes; paired window-bootstrap 95% percentile intervals (10,000 resamples, seed 20261102) for each no-C setup against its
  with-C version, and among the no-C setups (trees vs LR, LM alone vs LR, LM alone vs trees).
- Stated before running (Inferred): every setup gets dearer without C, the trees and LM alone more than LR (Phase 20's
  tab-nocdv lost about 0.05 AUC); the ranking trees < LR <= LM alone holds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import eval_lm_nuc as nuc, eval_parity as par, feature_sets as fs, history, ieee_data, tabular as tb
from .runs import stamp

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "eval_noc"
NOC = "lm-rich-noc"


def score(adapter: Path, log=print) -> None:
    t = ieee_data.load()
    n = len(t)
    rows = np.unique(np.concatenate([w.rows for w in tb.dev30(n) + tb.heldout_dev(n)]))
    hist = {**history.load(), "noc": True}
    man = json.loads((adapter / "manifest.json").read_text(encoding="utf-8"))
    if not man.get("noc"):
        raise SystemExit(f"{adapter.name} wasn't trained with --noc")
    model = nuc.load_model(adapter)
    t0 = time.perf_counter()
    q = nuc.score_rows(model, t, rows, hist=hist, log=log)
    info = {"setup": NOC, "adapter": adapter.name, "adapter_sha256": man.get("adapter_sha256"), "rows": int(len(rows)), "score_s": round(time.perf_counter() - t0, 1), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    nuc.SCORES.mkdir(parents=True, exist_ok=True)
    np.savez(nuc.SCORES / f"{NOC}.npz", rows=rows, q=q, info=np.array(json.dumps(info)))
    log(json.dumps(info))


def ranking(rows: dict, names: tuple[str, ...]) -> list[str]:
    return sorted(names, key=lambda k: rows[k]["mean_cost"])


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", metavar="ADAPTER_DIR", type=Path)
    a = ap.parse_args()
    log = lambda m: print(m, flush=True)
    if a.score:
        score(a.score, log)
        return
    t = ieee_data.load()
    f = fs.FeatureSet((), "C", None)
    setups = {"noc-logreg": ("fit", f), "noc-trees": ("fit", f), NOC: ("score", nuc.saved(NOC, len(t))[0]), **par.references(t, log)}
    pairs = (("noc-logreg", "tab-logreg"), ("noc-trees", "tab-all"), (NOC, "lm-rich"),
             ("noc-trees", "noc-logreg"), (NOC, "noc-logreg"), (NOC, "noc-trees"))
    out = par.run(setups, pairs, "Without C columns", "fraud/eval_noc.py", RESULTS, log)
    s = out["setups"]
    out["ranking"] = {"with_c": ranking(s, ("tab-logreg", "tab-all", "lm-rich")), "without_c": ranking(s, ("noc-logreg", "noc-trees", NOC))}
    strip = {"tab-logreg": "LR", "noc-logreg": "LR", "tab-all": "trees", "noc-trees": "trees", "lm-rich": "LM alone", NOC: "LM alone"}
    out["ranking"]["changed"] = [strip[k] for k in out["ranking"]["with_c"]] != [strip[k] for k in out["ranking"]["without_c"]]
    path = RESULTS / f"{out['eval']}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = RESULTS / f"{out['eval']}.md"
    md.write_text(md.read_text(encoding="utf-8") + f"\nRanking by dev cost, with C: {', '.join(strip[k] for k in out['ranking']['with_c'])}; without C: {', '.join(strip[k] for k in out['ranking']['without_c'])}. Changed: {'yes' if out['ranking']['changed'] else 'no'}.\n", encoding="utf-8")
    log(f"ranking: {out['ranking']}")


if __name__ == "__main__":
    main()
