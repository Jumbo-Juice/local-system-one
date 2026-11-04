"""LoRA fine-tune of d1-Omni-2B on IEEE-CIS fit rows → fraud/models/<time>_lora-d1-ieee/.

    python -m fraud.finetune_ieee --dry-run      # the training set only: counts and the row hash
    python -m fraud.finetune_ieee --steps 20     # smoke run: s/step and peak memory
    python -m fraud.finetune_ieee --train        # the training run
    python -m fraud.finetune_ieee --train --full # Phase 12: every fit fraud (--dry-run and --steps take --full too)
    python -m fraud.finetune_ieee --train --full --seed 1     # Phase 21: a second seed; or --passes 2; or --rich for the longer prompt
    python -m fraud.finetune_ieee --train --full --rich --weighted   # Phase 25: each row's loss counts by what a mistake on it costs

What it trains (pre-registered in fraud/eval_ieee.py, Phase 11 of fraud/ROADMAP.md). Owner, 2026-11-02: "train on ieee cis, compare
paysim fine tune with ieee cis fine tune". Everything but the data matches the PaySim adapter 20261102-152745_lora-d1, so the
comparison changes one thing (Implementation choice):
- Model: fraud.d1torch, starting from the frozen weights, not from the PaySim adapter.
- Rows: fit rows only (the first 60% by time). The PaySim adapter's size and ratio: 3,021 fraud + 9,063 legit, seed 0, shuffled.
  That leaves most fit fraud unused: the fit rows hold 11,988.
- Prompt: fraud.ieee_brain as the lone "choice" prompt of each example, in a seeded option order. Target: the index of
  "no: approve it" (legit) or "yes: decline it" (fraud).
- LoRA, optimiser, schedule, loss: fraud.finetune_d1's. 1,510 batches of 8, one pass.
- No hour cap (the owner set none this time). A safety stop at CAP_S saves what it has and records stopped_early.

--full (Phase 12, pre-registered in fraud/eval_ieee_full.py): the only change is more rows. Every fit fraud, 11,988, plus three legit
rows for each, one pass: 47,952 rows and 5,994 batches, so the LR schedule is sized to 5,994. Output fraud/models/<time>_lora-d1-ieee-full/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np

from system_one import Decision

from . import brain, finetune, ieee_brain, ieee_data
from .finetune import MODELS, SEED
from .finetune_d1 import LORA
from .finetune_d3 import BATCH, EPOCH_BATCHES, encode, train
from .ieee_data import Ieee

N_FRAUD, N_LEGIT = 3021, 9063  # the PaySim adapter's rows
LEGIT_PER_FRAUD = 3
CAP_S, CAP_FULL_S = 3 * 3600.0, 10 * 3600.0
SAVE_MARGIN_S = 60.0
CHECKPOINT_EVERY = 250  # --full only: about 10 min of work at risk if the process dies
WEIGHT_CAP_PCT = 99.0


def training_rows(t: Ieee, seed: int = SEED, full: bool = False) -> np.ndarray:
    """N_FRAUD + N_LEGIT rows from the fit split, in a seeded random order. Never a dev or test row.
    full: every fit fraud and LEGIT_PER_FRAUD legit rows for each."""
    a, _ = ieee_data.split_bounds(len(t))
    y = t.is_fraud[:a]
    rng = np.random.default_rng(seed)
    n_fraud = int(y.sum()) if full else N_FRAUD
    n_legit = n_fraud * LEGIT_PER_FRAUD if full else N_LEGIT
    fraud = rng.choice(np.flatnonzero(y == 1), n_fraud, replace=False)
    legit = rng.choice(np.flatnonzero(y == 0), n_legit, replace=False)
    return rng.permutation(np.r_[fraud, legit]).astype(np.int64)


def cost_weights(amount: np.ndarray, fraud: np.ndarray, c, cap_pct: float = WEIGHT_CAP_PCT) -> np.ndarray:
    """Phase 25 (fraud/eval_lm_cost.py): what a mistake on each row costs. A fraud approved costs its amount, a legit row declined its friction. Capped at the class's percentile and scaled to mean 1 within each class, so the class balance doesn't move."""
    fraud = np.asarray(fraud, bool)
    v = np.where(fraud, amount, c.friction(amount)).astype(np.float64)
    w = np.empty(len(v))
    for part in (fraud, ~fraud):
        capped = np.minimum(v[part], np.percentile(v[part], cap_pct))
        w[part] = capped / capped.mean()
    return w


def examples(t: Ieee, rows: np.ndarray, seed: int = SEED, hist: dict | None = None) -> list[tuple[Decision, int]]:
    """(decision, index of the target option) per row; options as listed or reversed by a seeded coin (as finetune.examples)."""
    flip = np.random.default_rng([seed, 1]).integers(0, 2, len(rows)).astype(bool)
    out = []
    for i, rev in zip(rows, flip):
        d = ieee_brain.decision(t, int(i), hist)
        if rev:
            d = Decision(d.instruction, d.options[::-1], state=d.state, context=d.context)
        want = brain.OPTIONS[brain.ACTIONS.index("decline" if t.is_fraud[i] else "approve")]
        out.append((d, d.options.index(want)))
    return out


def main() -> None:
    t_process = time.perf_counter()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="build the training set, print counts, do not train")
    g.add_argument("--steps", type=int, help="a smoke run: this many batches, to measure s/step and memory")
    g.add_argument("--train", action="store_true", help="the training run")
    ap.add_argument("--full", action="store_true", help="every fit fraud (Phase 12)")
    ap.add_argument("--seed", type=int, default=SEED, help="Phase 21: another training seed (default 0, as before)")
    ap.add_argument("--passes", type=int, default=1, help="Phase 21: passes over the rows, with the LR schedule sized to all of them")
    ap.add_argument("--rich", action="store_true", help="Phase 21: the longer prompt, ieee_brain.describe_rich")
    ap.add_argument("--signal", action="store_true", help="Phase 21: the longer prompt and the tabular risk score (fraud.lm_signal)")
    ap.add_argument("--weighted", action="store_true", help="Phase 25: each row's loss counts by what a mistake on it costs")
    args = ap.parse_args()

    t = ieee_data.load()
    rows = training_rows(t, seed=args.seed, full=args.full)
    n_fraud = int(t.is_fraud[rows].sum())
    info = {"dataset": "IEEE-CIS", "rows": len(rows), "fraud": n_fraud, "legit": len(rows) - n_fraud,
            "rows_sha256": finetune.rows_hash(rows)}
    print(json.dumps(info), flush=True)
    if args.dry_run:
        return

    from .d1torch import D1Torch
    from .runs import stamp

    steps = args.steps or (len(rows) // BATCH * args.passes if args.full else EPOCH_BATCHES)
    cap = CAP_FULL_S if args.full else CAP_S
    hist = None
    if args.rich or args.signal:
        from . import history

        hist = history.load()
        if args.signal:
            from .lm_signal import load

            hist["signal"] = load()
    exs = examples(t, rows, args.seed, hist)
    info["reversed"] = int(sum(d.options != brain.OPTIONS for d, _ in exs))
    model = D1Torch.load()
    enc = encode(model, exs)
    info["prompt_tokens_mean"] = float(np.mean([len(e[0]) for e in enc]))
    info["prompt_tokens_max"] = int(max(len(e[0]) for e in enc))
    info["base_model"] = model.info()
    print(f"encoded {len(enc)} examples, {info['reversed']} reversed, {info['prompt_tokens_mean']:.0f} tokens mean / "
          f"{info['prompt_tokens_max']} max ({time.perf_counter() - t_process:.0f} s since start)", flush=True)

    extra = f"{'-seed' + str(args.seed) if args.seed != SEED else ''}{'-pass' + str(args.passes) if args.passes != 1 else ''}{'-rich' if args.rich else ''}{'-signal' if args.signal else ''}{'-weighted' if args.weighted else ''}"
    out = MODELS / f"{stamp()}_lora-d1-ieee{'-full' if args.full else ''}{extra}{'-smoke' if args.steps else ''}"
    out.mkdir(parents=True, exist_ok=True)
    deadline = t_process + cap - SAVE_MARGIN_S if args.train else math.inf
    every = (10 if args.steps else CHECKPOINT_EVERY) if args.full else 0
    weights = None
    if args.weighted:
        from .eval_ieee import COSTS

        weights = cost_weights(t.amount[rows], t.is_fraud[rows], COSTS)
        info["weights"] = {"cap_pct": WEIGHT_CAP_PCT, "max": float(weights.max()), "effective_share": float(weights.sum() ** 2 / (weights ** 2).sum() / len(weights))}
        print(json.dumps(info["weights"]), flush=True)
    info.update(train(model, enc, out, steps, deadline=deadline, log=lambda m: print(m, flush=True), lora=LORA, checkpoint_every=every, seed=args.seed, weights=weights))
    info.update(passes=args.passes, rich=args.rich or args.signal, signal=args.signal, weighted=args.weighted)
    info["process_s"] = time.perf_counter() - t_process
    info["cap_s"] = cap
    info["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / "manifest.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"saved {out}  steps {info['steps_done']}/{steps}{' (stopped early)' if info['stopped_early'] else ''}  "
          f"loss first100 {info['loss_first100']:.4f} -> last100 {info['loss_last100']:.4f}  "
          f"peak {info['peak_mem_gib']} GiB  process {info['process_s'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
