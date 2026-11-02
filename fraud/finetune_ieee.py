"""LoRA fine-tune of d1-Omni-2B on IEEE-CIS fit rows → fraud/models/<time>_lora-d1-ieee/.

    python -m fraud.finetune_ieee --dry-run      # the training set only: counts and the row hash
    python -m fraud.finetune_ieee --steps 20     # smoke run: s/step and peak memory
    python -m fraud.finetune_ieee --train        # the training run

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
from .finetune_d3 import EPOCH_BATCHES, encode, train
from .ieee_data import Ieee

N_FRAUD, N_LEGIT = 3021, 9063  # the PaySim adapter's rows
CAP_S = 3 * 3600.0
SAVE_MARGIN_S = 60.0


def training_rows(t: Ieee, seed: int = SEED) -> np.ndarray:
    """N_FRAUD + N_LEGIT rows from the fit split, in a seeded random order. Never a dev or test row."""
    a, _ = ieee_data.split_bounds(len(t))
    y = t.is_fraud[:a]
    rng = np.random.default_rng(seed)
    fraud = rng.choice(np.flatnonzero(y == 1), N_FRAUD, replace=False)
    legit = rng.choice(np.flatnonzero(y == 0), N_LEGIT, replace=False)
    return rng.permutation(np.r_[fraud, legit]).astype(np.int64)


def examples(t: Ieee, rows: np.ndarray, seed: int = SEED) -> list[tuple[Decision, int]]:
    """(decision, index of the target option) per row; options as listed or reversed by a seeded coin (as finetune.examples)."""
    flip = np.random.default_rng([seed, 1]).integers(0, 2, len(rows)).astype(bool)
    out = []
    for i, rev in zip(rows, flip):
        d = ieee_brain.decision(t, int(i))
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
    args = ap.parse_args()

    t = ieee_data.load()
    rows = training_rows(t)
    n_fraud = int(t.is_fraud[rows].sum())
    info = {"dataset": "IEEE-CIS", "rows": len(rows), "fraud": n_fraud, "legit": len(rows) - n_fraud,
            "rows_sha256": finetune.rows_hash(rows)}
    print(json.dumps(info), flush=True)
    if args.dry_run:
        return

    from .d1torch import D1Torch
    from .runs import stamp

    steps = EPOCH_BATCHES if args.train else args.steps
    exs = examples(t, rows)
    info["reversed"] = int(sum(d.options != brain.OPTIONS for d, _ in exs))
    model = D1Torch.load()
    enc = encode(model, exs)
    info["prompt_tokens_mean"] = float(np.mean([len(e[0]) for e in enc]))
    info["prompt_tokens_max"] = int(max(len(e[0]) for e in enc))
    info["base_model"] = model.info()
    print(f"encoded {len(enc)} examples, {info['reversed']} reversed, {info['prompt_tokens_mean']:.0f} tokens mean / "
          f"{info['prompt_tokens_max']} max ({time.perf_counter() - t_process:.0f} s since start)", flush=True)

    out = MODELS / f"{stamp()}_lora-d1-ieee{'-smoke' if args.steps else ''}"
    out.mkdir(parents=True, exist_ok=True)
    deadline = t_process + CAP_S - SAVE_MARGIN_S if args.train else math.inf
    info.update(train(model, enc, out, steps, deadline=deadline, log=lambda m: print(m, flush=True), lora=LORA))
    info["process_s"] = time.perf_counter() - t_process
    info["cap_s"] = CAP_S
    info["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / "manifest.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"saved {out}  steps {info['steps_done']}/{steps}{' (stopped early)' if info['stopped_early'] else ''}  "
          f"loss first100 {info['loss_first100']:.4f} -> last100 {info['loss_last100']:.4f}  "
          f"peak {info['peak_mem_gib']} GiB  process {info['process_s'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
