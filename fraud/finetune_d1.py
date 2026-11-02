"""LoRA fine-tune of d1-Omni-2B on training steps, one hour → fraud/models/<time>_lora-d1/.

    python -m fraud.finetune_d1 --steps 20           # smoke run: s/step and peak memory (no cap rule)
    python -m fraud.finetune_d1 --plan <smoke dir>   # the training run: N batches from the smoke s/step

What it trains (pre-registered with the eval in fraud/eval_d1ft.py, Phase 9 of fraud/ROADMAP.md):
- Rows and examples: exactly Phase 5's and Phase 7's (fraud.finetune.training_rows / examples), row
  hash 983ed4fa….
- Model: fraud.d1torch (the LFM3 text layers, tied-embedding code readout). Prompt: prompt v4 as the
  lone "choice" prompt of each example, in its seeded option order.
- Target: the option index of "no: approve it" (legit) or "yes: decline it" (fraud); loss:
  cross-entropy over the three option scores divided by calibration.json's temperature (3.383), the
  function the decider reads.
- LoRA r=16, alpha=32, dropout 0.05 on the attention projections (q/k/v/o_proj) and the convolution
  layers' in_proj and out_proj, so every layer is adapted; AdamW lr 2e-4, 30 warm-up steps, linear
  decay to 0 over N; batch 8; bf16 weights; gradient checkpointing (as Phases 5 and 7).
- The cap (owner, 2026-11-02: "train d1 omni for an hour"): the training process takes at most CAP_S
  of wall clock, start-up and saving included. N = min(epoch, floor((CAP_S - SAVE_MARGIN_S - the
  smoke run's start-up) / the smoke's s/step after its first 3 batches)); no floor on N: an hour is
  what the owner asked for. The process stops stepping at CAP_S - SAVE_MARGIN_S, saves and records
  stopped_early.
The folder (git-ignored) holds `adapter/` and `manifest.json`, as Phase 7.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from . import brain, finetune
from .finetune import MODELS
from .finetune_d3 import EPOCH_BATCHES, encode, train

LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj"]}
CAP_S = 3600.0         # the training run's wall-clock cap, the whole process (owner, 2026-11-02)
SAVE_MARGIN_S = 60.0   # stop stepping this long before the cap: saving the adapter and manifest


def plan_steps(s_per_step: float, startup_s: float) -> int:
    """N = min(epoch, floor(what the hour leaves for steps / s per step))."""
    return min(EPOCH_BATCHES, math.floor((CAP_S - SAVE_MARGIN_S - startup_s) / s_per_step))


def main() -> None:
    t_process = time.perf_counter()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--steps", type=int, help="a smoke run: this many batches, to measure s/step and memory")
    g.add_argument("--plan", type=Path, metavar="SMOKE_DIR", help="the training run: N batches from the smoke s/step")
    args = ap.parse_args()

    from .capture import load_all
    from .d1torch import D1Torch
    from .runs import stamp

    if args.plan:
        smoke = json.loads((args.plan / "manifest.json").read_text(encoding="utf-8"))
        startup = smoke["process_s"] - smoke["train_s"]
        steps = plan_steps(smoke["step_s_mean"], startup)
        print(f"smoke {args.plan.name}: {smoke['step_s_mean']:.2f} s/step, start-up {startup:.0f} s -> N = {steps} "
              f"batches ({steps / EPOCH_BATCHES:.0%} of the epoch)", flush=True)
    else:
        steps = args.steps

    t, s = load_all()
    rows = finetune.training_rows(t, finetune.held_out_steps(t))
    n_fraud = int(t.is_fraud[rows].sum())
    info = {"rows": len(rows), "fraud": n_fraud, "legit": len(rows) - n_fraud, "rows_sha256": finetune.rows_hash(rows)}
    print(json.dumps(info), flush=True)

    exs = finetune.examples(t, s, rows)
    info["reversed"] = int(sum(d.options != brain.OPTIONS for d, _ in exs))
    model = D1Torch.load()
    enc = encode(model, exs)
    info["prompt_tokens_mean"] = float(np.mean([len(e[0]) for e in enc]))
    info["prompt_tokens_max"] = int(max(len(e[0]) for e in enc))
    info["base_model"] = model.info()
    print(f"encoded {len(enc)} examples, {info['reversed']} reversed, {info['prompt_tokens_mean']:.0f} tokens mean / "
          f"{info['prompt_tokens_max']} max ({time.perf_counter() - t_process:.0f} s since start)", flush=True)
    if args.plan:
        info["smoke"] = {"dir": args.plan.name, "step_s_mean": smoke["step_s_mean"]}

    out = MODELS / f"{stamp()}_lora-d1{'-smoke' if args.steps else ''}"
    out.mkdir(parents=True, exist_ok=True)
    deadline = t_process + CAP_S - SAVE_MARGIN_S if args.plan else math.inf
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
