"""LoRA fine-tune of d3-nano (Decision 3.0) on training steps → fraud/models/<time>_lora-d3/.

    python -m fraud.finetune_d3 --dry-run            # the training set only: counts and the row hash
    python -m fraud.finetune_d3 --steps 20           # smoke run: s/step and peak memory (no cap rule)
    python -m fraud.finetune_d3 --plan <smoke dir>   # the training run: N batches from the smoke s/step

What it trains (pre-registered with the eval in fraud/eval_d3ft.py, Phase 7 of fraud/ROADMAP.md):
- Rows and examples: exactly Phase 5's (fraud.finetune.training_rows / examples: steps 1–300
  minus dev hours, all fraud + 3× legit, seed 0, the seeded order coin), row hash 983ed4fa….
- Model: fraud.d3torch (d3-nano's backbone with its non-causal softmax layers and its frozen FP32
  readout). Prompt: what d3.D3Decider sends (prompt v4 as one "d3" choice question).
- Target: the option index of "no: approve it" (legit) or "yes: decline it" (fraud); loss:
  cross-entropy over the three option codes of the readout at the last position.
- LoRA r=16, alpha=32, dropout 0.05 on q/k/v/o_proj and in_proj_qkv/in_proj_z/out_proj; AdamW
  lr 2e-4, 30 warm-up steps, linear decay to 0 over N; batch 8; bf16 weights; gradient checkpointing.
- The cap (owner, 2026-10-21): N = min(1,510, floor(6,600 s / smoke s/step)), the mean s/step of
  the smoke run after its first 3 batches; no training if N < 755. The process stops stepping
  CAP_S - SAVE_MARGIN_S after it started, saves the adapter and records stopped_early.
The folder (git-ignored) holds `adapter/` and `manifest.json` (config, row hash, loss curve, step
times, peak memory, versions, adapter sha256, steps planned / done, process time).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from . import brain, finetune
from .finetune import BATCH, LR, MODELS, SEED, WARMUP_STEPS

# Phase 5's r/alpha/dropout; targets: the softmax-attention projections of the 6 attention layers and
# the Gated DeltaNet projections of the other 18, so every layer is adapted (owner, 2026-10-21).
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj"]}
N_OPTIONS = len(brain.OPTIONS)
EPOCH_BATCHES = 1510       # 12,084 Phase 5 examples // batch 8
STEP_BUDGET_S = 6600.0     # 1 h 50 min of training steps
MIN_BATCHES = 755          # half an epoch; below it training does not start (owner, 2026-10-21)
CAP_S = 7200.0             # the training run's wall-clock cap, the whole process (owner, 2026-10-21)
SAVE_MARGIN_S = 120.0      # stop stepping this long before the cap: saving the adapter and manifest


def plan_steps(s_per_step: float) -> tuple[int, bool]:
    """(N, ok): N = min(epoch, floor(step budget / s per step)); ok is False when N < MIN_BATCHES."""
    n = min(EPOCH_BATCHES, math.floor(STEP_BUDGET_S / s_per_step))
    return n, n >= MIN_BATCHES


def lr_factor(k: int, steps: int) -> float:
    """Phase 5's schedule: linear warm-up over WARMUP_STEPS, linear decay to 0 at ``steps``."""
    return min(1.0, (k + 1) / WARMUP_STEPS) * max(0.0, 1 - k / steps)


def encode(model, exs) -> list[tuple[list[int], int]]:
    """(prompt token ids, index of the target option) per finetune.examples item, rendered as
    d3.request renders it for the decider: state = context + state, instructions, options."""
    return [(model.encode(f"{d.context}\n\n{d.state}", d.instruction, list(d.options)), target) for d, target in exs]


def train(model, enc, out_dir: Path, steps: int, deadline: float = math.inf, clock=time.perf_counter,
          log=print) -> dict:
    """LoRA on model.lm for ``steps`` batches of the given order; the LR schedule is sized to
    ``steps``. Before each batch, if clock() >= deadline, training stops; the adapter is saved
    either way."""
    import torch
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(SEED)
    lm, dev = model.lm, str(model.device)
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    peft_model = get_peft_model(lm, LoraConfig(**LORA))
    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: lr_factor(k, steps))
    sync = torch.xpu.synchronize if dev.startswith("xpu") else (lambda: None)
    if dev.startswith("xpu"):
        torch.xpu.reset_peak_memory_stats()

    lm.train()
    losses, lrs, step_s, stopped, t_start = [], [], [], False, time.perf_counter()
    for k in range(steps):
        if clock() >= deadline:
            stopped = True
            log(f"hard stop before batch {k + 1}: the 2-hour cap")
            break
        e = k % (len(enc) // BATCH)
        chunk = enc[e * BATCH:(e + 1) * BATCH]
        sync()
        t0 = time.perf_counter()
        logits = model.logits([c[0] for c in chunk], N_OPTIONS)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([c[1] for c in chunk], device=dev))
        loss.backward()
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        sync()
        losses.append(round(loss.item(), 5))
        step_s.append(time.perf_counter() - t0)
        if k % 25 == 0 or k == steps - 1:
            eta = (steps - k - 1) * float(np.median(step_s[-25:]))
            log(f"step {k + 1:5d}/{steps}  loss(25) {np.mean(losses[-25:]):.4f}  {step_s[-1]:.2f} s/step  "
                f"eta {eta / 60:5.1f} min")
    lm.eval()

    adapter = Path(out_dir) / "adapter"
    peft_model.save_pretrained(adapter)
    weights = adapter / "adapter_model.safetensors"
    import peft
    import transformers
    timed = step_s[3:] or step_s
    return {
        "lora": LORA, "lr": LR, "warmup_steps": WARMUP_STEPS, "batch": BATCH, "seed": SEED,
        "steps": steps, "steps_done": len(losses), "stopped_early": stopped, "examples": len(enc),
        "losses": losses, "lrs": lrs, "loss_first100": float(np.mean(losses[:100])) if losses else None,
        "loss_last100": float(np.mean(losses[-100:])) if losses else None,
        "step_s_mean": float(np.mean(timed)) if timed else None,
        "step_s_p50": float(np.median(timed)) if timed else None, "train_s": time.perf_counter() - t_start,
        "peak_mem_gib": torch.xpu.max_memory_allocated() / 2**30 if dev.startswith("xpu") else None,
        "device": dev, "torch": torch.__version__, "peft": peft.__version__, "transformers": transformers.__version__,
        "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    }


def main() -> None:
    t_process = time.perf_counter()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="build the training set, print counts, do not train")
    g.add_argument("--steps", type=int, help="a smoke run: this many batches, to measure s/step and memory")
    g.add_argument("--plan", type=Path, metavar="SMOKE_DIR", help="the training run: N batches from the smoke s/step")
    args = ap.parse_args()

    from .capture import load_all
    from .d3torch import D3Torch
    from .runs import stamp

    if args.plan:
        smoke = json.loads((args.plan / "manifest.json").read_text(encoding="utf-8"))
        steps, ok = plan_steps(smoke["step_s_mean"])
        print(f"smoke {args.plan.name}: {smoke['step_s_mean']:.2f} s/step -> N = {steps} batches"
              f" ({steps / EPOCH_BATCHES:.0%} of the epoch)", flush=True)
        if not ok:
            print(f"N < {MIN_BATCHES}: not training; the owner decides (pre-registration, fraud/eval_d3ft.py)")
            sys.exit(2)
    else:
        steps = args.steps

    t, s = load_all()
    held = finetune.held_out_steps(t)
    rows = finetune.training_rows(t, held)
    n_fraud = int(t.is_fraud[rows].sum())
    info = {"rows": len(rows), "fraud": n_fraud, "legit": len(rows) - n_fraud,
            "rows_sha256": finetune.rows_hash(rows)}
    print(json.dumps(info), flush=True)
    if args.dry_run:
        return

    exs = finetune.examples(t, s, rows)
    info["reversed"] = int(sum(d.options != brain.OPTIONS for d, _ in exs))
    model = D3Torch.load()
    enc = encode(model, exs)
    info["prompt_tokens_mean"] = float(np.mean([len(e[0]) for e in enc]))
    info["prompt_tokens_max"] = int(max(len(e[0]) for e in enc))
    info["base_model"] = model.info()
    print(f"encoded {len(enc)} examples, {info['reversed']} reversed, "
          f"{info['prompt_tokens_mean']:.0f} tokens mean / {info['prompt_tokens_max']} max "
          f"({time.perf_counter() - t_process:.0f} s since start)", flush=True)
    if args.plan:
        info["smoke"] = {"dir": args.plan.name, "step_s_mean": smoke["step_s_mean"]}

    out = MODELS / f"{stamp()}_lora-d3{'-smoke' if args.steps else ''}"
    out.mkdir(parents=True, exist_ok=True)
    deadline = t_process + CAP_S - SAVE_MARGIN_S if args.plan else math.inf
    info.update(train(model, enc, out, steps, deadline=deadline, log=lambda m: print(m, flush=True)))
    info["process_s"] = time.perf_counter() - t_process
    info["cap_s"] = CAP_S
    info["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / "manifest.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"saved {out}  steps {info['steps_done']}/{steps}{' (stopped early)' if info['stopped_early'] else ''}  "
          f"loss first100 {info['loss_first100']:.4f} -> last100 {info['loss_last100']:.4f}  "
          f"peak {info['peak_mem_gib']} GiB  process {info['process_s'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
