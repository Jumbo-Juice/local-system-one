"""LoRA fine-tune of the fraud brain (Qwen2.5-1.5B) on training steps → fraud/models/<time>_lora/.

    python -m fraud.finetune --dry-run      # the training set only: counts and the row hash
    python -m fraud.finetune                # train (about 1 h on the Lenovo), save adapter + manifest
    python -m fraud.finetune --steps 20     # a short smoke run (same data, first 20 batches)

What it trains (pre-registered with the eval in fraud/eval_ft.py, Phase 5 of fraud/ROADMAP.md):
- Rows: steps 1–300, TRANSFER and CASH_OUT only (the rows the type filter passes to the model),
  minus every hour of dev0–dev9 (those pick the thresholds). All fraud rows, plus
  LEGIT_PER_FRAUD legit rows per fraud row sampled uniformly with SEED, in a seeded random order.
- Prompt: exactly what the hybrid sends (brain.decision, frozen prompt v4, rendered by the engine).
  Each example is shown with its options as listed or reversed (a seeded coin), because order
  debias reads both orders at inference.
- Target: the label token of "no: approve it" for legit rows and "yes: decline it" for fraud rows.
  Loss: cross-entropy at the answer position only. "unsure" is never a target; the review action
  comes from the dev thresholds (brain.ThresholdDecider).
- LoRA r=16, alpha=32, dropout 0.05 on q/k/v/o_proj; AdamW lr 2e-4, linear warm-up (30 steps) and
  decay to 0; batch 8; 1 epoch; bf16 weights; gradient checkpointing; torch seed SEED.
The adapter folder (git-ignored by the repo's `models/` rule) holds `adapter/` and `manifest.json`
(config, training-row hash, loss curve, step times, peak memory, versions, adapter sha256).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from system_one import Decision

from . import brain, windows
from .data import TYPE_ID, Transactions

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
REPO = ROOT.parent

SEED = 0
LEGIT_PER_FRAUD = 3
DEV_SEEDS = tuple(range(0, 10))  # = fraud.eval.DEV_SEEDS (not imported: eval pulls in the engine stack)
MODEL_TYPES = (TYPE_ID["TRANSFER"], TYPE_ID["CASH_OUT"])
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}
LR, WARMUP_STEPS, BATCH, EPOCHS = 2e-4, 30, 8, 1


# ---------------------------------------------------------------- the training set

def held_out_steps(t: Transactions, size: int = 500, fraud_target: int = 25) -> set[int]:
    """Every hour covered by dev0–dev9."""
    out: set[int] = set()
    for k in DEV_SEEDS:
        a, b = windows.make(t, "dev", k, size=size, fraud_target=fraud_target).steps
        out |= set(range(a, b + 1))
    return out


def training_rows(t: Transactions, held: set[int], seed: int = SEED) -> np.ndarray:
    """All fraud + LEGIT_PER_FRAUD× legit rows of the training pool, in a seeded random order."""
    lo, hi = windows.TRAIN
    pool = ((t.step >= lo) & (t.step <= hi) & np.isin(t.type, MODEL_TYPES)
            & ~np.isin(t.step, sorted(held)))
    fraud = np.flatnonzero(pool & (t.is_fraud == 1))
    legit = np.flatnonzero(pool & (t.is_fraud == 0))
    rng = np.random.default_rng(seed)
    pick = rng.choice(legit, LEGIT_PER_FRAUD * len(fraud), replace=False)
    return rng.permutation(np.r_[fraud, pick]).astype(np.int64)


def examples(t: Transactions, s: dict, rows: np.ndarray, seed: int = SEED) -> list[tuple[Decision, int]]:
    """(decision, index of the target option) per row; options as listed or reversed by a seeded coin."""
    flip = np.random.default_rng([seed, 1]).integers(0, 2, len(rows)).astype(bool)
    out = []
    for i, rev in zip(rows, flip):
        d = brain.decision(t, s, int(i))
        if rev:
            d = Decision(d.instruction, d.options[::-1], state=d.state, context=d.context)
        want = brain.OPTIONS[brain.ACTIONS.index("decline" if t.is_fraud[i] else "approve")]
        out.append((d, d.options.index(want)))
    return out


def encode(engine, exs: list[tuple[Decision, int]]) -> list[tuple[list[int], int]]:
    """(prompt token ids, target label token) per example, through the engine's own prompt and
    label-token mapping, so training sees exactly what inference reads."""
    out = []
    for d, target in exs:
        labels = engine.labels_for(len(d.options))
        text = engine.render(d, labels)
        ids = engine.backend.encode(text)
        _, tokens, _ = engine._map(d, text, ids, labels)
        out.append((ids, int(tokens[target])))
    return out


def rows_hash(rows: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(rows, np.int64).tobytes()).hexdigest()


# ---------------------------------------------------------------- training

def make_train_engine():
    """The repo's default engine (config/default.toml) with the model's own bf16 head and no
    prefix cache: its tokenizer renders the training prompts and its model is the one trained."""
    import torch

    from system_one import load_config, make_engine

    torch.manual_seed(SEED)
    cfg = load_config(REPO / "config" / "default.toml")
    cfg["backend"].update({"head_dtype": "model", "prefix_cache": False})
    return make_engine(cfg)


def train(engine, enc: list[tuple[list[int], int]], out_dir: Path, steps: int | None = None, log=print) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(SEED)
    backend = engine.backend
    dev = backend.device
    model = backend.model
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
    params = [p for p in model.parameters() if p.requires_grad]
    total = (len(enc) // BATCH) * EPOCHS if steps is None else steps
    opt = torch.optim.AdamW(params, lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / WARMUP_STEPS) * max(0.0, 1 - k / total))
    sync = torch.xpu.synchronize if dev.startswith("xpu") else (lambda: None)
    if dev.startswith("xpu"):
        torch.xpu.reset_peak_memory_stats()

    model.train()
    losses, step_s, t_start = [], [], time.perf_counter()
    for k in range(total):
        e = k % (len(enc) // BATCH)
        chunk = enc[e * BATCH:(e + 1) * BATCH]
        sync()
        t0 = time.perf_counter()
        ids, mask, pos = backend._pad_left([c[0] for c in chunk])
        logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False,
                       logits_to_keep=1).logits[:, -1, :].float()
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([c[1] for c in chunk], device=dev))
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        sync()
        losses.append(round(loss.item(), 5))
        step_s.append(time.perf_counter() - t0)
        if k % 25 == 0 or k == total - 1:
            recent = float(np.mean(losses[-25:]))
            eta = (total - k - 1) * float(np.median(step_s[-25:]))
            log(f"step {k + 1:5d}/{total}  loss(25) {recent:.4f}  {step_s[-1]:.2f} s/step  eta {eta / 60:5.1f} min")

    adapter = out_dir / "adapter"
    model.save_pretrained(adapter)
    weights = adapter / "adapter_model.safetensors"
    import peft
    import transformers
    return {
        "lora": LORA, "lr": LR, "warmup_steps": WARMUP_STEPS, "batch": BATCH, "epochs": EPOCHS,
        "steps": total, "examples": len(enc), "seed": SEED, "base_model": backend.model_name,
        "losses": losses,
        "loss_first100": float(np.mean(losses[:100])), "loss_last100": float(np.mean(losses[-100:])),
        "step_s_p50": float(np.median(step_s[3:] or step_s)), "train_s": time.perf_counter() - t_start,
        "peak_mem_gib": torch.xpu.max_memory_allocated() / 2**30 if dev.startswith("xpu") else None,
        "device": dev, "torch": torch.__version__, "peft": peft.__version__, "transformers": transformers.__version__,
        "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="build the training set, print counts, do not train")
    ap.add_argument("--steps", type=int, help="stop after this many batches (smoke run)")
    args = ap.parse_args()

    from .capture import load_all
    from .runs import stamp

    t, s = load_all()
    held = held_out_steps(t)
    rows = training_rows(t, held)
    n_fraud = int(t.is_fraud[rows].sum())
    info = {"rows": len(rows), "fraud": n_fraud, "legit": len(rows) - n_fraud,
            "held_out_steps": sorted(held), "rows_sha256": rows_hash(rows),
            "types": {k: int((t.type[rows] == TYPE_ID[k]).sum()) for k in ("TRANSFER", "CASH_OUT")}}
    print(json.dumps({k: v for k, v in info.items() if k != "held_out_steps"}), flush=True)
    if args.dry_run:
        return

    exs = examples(t, s, rows)
    info["reversed"] = int(sum(d.options != brain.OPTIONS for d, _ in exs))
    engine = make_train_engine()
    enc = encode(engine, exs)
    info["prompt_tokens_mean"] = float(np.mean([len(e[0]) for e in enc]))
    info["prompt_tokens_max"] = int(max(len(e[0]) for e in enc))
    print(f"encoded {len(enc)} examples, {info['reversed']} reversed, "
          f"{info['prompt_tokens_mean']:.0f} tokens mean / {info['prompt_tokens_max']} max", flush=True)

    out = MODELS / f"{stamp()}_lora{'-smoke' if args.steps else ''}"
    out.mkdir(parents=True, exist_ok=True)
    info.update(train(engine, enc, out, steps=args.steps, log=lambda m: print(m, flush=True)))
    info["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / "manifest.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"saved {out}  loss first100 {info['loss_first100']:.4f} → last100 {info['loss_last100']:.4f}  "
          f"peak {info['peak_mem_gib']} GiB", flush=True)


if __name__ == "__main__":
    main()
