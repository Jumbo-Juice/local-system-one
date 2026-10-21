"""Exploratory (Phase 7, after the reproduction check failed R3; not pre-registered): our d3torch forward
with simulated llama.cpp Q8_0 weights vs the Phase 6 Q8_0 traces and our BF16 repro traces, first N dev0 rows.

    .venv/Scripts/python -u fraud/dev/d3_q8sim.py 150
"""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import numpy as np, torch
from fraud import d3
from fraud.brain import OPTIONS
from fraud.capture import load_all
from fraud.d3torch import D3Torch
from fraud.eval_ft import trace_file
from fraud.runs import read_run


def q8_0_(w):
    """In place: blocks of 32 along the input dim, int8 with one fp16 scale (llama.cpp Q8_0), back to bf16."""
    rows, cols = w.shape
    if cols % 32:
        return False
    for a in range(0, rows, 4096):  # row chunks: the 248k-row embedding does not fit as one FP32 copy
        x = w.data[a:a + 4096].float().view(-1, cols // 32, 32)
        d = (x.abs().amax(-1, keepdim=True) / 127).half().float()
        q = torch.where(d > 0, (x / d.clamp_min(1e-30)).round().clamp(-127, 127), torch.zeros_like(x))
        w.data[a:a + 4096].copy_((q * d).view(-1, cols).to(w.dtype))
    return True


N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
t, s = load_all()
ours = [r for r in read_run(trace_file("20261021-130745", "repro", "hybrid-d3torch", "dev0")) if r.get("type") == "tx" and r["by"] == "model"][:N]
ref = {r["row"]: r for r in read_run(trace_file("20261021-020632", "dev", "hybrid-d3nano", "dev0")) if r.get("type") == "tx" and r["by"] == "model"}
m = D3Torch.load()
done = 0
with torch.no_grad():
    for name, mod in m.lm.named_modules():
        if isinstance(mod, (torch.nn.Linear, torch.nn.Embedding)) and mod.weight.dim() == 2:
            done += q8_0_(mod.weight)
print("quantized matrices:", done, flush=True)
t0 = time.perf_counter()
sim = [m.post(d3.request(t, s, r["row"], "both"))["answers"] for r in ours]
print(f"{len(ours)} rows in {time.perf_counter() - t0:.0f} s")
for j, q in enumerate(("fwd", "rev")):
    o = np.array([a[q]["probabilities"][OPTIONS[2]] for a in sim])
    a = np.array([r["orders"][j]["decline"] for r in ours])
    b = np.array([ref[r["row"]]["orders"][j]["decline"] for r in ours])
    print(f"{q}: |simQ8 - Q8_0 trace| median {np.median(abs(o - b)):.4f} (signed mean {np.mean(o - b):+.4f}) | "
          f"|BF16 ours - Q8_0 trace| median {np.median(abs(a - b)):.4f} (signed {np.mean(a - b):+.4f}) | "
          f"|simQ8 - BF16 ours| median {np.median(abs(o - a)):.4f} (signed {np.mean(o - a):+.4f})")
