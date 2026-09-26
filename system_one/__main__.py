"""Command line.

    python -m system_one [--config FILE] check
    python -m system_one [--config FILE] decide --question "..." --options a b c [--state "..."]

``check`` verifies a backend on a new machine: one forward pass returns full next-token
logits for a batch, labels map to distinct single tokens, and batched decisions match
sequential ones.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from . import Decision, load_config, make_engine


def check(engine) -> int:
    be = engine.backend
    print("backend:", json.dumps(be.info()))
    seqs = [be.encode(be.render_chat("Answer briefly.", "Capital of France? One word.")), be.encode("One two three")]
    t = time.perf_counter()
    logits = be.next_token_logits(seqs)
    print(f"logits: shape {logits.shape} (vocab {be.vocab_size}), finite={bool(np.isfinite(logits).all())}, "
          f"first call {1000 * (time.perf_counter() - t):.0f} ms, top tokens "
          f"{[be.decode([int(r.argmax())]) for r in logits]}")
    ok = logits.shape == (2, be.vocab_size) and bool(np.isfinite(logits).all())
    for n in (5, 26, 60):
        prep = engine.prepare(Decision("pick", tuple(f"option {i}" for i in range(n))))
        distinct = len(set(prep.tokens)) == n
        print(f"labels for {n} options: {prep.labels[:3]}... distinct single tokens: {distinct}")
        ok &= distinct
    targets = ("north", "south", "east", "west", "north", "east")
    ds = [Decision("Which move brings you closer to the target?", ("north", "south", "east", "west"),
                   state=f"The target is {i + 1} cell{'s' if i else ''} {d} of you.")
          for i, d in enumerate(targets)]
    batched, seq = engine.decide_batch(ds), engine.decide_sequential(ds)
    dp = max(abs(a - b) for rb, rs in zip(batched, seq) for a, b in zip(rb.probs, rs.probs))
    same = sum(rb.index == rs.index for rb, rs in zip(batched, seq))
    print(f"batched vs sequential: same choice {same}/{len(ds)}, max |dp| {dp:.4f}; "
          f"choices {[r.choice for r in batched]}")
    right = sum(r.choice == t for r, t in zip(batched, targets))
    print(f"model answers correct: {right}/{len(ds)} (informational: this check tests the "
          f"mechanics, not the model's judgement)")
    print("OK (mechanics)" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m system_one")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    d = sub.add_parser("decide")
    d.add_argument("--question", required=True)
    d.add_argument("--options", nargs="+", required=True)
    d.add_argument("--state", default="")
    d.add_argument("--context", default="")
    args = ap.parse_args()
    engine = make_engine(load_config(args.config))
    if args.cmd == "check":
        return check(engine)
    r = engine.decide(Decision(args.question, tuple(args.options), state=args.state, context=args.context))
    for opt, p in r.ranked():
        print(f"{p:6.3f}  {opt}{'   <- choice' if opt == r.choice else ''}")
    print(f"outside mass {r.outside_mass:.4f}  method {r.method}  prompt tokens {r.prompt_tokens}  "
          f"forward {1000 * engine.last_stats['forward_s']:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
