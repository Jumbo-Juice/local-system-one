"""Prompt development on dev windows only (never test windows)."""
import sys, time
import numpy as np
from system_one import load_config, make_engine, Decision
from fraud import brain, costs, windows, rules
from fraud.capture import load_all
from fraud.data import TYPES

t, s = load_all()
cfg = load_config(None); cfg["engine"]["order_debias"] = True; cfg["backend"]["head_dtype"] = "model"
eng = make_engine(cfg); brain.warm_up(eng)
DEV = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1, 2]
rows = []
for k in DEV:
    w = windows.make(t, "dev", k)
    rows += [int(i) for i in w.rows if t.type[i] not in rules.FRAUD_FREE]
y = t.is_fraud[rows]
print("decisions", len(rows), "fraud", int(y.sum()), flush=True)

def times(n): return "never" if n == 0 else "once" if n == 1 else f"{n} times"

def d_v1(i):
    kind = TYPES[t.type[i]]
    din = int(s["dest_in_before"][i]); same = int(s["same_amount_step_before"][i])
    recv = "NEW account: it never received money before" if din == 0 else f"known account: received money {times(din)} before"
    amt = f"the SAME amount was already moved {times(same)} earlier this hour" if same else "no earlier transaction this hour had this amount"
    return "\n".join([f"{kind} of {t.amount[i]:,.2f}, hour {int(s['hour'][i])}:00.",
                      f"Receiver: {recv}.", f"Amount: {amt}.",
                      f"Round amount: {'yes' if s['round_amount'][i] else 'no'}."])

CTX0 = brain.CONTEXT
CTX2 = ("Mobile-money service. Fraudsters take over an account, TRANSFER all its money to a fresh account "
        "that never received money before, then CASH_OUT exactly that amount within the same hour. "
        "Legitimate TRANSFERs usually go to known accounts; legitimate CASH_OUTs rarely repeat an amount "
        "moved earlier in the hour. Most transactions are legitimate.")
OPT3 = ("approve: normal customer activity", "review: suspicious, ask an analyst", "decline: fraud, block it")

OPT4 = ("no: approve it", "unsure: send it to an analyst", "yes: decline it")
Q4 = "Does this transaction match the fraud pattern?"
CTX6 = CTX2 + " Only about 1 in 10 TRANSFERs or CASH_OUTs here is fraud."
VARIANTS = {
    "v0": (lambda i: brain.describe(t, s, i), CTX0, brain.OPTIONS),
    "v1": (d_v1, CTX0, brain.OPTIONS),
    "v2": (d_v1, CTX2, brain.OPTIONS),
    "v3": (d_v1, CTX2, OPT3),
    "v4": (d_v1, CTX2, OPT4, Q4),
    "v6": (d_v1, CTX6, brain.OPTIONS),
    "v7": (d_v1, CTX6, OPT4, Q4),
}
pick = sys.argv[2].split(",") if len(sys.argv) > 2 else list(VARIANTS)
for name in pick:
    desc, ctx, opts, *q = VARIANTS[name]
    q = q[0] if q else brain.QUESTION
    P, ms, toks = [], [], []
    for i in rows:
        d = Decision(q, opts, state=desc(i), context=ctx)
        t0 = time.perf_counter(); r = eng.decide_batch([d])[0]; ms.append((time.perf_counter() - t0) * 1000)
        P.append(r.probs); toks.append(r.prompt_tokens)
    P = np.array(P); act = P.argmax(1)
    def auc(p):
        pos, neg = p[y == 1], p[y == 0]
        return (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    risk = P[:, 1] + P[:, 2]
    tbl = {a: (int(((act == k) & (y == 1)).sum()), int(((act == k) & (y == 0)).sum())) for k, a in enumerate(brain.ACTIONS)}
    print(f"{name}: AUC(review+decline) {auc(risk):.3f} AUC(decline) {auc(P[:, 2]):.3f}  actions (fraud, legit) {tbl}  "
          f"p50 {np.percentile(ms, 50):.0f} p90 {np.percentile(ms, 90):.0f} ms  tokens {np.mean(toks):.0f}", flush=True)
