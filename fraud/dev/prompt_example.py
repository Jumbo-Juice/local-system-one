"""The IEEE-CIS prompts verbatim, filled for one fit row, for the paper's prompt appendix → fraud/results/prompt_example.json.

    python -m fraud.dev.prompt_example

The row (Implementation choice, a written rule, not picked by eye): the first fit row, in time order, whose card has earlier
payments and whose device type is recorded, so every line of the rich prompt is filled. A fit row, so no dev or test row is
shown. Its label is recorded but plays no part in the choice. The text is what d1torch.encode feeds the model: the chat
template's tokens, the state, the question and the options in the order the final setups were scored (rev: A decline, B review,
C approve), up to the answer slot. The BOS token is left out.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .. import history, ieee_brain, ieee_data, lm_signal
from ..d1torch import prefix_text, suffix_text

OUT = Path(__file__).resolve().parent.parent / "results" / "prompt_example.json"


def main() -> None:
    t = ieee_data.load()
    hist = history.load()
    a = ieee_data.split_bounds(len(t))[0]
    before = hist["history"]["card_log_before"]
    i = next(int(k) for k in range(a) if before[k] > 0 and t.text("DeviceType", k))
    opts = ieee_brain.OPTIONS[::-1]
    full = lambda d: prefix_text("", f"{d.context}\n\n{d.state}") + suffix_text(d.instruction, opts)
    plain = ieee_brain.decision(t, i)
    rich = ieee_brain.decision(t, i, hist)
    signal = ieee_brain.decision(t, i, {**hist, "signal": lm_signal.load()})
    out = {"script": "fraud/dev/prompt_example.py", "label": "fit row (illustration)", "row_rule": "first fit row with earlier card payments and a recorded device type",
           "row": i, "is_fraud": int(t.is_fraud[i]), "options_order": "rev (A decline, B review, C approve)",
           "context": ieee_brain.CONTEXT, "question": plain.instruction, "options": list(opts),
           "state_plain": plain.state, "state_rich": rich.state, "signal_line": signal.state.splitlines()[-1],
           "full_plain": full(plain), "full_rich": full(rich)}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out["full_rich"])


if __name__ == "__main__":
    main()
