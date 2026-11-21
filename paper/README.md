# The paper (Typst, IEEE two-column)

*Can a Small Fine-Tuned LLM Make Fraud Decisions? A Locked, Cost-Based Evaluation*: the portfolio version of the
paper, revised under `REVISION_PROMPT.md` (every step and decision in `REVISION_LOG.md`).

It is an evaluation study. It asks whether a small fine-tuned language model, read as a single-pass three-way
classifier (approve, review, decline), lowers dollar decision cost against logistic regression and boosted trees,
and whether any difference comes from the model, the information it sees, or the decision rule. Main text about
8 pages; appendices A–H.

The design is the "Paper" section of `docs/superpowers/specs/2026-11-12-final-paper-and-deck-design.md`.
Every command is in `RUN-GUIDE.md` → Paper.

## Files

| file | what |
|---|---|
| `main.typ` | the paper, in this order: the main text, 30 references and appendices A–H. The main text holds Fig. 1 (the pipeline), Fig. 2 (the customer-line trade-off), Fig. 3 (final-test differences), Table I (PaySim option order), Table II (the final test, with the GBDT read-out) and Table III (where the difference comes from). |
| `gen_tables.py` | writes the block between `// generated: start` and `// generated: end` in `main.typ`: the trade-off figure and the appendix tables and figures. It writes layout and literal `v()` keys only, never a number. |
| `build_data.py` | writes `data.json`: every `v("path\|format")` key that `main.typ` reads. It resolves each from the deck's build plus the `rev` block, which reads the revision's committed results in `fraud/results/`. |
| `revision_manual.json` | the few prose-only revision constants (feature count, dev window counts), each with its `_source` |
| `data.json` | the numbers, as text; rebuilt and compared by `tests/test_paper.py` |
| `vendor/charged-ieee/` | the IEEE template (Typst Universe `charged-ieee` 0.1.4), vendored with its licence and `SOURCE.txt`, so a build needs no network |
| `exports/transforming-llms-fraud.pdf` | the compiled paper (committed; `.gitignore` keeps `exports/` in) |

## Numbers come from data

No number in the text is typed. `main.typ` reads them with `#v("<path>|<format>")`: the path is into
`slides/build_data.py`'s `build()` (the committed results in `fraud/results/`, the pinned PaySim trace and
`slides/manual_data.json`), the format one of the deck's (`int`, `signed`, `ci`, `range`, `d2`, `usd2`, `pct`,
`abs`, `auc`, `dist`, …; none means plain text). Fig. 2 reads the raw values (no format) and draws them. A number
the deck's build lacks is added there, never typed: additive keys in `slides/build_data.py`, or a block in
`slides/manual_data.json` with a `_source` (the paper added `conventions`: the interval level and the latency
quantile).

## Rules the text follows (checked by `tests/test_paper.py`)

- No phase numbers, setup ids, window ids, file paths or "econometric" in the body. File paths appear only in the
  "Code and results" availability statement (Data and Code Availability, after the conclusion), between the `code-and-data` markers.
- Every number carries its label: `dev`, `held-out` (the registered final test) or `read-out`.
- The settled framing appears verbatim: the evaluation title, "no detectable saving", "not a fairness audit". The
  standard terms are named: Bayes minimum-risk rule, Platt scaling, operating point, verbalizer, option-order bias,
  post-hoc.
- No student IDs, group name, "proposal" or "deck". The authors are listed with Siam University only, and an
  acknowledgement says the work began as a course project.
- References:
  - 30 entries, each checked against a primary source (`REVISION_LOG.md`, Stage 4);
  - listed by key (`REFS`) in order of first citation in the text, and each cited;
  - System One is credited in a footnote, not in the list.
- The one IEEE-CIS row shown is a fit row, in the prompt appendix (`fraud/results/prompt_example.json`).
- The generated block equals what `gen_tables.py` writes.

## Fonts

The template asks for TeX Gyre Termes and TeX Gyre Cursor, which this machine doesn't have. `main.typ` names
`("TeX Gyre Termes", "Times New Roman")` in show-set rules on `text`, `place` and `footnote.entry` before the
template, in a set rule after it, and around the abstract and index terms, so the whole paper, title and abstract
included, is set in Windows' Times New Roman (`pdffonts` lists only Times New Roman and the math font). Typst
still prints eight "unknown font family" warnings: seven for Termes (one from the template, six from main.typ's
uses of the tuple) and one for Cursor (the template). They are harmless.
