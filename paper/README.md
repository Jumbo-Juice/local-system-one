# The paper (Typst, IEEE two-column)

Frog Watch, Group 13: *Transforming LLMs into Classifiers for Fraud Detection*. The written companion to the
final deck in `slides/`: how a small language model becomes a three-way decision model (approve, review,
decline) read in one forward pass, how it was trained and judged in dollars, and what held on a locked final
test. It is where every bias is discussed at length; the deck shows only two.

The design is the "Paper" section of `docs/superpowers/specs/2026-11-12-final-paper-and-deck-design.md`.
Every command is in `RUN-GUIDE.md` → Paper.

## Files

| file | what |
|---|---|
| `main.typ` | the paper: sections, Fig. 1 (the pipeline) and Fig. 2 (final-test differences with intervals) drawn in Typst, Table I (the worked payment in both option orders), Table II (the final test), the 14 references |
| `build_data.py` | writes `data.json`: every `v("path\|format")` key that `main.typ` reads, resolved from the deck's build and formatted as the deck formats it |
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
  "Code and data" footnote, between the `code-and-data` markers.
- The one payment shown is the committed PaySim trace's, labelled PaySim; IEEE-CIS appears only at window level.
- "softmax (multinomial logistic)" at most once; the negatives said plainly: "no setup showed a distinguishable
  saving", "fairness was not measured".
- References [1]–[13] as in the midterm plan, [14] the d1-Omni-2B model card, written as a numbered list between
  the `references` markers; each is cited in the text.

## Fonts

The template asks for TeX Gyre Termes and TeX Gyre Cursor, which this machine doesn't have. `main.typ` names
`("TeX Gyre Termes", "Times New Roman")` in show-set rules on `text`, `place` and `footnote.entry` before the
template, in a set rule after it, and around the abstract and index terms, so the whole paper, title and abstract
included, is set in Windows' Times New Roman (`pdffonts` lists only Times New Roman and the math font). Typst
still prints eight "unknown font family" warnings: seven for Termes (one from the template, six from main.typ's
uses of the tuple) and one for Cursor (the template). They are harmless.
