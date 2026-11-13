# Final presentation (reveal.js 6.0.2)

Frog Watch, Group 13: *Transforming LLMs into classifiers for fraud detection*. 15 static slides and an
unpresented references slide, in the midterm's look: white page, green ink, Archivo, numbered component bars and
pipeline pages. 15 minutes everything included: 12:00 of talk, 3:00 of questions. It builds on the midterm in
`../midterm-proposal/`, which stays as it is.

Every command is in `RUN-GUIDE.md`. The design is `docs/superpowers/specs/2026-11-12-final-paper-and-deck-design.md`.

## The talk

| # | slide | presenter | time |
|---|---|---|---|
| 01 | Title | Juji | 0:25 |
| 02 | The problem: approve, review or decline, before the money moves | Juji | 0:45 |
| 03 | How we judge a decision maker: accuracy flips to dollars (two steps) | Juji | 1:00 |
| 04 | Data and split: PaySim, IEEE-CIS, windows, the locked final test | Juji | 0:45 |
| 05 | Method: the rule, on the PaySim transfer | Swan | 0:45 |
| 06 | Method: logistic regression | Swan | 0:50 |
| 07 | Method: boosted trees | Swan | 0:45 |
| 08 | Method: a language model as the decision maker (logits, softmax) | Juji | 1:15 |
| 09 | From probabilities to an action: the options reversed (two steps), averaged, cut-offs | Juji | 0:45 |
| 10 | Method: training a LoRA adapter; PaySim before and after | Juji | 1:00 |
| 11 | Results: the final test, every setup against logistic regression | Ryu | 1:00 |
| 12 | Results: the trees' dev win against new windows (two steps) | Ryu | 0:45 |
| 13 | Results: what did work (customers, speed) and what the language setup copies | Ryu | 0:45 |
| 14 | Biases: assumed costs, option-order bias | Ponh | 0:45 |
| 15 | Takeaways | Ponh | 0:25 |
| 16 | References [1]–[14], not presented | | |

Presenters on the corner tags are a proposal (Implementation choice): edit `.tag` in `index.html`.

## Presenting

Space, N, the right arrow and a clicker's page keys walk the talk, one step at a time; left goes back.
**S** opens the speaker view with notes, timings and sources. There are no backup sub-slides.

## How it's built

- **Numbers come from data.** `build_data.py` reads the committed results and writes `data.js`. `index.html`
  binds with `{{path|format}}`, in the notes too. Numbers that exist only in prose are in `manual_data.json`, each
  with its source. `tests/test_slides.py` rebuilds `data.js` and checks the key numbers.
- **One payment through every method.** A real PaySim transfer from the pinned trace, picked by a written rule
  (`worked_block`). The language model slides show its recorded logits, softmax and both option orders. Rules,
  logistic regression and trees recorded no score for it, so their slides use made-up inputs, labelled
  "illustration" on the slide, and `illustration_block` derives the rest.
- **A comparison stated in prose is the console's sentence.** Every difference runs in registered order, the first
  setup minus the second (`build_data.diff()` refuses the other way round). Each difference in `data.js` carries `said`, the
  analyst console's sentence word for word (`tests/test_slides.py` runs the console's own code to check). Charts and
  tiles keep their numbers, and their verdict comes from the data through `|dist`, never typed.
- **Static.** No scenes, no Auto-Animate, no sliders. `data-steps="n"` adds n reveal fragments and `data-from="k"`
  shows an element from step k, so a slide's picture is a function of its step: forward, backward and re-entry land
  on the same picture. `check_deck.mjs` tests that. Charts are static SVG drawn by `js/charts.js` from bound numbers;
  a chart row with `data-from` appears at its step on the same axis as the others.
- **Print.** `?print-pdf` draws every slide at its last step (`js/core.js`), one page per slide.
- **No network.** reveal.js 6.0.2 is in `vendor/reveal/` (MIT, sha512 checked against npm on 2026-11-04) and
  Archivo in `vendor/fonts/` (SIL OFL). No package is added: `cdp.mjs` drives headless Edge with Node's own WebSocket.
- **No model runs here,** and no IEEE-CIS row appears: window-level results, plus the committed PaySim trace,
  labelled as PaySim.
- **Visible text** has no phase numbers, setup ids or file paths; those go in the notes' `Source:` lines.
