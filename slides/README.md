# Final presentation (reveal.js 6.0.2)

Frog Watch, Group 13: *Transforming LLMs into classifiers for fraud detection*. 15 static slides and an
unpresented references slide, in the midterm's look (being rewritten: the frame, the title and a sample slide are in). 15 minutes everything included: 12:00 of talk, 3:00 of questions. It builds on the
midterm in `../midterm-proposal/`, and takes its look from there and from `../slides/`.

Every command is in `RUN-GUIDE.md`. The brief is `slides-handoff.md`.

## Presenting

Space, N, the right arrow and a clicker's page keys walk the talk, one step at a time; left goes back.
**S** opens the speaker view with notes, timings and sources. There are no backup sub-slides.

Presenters on the corner tags are a proposal (Implementation choice): edit `.tag` in `index.html`.

## How it's built

- **Numbers come from data.** `build_data.py` reads the committed results and writes `data.js`. `index.html`
  binds with `{{path|format}}`, in the notes too. Numbers that exist only in prose are in `manual_data.json`, each
  with its source. `tests/test_slides.py` rebuilds `data.js` and checks the key numbers.
- **A comparison stated in prose is the console's sentence.** Every difference runs in registered order, the first
  setup minus the second (`build_data.diff()` refuses the other way round). Each difference in `data.js` carries `said`, the
  analyst console's sentence word for word (`tests/test_slides.py` runs the console's own code to check). Tiles and
  tables keep their numbers, and their verdict comes from the data through `|dist`, never typed.
- **Static.** No scenes, no Auto-Animate, no sliders. `data-steps="n"` adds n reveal fragments and `data-from="k"`
  shows an element from step k, so a slide's picture is a function of its step: forward, backward and re-entry land
  on the same picture. `check_deck.mjs` tests that. Charts are static SVG drawn by `js/charts.js` from bound numbers.
- **Print.** `?print-pdf` draws every slide at its last step (`js/core.js`), one page per slide.
- **No network.** reveal.js 6.0.2 is in `vendor/reveal/` (MIT, sha512 checked against npm on 2026-11-04) and
  Archivo in `vendor/fonts/` (SIL OFL). No package is added: `cdp.mjs` drives headless Edge with Node's own WebSocket.
- **No model runs here,** and no IEEE-CIS row appears: window-level results, plus the committed PaySim trace,
  labelled as PaySim.
