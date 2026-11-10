# Final presentation (reveal.js 6.0.2)

Frog Watch, Group 13: *Transforming LLMs into classifiers for fraud detection*. 19 main slides, 5 backup
sub-slides, references. 15 minutes everything included: 12:00 of talk, 3:00 of questions. It builds on the
midterm in `../midterm-proposal/`, and takes its look from there and from `../slides/`.

Every command is in `RUN-GUIDE.md`. The brief is `slides-handoff.md`.

## Presenting

Space, N, the right arrow and a clicker's page keys walk the talk, one step at a time. At the end of a slide's steps, the down arrow opens
its backup sub-slide, if it has one; left goes back. **S** opens the speaker view with notes, timings and sources. Sliders and
buttons give the keyboard back as soon as they're used.

Presenters on the corner tags are a proposal (Implementation choice): edit `.tag` in `index.html`.

## How it's built

- **Numbers come from data.** `build_data.py` reads the committed results and writes `data.js`. `index.html`
  binds with `{{path|format}}`, in the notes too. Numbers that exist only in prose are in `manual_data.json`, each
  with its source. `tests/test_slides.py` rebuilds `data.js` and checks the key numbers.
- **A slide's state is a pure function of its step.** `data-steps="n"` adds n reveal fragments, `data-from="k"`
  shows an element from step k, and a scene's `set(step, animate)` in `js/scenes.js` is absolute. So forward,
  backward and re-entry land on the same picture. `check_deck.mjs` tests that.
- **Motion tokens** are at the top of `deck.css`: three durations, two easings. Motion is transforms and opacity.
- **Print.** `?print-pdf` draws every final state at once. A slide that only sets up an Auto-Animate has
  `data-print-skip`. Controls have `no-print`, and what replaces them has `print-only`.
- **No network.** reveal.js 6.0.2 is in `vendor/reveal/` (MIT, sha512 checked against npm on 2026-11-04) and
  Archivo in `vendor/fonts/` (SIL OFL). No package is added: `cdp.mjs` drives headless Edge with Node's own WebSocket.
- **No model runs here,** and no IEEE-CIS row appears: window-level results, plus the committed PaySim trace,
  labelled as PaySim.
