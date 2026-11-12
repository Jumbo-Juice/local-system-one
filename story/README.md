# The story

One scroll page that puts the whole project's findings to one question, for a newcomer reading alone:
*Can a model that never writes be trusted to decide?* Six trials, each ending on a stamp (yes, no, only
with help, can't tell), then the tally, the limits and the sources. The words are defined in
`GLOSSARY.md` → The story; why this page reads both applications is in
`docs/adr/0001-the-story-reads-both-applications.md`.

Every command is in `RUN-GUIDE.md` → The story.

## How it's built

- **Numbers come from data.** `build_data.py` reads the committed results and the pinned runs and writes
  `data.js`. Fraud numbers come through `slides/build_data.py`, the deck's own build, so the story, the
  deck and the console agree. Numbers that exist only in prose are in `manual_data.json`, each with its
  source. `index.html` binds with `data-bind="path|format"`; `tests/test_story.py` rebuilds `data.js`
  and checks the key numbers and every binding.
- **A comparison is one sentence,** the first setup minus the second, worded as the console and the deck
  word it (`S.compare` in `js/core.js`), with plain names for the setups.
- **A figure's picture is a function of the step in view.** Each trial is a column of steps beside a
  figure that stays put. A step names its panel (`data-panel`), and a panel may have a hook in
  `S.hooks` (`enter`, `scrub`). Forward, backward and a jump from the bar land on the same picture.
- **Replays are recorded runs.** `js/replay.js` draws a pinned game run tick by tick; the scroll moves
  time. The order flip is one recorded decision with the scores the run stored for each order. The rule
  explorer is arithmetic in the page.
- **Motion** is GSAP 3.15.1 with ScrollTrigger, vendored in `vendor/gsap/` (checked against the upstream
  tag, see `SOURCE.txt`; GreenSock's "no charge" licence, not MIT). One ease everywhere (`S.to`,
  `S.from`). `prefers-reduced-motion`, or `?reduced=1`, shows final states. Without the library the page
  shows every panel, stacked.
- **No network.** Figtree is in `vendor/fonts/` (SIL OFL). The only links out are the sources a reader
  may follow.
- **No model runs here,** and no IEEE-CIS row appears: results per window, pinned game runs and the
  committed PaySim trace, labelled as PaySim.

## Design

The look is the category standard for a scrolling explainer, chosen by the owner, with Distill and The
Pudding as the craft bar. The durable decisions are in `DESIGN.md` at the repo root and the page's own
brief in `.impeccable/surfaces/story-index-html.md`.
