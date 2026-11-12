# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Static HTML, CSS and JavaScript with no build step and no Node project. Libraries are vendored and
checked (reveal.js for the deck, GSAP for the story), so every page runs with no network. Pages are
served locally with `python -m http.server` or their own small Python server; the story may later
be hosted on GitHub Pages unchanged. (Owner's answer, 2026-11-12: "all local for now but i might
host this on github io".)

## Users

- **The story:** a newcomer reading alone, such as a classmate, a grader or a curious visitor who
  missed the talk. Nobody is there to explain, so every term is explained where it first appears.
- **The deck** (`slides/`): a room, with a presenter.
- **The fraud console** (`fraud/viewer/`): an analyst replaying recorded runs.
- **The Master Viewer** (`viewer/`): someone watching recorded game runs.

## Product Purpose

local-system-one is a small local AI that makes every decision with one multiple-choice look: one
forward pass, no writing. It was tested on three games and on fraud detection, and the project
reports what worked and what didn't. Success for a reader is being able to answer, with the
evidence in hand, "Can a model that never writes be trusted to decide?"

## Positioning

Every result was pre-registered, measured on recorded runs and reported with its failures. The
honest answer is the product: it is quick enough to decide in real time, but every gain measured
came from the structure around the model or from training it, not from its judgement as it comes.

## Operating Context

- The story is read once, top to bottom, by scrolling, on a laptop or a phone.
- It asks the question in six trials. Each trial ends on one stamp: yes, no, only with help or
  can't tell. The words are defined in `GLOSSARY.md`.
- Two stops invite a try: flipping the order of the options, and weighing a payment's amount.
- Game moments are replayed from pinned run traces.

## Capabilities and Constraints

- No page runs a model. Everything shown is a recorded run or arithmetic on committed results.
- No IEEE-CIS row appears anywhere: window-level results only. PaySim traces are labelled PaySim.
- Numbers are built from committed result files, never typed, and a test checks them.
- Model scores are uncalibrated and are never shown as probabilities.
- A comparison is "distinguishable" or "not distinguishable"; nothing softer.
- The game demos and the fraud app stay apart. Only the story reads both
  (`docs/adr/0001-the-story-reads-both-applications.md`).
- This project is not Jev and does not reproduce Jev.

## Brand Commitments

- Named on the story: the project and its sources only. No people and no course.
- The story must not read as a sales page, a chopped-up deck or a lab notebook (no phase numbers,
  setup ids or file paths in the prose).
- Plain words for a newcomer; the glossary's term for each concept and no synonyms.

## Evidence on Hand

- Engine and game results: `README.md`, `docs/research.md`, `docs/lessons-learned.md`,
  `bench/results/`.
- Fraud results: `fraud/results/`, `fraud/SUMMARY.md`, `fraud/RESULTS-*.md`; the deck's data build
  `slides/build_data.py`.
- Pinned run traces: `runs/shooter/*.pinned.jsonl`, `runs/dungeon/*.pinned.jsonl`,
  `fraud/runs/*.pinned.jsonl`.
- Figures and GIFs: `docs/media/`.
- Not on hand, and not to be invented: final-test traces, IEEE-CIS rows, fairness results,
  testimonials, deployment claims.

## Product Principles

1. Report the negative results as plainly as the positive ones.
2. A claim is only as strong as its evidence, and says which kind it is.
3. One word per concept.
4. The reader sets the pace; nothing depends on motion alone.

## Accessibility & Inclusion

Honour `prefers-reduced-motion` by showing final states. No information exists only in motion or
only in colour. No sideways scroll at 375 px.
