---
status: accepted
---

# The story reads both applications and belongs to neither

The game demos and the fraud app are kept apart: neither imports from, writes for, or appears in
the other's pages. On 2026-11-12 the owner asked for a story that puts the whole project's findings
to one question ("Can a model that never writes be trusted to decide?"), with game and fraud
evidence side by side in the same trial. We decided the story is a third surface in its own
top-level folder. It only reads what both applications have committed (results, pinned runs,
figures); nothing in `demo/`, `viewer/`, `fraud/` or `slides/` imports from it or writes for it,
and the kept-apart rule between the two applications stays as it is.

## Considered options

- **Rising stakes, one act per application** (engine, then games, then fraud). It keeps the two
  apart on the page. The owner chose the single question instead, because the same answer, "only
  with help", shows up in both, and that is the finding.
- **Add a story mode to the deck or the console.** The deck is fraud only and written for a
  presenter; the console replays runs for an analyst. Neither is for a newcomer reading alone.

## Consequences

- The story runs no model and shows no IEEE-CIS row, as the deck does: window-level results and
  committed PaySim traces only.
- Its numbers are built from the committed result files, never typed, so they can't drift from the
  deck's or the console's.
- GSAP is vendored for it, at the owner's request. It is the only page in the repo that uses it.
