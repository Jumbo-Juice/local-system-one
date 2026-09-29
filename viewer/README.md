# Master Viewer

One browser page for the runs of every game demo:

```bash
.venv/Scripts/python -m viewer
```

It opens a start page with three ways to watch:

- **Load a trace**: pick or drop any `trace.jsonl`.
- **Random run**: one run from the pool, in the full view.
- **Auto-demo**: two panes play the pool from newest to oldest. When a pane's run ends, it shows
  the outcome for 3 s and takes the next older run. After the oldest, the pool is read again and
  the panes start over at the newest.

A filter (All / Shooter / Dungeon / ...) limits Random run and Auto-demo to one game. The page only
replays recorded decisions; it never runs a model.

```bash
.venv/Scripts/python -m viewer runs/shooter/<run>.jsonl           # open one run
.venv/Scripts/python -m viewer --host 0.0.0.0                     # reachable from other devices
.venv/Scripts/python -m viewer --bundle runs/shooter/<run>.jsonl  # one self-contained .html to share
```

## Files

| file | what it does |
|---|---|
| `index.html` | the page: markup and styles for the start page, the full view and the auto-demo |
| `shell.js` | modes, the run pool, playback, the auto-demo queue, and the page around the stage (decision inspector, "About this run") |
| `core.js` | `Games` (the registry of game viewers) and `Run` (trace.jsonl → a run; playback time → tick) |
| `paint.js` | canvas helpers shared by the game viewers (text, fit, wrap, panels, pills, colour maths) |
| `games/<game>.js` | **one viewer per game demo**: draws that game's stage |
| `server.py` | `python -m viewer`: serves this folder, the pools `runs/<game>/` and `/api/runs`; builds `--bundle` pages |

The shell knows nothing about any game's rules. A game viewer knows nothing about pools, modes or
playback controls.

## Adding a viewer for a new game demo

Every game demo gets its own viewer. For a demo named `platformer`:

1. The demo's capture writes its runs to `runs/platformer/` (`demo.runs.write_run("platformer", ...)`)
   with `"scenario": "platformer"` in the trace header.
2. Add `viewer/games/platformer.js` and list it in `index.html` next to the other games.
3. Start from `games/dungeon.js` (the smaller one). The file calls `Games.add({...})` with:

| field | what it is |
|---|---|
| `scenario` | `"platformer"`: the header's scenario and the pool's folder name |
| `title` | shown in the filter, pane titles and page header |
| `tiers` | the decision tiers in the order to show them, e.g. `["strategy", "target", "move"]` |
| `tierLabels` | display names per tier |
| `prepare(run)` | per-run work done once: static map layer, timeline, lookups (store them on `run`) |
| `drawStage(ctx, run, clock, playing)` | the full 1920×1080 stage at playback time `clock` |
| `pane(run)` | the part of the stage an auto-demo pane shows, `{x, y, w, h}` in stage coordinates |
| `drawPane(ctx, run, clock)` | draws that part (the shell has already shifted `ctx` to it) |
| `verdict(run)` | `[word, tone, detail]` for the outcome banner; tone is `mint`, `red` or `amber` |
| `facts(run)` | titled fact cards: `[[title, [[label, value, hint?], ...]], ...]` |
| `notes(run)` | titled note cards: `[[title, [bullet, ...]], ...]` ("Playback and video" is added by the shell) |
| `inspectNote` | one paragraph above the per-tick decision inspector |
| `exampleFrom` | which decision the header's `example_prompt` came from, e.g. `"move head"` |

`Run.parse` already gives every run `header`, `ticks`, `end`, `n`, `start`/`dur`/`total` (playback
times), `last[tier]` (the last tick that decided each tier), `map` and latency figures; use
`Run.moment(run, clock)` for the tick, its fraction and the world before and after it, and
`Run.held(run, tier, i)` for the decision in force.

"About this run" keeps one layout for every game: an outcome banner, facts as titled cards with one
fact per row (a short value; any explanation goes in the hint under it), and notes as short bullets
grouped by topic.
