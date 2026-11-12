---
name: local-system-one, the story
description: A scrolling explainer on white. One blue for the model, four stamp colours, flat figures that stay put.
colors:
  blue: "#2563eb"
  blue-ink: "#1d4ed8"
  blue-soft: "#dbeafe"
  yes: "#15803d"
  yes-soft: "#dcfce7"
  no: "#b91c1c"
  no-soft: "#fee2e2"
  help: "#b45309"
  help-soft: "#fef3c7"
  cant: "#475569"
  cant-soft: "#e2e8f0"
  bg: "#ffffff"
  ink: "#111827"
  ink-2: "#374151"
  muted: "#4b5563"
  rule: "#e5e7eb"
  rule-2: "#d1d5db"
  track: "#f3f4f6"
typography:
  display:
    fontFamily: "Figtree, system-ui, sans-serif"
    fontSize: "clamp(2.25rem, 1.2rem + 4vw, 4rem)"
    fontWeight: 800
    lineHeight: 1.04
    letterSpacing: "-0.03em"
  headline:
    fontFamily: "Figtree, system-ui, sans-serif"
    fontSize: "clamp(1.75rem, 1.1rem + 2.6vw, 2.75rem)"
    fontWeight: 800
    lineHeight: 1.1
    letterSpacing: "-0.025em"
  title:
    fontFamily: "Figtree, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 700
    lineHeight: 1.6
  lead:
    fontFamily: "Figtree, system-ui, sans-serif"
    fontSize: "1.1875rem"
    fontWeight: 400
    lineHeight: 1.6
  body:
    fontFamily: "Figtree, system-ui, sans-serif"
    fontSize: "1.125rem"
    fontWeight: 400
    lineHeight: 1.6
    fontFeature: "tnum"
  label:
    fontFamily: "Figtree, system-ui, sans-serif"
    fontSize: "0.9375rem"
    fontWeight: 400
    lineHeight: 1.5
  chart-label:
    fontFamily: "Figtree, sans-serif"
    fontSize: "11px"
    fontWeight: 500
rounded:
  bar: "2px"
  map: "4px"
  pill: "999px"
  circle: "50%"
spacing:
  bar-h: "3.5rem"
  gutter: "clamp(1rem, 4vw, 4rem)"
  gap: "clamp(2rem, 5vw, 4.5rem)"
  col: "42.5rem"
  steps-col: "27rem"
  wrap: "76rem"
  section: "clamp(3rem, 10vh, 6rem)"
components:
  top-bar:
    backgroundColor: "{colors.bg}"
    textColor: "{colors.ink-2}"
    height: "{spacing.bar-h}"
  slot:
    backgroundColor: "{colors.bg}"
    textColor: "{colors.muted}"
    rounded: "{rounded.circle}"
    size: "1.75rem"
  slot-stamped-yes:
    backgroundColor: "{colors.yes}"
    textColor: "{colors.bg}"
  slot-stamped-no:
    backgroundColor: "{colors.no}"
    textColor: "{colors.bg}"
  slot-stamped-help:
    backgroundColor: "{colors.help}"
    textColor: "{colors.bg}"
  slot-stamped-cant:
    backgroundColor: "{colors.cant}"
    textColor: "{colors.bg}"
  stamp-mark:
    backgroundColor: "{colors.yes}"
    textColor: "{colors.bg}"
    rounded: "{rounded.circle}"
    size: "4.25rem"
  choice-pill:
    backgroundColor: "{colors.bg}"
    textColor: "{colors.blue-ink}"
    rounded: "{rounded.pill}"
    padding: "0.5rem 0.9rem"
  choice-pill-hover:
    backgroundColor: "{colors.blue-soft}"
  choice-pill-selected:
    backgroundColor: "{colors.blue}"
    textColor: "{colors.bg}"
  bar-track:
    backgroundColor: "{colors.track}"
    rounded: "{rounded.bar}"
    height: "0.875rem"
  bar-fill:
    backgroundColor: "{colors.ink-2}"
    rounded: "{rounded.bar}"
  bar-fill-model:
    backgroundColor: "{colors.blue}"
  source-note:
    textColor: "{colors.ink-2}"
    typography: "{typography.label}"
---

# Design System: local-system-one, the story

**Scope.** This design system is the story's (`story/`). The repo has three older surfaces with their own looks: the reveal.js deck (`slides/`), the fraud analyst console (`fraud/viewer/`) and the game viewer (`viewer/`). This document does not govern them. Do not restyle them to match.

Everything below was read from the shipped files: `story/story.css`, `story/index.html` and `story/js/`. Where the build and the plan differ, the build is recorded.

## Overview

**Creative North Star: "The Figure That Stays Put"**

The story is a scrolling explainer in the category standard. The owner chose that standard and named Distill and The Pudding as the craft bar. The page is white, the ink is near black and the rules are hairline grey. Text passes a figure that holds still. Each trial ends on a stamp.

Colour is sparse and every colour has one job. One blue means the model itself or something the reader can touch. Four stamp colours mean the four verdicts. Everything else is ink and grey. Figures are flat SVG, canvas or CSS with direct labels. There are no cards, no panels with fills and no decoration around a figure.

The reader is a newcomer reading alone. The page is calm so the evidence can be seen before the verdict. It is not a dark AI landing page with glowing charts and a hero metric.

**Key Characteristics:**
- White ground, near-black ink, hairline grey rules.
- One blue with a fixed meaning. Four stamp colours with fixed meanings.
- One typeface, Figtree, with tabular numerals everywhere.
- Flat figures with direct labels.
- A figure that stays put while text steps pass it.
- One ease for all motion. The stamp landing is the one authored moment.
- Reduced motion shows final states. No fact lives only in motion or only in colour.

## Colors

A white page with grey structure, one blue and four verdict colours.

### Primary
- **Model Blue** (`blue`): the model's own bar, line, dot or trail in a figure. Also the fill of a selected control, the slider track and thumb, the scroll progress line and the focus ring.
- **Model Blue Ink** (`blue-ink`): blue as text. Links, the top option's letter, slider read-outs, control labels and the "Where this comes from" toggle.
- **Model Blue Soft** (`blue-soft`): the hover fill of a choice pill and the text selection colour.

### Secondary
The stamp colours. Each has a strong value and a soft value. The soft values are defined as tokens but the build does not use them yet.
- **Yes Green** (`yes`): the yes stamp. In replays, the exit and a healthy health bar.
- **No Red** (`no`): the no stamp. In figures, danger: a harmful option, an enemy, a failed run, low health.
- **Only With Help Amber** (`help`): the only-with-help stamp. In replays, the key.
- **Can't Tell Slate** (`cant`): the can't-tell stamp. In charts, the quiet reference line and its label.

### Neutral
- **Page White** (`bg`): the ground. Also the gap ring around dots and thumbs that sit on a track.
- **Ink** (`ink`): headings, body text, zero lines and limit marks.
- **Ink 2** (`ink-2`): secondary text, bar links, and every bar or line that is not the model.
- **Muted** (`muted`): captions, hints, axis labels and the trial number inside a heading.
- **Rule** (`rule`): hairlines between sections and groups, chart grid lines.
- **Rule 2** (`rule-2`): stronger hairlines, chart axes, empty dots, unstamped slot borders.
- **Track** (`track`): the empty part of a bar and the tint behind a file path.

### Named Rules
**The One Blue Rule.** Blue means the model itself (its lean, its time, its cost) or something the reader can touch. Nothing else is blue. A comparison bar that is not the model is Ink 2.

**The Four Stamps Rule.** Green, red, amber and slate are the four verdicts. A stamp colour never appears without its drawn mark or its word. Inside a figure the same four keep the nearest meaning: good, harm, needs something, unknown.

**The Mark Carries It Rule.** Colour never stands alone. Each stamp has its own mark (tick, cross, plus, question mark) and its own words.

## Typography

**Display Font:** Figtree (variable, weights 300 to 900, self-hosted), with system-ui and sans-serif behind it.
**Body Font:** Figtree.
**Label/Mono Font:** none. File paths are set in Figtree on a Track tint.

**Character:** One friendly geometric sans does every job. Headings are heavy and tight. Body text is roomy.

### Hierarchy
- **Display** (800, fluid 2.25rem to 4rem, line height 1.04, tracking -0.03em): the question at the top. Used once. The closing takeaway uses the same weight and tracking at 1.75rem to 3.25rem, line height 1.1.
- **Headline** (800, fluid 1.75rem to 2.75rem, line height 1.1, tracking -0.025em): trial headings, closing section headings and the stamp line.
- **Title** (700, 1rem): the title above a figure. Source group headings are 700 at 1.0625rem.
- **Lead** (400, 1.1875rem, line height 1.6): the opening paragraphs and the sentence under a stamp.
- **Body** (400, 1.125rem, line height 1.6): step text. Paragraphs in steps stop at 34rem wide. Emphasis is weight 700.
- **Label** (400 to 700, 0.9375rem, line height 1.35 to 1.5): captions, hints, keys, option text and source notes. Captions stop at 60ch. The smallest text is 0.8125rem, for marks on a track and narrow-screen captions.
- **Chart label** (500, 11px, Muted; 700 for a direct label on a line): text inside an SVG chart.
- **Big number** (800, 1.5rem, line height 1.15): a fact above its one-line explanation, and the counts in the tally.

### Named Rules
**The One Family Rule.** Figtree throughout. Hierarchy comes from weight (400, 500, 600, 700, 800) and size.

**The Tabular Rule.** `font-variant-numeric: tabular-nums` is set on the body. Numbers line up everywhere and do not shift when they change.

**The Plain Words Rule.** Prose uses plain words for a newcomer and the glossary's terms: trial, stamp, help, can't tell, distinguishable. No phase numbers and no setup ids. File paths appear only in "Where this comes from" notes. Headings sit alone, with no label above them.

## Layout

The page is one column of sections inside a wrap (76rem wide at most) with a fluid gutter (1rem to 4rem). Long text holds to a reading column (42.5rem).

- **Top bar.** Sticky, 3.5rem tall, white, with a hairline under it. Project name left, links and six numbered slots right. A 3px blue progress line sits on its bottom edge.
- **Opening.** Two columns: the reading column and the first figure, with a fluid gap (2rem to 4.5rem).
- **Trial.** A hairline, a heading, then two columns. Text steps are on the left (27rem at most). The figure is on the right, sticky under the bar and centred in the height that is left. Each step is about a screen tall (82svh; 60svh for the first, 64svh for the last). A long step is 190svh and its text sticks at 36svh.
- **Stamp, then sources.** After the steps, the stamp sits in the reading column under a 2px ink rule. A closed "Where this comes from" note follows.
- **Closing.** The tally (a ruled list up to 52rem wide), the limits and the sources, each opened by a hairline.
- **Section rhythm.** Sections are separated by 3rem to 6rem, scaled by viewport height.

**Narrow screens (under 60rem).** The opening becomes one column. In a trial the figure sticks at the top, full width, on white with a hairline under it. The steps pass below it. Steps are 72svh, long steps 170svh. Figure text and gaps tighten, most keys hide, and option text is cut to one line. **Under 42rem** the bar keeps only the name and the slots, and a tally row stacks its verdict under its question. There is no sideways scroll at 375px.

**Without the motion library** every panel is shown, stacked in order, and every slot is filled.

## Elevation & Depth

The system is flat. There are no card shadows and no layered surfaces. Depth is order only: the bar and the narrow-screen figure stick above the text on opaque white, and each is closed by a hairline.

Structure comes from rules, not boxes. A 1px Rule line separates sections and groups. A 1px Rule 2 line rules the tally. A 2px Ink line opens a stamp.

### Named Rules
**The Flat Figure Rule.** Figures carry no shadows, no gradients and no fills behind them. A mark that sits on a track gets a 2px or 3px white ring to separate it, not a shadow.

## Shapes

Three shapes, each with a meaning.
- **Circles** (50%) are things that are counted or placed: slots, stamp marks, dots, the reader's position, the slider thumb.
- **Pills** (999px) are things the reader can press or drag: choice pills, the slider track.
- **Near-square bars** (2px) are measures: bar tracks and fills, key swatches, square dots. The replay map has 4px corners.

Text blocks have no boxes, borders or radius. Lines are 1px, 1.5px (slot and pill borders, empty dots) or 2px (zero lines, limit marks, the stamp rule). Drawn marks are open strokes on a 24 unit grid: stroke 2.6, round caps and joins, no fill.

## Components

Components are plain and quiet. The stamp is the only one with presence.

### Top bar and slots
- **Bar:** white, 3.5rem, hairline below. Name at 700, 1.0625rem. Links at 500, 0.9375rem, Ink 2, underlined on hover.
- **Slot:** a 1.75rem circle with a 1.5px Rule 2 border and a Muted number (700, 0.8125rem). Hover darkens border and number.
- **Stamped slot:** fills with its stamp colour and swaps the number for the white mark. It fills when its trial's stamp scrolls into view and empties if the reader scrolls back above it. Its accessible name states the verdict.
- **Progress line:** 3px, blue, grows left to right with the scroll.

### The stamp (signature)
- **Layout:** a 2px Ink rule, then a 4.25rem circle in the stamp colour with a white mark, beside the line "Stamped **verdict**" in Headline type. The verdict word takes the stamp colour. One Lead sentence follows, full width.
- **Marks:** tick for yes, cross for no, plus for only with help, question mark for can't tell.
- **Landing:** the circle drops from 1.9 scale and -14 degrees to rest in 0.7s. The mark draws itself in 0.6s after 0.18s. The line slides in from 18px left. The matching slot in the bar pulses from 1.7 scale. This is the one authored moment on the page.
- **Tally row:** the same mark at 2.25rem, the question at 600, the verdict at 800 in its colour, ruled with Rule 2.

### Figures
- **Frame:** a title (700, 1rem), the figure, then a Muted caption. No border, no background.
- **Panels:** one figure holds several panels in the same grid cell. A step shows its panel with a 0.45s fade. Bars in a new panel grow from zero in 0.8s, 0.05s apart.
- **Bar row:** label left (600), value right (700), a 0.875rem Track bar below with a 2px radius. The fill is Ink 2. The model's fill is blue. A quiet fill is Rule 2. A bad fill is red. A limit is a 2px Ink line with a small label above it.
- **Options:** a lettered list with thin bars (0.625rem). The option the model leans to has a blue letter, bold text and a blue bar. A harmful option has red text, and a red bar and letter if it is on top.
- **Dots:** 0.8rem circles, empty with a Rule 2 border, filled Ink 2, or red. One dot is one run.
- **Charts:** SVG. Axes in Rule 2, grid in Rule, lines 2.5 wide with round joins. The model's line is blue. Others are Ink 2 or dashed. Lines are labelled directly at 700 in their own colour.
- **Interval:** a dot with a white ring on a pale band of the same colour, against a 2px Ink line labelled "no difference".
- **Replay:** an SVG map on a Rule ground with white floor. The agent is a blue dot with a faint blue trail. Enemies are red, the key amber, the exit green. A ruler under it shows the position, and a ruled card states what was chosen.
- **Facts:** a big number over one Muted line, in an auto-fit grid.

### Controls
- **Choice pill:** a pill with a 1.5px blue border, blue-ink text (600, 0.9375rem) and 0.5rem by 0.9rem padding. Hover fills it Blue Soft. Selected fills it blue with white text. It is a real radio input.
- **Slider:** a 0.375rem pill track, blue up to the value and Rule 2 after it, with a 1.5rem blue thumb ringed in white. The label is 700, the read-out is blue-ink, and a Muted hint sits below. The thumb grows to 1.12 on hover.
- **Focus:** a 3px blue outline, 3px away, on everything focusable.
- **Links:** blue-ink, underlined 1px, 2px on hover.

### "Where this comes from" note
A closed disclosure after each stamp. The toggle is blue-ink at 600 with a small chevron that turns when open. Inside, a list at 0.9375rem in Ink 2. Each item starts with a bold evidence label (Observed, Documented, Inferred, Implementation choice). File paths appear here and nowhere else, on a Track tint.

### Motion
- **One ease.** Exponential ease-out everywhere: `cubic-bezier(0.16, 1, 0.3, 1)` in CSS and `expo.out` in GSAP.
- **Durations.** 0.2s for hovers. 0.45s for panel and step fades. 0.6s to 0.8s for entrances. 1.2s for a grid that counts up.
- **Steps.** A step becomes active when its top crosses 62% of the viewport height (78% on narrow screens). Text outside the active step rests at 0.62 opacity.
- **Scrub.** Long steps tie a figure to scroll position. The build has four: three across the two replays and one on the race chart.
- **Stateless.** A figure's picture depends only on the step in view. Scrolling forward, scrolling back and reloading land on the same picture.
- **Reduced motion.** With `prefers-reduced-motion`, every change is set at once to its final state and smooth scrolling is off.

## Do's and Don'ts

### Do:
- **Do** keep blue for the model itself or for something the reader can touch.
- **Do** pair every stamp colour with its mark and its word.
- **Do** label figures directly, and state the unit in the caption.
- **Do** build figures from flat SVG, canvas or CSS on the white page.
- **Do** separate with hairlines (1px Rule or Rule 2). Open a stamp with the 2px Ink rule.
- **Do** use Figtree with tabular numerals for every number.
- **Do** use the one ease and show final states under reduced motion.
- **Do** keep the figure in place while the text moves, and keep it a function of the step in view.
- **Do** write in plain words with the glossary's terms, and keep file paths inside "Where this comes from" notes.
- **Do** check every new layout at 375px for sideways scroll.

### Don't:
- **Don't** use blue for decoration, for a non-model series or for emphasis in prose.
- **Don't** use a stamp colour as a general accent.
- **Don't** put shadows, gradients or filled cards on or around a figure.
- **Don't** add a second typeface or a monospace face.
- **Don't** put a small label above a heading.
- **Don't** add a second authored animation that competes with the stamp landing.
- **Don't** carry a fact in motion alone or in colour alone.
- **Don't** show model scores as odds or percentages of certainty.
- **Don't** use phase numbers, setup ids or file paths in the prose.
- **Don't** restyle the deck, the fraud console or the game viewer to match this system.
