// Static SVG charts, drawn once after the numbers are bound. No motion: what loads is what prints.
// One SVG unit is one slide pixel (the SVG is drawn at its own width, never stretched), so a 30 px label is
// 30 px on the 1920 x 1080 slide. Colours come from deck.css through classes, i.e. from its CSS variables.
//
// In the HTML, a chart is a [data-chart] element whose <i> children are its rows, every number a binding:
//   <div class="chart" data-chart="bars" data-fmt="d2" data-max="1">
//     <i data-label="approve" data-value="{{worked.orders.listed.0}}"></i> …  (data-hl marks a row in green)
//   <div class="chart" data-chart="interval" data-fmt="signed" data-axis="USD per window">
//     <i data-label="…" data-mean="{{final.diffs.x.mean}}" data-lo="{{final.diffs.x.ci.0}}" data-hi="{{final.diffs.x.ci.1}}"></i>
// Optional on the chart: data-width (px, default 760), data-label-w (px, default 240), data-value-w (px, default 150).
// A row with data-from="k" is drawn in a <g data-from="k">, so it appears at step k on the same axis as the others.
(function () {
  const NS = 'http://www.w3.org/2000/svg'
  const ROW = 64

  function node(tag, attrs = {}, text) {
    const n = document.createElementNS(NS, tag)
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v)
    if (text !== undefined) n.textContent = text
    return n
  }

  function frame(el, opts, rows, extra = 0) {
    const width = Number(opts.width ?? 760)
    const labelW = Number(opts.labelW ?? 240)
    const valueW = Number(opts.valueW ?? 150)
    const height = rows * ROW + extra
    const svg = node('svg', { class: 'chart-svg', width, height, viewBox: `0 0 ${width} ${height}`, role: 'img' })
    if (opts.title) svg.setAttribute('aria-label', opts.title)
    el.replaceChildren(svg)
    const x0 = labelW + 16, x1 = width - valueW - 16
    return { svg, width, x0, x1 }
  }

  // one group per row; a row with a step carries it, so core.js shows it with the slide's other steps. The class keeps the
  // markup the same before and after a toggle (an element with no class attribute gets class="" once toggled off)
  const rowGroup = (svg, r) => svg.appendChild(node('g', r.from ? { class: 'step-row', 'data-from': r.from } : {}))

  const fmtOf = name => (name && window.Deck.formats[name]) || window.Deck.formats.d2

  // rows: {label, value, hl}. A bar per row from zero; negatives go left of a zero line.
  function bars(el, rows, opts = {}) {
    const fmt = fmtOf(opts.fmt)
    const { svg, width, x0, x1 } = frame(el, opts, rows.length, 8)
    const values = rows.map(r => r.value)
    const lo = Math.min(0, ...values, opts.min ?? 0)
    const hi = Math.max(0, ...values, opts.max ?? 0)
    const x = v => x0 + (v - lo) / ((hi - lo) || 1) * (x1 - x0)
    rows.forEach((r, i) => {
      const y = 4 + i * ROW + ROW / 2
      const a = Math.min(x(0), x(r.value)), b = Math.max(x(0), x(r.value))
      const g = rowGroup(svg, r)
      g.append(node('text', { class: 'lab', x: 0, y, 'dominant-baseline': 'central' }, r.label))
      g.append(node('rect', { class: `bar${r.hl ? ' hl' : ''}`, x: a, y: y - 18, width: Math.max(b - a, 2), height: 36 }))
      g.append(node('text', { class: `val${r.hl ? ' hl' : ''}`, x: width, y, 'text-anchor': 'end', 'dominant-baseline': 'central' }, fmt(r.value)))
    })
    svg.append(node('line', { class: 'zero', x1: x(0), x2: x(0), y1: 0, y2: rows.length * ROW + 8 }))
    return svg
  }

  // rows: {label, mean, ci: [lo, hi], hl}. A zero line, and per row one bar for the interval and one dot for the mean.
  // A row whose interval leaves out zero is drawn in green; the words on the slide say what that means.
  function interval(el, rows, opts = {}) {
    const fmt = fmtOf(opts.fmt ?? 'signed')
    const axis = 56
    const { svg, width, x0, x1 } = frame(el, opts, rows.length, axis)
    const ends = rows.flatMap(r => r.ci)
    const span = Math.max(0, ...ends) - Math.min(0, ...ends) || 1
    const lo = Math.min(0, ...ends) - span * 0.04, hi = Math.max(0, ...ends) + span * 0.04
    const x = v => x0 + (v - lo) / (hi - lo) * (x1 - x0)
    const bottom = rows.length * ROW
    rows.forEach((r, i) => {
      const y = i * ROW + ROW / 2
      const cls = r.ci[0] > 0 || r.ci[1] < 0 || r.hl ? ' hl' : ''
      svg.append(node('line', { class: 'row-rule', x1: 0, x2: width, y1: (i + 1) * ROW, y2: (i + 1) * ROW }))
      const g = rowGroup(svg, r)
      g.append(node('text', { class: 'lab', x: 0, y, 'dominant-baseline': 'central' }, r.label))
      g.append(node('line', { class: `ci${cls}`, x1: x(r.ci[0]), x2: x(r.ci[1]), y1: y, y2: y }))
      g.append(node('circle', { class: `dot${cls}`, cx: x(r.mean), cy: y, r: 11 }))
      g.append(node('text', { class: `val${cls}`, x: width, y, 'text-anchor': 'end', 'dominant-baseline': 'central' }, fmt(r.mean)))
    })
    svg.append(node('line', { class: 'zero', x1: x(0), x2: x(0), y1: 0, y2: bottom + 8 }))
    svg.append(node('text', { class: 'tick', x: x(0), y: bottom + 38, 'text-anchor': 'middle' }, '0'))
    if (opts.axis) svg.append(node('text', { class: 'tick', x: x1, y: bottom + 38, 'text-anchor': 'end' }, opts.axis))
    return svg
  }

  const num = s => (s === undefined ? undefined : Number(s))

  // Draw every [data-chart] under root from its <i> rows.
  function drawAll(root) {
    for (const el of root.querySelectorAll('[data-chart]')) {
      const d = el.dataset
      const opts = { fmt: d.fmt, width: num(d.width), labelW: num(d.labelW), valueW: num(d.valueW), max: num(d.max), min: num(d.min), axis: d.axis, title: d.title }
      const rows = [...el.querySelectorAll('i')].map(i => ({
        label: i.dataset.label, hl: 'hl' in i.dataset, from: i.dataset.from,
        value: num(i.dataset.value), mean: num(i.dataset.mean), ci: [num(i.dataset.lo), num(i.dataset.hi)],
      }))
      for (const r of rows) for (const v of [r.value, r.mean, ...r.ci]) if (v !== undefined && !Number.isFinite(v)) throw new Error(`chart row "${r.label}" has a number that is not a number`)
      const draw = { bars, interval }[d.chart]
      if (!draw) throw new Error(`unknown chart: ${d.chart}`)
      draw(el, rows, opts)
    }
  }

  window.Charts = { bars, interval, drawAll }
})()
