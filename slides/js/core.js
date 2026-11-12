// The deck's small runtime: formats, {{bindings}}, marks and steps. The deck is static: a slide's picture is a
// pure function of its step, and printing shows every slide at its last step.
(function () {
  const PRINT = /print-pdf/.test(location.search)
  const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches || /reduced=1/.test(location.search)
  if (PRINT) document.documentElement.classList.add('printing')
  if (REDUCED) document.documentElement.classList.add('reduced')

  const MINUS = '−'
  const group = (x, digits = 0) => Math.abs(x).toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })
  const int = x => (Math.round(x) < 0 ? MINUS : '') + group(Math.round(x))
  const signed = x => (Math.round(x) > 0 ? '+' : Math.round(x) < 0 ? MINUS : '') + group(Math.round(x))
  const formats = {
    int, signed,
    ci: a => `${int(a[0])} to ${int(a[1])}`,
    range: a => `${int(a[0])}–${int(a[1])}`,
    auc: x => x.toFixed(3).replace(/^0/, ''),
    auc4: x => x.toFixed(4).replace(/^0/, ''),
    aucrange: a => `${a[0].toFixed(3).replace(/^0/, '')}–${a[1].toFixed(3).replace(/^0/, '')}`,
    num: x => String(x),
    abs: x => group(Math.round(x)),
    pct: x => `${Math.round(x * 100)}%`,
    d2: x => (x.toFixed(2).startsWith('-') ? MINUS : '') + Math.abs(x).toFixed(2),
    usd2: x => group(x, 2),
    dist: d => (d.distinguishable ? 'distinguishable' : 'not distinguishable'),  // a verdict word from the data, never typed
  }

  const get = path => path.split('.').reduce((node, part) => node[Array.isArray(node) ? Number(part) : part], window.DATA)

  // {{final.setups.baseline.mean|int}} in the HTML, speaker notes included
  function bindAll(root) {
    root.innerHTML = root.innerHTML.replace(/\{\{\s*([\w.]+)(?:\|(\w+))?\s*\}\}/g, (_, path, fmt) => {
      const value = get(path)
      return fmt ? formats[fmt](value) : String(value)
    })
  }

  // ✓ / ✗ / ~ as one-stroke paths, so they never fall back to a missing glyph: <span data-mark="yes">
  const MARKS = { yes: 'M3.5 10.5 L8 15 L16.5 4.5', no: 'M4.5 4.5 L15.5 15.5 M15.5 4.5 L4.5 15.5', partly: 'M3 11.5 C5.5 6.5 8 6.5 10 10 C12 13.5 14.5 13.5 17 8.5' }
  const mark = kind => `<svg class="mark" viewBox="0 0 20 20" role="img" aria-label="${kind}"><path d="${MARKS[kind]}"/></svg>`
  function marks(root) {
    for (const m of root.querySelectorAll('[data-mark]')) m.outerHTML = mark(m.dataset.mark)
  }

  // Steps: data-steps="n" on a section gives it n clicks; data-from="k" shows an element from click k on.
  // A step's picture depends only on the step number, so forward, backward and re-entry agree.
  function showStep(sec, step) {
    for (const node of sec.querySelectorAll('[data-from]')) node.classList.toggle('shown', step >= Number(node.dataset.from))
  }
  // Printing: one page per slide, every step shown. main.js calls this after the numbers are bound.
  function printAll() {
    if (PRINT) for (const node of document.querySelectorAll('[data-from]')) node.classList.add('shown')
  }

  window.Deck = { PRINT, REDUCED, formats, get, bindAll, mark, marks, showStep, printAll }
})()
