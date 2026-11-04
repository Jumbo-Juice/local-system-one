// The deck's small runtime: formats, {{bindings}}, motion, and the rule that keeps going backwards safe.
//
// A slide's state is a pure function of its step, and a step is a reveal fragment, so the presenter sets
// the pace and nothing depends on what played before. A scene's set(step, animate) always lands on the
// same final state, whether it arrives there with motion or at once.
(function () {
  const PRINT = /print-pdf/.test(location.search)
  const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches || /reduced=1/.test(location.search)
  if (PRINT) document.documentElement.classList.add('printing')

  const style = getComputedStyle(document.documentElement)
  const ms = name => parseFloat(style.getPropertyValue(name))
  const motion = {
    fast: ms('--d-fast'), base: ms('--d-base'), slow: ms('--d-slow'),
    out: style.getPropertyValue('--e-out').trim(), move: style.getPropertyValue('--e-move').trim(),
  }

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
    num: x => String(x),
    abs: x => group(Math.round(x)),
    pct: x => `${Math.round(x * 100)}%`,
    d2: x => x.toFixed(2),
    usd2: x => group(x, 2),
  }

  const get = path => path.split('.').reduce((node, part) => node[Array.isArray(node) ? Number(part) : part], window.DATA)

  // {{final.setups.baseline.mean|int}} in the HTML, speaker notes included
  function bindAll(root) {
    root.innerHTML = root.innerHTML.replace(/\{\{\s*([\w.]+)(?:\|(\w+))?\s*\}\}/g, (_, path, fmt) => {
      const value = get(path)
      return fmt ? formats[fmt](value) : String(value)
    })
  }

  // ✓ / ✗ / ~ as one-stroke paths, so they never fall back to a missing glyph
  const MARKS = { yes: 'M3.5 10.5 L8 15 L16.5 4.5', no: 'M4.5 4.5 L15.5 15.5 M15.5 4.5 L4.5 15.5', partly: 'M3 11.5 C5.5 6.5 8 6.5 10 10 C12 13.5 14.5 13.5 17 8.5' }
  const mark = kind => `<svg class="mark" viewBox="0 0 20 20" role="img" aria-label="${kind}"><path d="${MARKS[kind]}"/></svg>`

  // A number that counts to its value. Same value, same end state; a newer count replaces an older one.
  function count(el, to, fmt = int, animate = true, duration = motion.slow) {
    const from = el._value ?? 0
    el._value = to
    const token = (el._count = {})
    const finish = () => { if (el._count === token) el.textContent = fmt(to) }
    if (!animate || REDUCED || PRINT || from === to) return finish()
    const start = performance.now()
    const tick = now => {
      if (el._count !== token) return
      const t = Math.min(1, (now - start) / duration)
      const eased = 1 - Math.pow(1 - t, 3)
      el.textContent = fmt(from + (to - from) * eased)
      if (t < 1) requestAnimationFrame(tick)
    }
    requestAnimationFrame(tick)
    setTimeout(finish, duration + 120) // a hidden page gets no frames: still land on the value
  }

  // Short entrances: they play forward only, never hold a final state, and a jump cancels them.
  function rise(els, { stagger = 90, y = 24 } = {}) {
    if (REDUCED || PRINT) return []
    return [...els].map((el, i) => el.animate(
      [{ opacity: 0, transform: `translateY(${y}px)` }, { opacity: 1, transform: 'none' }],
      { duration: motion.base, delay: i * stagger, easing: motion.out, fill: 'backwards' },
    ))
  }

  function el(tag, attrs = {}, ...kids) {
    const node = document.createElement(tag)
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') node.className = v
      else if (k === 'style') node.setAttribute('style', v)
      else if (k === 'html') node.innerHTML = v
      else node.setAttribute(k, v)
    }
    for (const kid of kids) node.append(kid)
    return node
  }

  // A control should never keep the keyboard: after a drag or click, arrows and space drive the deck again.
  function letGoOfKeys(control) {
    for (const ev of ['pointerup', 'change', 'click']) control.addEventListener(ev, () => control.blur())
  }

  window.Deck = { PRINT, REDUCED, motion, formats, get, bindAll, mark, count, rise, el, letGoOfKeys, scenes: {} }
})()
