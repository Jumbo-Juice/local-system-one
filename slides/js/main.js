// Wiring: bind the numbers, build the scenes, turn data-steps into reveal fragments, start reveal. A section's
// state follows from how many of its step fragments are visible, so reveal's own bookkeeping keeps it right.
(function () {
  const { PRINT, REDUCED, motion, scenes, bindAll, mark, el } = window.Deck
  const ACT = { A: { label: 'A · The setup', transition: 'fade', cls: '' }, B: { label: 'B · The journey', transition: 'slide', cls: '' }, C: { label: 'C · The verdict', transition: 'fade', cls: 'ink' } }
  const slidesRoot = document.querySelector('.slides')

  bindAll(slidesRoot)
  for (const m of slidesRoot.querySelectorAll('[data-mark]')) m.outerHTML = mark(m.dataset.mark)

  // In print a before-picture is dropped: each slide is one page, in its final state.
  if (PRINT) for (const sec of slidesRoot.querySelectorAll('[data-print-skip]')) sec.remove()

  const leaves = [...slidesRoot.querySelectorAll('section')].filter(s => !s.querySelector('section'))
  const stops = new Set(leaves.map(s => s.dataset.n)).size
  const MARKS = 21 // how many timing marks fit down the sheet
  for (const sec of leaves) {
    const page = sec.querySelector('.page')
    const done = Math.ceil(Number(sec.dataset.n) / stops * MARKS)
    if (page) page.append(el('div', { class: 'marks', style: `--done:${done}` }))
    const act = ACT[sec.dataset.act]
    if (act && !sec.dataset.transition) sec.dataset.transition = act.transition
    if (sec.dataset.act === 'B') sec.dataset.backgroundColor = '#f3f8f6'
    const n = Number(sec.dataset.steps ?? 0)
    for (let i = 0; i < n; i++) sec.append(el('i', { class: 'fragment step-mark', 'data-fragment-index': i }))
  }

  for (const sec of leaves) scenes[sec.dataset.scene]?.mount?.(sec)

  const stepOf = sec => sec.querySelectorAll(':scope > .step-mark.visible').length

  function apply(sec, step, animate) {
    const go = animate && !REDUCED
    sec.classList.toggle('still', !go)
    for (const node of sec.querySelectorAll('[data-from]')) node.classList.toggle('shown', step >= Number(node.dataset.from))
    scenes[sec.dataset.scene]?.set(step, go, sec)
    if (!go) { void sec.offsetWidth; sec.classList.remove('still') }
  }

  // Entering a slide never replays a half-finished state: forward starts at step 0, backward at the last step.
  function enter(sec, previous) {
    if (previous) scenes[previous.dataset.scene]?.leave?.(previous)
    if (!sec) return
    const step = stepOf(sec)
    apply(sec, step, false)
    if (step === 0 && previous) scenes[sec.dataset.scene]?.enter?.(sec)
    actWipe(sec, previous)
  }

  const wipe = el('div', { class: 'act-wipe' })
  document.body.append(wipe)
  let wipeAnim = null
  function actWipe(sec, previous) {
    if (!previous || REDUCED) return
    const to = sec.dataset.act, from = previous.dataset.act
    if (!to || to === from || Number(sec.dataset.n) < Number(previous.dataset.n)) return
    wipeAnim?.cancel()
    wipe.textContent = ACT[to].label
    wipe.className = `act-wipe ${ACT[to].cls}`
    wipeAnim = wipe.animate(
      [{ opacity: 1, transform: 'translateX(-100%)' }, { opacity: 1, transform: 'none', offset: .35 }, { opacity: 1, transform: 'none', offset: .65 }, { opacity: 1, transform: 'translateX(100%)' }],
      { duration: motion.slow * 1.5, easing: motion.move },
    )
  }

  if (PRINT) {
    for (const sec of leaves) {
      const n = Number(sec.dataset.steps ?? 0)
      for (const node of sec.querySelectorAll('[data-from]')) node.classList.add('shown')
      scenes[sec.dataset.scene]?.set(n, false, sec)
      scenes[sec.dataset.scene]?.print?.(sec)
    }
  }

  const stepSection = e => e.fragment?.closest('section')
  Reveal.on('ready', e => { if (!PRINT) enter(e.currentSlide, null) })
  Reveal.on('slidechanged', e => enter(e.currentSlide, e.previousSlide))
  Reveal.on('fragmentshown', e => { const s = stepSection(e); if (s) apply(s, stepOf(s), true) })
  Reveal.on('fragmenthidden', e => { const s = stepSection(e); if (s) apply(s, stepOf(s), true) })

  const right = () => Reveal.right()
  const left = () => Reveal.left()
  Reveal.initialize({
    width: 1920,
    height: 1080,
    margin: 0,
    minScale: 0.05,
    maxScale: 4,
    center: false,
    hash: true,
    controls: false,
    progress: false,
    slideNumber: false,
    transition: REDUCED ? 'none' : 'fade',
    transitionSpeed: 'fast',
    backgroundTransition: REDUCED ? 'none' : 'fade',
    autoAnimate: !REDUCED,
    autoAnimateDuration: motion.slow / 1000,
    autoAnimateEasing: motion.out,
    // Space, N and a clicker's page keys walk the talk and skip the backups; only the down arrow opens one.
    keyboard: { 32: right, 78: right, 34: right, 80: left, 33: left },
    pdfSeparateFragments: false, // one page per slide, every step shown
    pdfMaxPagesPerSlide: 1,
    plugins: [window.RevealNotes],
  })
  window.__deck = { leaves, stepOf, busy: () => !!(scenes.replay.timer || scenes.race.raf) }
})()
