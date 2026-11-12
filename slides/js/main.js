// Wiring: bind the numbers, draw the marks and charts, turn data-steps into reveal fragments, start reveal.
// A section's step is how many of its step fragments are visible, so reveal's own bookkeeping keeps it right.
(function () {
  const { PRINT, REDUCED, bindAll, marks, showStep, printAll } = window.Deck
  const slidesRoot = document.querySelector('.slides')

  bindAll(slidesRoot)
  marks(slidesRoot)
  window.Charts.drawAll(slidesRoot)
  printAll()

  const leaves = [...slidesRoot.querySelectorAll('section')].filter(s => !s.querySelector('section'))
  for (const sec of leaves) {
    const n = Number(sec.dataset.steps ?? 0)
    for (let i = 0; i < n; i++) {
      const step = document.createElement('i')
      step.className = 'fragment step-mark'
      step.dataset.fragmentIndex = i
      sec.append(step)
    }
  }

  const stepOf = sec => sec.querySelectorAll(':scope > .step-mark.visible').length
  const show = sec => { if (sec && !PRINT) showStep(sec, stepOf(sec)) }
  const stepSection = e => e.fragment?.closest('section')

  Reveal.on('ready', e => show(e.currentSlide))
  Reveal.on('slidechanged', e => show(e.currentSlide))
  Reveal.on('fragmentshown', e => show(stepSection(e)))
  Reveal.on('fragmenthidden', e => show(stepSection(e)))

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
    backgroundTransition: 'none',
    autoAnimate: false,
    // Space, N, the right arrow and a clicker's page keys walk the talk, one step at a time.
    keyboard: { 32: right, 78: right, 34: right, 80: left, 33: left },
    pdfSeparateFragments: false, // one page per slide, every step shown
    pdfMaxPagesPerSlide: 1,
    plugins: [window.RevealNotes],
  })
  window.__deck = { leaves, stepOf }
})()
