// Walks the live deck at 1920 x 1080: node slides/check_deck.mjs [folder for screenshots]
// Forward through every step, then backward, then re-entry, and fails if a state doesn't match itself. Also
// presses keys faster than the animations, which must still land on the same final state. No network, 0 console errors.
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { launch, serve, sleep } from './cdp.mjs'

const ROOT = path.dirname(fileURLToPath(import.meta.url))
const SHOTS = process.argv[2]
const SETTLE = 1000 // longer than any token's duration

// What a slide currently shows: which steps are in, every bar, every number, and whether the race and replay are at rest
const signature = `(() => {
  const sec = Reveal.getCurrentSlide()
  const bits = [...sec.querySelectorAll('[data-from]')].map(n => n.classList.contains('shown') ? 1 : 0).join('')
  const bars = [...sec.querySelectorAll('.hbar-fill, .wbar')].map(n => getComputedStyle(n).transform).join('|')
  const on = [...sec.querySelectorAll('.on, .dots, .ivals, .judged, .locked')].length
  const vals = [...sec.querySelectorAll('.val, .v, .per, [data-corr], [data-p90], [data-answer], [data-seen], [data-declined]')].map(n => n.textContent).join('|')
  const seg = [...sec.querySelectorAll('.seg, .needle, .replay-rows')].map(n => getComputedStyle(n).transform).join('|')
  const lines = [...sec.querySelectorAll('polyline')].map(n => n.getAttribute('points')).join('|')
  return JSON.stringify({ bits, bars, on, vals, seg, lines })
})()`
const where = `(() => { const i = Reveal.getIndices(); return i.h + '.' + i.v + '.' + (i.f ?? -1) })()`

const server = await serve(ROOT)
const edge = await launch()
const settle = async () => {
  await sleep(SETTLE)
  for (let i = 0; i < 200 && await edge.evaluate('__deck.busy()'); i++) await sleep(100)
  await sleep(100)
}
const failures = []
const states = new Map()

try {
  await edge.goto(server.url)
  await sleep(600)

  // 1. forward through everything on the main line, recording each settled state
  const order = []
  for (let n = 0; n < 300; n++) {
    await settle()
    const at = await edge.evaluate(where)
    if (order.includes(at)) break
    order.push(at)
    states.set(at, await edge.evaluate(signature))
    if (SHOTS) { fs.mkdirSync(SHOTS, { recursive: true }); await edge.shot(path.join(SHOTS, `f${String(n).padStart(3, '0')}_${at}.png`)) }
    const before = at
    await edge.evaluate('Reveal.right()')
    if (await edge.evaluate(where) === before) break
  }
  console.log(`forward: ${order.length} states`)

  // 2. backward: every state must match what forward left there. Steps on a slide reached from its end are final states.
  for (let k = order.length - 2; k >= 0; k--) {
    await edge.evaluate('Reveal.left()')
    await settle()
    const at = await edge.evaluate(where)
    if (at !== order[k]) { failures.push(`back: expected ${order[k]}, at ${at}`); await edge.evaluate(`Reveal.slide(${order[k].split('.').map(x => Number(x)).join(',')})`); await settle() }
    const got = await edge.evaluate(signature)
    if (got !== states.get(order[k])) failures.push(`back differs at ${order[k]}:\n  fwd ${states.get(order[k])}\n  bak ${got}`)
  }
  console.log('backward: done')

  // 3. re-entry from each slide's neighbour, at its first and last states
  const slides = [...new Set(order.map(s => s.split('.').slice(0, 2).join('.')))]
  for (const s of slides) {
    const [h, v] = s.split('.').map(Number)
    await edge.evaluate(`Reveal.slide(${h}, ${v})`)
    await settle()
    await edge.evaluate(`Reveal.slide(${h + 1}, 0)`)
    await settle()
    await edge.evaluate('Reveal.slide(0, 0)')
    await settle()
    await edge.evaluate(`Reveal.slide(${h}, ${v}, -1)`)
    await settle()
    const first = order.find(o => o.startsWith(`${s}.`))
    const got = await edge.evaluate(signature)
    if (got !== states.get(first)) failures.push(`re-entry differs at ${first}:\n  fwd ${states.get(first)}\n  got ${got}`)
  }
  console.log(`re-entry: ${slides.length} slides`)

  // 4. presenter faster than the motion: rapid steps land on the same state
  for (const s of slides) {
    const [h, v] = s.split('.').map(Number)
    const last = order.filter(o => o.startsWith(`${s}.`)).at(-1)
    await edge.evaluate(`Reveal.slide(${h}, ${v}, -1)`)
    const steps = order.filter(o => o.startsWith(`${s}.`)).length - 1
    for (let i = 0; i < steps; i++) { await edge.evaluate('Reveal.nextFragment()'); await sleep(40) }
    await settle()
    const got = await edge.evaluate(signature)
    if (got !== states.get(last)) failures.push(`rapid differs at ${last}:\n  fwd ${states.get(last)}\n  got ${got}`)
  }
  console.log('rapid: done')

  if (edge.errors.length) failures.push(`console errors: ${edge.errors.join(' | ')}`)
} finally {
  edge.close()
  server.close()
}
console.log(failures.length ? `FAIL\n${failures.join('\n')}` : 'all states match, 0 console errors')
process.exit(failures.length ? 1 : 0)
