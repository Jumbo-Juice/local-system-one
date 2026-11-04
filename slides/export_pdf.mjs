// The deck as a PDF, with no new package: node slides/export_pdf.mjs
// One 1920 x 1080 page per slide, each in its final state, fonts embedded, text kept as text. Also writes
// exports/png/p-NN.png per page and fails if anything overflows its page or is set below the minimum sizes.
// A pass is not the end: open the PNGs and look at them.
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { launch, serve, sleep } from './cdp.mjs'

const ROOT = path.dirname(fileURLToPath(import.meta.url))
const PDF = path.join(ROOT, 'exports', 'FrogWatch(Group 13)_Final_128-331.pdf')
const MIN = { text: 30, h1: 64, refs: 26 }

function checkPages(min) {
  return [...document.querySelectorAll('.pdf-page')].map((p, i) => {
    const box = p.getBoundingClientRect()
    const out = []
    const foot = p.querySelector('.page > .foot')?.getBoundingClientRect()
    if (foot) for (const el of p.querySelectorAll('.main *')) {
      const r = el.getBoundingClientRect()
      if (r.width && r.height && r.bottom > foot.top + 1 && getComputedStyle(el).display !== 'none') { out.push(`runs into the footnote: ${el.className?.baseVal ?? el.className ?? el.localName}`); break }
    }
    for (const el of p.querySelectorAll('.page *')) {
      if (el.closest('.notes, .speaker-notes')) continue
      const style = getComputedStyle(el)
      if (style.display === 'none') continue
      const r = el.getBoundingClientRect()
      const name = el.className?.baseVal ?? el.className ?? el.localName
      if (r.width && (r.left < box.left - 1 || r.right > box.right + 1 || r.bottom > box.bottom + 1)) out.push(`overflows the page: ${name}`)
      if (el.scrollWidth > el.clientWidth + 1 && style.overflowX === 'hidden') out.push(`clipped: ${name}`)
      const own = [...el.childNodes].some(n => n.nodeType === 3 && n.textContent.trim())
      if (!own || el.closest('svg')) continue
      const floor = el.closest('h1') ? min.h1 : el.closest('.refs') ? min.refs : min.text
      if (parseFloat(style.fontSize) < floor - 0.01) out.push(`${parseFloat(style.fontSize)}px text: "${el.textContent.trim().slice(0, 40)}"`)
    }
    return { page: i + 1, out: [...new Set(out)] }
  })
}

const server = await serve(ROOT)
const edge = await launch()
let failed = false
try {
  await edge.goto(`${server.url}?print-pdf`)
  await edge.evaluate(`new Promise(r => { const t = () => document.querySelector('.pdf-page') ? r() : setTimeout(t, 50); t() })`)
  await sleep(500)
  const problems = await edge.evaluate(`(${checkPages.toString()})(${JSON.stringify(MIN)})`)
  const { data } = await edge.send('Page.printToPDF', { printBackground: true, preferCSSPageSize: true })
  fs.writeFileSync(PDF, Buffer.from(data, 'base64'))
  // the PNGs come from the PDF itself, so what you look at is what ships
  const png = path.join(ROOT, 'exports', 'png')
  fs.mkdirSync(png, { recursive: true })
  for (const f of fs.readdirSync(png)) if (f.endsWith('.png')) fs.rmSync(path.join(png, f)) // only the files: Windows won't delete a folder someone has open
  const rendered = spawnSync('pdftoppm', ['-r', '96', '-png', PDF, path.join(png, 'p')])
  if (rendered.error) console.log('pdftoppm not found: no PNGs')
  for (const { page, out } of problems) {
    console.log(`${String(page).padStart(2, '0')}  ${out.length ? `FAIL ${out.join('; ')}` : 'ok'}`)
    if (out.length) failed = true
  }
  if (edge.errors.length) { failed = true; console.log('console errors:', edge.errors) }
  console.log(`${problems.length} pages -> ${PDF}`)
} finally {
  edge.close()
  server.close()
}
process.exit(failed ? 1 : 0)
