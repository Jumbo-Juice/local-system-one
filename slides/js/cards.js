// 14 prediction cards: built from the registered predictions before the steps are wired, so each card's
// data-from makes it flip on its own step. Print shows both faces (deck.css, html.printing).
(function () {
  const { el, mark } = window.Deck
  const box = document.querySelector('[data-cards]')
  if (!box) return
  const rows = window.DATA.manual.predictions.rows
  rows.forEach((r, i) => {
    const front = el('div', { class: 'face front', html: `<span class="k">Predicted</span><p>${r.said}</p>` })
    const back = el('div', { class: `face back ${r.verdict}`, html: `<p class="out">${r.got}</p><span class="tagl">${mark(r.verdict)}${r.tag}</span>` })
    box.append(el('div', { class: 'flip', 'data-from': i + 1 }, el('div', { class: 'inner' }, front, back)))
  })
  const count = v => rows.filter(r => r.verdict === v).length
  box.append(el('div', { class: 'tally', 'data-from': rows.length, html: `<p><b>${count('no')}</b> wrong</p><p><b>${count('partly')}</b> partly</p><p><b>${count('yes')}</b> right</p>` }))
})()
