// One scene per dynamic slide. mount(sec) builds it once; set(step, animate, sec) is absolute, so forward,
// backward and re-entry all land on the same picture; leave() stops anything running; print(sec) adds what a
// static page needs in place of a control. Every number comes from window.DATA (build_data.py).
(function () {
  const { scenes, el, count, mark, rise, formats: f, REDUCED, motion, letGoOfKeys } = window.Deck
  const D = () => window.DATA
  const setW = (node, w) => node.style.setProperty('--w', w)

  function bar(cls = '') {
    const fill = el('span', { class: `hbar-fill ${cls}` })
    return { track: el('span', { class: 'hbar-track track' }, fill), fill }
  }

  // ------------------------------------------------------------ 1 title
  scenes.title = {
    enter(sec) { rise(sec.querySelectorAll('.title-h .line'), { stagger: 160, y: 40 }); rise(sec.querySelectorAll('.title-opts .o'), { stagger: 120, y: 12 }) },
    set() {},
  }

  // ------------------------------------------------------------ 3 one look: a recorded PaySim decision
  scenes.look = {
    mount(sec) {
      const d = D().paysim.decision
      const box = sec.querySelector('.opts')
      this.rows = [['A', 'approve'], ['B', 'review'], ['C', 'decline']].map(([letter, name]) => {
        const b = bar(name === d.action ? 'ours' : '')
        const val = el('span', { class: 'val' }, '0.00')
        const bub = el('span', { class: 'bub' }, letter)
        const line = el('div', { class: 'opt' }, bub, el('span', { class: 'name' }, name), b.track, val)
        box.append(line)
        return { line, bub, val, fill: b.fill, score: d.scores[name], hit: name === d.action }
      })
    },
    set(step, animate) {
      for (const r of this.rows) {
        setW(r.fill, step >= 1 ? r.score : 0)
        count(r.val, step >= 1 ? r.score : 0, f.d2, animate)
        r.line.classList.toggle('on', step >= 2 && r.hit)
        r.bub.classList.toggle('on', step >= 2 && r.hit)
      }
    },
  }

  // ------------------------------------------------------------ 5 replay of a recorded PaySim window
  scenes.replay = {
    mount(sec) {
      const p = D().paysim
      this.rows = sec.querySelector('.replay-rows')
      for (const r of p.replay) {
        this.rows.append(el('div', { class: `rrow ${r.by === 'model' ? 'm' : ''}` },
          el('span', { class: 'i' }, `#${r.i}`), el('span', {}, r.type), el('span', { class: 'amt' }, f.usd2(r.amount)), el('span', { class: 'act' }, r.action)))
      }
      this.n = p.replay.length
      this.seenEl = sec.querySelector('[data-seen]')
      this.declinedEl = sec.querySelector('[data-declined]')
      const button = sec.querySelector('[data-toggle]')
      this.button = button
      button.addEventListener('click', () => (this.timer ? this.stop() : this.play()))
      letGoOfKeys(button)
    },
    show(k) {
      this.at = k
      const visible = 7
      this.rows.style.transform = `translateY(${-Math.max(0, k - visible) * 66}px)`
      const shown = D().paysim.replay.slice(0, k).filter(r => r.by === 'model')
      this.seenEl.textContent = f.int(shown.length)
      this.declinedEl.textContent = f.int(shown.filter(r => r.action === 'decline').length)
    },
    play() {
      if (REDUCED) return this.show(this.n)
      this.button.textContent = 'Pause'
      this.timer = setInterval(() => {
        if (this.at >= this.n) return this.stop()
        this.show(this.at + 1)
      }, 170)
    },
    stop() { clearInterval(this.timer); this.timer = null; this.button.textContent = 'Play' },
    leave() { this.stop() },
    set(step, animate) {
      this.stop()
      const fromStart = this.last === 0
      this.last = step
      if (step === 0) this.show(0)
      else if (step === 1 && animate && fromStart) { this.show(0); this.play() } else this.show(this.n) // stepping back never replays
    },
    // reveal turns off overflow clipping in print, so keep only the rows that fit, with no scroll offset
    print() {
      this.show(this.n)
      for (const row of [...this.rows.children].slice(0, -7)) row.remove()
      this.rows.style.transform = 'none'
    },
  }

  // ------------------------------------------------------------ 6 the split and FINAL's locked windows
  scenes.split = {
    mount(sec) {
      const s = D().manual.split
      const shares = { fit: s.fit / s.rows, dev: s.dev / s.rows, test: s.test / s.rows }
      this.parts = { fit: sec.querySelector('.seg.fit'), dev: sec.querySelector('.seg.devseg'), test: sec.querySelector('.seg.test') }
      this.at = { fit: 0, dev: shares.fit, test: shares.fit + shares.dev, shares }
      for (const [k, node] of Object.entries({ fit: sec.querySelector('[data-l=fit]'), dev: sec.querySelector('[data-l=dev]'), test: sec.querySelector('[data-l=test]') })) node.style.left = `${this.at[k] * 100}%`
      const testStart = s.fit + s.dev
      this.zoom = sec.querySelector('.zoom')
      const barEl = sec.querySelector('.zoom-bar')
      Object.values(D().final.window_rows).forEach(([start, end], i) => {
        const x = ((start + end) / 2 - testStart) / s.test
        const t = el('span', { class: 'tick', style: `left:${x * 100}%; transition-delay:${i * 22}ms` })
        barEl.append(t)
      })
    },
    set(step) {
      const { shares } = this.at
      // the three segments slide in from the left of the bar: fit stays, dev and test take their places
      this.parts.fit.style.transform = step >= 1 ? `scaleX(${shares.fit})` : 'scaleX(1)'
      this.parts.dev.style.transform = step >= 1 ? `translateX(${this.at.dev * 1704}px) scaleX(${shares.dev})` : `translateX(1704px) scaleX(${shares.dev})`
      this.parts.test.style.transform = step >= 1 ? `translateX(${this.at.test * 1704}px) scaleX(${shares.test})` : `translateX(1704px) scaleX(${shares.test})`
      this.zoom.classList.toggle('on', step >= 2)
      this.zoom.classList.toggle('locked', step >= 3)
    },
  }

  // ------------------------------------------------------------ 7 rule explorer: arithmetic in the page, no model
  const LOG_MIN = Math.log(1), LOG_MAX = Math.log(5000)
  const amountFrom = t => Math.round(Math.exp(LOG_MIN + (LOG_MAX - LOG_MIN) * t))
  const amountTo = a => (Math.log(a) - LOG_MIN) / (LOG_MAX - LOG_MIN)
  function expected(amount, p, line) {
    const c = D().costs
    const friction = Math.max(c.friction_rate * amount, c.friction_min)
    const fees = c.review_fee + D().cost.lambda
    return { approve: p * amount, decline: (1 - p) * (friction + (line ? 10 : 0)), review: fees }
  }
  function cheapest(e) { // as fraud/rule.py: review only when strictly below both, else the cheaper of the two
    const fallback = e.approve <= e.decline ? 'approve' : 'decline'
    return e.review < Math.min(e.approve, e.decline) ? 'review' : fallback
  }
  const PRESETS = [
    { amount: 30, p: .02, line: false },
    { amount: 1500, p: .02, line: false },
    { amount: 30, p: .2, line: false },
    { amount: 30, p: .2, line: true },
  ]
  scenes.rule = {
    mount(sec) {
      this.amount = sec.querySelector('[data-amount]')
      this.p = sec.querySelector('[data-p]')
      this.lineBtn = sec.querySelector('[data-line]')
      this.amountVal = sec.querySelector('[data-amount-val]')
      this.pVal = sec.querySelector('[data-p-val]')
      this.answer = sec.querySelector('[data-answer]')
      const box = sec.querySelector('.ebars')
      const formula = { approve: 'p × amount', decline: '(1 − p) × friction', review: `${D().costs.review_fee} + λ ${D().cost.lambda}` }
      this.bars = ['approve', 'review', 'decline'].map((name, i) => {
        const b = bar()
        const val = el('span', { class: 'val' }, '0')
        const bub = el('span', { class: 'bub' }, 'ABC'[i])
        const line = el('div', { class: 'ebar' }, bub, el('span', { class: 'name' }, name), b.track, val, el('span', {}), el('span', {}), el('span', { class: 'how' }, formula[name]))
        box.append(line)
        return { name, line, bub, val, fill: b.fill, how: line.querySelector('.how') }
      })
      this.state = { ...PRESETS[0] }
      this.amount.addEventListener('input', () => { this.state.amount = amountFrom(+this.amount.value); this.draw(false) })
      this.p.addEventListener('input', () => { this.state.p = +this.p.value; this.draw(false) })
      this.lineBtn.addEventListener('click', () => { this.state.line = !this.state.line; this.draw(true) })
      for (const c of [this.amount, this.p, this.lineBtn]) letGoOfKeys(c)
    },
    draw(animate) {
      const { amount, p, line } = this.state
      const e = expected(amount, p, line)
      const top = Math.max(e.approve, e.decline, e.review, 1)
      const best = cheapest(e)
      this.amount.value = amountTo(amount)
      this.p.value = p
      this.amountVal.textContent = `${f.int(amount)} USD`
      this.pVal.textContent = p.toFixed(2)
      this.lineBtn.classList.toggle('on', line)
      this.lineBtn.textContent = line ? 'Customer line on: +10 USD on a decline' : 'Customer line off'
      for (const b of this.bars) {
        setW(b.fill, e[b.name] / top)
        b.fill.classList.toggle('ours', b.name === best)
        b.line.classList.toggle('on', b.name === best)
        b.bub.classList.toggle('on', b.name === best)
        count(b.val, e[b.name], f.d2, animate, motion.base)
        if (b.name === 'decline') b.how.textContent = line ? '(1 − p) × (friction + 10)' : '(1 − p) × friction' // as fraud/rule.py prices it
      }
      this.answer.textContent = `Cheapest: ${best}`
    },
    set(step, animate) {
      this.state = { ...PRESETS[Math.min(step, PRESETS.length - 1)] }
      this.draw(animate)
    },
    print(sec) {
      const body = sec.querySelector('[data-worked]')
      for (const [i, s] of PRESETS.entries()) {
        const e = expected(s.amount, s.p, s.line)
        const best = cheapest(e)
        const cell = name => `<td class="r"${name === best ? ' style="color:var(--c-ours);font-weight:700"' : ''}>${e[name].toFixed(2)}</td>`
        body.insertAdjacentHTML('beforeend', `<tr><td>${f.int(s.amount)} USD, p ${s.p.toFixed(2)}${s.line ? ', customer line' : ''}</td>${cell('approve')}${cell('review')}${cell('decline')}<td>${best}</td></tr>`)
      }
    },
  }

  // ------------------------------------------------------------ 8 the ladder, dev only
  const LADDER_MAX = 4000
  scenes.ladder = {
    mount(sec) {
      const s = D().dev.setups
      const box = sec.querySelector('.ladder-rows')
      this.rows = [
        ['frozen', 'No fine-tune', 'frozen', 0], ['phase12', 'Fine-tuned', 'one pass', 1],
        ['seed1', 'Same recipe, seed 1', 'how much one adapter moves', 2], ['rich', 'More columns', 'no tree score', 3],
      ].map(([key, name, small, from]) => {
        const b = bar(key === 'rich' ? 'ours' : '')
        const val = el('span')
        const line = el('div', { class: 'lrow' }, el('span', { class: 'name', html: `${name}<small>${small}</small>` }), b.track, el('span', { class: 'val' }, val))
        box.append(line)
        return { key, line, fill: b.fill, val, from, mean: s[key].mean }
      })
      const plot = sec.querySelector('.ladder')
      this.refs = [['logreg', 'logreg'], ['trees', 'trees']].map(([key, label]) => {
        const line = el('div', { class: `refline ${key}`, style: `left:calc(430px + (100% - 650px) * ${s[key].mean / LADDER_MAX}); bottom:0` }, el('span', {}, `${label} ${f.int(s[key].mean)}`))
        plot.append(line)
        return line
      })
    },
    set(step, animate) {
      for (const r of this.rows) {
        const on = step >= r.from
        setW(r.fill, on ? r.mean / LADDER_MAX : 0)
        r.line.style.opacity = on ? 1 : 0
        r.line.classList.toggle('dimmed', step > r.from && r.key !== 'rich')
        count(r.val, on ? r.mean : 0, f.int, animate)
      }
      for (const ref of this.refs) ref.classList.toggle('on', step >= 4)
    },
  }

  // ------------------------------------------------------------ 9 the shortcut
  scenes.shortcut = {
    mount(sec) {
      const s = D().dev.setups
      const box = sec.querySelector('.ladder-rows')
      this.rows = [['rich', 'More columns', 'no tree score', 0], ['signal', 'Reads a tree score', 'looks cheapest', 0], ['signal_alone', 'The tree score alone', 'no language model', 1]].map(([key, name, small, from]) => {
        const b = bar(key === 'rich' ? 'ours' : '')
        const val = el('span')
        const line = el('div', { class: 'lrow' }, el('span', { class: 'name', html: `${name}<small>${small}</small>` }), b.track, el('span', { class: 'val' }, val))
        box.append(line)
        return { key, line, fill: b.fill, val, from, mean: s[key].mean }
      })
      this.corr = sec.querySelector('[data-corr]')
    },
    set(step, animate) {
      for (const r of this.rows) {
        const on = step >= r.from
        setW(r.fill, on ? r.mean / LADDER_MAX : 0)
        r.line.style.opacity = on ? 1 : 0
        count(r.val, on ? r.mean : 0, f.int, animate)
        if (r.key === 'signal') r.fill.classList.toggle('outline', step >= 2)
      }
      count(this.corr, step >= 1 ? D().dev.echo.rank_corr : 0, f.auc, animate)
    },
  }

  // ------------------------------------------------------------ 10 window race: cumulative cost over FINAL's 20 windows
  const RACE = [
    { key: 'approve_all', name: 'Approve everything', color: 'var(--c-fraud)' },
    { key: 'rich_amt', name: 'd1, no trees, rule', color: 'var(--c-ours)' },
    { key: 'rich_cust', name: 'the same, customer line', color: 'var(--c-miss)', dash: '1 15' },
    { key: 'baseline', name: 'Baseline: logreg, rule', color: 'var(--c-ink)' },
  ]
  scenes.race = {
    mount(sec) {
      const setups = D().final.setups
      const W = 1100, H = 480, L = 110, B = 44, T = 10
      this.series = RACE.map(r => {
        let sum = 0
        return { ...r, cum: [0, ...setups[r.key].windows.map(v => (sum += v))], mean: setups[r.key].mean }
      })
      const top = 110000
      this.sx = k => L + (W - L - 20) * (k / 20)
      this.sy = v => T + (H - T - B) * (1 - v / top)
      const svg = sec.querySelector('svg')
      svg.setAttribute('viewBox', `0 0 ${W} ${H}`)
      let grid = ''
      for (const v of [0, 25000, 50000, 75000, 100000]) grid += `<line x1="${L}" x2="${W - 20}" y1="${this.sy(v)}" y2="${this.sy(v)}"/><text x="${L - 12}" y="${this.sy(v) + 10}" text-anchor="end">${v / 1000}k</text>`
      for (const k of [1, 5, 10, 15, 20]) grid += `<text x="${this.sx(k)}" y="${H - 6}" text-anchor="middle">${k}</text>`
      grid += `<text x="${L - 12}" y="${H - 6}" text-anchor="end">window</text>` // the axes say what they count: windows along, cumulative USD up (the heading beside the plot)
      svg.innerHTML = `<g class="grid">${grid}</g>` + this.series.map(s => `<polyline class="line" stroke="${s.color}" ${s.dash ? `stroke-dasharray="${s.dash}"` : ''}/>`).join('')
      const lines = svg.querySelectorAll('polyline')
      const box = sec.querySelector('.totals')
      this.series.forEach((s, i) => {
        s.line = lines[i]
        s.val = el('span', { class: 'v' }, '0')
        s.per = el('span', { class: 'per' }, '')
        box.append(el('div', { class: 'total' }, el('span', { class: 'who3', html: `<i class="sw" style="background:${s.color}"></i>${s.name}` }), s.val, s.per))
      })
      this.scrub = sec.querySelector('input')
      this.at = sec.querySelector('.at')
      this.playBtn = sec.querySelector('[data-play]')
      this.scrub.addEventListener('input', () => { this.pause(); this.draw(+this.scrub.value) })
      this.playBtn.addEventListener('click', () => (this.raf ? this.pause() : this.play()))
      for (const c of [this.scrub, this.playBtn]) letGoOfKeys(c)
    },
    draw(k) {
      this.k = k
      const whole = Math.floor(k)
      for (const s of this.series) {
        const pts = []
        for (let i = 0; i <= whole; i++) pts.push(`${this.sx(i)},${this.sy(s.cum[i])}`)
        const part = k - whole
        const now = whole < 20 ? s.cum[whole] + (s.cum[whole + 1] - s.cum[whole]) * part : s.cum[20]
        if (part > 0) pts.push(`${this.sx(k)},${this.sy(now)}`)
        s.line.setAttribute('points', pts.join(' '))
        s.val.textContent = f.int(now)
        s.per.textContent = k > 0 ? `${f.int(now / k)} per window` : ' '
      }
      this.scrub.value = k
      this.at.textContent = `window ${Math.ceil(k)} of 20`
    },
    play() {
      if (REDUCED) return this.draw(20)
      if (this.k >= 20) this.draw(0)
      this.playBtn.textContent = 'Pause'
      const speed = 20 / 8000 // the whole race takes 8 s
      let last = performance.now()
      const tick = now => {
        const k = Math.min(20, this.k + Math.max(0, now - last) * speed) // a frame's timestamp can predate `last`
        last = now
        this.draw(k)
        this.raf = k < 20 ? requestAnimationFrame(tick) : (this.pause(), null)
      }
      this.raf = requestAnimationFrame(tick)
    },
    pause() { cancelAnimationFrame(this.raf); this.raf = null; this.playBtn.textContent = 'Play' },
    leave() { this.pause() },
    set(step, animate) {
      this.pause()
      const fromStart = this.last === 0
      this.last = step
      if (step === 0) this.draw(0)
      else if (step === 1 && animate && fromStart) { this.draw(0); this.play() } else this.draw(20) // stepping back never replays
    },
    print() { this.draw(20) },
  }

  // ------------------------------------------------------------ 11 CI forest
  scenes.forest = {
    mount(sec) {
      const d = D().final.diffs
      const panel = (box, rows, lo, hi, ticks) => {
        const x = v => (v - lo) / (hi - lo) * 100
        const built = rows.map(([key, name]) => {
          const r = d[key]
          const excl = r.distinguishable
          const ci = el('span', { class: 'fci', style: `left:${x(r.ci[0])}%; width:${x(r.ci[1]) - x(r.ci[0])}%; transform-origin:${(r.mean - r.ci[0]) / (r.ci[1] - r.ci[0]) * 100}% 50%` })
          const dot = el('span', { class: 'fdot', style: `left:${x(r.mean)}%` })
          const line = el('div', { class: `frow ${excl ? 'excl-yes' : ''}` },
            el('span', { class: 'fname', html: name }), el('div', { class: 'fplot' }, ci, dot),
            el('span', { class: 'fval', html: `<b>${f.signed(r.mean)}</b><span class="ftag">${mark(excl ? 'yes' : 'partly')}${excl ? 'excludes 0' : 'crosses 0'}</span>` }))
          box.append(line)
          return { line, excl }
        })
        const axisRow = el('div', { style: 'display:contents' }, el('span'), el('div', { class: 'faxis' }, ...ticks.map(t => el('span', { style: `left:${x(t)}%` }, f.signed(t)))), el('span'))
        box.append(axisRow)
        return { built }
      }
      this.top = panel(sec.querySelector('[data-forest=base]'), [
        ['rich_amt_signal_amt', 'd1, no trees <span class="dim">vs</span> d1 reads trees'],
        ['signal_amt_baseline', 'd1 reads trees <span class="dim">vs baseline</span>'],
        ['signal_cust_baseline', '…customer line <span class="dim">vs baseline</span>'],
        ['rich_cust_rich_amt', 'd1, no trees: line <span class="dim">vs no line</span> *'],
        ['rich_cust_baseline', 'd1, no trees, line <span class="dim">vs baseline</span> *'],
        ['rich_amt_baseline', 'd1, no trees <span class="dim">vs baseline</span>'],
      ], -1000, 1000, [-1000, -500, 0, 500, 1000])
      this.bottom = panel(sec.querySelector('[data-forest=approve]'), [
        ['signal_amt_approve', 'd1 reads trees <span class="dim">vs approve all</span>'],
        ['rich_cust_approve', 'd1, no trees, line <span class="dim">vs approve all</span> *'],
        ['rich_amt_approve', 'd1, no trees <span class="dim">vs approve all</span>'],
      ], -4500, 500, [-4000, -2000, 0])
      this.sweep = sec.querySelector('[data-sweep]')
      this.anims = []
    },
    set(step, animate) {
      for (const a of this.anims) a.cancel()
      this.anims = []
      this.sweep.classList.toggle('on', step >= 3)
      const all = [...this.top.built, ...this.bottom.built]
      for (const r of all) {
        r.line.classList.toggle('dots', step >= 1)
        r.line.classList.toggle('ivals', step >= 2)
        r.line.classList.toggle('excl', step >= 3 && r.excl)
        r.line.classList.toggle('judged', step >= 3)
      }
      if (step === 3 && animate && !REDUCED) {
        this.anims.push(this.sweep.animate([{ transform: 'translateX(-363px)' }, { transform: 'none' }], { duration: motion.slow * 1.5, easing: motion.move })) // half the plot: CSS places the line, since a hidden section has no layout to measure
      }
    },
  }

  // ------------------------------------------------------------ 11 backup: cost against speed (the console's Cost and speed panel)
  // Static: no steps. Cheaper at the top, faster to the left (log scale). The shaded corner is final.speed's ideal rule.
  scenes.costspeed = {
    mount(sec) {
      const sp = D().final.speed
      const W = 1700, H = 560, L = 150, R = 520, T = 10, B = 84, pw = W - L - R, ph = H - T - B
      const ms = sp.points.map(p => p.ms), costs = sp.points.map(p => p.cost)
      const lo = Math.floor(Math.log10(Math.min(...ms))), hi = Math.ceil(Math.log10(Math.max(sp.bar, ...ms)))
      const x = v => L + pw * (Math.log10(v) - lo) / (hi - lo)
      const a = Math.floor(Math.min(...costs) / 200) * 200 - 100, b = Math.ceil(Math.max(...costs) / 200) * 200 + 100
      const y = v => T + ph * (v - a) / (b - a)
      const edge = sp.edge ?? b
      let g = `<rect class="ideal" x="${L}" y="${T}" width="${x(sp.bar) - L}" height="${y(edge) - T}"/>`
      g += `<text class="ideal-tag" x="${L + 16}" y="${T + 36}">ideal: fast, and not distinguishably dearer than the cheapest</text>`
      for (let v = Math.ceil(a / 200) * 200; v <= b; v += 200) g += `<line class="gl" x1="${L}" x2="${L + pw}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 14}" y="${y(v) + 10}" text-anchor="end">${f.int(v)}</text>`
      for (let p = lo; p <= hi; p++) g += `<line class="gl" x1="${x(10 ** p)}" x2="${x(10 ** p)}" y1="${T}" y2="${T + ph}"/><text x="${x(10 ** p)}" y="${H - 44}" text-anchor="middle">${10 ** p >= 1 ? 10 ** p : (10 ** p).toFixed(-p)} ms</text>`
      g += `<text x="${L + pw / 2}" y="${H - 4}" text-anchor="middle">p90 time to decide one payment, log scale: faster to the left</text>`
      g += `<text x="28" y="${T + ph / 2}" text-anchor="middle" transform="rotate(-90 28 ${T + ph / 2})">USD per window, cheaper at the top</text>`
      g += `<line class="bar" x1="${x(sp.bar)}" x2="${x(sp.bar)}" y1="${T}" y2="${T + ph}"/><text x="${x(sp.bar) + 10}" y="${T + ph - 14}">${sp.bar} ms bar</text>`
      // Labels: beside the dot if it has room before the bar, else in a column right of the plot on a leader line.
      const col = L + pw + 30
      const placed = [...sp.points].sort((p, q) => p.cost - q.cost).map(p => {
        const cx = x(p.ms)
        return { p, cx, cy: y(p.cost), lx: cx + 30 + 420 > x(sp.bar) ? col : cx + 30, ly: y(p.cost) }
      })
      placed.forEach((o, i) => { for (const q of placed.slice(0, i)) if (q.lx === o.lx && o.ly - q.ly < 70) o.ly = q.ly + 70 })
      for (const { p, cx, cy, lx, ly } of placed) {
        if (lx === col) g += `<polyline class="lead" points="${cx + 14},${cy} ${col - 40},${ly} ${col - 10},${ly}"/>`
        g += `<circle class="dot${p.ideal ? ' in' : ''}" cx="${cx}" cy="${cy}" r="${p.cpu ? 15 : 12}"/>`
        g += `<text class="lab${p.cpu ? ' strong' : ''}" x="${lx}" y="${ly + 4}">${p.name}<tspan class="sub" x="${lx}" dy="32">${f.int(p.cost)} USD · ${p.ms < 1 ? p.ms.toFixed(3) : p.ms.toFixed(1)} ms${p.cpu ? ', laptop CPU' : ''}</tspan></text>`
      }
      const svg = sec.querySelector('.costspeed svg')
      svg.setAttribute('viewBox', `0 0 ${W} ${H}`)
      svg.innerHTML = g
    },
    set() {},
  }

  // ------------------------------------------------------------ 12 customer line toggle
  const TOGGLE = ['baseline', 'rich_amt', 'rich_cust']
  const METRICS = [
    { key: 'good_declined', name: 'Good customers declined', max: 1200, cls: 'ours' },
    { key: 'frauds_stopped', name: 'Frauds stopped, of 630', max: 630, cls: '' },
    { key: 'fraud_let_through', name: 'Fraud USD let through', max: 40000, cls: 'fraud' },
    { key: 'mean', name: 'Mean cost per window, USD', max: 3000, cls: '' },
  ]
  scenes.toggle = {
    mount(sec) {
      const s = D().final.setups
      const box = sec.querySelector('.mrows')
      this.rows = METRICS.map(m => {
        const b = bar(m.cls)
        const val = el('span', { class: 'val' }, '0')
        const line = el('div', { class: 'mrow' }, el('span', { class: 'name' }, m.name), b.track, val)
        box.append(line)
        const left = s.baseline[m.key] / m.max
        b.track.append(el('span', { class: 'base-tick', style: `left:${left * 100}%`, title: 'baseline' }))
        return { ...m, fill: b.fill, val }
      })
      this.buttons = [...sec.querySelectorAll('[data-pick]')]
      this.readout = sec.querySelector('.seg-readout')
      for (const b of this.buttons) {
        b.addEventListener('click', () => this.pick(b.dataset.pick, true))
        letGoOfKeys(b)
      }
    },
    pick(key, animate) {
      const s = D().final.setups[key]
      for (const r of this.rows) {
        setW(r.fill, s[r.key] / r.max)
        count(r.val, s[r.key], f.int, animate)
      }
      for (const b of this.buttons) b.classList.toggle('on', b.dataset.pick === key)
      this.readout.classList.toggle('on', key === 'rich_cust')
    },
    set(step, animate) { this.pick(TOGGLE[step], animate) },
  }

  // ------------------------------------------------------------ 13 weights: schematic heights, labelled so
  scenes.weights = {
    mount(sec) {
      this.small = sec.querySelector('[data-w=small]')
      this.large = sec.querySelector('[data-w=large]')
      this.smallVal = sec.querySelector('[data-wv=small]')
      this.largeVal = sec.querySelector('[data-wv=large]')
    },
    set(step) {
      const split = step >= 1
      this.small.style.setProperty('--h', split ? .012 : .137)
      this.large.style.setProperty('--h', split ? 1 : .137)
      this.smallVal.textContent = split ? 'well under 1' : 'counts 1'
      this.largeVal.textContent = split ? `up to ${D().manual.training.largest_weight}` : 'counts 1'
    },
  }

  // ------------------------------------------------------------ 15 waterfall
  scenes.waterfall = {
    mount(sec) {
      const w = D().cost.waterfall
      const AXIS_FROM = 2000, AXIS_TO = 2300, PLOT = 440, BASE = 80
      const y = v => (v - AXIS_FROM) / (AXIS_TO - AXIS_FROM) * PLOT
      const steps = [
        { name: 'lm-rich', from: AXIS_FROM, to: w.start, cls: 'start', text: f.int(w.start) },
        { name: 'fraud let through', from: w.start, to: w.start + w.fraud, cls: 'up', text: f.signed(w.fraud) },
        { name: 'wrong declines', from: w.start + w.fraud, to: w.start + w.fraud + w.declines, cls: 'down', text: f.signed(w.declines) },
        { name: 'review fees', from: w.start + w.fraud + w.declines, to: w.end, cls: 'down', text: f.signed(w.reviews) },
        { name: 'lm-weighted', from: AXIS_FROM, to: w.end, cls: 'start', text: f.int(w.end) },
      ]
      const box = sec.querySelector('.wf')
      this.cols = steps.map(s => {
        const lo = Math.min(s.from, s.to), hi = Math.max(s.from, s.to)
        const barEl = el('span', { class: `wf-bar ${s.cls}`, style: `bottom:${BASE + y(lo)}px; height:${y(hi) - y(lo)}px; transform-origin:${s.to >= s.from ? 'bottom' : 'top'} center` })
        const val = el('span', { class: 'wf-val', style: `bottom:${BASE + y(hi) + 6}px` }, s.text)
        const col = el('div', { class: 'wf-col' }, barEl, val, el('span', { class: 'wf-name' }, s.name))
        box.append(col)
        return col
      })
      box.append(el('span', { class: 'wf-base' }))
      const bands = sec.querySelector('.bands')
      this.bands = D().manual.bands.rows.map(r => {
        const b = bar(r.label.startsWith('500') ? 'ours' : '')
        bands.append(el('div', { class: 'brow' }, el('span', {}, r.label), b.track, el('span', { class: 'val' }, f.pct(r.rich / r.frauds))))
        return { fill: b.fill, share: r.rich / r.frauds }
      })
    },
    set(step) {
      this.cols.forEach((c, i) => c.classList.toggle('on', step >= i))
      for (const b of this.bands) setW(b.fill, step >= 5 ? b.share : 0)
    },
  }

  // ------------------------------------------------------------ 16 latency needle
  scenes.speed = {
    mount(sec) { this.needle = sec.querySelector('.needle'); this.val = sec.querySelector('[data-p90]') },
    set(step, animate) {
      const p90 = D().final.latency.p90
      const angle = -90 + (step >= 1 ? p90 / 300 * 180 : 0)
      this.needle.style.setProperty('--a', `${angle}deg`)
      count(this.val, step >= 1 ? p90 : 0, f.int, animate)
    },
  }

})()
