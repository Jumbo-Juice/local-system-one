// Headless Edge over its debugging protocol, and a small static server. No package: Node's own WebSocket.
import { spawn } from 'node:child_process'
import fs from 'node:fs'
import http from 'node:http'
import os from 'node:os'
import path from 'node:path'

const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.ttf': 'font/ttf', '.txt': 'text/plain', '.png': 'image/png' }

export function serve(root) {
  const server = http.createServer((req, res) => {
    const name = path.join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname.replace(/\/$/, '/index.html')))
    if (!name.startsWith(root) || !fs.existsSync(name)) { res.writeHead(404); return res.end() }
    res.writeHead(200, { 'content-type': TYPES[path.extname(name)] ?? 'application/octet-stream' })
    fs.createReadStream(name).pipe(res)
  })
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve({ url: `http://127.0.0.1:${server.address().port}/`, close: () => server.close() })))
}

const EDGE = process.env.EDGE ?? 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'
export const sleep = ms => new Promise(r => setTimeout(r, ms))

export async function launch(width = 1920, height = 1080) {
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'deck-edge-'))
  const proc = spawn(EDGE, ['--headless=new', '--disable-gpu', '--hide-scrollbars', '--no-first-run', '--remote-debugging-port=0', `--user-data-dir=${profile}`, `--window-size=${width},${height}`, 'about:blank'], { stdio: 'ignore' })
  const portFile = path.join(profile, 'DevToolsActivePort')
  for (let i = 0; i < 100 && !fs.existsSync(portFile); i++) await sleep(100)
  const port = fs.readFileSync(portFile, 'utf8').split('\n')[0]
  const target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find(t => t.type === 'page')
  const ws = new WebSocket(target.webSocketDebuggerUrl)
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject })
  let id = 0
  const pending = new Map()
  const handlers = []
  ws.onmessage = ({ data }) => {
    const msg = JSON.parse(data)
    if (msg.id && pending.has(msg.id)) {
      const { resolve, reject } = pending.get(msg.id)
      pending.delete(msg.id)
      msg.error ? reject(new Error(msg.error.message)) : resolve(msg.result)
    } else for (const h of handlers) h(msg)
  }
  const send = (method, params = {}) => new Promise((resolve, reject) => { pending.set(++id, { resolve, reject }); ws.send(JSON.stringify({ id, method, params })) })
  const evaluate = async expression => {
    const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true })
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text)
    return r.result.value
  }
  const errors = []
  handlers.push(msg => {
    if (msg.method === 'Runtime.exceptionThrown') errors.push(msg.params.exceptionDetails.exception?.description ?? msg.params.exceptionDetails.text)
    if (msg.method === 'Runtime.consoleAPICalled' && msg.params.type === 'error') errors.push(msg.params.args.map(a => a.value ?? a.description).join(' '))
    if (msg.method === 'Log.entryAdded' && msg.params.entry.level === 'error') errors.push(`${msg.params.entry.text} ${msg.params.entry.url ?? ''}`)
  })
  await send('Page.enable'); await send('Runtime.enable'); await send('Log.enable')
  await send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false })
  const goto = async url => {
    await send('Page.navigate', { url })
    for (let i = 0; i < 200; i++) {
      await sleep(100)
      if (await evaluate(`document.readyState === 'complete' && !!window.Reveal && Reveal.isReady() && document.fonts.status === 'loaded'`).catch(() => false)) return
    }
    throw new Error(`deck never got ready: ${url} ${errors.join(' | ')}`)
  }
  const shot = async (file, clip) => {
    const { data } = await send('Page.captureScreenshot', { format: 'png', ...(clip ? { clip: { ...clip, scale: 1 } } : {}), captureBeyondViewport: !!clip })
    fs.writeFileSync(file, Buffer.from(data, 'base64'))
  }
  const close = () => { ws.close(); proc.kill(); try { fs.rmSync(profile, { recursive: true, force: true }) } catch {} }
  return { send, evaluate, goto, shot, close, errors }
}
