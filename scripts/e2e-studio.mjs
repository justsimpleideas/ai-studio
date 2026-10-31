// End-to-end test of the real Electron app, driven over the Chrome DevTools Protocol.
//
// The image engine runs for real (Python process, per-launch token, loopback API,
// studio-image: protocol, IPC) but with the test-only fake model from
// image-engine/tests/ui_harness.py, so no GPU or weights are needed. The app gets a
// throwaway profile folder, so real settings and galleries are never touched.
//
//   npm run build && npm run e2e [-- --shots <dir>]
//
// Needs a Python with Pillow on PATH (or STUDIO_PYTHON=/path/to/python). With torch
// and diffusers in that Python too, the Train tab trains for real on a tiny CPU model;
// without them it checks that training fails with a clear message.

import { spawn, spawnSync, execSync } from 'node:child_process'
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import net from 'node:net'
import { tmpdir } from 'node:os'
import { delimiter, dirname, join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const ROOT = resolve(import.meta.dirname, '..')
const DEBUG_PORT = 9341
const ENGINE_PORT = 5198
const shotsArg = process.argv.indexOf('--shots')
const SHOTS = shotsArg > 0 ? process.argv[shotsArg + 1] : null
const electronExe = join(ROOT, 'node_modules', 'electron', 'dist', process.platform === 'win32' ? 'electron.exe' : 'electron')

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
let passed = 0
const failures = []
const consoleErrors = []

function check(name, ok, detail = '') {
  if (ok) {
    passed++
    console.log(`  ok   ${name}`)
  } else {
    failures.push(name)
    console.log(`  FAIL ${name}${detail ? ` -- ${detail}` : ''}`)
  }
}

function portInUse(port) {
  return new Promise((done) => {
    const socket = net.connect(port, '127.0.0.1')
    socket.once('connect', () => (socket.destroy(), done(true)))
    socket.once('error', () => done(false))
  })
}

async function waitPortFree(port, ms) {
  const deadline = Date.now() + ms
  while (Date.now() < deadline) {
    if (!(await portInUse(port))) return true
    await sleep(200)
  }
  return false
}

// ------------------------------------------------------------ launch

if (await portInUse(ENGINE_PORT)) {
  console.error(`Port ${ENGINE_PORT} is busy; close any running studio first.`)
  process.exit(2)
}

const profile = mkdtempSync(join(tmpdir(), 'studio-e2e-'))
// The folder the test opens as a workspace, standing in for the native folder picker.
const workspaceDir = join(mkdtempSync(join(tmpdir(), 'studio-e2e-ws-')), 'demo-project')
mkdirSync(workspaceDir)
writeFileSync(join(workspaceDir, 'readme.txt'), 'A demo project.')
// Where "Save as…" writes, standing in for the native save dialog.
const exportPath = join(dirname(workspaceDir), 'exported.png')
// Where the gallery's multi-select saves copies, standing in for the folder picker.
const exportFolder = join(dirname(workspaceDir), 'saved-copies')
mkdirSync(exportFolder, { recursive: true })
// Where downloaded models would go, standing in for the folder picker (nothing is downloaded).
const modelsDir = join(dirname(workspaceDir), 'models')
// Synthetic training photos, standing in for the photo-folder picker of the Train tab.
const trainingPhotos = join(dirname(workspaceDir), 'photos')
const PYTHON = process.env.STUDIO_PYTHON ?? 'python'
execSync(`"${PYTHON}" "${join(ROOT, 'scripts', 'e2e-training-photos.py')}" "${trainingPhotos}"`, { stdio: 'inherit' })
// Two other galleries to merge in, standing in for the folder picker of the merge (one per pick).
const mergeGalleries = join(dirname(workspaceDir), 'other-galleries')
execSync(`"${PYTHON}" "${join(ROOT, 'scripts', 'e2e-merge-galleries.py')}" "${mergeGalleries}"`, { stdio: 'inherit' })
const TRAINS_FOR_REAL = spawnSync(PYTHON, ['-c', 'import torch, diffusers, safetensors'], { stdio: 'ignore' }).status === 0
// A local stand-in for Wikimedia Commons and Openverse, so collecting photos is tested without the internet.
const FAKE_SOURCES_PORT = 9362
const fakeSources = spawn(PYTHON, ['-m', 'tests.fake_photo_sources', String(FAKE_SOURCES_PORT)], { cwd: join(ROOT, 'image-engine'), stdio: 'ignore' })
const collectedPhotos = join(dirname(workspaceDir), 'collected')
// Where the gallery and the chats are moved to, standing in for the folder pickers
// of Settings → Where things are kept.
const movedGallery = join(dirname(workspaceDir), 'moved-gallery')
const movedChats = join(dirname(workspaceDir), 'moved-chats')
const env = { ...process.env }
delete env.ELECTRON_RUN_AS_NODE
Object.assign(env, {
  STUDIO_USER_DATA: profile,
  STUDIO_IMAGE_ENGINE_MODULE: 'tests.ui_harness',
  STUDIO_PYTHON: process.env.STUDIO_PYTHON ?? 'python',
  STUDIO_PICK_FOLDER: workspaceDir,
  STUDIO_SAVE_AS: exportPath,
  STUDIO_PICK_EXPORT_FOLDER: exportFolder,
  // The start-picture picker answers with the exported PNG, which exists by the time it is asked.
  STUDIO_PICK_START_IMAGE: exportPath,
  STUDIO_PICK_MODELS_FOLDER: modelsDir,
  STUDIO_PICK_TRAINING_FOLDER: trainingPhotos,
  STUDIO_PICK_COLLECT_FOLDER: collectedPhotos,
  STUDIO_PICK_GALLERY_FOLDER: movedGallery,
  STUDIO_PICK_CHATS_FOLDER: movedChats,
  STUDIO_PICK_MERGE_SOURCES: [join(mergeGalleries, 'pc1', 'outputs'), join(mergeGalleries, 'pc2')].join(delimiter),
  STUDIO_COLLECT_COMMONS_API: `http://127.0.0.1:${FAKE_SOURCES_PORT}/commons/api.php`,
  STUDIO_COLLECT_OPENVERSE_API: `http://127.0.0.1:${FAKE_SOURCES_PORT}/openverse/v1/images/`,
  HARNESS_STEP_DELAY: '0.15',
  // No tray balloons popping up on the desktop during the test.
  STUDIO_NO_BALLOON: '1'
})

const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])

/** Every PNG under a folder, newest first. */
function pngsUnder(dir) {
  const found = []
  for (const entry of readdirSync(dir, { withFileTypes: true, recursive: true })) {
    if (entry.isFile() && entry.name.endsWith('.png')) found.push(join(entry.parentPath, entry.name))
  }
  return found.sort().reverse()
}

/** Drag a gallery tile onto a folder row (or "All"), the way the browser delivers it. */
async function dragTile(itemId, targetSelector) {
  return js(`(() => {
    const tile = document.querySelector(${q(inImage(`.tile[data-id="${itemId}"]`))});
    const target = document.querySelector(${q(targetSelector)});
    if (!tile || !target) return false;
    const data = new DataTransfer();
    tile.dispatchEvent(new DragEvent("dragstart", { dataTransfer: data, bubbles: true, cancelable: true }));
    for (const type of ["dragover", "drop"]) {
      target.dispatchEvent(new DragEvent(type, { dataTransfer: data, bubbles: true, cancelable: true }));
    }
    return true;
  })()`)
}
const folderRow = (name) =>
  `[...document.querySelectorAll(${q(inImage('.folder-row'))})].find((r) => r.querySelector(".folder-name")?.textContent === ${q(name)})`
const folderCount = (name) => js(`${folderRow(name)}?.querySelector(".folder-count")?.textContent ?? ""`)
async function nameFolder(name) {
  await setValue(inImage('.folder-name-input input'), name)
  await js(`document.querySelector(${q(inImage('.folder-name-input'))}).requestSubmit()`)
}

/** Start the app a second time with the same profile; resolves true once that copy has exited. */
function launchSecond() {
  return new Promise((done) => {
    const second = spawn(electronExe, ['.'], { cwd: ROOT, env, stdio: 'ignore' })
    const timer = setTimeout(() => (second.kill(), done(false)), 15000)
    second.once('exit', () => (clearTimeout(timer), done(true)))
  })
}

/**
 * Close the window the way its close button does: Windows sends it WM_CLOSE. (A page
 * calling window.close() skips the window's close event, so it would test nothing.)
 */
function closeLikeTheCloseButton() {
  const script = `
    $h = (Get-Process -Id ${app.pid}).MainWindowHandle
    Add-Type -Name Win -Namespace E2E -MemberDefinition '[DllImport("user32.dll")] public static extern bool PostMessage(IntPtr h, uint m, IntPtr w, IntPtr l);'
    if ($h -eq 0) { exit 3 }
    [E2E.Win]::PostMessage($h, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero) | Out-Null`
  return spawnSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', script], { stdio: 'ignore' }).status === 0
}

/** Tell the window it lost focus, as choosing another program does: WM_ACTIVATE with WA_INACTIVE. */
function deactivateLikeAnotherWindow() {
  const script = `
    $h = (Get-Process -Id ${app.pid}).MainWindowHandle
    Add-Type -Name Win -Namespace E2E -MemberDefinition '[DllImport("user32.dll")] public static extern bool PostMessage(IntPtr h, uint m, IntPtr w, IntPtr l);'
    if ($h -eq 0) { exit 3 }
    [E2E.Win]::PostMessage($h, 0x0006, [IntPtr]::Zero, [IntPtr]::Zero) | Out-Null`
  return spawnSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', script], { stdio: 'ignore' }).status === 0
}

/** Minimize the window the way its title bar button does: WM_SYSCOMMAND with SC_MINIMIZE. */
function minimizeLikeTheButton() {
  const script = `
    $h = (Get-Process -Id ${app.pid}).MainWindowHandle
    Add-Type -Name Win -Namespace E2E -MemberDefinition '[DllImport("user32.dll")] public static extern bool PostMessage(IntPtr h, uint m, IntPtr w, IntPtr l);'
    if ($h -eq 0) { exit 3 }
    [E2E.Win]::PostMessage($h, 0x0112, [IntPtr]0xF020, [IntPtr]::Zero) | Out-Null`
  return spawnSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', script], { stdio: 'ignore' }).status === 0
}

// While someone uses the desktop, other windows can cover the test window. Windows then
// reports it occluded, Chromium stops painting it, and lazy-loaded pictures never start
// loading. The test window is treated as always visible instead.
const ALWAYS_VISIBLE = ['--disable-features=CalculateNativeWinOcclusion', '--disable-backgrounding-occluded-windows']

function launch() {
  return spawn(electronExe, ['.', `--remote-debugging-port=${DEBUG_PORT}`, ...ALWAYS_VISIBLE], { cwd: ROOT, env, stdio: 'ignore' })
}

async function connect() {
  for (let i = 0; i < 100; i++) {
    try {
      const targets = await (await fetch(`http://127.0.0.1:${DEBUG_PORT}/json/list`)).json()
      const page = targets.find((t) => t.type === 'page' && !t.url.startsWith('devtools'))
      if (page) return page.webSocketDebuggerUrl
    } catch {
      /* still starting */
    }
    await sleep(200)
  }
  throw new Error('Electron did not expose a debugging target')
}

let app = launch()
let ws
let nextId = 0
const pending = new Map()

async function attach() {
  ws = new WebSocket(await connect())
  await new Promise((res, rej) => {
    ws.onopen = res
    ws.onerror = rej
  })
  ws.onmessage = (msg) => {
    const data = JSON.parse(msg.data)
    if (data.id && pending.has(data.id)) {
      const { resolve: ok, reject } = pending.get(data.id)
      pending.delete(data.id)
      data.error ? reject(new Error(data.error.message)) : ok(data.result)
    } else if (data.method === 'Runtime.exceptionThrown') {
      consoleErrors.push(data.params.exceptionDetails.exception?.description ?? data.params.exceptionDetails.text)
    } else if (data.method === 'Runtime.consoleAPICalled' && data.params.type === 'error') {
      consoleErrors.push(data.params.args.map((a) => a.value ?? a.description).join(' '))
    }
  }
  await send('Runtime.enable')
  await send('Page.enable')
}

const send = (method, params = {}) =>
  new Promise((ok, reject) => {
    const id = ++nextId
    pending.set(id, { resolve: ok, reject })
    ws.send(JSON.stringify({ id, method, params }))
  })

async function js(expression) {
  const res = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true })
  if (res.exceptionDetails) throw new Error(`${expression}: ${res.exceptionDetails.exception?.description}`)
  return res.result.value
}

async function waitFor(expression, timeout = 10000) {
  const deadline = Date.now() + timeout
  while (Date.now() < deadline) {
    try {
      if (await js(expression)) return true
    } catch {
      /* navigating */
    }
    await sleep(80)
  }
  return false
}

const q = (sel) => JSON.stringify(sel)
const inImage = (sel) => `.studio-pane:not([hidden]) .image-studio ${sel}`
const click = (sel) => js(`(() => { const el = document.querySelector(${q(sel)}); if (!el) throw new Error("missing ${sel}"); el.click(); return true })()`)
const clickText = (sel, text) =>
  js(`(() => { const el = [...document.querySelectorAll(${q(sel)})].find(e => e.textContent.trim().startsWith(${q(text)})); if (!el) throw new Error("missing ${text}"); el.click(); return true })()`)
const setValue = (sel, value) =>
  js(`(() => {
    const el = document.querySelector(${q(sel)});
    const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, "value").set.call(el, ${q(String(value))});
    el.dispatchEvent(new Event("input", { bubbles: true }));
    return true;
  })()`)
const count = (sel) => js(`document.querySelectorAll(${q(sel)}).length`)
/** Drop a file on the image side the way the browser delivers one: dragenter, dragover, drop. */
async function dropFile(bytes, name) {
  await js(`(() => {
    const bytes = Uint8Array.from(atob(${q(bytes.toString('base64'))}), (c) => c.charCodeAt(0));
    const data = new DataTransfer();
    data.items.add(new File([bytes], ${q(name)}, { type: "image/png" }));
    const target = document.querySelector(${q(inImage('.generate-panel'))});
    for (const type of ["dragenter", "dragover", "drop"]) {
      target.dispatchEvent(new DragEvent(type, { dataTransfer: data, bubbles: true, cancelable: true }));
    }
    return true;
  })()`)
}

async function ctrlB() {
  for (const type of ['keyDown', 'keyUp']) {
    await send('Input.dispatchKeyEvent', { type, key: 'b', code: 'KeyB', windowsVirtualKeyCode: 66, modifiers: 2 })
  }
}
const text = (sel) => js(`document.querySelector(${q(sel)})?.textContent ?? ""`)

async function shot(name) {
  if (!SHOTS) return
  mkdirSync(SHOTS, { recursive: true })
  const { data } = await send('Page.captureScreenshot', { format: 'png' })
  writeFileSync(join(SHOTS, `${name}.png`), Buffer.from(data, 'base64'))
}

// ------------------------------------------------------------ scenario

try {
  await attach()
  await waitFor(`document.readyState === "complete" && !!document.querySelector(".rail")`)

  console.log('shell')
  check('rail shows Chat, Image and Settings', (await count('.rail-button')) === 3)
  check('chat side is shown first', await js(`!document.querySelector(".studio-pane:not([hidden]) .image-studio")`))
  check('chat side asks for a GGUF model', (await text('.studio-pane:not([hidden])')).includes('Load a GGUF model'))
  check('window title', (await js('document.title')) === 'Local AI Studio')
  check('engine not started before the image side is opened', !(await portInUse(ENGINE_PORT)))
  await shot('01-chat')

  console.log('workspace')
  check('asks for a folder before anything else', (await text('.studio-pane:not([hidden]) .transcript')).includes('Open a folder to start'))
  await clickText('.ws-head-actions button', 'Open folder')
  check('the opened folder becomes the workspace', await waitFor(`document.querySelector(".ws-current")?.textContent.includes("demo-project")`))
  check('a new chat is waiting in it', await waitFor(`!!document.querySelector(".chat-item.draft")`))
  check('nothing is written into the opened folder', readdirSync(workspaceDir).join() === 'readme.txt')
  const climb = await js(`window.api.workspace.switchTo("../../somewhere").then(() => "accepted", (e) => e.message)`)
  check('a workspace id that is a path is refused', /no longer exists/.test(climb), climb)
  const sendBare = await js(`window.api.sendMessage("hi").then(() => "accepted", (e) => e.message)`)
  check('without a model nothing is sent', /Load a model first/.test(sendBare), sendBare)
  check('and no chat is kept for it', (await count('.chat-list .chat-item:not(.draft)')) === 0)

  await clickText('.tabs button', 'Rules')
  const projectRules = '.panel > section:first-child'
  check('project rules belong to the open folder', await waitFor(`document.querySelector(${q(projectRules + ' h3')})?.textContent === "Project rules · demo-project"`))
  await clickText(`${projectRules} button`, 'Add a rule')
  check('a rule editor appears', await waitFor(`!!document.querySelector(${q(projectRules + ' .rule-editor textarea')})`))
  await setValue(`${projectRules} .rule-editor textarea`, 'Use two-space indents.')
  await clickText(`${projectRules} button.primary`, 'Save')
  check('the prompt names the working folder and carries the project rule', await waitFor(`(() => { const t = document.querySelector(".panel pre.diff")?.textContent ?? ""; return t.includes("Your working folder is") && t.includes("demo-project") && t.includes("Use two-space indents.") })()`))
  const saved = readdirSync(join(profile, 'userData', 'workspaces'))
  check('the workspace is kept in the app data folder', saved.length === 1 && readFileSync(join(profile, 'userData', 'workspaces', saved[0], 'workspace.json'), 'utf8').includes('two-space'), saved.join())

  await clickText('.tabs button', 'Permissions')
  check('the working folder is shared without asking', await waitFor(`document.querySelector(".panel")?.textContent.includes("working folder")`))
  check('secrets and version control are protected by default', await js(`(() => { const rows = [...document.querySelectorAll(".protect-row")].map((r) => r.textContent); return rows.some((t) => t.includes(".env")) && rows.some((t) => t.includes(".git")) })()`))
  check('the change history is there, empty for now', await js(`document.querySelector(".panel")?.textContent.includes("No changes yet.")`))
  await shot('01c-permissions')

  console.log('internet')
  check('a new folder is offline', await js(`document.querySelector(".level-offline input")?.checked === true`))
  // Turning internet on asks first; cancelling keeps the folder offline.
  await js(`window.__confirmed = []; window.confirm = (m) => { window.__confirmed.push(m); return false }; true`)
  await click('.level-search input')
  check('turning the internet on warns first', await js(`window.__confirmed.length === 1 && window.__confirmed[0].includes("Allow the agent to use the internet")`))
  check('cancelling the warning keeps it offline', await waitFor(`document.querySelector(".level-offline input")?.checked === true`))
  await js(`window.confirm = () => true; true`)
  await click('.level-search input')
  check('confirming switches to search only', await waitFor(`document.querySelector(".level-search input")?.checked === true`))
  check('the internet log starts empty', await js(`document.querySelector(".internet")?.textContent.includes("No requests yet.")`))
  await shot('01d-internet')
  await click('.level-offline input')
  check('and back to offline without a warning', await waitFor(`document.querySelector(".level-offline input")?.checked === true`))
  const outside = await js(`fetch("https://example.com/").then(() => "reached", () => "blocked")`)
  check('the window itself cannot reach the internet', outside === 'blocked', outside)
  await clickText('.tabs button', 'Chat')
  check('the sidebar counts the project rule', await waitFor(`document.querySelector(".ws-foot")?.textContent.includes("1 project rule")`))
  await shot('01b-workspace')

  console.log('layout')
  check('the top bar stays one line high', await js(`[...document.querySelectorAll(".topbar")].every((b) => b.getBoundingClientRect().height <= 56)`))
  check('nothing makes the window scroll sideways', await js(`document.documentElement.scrollWidth <= document.documentElement.clientWidth`))
  await click('.sidebar-toggle')
  check('the folder and chat list can be hidden', await waitFor(`!document.querySelector(".ws-pane")`))
  check('the folder and its internet level stay in view', await waitFor(`(() => { const t = document.querySelector(".chat-head")?.textContent ?? ""; return t.includes("demo-project") && t.includes("Offline") })()`))
  await ctrlB()
  check('Ctrl+B brings them back', await waitFor(`!!document.querySelector(".ws-pane") && !document.querySelector(".chat-head")`))

  console.log('image engine start')
  await clickText('.rail-button', 'Image')
  check('image side appears', await waitFor(`!!document.querySelector(${q(inImage('.generate-panel'))})`))
  check('engine comes up and reports the model state', await waitFor(`document.querySelector(${q(inImage('.status-chip'))})?.textContent.includes("Image model not loaded")`, 30000))
  check('engine listens on its port', await portInUse(ENGINE_PORT))

  console.log('security from the window')
  const direct = await js(`fetch("http://127.0.0.1:${ENGINE_PORT}/api/status").then(() => "reachable", (e) => "blocked")`)
  check('window cannot call the engine directly', direct === 'blocked', direct)
  check('no Node in the window', await js(`typeof require === "undefined" && typeof process === "undefined"`))
  check('bridge exposes named image operations only', await js(`typeof window.api.image.submit === "function" && !("invoke" in window.api) && !("invoke" in window.api.image)`))
  const bogus = await js(`window.api.image.cancel("../../api/status").then(() => "accepted", (e) => e.message)`)
  check('main refuses malformed ids', /Invalid id/.test(bogus), bogus)
  // A dropped file or a followed link would otherwise load in place of the studio, preload bridge included.
  const elsewhere = join(dirname(workspaceDir), 'elsewhere.html')
  writeFileSync(elsewhere, '<title>Elsewhere</title>')
  await js(`(location.href = ${q(pathToFileURL(elsewhere).href)}, true)`)
  await sleep(1500)
  check(
    'the window cannot be navigated away from the studio',
    (await js('document.title')) === 'Local AI Studio' && (await js(`!!document.querySelector(".rail")`))
  )

  console.log('generate')
  await setValue(inImage('textarea.prompt'), 'A lighthouse on a rocky coast at dusk')
  await setValue(inImage('input[type="range"]'), 2)
  await click(inImage('.generate-button'))
  check('Generate opens the queue window', await waitFor(`!!document.querySelector(${q(inImage('.queue-window'))})`))
  check('with the new job in it', await waitFor(`document.querySelectorAll(${q(inImage('.queue-row'))}).length === 1 && document.querySelector(${q(inImage('.queue-prompt'))}).textContent === "A lighthouse on a rocky coast at dusk"`))
  await clickText(inImage('.queue-window button'), 'Close')
  check('and it closes', await waitFor(`!document.querySelector(${q(inImage('.queue-window'))})`))
  check('progress card appears', await waitFor(`!!document.querySelector(${q(inImage('.progress-card'))})`))
  check('progress reports steps', await waitFor(`/Step \\d+ of 8/.test(document.querySelector(${q(inImage('.progress-card'))})?.textContent ?? "")`))
  await shot('02-progress')
  check('two images land in the filmstrip', await waitFor(`document.querySelectorAll(${q(inImage('.film'))}).length === 2`, 20000))
  check(
    'stage image is served through studio-image:',
    await waitFor(`(() => { const i = document.querySelector(${q(inImage('.stage-image img'))}); return i && i.src.startsWith("studio-image://gallery/") && i.complete && i.naturalWidth === 1024 })()`)
  )
  check(
    'thumbnails decode',
    await waitFor(`[...document.querySelectorAll(${q(inImage('.film img'))})].every(i => i.complete && i.naturalWidth > 0)`)
  )
  await shot('03-result')

  console.log('queue')
  // Long jobs (50 steps at the harness's 0.15 s a step), all cancelled before they finish an image.
  const submitLong = (prompt) =>
    js(`window.api.image.submit({ prompt: ${q(prompt)}, negative_prompt: "", width: 512, height: 512, steps: 50, guidance: 0, seed: 5, count: 1 })`)
  const qa = await submitLong('queue: first')
  const qb = await submitLong('queue: second')
  const qc = await submitLong('queue: third')
  const queueJob = (id) => `window.api.image.jobs().then((all) => all.find((j) => j.id === ${q(id)}))`
  const row = (id, sel = '') => inImage(`.queue-row[data-job="${id}"] ${sel}`)
  const rowPrompts = `[...document.querySelectorAll(${q(inImage('.queue-row .queue-prompt'))})].map((p) => p.textContent).join("|")`
  const rowPlaces = `[...document.querySelectorAll(${q(inImage('.queue-row .queue-place'))})].map((p) => p.textContent).join("|")`
  const card = `(document.querySelector(${q(inImage('.progress-card'))})?.textContent ?? "")`
  check('the progress card says Cancel is for the job it shows', await waitFor(`${card}.includes("Cancel this one") && ${card}.includes("queue: first")`), await js(card))
  check('and how many wait behind it', await waitFor(`${card}.includes("Queue · 2 waiting")`))
  await click(inImage('.progress-card .queue-button'))
  check('the queue window lists them in the order they run', await waitFor(`${rowPrompts} === "queue: first|queue: second|queue: third"`), await js(rowPrompts))
  check('marking the one being made and the places in line', await waitFor(`${rowPlaces} === "Making now|Next|2nd in line"`))
  await shot('03b-queue')

  await clickText(row(qb.id, 'button'), 'Edit')
  check('a queued job opens for editing', await waitFor(`!!document.querySelector(${q(row(qb.id, '.queue-editor'))})`))
  check('with its own settings', await js(`document.querySelector(${q(row(qb.id, '.queue-editor textarea.prompt'))}).value === "queue: second"`))
  await setValue(row(qb.id, '.queue-editor textarea.prompt'), 'queue: second, changed')
  await clickText(row(qb.id, '.queue-editor [aria-label="Aspect ratio"] button'), '16:9')
  await clickText(row(qb.id, '.queue-editor [aria-label="Resolution"] button'), 'Draft')
  await setValue(row(qb.id, '.queue-editor input[aria-label="Images"]'), 2)
  await shot('03c-queue-edit')
  await clickText(row(qb.id, '.queue-editor button'), 'Save changes')
  check(
    'saving changes the queued job itself: same job, new prompt, size and count',
    await waitFor(`${queueJob(qb.id)}.then((j) => j.status === "queued" && j.request.prompt === "queue: second, changed" && j.request.width > j.request.height && j.request.width * j.request.height < 600000 && j.request.count === 2 && j.request.seed === 5)`)
  )
  check('it keeps its place in line', await waitFor(`${rowPrompts} === "queue: first|queue: second, changed|queue: third" && !document.querySelector(${q(inImage('.queue-editor'))})`))

  await clickText(row(qc.id, 'button'), 'Remove from queue')
  check('removing one cancels it before it starts', await waitFor(`${queueJob(qc.id)}.then((j) => j.status === "cancelled" && j.started === null)`))
  check('and takes it off the list', await waitFor(`${rowPrompts} === "queue: first|queue: second, changed"`))

  await clickText(row(qa.id, 'button'), 'Edit')
  check('editing the one being made says it will start again', await waitFor(`!!document.querySelector(${q(row(qa.id, '.queue-restart-note'))})`))
  await setValue(row(qa.id, '.queue-editor textarea.prompt'), 'queue: first, again')
  await clickText(row(qa.id, '.queue-editor button'), 'Save and restart')
  check('saving stops it', await waitFor(`${queueJob(qa.id)}.then((j) => j.status === "cancelled")`))
  check(
    'and its changed copy runs next, ahead of what was queued',
    await waitFor(`window.api.image.jobs().then((all) => all.some((j) => j.request.prompt === "queue: first, again" && j.status === "running") && all.find((j) => j.id === ${q(qb.id)}).status === "queued")`)
  )
  check('the window shows the copy first', await waitFor(`${rowPrompts} === "queue: first, again|queue: second, changed" && ${rowPlaces} === "Making now|Next"`))

  await clickText(inImage('.progress-card button'), 'Cancel this one')
  check("the card's Cancel stops the job it shows, and the next one starts", await waitFor(`${queueJob(qb.id)}.then((j) => j.status === "running")`))
  check('which the card then shows', await waitFor(`${card}.includes("queue: second, changed")`))
  await clickText(row(qb.id, 'button'), 'Cancel this one')
  check('cancelling from the window works too, and empties it', await waitFor(`${queueJob(qb.id)}.then((j) => j.status === "cancelled") && document.querySelector(${q(inImage('.queue-empty'))}) !== null`))
  check(
    'none of them made an image',
    await js(`window.api.image.jobs().then((all) => all.filter((j) => j.request.prompt?.startsWith("queue:")).every((j) => j.images.length === 0))`)
  )
  await click(inImage('.queue-foot input[type=checkbox]'))
  check('opening after Generate can be switched off, and is remembered', await waitFor(`localStorage.getItem("lis.queue.openOnGenerate.v1") === "false"`))
  await js(`window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }))`)
  check('Escape closes the queue window', await waitFor(`!document.querySelector(${q(inImage('.queue-window'))})`))

  console.log('detail, gallery')
  await click(inImage('.stage-image'))
  check('detail opens', await waitFor(`!!document.querySelector(${q(inImage('.modal.detail'))})`))
  await clickText(inImage('.detail-actions button'), '☆ Favorite')
  check('favorite toggles', await waitFor(`!!document.querySelector(${q(inImage('.detail-actions button[aria-pressed="true"]'))})`))
  check('save offers a dialog, not a download link', (await count(inImage('.detail-actions a[download]'))) === 0)
  await shot('04-detail')
  await js(`window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }))`)
  await clickText(inImage('.tabs button'), 'Gallery')
  check('gallery lists both images', await waitFor(`document.querySelectorAll(${q(inImage('.tile'))}).length === 2`))
  check('gallery thumbnails decode', await waitFor(`[...document.querySelectorAll(${q(inImage('.tile img'))})].every(i => i.complete && i.naturalWidth > 0)`))
  check('favorite badge shows', (await count(inImage('.tile .badge'))) === 1)
  await shot('05-gallery')

  console.log('settings inside images')
  const first = await js(`window.api.image.gallery({ offset: 0, limit: 1 }).then((p) => p.items[0])`)
  check(
    'each image records the app and the exact model revision',
    first?.provenance?.app?.startsWith('Local AI Studio ') && first?.provenance?.model_revision?.startsWith('example/fake-model@'),
    JSON.stringify(first?.provenance)
  )
  const original = readFileSync(pngsUnder(join(profile, 'image', 'outputs')).find((p) => p.includes(first.id.slice(0, 6))))
  check("the gallery's own copy keeps its settings", original.includes('local-ai-studio') && original.includes('lighthouse'))
  await js(`window.api.image.saveImageAs(${q(first.id)})`)
  const stripped = readFileSync(exportPath)
  check(
    'a saved copy leaves the prompt and settings out by default',
    stripped.subarray(0, 8).equals(PNG_SIGNATURE) && !stripped.includes('lighthouse') && !stripped.includes('local-ai-studio')
  )
  await js(`window.api.image.setIncludeSettingsInExports(true)`)
  await js(`window.api.image.saveImageAs(${q(first.id)})`)
  check('and keeps them when that is switched on', readFileSync(exportPath).includes('local-ai-studio'))
  await js(`window.api.image.setIncludeSettingsInExports(false)`)

  await clickText(inImage('.tabs button'), 'Image')
  await setValue(inImage('textarea.prompt'), 'something else entirely')
  await dropFile(stripped, 'shared-copy.png')
  check('dropping a stripped copy says it has no settings', await waitFor(`document.querySelector(${q(inImage('.banner'))})?.textContent.includes("no settings saved")`))
  check('and leaves the form alone', await js(`document.querySelector(${q(inImage('textarea.prompt'))}).value === "something else entirely"`))
  await dropFile(original, 'from-gallery.png')
  check('dropping a gallery image brings its prompt back', await waitFor(`document.querySelector(${q(inImage('textarea.prompt'))}).value === ${q(first.prompt)}`))
  check(
    'with its seed locked, so it can be made again',
    await js(`(() => { const f = JSON.parse(localStorage.getItem("lis.form.v1")); return f.seed === ${first.seed} && f.randomSeed === false })()`)
  )
  check('the window says where the settings came from', await waitFor(`document.querySelector(${q(inImage('.banner'))})?.textContent.includes("from-gallery.png")`))

  console.log('settings')
  await clickText(inImage('.tabs button'), 'Settings')
  check('engine shows as running', await waitFor(`document.querySelector(${q(inImage('.settings'))})?.textContent.includes("Running")`))
  check('model folder is shown read-only with a picker', (await count(inImage('.settings input:not([type])'))) === 0)
  await clickText(inImage('.mode'), 'Balanced')
  check('memory mode saves', await waitFor(`document.querySelector(${q(inImage('.settings'))})?.textContent.includes("Saved.")`))
  await shot('06-settings')
  check('the video model has its own card', await waitFor(`document.querySelector(${q(inImage('.settings'))})?.textContent.includes("fake-video")`))
  check('settings say how videos will be saved', await waitFor(`/saved as MP4|saved as animated WebP/.test(document.querySelector(${q(inImage('.settings'))})?.textContent ?? "")`))

  // Settings search: cards and settings whose text has every word; a matching card title keeps the card whole.
  const shownCards = `[...document.querySelectorAll(${q(inImage('.settings > .card'))})].filter((c) => getComputedStyle(c).display !== "none").map((c) => c.querySelector("h2")?.textContent)`
  await setValue(inImage('.settings-search input'), 'ffmpeg')
  check('searching settings keeps the cards that mention it', await waitFor(`(() => { const s = ${shownCards}; return s.length === 1 && s[0] === "Video model" })()`), String(await js(shownCards)))
  check('and within them only the settings that do', await js(`[...document.querySelectorAll(${q(inImage('.settings > .card .field'))})].filter((f) => getComputedStyle(f).display !== "none").every((f) => f.textContent.toLowerCase().includes("ffmpeg"))`))
  await setValue(inImage('.settings-search input'), 'hidden folders')
  check('a card title keeps its card whole', await waitFor(`(() => { const s = ${shownCards}; return s.includes("Hidden folders") })()`))
  await setValue(inImage('.settings-search input'), 'no such setting anywhere')
  check('no match says so', await waitFor(`(() => { const e = document.querySelector(${q(inImage('.search-empty'))}); return !!e && !e.hidden && ${shownCards}.length === 0 })()`))
  await setValue(inImage('.settings-search input'), '')
  check('clearing the search shows every card again', await waitFor(`${shownCards}.length === document.querySelectorAll(${q(inImage('.settings > .card'))}).length`))

  console.log('engine install and model downloads')
  await clickText(inImage('.setup button'), 'Install the engine')
  check('installing first says what it downloads and from where', await waitFor(`document.querySelector(${q(inImage('.setup-box'))})?.textContent.includes("official PyTorch and PyPI indexes")`))
  await clickText(inImage('.setup-box button'), 'Cancel')
  check('and can be backed out of', await waitFor(`!document.querySelector(${q(inImage('.setup .setup-box'))})`))
  check('every supported model is offered for download', (await count(inImage('.download-row'))) === 4)
  const hardware = await js(`window.api.image.hardware()`)
  check('the list says which computer it is judged for', await waitFor(`document.querySelector(${q(inImage('.hardware'))})?.textContent.startsWith("This PC: ")`))
  check('naming its GPU, or that there is none', (await text(inImage('.hardware'))).includes(hardware.gpu ? hardware.gpu.name : 'No NVIDIA GPU found'))
  check('every model says how it would run here', (await count(inImage('.download-row .fit-note'))) === 4)
  check('a model marked Recommended runs well here', await js(`[...document.querySelectorAll(${q(inImage('.download-row'))})].every((row) => !row.querySelector(".tag:not(.fit-slow):not(.fit-no)") || row.querySelector(".fit-note.fit-good"))`))
  await clickText(inImage('.download-row button'), 'Download')
  check('a download first shows where the model comes from', await waitFor(`document.querySelector(${q(inImage('.download-row .facts'))})?.textContent.includes("@ f332072")`))
  check('and its license', (await text(inImage('.download-row .facts'))).includes('Apache License 2.0'))
  check('and the folder it goes to', await waitFor(`/image-model$/.test(document.querySelector(${q(inImage('.download-row .facts .path'))})?.textContent ?? "")`))
  const startDisabled = () => js(`[...document.querySelectorAll(${q(inImage('.download-row button.primary'))})].find((b) => b.textContent.startsWith("Start download"))?.disabled`)
  check('nothing starts until the license is accepted', (await startDisabled()) === true)
  await click(inImage('.download-row .toggle input'))
  check('accepting it enables the download', (await startDisabled()) === false)
  await clickText(inImage('.download-row .facts button'), 'Change folder')
  check('the models folder can be changed', await waitFor(`document.querySelector(${q(inImage('.download-row .facts .path'))})?.textContent.startsWith(${q(modelsDir)})`))
  check('and is remembered', await js(`window.api.getConfig().then((c) => c.image.modelsFolder === ${q(modelsDir)})`))
  await shot('06b-model-download')
  await clickText(inImage('.download-row .setup-box button'), 'Cancel')
  check('backing out starts nothing', (await js(`window.api.image.downloadState()`)).phase === 'idle')
  // MP4 needs ffmpeg on this machine (it is not bundled); without it the MP4 checks below are skipped.
  const mp4 = (await js(`window.api.image.status()`)).video?.encoder === 'mp4'
  const mp4Check = (name, ok, detail) => (mp4 ? check(name, ok, detail) : console.log(`  skip ${name} (no ffmpeg here)`))

  console.log('video')
  await clickText(inImage('.tabs button'), 'Video')
  check('video tab appears', await waitFor(`!!document.querySelector(${q(inImage('.video-panel'))})`))
  await setValue(inImage('.video-panel textarea.prompt'), 'Waves roll onto a beach at sunset. The camera pans slowly.')
  await clickText(inImage('.video-panel [aria-label="Size"] button'), '480p')
  await clickText(inImage('.video-panel [aria-label="Length"] button'), '2 s')
  await click(inImage('.video-panel .disclosure'))
  await setValue(inImage('.video-panel .advanced input[type="range"]'), 10) // steps: a quick fake run
  check('the form says what it will make', await js(`document.querySelector(${q(inImage('.video-panel'))}).textContent.includes("832 × 480") && document.querySelector(${q(inImage('.video-panel'))}).textContent.includes("49 frames")`))
  await click(inImage('.video-panel .generate-button'))
  check('video progress reports steps', await waitFor(`/Step \\d+ of 10/.test(document.querySelector(${q(inImage('.progress-card'))})?.textContent ?? "")`))
  check('the video lands in the filmstrip', await waitFor(`document.querySelectorAll(${q(inImage('.film'))}).length === 1`, 30000))
  const playable = `(() => { const v = document.querySelector(${q(inImage('.stage-video video'))}); return !!v && v.src.startsWith("studio-image://video/") && v.readyState >= 2 && v.videoWidth === 832 && Math.abs(v.duration - 49 / 24) < 0.1 })()`
  const videoState = `(() => { const v = document.querySelector(${q(inImage('.stage-video video'))}); return v ? JSON.stringify({ src: v.src, ready: v.readyState, net: v.networkState, error: v.error && [v.error.code, v.error.message], w: v.videoWidth, d: v.duration }) : "no video element" })()`
  mp4Check('it plays through studio-image: as an 832×480, 2-second MP4', mp4 && (await waitFor(playable, 15000)), mp4 && (await js(videoState)))
  mp4Check(
    'and seeks, which needs byte ranges through the protocol',
    mp4 && await js(`new Promise((done) => { const v = document.querySelector(${q(inImage('.stage-video video'))}); v.pause(); v.addEventListener("seeked", () => done(Math.abs(v.currentTime - 1.5) < 0.2), { once: true }); v.currentTime = 1.5; setTimeout(() => done(false), 5000) })`)
  )
  check('the status chip follows the video model', await waitFor(`document.querySelector(${q(inImage('.status-chip'))})?.textContent.includes("Video model ready")`))
  const engineStatus = await js(`window.api.image.status()`)
  check('the image model made room for it', engineStatus.models?.image?.state === 'unloaded' && engineStatus.models?.video?.state === 'ready')
  await shot('07-video')

  const [firstVideo] = (await js(`window.api.image.gallery({ media: "video" })`)).items
  check('the video is recorded with its settings', firstVideo?.media === 'video' && firstVideo.frames === 49 && firstVideo.fps === 24 && firstVideo.mime === (mp4 ? 'video/mp4' : 'image/webp'))
  await js(`window.api.image.saveImageAs(${q(firstVideo.id)})`)
  const savedVideo = readFileSync(exportPath)
  mp4Check('"Save as" writes the MP4, with no prompt inside', savedVideo.subarray(4, 8).toString() === 'ftyp' && !savedVideo.includes('Waves roll'))
  await js(`window.api.image.saveAsGif(${q(firstVideo.id)}, { width: 320, fps: 12 })`)
  const savedGif = readFileSync(exportPath)
  check(
    '"Save as GIF" writes a looping GIF at the size asked, with no prompt inside',
    savedGif.subarray(0, 6).toString() === 'GIF89a' && savedGif.readUInt16LE(6) === 320 && savedGif.readUInt16LE(8) === 185 &&
      savedGif.includes('NETSCAPE2.0') && !savedGif.includes('Waves roll'),
    `${savedGif.subarray(0, 6)} ${savedGif.readUInt16LE(6)}x${savedGif.readUInt16LE(8)} ${savedGif.length} bytes`
  )
  check(
    'and refuses options outside the limits',
    (await js(`window.api.image.saveAsGif(${q(firstVideo.id)}, { width: 5000, fps: 12 }).then(() => "saved", (e) => e.message)`)).includes('Invalid GIF options')
  )

  await click(inImage('.stage-details'))
  check('video details open with a player', await waitFor(`!!document.querySelector(${q(inImage('.modal.detail .video-frame'))})`))
  check('they show the length', await waitFor(`document.querySelector(${q(inImage('.modal.detail .meta'))})?.textContent.includes("49 frames at 24 fps")`))
  await clickText(inImage('.detail-actions button'), 'Save as GIF…')
  check(
    'a video offers a GIF copy, 480 wide at 12 fps to start',
    await waitFor(`document.querySelector(${q(inImage('.gif-export .gif-summary'))})?.textContent.includes("480 × 277 · 25 frames")`)
  )
  await clickText(inImage('.gif-export [aria-label="GIF width"] button'), '320')
  check('the summary follows the choice', await waitFor(`document.querySelector(${q(inImage('.gif-export .gif-summary'))})?.textContent.includes("320 × 185")`))
  await clickText(inImage('.gif-export button'), 'Save GIF…')
  check('saving it from the window works', await waitFor(`[...document.querySelectorAll(${q(inImage('.gif-export button'))})].some((b) => b.textContent === "Saved")`, 15000))
  check('and writes the GIF', readFileSync(exportPath).readUInt16LE(6) === 320)
  await shot('07b-video-gif')
  await clickText(inImage('.detail-actions button'), 'Continue from last frame')
  const startShown = `(() => { const i = document.querySelector(${q(inImage('.start-picture img'))}); return !!i && i.src.startsWith("studio-image://input/") && i.complete && i.naturalWidth === 832 })()`
  check('continuing puts the last frame in as the start picture', await waitFor(startShown))
  check('at the same size, so the clips join', await js(`document.querySelector(${q(inImage('.video-panel'))}).textContent.includes("832 × 480")`))
  await click(inImage('.video-panel .generate-button'))
  check('the second clip lands too', await waitFor(`document.querySelectorAll(${q(inImage('.film'))}).length === 2`, 30000))
  const videos = (await js(`window.api.image.gallery({ media: "video" })`)).items
  check('it records the picture it started from', typeof videos[0]?.start_image === 'string' && videos[0].start_image.length === 16)

  console.log('distilled video model')
  // A folder labelled like the 3-step distilled model: the engine tells the two apart by it.
  const fastDir = join(profile, 'fast-video-model')
  mkdirSync(fastDir, { recursive: true })
  writeFileSync(join(fastDir, 'model_index.json'), JSON.stringify({ _class_name: 'WanDMDPipeline' }))
  const videoDirBefore = (await js(`window.api.image.settings()`)).video_model_dir
  await js(`window.api.image.saveSettings({ video_model_dir: ${q(fastDir)} })`)
  check('the engine recognises the distilled model', await waitFor(`window.api.image.status().then((s) => s.video?.profile?.name === "fast" && s.video.defaults.steps === 3)`))
  await clickText(inImage('.video-panel [aria-label="Length"] button'), '2 s')
  const fastPanel = `document.querySelector(${q(inImage('.video-panel'))})`
  if (!(await js(`!!document.querySelector(${q(inImage('.video-panel .advanced'))})`))) await click(inImage('.video-panel .disclosure'))
  check(
    'its steps, guidance and negative prompt are shown as set by the model',
    await waitFor(`!!document.querySelector(${q(inImage('.video-panel .fixed-note'))}) && !${fastPanel}.querySelector('input[type="range"][max="50"]')`)
  )
  check('a short clip warns that it was trained on 5-second ones', await waitFor(`!!document.querySelector(${q(inImage('.video-panel .fast-length-note'))})`))
  await js(`window.api.image.saveSettings({ video_model_dir: ${q(videoDirBefore)} })`)
  check('back on the original model the steps slider returns', await waitFor(`!!${fastPanel}.querySelector('input[type="range"][max="50"]')`, 10000))

  console.log('start pictures')
  await js(`window.api.image.saveImageAs(${q(first.id)})`) // the picker answers with this file; the video export replaced it
  await click(inImage('.start-actions button'))
  check('a picture chosen from disk becomes the start picture', await waitFor(`(() => { const i = document.querySelector(${q(inImage('.start-picture img'))}); return !!i && i.complete && i.naturalWidth === 1024 })()`))
  check('and the video takes its square shape', await js(`document.querySelector(${q(inImage('.video-panel [aria-label="Shape"] [aria-checked="true"]'))})?.textContent.includes("1:1")`))
  await clickText(inImage('.start-actions button'), 'Remove')
  check('it can be removed', await waitFor(`!document.querySelector(${q(inImage('.start-picture'))})`))
  await dropFile(original, 'dropped-on-video.png')
  check('a picture dropped on the video tab becomes the start picture', await waitFor(startShown.replace('=== 832', '=== 1024')))
  await shot('08-video-start')

  console.log('gallery with videos')
  await clickText(inImage('.tabs button'), 'Gallery')
  await clickText(inImage('.media-filter button'), 'Videos')
  check('the gallery can show videos only', await waitFor(`document.querySelectorAll(${q(inImage('.tile'))}).length === 2 && document.querySelectorAll(${q(inImage('.tile .video-badge'))}).length === 2`))
  await clickText(inImage('.media-filter button'), 'Images')
  check('or images only', await waitFor(`document.querySelectorAll(${q(inImage('.tile'))}).length === 2 && document.querySelectorAll(${q(inImage('.tile .video-badge'))}).length === 0`))
  await click(inImage('.tile'))
  await clickText(inImage('.detail-actions button'), 'Animate')
  check('"Animate" opens the video tab with that image to start from', await waitFor(`!!document.querySelector(${q(inImage('.video-panel'))}) && ${startShown.replace('=== 832', '=== 1024')}`))
  check('and keeps the prompt already written', await js(`document.querySelector(${q(inImage('.video-panel textarea.prompt'))}).value.startsWith("Waves roll")`))
  await clickText(inImage('.tabs button'), 'Gallery')
  await clickText(inImage('.media-filter button'), 'All')
  await shot('09-gallery-mixed')

  // Thumbnail size, as a file manager's view sizes: the size it has always had, then two smaller.
  const gridSize = () => js(`document.querySelector(${q(inImage('.gallery-grid'))}).dataset.size`)
  const firstTileWidth = async () => {
    // The grid reloads when the filter changes, so wait for a thumbnail to be there.
    await waitFor(`!!document.querySelector(${q(inImage('.tile'))})`)
    return js(`document.querySelector(${q(inImage('.tile'))}).getBoundingClientRect().width`)
  }
  const sizeButton = (label) => `[...document.querySelectorAll(${q(inImage('.tile-size button'))})].find((b) => b.textContent === ${q(label)})`
  check('thumbnails start at the size they have always been', (await gridSize()) === 'large')
  check('and the buttons say so', await js(`(${sizeButton('Large')}).getAttribute("aria-checked") === "true"`))
  const largeWidth = await firstTileWidth()
  await js(`(${sizeButton('Small')}).click()`)
  check('a smaller size can be chosen', await waitFor(`document.querySelector(${q(inImage('.gallery-grid'))}).dataset.size === "small"`))
  const smallWidth = await firstTileWidth()
  check('and the thumbnails really are smaller', smallWidth < largeWidth, `${smallWidth} is not under ${largeWidth}`)
  await js(`(${sizeButton('Medium')}).click()`)
  const mediumWidth = await firstTileWidth()
  check('with a size in between', mediumWidth > smallWidth && mediumWidth < largeWidth, `${smallWidth} < ${mediumWidth} < ${largeWidth}`)
  check('the choice is kept for next time', await js(`localStorage.getItem("lis.gallery.tileSize.v1") === "medium"`))
  await shot('09b-gallery-medium')
  await js(`(${sizeButton('Large')}).click()`)
  check('and the size it started at can be chosen again', await waitFor(`document.querySelector(${q(inImage('.gallery-grid'))}).dataset.size === "large"`))

  // A mosaic over the thumbnails, for a screen other people can see. Off to begin with.
  const mosaicOf = () => js(`document.querySelector(${q(inImage('.gallery-grid'))}).dataset.mosaic`)
  const mosaicIs = (level) => waitFor(`document.querySelector(${q(inImage('.gallery-grid'))}).dataset.mosaic === ${q(level)}`)
  const mosaicButton = (label) => `[...document.querySelectorAll(${q(inImage('.mosaic-level button'))})].find((b) => b.textContent === ${q(label)})`
  const thumbDrawing = () =>
    js(`(() => {
      const tile = document.querySelector(${q(inImage('.tile'))});
      const canvas = tile.querySelector("canvas.mosaic");
      const shown = canvas ?? tile.querySelector("img");
      return {
        kind: canvas ? "mosaic" : "picture",
        rendering: getComputedStyle(shown).imageRendering,
        // How many blocks across the picture is drawn with, and whether it was drawn at all.
        blocks: canvas ? canvas.width : null,
        drawn: canvas ? canvas.dataset.drawn === "yes" : null,
        // The whole tile is covered by what is shown.
        coverage: shown.getBoundingClientRect().width / tile.getBoundingClientRect().width,
      };
    })()`)
  const tileBox = () =>
    js(`(() => { const r = document.querySelector(${q(inImage('.tile'))}).getBoundingClientRect(); return [Math.round(r.width), Math.round(r.height)] })()`)
  const boxBefore = await tileBox()
  const plain = await thumbDrawing()
  check('thumbnails are not mosaicked to begin with', (await mosaicOf()) === 'off' && plain.kind === 'picture', JSON.stringify(plain))
  await js(`(${mosaicButton('Weak')}).click()`)
  check('a mosaic can be put over them', await mosaicIs('weak'))
  await waitFor(`document.querySelector(${q(inImage('.tile canvas.mosaic'))})?.dataset.drawn === "yes"`)
  const weak = await thumbDrawing()
  check(
    'and the thumbnail is then drawn as blocks, not blurred',
    weak.kind === 'mosaic' && weak.rendering === 'pixelated' && weak.blocks === 24 && weak.drawn,
    JSON.stringify(weak)
  )
  // All of the tile but its own 1px border, which is what a plain thumbnail fills too.
  check('covering the whole thumbnail', weak.coverage >= 0.98, `${weak.coverage}`)
  // The first attempt laid the thumbnail out small and the tiles collapsed into strips.
  // Within a pixel: the tile's own aspect ratio counts its border, the thumbnail's height did not.
  const boxAfter = await tileBox()
  const sameBox = boxAfter.every((side, i) => Math.abs(side - boxBefore[i]) <= 1)
  check('and the grid keeps its shape', sameBox, `${boxAfter} was ${boxBefore}`)
  await js(`(${mosaicButton('Strong')}).click()`)
  await mosaicIs('strong')
  await waitFor(`document.querySelector(${q(inImage('.tile canvas.mosaic'))})?.dataset.drawn === "yes"`)
  const strong = await thumbDrawing()
  check('a stronger mosaic draws fewer, bigger blocks', strong.blocks === 8 && strong.drawn && strong.coverage >= 0.98, JSON.stringify(strong))
  check('the choice is kept for next time', await js(`localStorage.getItem("lis.gallery.mosaic.v1") === "strong"`))
  await shot('09c-gallery-mosaic')
  // The mosaic is a cover for the grid: opening an item still shows it as it is.
  // A still, by id: the newest tile may be a video, which has no <img> to look at.
  const [aStill] = (await js(`window.api.image.gallery({ media: "image" })`)).items
  await click(inImage(`.tile[data-id='${aStill.id}']`))
  const detail = await js(`(() => { const s = getComputedStyle(document.querySelector(${q(inImage('.detail-image img'))})); return [s.imageRendering, s.transform].join(" ") })()`)
  check('but an item opened from it is not mosaicked', detail === 'auto none', detail)
  await js(`document.querySelector(${q(inImage('.modal-backdrop'))}).dispatchEvent(new MouseEvent("mousedown", { bubbles: true }))`)
  await js(`(${mosaicButton('No mosaic')}).click()`)
  check('and the mosaic can be taken off again', (await mosaicIs('off')) && (await thumbDrawing()).kind === 'picture')

  console.log('gallery folders')
  await clickText(inImage('.tabs button'), 'Gallery')
  check('the gallery has All, Favorites and a folder list', await waitFor(`!!document.querySelector(${q(inImage('.folder-list'))}) && document.querySelector(${q(inImage('.folder-list'))}).textContent.includes("Favorites")`))
  await click(inImage('.folder-head button'))
  await nameFolder('Trips')
  check('a folder can be created', await waitFor(`!!(${folderRow('Trips')})`))
  check('and opens, empty', await waitFor(`document.querySelector(${q(inImage('.gallery-heading'))})?.textContent === "Trips" && document.querySelectorAll(${q(inImage('.tile'))}).length === 0`))
  await click(inImage('.folder-item'))  // back to All
  const everything = (await js(`window.api.image.gallery({})`)).items
  const [favorite] = (await js(`window.api.image.gallery({ favorites: true })`)).items
  const video = everything.find((i) => i.media === 'video')
  await waitFor(`document.querySelectorAll(${q(inImage('.tile'))}).length === ${everything.length}`)
  check('a tile can be dragged onto a folder', await dragTile(favorite.id, `.studio-pane:not([hidden]) .image-studio .folder-row`))
  check('which then counts it', await waitFor(`(${folderRow('Trips')})?.querySelector(".folder-count")?.textContent === "1"`))
  await click(inImage(`.tile[data-id='${video.id}']`))
  await js(`(() => { const s = document.querySelector(${q(inImage('.detail-folder select'))}); const opt = [...s.options].find((o) => o.textContent === "Trips"); s.value = opt.value; s.dispatchEvent(new Event("change", { bubbles: true })); return true })()`)
  check('or moved there from its details', await waitFor(`(${folderRow('Trips')})?.querySelector(".folder-count")?.textContent === "2"`))
  await js(`document.querySelector(${q(inImage('.modal-backdrop'))}).dispatchEvent(new MouseEvent("mousedown", { bubbles: true }))`)
  await js(`${folderRow('Trips')}.querySelector(".folder-item").click()`)
  check('the folder shows what is in it', await waitFor(`document.querySelectorAll(${q(inImage('.tile'))}).length === 2`))
  await clickText(inImage('.folder-item'), '★ Favorites')
  check(
    'favorites show every favorite, whatever folder it is in',
    await waitFor(`document.querySelector(${q(inImage('.gallery-heading'))})?.textContent === "Favorites" && document.querySelectorAll(${q(inImage('.tile'))}).length === 1 && !!document.querySelector(${q(inImage(`.tile[data-id="${favorite.id}"]`))})`)
  )
  check('"All" still shows everything', (await js(`window.api.image.gallery({})`)).total === everything.length)

  // The folder list can be put away, by its button or Ctrl+B, and the choice is kept.
  const folderListShown = `!!document.querySelector(${q(inImage('.folder-list'))})`
  await click(inImage('.folders-toggle'))
  check('the folder list can be hidden', await waitFor(`!${folderListShown}`))
  check('and the choice is kept', await js(`localStorage.getItem("lis.gallery.folders.v1") === "hidden"`))
  await js(`window.dispatchEvent(new KeyboardEvent("keydown", { key: "b", ctrlKey: true, bubbles: true }))`)
  check('Ctrl+B brings it back', await waitFor(folderListShown))

  // Putting the folders in an order by dragging them. A press is not a drag: the pointer
  // has to travel (REORDER_SLOP in GalleryView.tsx) before anything moves, so a click
  // never reorders the list.
  // Scoped to the gallery's own list: the Train tab's LoRA library uses .folder-row too.
  const GALLERY_ROWS = inImage('.gallery-view .folder-list .folder-row')
  const namesExpression = `[...document.querySelectorAll(${q(`${GALLERY_ROWS} .folder-name`)})].map((n) => n.textContent).join()`
  const folderNames = () => js(namesExpression)
  const orderIs = (names) => waitFor(`${namesExpression} === ${q(names)}`)
  const galleryRow = (name) =>
    `[...document.querySelectorAll(${q(GALLERY_ROWS)})].find((r) => r.querySelector(".folder-name")?.textContent === ${q(name)})`
  const engineFolderNames = async () => (await js(`window.api.image.folders()`)).map((f) => f.name).join()
  /** Press a folder row, move the pointer to `aim` (a function of the target row's box), let it go, then click as a mouse would. */
  const dragFolder = (name, target, aim) =>
    js(`(() => {
      const rows = [...document.querySelectorAll(${q(GALLERY_ROWS)})];
      const named = (text) => rows.find((r) => r.querySelector(".folder-name")?.textContent === text);
      const from = named(${q(name)}), to = named(${q(target)});
      if (!from || !to) throw new Error("no such folder row");
      const a = from.getBoundingClientRect(), b = to.getBoundingClientRect();
      const x = a.left + 20, y = a.top + a.height / 2;
      const aimY = (${aim})(b);
      from.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true, button: 0, isPrimary: true, clientX: x, clientY: y }));
      window.dispatchEvent(new PointerEvent("pointermove", { bubbles: true, clientX: x, clientY: aimY }));
      window.dispatchEvent(new PointerEvent("pointerup", { bubbles: true, clientX: x, clientY: aimY }));
      // The browser sends a click after the pointer goes up on the same element.
      from.querySelector(".folder-item").click();
      return true;
    })()`)
  // One at a time: creating a folder is a round trip, and the name box stays open until it is done.
  const newFolder = async (name) => {
    await click(inImage('.folder-head button'))
    await nameFolder(name)
    await waitFor(`!!(${galleryRow(name)})`)
  }
  await newFolder('Alps')
  await newFolder('Zoo')
  check('folders are in alphabetical order to begin with', await orderIs('Alps,Trips,Zoo'), await folderNames())

  // A press that barely moves: the folder is selected, and the list is left alone.
  await clickText(inImage('.folder-item'), 'All')
  await dragFolder('Zoo', 'Zoo', '(b) => b.top + b.height / 2 + 3')
  check('a press that barely moves does not reorder anything', (await folderNames()) === 'Alps,Trips,Zoo')
  check('and still selects the folder, as a click', await waitFor(`document.querySelector(${q(inImage('.gallery-heading'))})?.textContent === "Zoo"`))

  // A deliberate pull: Zoo lands in front of Trips, and the click that ends it is not a selection.
  await clickText(inImage('.folder-item'), 'All')
  await dragFolder('Zoo', 'Trips', '(b) => b.top + 2')
  check('dragging a folder puts it where it was dropped', await orderIs('Alps,Zoo,Trips'), await folderNames())
  check('and the drag does not open the folder it was dragging', await js(`document.querySelector(${q(inImage('.gallery-heading'))})?.textContent === "All"`))
  check('the engine keeps the order', (await engineFolderNames()) === 'Alps,Zoo,Trips')

  // Dropped below the last row: the end of the list.
  await dragFolder('Alps', 'Trips', '(b) => b.bottom + 30')
  check('a folder dropped past the last one goes to the end', await orderIs('Zoo,Trips,Alps'), await folderNames())

  // Without a mouse: Alt and an arrow move the folder in focus one place.
  await js(`(${galleryRow('Alps')}).querySelector(".folder-item").focus()`)
  await js(`(${galleryRow('Alps')}).querySelector(".folder-item").dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowUp", altKey: true, bubbles: true }))`)
  check('Alt+Up moves a folder up one place', await orderIs('Zoo,Alps,Trips'), await folderNames())
  await js(`(${galleryRow('Zoo')}).querySelector(".folder-item").dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowDown", altKey: true, bubbles: true }))`)
  check('and Alt+Down moves one down', await orderIs('Alps,Zoo,Trips'), await folderNames())
  await newFolder('Aaa')
  check('a new folder joins the end once the folders have an order', await orderIs('Alps,Zoo,Trips,Aaa'), await folderNames())

  // The three extra folders go again, so the rest of the test sees the gallery it expects.
  for (const name of ['Aaa', 'Alps', 'Zoo']) {
    await js(`(${galleryRow(name)}).querySelector('[aria-label="Options for ${name}"]').click()`)
    await clickText(inImage('.folder-menu button'), 'Delete folder…')
    await clickText(inImage('.folder-confirm button'), 'Delete folder')
    await waitFor(`!(${galleryRow(name)})`)
  }
  check('only the folder the rest of the test uses is left', (await folderNames()) === 'Trips', await folderNames())

  // Where the gallery opens: All by default, or where it was left.
  await clickText(inImage('.folder-item'), 'Trips')
  await waitFor(`document.querySelector(${q(inImage('.gallery-heading'))})?.textContent === "Trips"`)
  await clickText(inImage('.tabs button'), 'Image')
  await clickText(inImage('.tabs button'), 'Gallery')
  check('the gallery opens on All by default', await waitFor(`document.querySelector(${q(inImage('.gallery-heading'))})?.textContent === "All"`))
  await clickText(inImage('.folder-item'), 'Trips')
  await clickText(inImage('.tabs button'), 'Settings')
  await clickText(inImage('[aria-label="The gallery opens on"] button'), 'Where it was left')
  await clickText(inImage('.tabs button'), 'Gallery')
  check('or where it was left, when chosen', await waitFor(`document.querySelector(${q(inImage('.gallery-heading'))})?.textContent === "Trips"`))
  await clickText(inImage('.folder-item'), 'All')
  await clickText(inImage('.tabs button'), 'Settings')
  await clickText(inImage('[aria-label="The gallery opens on"] button'), 'Favorites')
  await clickText(inImage('.tabs button'), 'Gallery')
  check('or on Favorites', await waitFor(`document.querySelector(${q(inImage('.gallery-heading'))})?.textContent === "Favorites"`))
  await clickText(inImage('.tabs button'), 'Settings')
  await clickText(inImage('[aria-label="The gallery opens on"] button'), 'A folder')
  check('or on a chosen folder, picked from a list', await waitFor(`[...document.querySelectorAll(${q(inImage('[aria-label="The folder the gallery opens on"] option'))})].some((o) => o.textContent === "Trips")`))
  await js(`(() => { const s = document.querySelector(${q(inImage('[aria-label="The folder the gallery opens on"]'))}); const id = [...s.options].find((o) => o.textContent === "Trips").value; const set = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value").set; set.call(s, id); s.dispatchEvent(new Event("change", { bubbles: true })) })()`)
  await clickText(inImage('.tabs button'), 'Gallery')
  check('which it then opens on', await waitFor(`document.querySelector(${q(inImage('.gallery-heading'))})?.textContent === "Trips"`))
  await clickText(inImage('.tabs button'), 'Settings')
  await clickText(inImage('[aria-label="The gallery opens on"] button'), 'All')
  await clickText(inImage('.tabs button'), 'Gallery')
  await waitFor(`document.querySelector(${q(inImage('.gallery-heading'))})?.textContent === "All"`)

  // A folder's own actions: only ⋯ (and a lock for an open PIN folder) on the row; rename and delete in its menu.
  const openMenu = (name) => js(`${folderRow(name)}.querySelector('[aria-label="Options for ${name}"]').click()`)
  const folderMenuOpen = `!!document.querySelector(${q(inImage('.folder-menu'))})`
  check('a folder row offers only its menu', await js(`[...(${folderRow('Trips')}).querySelectorAll(".folder-actions button")].map((b) => b.textContent.trim()).join() === "⋯"`))
  await openMenu('Trips')
  check('its menu opens', await waitFor(folderMenuOpen))
  await js(`document.querySelector(${q(inImage('.gallery-heading'))}).dispatchEvent(new PointerEvent("pointerdown", { bubbles: true }))`)
  check('and closes on a click anywhere else', await waitFor(`!${folderMenuOpen}`))
  await openMenu('Trips')
  await waitFor(folderMenuOpen)
  await js(`document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }))`)
  check('or on Escape', await waitFor(`!${folderMenuOpen}`))
  await openMenu('Trips')
  await clickText(inImage('.folder-menu button'), 'Rename')
  await nameFolder('Holidays')
  check('a folder can be renamed', await waitFor(`!!(${folderRow('Holidays')}) && !(${folderRow('Trips')})`))
  await click(inImage('.folder-head button'))
  await nameFolder('holidays')
  check('two folders cannot share a name', await waitFor(`document.querySelector(${q(inImage('.banner'))})?.textContent.includes("already a folder")`))
  await js(`document.querySelector(${q(inImage('.folder-name-input input'))})?.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }))`)
  await shot('11-gallery-folders')

  check('dragging onto All takes an item out of its folder', await dragTile(favorite.id, `.studio-pane:not([hidden]) .image-studio .folder-item`))
  check('and its folder counts one fewer', await waitFor(`(${folderRow('Holidays')})?.querySelector(".folder-count")?.textContent === "1"`))

  // Right-click on a tile: move to a folder, take out of it, delete (asked first).
  const rightClick = (id) =>
    js(`(() => { const t = document.querySelector(${q(inImage(`.tile[data-id="${id}"]`))}); if (!t) return false; const r = t.getBoundingClientRect(); t.dispatchEvent(new MouseEvent("contextmenu", { bubbles: true, cancelable: true, clientX: r.left + 10, clientY: r.top + 10 })); return true })()`)
  const menuOpen = `!!document.querySelector(${q(inImage('.tile-menu'))})`
  check('right-clicking a tile opens its menu', (await rightClick(favorite.id)) && (await waitFor(menuOpen)))
  check('which offers the folders to move it to', (await text(inImage('.tile-menu'))).includes('Holidays'))
  await clickText(inImage('.tile-menu button'), 'Holidays')
  check('moving there files it', await waitFor(`(${folderRow('Holidays')})?.querySelector(".folder-count")?.textContent === "2"`))
  check('and closes the menu', await waitFor(`!${menuOpen}`))
  await rightClick(favorite.id)
  await waitFor(menuOpen)
  await clickText(inImage('.tile-menu button'), 'Take out of')
  check('it can be taken out of its folder the same way', await waitFor(`(${folderRow('Holidays')})?.querySelector(".folder-count")?.textContent === "1"`))
  await rightClick(favorite.id)
  await waitFor(menuOpen)
  await clickText(inImage('.tile-menu button'), 'Delete')
  check('deleting from the menu asks first', await waitFor(`document.querySelector(${q(inImage('.tile-menu-confirm'))})?.textContent.includes("cannot be undone")`))
  await clickText(inImage('.tile-menu-confirm button'), 'Cancel')
  await js(`document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }))`)
  check('and Escape closes it, leaving the item', (await waitFor(`!${menuOpen}`)) && (await js(`window.api.image.gallery({})`)).total === everything.length)

  await openMenu('Holidays')
  await clickText(inImage('.folder-menu button'), 'Delete folder')
  check('deleting asks first, and says the items stay', await waitFor(`document.querySelector(${q(inImage('.folder-confirm'))})?.textContent.includes("stay in All")`))
  await clickText(inImage('.folder-confirm button'), 'Delete folder')
  check('the folder goes', await waitFor(`!(${folderRow('Holidays')})`))
  check('and nothing that was in it is lost', (await js(`window.api.image.gallery({})`)).total === everything.length)
  check('it is back in no folder', (await js(`window.api.image.gallery({})`)).items.find((i) => i.id === video.id)?.folder_id === null)

  console.log('choosing several, folders by keyboard')
  const keyDown = (init) => js(`window.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, ...${JSON.stringify(init)} }))`)
  const ctrlM = { key: 'm', code: 'KeyM', ctrlKey: true }
  const digit = (n) => ({ key: String(n), code: `Digit${n}` })
  const clickTile = (id, mods = {}) =>
    js(`document.querySelector(${q(inImage(`.tile[data-id="${id}"]`))}).dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, ...${JSON.stringify(mods)} }))`)
  const barText = `(document.querySelector(${q(inImage('.selection-bar'))})?.textContent ?? "")`
  const foldersOf = (ids) => `window.api.image.gallery({}).then((p) => ${JSON.stringify(ids)}.map((id) => p.items.find((i) => i.id === id)?.folder_id ?? null))`
  const until = async (test, ms = 10000) => {
    for (const end = Date.now() + ms; Date.now() < end; await sleep(100)) if (test()) return true
    return false
  }
  const folderOf = (id) => `window.api.image.gallery({}).then((p) => p.items.find((i) => i.id === ${q(id)})?.folder_id ?? null)`
  await click(inImage('.folder-head button'))
  await nameFolder('Picks A')
  await waitFor(`!!(${folderRow('Picks A')})`)
  await click(inImage('.folder-head button'))
  await nameFolder('Picks B')
  await waitFor(`!!(${folderRow('Picks B')})`)
  const listed = await js(`window.api.image.folders()`)
  const keyOf = (name) => listed.findIndex((f) => f.name === name) + 1
  const idOf = (name) => listed.find((f) => f.name === name).id
  await click(inImage('.folder-item')) // All
  await waitFor(`document.querySelectorAll(${q(inImage('.tile'))}).length === ${everything.length}`)
  check('there are enough items to choose from', everything.length >= 4, String(everything.length))
  const order = await js(`[...document.querySelectorAll(${q(inImage('.tile'))})].map((t) => t.dataset.id)`)

  await clickTile(order[0], { ctrlKey: true })
  check('Ctrl+click chooses a tile instead of opening it', await waitFor(`${barText}.includes("1 chosen") && !document.querySelector(${q(inImage('.modal.detail'))})`))
  await clickTile(order[2], { shiftKey: true })
  check('Shift+click chooses the run up to it', await waitFor(`${barText}.includes("3 chosen") && document.querySelectorAll(${q(inImage('.tile.selected'))}).length === 3`))
  await keyDown(ctrlM)
  check('the shortcut opens a numbered folder list for them', await waitFor(`document.querySelector(${q(inImage('.selection-bar .folder-picker'))})?.textContent.includes("Picks A")`))
  await shot('05b-gallery-choose')
  await keyDown(digit(keyOf('Picks A')))
  check(
    'a number moves all of them to that folder at once',
    await waitFor(`${foldersOf(order.slice(0, 3))}.then((f) => f.every((x) => x === ${q(idOf('Picks A'))}))`)
  )
  check('and says so', await waitFor(`document.querySelector(${q(inImage('.banner'))})?.textContent.includes("Moved 3 items")`))
  check('which ends the choosing', await waitFor(`!document.querySelector(${q(inImage('.selection-bar'))})`))

  await click(inImage('.select-toggle'))
  await clickTile(order[3])
  await clickTile(order[1])
  check('in Select mode a plain click chooses', await waitFor(`${barText}.includes("2 chosen")`))
  await clickText(inImage('.selection-bar button'), 'Save copies')
  check('saving offers what to keep in the copies', await waitFor(`document.querySelector(${q(inImage('.save-many'))})?.textContent.includes(".txt")`))
  check('keeping the prompt inside starts as the Save as setting has it (off)', await js(`!document.querySelectorAll(${q(inImage('.save-many input[type=checkbox]'))})[0].checked`))
  await js(`document.querySelectorAll(${q(inImage('.save-many input[type=checkbox]'))})[1].click()`)
  await clickText(inImage('.save-many button'), 'Choose a folder and save')
  check('saving says where they went', await waitFor(`document.querySelector(${q(inImage('.banner'))})?.textContent.includes("Saved 2 items to")`))
  const savedNow = readdirSync(exportFolder).sort()
  const savedMedia = savedNow.filter((f) => !f.endsWith('.txt'))
  check('both are there, each with its prompt file', savedMedia.length === 2 && savedNow.length === 4, savedNow.join(', '))
  const savedPng = savedMedia.find((f) => f.endsWith('.png'))
  check(
    'an image copy leaves the prompt out, as chosen, and its .txt holds it',
    !savedPng ||
      (!readFileSync(join(exportFolder, savedPng)).includes('local-ai-studio') &&
        readFileSync(join(exportFolder, savedPng.replace(/[.]png$/, '.txt')), 'utf8').includes('Seed:')),
    savedPng
  )
  await clickText(inImage('.selection-bar button'), 'Save copies')
  await waitFor(`!!document.querySelector(${q(inImage('.save-many'))})`)
  await clickText(inImage('.save-many button'), 'Choose a folder and save')
  check(
    'saving again never overwrites: the new copies get numbers (the .txt option starts off again)',
    (await until(() => readdirSync(exportFolder).length === 6)) && readdirSync(exportFolder).filter((f) => f.includes(' (2).')).length === 2,
    readdirSync(exportFolder).join(', ')
  )
  await clickText(inImage('.selection-bar button'), 'Delete')
  check('deleting the chosen ones asks first', await waitFor(`document.querySelector(${q(inImage('.selection-bar [role=alertdialog]'))})?.textContent.includes("Delete 2 items for good")`))
  await clickText(inImage('.selection-bar [role=alertdialog] button'), 'Cancel')
  await keyDown({ key: 'a', code: 'KeyA', ctrlKey: true })
  check('Ctrl+A chooses every item shown', await waitFor(`${barText}.includes("${everything.length} chosen")`))
  await keyDown({ key: 'Escape', code: 'Escape' })
  check('and Escape lets go', await waitFor(`!document.querySelector(${q(inImage('.selection-bar'))}) && !document.querySelector(${q(inImage('.tile.selected'))})`))
  check('nothing was deleted', (await js(`window.api.image.gallery({})`)).total === everything.length)

  await clickTile(order[1])
  check('a plain click opens the item again', await waitFor(`!!document.querySelector(${q(inImage('.modal.detail'))})`))
  const detailSrc = `(document.querySelector(${q(inImage('.modal.detail .detail-image img, .modal.detail .detail-image video'))})?.getAttribute("src") ?? "")`
  const firstSrc = await js(detailSrc)
  await keyDown(ctrlM)
  check('in its details the shortcut opens the numbered folder list', await waitFor(`!!document.querySelector(${q(inImage('.detail-folder-wrap .folder-picker'))})`))
  await keyDown(digit(keyOf('Picks B')))
  check('a number files it there', await waitFor(`${folderOf(order[1])}.then((f) => f === ${q(idOf('Picks B'))})`))
  check('and stays on the same item, with the list closed', await waitFor(`${detailSrc} === ${q(firstSrc)} && !document.querySelector(${q(inImage('.folder-picker'))})`))
  await js(`document.querySelector(${q(inImage('.detail-folder select'))}).focus()`)
  await js(`document.querySelector(${q(inImage('.detail-folder select'))}).dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight", code: "ArrowRight", bubbles: true, cancelable: true }))`)
  check('the right arrow goes to the next item even from the folder list', await waitFor(`${detailSrc} !== ${q(firstSrc)}`))
  check('and leaves the folder of the one before alone', await js(`${folderOf(order[1])}.then((f) => f === ${q(idOf('Picks B'))})`))
  await keyDown({ key: 'ArrowLeft', code: 'ArrowLeft' })
  check('the left arrow goes back', await waitFor(`${detailSrc} === ${q(firstSrc)}`))
  await keyDown(ctrlM)
  await waitFor(`!!document.querySelector(${q(inImage('.detail-folder-wrap .folder-picker'))})`)
  await keyDown(digit(0))
  check('0 takes it out of its folder', await waitFor(`${folderOf(order[1])}.then((f) => f === null)`))
  await keyDown({ key: 'Escape', code: 'Escape' })
  await waitFor(`!document.querySelector(${q(inImage('.modal.detail'))})`)
  for (const name of ['Picks A', 'Picks B']) {
    await openMenu(name)
    await clickText(inImage('.folder-menu button'), 'Delete folder')
    await clickText(inImage('.folder-confirm button'), 'Delete folder')
    await waitFor(`!(${folderRow(name)})`)
  }
  check('with those folders gone, everything is back in no folder', (await js(`window.api.image.gallery({})`)).items.every((i) => i.id === favorite.id || i.folder_id === null))

  console.log('hidden and locked folders')
  const pinField = (label) => inImage(`.folder-pin input[aria-label="${label}"]`)
  const submitPin = () => js(`document.querySelector(${q(inImage('.folder-pin'))}).requestSubmit()`)
  const tileShown = (id) => `!!document.querySelector(${q(inImage(`.tile[data-id="${id}"]`))})`
  await click(inImage('.folder-head button'))
  await nameFolder('Vault')
  check('another folder is made', await waitFor(`!!(${folderRow('Vault')})`))
  await click(inImage('.folder-item'))  // All
  await waitFor(tileShown(favorite.id))
  // Onto that folder by id, not "the first row": a list with one row more than the
  // test expected would otherwise drop the item somewhere else and fail obscurely.
  const vaultRow = (await js(`window.api.image.folders()`)).find((f) => f.name === 'Vault')?.id
  check(
    'and given an item',
    (await dragTile(favorite.id, `${GALLERY_ROWS}[data-folder-id="${vaultRow}"]`)) && (await waitFor(`(${folderRow('Vault')})?.querySelector(".folder-count")?.textContent === "1"`)),
    `tile ${await js(tileShown(favorite.id))}, rows ${await count(GALLERY_ROWS)}, heading ${await js(`document.querySelector(${q(inImage('.gallery-heading'))})?.textContent`)}`
  )
  await openMenu('Vault')
  await clickText(inImage('.folder-menu button'), 'Lock with a PIN')
  await setValue(pinField('PIN'), '246810')
  await setValue(pinField('PIN again'), '135790')
  await submitPin()
  check('two different PINs are refused', await waitFor(`document.querySelector(${q(inImage('.folder-pin [role=alert]'))})?.textContent.includes("not the same")`))
  await setValue(pinField('PIN'), '246810')
  await setValue(pinField('PIN again'), '246810')
  await submitPin()
  check('a folder can be locked with a PIN, and stays open for now', await waitFor(`window.api.image.folders().then((f) => f.some((x) => x.name === "Vault" && x.locked && x.unlocked))`))
  check('an open PIN folder shows a lock button beside its menu', await js(`[...(${folderRow('Vault')}).querySelectorAll(".folder-actions button")].map((b) => b.textContent.trim()).join() === "🔒,⋯"`))
  await js(`${folderRow('Vault')}.querySelector('[aria-label="Lock Vault now"]').click()`)
  check('locking it shows a lock instead of a count', await waitFor(`!!(${folderRow('Vault')})?.querySelector(".folder-lock")`))
  check('and takes what it holds out of All', await waitFor(`!(${tileShown(favorite.id)}) && document.querySelectorAll(${q(inImage('.tile'))}).length === ${everything.length - 1}`))
  check('and out of Favorites and search', (await js(`window.api.image.gallery({ favorites: true })`)).total === 0)
  await js(`${folderRow('Vault')}.querySelector(".folder-item").click()`)
  await setValue(pinField('PIN'), '000000')
  await submitPin()
  check('a wrong PIN is refused', await waitFor(`document.querySelector(${q(inImage('.folder-pin [role=alert]'))})?.textContent.includes("Wrong PIN")`))
  await setValue(pinField('PIN'), '246810')
  await submitPin()
  check('the right one opens it', await waitFor(`document.querySelector(${q(inImage('.gallery-heading'))})?.textContent === "Vault" && ${tileShown(favorite.id)}`))
  await shot('12-gallery-locked-folder')
  // Ctrl+Shift+L locks every open folder; a view of one goes to Favorites.
  await js(`window.dispatchEvent(new KeyboardEvent("keydown", { key: "L", ctrlKey: true, shiftKey: true, bubbles: true }))`)
  check('Ctrl+Shift+L locks every open folder', await waitFor(`window.api.image.folders().then((f) => f.every((x) => !x.unlocked))`))
  check('and leaves the locked folder for Favorites', await waitFor(`document.querySelector(${q(inImage('.gallery-heading'))})?.textContent === "Favorites" && !!(${folderRow('Vault')})?.querySelector(".folder-lock")`))
  check('without showing what it holds', !(await js(tileShown(favorite.id))))
  await js(`${folderRow('Vault')}.querySelector(".folder-item").click()`)
  await setValue(pinField('PIN'), '246810')
  await submitPin()
  await waitFor(`document.querySelector(${q(inImage('.gallery-heading'))})?.textContent === "Vault"`)
  await openMenu('Vault')
  await clickText(inImage('.folder-menu button'), 'Hide')
  check('a hidden folder leaves the folder list', await waitFor(`!(${folderRow('Vault')})`))
  check('and the view goes back to All, without its items', await waitFor(`document.querySelector(${q(inImage('.gallery-heading'))})?.textContent === "All" && !(${tileShown(favorite.id)})`))
  await clickText(inImage('.tabs button'), 'Settings')
  await clickText(inImage('.settings label.toggle'), 'Show hidden folders')
  check('settings can show hidden folders', await waitFor(`window.api.image.showHiddenFolders()`))
  await clickText(inImage('.tabs button'), 'Gallery')
  check('which come back, marked hidden', await waitFor(`(${folderRow('Vault')})?.querySelector(".folder-tag")?.textContent === "hidden"`))
  await click(inImage('.folder-item'))

  console.log('without ffmpeg')
  await js(`window.api.image.saveSettings({ ffmpeg_path: ${q(join(dirname(workspaceDir), 'missing', 'ffmpeg.exe'))} })`)
  await clickText(inImage('.tabs button'), 'Video')
  check('the video tab warns that videos will be WebP', await waitFor(`document.querySelector(${q(inImage('.video-panel'))})?.textContent.includes("animated WebP")`, 8000))
  await clickText(inImage('.start-actions button'), 'Remove')
  await click(inImage('.video-panel .generate-button'))
  check('a video still comes out', await waitFor(`document.querySelectorAll(${q(inImage('.film'))}).length === 3`, 30000))
  check('as an animated WebP that plays as a picture', await waitFor(`(() => { const i = document.querySelector(${q(inImage('.stage-video img.video-frame'))}); return !!i && i.src.startsWith("studio-image://video/") && i.complete && i.naturalWidth > 0 })()`))
  await js(`window.api.image.saveSettings({ ffmpeg_path: "" })`)

  console.log('what shows once something is made')
  const toggle = (label) => clickText(inImage('.settings label.toggle'), label)
  const stageText = () => text(inImage('.stage'))
  await clickText(inImage('.tabs button'), 'Settings')
  await toggle('Show a finished video on screen')
  await clickText(inImage('.tabs button'), 'Video')
  check('with results off, the finished video leaves the stage', await waitFor(`!document.querySelector(${q(inImage('.stage-video'))})`))
  check('and the stage says it is done', (await stageText()).includes('Done. It is in the gallery.'))
  check('while the strip stays', (await count(inImage('.film'))) === 3)
  await click(inImage('.film'))
  check('a thumbnail picked from the strip still shows', await waitFor(`!!document.querySelector(${q(inImage('.stage-video'))})`))
  await clickText(inImage('.tabs button'), 'Settings')
  await toggle('Show this session’s videos as thumbnails under it')
  check('the choices are kept', await js(`(() => { const d = JSON.parse(localStorage.getItem("lis.display.v1")); return d.video.showResult === false && d.video.showStrip === false && d.image.showResult && d.image.showStrip })()`))
  await clickText(inImage('.tabs button'), 'Video')
  check('with the strip off, no thumbnails show', await waitFor(`document.querySelectorAll(${q(inImage('.film'))}).length === 0`))
  await clickText(inImage('.tabs button'), 'Image')
  check('and images keep their own choices', await waitFor(`document.querySelectorAll(${q(inImage('.film'))}).length > 0 && !!document.querySelector(${q(inImage('.stage-image'))})`))
  await clickText(inImage('.tabs button'), 'Settings')
  await toggle('Show this session’s images as thumbnails under it')
  await clickText(inImage('.tabs button'), 'Image')
  check('the image strip turns off on its own', await waitFor(`document.querySelectorAll(${q(inImage('.film'))}).length === 0 && !!document.querySelector(${q(inImage('.stage-image'))})`))
  await clickText(inImage('.tabs button'), 'Settings')
  await toggle('Show this session’s images as thumbnails under it')
  await toggle('Show this session’s videos as thumbnails under it')
  await toggle('Show a finished video on screen')
  check('and all of it turns back on', await js(`(() => { const d = JSON.parse(localStorage.getItem("lis.display.v1")); return d.video.showResult && d.video.showStrip && d.image.showResult && d.image.showStrip })()`))

  console.log('training')
  const train = (sel) => inImage(`.train-view ${sel}`)
  const cardsText = `[...document.querySelectorAll(${q(train('.caption-text'))})].map((e) => e.textContent)`
  await clickText(inImage('.tabs button'), 'Train')
  check('the Train tab opens', await waitFor(`!!document.querySelector(${q(train('.train-new'))})`))
  await setValue(train('.train-new input'), 'Example coupes')
  await js(`document.querySelector(${q(train('.train-new'))}).requestSubmit()`)
  check('a project is made from the chosen folder and scanned', await waitFor(`document.querySelectorAll(${q(train('.caption-card'))}).length === 4`, 20000))
  check('captions come from the folder names', await js(`${cardsText}.every((t) => t === "Example Motors Coupe GT")`))
  check('a new project is about any subject, and asks what the photos show', await js(`(() => { const t = document.querySelector(${q(train('.subject-section'))})?.textContent ?? ""; return t.includes("Not said what the photos show yet") && t.includes("Name / Variant") })()`))
  await clickText(train('.subject-section button'), 'Edit subject')
  await setValue(train('.subject-section input[aria-label="What the photos show"]'), 'coupe')
  await setValue(train('.subject-section input[aria-label="Trigger word"]'), 'excoupe')
  check('the editor shows an example caption', await waitFor(`document.querySelector(${q(train('.subject-preview'))})?.textContent.includes("excoupe, <Name> <Variant>, coupe, front view")`))
  await setValue(train('.subject-section input[aria-label="Trigger word"]'), 'ex coupe')
  check('a bad definition is explained, and cannot be saved', await waitFor(`(() => { const e = document.querySelector(${q(train('.subject-editor .error-text'))}); const b = [...document.querySelectorAll(${q(train('.subject-section button'))})].find((x) => x.textContent.startsWith("Save")); return !!e && e.textContent.includes("one word") && b.disabled })()`))
  await setValue(train('.subject-section input[aria-label="Trigger word"]'), 'excoupe')
  await clickText(train('.subject-section button'), 'Save and rescan')
  check('the subject is saved and the captions follow it', await waitFor(`${cardsText}.length === 4 && ${cardsText}.every((t) => t === "excoupe, Example Motors Coupe GT, coupe")`, 20000))
  check('the duplicate and the too-small photo are reported', await waitFor(`(() => { const t = document.querySelector(${q(train('.train-main'))}).textContent; return t.includes("1 duplicates") && t.includes("1 too small") })()`))
  check('photos show as thumbnails', await waitFor(`(() => { const i = document.querySelector(${q(train('.caption-card img'))}); return !!i && i.src.startsWith("studio-image://training/") && i.complete && i.naturalWidth > 0 })()`),
    await js(`(async () => {
      const imgs = [...document.querySelectorAll(${q(train('.caption-card img'))})].map((i) => [i.getAttribute("src").slice(-22), i.complete, i.naturalWidth]);
      const src = document.querySelector(${q(train('.caption-card img'))})?.src;
      const fresh = await new Promise((r) => { const i = new Image(); const t = setTimeout(() => r("pending 3s"), 3000); i.onload = () => (clearTimeout(t), r("load")); i.onerror = () => (clearTimeout(t), r("error")); i.src = src; });
      const gallery = await window.api.image.gallery({ limit: 1 });
      const g = await new Promise((r) => { const i = new Image(); const t = setTimeout(() => r("pending 3s"), 3000); i.onload = () => (clearTimeout(t), r("load")); i.onerror = () => (clearTimeout(t), r("error")); i.src = gallery.items[0].thumb_url; });
      const open = performance.getEntriesByType("resource").filter((e) => e.name.startsWith("studio-image:") && e.responseEnd === 0).map((e) => e.name.slice(15, 60));
      return JSON.stringify({ imgs, fresh, galleryThumb: g, videos: document.querySelectorAll("video").length, unfinished: open });
    })()`))
  await clickText(train('.caption-card .caption-actions button'), 'Edit')
  await js(`(() => { const s = document.querySelector(${q(train('.caption-editor select'))}); s.value = "side"; s.dispatchEvent(new Event("change", { bubbles: true })); return true })()`)
  await clickText(train('.caption-editor button'), 'Save')
  check('a caption can be corrected by hand', await waitFor(`${cardsText}.some((t) => t === "excoupe, Example Motors Coupe GT, coupe, side view")`))
  const firstPhoto = await js(`document.querySelectorAll(${q(train('.caption-card'))})[3].querySelector(".caption-text").textContent`)
  await js(`[...document.querySelectorAll(${q(train('.caption-card'))})][3].querySelector(".danger-text").click()`)
  check('a photo can be excluded', await waitFor(`!!document.querySelector(${q(train('.caption-card.excluded'))})`) && firstPhoto.length > 0)
  await clickText(train('.train-section button'), 'Describe photos')
  check('the captioning model describes the photos', await waitFor(`${cardsText}.filter((t) => t === "excoupe, Example Motors Coupe GT, coupe, front view, grey, grey, daylight, whole in frame").length === 3`, 20000),
    await js(`JSON.stringify(${cardsText})`))
  check('a hand correction outlives the model', await js(`${cardsText}.some((t) => t.startsWith("excoupe, Example Motors Coupe GT, coupe, side view"))`))
  check('and it says how many it looked at', await waitFor(`document.querySelector(${q(train('.train-main'))}).textContent.includes("4 of 4 described")`))
  await shot('12-train-captions')

  await setValue(train('.train-settings input[type="number"]'), 4)
  await js(`document.querySelectorAll(${q(train('.train-settings input[type="number"]'))})[1].focus()`)
  await js(`(() => {
    const [steps, every] = document.querySelectorAll(${q(train('.train-settings input[type="number"]'))});
    const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
    set.call(every, "2"); every.dispatchEvent(new Event("input", { bubbles: true }));
    const side = [...document.querySelectorAll(${q(train('.train-settings select'))})].find((s) => [...s.options].some((o) => o.value === "512"));
    side.value = "512"; side.dispatchEvent(new Event("change", { bubbles: true }));
    return steps.value === "4";
  })()`)
  await clickText(train('button'), 'Start a new run')
  if (TRAINS_FOR_REAL) {
    check('a run trains to its last step', await waitFor(`(() => { const r = document.querySelector(${q(train('.run-card'))}); return !!r && r.textContent.includes("finished") && r.textContent.includes("step 4 / 4") })()`, 120000))
    await clickText(train('.run-card button'), 'Show loss')
    check('its loss can be shown as a chart', await waitFor(`!!document.querySelector(${q(train('.loss-plot svg path.avg'))})`))
    await clickText(train('.run-card button'), 'Compare')
    check('checkpoints are compared with the base model', await waitFor(`(() => { const g = document.querySelector(${q(train('.compare-grid'))}); return !!g && g.querySelectorAll("img").length >= 3 })()`, 90000))
    check('comparison images land in a gallery folder', (await js(`window.api.image.folders()`)).some((f) => f.name === 'Training: Example coupes' && f.count >= 3))
    await shot('13-train-run')
    await clickText(train('.run-card button'), 'Use as LoRA')
    check('a checkpoint goes into the LoRA library', await waitFor(`document.querySelector(${q(train('.lora-library'))}).textContent.includes("Example coupes")`))
    const [lora] = await js(`window.api.training.loras()`)
    await clickText(inImage('.tabs button'), 'Image')
    check('the Image tab offers the LoRA', await waitFor(`[...document.querySelectorAll(${q(inImage('.lora-picker option'))})].some((o) => o.value === ${q(lora.id)})`))
    await js(`(() => { const s = document.querySelector(${q(inImage('.lora-picker select'))}); s.value = ${q(lora.id)}; s.dispatchEvent(new Event("change", { bubbles: true })); return true })()`)
    check('and shows it with a strength', await waitFor(`!!document.querySelector(${q(inImage('.lora-row input[type="range"]'))})`))
    const before = (await js(`window.api.image.gallery({})`)).total
    await click(inImage('.generate-button'))
    check('an image is made with the LoRA, and says so', await waitFor(`window.api.image.gallery({}).then((g) => g.total > ${before} && g.items[0].provenance?.loras?.[0]?.id === ${q(lora.id)})`, 30000))
    await js(`document.querySelector(${q(inImage('.lora-row .icon-button'))}).click()`)
    check('the LoRA can be taken off again', await waitFor(`!document.querySelector(${q(inImage('.lora-row'))})`))
  } else {
    check('without a trainable model, training fails with a clear message', await waitFor(`document.querySelector(${q(train('.train-main'))})?.textContent.includes("cannot be trained")`, 20000))
  }

  console.log('collecting photos')
  await clickText(inImage('.tabs button'), 'Train')
  await clickText(train('.folder-item'), 'Collect photos from the web')
  check('the collect panel says what is sent where', await waitFor(`(() => { const t = document.querySelector(${q(train('.collect-notice'))})?.textContent ?? ""; return t.includes("commons.wikimedia.org") && t.includes("CC BY") })()`))
  await setValue(train('.collect-panel input[aria-label="Project name"]'), 'Collected coupes')
  await setValue(train('.collect-panel input[aria-label="What the photos show"]'), 'coupe')
  check('the columns are the subject\'s folder levels', await waitFor(`(() => { const h = document.querySelector(${q(train('.collect-row.head'))})?.textContent ?? ""; return h.includes("Name") && h.includes("Variant (optional)") && h.includes("Search words") && !h.includes("Make") })()`))
  for (const [key, value] of [['Name', 'Example Motors Coupe GT'], ['Variant', 'Mk2'], ['category', 'Example Coupe']]) {
    await setValue(train(`.collect-row input[aria-label="Row 1 ${key}"]`), value)
  }
  await setValue(train('.collect-row input[aria-label="Row 1 photos"]'), 3)
  await clickText(train('.collect-panel button'), 'Choose a folder and collect')
  check('openly licensed photos are collected', await waitFor(`document.querySelector(${q(train('.collect-progress'))})?.textContent.includes("3 photos")`, 30000))
  check('and a project is ready', await waitFor(`!!document.querySelector(${q(train('.collect-progress button.primary'))})`, 15000))
  const credits = (() => { try { return readFileSync(join(collectedPhotos, 'CREDITS.md'), 'utf8') } catch { return '' } })()
  check('credits are written next to the photos', credits.includes('Photographer 101') && credits.includes('CC BY 4.0'))
  check('share-alike and personality-rights photos are not taken', !credits.includes('coupe-sa') && !credits.includes('coupe-people'))
  await shot('14-collect')
  await clickText(train('.collect-progress button'), 'Open the project')
  check('the collected project opens with its subject and captions', await waitFor(`document.querySelectorAll(${q(train('.caption-card'))}).length === 3 && ${cardsText}.every((t) => t === "Example Motors Coupe GT Mk2, coupe")`, 20000))

  console.log('merging galleries')
  const merge = (sel) => inImage(`.merge ${sel}`)
  const mergeText = () => text(inImage('.merge'))
  // Hidden folders hidden, as they are when the studio starts (the steps above showed them).
  await js(`window.api.image.setShowHiddenFolders(false)`)
  const totalBefore = (await js(`window.api.image.gallery({})`)).total
  await clickText(inImage('.tabs button'), 'Image')
  await clickText(inImage('.tabs button'), 'Settings')
  await clickText(merge('button'), 'Add a gallery')
  check('a gallery to merge can be chosen', await waitFor(`document.querySelectorAll(${q(merge('.merge-source'))}).length === 1`))
  // The second pick is the folder holding an outputs folder: its outputs folder is taken.
  await clickText(merge('button'), 'Add a gallery')
  check('and another, by the folder that holds it', await waitFor(`document.querySelectorAll(${q(merge('.merge-source'))}).length === 2`))
  check('each says what it holds', await waitFor(`document.querySelector(${q(merge('.merge-source'))})?.textContent.includes("3 images")`))
  check('names that differ only in case are a clash, shown before anything is copied', await waitFor(`!!document.querySelector(${q(merge('.merge-conflicts'))})?.textContent.includes("“Beach”")`))
  check('and the merge waits until it is resolved', await js(`[...document.querySelectorAll(${q(merge('button.primary'))})].every((b) => b.disabled)`))
  check('a copy of an item already coming is skipped, and said so', (await mergeText()).includes('1 item is already here'))
  check('a hidden folder of another gallery is not named', !(await mergeText()).includes('Private'))
  await clickText(merge('button'), 'Rename them for me')
  check('names can be suggested for every clash', await waitFor(`!document.querySelector(${q(merge('.merge-conflicts'))})`))
  check('so it can merge', await waitFor(`[...document.querySelectorAll(${q(merge('button.primary'))})].some((b) => !b.disabled && b.textContent === "Merge 4 items")`))
  check('a merge can be stopped, which adds nothing', (await js(`window.api.image.cancelMerge().then(() => true)`)) && (await js(`window.api.image.gallery({})`)).total === totalBefore)
  await clickText(merge('button.primary'), 'Merge')
  check('merging copies them in', await waitFor(`document.querySelector(${q(merge('.merge-result'))})?.textContent.includes("Merged 4 items")`, 20000))
  check('every item comes, but the one already there', await waitFor(`window.api.image.gallery({}).then((p) => p.total === ${totalBefore + 3})`))
  const merged = await js(`window.api.image.folders()`)
  check('folders come with their names, the clash renamed', merged.some((f) => f.name === 'Beach') && merged.some((f) => f.name === 'beach (pc2)'))
  check('favourites come as favourites', (await js(`window.api.image.gallery({ favorites: true })`)).items.some((i) => i.prompt === 'a merged sandcastle'))
  check('with their prompts and settings', (await js(`window.api.image.gallery({ q: "merged wave" })`)).items[0]?.seed === 201)
  check('a hidden folder stays hidden', !merged.some((f) => f.name === 'Private'))
  await js(`window.api.image.setShowHiddenFolders(true)`)
  const privateFolder = (await js(`window.api.image.folders()`)).find((f) => f.name === 'Private')
  check('and locked, with the PIN it had', !!privateFolder?.locked && !!(await js(`window.api.image.unlockFolder(${q(privateFolder?.id ?? '')}, "135790").then((f) => f.unlocked)`)))
  await js(`window.api.image.lockFolder(${q(privateFolder?.id ?? '')}).then(() => window.api.image.setShowHiddenFolders(false))`)

  console.log('switching sides keeps state')
  await clickText('.rail-button', 'Chat')
  check('chat side back', await waitFor(`!document.querySelector(".studio-pane:not([hidden]) .image-studio")`))
  await clickText('.rail-button', 'Image')
  await clickText(inImage('.tabs button'), 'Image')
  // Two from the image section, less the favorite, which went into the locked Vault and so left
  // this window's strip; plus the one made with a LoRA when the Train tab trained for real.
  check('session images survive the switch', (await count(inImage('.film'))) === (TRAINS_FOR_REAL ? 2 : 1), await js(`[...document.querySelectorAll(${q(inImage('.film'))})].map((f) => f.title).join(" | ")`))
  check('prompt survives the switch', await js(`document.querySelector(${q(inImage('textarea.prompt'))}).value.includes("lighthouse")`))

  check('no console errors', consoleErrors.length === 0, consoleErrors.join(' | '))

  // The choice is stored for next time. (The restart below is a hard kill, which
  // can beat the browser's write to disk, so the stored value is checked here.)
  await clickText('.rail-button', 'Chat')
  await click('.sidebar-toggle')
  check('hiding the sidebar is remembered', await js(`localStorage.getItem("studio.chatSidebar") === "hidden"`))
  await ctrlB()
  check('and so is showing it', await js(`localStorage.getItem("studio.chatSidebar") === "shown"`))

  console.log('closing the window')
  await clickText('.rail-button', 'Settings')
  check('studio settings open', await waitFor(`!!document.querySelector(".app-settings")`))
  check('closing the window quits by default', await js(`document.querySelector('.app-settings input[value=quit]').checked`))
  check('the per-model choices wait until background is chosen', await js(`[...document.querySelectorAll(".keep-models .keep-row button")].every((b) => b.disabled)`))
  const privacyRow = (label) => `[...document.querySelectorAll(".app-settings .keep-row")].find((r) => r.textContent.startsWith(${q(label)}))`
  check('open folders stay open on minimizing, by default', await js(`(${privacyRow('When the window is minimized')}).querySelector("button.active").textContent === "Leave them open"`))
  check('notices are on by default and wait until background is chosen', await js(`["A chat reply arrives", "An image or video is finished"].every((label) => { const r = [...document.querySelectorAll(".app-settings .keep-row")].find((x) => x.textContent.startsWith(label)); return r && r.querySelector("button.active").textContent === "Notify me" && [...r.querySelectorAll("button")].every((b) => b.disabled) })`))
  check('and lock again on closing to the background, a choice that waits until background is chosen', await js(`(() => { const r = ${privacyRow('When the window closes to the background')}; return r.querySelector("button.active").textContent === "Lock them again" && [...r.querySelectorAll("button")].every((b) => b.disabled) })()`))
  await click('.app-settings input[value=background]')
  check('background can be chosen', await waitFor(`window.api.getConfig().then((c) => c.background.closeBehavior === "background")`))
  check('every model stays on the GPU by default', await js(`window.api.getConfig().then((c) => Object.values(c.background.keepLoaded).every(Boolean))`))
  await js(`(${privacyRow('An image or video is finished')}).querySelectorAll("button")[1].click()`)
  check('a notice can be turned off, on its own', await waitFor(`window.api.getConfig().then((c) => c.notifications.mediaDone === false && c.notifications.chatReply === true)`))
  check('and shows as off', await waitFor(`(${privacyRow('An image or video is finished')}).querySelector("button.active").textContent === "Stay quiet"`))
  await js(`(${privacyRow('An image or video is finished')}).querySelectorAll("button")[0].click()`)
  check('and on again', await waitFor(`window.api.getConfig().then((c) => c.notifications.mediaDone === true)`))
  check('a malformed notice setting is refused', await js(`window.api.setNotifications({ chatReply: "yes", mediaDone: true }).then(() => false, () => true)`))
  await shot('10-app-settings')

  // Where things are kept: moving the gallery changes where new media goes and
  // leaves the old folder alone (decisions S18). The chats folder is only read
  // here: moving it would need a restart, which the test does its own way.
  const folderRowOf = (label) => `[...document.querySelectorAll(".data-folder")].find((r) => r.textContent.startsWith(${q(label)}))`
  const galleryPath = () => js(`(${folderRowOf('Gallery')}).querySelector(".data-folder-path").textContent`)
  const defaultGallery = join(profile, 'image', 'outputs')
  const beforeMove = (await js(`window.api.image.gallery({})`)).items.length
  check('both folders are listed where they started', await js(`(${folderRowOf('Chats and settings')}).querySelector(".data-folder-path").textContent === ${q(join(profile, 'userData'))}`) && (await galleryPath()) === defaultGallery)
  check('and neither offers to go back to a default it is already on', await js(`[...document.querySelectorAll(".data-folder-actions")].every((a) => a.querySelectorAll("button")[1].disabled)`))
  check('the gallery has something in it to leave behind', beforeMove > 0 && pngsUnder(defaultGallery).length > 0)

  await js(`(${folderRowOf('Gallery')}).querySelectorAll(".data-folder-actions button")[0].click()`)
  check('the gallery folder can be moved', await waitFor(`(${folderRowOf('Gallery')}).querySelector(".data-folder-path").textContent === ${q(movedGallery)}`))
  check('which needs no restart, unlike the chats folder', await js(`!document.querySelector(".data-folder-restart")`))
  check('the engine comes back on it', await waitFor(`window.api.image.engineState().then((s) => s.state === "running")`, 90000))
  check('the new folder starts empty', await waitFor(`window.api.image.gallery({}).then((p) => p.items.length === 0)`))
  check('and nothing was taken out of the old one', pngsUnder(defaultGallery).length > 0)

  await js(`(${folderRowOf('Gallery')}).querySelectorAll(".data-folder-actions button")[1].click()`)
  check('the default can be chosen again', await waitFor(`(${folderRowOf('Gallery')}).querySelector(".data-folder-path").textContent === ${q(defaultGallery)}`))
  check('and everything in it is there as it was', await waitFor(`window.api.image.engineState().then((s) => s.state === "running")`, 90000) && await waitFor(`window.api.image.gallery({}).then((p) => p.items.length === ${beforeMove})`))

  // Keyboard shortcuts: every one listed; a changeable one takes the next keys pressed.
  const kbdOf = (label) => js(`[...document.querySelectorAll(".shortcut-row")].find((r) => r.textContent.startsWith(${q(label)}))?.querySelector("kbd")?.textContent`)
  // By row, not by an aria-label selector: click() puts its selector in a string unescaped.
  const changeKeys = (label) => js(`[...document.querySelectorAll(".shortcut-row")].find((r) => r.textContent.startsWith(${q(label)})).querySelector("button").click()`)
  const pressKeys = (key, code, mods) => js(`window.dispatchEvent(new KeyboardEvent("keydown", { key: ${q(key)}, code: ${q(code)}, ${mods}, bubbles: true }))`)
  check('settings list every shortcut, the fixed ones too', (await kbdOf('Lock every open folder')) === 'Ctrl+Shift+L' && (await kbdOf('Generate')) === 'Ctrl+Enter' && (await count('.shortcut-row.fixed')) >= 4)
  await changeKeys('Show or hide the sidebar')
  check('changing one waits for the keys', await waitFor(`!!document.querySelector(".shortcut-row kbd.recording")`))
  await pressKeys('s', 'KeyS', 'ctrlKey: true, altKey: true')
  check('and takes them', await waitFor(`window.api.getConfig().then((c) => c.shortcuts.toggleSidebar === "Ctrl+Alt+S")`))
  check('showing the new keys', await waitFor(`[...document.querySelectorAll(".shortcut-row")].find((r) => r.textContent.startsWith("Show or hide the sidebar"))?.querySelector("kbd")?.textContent === "Ctrl+Alt+S"`))
  await changeKeys('Lock every open folder')
  await pressKeys('s', 'KeyS', 'ctrlKey: true, altKey: true')
  check('keys another shortcut has are refused, saying which', await waitFor(`document.querySelector(".shortcut-problem")?.textContent.includes("Show or hide the sidebar")`))
  await pressKeys('c', 'KeyC', 'ctrlKey: true')
  check('so are editing keys', await waitFor(`document.querySelector(".shortcut-problem")?.textContent.includes("editing text")`))
  await pressKeys('h', 'KeyH', 'shiftKey: true')
  check('and keys that would get in the way of typing', await waitFor(`document.querySelector(".shortcut-problem")?.textContent.includes("Ctrl or Alt")`))
  await pressKeys('Escape', 'Escape', 'ctrlKey: false')
  check('Esc stops changing it, keeping the old keys', (await waitFor(`!document.querySelector(".shortcut-row kbd.recording")`)) && (await js(`window.api.getConfig().then((c) => c.shortcuts.lockFolders === "Ctrl+Shift+L")`)))
  await clickText('.rail-button', 'Chat')
  const sidebarState = () => js(`localStorage.getItem("studio.chatSidebar")`)
  const before = await sidebarState()
  await pressKeys('s', 'KeyS', 'ctrlKey: true, altKey: true')
  check('the new keys work at once', await waitFor(`localStorage.getItem("studio.chatSidebar") !== ${q(before)}`))
  await ctrlB()
  check('and the old ones no longer do', (await sleep(300), (await sidebarState()) !== before))
  await pressKeys('s', 'KeyS', 'ctrlKey: true, altKey: true')
  await clickText('.rail-button', 'Settings')
  await clickText('.shortcuts button', 'Put every shortcut back')
  check('every shortcut can go back to its default', await waitFor(`window.api.getConfig().then((c) => c.shortcuts.toggleSidebar === "Ctrl+B")`))

  // Dark by default; light at the click of a button, everywhere at once.
  const bgColour = () => js(`getComputedStyle(document.documentElement).getPropertyValue("--bg").trim()`)
  check('the studio is dark by default', (await js(`window.api.getConfig().then((c) => c.theme)`)) === 'dark' && (await bgColour()) === '#0b0b0c')
  await clickText('.app-settings [aria-label="Theme"] button', 'Light')
  check('it can be made light', await waitFor(`getComputedStyle(document.documentElement).getPropertyValue("--bg").trim() === "#f3f3f5"`))
  check('and the choice is kept', await js(`window.api.getConfig().then((c) => c.theme === "light")`))
  await shot('20-light-settings')
  await clickText('.rail-button', 'Chat')
  await shot('21-light-chat')
  await clickText('.rail-button', 'Image')
  await clickText(inImage('.tabs button'), 'Image')
  await shot('22-light-image')
  await clickText(inImage('.tabs button'), 'Gallery')
  await sleep(600)
  await shot('23-light-gallery')
  await clickText(inImage('.tabs button'), 'Settings')
  await shot('24-light-image-settings')
  await clickText('.rail-button', 'Settings')
  await clickText('.app-settings [aria-label="Theme"] button', 'Dark')
  check('and dark again', await waitFor(`getComputedStyle(document.documentElement).getPropertyValue("--bg").trim() === "#0b0b0c"`))
  await js(`window.api.image.loadModel()`)
  check('the image model is loaded', await waitFor(`window.api.image.status().then((s) => s.model.state === "ready")`))

  check('the close button can be pressed', closeLikeTheCloseButton())
  check('closing the window hides it', await waitFor(`document.visibilityState === "hidden"`))
  check('and the studio and its engine keep running', await portInUse(ENGINE_PORT))
  await sleep(4500) // longer than one background check
  check('a model set to stay loaded stays on the GPU', await js(`window.api.image.status().then((s) => s.model.state === "ready")`))
  check('a second launch exits and brings the window back', (await launchSecond()) && (await waitFor(`document.visibilityState === "visible"`)))
  check('hidden folders are hidden again after the window closed', (await js(`window.api.image.showHiddenFolders()`)) === false && !(await js(`window.api.image.folders()`)).some((f) => f.name === 'Vault'))
  await js(`window.api.image.setShowHiddenFolders(true)`)
  check('and unlocked folders locked again', (await js(`window.api.image.folders()`)).some((f) => f.name === 'Vault' && f.locked && !f.unlocked))
  await js(`window.api.image.setShowHiddenFolders(false)`)

  await js(`[...document.querySelectorAll(".keep-row")].find((r) => r.textContent.startsWith("Image model")).querySelectorAll("button")[1].click()`)
  check('a model can be set to unload', await waitFor(`window.api.getConfig().then((c) => c.background.keepLoaded.image === false)`))
  closeLikeTheCloseButton()
  check('then closing the window unloads it once idle', await waitFor(`window.api.image.status().then((s) => s.model.state === "unloaded")`, 10000))
  check('while the engine stays up', await portInUse(ENGINE_PORT))
  await launchSecond()
  check('and the window comes back again', await waitFor(`document.visibilityState === "visible"`))

  // Folder locking without quitting: kept open on closing to the background, or locked on minimizing.
  const vaultId = (await js(`window.api.image.setShowHiddenFolders(true).then(() => window.api.image.folders())`)).find((f) => f.name === 'Vault').id
  const vaultOpen = `window.api.image.folders().then((f) => f.some((x) => x.name === "Vault" && x.unlocked))`
  await js(`window.api.image.unlockFolder(${q(vaultId)}, "246810")`)
  await js(`(${privacyRow('When the window closes to the background')}).querySelectorAll("button")[1].click()`)
  check('open folders can be left open in the background', await waitFor(`window.api.getConfig().then((c) => c.folderPrivacy.lockOnBackground === false)`))
  closeLikeTheCloseButton()
  await waitFor(`document.visibilityState === "hidden"`)
  await sleep(1000)
  check('and then stay open after the window closed', await js(vaultOpen))
  await launchSecond()
  await waitFor(`document.visibilityState === "visible"`)
  await js(`(${privacyRow('When the window is minimized')}).querySelectorAll("button")[0].click()`)
  check('minimizing can lock them', await waitFor(`window.api.getConfig().then((c) => c.folderPrivacy.lockOnMinimize === true)`))
  check('the window can be minimized', minimizeLikeTheButton())
  // The same reset as quitting: shown hidden folders are hidden again, and open ones locked.
  check('and minimizing hides shown hidden folders again', await waitFor(`window.api.image.showHiddenFolders().then((shown) => !shown)`, 8000))
  check('and locks open folders again', (await js(`window.api.image.setShowHiddenFolders(true).then(() => window.api.image.folders())`)).some((f) => f.name === 'Vault' && f.locked && !f.unlocked))
  await launchSecond()
  await waitFor(`document.visibilityState === "visible"`)
  // Switching to another window can lock them too.
  await js(`window.api.image.unlockFolder(${q(vaultId)}, "246810")`)
  await js(`(${privacyRow('When another window is chosen')}).querySelectorAll("button")[0].click()`)
  check('switching to another window can lock them', await waitFor(`window.api.getConfig().then((c) => c.folderPrivacy.lockOnBlur === true)`))
  check('the window can lose focus', deactivateLikeAnotherWindow())
  check('and then open folders lock again', await waitFor(`window.api.image.showHiddenFolders().then((shown) => !shown)`, 8000))
  await launchSecond()
  await waitFor(`document.visibilityState === "visible"`)
  await js(`window.api.setFolderPrivacy({ lockOnBlur: false, lockOnMinimize: false, lockOnBackground: true }).then(() => window.api.image.setShowHiddenFolders(false))`)

  await click('.app-settings input[value=quit]')
  check('quit can be chosen again', await waitFor(`window.api.getConfig().then((c) => c.background.closeBehavior === "quit")`))
  await clickText('.rail-button', 'Chat')

  console.log('shutdown')
  ws.close()
  // Hard kill: the engine must notice its parent is gone (stdin closes) and exit by itself.
  if (process.platform === 'win32') execSync(`taskkill /PID ${app.pid} /T /F`, { stdio: 'ignore' })
  else app.kill('SIGKILL')
  app = null
  check('engine exits when the app is killed', await waitPortFree(ENGINE_PORT, 10000))

  console.log('restart')
  app = launch()
  await attach()
  check('the last folder is open again after a restart', await waitFor(`document.querySelector(".ws-current")?.textContent.includes("demo-project")`, 15000))
  check('its project rule is still there', await waitFor(`document.querySelector(".ws-foot")?.textContent.includes("1 project rule")`))

  console.log('moving the chats folder')
  // The chats folder is read once, at startup, so a new one is used from the next
  // start, and the folder left behind keeps its chats (decisions S18). The window's
  // own "Restart now" calls app.relaunch(); the test restarts the studio the way it
  // does everywhere else, so the process it then drives is one it owns.
  const chatsRow = `[...document.querySelectorAll(".data-folder")].find((r) => r.textContent.startsWith("Chats and settings"))`
  const chatsPath = () => js(`(${chatsRow}).querySelector(".data-folder-path").textContent`)
  await clickText('.rail-button', 'Settings')
  await waitFor(`!!document.querySelector(".app-settings")`)
  await js(`(${chatsRow}).querySelectorAll(".data-folder-actions button")[0].click()`)
  check('a new chats folder is taken, and asks for a restart', (await waitFor(`!!document.querySelector(".data-folder-restart")`)) && (await chatsPath()) === movedChats)
  await clickText('.rail-button', 'Chat')
  check('the studio stays on the folder it started with until then', await js(`document.querySelector(".ws-current")?.textContent.includes("demo-project")`))

  async function restartStudio() {
    ws.close()
    if (process.platform === 'win32') execSync(`taskkill /PID ${app.pid} /T /F`, { stdio: 'ignore' })
    else app.kill('SIGKILL')
    await waitPortFree(ENGINE_PORT, 10000)
    app = launch()
    await attach()
  }

  await restartStudio()
  check('the next start uses it', (await js(`window.api.locations.get().then((v) => v.userData.path)`)) === movedChats)
  check('and nothing is left over to restart for', await js(`window.api.locations.get().then((v) => v.restartNeeded === false)`))
  check('an empty folder means an empty studio: no workspace, no chats', await waitFor(`!document.querySelector(".ws-current") && document.querySelectorAll(".chat-item").length === 0`, 15000))
  check('while the folder left behind still holds its chats', readdirSync(join(profile, 'userData', 'workspaces')).length > 0)

  await clickText('.rail-button', 'Settings')
  await waitFor(`!!document.querySelector(".app-settings")`)
  await js(`(${chatsRow}).querySelectorAll(".data-folder-actions button")[1].click()`)
  check('the old folder can be chosen again', (await waitFor(`!!document.querySelector(".data-folder-restart")`)) && (await chatsPath()) === join(profile, 'userData'))
  await restartStudio()
  check('and has everything it had: the workspace, its chats and its rule', await waitFor(`document.querySelector(".ws-current")?.textContent.includes("demo-project")`, 15000) && (await waitFor(`document.querySelector(".ws-foot")?.textContent.includes("1 project rule")`)))
  check('the chats folder is back to the default', await js(`window.api.locations.get().then((v) => v.userData.isDefault === true)`))

  console.log('closing the window, quit mode')
  const exited = new Promise((done) => app.once('exit', () => done(true)))
  closeLikeTheCloseButton()
  check('closing the window quits the studio', await Promise.race([exited, sleep(15000).then(() => false)]))
  app = null
  check('and stops the engine with it', await waitPortFree(ENGINE_PORT, 10000))
} catch (err) {
  failures.push(`crashed: ${err.message}`)
  console.error(err)
} finally {
  try {
    ws?.close()
  } catch {
    /* already closed */
  }
  if (app) app.kill()
  fakeSources.kill()
  await sleep(500)
  try {
    rmSync(profile, { recursive: true, force: true })
    rmSync(dirname(workspaceDir), { recursive: true, force: true })
  } catch {
    /* files may still be held briefly */
  }
}

console.log(`\n${passed} passed, ${failures.length} failed`)
process.exit(failures.length ? 1 : 0)
