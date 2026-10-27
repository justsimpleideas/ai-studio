// End-to-end test of the real Electron app, driven over the Chrome DevTools Protocol.
//
// The image engine runs for real (Python process, per-launch token, loopback API,
// studio-image: protocol, IPC) but with the test-only fake model from
// image-engine/tests/ui_harness.py, so no GPU or weights are needed. The app gets a
// throwaway profile folder, so real settings and galleries are never touched.
//
//   npm run build && npm run e2e [-- --shots <dir>]
//
// Needs a Python with Pillow on PATH (or STUDIO_PYTHON=/path/to/python).

import { spawn, spawnSync, execSync } from 'node:child_process'
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import net from 'node:net'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
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
// Where downloaded models would go, standing in for the folder picker (nothing is downloaded).
const modelsDir = join(dirname(workspaceDir), 'models')
const env = { ...process.env }
delete env.ELECTRON_RUN_AS_NODE
Object.assign(env, {
  STUDIO_USER_DATA: profile,
  STUDIO_IMAGE_ENGINE_MODULE: 'tests.ui_harness',
  STUDIO_PYTHON: process.env.STUDIO_PYTHON ?? 'python',
  STUDIO_PICK_FOLDER: workspaceDir,
  STUDIO_SAVE_AS: exportPath,
  // The start-picture picker answers with the exported PNG, which exists by the time it is asked.
  STUDIO_PICK_START_IMAGE: exportPath,
  STUDIO_PICK_MODELS_FOLDER: modelsDir,
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

function launch() {
  return spawn(electronExe, ['.', `--remote-debugging-port=${DEBUG_PORT}`], { cwd: ROOT, env, stdio: 'ignore' })
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

  console.log('engine install and model downloads')
  await clickText(inImage('.setup button'), 'Install the engine')
  check('installing first says what it downloads and from where', await waitFor(`document.querySelector(${q(inImage('.setup-box'))})?.textContent.includes("official PyTorch and PyPI indexes")`))
  await clickText(inImage('.setup-box button'), 'Cancel')
  check('and can be backed out of', await waitFor(`!document.querySelector(${q(inImage('.setup .setup-box'))})`))
  check('every supported model is offered for download', (await count(inImage('.download-row'))) === 3)
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

  await js(`${folderRow('Trips')}.querySelector('[aria-label="Rename Trips"]').click()`)
  await nameFolder('Holidays')
  check('a folder can be renamed', await waitFor(`!!(${folderRow('Holidays')}) && !(${folderRow('Trips')})`))
  await click(inImage('.folder-head button'))
  await nameFolder('holidays')
  check('two folders cannot share a name', await waitFor(`document.querySelector(${q(inImage('.banner'))})?.textContent.includes("already a folder")`))
  await js(`document.querySelector(${q(inImage('.folder-name-input input'))})?.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }))`)
  await shot('11-gallery-folders')

  check('dragging onto All takes an item out of its folder', await dragTile(favorite.id, `.studio-pane:not([hidden]) .image-studio .folder-item`))
  check('and its folder counts one fewer', await waitFor(`(${folderRow('Holidays')})?.querySelector(".folder-count")?.textContent === "1"`))
  await js(`${folderRow('Holidays')}.querySelector('[aria-label="Delete Holidays"]').click()`)
  check('deleting asks first, and says the items stay', await waitFor(`document.querySelector(${q(inImage('.folder-confirm'))})?.textContent.includes("stay in All")`))
  await clickText(inImage('.folder-confirm button'), 'Delete folder')
  check('the folder goes', await waitFor(`!(${folderRow('Holidays')})`))
  check('and nothing that was in it is lost', (await js(`window.api.image.gallery({})`)).total === everything.length)
  check('it is back in no folder', (await js(`window.api.image.gallery({})`)).items.find((i) => i.id === video.id)?.folder_id === null)
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

  console.log('switching sides keeps state')
  await clickText('.rail-button', 'Chat')
  check('chat side back', await waitFor(`!document.querySelector(".studio-pane:not([hidden]) .image-studio")`))
  await clickText('.rail-button', 'Image')
  await clickText(inImage('.tabs button'), 'Image')
  check('session images survive the switch', (await count(inImage('.film'))) === 2)
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
  check('the per-model choices wait until background is chosen', await js(`[...document.querySelectorAll(".keep-row button")].every((b) => b.disabled)`))
  await click('.app-settings input[value=background]')
  check('background can be chosen', await waitFor(`window.api.getConfig().then((c) => c.background.closeBehavior === "background")`))
  check('every model stays on the GPU by default', await js(`window.api.getConfig().then((c) => Object.values(c.background.keepLoaded).every(Boolean))`))
  await shot('10-app-settings')
  await js(`window.api.image.loadModel()`)
  check('the image model is loaded', await waitFor(`window.api.image.status().then((s) => s.model.state === "ready")`))

  check('the close button can be pressed', closeLikeTheCloseButton())
  check('closing the window hides it', await waitFor(`document.visibilityState === "hidden"`))
  check('and the studio and its engine keep running', await portInUse(ENGINE_PORT))
  await sleep(4500) // longer than one background check
  check('a model set to stay loaded stays on the GPU', await js(`window.api.image.status().then((s) => s.model.state === "ready")`))
  check('a second launch exits and brings the window back', (await launchSecond()) && (await waitFor(`document.visibilityState === "visible"`)))

  await js(`[...document.querySelectorAll(".keep-row")].find((r) => r.textContent.startsWith("Image model")).querySelectorAll("button")[1].click()`)
  check('a model can be set to unload', await waitFor(`window.api.getConfig().then((c) => c.background.keepLoaded.image === false)`))
  closeLikeTheCloseButton()
  check('then closing the window unloads it once idle', await waitFor(`window.api.image.status().then((s) => s.model.state === "unloaded")`, 10000))
  check('while the engine stays up', await portInUse(ENGINE_PORT))
  await launchSecond()
  check('and the window comes back again', await waitFor(`document.visibilityState === "visible"`))

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
