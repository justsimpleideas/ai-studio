import path from 'node:path'
import fs from 'node:fs'
import { randomUUID } from 'node:crypto'
import { app, BrowserWindow, dialog, ipcMain, Menu, nativeImage, protocol, safeStorage, session, shell, Tray } from 'electron'
import {
  MODEL_SLOTS,
  composeSystemSections,
  isCloseBehavior,
  isKvCacheType,
  parseBackground,
  type BackgroundPreferences,
  type GlobalRule,
  type SystemSection
} from '@shared/config.js'
import { withDefaultProtections, type ClassMode, type ProtectedPath, type WorkspaceRoot } from '@shared/permissions.js'
import type { ChatTarget } from '@shared/engine-protocol.js'
import { chatTitle, type Workspace, type WorkspaceSnapshot, type WorkspaceState } from '@shared/workspace.js'
import { isWebTool, type FileToolCall, type RiskClass, type WebToolCall } from '@shared/tools.js'
import { isInternetLevel, toolsFor } from '@shared/network.js'
import type { ApprovalRequest, ApprovalResponse, AuditEntry, ToolOutcome } from '@shared/broker-types.js'
import type { EngineToMain, LoadRequest } from '@shared/engine-protocol.js'
import { validateToolCall } from '@shared/validate.js'
import { IMAGE_ENGINE_PORT, IMAGE_SCHEME } from '@shared/image.js'
import { BackgroundUnloader, type UnloadTarget } from './background.js'
import { Broker } from './broker/index.js'
import { canonicalize, comparablePath } from './broker/paths.js'
import { EngineSupervisor } from './engine/supervisor.js'
import { isGgufFile } from './gguf.js'
import { GpuArbiter } from './gpu.js'
import { MODEL_CATALOG } from '@shared/setup.js'
import { ModelDownloads } from './image/downloads.js'
import { ModelImporter, type EngineFolders } from './image/importer.js'
import { ModelInspector } from './image/inspect.js'
import { ImageEngine } from './image/engine.js'
import { registerImageHandlers, serveImage } from './image/handlers.js'
import { PhotoCollector } from './image/collect.js'
import { registerTrainingHandlers } from './image/training-handlers.js'
import { EngineSetup } from './image/setup.js'
import { Journal } from './tools/fs-tools.js'
import { ConfigStore } from './store.js'
import { ChatRecorder } from './chat-recorder.js'
import { isSafeId, WorkspaceStore } from './workspaces.js'
import { WebGate } from './net/gate.js'
import { NetworkLog } from './net/log.js'
import { SecretStore } from './secrets.js'

/**
 * Wiring.
 *
 * This file holds no policy and no file access of its own. It owns the window,
 * it owns the engine process, and it carries messages between the two through
 * the broker. Anything that decides something lives somewhere else, and is
 * tested there.
 */

// A separate profile folder, used by the end-to-end test so it never touches real
// settings, chats or galleries. Must be set before anything reads the path.
const userDataOverride = process.env['STUDIO_USER_DATA']
if (userDataOverride) app.setPath('userData', userDataOverride)

// One studio per profile. A second launch, say while this one runs in the
// background, brings this window back instead of starting another copy that
// would fight it for the image engine's port and the GPU. (The lock is per
// userData folder, so the end-to-end test's throwaway profile is separate.)
if (!app.requestSingleInstanceLock()) app.exit(0)

// Must happen before the app is ready: the window shows generated images through
// this private scheme, served below from the image engine (decisions S3).
protocol.registerSchemesAsPrivileged([
  { scheme: IMAGE_SCHEME, privileges: { standard: true, secure: true, stream: true } }
])

const userDataRoot = path.join(app.getPath('userData'), 'userData')
const store = new ConfigStore(path.join(userDataRoot, 'config.json'))
const workspaces = new WorkspaceStore(path.join(userDataRoot, 'workspaces'))
const secrets = new SecretStore(path.join(userDataRoot, 'secrets.json'), safeStorage)

let window: BrowserWindow | null = null

/** Approval prompts waiting on a click. */
const pendingApprovals = new Map<string, (response: ApprovalResponse) => void>()

function toRenderer(channel: string, payload: unknown): void {
  window?.webContents.send(channel, payload)
}

function askUser(request: ApprovalRequest): Promise<ApprovalResponse> {
  return new Promise<ApprovalResponse>((resolve) => {
    if (!window) {
      // No window means nobody can answer, and an unanswerable prompt must
      // not become a silent yes.
      resolve({ allow: false, grant: null })
      return
    }
    pendingApprovals.set(request.id, resolve)
    toRenderer('approval:request', request)
    // In the background the prompt waits unseen; say so from the tray.
    if (!window.isVisible()) trayMessage('The agent is waiting for your approval. Open the studio to answer.')
  })
}

// ------------------------------------------------------------ workspaces

/** The open folder, or null before one has been opened. */
let activeWorkspace: Workspace | null = null
/** The open chat of the active workspace. */
let recorder: ChatRecorder | null = null
/**
 * Which chat the engine currently holds. Null after a load, an unload or a
 * crash, when it holds none; the next prompt then carries the chat's history.
 */
let engineChatId: string | null = null

/**
 * One broker and one journal per workspace: session grants, "always" grants and
 * undo all stay with the folder they were given for.
 */
const brokers = new Map<string, Broker>()
const journals = new Map<string, Journal>()

function workspaceRoots(ws: Workspace): WorkspaceRoot[] {
  return [
    { id: 'working-folder', path: ws.path, label: ws.name, enabled: true, addedAt: ws.createdAt },
    ...ws.extraRoots
  ]
}

function journalFor(id: string): Journal {
  let journal = journals.get(id)
  if (!journal) {
    // Kept on disk, so a reply can still be reverted after a restart.
    journal = new Journal(workspaces.backupsDir(id), workspaces.journalFile(id), () => {
      if (activeWorkspace?.id === id) toRenderer('journal:changed', null)
    })
    journals.set(id, journal)
  }
  return journal
}

function brokerFor(ws: Workspace): Broker {
  const id = ws.id
  let broker = brokers.get(id)
  if (!broker) {
    broker = new Broker({
      // Read fresh on every call, so a folder switched off a moment ago is
      // already out of reach.
      getSettings: () => {
        const current = workspaces.get(id)
        return {
          ...store.get().permissions,
          roots: current ? workspaceRoots(current) : [],
          rules: current?.policyRules ?? [],
          protectedPaths: current?.protectedPaths ?? []
        }
      },
      saveRule: async (rule) => {
        await workspaces.update(id, (w) => {
          w.policyRules.push(rule)
        })
        if (activeWorkspace?.id === id) activeWorkspace = workspaces.get(id)
        pushState()
      },
      prompt: askUser,
      trash: (target) => shell.trashItem(target),
      audit: (entry: AuditEntry) => toRenderer('audit:entry', entry),
      journal: journalFor(id),
      workingFolder: ws.path
    })
    brokers.set(id, broker)
  }
  return broker
}

function systemFor(ws: Workspace): SystemSection[] {
  return composeSystemSections(store.get().rules, {
    workingFolder: ws.path,
    extraFolders: ws.extraRoots.filter((r) => r.enabled).map((r) => r.path),
    projectRules: ws.rules,
    readOnly: ws.protectedPaths.filter((p) => p.enabled && p.level === 'read-only').map((p) => p.pattern),
    internet: ws.internet
  })
}

// ------------------------------------------------------------ internet

const gates = new Map<string, WebGate>()
const networkLogs = new Map<string, NetworkLog>()

function networkLogFor(id: string): NetworkLog {
  let log = networkLogs.get(id)
  if (!log) {
    log = new NetworkLog(workspaces.networkLogFile(id), () => {
      if (activeWorkspace?.id === id) toRenderer('network:changed', null)
    })
    networkLogs.set(id, log)
  }
  return log
}

/** The one road from the agent to the internet, for one workspace. */
function gateFor(ws: Workspace): WebGate {
  const id = ws.id
  let gate = gates.get(id)
  if (!gate) {
    gate = new WebGate({
      // Read fresh on every request: switching a folder back to offline takes effect at once.
      level: () => workspaces.get(id)?.internet ?? 'offline',
      askFirst: () => store.get().permissions.modes.network !== 'auto',
      searchKey: () => secrets.get('searchKey'),
      prompt: askUser,
      log: networkLogFor(id),
      leaks: () => brokerFor(ws).contents,
      files: {
        checkNewFile: (raw) => brokerFor(ws).checkNewFile(raw),
        saveNewFile: (resolved, bytes, ctx) => brokerFor(ws).saveNewFile(resolved, bytes, { promptId: ctx.promptId })
      }
    })
    gates.set(id, gate)
  }
  return gate
}

/**
 * Nothing in the window ever needs the internet: it is served from disk, talks
 * to the main process over IPC and shows images through studio-image:. Any
 * other request from it (a stray link, a page script, a library phoning home)
 * is cancelled here, whatever the internet level. The agent's own requests do
 * not come through here at all; they go through the gate above.
 */
function isLocalRequest(url: string): boolean {
  if (/^(file|data|blob|devtools|chrome|chrome-extension|studio-image):/i.test(url)) return true
  try {
    const u = new URL(url)
    return (u.protocol === 'http:' || u.protocol === 'ws:') && ['localhost', '127.0.0.1', '[::1]'].includes(u.hostname)
  } catch {
    return false
  }
}

/** Which chat a message to the engine is for, with its history when the engine must switch to it. */
function chatTarget(rec: ChatRecorder): ChatTarget {
  const chatId = rec.chat.meta.id
  if (engineChatId === chatId) return { chatId }
  engineChatId = chatId
  return { chatId, history: rec.chat.history ?? [] }
}

function workspaceState(): WorkspaceState {
  return {
    workspaces: workspaces.list(),
    active: activeWorkspace,
    chats: activeWorkspace ? workspaces.listChats(activeWorkspace.id) : [],
    chatId: recorder?.chat.meta.id ?? null
  }
}

function snapshot(): WorkspaceSnapshot {
  return { state: workspaceState(), chat: recorder?.view() ?? null }
}

function pushState(): void {
  toRenderer('workspace:changed', workspaceState())
}

let saving: Promise<unknown> = Promise.resolve()

/** Write the open chat, if it has anything in it yet. Saves are queued, never overlapped. */
function saveChat(): void {
  const ws = activeWorkspace
  const rec = recorder
  if (!ws || !rec || !rec.worthSaving) return
  const copy = structuredClone(rec.chat)
  saving = saving
    .catch(() => undefined)
    .then(() => workspaces.saveChat(ws.id, copy))
    .then(() => pushState())
    .catch((err: Error) => toRenderer('studio:notice', `This chat could not be saved: ${err.message}`))
}

/**
 * Open `next` as the chat. What the agent has read belongs to the conversation
 * that read it, so the broker forgets it: in the new chat an existing file must
 * be read again before it can be changed.
 */
function setRecorder(next: ChatRecorder | null): void {
  recorder = next
  if (activeWorkspace) brokerFor(activeWorkspace).resetConversation()
}

function assertIdle(): void {
  if (chat.activePrompts.size > 0 || pendingApprovals.size > 0) {
    throw new Error('Wait for the current reply to finish, or stop it, first.')
  }
}

/** Make `ws` the open workspace, starting a new chat in it. */
async function activate(ws: Workspace): Promise<void> {
  activeWorkspace = await workspaces.open(ws.path)
  setRecorder(ChatRecorder.fresh())
  await store.update((c) => {
    c.lastWorkspaceId = ws.id
  })
  measureUsage()
}

/** Pick a folder: a native dialog, or for the end-to-end test a folder named up front. */
async function pickFolder(title: string, buttonLabel: string): Promise<string | null> {
  const preset = process.env['STUDIO_PICK_FOLDER']
  if (preset) return preset
  const result = await dialog.showOpenDialog({ title, properties: ['openDirectory'], buttonLabel })
  return result.canceled ? null : (result.filePaths[0] ?? null)
}

/**
 * What the chat side is doing, mirrored from the engine's messages so the GPU
 * arbiter can tell whether the chat model may be unloaded right now.
 */
const chat = {
  loaded: false,
  loading: false,
  activePrompts: new Set<string>(),
  unloadWaiters: [] as Array<() => void>
}

function chatSettled(): void {
  engineChatId = null
  chat.loaded = false
  chat.loading = false
  chat.activePrompts.clear()
  for (const resolve of chat.unloadWaiters.splice(0)) resolve()
}

function chatBusyReason(): string | null {
  if (chat.loading) return 'The chat model is loading.'
  return chat.activePrompts.size > 0 ? 'The chat model is answering.' : null
}

/** Unload the chat model; resolves once the engine says so (or after 30 s regardless). */
function unloadChat(): Promise<void> {
  return new Promise<void>((resolve) => {
    const timer = setTimeout(resolve, 30_000)
    chat.unloadWaiters.push(() => {
      clearTimeout(timer)
      resolve()
    })
    engine.send({ type: 'unload' })
  })
}

/**
 * The load in flight. Kept so that an 8-bit context which fails to build, or
 * takes the engine down with it (the library calls the option experimental),
 * is retried once at 16-bit instead of leaving the user with no model.
 */
let pendingLoad: LoadRequest | null = null

function sendLoad(request: LoadRequest): void {
  pendingLoad = request
  chat.loading = true
  engine.send({ type: 'load', request })
}

/** True when the failed load was 8-bit and has been sent again at 16-bit. */
function retryAt16Bit(): boolean {
  const failed = pendingLoad
  pendingLoad = null
  if (!failed || failed.kvCache !== 'q8_0') return false
  void store
    .update((c) => {
      c.model.kvCache = 'f16'
    })
    .then((config) => toRenderer('config:changed', config))
  toRenderer(
    'studio:notice',
    '8-bit context memory did not work with this model on this GPU. It was switched back to 16-bit and the model is loading again.'
  )
  sendLoad({ ...failed, kvCache: 'f16' })
  return true
}

const engine = new EngineSupervisor(
  (message) => void handleEngineMessage(message),
  (code) => {
    chatSettled()
    if (retryAt16Bit()) return
    toRenderer('engine:event', {
      type: 'error',
      id: null,
      message:
        code === 0
          ? 'The model process stopped.'
          : `The model process stopped unexpectedly (exit code ${code}). The model is no longer loaded.`,
      fatal: true
    } satisfies EngineToMain)
  }
)

/**
 * A tool call arriving from the model.
 *
 * Validated first, then brokered. The result goes back to the engine either way,
 * because a model that is told nothing will sit there having silently lost its
 * turn.
 */
async function handleEngineMessage(message: EngineToMain): Promise<void> {
  if (message.type === 'loaded') pendingLoad = null
  if (message.type === 'error' && message.fatal && retryAt16Bit()) return
  trackChat(message)
  record(message)
  if (message.type === 'history') return
  if (message.type !== 'toolCall') {
    toRenderer('engine:event', message)
    // What instructions and tool descriptions cost is worth seeing before the
    // first message, so a freshly loaded model is measured straight away.
    if (message.type === 'loaded') measureUsage()
    return
  }

  const validation = validateToolCall(message.call, randomUUID())
  if (!validation.ok) {
    engine.send({ type: 'toolResult', callId: message.callId, ok: false, payload: validation.error })
    const rejected: ToolOutcome = {
      ok: false,
      tool: (message.call.name || 'unknown') as ToolOutcome['tool'],
      resolved: null,
      error: validation.error,
      refused: true
    }
    recorder?.outcome(rejected)
    toRenderer('tool:outcome', rejected)
    return
  }

  // Folders cannot be switched while a reply is running, so the workspace open
  // now is the one the prompt was sent in. Web tools go to the internet gate,
  // file tools to the broker; neither can reach the other's capability.
  const call = validation.call
  const outcome: ToolOutcome = !activeWorkspace
    ? { ok: false, tool: call.name, resolved: null, error: 'No folder is open.', refused: true }
    : isWebTool(call.name)
      ? await gateFor(activeWorkspace).execute(call as WebToolCall, {
          promptId: message.promptId,
          chatId: recorder?.chat.meta.id ?? null
        })
      : await brokerFor(activeWorkspace).execute(call as FileToolCall, { promptId: message.promptId })
  recorder?.outcome(outcome)
  saveChat()
  toRenderer('tool:outcome', outcome)
  engine.send({
    type: 'toolResult',
    callId: message.callId,
    ok: outcome.ok,
    payload: outcome.ok
      ? outcome.result
      : outcome.hint
        ? `${outcome.error} ${outcome.hint}`
        : outcome.error
  })
}

/** Keep the open chat's saved copy in step with what the window is showing. */
function record(message: EngineToMain): void {
  const rec = recorder
  if (!rec) return
  switch (message.type) {
    case 'chunk':
      rec.chunk(message.id, message.text)
      break
    case 'complete':
      rec.complete(message.id, message.text)
      saveChat()
      break
    case 'aborted':
      rec.aborted(message.id)
      saveChat()
      break
    case 'error':
      if (message.id !== null && rec.owns(message.id)) {
        rec.error(message.id, message.message)
        saveChat()
      }
      break
    case 'history':
      if (message.chatId === rec.chat.meta.id) {
        rec.history(message.history)
        saveChat()
      }
      break
  }
}

function trackChat(message: EngineToMain): void {
  switch (message.type) {
    case 'loadProgress':
      chat.loading = true
      break
    case 'loaded':
      chat.loaded = true
      chat.loading = false
      // A freshly loaded model holds no conversation yet.
      engineChatId = null
      break
    case 'unloaded':
      chatSettled()
      break
    case 'complete':
    case 'aborted':
      chat.activePrompts.delete(message.id)
      break
    case 'error':
      if (message.fatal) chatSettled()
      else {
        chat.loading = false
        if (message.id) chat.activePrompts.delete(message.id)
        else chat.activePrompts.clear()
      }
      break
  }
}

const imageEngineDir = app.isPackaged
  ? path.join(process.resourcesPath, 'image-engine')
  : path.join(app.getAppPath(), 'image-engine')
const imageDataDir = path.join(app.getPath('userData'), 'image')
const imagePython = (): string => process.env['STUDIO_PYTHON'] ?? store.get().image.pythonPath ?? 'python'

const imageEngine = new ImageEngine({
  python: imagePython,
  engineDir: imageEngineDir,
  dataDir: imageDataDir,
  port: IMAGE_ENGINE_PORT,
  // Only for end-to-end tests: run the engine with a fake model (see image-engine/tests/ui_harness.py).
  module: process.env['STUDIO_IMAGE_ENGINE_MODULE'] ?? 'image_engine',
  appVersion: app.getVersion(),
  onEvent: (event) => toRenderer('image:event', event),
  onState: (state) => toRenderer('image:engineState', state)
})

/**
 * Send only the newest of a fast stream of states (pip output, download bytes),
 * at most every `ms`, and always the last one.
 */
function latestEvery<T>(channel: string, ms: number): (state: T) => void {
  let pending: T | null = null
  let timer: NodeJS.Timeout | null = null
  return (state) => {
    pending = state
    if (timer) return
    toRenderer(channel, state)
    pending = null
    timer = setTimeout(() => {
      timer = null
      if (pending !== null) toRenderer(channel, pending)
      pending = null
    }, ms)
  }
}

const engineSetup = new EngineSetup({
  venvDir: path.join(app.getPath('userData'), 'image-python'),
  requirements: path.join(imageEngineDir, 'requirements.txt'),
  constraints: path.join(imageEngineDir, 'constraints.txt'),
  // Only for tests and offline machines: install from a folder of wheels.
  source: { wheelhouse: process.env['STUDIO_SETUP_WHEELHOUSE'] ?? null },
  onState: latestEvery('image:setupState', 150)
})

const modelDownloads = new ModelDownloads({
  python: imagePython,
  script: path.join(imageEngineDir, 'scripts', 'fetch_model.py'),
  async reuseSource(model, dest) {
    // The two video models share their text encoder and decoder; link those instead of downloading them again.
    if (model.kind !== 'video') return null
    const configured = imageEngine.state.state === 'running'
      ? ((await imageEngine.request('GET', '/api/settings').catch(() => null)) as { video_model_dir?: string } | null)
      : null
    const siblings = MODEL_CATALOG.filter((m) => m.kind === 'video' && m.id !== model.id).map((m) =>
      path.join(path.dirname(dest), m.folder)
    )
    for (const candidate of [configured?.video_model_dir, ...siblings]) {
      if (!candidate || path.resolve(candidate) === path.resolve(dest)) continue
      if (fs.existsSync(path.join(candidate, 'manifest.json'))) return candidate
    }
    return null
  },
  async applyFolder(model, dest) {
    if (model.part) return // a part of another model: the import uses it, the engine is not pointed at it
    await imageEngine.start()
    const key = { image: 'model_dir', video: 'video_model_dir', caption: 'caption_model_dir' }[model.kind]
    await imageEngine.request('PUT', '/api/settings', { [key]: dest })
  },
  onState: latestEvery('image:downloadState', 250)
})

const modelImporter = new ModelImporter({
  python: imagePython,
  script: path.join(imageEngineDir, 'scripts', 'import_model.py'),
  modelsFolder: () => store.get().image.modelsFolder ?? path.join(imageDataDir, 'models'),
  engineFolders: async () =>
    imageEngine.state.state === 'running'
      ? ((await imageEngine.request('GET', '/api/settings')) as EngineFolders)
      : null,
  knownFolders: () => store.get().image.partFolders,
  async rememberFolders(folders) {
    // Newest first, and a bounded list: a folder deleted long ago only costs one failed read.
    await store.update((c) => {
      c.image.partFolders = [...new Set([...folders, ...c.image.partFolders])].slice(0, 32)
    })
  },
  downloads: modelDownloads,
  async selectFolder(dest, target) {
    await imageEngine.start()
    await imageEngine.request('PUT', '/api/settings', { [target === 'video' ? 'video_model_dir' : 'model_dir']: dest })
  },
  onState: latestEvery('image:importState', 250)
})

const modelInspector = new ModelInspector({
  python: imagePython,
  script: path.join(imageEngineDir, 'scripts', 'inspect_model.py'),
  onState: latestEvery('image:inspectState', 250)
})

const photoCollector = new PhotoCollector({
  python: imagePython,
  script: path.join(imageEngineDir, 'scripts', 'collect_photos.py'),
  workDir: path.join(imageDataDir, 'collect'),
  // Set only by the end-to-end test, which runs a local stand-in for both services.
  apis: {
    ...(process.env['STUDIO_COLLECT_COMMONS_API'] ? { commons: process.env['STUDIO_COLLECT_COMMONS_API'] } : {}),
    ...(process.env['STUDIO_COLLECT_OPENVERSE_API'] ? { openverse: process.env['STUDIO_COLLECT_OPENVERSE_API'] } : {})
  },
  async makeProject(name, folder, subject) {
    await imageEngine.start()
    const { items } = (await imageEngine.request('GET', '/api/training/projects')) as { items: { id: string; photo_root: string }[] }
    const same = items.find((p) => path.resolve(p.photo_root).toLowerCase() === path.resolve(folder).toLowerCase())
    // An existing project keeps its own subject: the user may have refined it since.
    const id = same?.id ?? ((await imageEngine.request('POST', '/api/training/projects', { name, photo_root: folder, subject })) as { id: string }).id
    // A project that is busy is scanned by the user later; the photos are there either way.
    await imageEngine.request('POST', `/api/training/projects/${id}/scan`, {}).catch(() => undefined)
    return id
  },
  onState: latestEvery('training:collectState', 250)
})

const gpu = new GpuArbiter(
  {
    chat: {
      label: 'chat model',
      isLoaded: () => chat.loaded || chat.loading,
      busyReason: chatBusyReason,
      unload: unloadChat
    },
    image: {
      // The image engine holds one of its models at a time: image, video or captioning.
      get label() {
        const kind = imageEngine.loadedKind
        return kind === 'video' ? 'video model' : kind === 'caption' ? 'captioning model' : 'image model'
      },
      isLoaded: () => imageEngine.state.state === 'running' && imageEngine.modelLoaded,
      busyReason: () => imageEngine.busyReason,
      unload: async () => {
        await imageEngine.request('POST', '/api/model/unload', {})
      }
    }
  },
  (message) => toRenderer('studio:notice', message)
)

// ------------------------------------------------------------ background

/**
 * Closing the window quits by default. With "keep running in the background"
 * chosen, it hides the window instead; the studio lives on in the tray, work in
 * progress carries on, and each model either stays on the GPU or is unloaded once
 * idle, as chosen per model (config.background, main/background.ts).
 */
let quitting = false
let tray: Tray | null = null
let toldAboutTray = false

const imageModelTarget = (kind: 'image' | 'video'): UnloadTarget => ({
  slot: kind,
  isLoaded: () => imageEngine.state.state === 'running' && ['ready', 'loading'].includes(imageEngine.models[kind]),
  busyReason: () => imageEngine.busyReason,
  unload: async () => {
    await imageEngine.request('POST', '/api/model/unload', { kind })
  }
})

const backgroundUnloader = new BackgroundUnloader(
  [
    { slot: 'chat', isLoaded: () => chat.loaded || chat.loading, busyReason: chatBusyReason, unload: unloadChat },
    imageModelTarget('image'),
    imageModelTarget('video')
  ],
  () => store.get().background.keepLoaded
)

/** A short message from the tray icon, when the window is not there to show one. */
function trayMessage(content: string): void {
  if (!tray || process.env['STUDIO_NO_BALLOON'] || process.platform !== 'win32') return
  tray.displayBalloon({ title: 'Local AI Studio', content, iconType: 'info' })
}

function ensureTray(): void {
  if (tray) return
  const icon = windowIcon().icon
  tray = new Tray(icon ? nativeImage.createFromPath(icon) : nativeImage.createEmpty())
  tray.setToolTip('Local AI Studio')
  tray.setContextMenu(
    Menu.buildFromTemplate([
      { label: 'Open Local AI Studio', click: showWindow },
      { type: 'separator' },
      { label: 'Quit (unloads every model)', click: () => app.quit() }
    ])
  )
  tray.on('click', showWindow)
}

/**
 * Hidden gallery folders hide again and unlocked ones lock again whenever the window
 * goes away, as they do when the studio quits and the engine with it (gallery.py).
 */
function resetGalleryPrivacy(): void {
  if (imageEngine.state.state !== 'running') return
  void imageEngine.request('POST', '/api/gallery/privacy/reset', {}).catch(() => undefined)
}

function toBackground(): void {
  if (!window) return
  window.hide()
  resetGalleryPrivacy()
  ensureTray()
  backgroundUnloader.start()
  if (!toldAboutTray) {
    toldAboutTray = true
    trayMessage('Still running in the background. Open it again, or quit, from this icon.')
  }
}

function showWindow(): void {
  backgroundUnloader.stop()
  if (!window) {
    createWindow()
    return
  }
  if (window.isMinimized()) window.restore()
  window.show()
  window.focus()
}

app.on('second-instance', showWindow)

function measureUsage(): void {
  if (!activeWorkspace || !recorder || !chat.loaded) return
  engine.send({
    type: 'measureUsage',
    system: systemFor(activeWorkspace),
    tools: toolsFor(activeWorkspace.internet),
    ...chatTarget(recorder)
  })
}

// The icon on disk is for unpackaged runs; a packaged app carries it in the
// executable, set by electron-builder from the same resources folder.
function windowIcon(): { icon?: string } {
  const file = path.join(import.meta.dirname, '../../resources', process.platform === 'win32' ? 'icon.ico' : 'icon.png')
  return fs.existsSync(file) ? { icon: file } : {}
}

// The top bars in the renderer double as the title bar: the system caption
// buttons are drawn over their right end, in the same colours (--panel and
// --muted in tokens.css). One pixel short of the bars' 54 px, so their bottom
// border runs under the buttons too. macOS keeps its own title bar.
const titleBar: Electron.BrowserWindowConstructorOptions =
  process.platform === 'darwin'
    ? {}
    : { titleBarStyle: 'hidden', titleBarOverlay: { color: '#131314', symbolColor: '#8f8f97', height: 53 } }

function createWindow(): void {
  window = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 940,
    minHeight: 620,
    show: false,
    backgroundColor: '#0b0b0c',
    title: 'Local AI Studio',
    ...windowIcon(),
    ...titleBar,
    webPreferences: {
      preload: path.join(import.meta.dirname, '../preload/index.cjs'),
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false
    }
  })

  // A window with a title bar overlay never gets ready-to-show (Electron 44),
  // so the first finished load reveals it too. Whichever comes first wins.
  const reveal = (): void => {
    if (window && !window.isVisible()) window.show()
  }
  window.once('ready-to-show', reveal)
  window.webContents.once('did-finish-load', reveal)
  window.on('close', (event) => {
    // A real quit (from the tray, or the system shutting down) always closes.
    if (quitting || store.get().background.closeBehavior !== 'background') return
    event.preventDefault()
    toBackground()
  })
  window.on('closed', () => {
    window = null
    resetGalleryPrivacy()
    // Anything still waiting on a dialog that no longer exists is refused.
    for (const [id, resolve] of pendingApprovals) {
      pendingApprovals.delete(id)
      resolve({ allow: false, grant: null })
    }
  })

  // The window only ever shows the studio. Without this, a file dropped on it or
  // a followed link would replace the page, and the preload bridge, with all its
  // capabilities, would be handed to whatever was loaded.
  window.webContents.on('will-navigate', (event) => event.preventDefault())
  window.webContents.on('will-frame-navigate', (event) => {
    if (!event.isMainFrame) event.preventDefault()
  })

  window.webContents.setWindowOpenHandler(({ url }) => {
    void shell.openExternal(url)
    return { action: 'deny' }
  })

  const devUrl = process.env['ELECTRON_RENDERER_URL']
  if (devUrl) {
    void window.loadURL(devUrl)
  } else {
    void window.loadFile(path.join(import.meta.dirname, '../renderer/index.html'))
  }
}

function registerHandlers(): void {
  ipcMain.handle('config:get', () => store.get())

  ipcMain.handle('config:setClassMode', async (_e, risk: RiskClass, mode: ClassMode) => {
    return store.update((c) => {
      c.permissions.modes[risk] = mode
    })
  })

  ipcMain.handle('config:setConfirmEverything', async (_e, value: boolean) => {
    return store.update((c) => {
      c.permissions.confirmEverything = value
    })
  })

  // ---------------------------------------------------------- workspaces

  ipcMain.handle('workspace:get', () => snapshot())

  ipcMain.handle('workspace:openFolder', async () => {
    assertIdle()
    const picked = await pickFolder('Open a folder to work in', 'Open folder')
    if (!picked) return snapshot()
    // Stored canonical. A workspace kept as the string the picker returned would
    // stop matching the moment a junction on the way to it was repointed.
    const resolved = canonicalize(picked)
    if (!resolved.ok || !resolved.existed) throw new Error('That folder cannot be opened.')
    await activate(await workspaces.open(resolved.path))
    return snapshot()
  })

  ipcMain.handle('workspace:switch', async (_e, id: unknown) => {
    assertIdle()
    const ws = isSafeId(id) ? workspaces.get(id) : null
    if (!ws) throw new Error('That workspace no longer exists.')
    if (!fs.existsSync(ws.path)) throw new Error(`The folder ${ws.path} is no longer there.`)
    await activate(ws)
    return snapshot()
  })

  ipcMain.handle('workspace:forget', async (_e, id: unknown) => {
    assertIdle()
    const ws = isSafeId(id) ? workspaces.get(id) : null
    if (!ws) return snapshot()
    const options = {
      type: 'warning' as const,
      buttons: ['Remove', 'Cancel'],
      defaultId: 1,
      cancelId: 1,
      title: 'Remove workspace',
      message: `Remove ${ws.name} from the list?`,
      detail: 'Its chats, project rules and saved permissions are deleted. The folder and its files are not touched.'
    }
    const { response } = window
      ? await dialog.showMessageBox(window, options)
      : await dialog.showMessageBox(options)
    if (response !== 0) return snapshot()
    if (activeWorkspace?.id === ws.id) {
      activeWorkspace = null
      recorder = null
      await store.update((c) => {
        c.lastWorkspaceId = null
      })
    }
    brokers.delete(ws.id)
    journals.delete(ws.id)
    await workspaces.forget(ws.id)
    return snapshot()
  })

  ipcMain.handle('workspace:saveRules', async (_e, rules: unknown) => {
    const ws = activeWorkspace
    if (!ws) throw new Error('No folder is open.')
    activeWorkspace = await workspaces.update(ws.id, (w) => {
      w.rules = cleanRules(rules)
    })
    // The next prompt picks the new rules up; the conversation carries on.
    measureUsage()
    return workspaceState()
  })

  ipcMain.handle('workspace:addFolder', async () => {
    const ws = activeWorkspace
    if (!ws) throw new Error('No folder is open.')
    const picked = await pickFolder('Share another folder in this workspace', 'Share this folder')
    if (!picked) return workspaceState()
    const resolved = canonicalize(picked)
    if (!resolved.ok || !resolved.existed) throw new Error('That folder cannot be shared.')
    const root: WorkspaceRoot = {
      id: randomUUID(),
      path: resolved.path,
      label: path.basename(resolved.path) || resolved.path,
      enabled: true,
      addedAt: new Date().toISOString()
    }
    activeWorkspace = await workspaces.update(ws.id, (w) => {
      const known = [w.path, ...w.extraRoots.map((r) => r.path)].map(comparablePath)
      if (!known.includes(comparablePath(root.path))) w.extraRoots.push(root)
    })
    measureUsage()
    return workspaceState()
  })

  ipcMain.handle('workspace:setFolderEnabled', async (_e, id: unknown, enabled: unknown) => {
    const ws = activeWorkspace
    if (!ws) throw new Error('No folder is open.')
    activeWorkspace = await workspaces.update(ws.id, (w) => {
      const root = w.extraRoots.find((r) => r.id === id)
      if (root) root.enabled = enabled === true
    })
    measureUsage()
    return workspaceState()
  })

  ipcMain.handle('workspace:removeFolder', async (_e, id: unknown) => {
    const ws = activeWorkspace
    if (!ws) throw new Error('No folder is open.')
    activeWorkspace = await workspaces.update(ws.id, (w) => {
      w.extraRoots = w.extraRoots.filter((r) => r.id !== id)
    })
    measureUsage()
    return workspaceState()
  })

  ipcMain.handle('workspace:removeGrant', async (_e, id: unknown) => {
    const ws = activeWorkspace
    if (!ws) throw new Error('No folder is open.')
    activeWorkspace = await workspaces.update(ws.id, (w) => {
      w.policyRules = w.policyRules.filter((r) => r.id !== id)
    })
    return workspaceState()
  })

  ipcMain.handle('session:clearGrants', () => {
    if (activeWorkspace) brokerFor(activeWorkspace).clearSessionGrants()
  })

  // ---------------------------------------------------------- chats

  ipcMain.handle('chat:new', () => {
    assertIdle()
    if (!activeWorkspace) throw new Error('No folder is open.')
    setRecorder(ChatRecorder.fresh())
    measureUsage()
    return snapshot()
  })

  ipcMain.handle('chat:open', (_e, chatId: unknown) => {
    assertIdle()
    const ws = activeWorkspace
    if (!ws) throw new Error('No folder is open.')
    const saved = isSafeId(chatId) ? workspaces.loadChat(ws.id, chatId) : null
    if (!saved) throw new Error('That chat could not be read.')
    setRecorder(new ChatRecorder(saved))
    measureUsage()
    return snapshot()
  })

  ipcMain.handle('chat:delete', async (_e, chatId: unknown) => {
    const ws = activeWorkspace
    if (!ws || !isSafeId(chatId)) return snapshot()
    if (recorder?.chat.meta.id === chatId) {
      assertIdle()
      setRecorder(ChatRecorder.fresh())
      measureUsage()
    }
    await saving.catch(() => undefined)
    await workspaces.deleteChat(ws.id, chatId)
    return snapshot()
  })

  ipcMain.handle('rules:save', async (_e, rules: unknown) => {
    const config = await store.update((c) => {
      c.rules = cleanRules(rules)
    })
    // Global rules apply to the open chat from its next prompt; nothing is cleared.
    measureUsage()
    return config
  })

  ipcMain.handle('model:pick', async () => {
    const result = await dialog.showOpenDialog({
      title: 'Choose a GGUF model',
      properties: ['openFile'],
      // Some model managers keep GGUF files named only by their hash, with no
      // extension (blobs/sha256-…), so "All files" is offered too; the file is
      // checked by content on load.
      filters: [
        { name: 'GGUF model', extensions: ['gguf'] },
        { name: 'All files (for models stored without an extension)', extensions: ['*'] }
      ]
    })
    return result.canceled ? null : (result.filePaths[0] ?? null)
  })

  ipcMain.handle('model:load', async (_e, modelPath: string) => {
    if (!fs.existsSync(modelPath)) throw new Error('That model file no longer exists.')
    if (!(await isGgufFile(modelPath))) throw new Error('That file is not a GGUF model.')
    await gpu.claim('chat')
    chat.loading = true
    const config = store.get()
    await store.update((c) => {
      c.model.lastModelPath = modelPath
    })
    sendLoad({
      modelPath,
      backend: config.model.backend,
      gpuLayers: config.model.gpuLayers,
      contextSize: config.model.contextSize,
      kvCache: config.model.kvCache
    })
  })

  // Takes effect at the next load; the window offers to reload straight away.
  ipcMain.handle('config:setKvCache', async (_e, kv: unknown) => {
    if (!isKvCacheType(kv)) throw new Error('Unknown context memory type.')
    return store.update((c) => {
      c.model.kvCache = kv
    })
  })

  ipcMain.handle('config:setBackground', async (_e, prefs: unknown) => {
    // Rebuilt field by field; anything the window sends that is not a known value is refused.
    const raw = prefs as Partial<BackgroundPreferences> | null
    if (!raw || typeof raw !== 'object' || !isCloseBehavior(raw.closeBehavior)) throw new Error('Invalid setting.')
    const keep = raw.keepLoaded as Record<string, unknown> | undefined
    if (!keep || !MODEL_SLOTS.every((slot) => typeof keep[slot] === 'boolean')) throw new Error('Invalid setting.')
    return store.update((c) => {
      c.background = parseBackground(raw)
    })
  })

  ipcMain.handle('model:unload', () => {
    engine.send({ type: 'unload' })
  })

  ipcMain.handle('chat:send', (_e, text: unknown) => {
    const ws = activeWorkspace
    const rec = recorder
    if (!ws || !rec) throw new Error('Open a folder first.')
    // Refused here rather than left to fail in the engine, which would record
    // the message, and keep the chat, for a reply that can never come.
    if (!chat.loaded) throw new Error('Load a model first.')
    if (typeof text !== 'string' || !text.trim()) throw new Error('Nothing to send.')
    const id = randomUUID()
    chat.activePrompts.add(id)
    rec.userMessage(id, text)
    // Whatever this reply changes is grouped under the message that asked for it.
    journalFor(ws.id).label(id, chatTitle(text), rec.chat.meta.id)
    // Saved straight away, so the message survives even if the reply never comes.
    saveChat()
    engine.send({ type: 'prompt', id, text, system: systemFor(ws), tools: toolsFor(ws.internet), ...chatTarget(rec) })
    return id
  })

  ipcMain.handle('chat:abort', (_e, promptId: string) => {
    engine.send({ type: 'abort', id: promptId })
  })

  ipcMain.handle('usage:refresh', () => {
    measureUsage()
  })

  ipcMain.handle('approval:respond', (_e, id: string, response: ApprovalResponse) => {
    const resolve = pendingApprovals.get(id)
    if (resolve) {
      pendingApprovals.delete(id)
      resolve(response)
    }
  })

  // ---------------------------------------------------------- change history

  ipcMain.handle('journal:list', () => (activeWorkspace ? journalFor(activeWorkspace.id).groups() : []))

  ipcMain.handle('journal:revert', async (_e, groupId: unknown, force: unknown) => {
    assertIdle()
    if (!activeWorkspace) throw new Error('No folder is open.')
    if (typeof groupId !== 'string' || !groupId) throw new Error('Nothing to revert.')
    const broker = brokerFor(activeWorkspace)
    const result = await journalFor(activeWorkspace.id).revert(groupId, force === true)
    // Reverted files are no longer what the agent read.
    if (result.done) broker.resetConversation()
    return result
  })

  // ---------------------------------------------------------- internet

  ipcMain.handle('workspace:setInternet', async (_e, level: unknown) => {
    const ws = activeWorkspace
    if (!ws) throw new Error('No folder is open.')
    if (!isInternetLevel(level)) throw new Error('Not an internet level.')
    assertIdle()
    activeWorkspace = await workspaces.update(ws.id, (w) => {
      w.internet = level
    })
    measureUsage()
    return workspaceState()
  })

  ipcMain.handle('network:log', () => (activeWorkspace ? networkLogFor(activeWorkspace.id).recent(200) : []))

  ipcMain.handle('internet:keyStatus', () => ({ hasKey: secrets.has('searchKey') }))

  ipcMain.handle('internet:setKey', async (_e, key: unknown) => {
    if (key !== null && (typeof key !== 'string' || !/^[!-~]{8,200}$/.test(key.trim()))) {
      throw new Error('That does not look like an API key.')
    }
    await secrets.set('searchKey', key === null ? null : key.trim())
    return { hasKey: secrets.has('searchKey') }
  })

  ipcMain.handle('workspace:setProtected', async (_e, list: unknown) => {
    const ws = activeWorkspace
    if (!ws) throw new Error('No folder is open.')
    activeWorkspace = await workspaces.update(ws.id, (w) => {
      w.protectedPaths = withDefaultProtections(cleanProtections(list))
    })
    measureUsage()
    return workspaceState()
  })
}

/** Protections as they arrive from the window. Built-in ones can be switched off, not removed. */
function cleanProtections(input: unknown): ProtectedPath[] {
  if (!Array.isArray(input)) throw new Error('Protections must be a list.')
  return input
    .filter((p): p is Record<string, unknown> => typeof p === 'object' && p !== null)
    .filter((p) => typeof p['pattern'] === 'string' && (p['pattern'] as string).trim().length > 0)
    .map((p) => ({
      id: typeof p['id'] === 'string' && p['id'] ? (p['id'] as string) : randomUUID(),
      pattern: (p['pattern'] as string).trim().slice(0, 300),
      level: p['level'] === 'read-only' ? ('read-only' as const) : ('hidden' as const),
      enabled: p['enabled'] !== false,
      builtin: typeof p['id'] === 'string' && (p['id'] as string).startsWith('builtin:')
    }))
}

/** Rules as they arrive from the window, reduced to the fields a rule has. */
function cleanRules(input: unknown): GlobalRule[] {
  if (!Array.isArray(input)) throw new Error('Rules must be a list.')
  return input
    .filter((r): r is Record<string, unknown> => typeof r === 'object' && r !== null)
    .map((r) => ({
      id: typeof r['id'] === 'string' && r['id'] ? r['id'] : randomUUID(),
      title: typeof r['title'] === 'string' ? r['title'].slice(0, 200) : '',
      body: typeof r['body'] === 'string' ? r['body'].slice(0, 20_000) : '',
      enabled: r['enabled'] === true
    }))
}

/** Reopen the workspace that was open when the app last closed, if its folder is still there. */
function restoreLastWorkspace(): void {
  const id = store.get().lastWorkspaceId
  const ws = id ? workspaces.get(id) : null
  if (!ws || !fs.existsSync(ws.path)) return
  activeWorkspace = ws
  setRecorder(ChatRecorder.fresh())
}

app.whenReady().then(() => {
  // Groups the window on the Windows taskbar under this app rather than under
  // a generic Electron entry. Same id as appId in electron-builder.yml.
  if (process.platform === 'win32') app.setAppUserModelId('com.justsimpleideas.localaistudio')
  restoreLastWorkspace()
  session.defaultSession.webRequest.onBeforeRequest((details, callback) => {
    callback({ cancel: !isLocalRequest(details.url) })
  })
  registerHandlers()
  registerImageHandlers({
    ipcMain,
    dialog,
    store,
    imageEngine,
    gpu,
    getWindow: () => window,
    setup: engineSetup,
    downloads: modelDownloads,
    inspector: modelInspector,
    importer: modelImporter,
    defaultModelsFolder: path.join(imageDataDir, 'models'),
    openExternal: (url) => shell.openExternal(url)
  })
  registerTrainingHandlers({ ipcMain, dialog, imageEngine, gpu, getWindow: () => window, collector: photoCollector })
  protocol.handle(IMAGE_SCHEME, (request) => serveImage(imageEngine, request.url, request.headers.get('Range')))
  createWindow()
  engine.start()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  engine.stop()
  imageEngine.stop()
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', () => {
  quitting = true
  backgroundUnloader.stop()
  engine.stop()
  imageEngine.stop()
  // pip and the downloader leave resumable state behind, so stopping them is safe.
  engineSetup.cancel()
  modelDownloads.cancel()
  tray?.destroy()
  tray = null
})
