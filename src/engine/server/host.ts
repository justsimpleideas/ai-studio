import { spawn, type ChildProcess } from 'node:child_process'
import { randomBytes, randomUUID } from 'node:crypto'
import fsp from 'node:fs/promises'
import net from 'node:net'
import path from 'node:path'
import type { ChatHistoryItem } from 'node-llama-cpp'
import { joinSystemSections, type SystemSection } from '@shared/config.js'
import { modelDisplayName, type ChatTarget, type EngineToMain, type MainToEngine, type ModelInfo } from '@shared/engine-protocol.js'
import {
  DEFAULT_CONTEXT_CAP,
  DEFAULT_GENERATION,
  defaultContextSize,
  recommendedSampling,
  resolveSampling,
  type GenerationPreferences,
  type ModelTraits
} from '@shared/generation.js'
import { describeGguf, readGgufHeader, type ByteSource } from '@shared/gguf.js'
import { ReplyClock } from '@shared/reply-stats.js'
import { TOOL_NAMES, type ToolName } from '@shared/tools.js'
import type { TurnMeter } from '@shared/usage.js'
import { thoughtBudget } from '../thinking.js'
import { emptyServerLog, parseDeviceList, readServerLine, serverArgs, serverEnv, type ServerLog } from './args.js'
import { measureRemote, type MeasureClient } from './measure.js'
import {
  completionBody,
  parseArguments,
  readChunk,
  ReplyAccumulator,
  toServerMessages,
  type WireMessage
} from './messages.js'
import { SseParser } from './sse.js'

/**
 * The model host, served by llama-server.
 *
 * The same process role as engine/host.ts, with the same messages in and out,
 * so the rest of the studio does not know which it is talking to. The model
 * itself runs in a llama-server program the user installed, which this process
 * starts and stops. The server listens on the loopback address only, on a port
 * chosen as free just before, behind a key made for that one start and handed
 * to it through its environment. Nothing in this process, or in the server,
 * reaches anywhere but the other.
 *
 * Why it exists: a mixture-of-experts model bigger than the GPU runs far better
 * when only its experts live in system memory and everything else stays on the
 * GPU. llama.cpp can do that; the library inside the studio cannot ask for it.
 */

const parentPort = process.parentPort

function send(message: EngineToMain): void {
  parentPort.postMessage(message)
}

let server: ChildProcess | null = null
let baseUrl = ''
let apiKey = ''
let serverLog: ServerLog = emptyServerLog()
let loaded: ModelInfo | null = null
let traits: ModelTraits | null = null
let generation: GenerationPreferences = DEFAULT_GENERATION
/** The conversation, in the saved form, system message first. */
let history: ChatHistoryItem[] = []
let activeChatId: string | null = null
let activeSystemPrompt = ''
let activeSystem: SystemSection[] = []
let activeTools: ToolName[] = [...TOOL_NAMES]
let turnMeters: TurnMeter[] = []
const running = new Map<string, AbortController>()
const clocks = new Map<string, ReplyClock>()

interface PendingCall {
  promptId: string
  settle: (result: { ok: boolean; payload: unknown }) => void
}

const pendingCalls = new Map<string, PendingCall>()

/** Ask the application to run a tool; parked until it decides. Same contract as the built-in host. */
function requestTool(promptId: string, name: string, args: unknown): Promise<{ ok: boolean; payload: unknown }> {
  const callId = randomUUID()
  clocks.get(promptId)?.pause()
  return new Promise((resolve) => {
    pendingCalls.set(callId, {
      promptId,
      settle: (result) => {
        clocks.get(promptId)?.resume()
        resolve(result)
      }
    })
    send({ type: 'toolCall', callId, promptId, call: { name, args } })
  })
}

// ------------------------------------------------------------------ the server

class FileSource implements ByteSource {
  constructor(
    private readonly handle: fsp.FileHandle,
    readonly size: number
  ) {}

  async read(offset: number, length: number): Promise<Uint8Array> {
    const buffer = Buffer.alloc(length)
    const { bytesRead } = await this.handle.read(buffer, 0, length, offset)
    return buffer.subarray(0, bytesRead)
  }
}

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const probe = net.createServer()
    probe.on('error', reject)
    probe.listen(0, '127.0.0.1', () => {
      const { port } = probe.address() as net.AddressInfo
      probe.close(() => resolve(port))
    })
  })
}

async function request(route: string, init: RequestInit = {}): Promise<Response> {
  const response = await fetch(`${baseUrl}${route}`, {
    ...init,
    headers: { 'content-type': 'application/json', authorization: `Bearer ${apiKey}`, ...(init.headers ?? {}) }
  })
  if (!response.ok) {
    const text = await response.text().catch(() => '')
    throw new Error(`The server answered ${response.status} to ${route}${text ? `: ${text.slice(0, 300)}` : ''}`)
  }
  return response
}

const client: MeasureClient = {
  async tokenCount(text) {
    const body = (await (await request('/tokenize', { method: 'POST', body: JSON.stringify({ content: text, add_special: false }) })).json()) as {
      tokens?: unknown[]
    }
    return Array.isArray(body.tokens) ? body.tokens.length : 0
  },
  async render(messages, tools) {
    const body = (await (
      await request('/apply-template', { method: 'POST', body: JSON.stringify({ messages, ...(tools ? { tools } : {}) }) })
    ).json()) as { prompt?: string }
    return typeof body.prompt === 'string' ? body.prompt : ''
  }
}

async function stopServer(): Promise<void> {
  const child = server
  server = null
  loaded = null
  traits = null
  endSession()
  if (!child || child.exitCode !== null || child.signalCode !== null) {
    send({ type: 'sidecar', pid: null })
    return
  }
  await new Promise<void>((resolve) => {
    const timer = setTimeout(() => child.kill(), 3000)
    child.once('exit', () => {
      clearTimeout(timer)
      resolve()
    })
    child.kill()
  })
  send({ type: 'sidecar', pid: null })
}

/** `llama-server --list-devices`: prints the devices and exits, loading nothing. */
function listDevices(program: string): Promise<{ backend: string | null; devices: string[] }> {
  return new Promise((resolve) => {
    let out = ''
    const child = spawn(program, ['--list-devices'], {
      cwd: path.dirname(program),
      env: serverEnv(process.env, ''),
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true
    })
    const timer = setTimeout(() => child.kill(), 30_000)
    child.stdout?.on('data', (d: Buffer) => (out += d.toString()))
    child.stderr?.on('data', (d: Buffer) => (out += d.toString()))
    child.on('error', () => {
      clearTimeout(timer)
      resolve({ backend: null, devices: [] })
    })
    child.on('exit', () => {
      clearTimeout(timer)
      resolve(parseDeviceList(out))
    })
  })
}

/** Wait until /health says the model is in; fail as soon as the server has gone instead. */
async function waitForHealth(child: ChildProcess, limitMs: number): Promise<void> {
  const started = Date.now()
  while (Date.now() - started < limitMs) {
    if (child.exitCode !== null || child.signalCode !== null) {
      throw new Error(`llama-server stopped while loading (exit code ${child.exitCode ?? 'signal'}).`)
    }
    try {
      const response = await fetch(`${baseUrl}/health`, { headers: { authorization: `Bearer ${apiKey}` } })
      if (response.ok) return
    } catch {
      // Not listening yet.
    }
    await new Promise((resolve) => setTimeout(resolve, 500))
  }
  throw new Error('llama-server did not finish loading the model in time.')
}

const SAMPLE_MESSAGES: WireMessage[] = [
  { role: 'system', content: 'You are a careful assistant.' },
  { role: 'user', content: 'What is the capital of France?' },
  { role: 'assistant', content: 'Paris.' },
  { role: 'user', content: 'And of Italy?' }
]

async function load(message: MainToEngine & { type: 'load' }): Promise<void> {
  await stopServer()
  const { request: req } = message
  const program = req.server.path
  if (!program) throw new Error('No llama-server program is chosen.')
  await fsp.access(program)

  // The header, read by the studio's own strict reader: what the model is, and
  // the context it was trained for, before the server is started on it.
  const handle = await fsp.open(req.modelPath, 'r')
  let details
  let size: number
  let parameters = 0
  try {
    size = (await handle.stat()).size
    const { header } = await readGgufHeader(new FileSource(handle, size))
    details = describeGguf(header)
    for (const t of header.tensors) parameters += t.dims.reduce((a, b) => a * b, 1)
  } finally {
    await handle.close()
  }
  const trained = details.contextLength ?? DEFAULT_CONTEXT_CAP
  const contextSize = Math.min(req.contextSize ?? defaultContextSize(trained), trained)
  const moe = details.experts !== null

  // Which GPU it will use, asked of the program itself before the model goes
  // anywhere; its load log does not name the device at the default verbosity.
  const listed = await listDevices(program)
  const port = await freePort()
  apiKey = randomBytes(24).toString('hex')
  baseUrl = `http://127.0.0.1:${port}`
  serverLog = emptyServerLog()
  serverLog.backend = listed.backend
  serverLog.devices = listed.devices
  const args = serverArgs({
    modelPath: req.modelPath,
    port,
    contextSize,
    kvCache: req.kvCache,
    gpuLayers: req.gpuLayers,
    threads: req.server.threads,
    mixtureOfExperts: moe,
    expertsOnCpu: req.server.expertsOnCpu
  })
  // For the engine log (the path is the model's, already chosen by the user).
  console.log(`[engine] llama-server: ${args.filter((a) => a !== req.modelPath).join(' ')}`)
  send({ type: 'loadProgress', fraction: 0 })
  const child = spawn(program, args, {
    cwd: path.dirname(program),
    env: serverEnv(process.env, apiKey),
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true
  })
  server = child
  send({ type: 'sidecar', pid: child.pid ?? null })
  const take = (chunk: Buffer): void => {
    for (const raw of chunk.toString().split(/\r?\n/)) {
      const line = raw.trimEnd()
      if (!line) continue
      readServerLine(serverLog, line)
      console.log(`[llama-server] ${line}`)
    }
  }
  child.stdout?.on('data', take)
  child.stderr?.on('data', take)
  child.on('exit', (code) => {
    if (server !== child) return
    // Gone on its own: the model is no longer there, whatever the window thinks.
    server = null
    loaded = null
    send({ type: 'sidecar', pid: null })
    send({ type: 'error', id: null, message: `llama-server stopped (exit code ${code ?? 'signal'}). The model is no longer loaded.`, fatal: true })
    send({ type: 'unloaded' })
  })

  await waitForHealth(child, 20 * 60 * 1000)
  if (server !== child) throw new Error('The load was cancelled.')

  let n_ctx = contextSize
  try {
    const props = (await (await request('/props')).json()) as { default_generation_settings?: { n_ctx?: number } }
    if (typeof props.default_generation_settings?.n_ctx === 'number') n_ctx = props.default_generation_settings.n_ctx
  } catch {
    // The requested size stands.
  }

  const template = details.template?.text ?? null
  traits = {
    architecture: details.architecture ?? 'unknown',
    experts: details.experts,
    // The template is the authority: a model that can think has a switch for it, or a thought block.
    thinking: template !== null && /enable_thinking|<think>|<\|think\|>|reasoning/i.test(template),
    quantization: details.fileType,
    parameters
  }
  let rendered: string
  try {
    rendered = await client.render(SAMPLE_MESSAGES, null)
  } catch (err) {
    rendered = `(could not render: ${(err as Error).message})`
  }
  console.log(
    `[engine] loaded through llama-server: backend ${serverLog.backend ?? 'unknown'} ${JSON.stringify(serverLog.devices)}, ${serverLog.gpuLayers ?? '?'}/${serverLog.totalLayers ?? '?'} layers on the GPU, context ${n_ctx}, kv ${req.kvCache}${moe ? `, experts on CPU: ${req.server.expertsOnCpu ?? 'all'}` : ''}`
  )
  loaded = {
    path: req.modelPath,
    name: modelDisplayName(path.basename(req.modelPath), details.name ?? undefined),
    trainContextSize: trained,
    contextSize: n_ctx,
    // Asked for all of them; when the log did not say, that is what was placed.
    gpuLayers: serverLog.gpuLayers ?? (req.gpuLayers === null && serverLog.backend ? (details.layers ?? 0) : (req.gpuLayers ?? 0)),
    backend: serverLog.backend ?? 'gpu',
    devices: serverLog.devices,
    fileSizeBytes: size,
    kvCache: req.kvCache,
    engine: 'server',
    ...(moe ? { expertsOnCpu: req.server.expertsOnCpu ?? ('all' as const) } : {}),
    threads: req.server.threads,
    traits,
    recommendedSampling: recommendedSampling(traits),
    template: { wrapper: 'llama-server', usesFileTemplate: true, source: template, rendered: rendered.length > 6000 ? `${rendered.slice(0, 6000)}\n…` : rendered }
  }
  send({ type: 'loadProgress', fraction: 1 })
  send({ type: 'loaded', info: loaded })
}

// -------------------------------------------------------------- the conversation

function endSession(): void {
  history = []
  activeChatId = null
  activeSystemPrompt = ''
  activeSystem = []
  turnMeters = []
}

/** The conversation for `target`: switched to its saved history when the chat changed, with the current system message at its head. */
function ensureSession(system: SystemSection[], target: ChatTarget): void {
  const systemPrompt = joinSystemSections(system)
  const sameChat = activeChatId === target.chatId
  if (!sameChat) {
    history = Array.isArray(target.history) ? [...(target.history as ChatHistoryItem[])] : []
    turnMeters = []
    activeChatId = target.chatId
  }
  const rest = history[0]?.type === 'system' ? history.slice(1) : history
  history = [{ type: 'system', text: systemPrompt }, ...rest]
  activeSystemPrompt = systemPrompt
  activeSystem = system
}

async function reportUsage(): Promise<void> {
  if (!loaded) return
  try {
    const snapshot = await measureRemote(client, history, activeSystem, turnMeters, loaded.contextSize, activeTools)
    send({ type: 'usage', snapshot })
  } catch (err) {
    console.warn('Token usage could not be measured:', (err as Error).message)
  }
}

/** One streamed completion; returns how the reply ended and what the server counted. */
async function streamCompletion(
  promptId: string,
  messages: WireMessage[],
  reply: ReplyAccumulator,
  clock: ReplyClock,
  signal: AbortSignal
): Promise<{ finish: string | null; usage: { input: number; output: number } | null }> {
  const sampling = resolveSampling(generation.sampling, traits ?? { architecture: '', thinking: false })
  const body = completionBody({
    messages,
    tools: activeTools,
    sampling,
    thinking: generation.thinking,
    thinkingBudget: thoughtBudget(generation.thinking, generation.thinkingBudget)
  })
  const response = await request('/v1/chat/completions', { method: 'POST', body: JSON.stringify(body), signal })
  if (!response.body) throw new Error('The server sent no reply.')
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  const parser = new SseParser()
  let finish: string | null = null
  let usage: { input: number; output: number } | null = null
  const handle = (event: string): void => {
    if (event === '[DONE]') return
    const chunk = readChunk(event)
    if (chunk.reasoning) {
      clock.thought()
      reply.thought(chunk.reasoning)
      send({ type: 'thought', id: promptId, text: chunk.reasoning })
    }
    if (chunk.content) {
      clock.text()
      reply.endThought()
      reply.text(chunk.content)
      send({ type: 'chunk', id: promptId, text: chunk.content })
    }
    for (const delta of chunk.toolCalls) {
      clock.text()
      reply.toolCall(delta)
    }
    if (chunk.finish) finish = chunk.finish
    if (chunk.usage) usage = chunk.usage
  }
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    for (const event of parser.push(decoder.decode(value, { stream: true }))) handle(event)
  }
  for (const event of parser.flush()) handle(event)
  return { finish, usage }
}

async function runPrompt(message: MainToEngine & { type: 'prompt' }): Promise<void> {
  if (!loaded) throw new Error('No model is loaded.')
  const controller = new AbortController()
  running.set(message.id, controller)
  const clock = new ReplyClock()
  clocks.set(message.id, clock)
  const reply = new ReplyAccumulator()
  let inputTokens = 0
  let outputTokens = 0
  let turn = 0
  try {
    activeTools = message.tools
    generation = message.generation
    ensureSession(message.system, message)
    turn = history.filter((item) => item.type === 'user').length + 1
    history.push({ type: 'user', text: message.text })
    const messages = toServerMessages(history)
    // The model may call tools several times before it answers; each call is
    // run by the application and its result given back in a further request.
    for (let round = 0; round < 32; round++) {
      const before = reply.replyText.length
      const { finish, usage } = await streamCompletion(message.id, messages, reply, clock, controller.signal)
      if (usage) {
        inputTokens += usage.input
        outputTokens += usage.output
      }
      const calls = reply.takeCalls()
      if (finish !== 'tool_calls' && calls.length === 0) break
      if (calls.length === 0) break
      // What this round wrote, with the calls it made, then each call's result.
      messages.push({ role: 'assistant', content: reply.replyText.slice(before) || null, tool_calls: calls })
      for (const call of calls) {
        const params = parseArguments(call)
        const result = await requestTool(message.id, call.function.name, params)
        const payload = result.ok ? result.payload : { error: result.payload }
        reply.result(call, params, payload)
        messages.push({ role: 'tool', tool_call_id: call.id, content: JSON.stringify(payload) })
      }
    }
    reply.endThought()
    history.push({ type: 'model', response: reply.parts })
    sendStats(message.id, clock, outputTokens)
    send({ type: 'complete', id: message.id, text: reply.replyText })
  } catch (err) {
    if (controller.signal.aborted) {
      reply.endThought()
      if (reply.parts.length) history.push({ type: 'model', response: reply.parts })
      sendStats(message.id, clock, outputTokens)
      send({ type: 'aborted', id: message.id })
    } else {
      send({ type: 'error', id: message.id, message: (err as Error).message, fatal: false })
    }
  } finally {
    running.delete(message.id)
    clocks.delete(message.id)
    turnMeters.push({ turn, inputTokens, outputTokens })
    for (const [callId, pending] of pendingCalls) {
      if (pending.promptId !== message.id) continue
      pendingCalls.delete(callId)
      pending.settle({ ok: false, payload: 'The turn ended before this could run.' })
    }
    if (activeChatId === message.chatId) send({ type: 'history', chatId: message.chatId, history })
    await reportUsage()
  }
}

function sendStats(id: string, clock: ReplyClock, outputTokens: number): void {
  send({ type: 'stats', id, stats: clock.stats(outputTokens) })
}

parentPort.on('message', (event) => {
  const message = event.data as MainToEngine
  void (async () => {
    try {
      switch (message.type) {
        case 'load':
          await load(message)
          break
        case 'unload':
          await stopServer()
          send({ type: 'unloaded' })
          break
        case 'prompt':
          await runPrompt(message)
          break
        case 'abort':
          running.get(message.id)?.abort()
          break
        case 'toolResult': {
          const pending = pendingCalls.get(message.callId)
          if (pending) {
            pendingCalls.delete(message.callId)
            pending.settle({ ok: message.ok, payload: message.payload })
          }
          break
        }
        case 'measureUsage':
          if (running.size > 0 || !loaded) break
          activeTools = message.tools
          ensureSession(message.system, message)
          await reportUsage()
          break
      }
    } catch (err) {
      const id = 'id' in message ? (message.id as string) : null
      console.error(`[engine] ${message.type} failed:`, (err as Error).stack ?? (err as Error).message)
      send({ type: 'error', id, message: (err as Error).message, fatal: message.type === 'load' })
    }
  })()
})

process.on('exit', () => {
  // A last try at not leaving the server behind; the application also ends it by id.
  server?.kill()
})

send({ type: 'ready' })
