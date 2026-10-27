import { randomUUID } from 'node:crypto'
import {
  type ChatHistoryItem,
  defineChatSessionFunction,
  getLlama,
  LlamaChatSession,
  type ChatSessionModelFunctions,
  type Llama,
  type LlamaContext,
  type LlamaContextSequence,
  type LlamaModel,
  type TokenMeterState
} from 'node-llama-cpp'
import { modelDisplayName, type ChatTarget, type EngineToMain, type MainToEngine, type ModelInfo } from '@shared/engine-protocol.js'
import { joinSystemSections, type GpuBackend, type SystemSection } from '@shared/config.js'
import { TOOL_DESCRIPTIONS, TOOL_NAMES, TOOL_SCHEMAS, type ToolName } from '@shared/tools.js'
import type { TurnMeter } from '@shared/usage.js'
import { kvCacheOptions } from './kv-cache.js'
import { measureConversation } from './measure.js'

/**
 * The model host.
 *
 * Runs as an Electron utilityProcess for two reasons. llama.cpp is native code
 * that can take the process down with it when a GGUF is malformed or the GPU
 * runs out of memory, and a crash here should cost the user their session
 * rather than their application. And this is the process that handles the raw
 * output of a model, so it is the one that should hold no authority: it can ask
 * for a file to be read, but it has no idea how to read one.
 */

const parentPort = process.parentPort

function send(message: EngineToMain): void {
  parentPort.postMessage(message)
}

let llama: Llama | null = null
let activeBackend: GpuBackend | null = null
let model: LlamaModel | null = null
let context: LlamaContext | null = null
/**
 * The context's one sequence, kept for the life of the context. A context is
 * created with a fixed number of sequences (one, here), and a chat session does
 * not hand its sequence back when it is dropped, so taking a fresh one for
 * every new chat would fail with "No sequences left" on the second.
 */
let sequence: LlamaContextSequence | null = null
let session: LlamaChatSession | null = null
/** Which of the application's chats the session holds. */
let activeChatId: string | null = null
let activeSystemPrompt = ''
let activeSystem: SystemSection[] = []
/** The tools the current conversation's model is given; the file tools unless the workspace allows more. */
let activeTools: ToolName[] = [...TOOL_NAMES]
/** Per-prompt work for the current chat, from llama.cpp's own token meter. */
let turnMeters: TurnMeter[] = []

/** Prompts in flight, so an abort from the UI can reach the right generation. */
const running = new Map<string, AbortController>()

/**
 * Tool calls waiting on the other process to decide.
 *
 * A handler parks here while the broker checks policy and, when it needs to,
 * while a person reads a dialog. There is no timeout: a prompt sitting unanswered
 * because the user went to make coffee is a normal state, not a stuck one.
 */
interface PendingCall {
  promptId: string
  settle: (result: { ok: boolean; payload: unknown }) => void
}

const pendingCalls = new Map<string, PendingCall>()

function requestTool(promptId: string, name: ToolName, args: unknown): Promise<unknown> {
  const callId = randomUUID()
  return new Promise((resolve) => {
    pendingCalls.set(callId, {
      promptId,
      // A refusal is handed back to the model as an ordinary result rather than
      // thrown. The model needs to learn that this door is closed so it can try
      // something else; killing the turn would just leave the user staring at a
      // half-finished answer.
      settle: ({ ok, payload }) => resolve(ok ? payload : { error: payload })
    })
    send({ type: 'toolCall', callId, promptId, call: { name, args } })
  })
}

function buildFunctions(promptId: string, tools: readonly ToolName[]): ChatSessionModelFunctions {
  const functions: Record<string, unknown> = {}
  for (const name of tools) {
    functions[name] = defineChatSessionFunction({
      description: TOOL_DESCRIPTIONS[name],
      // The same schema the broker validates against. node-llama-cpp compiles it
      // into a GBNF grammar, so the model is constrained at the token level and
      // structurally cannot emit a call with a missing or misspelled argument.
      params: TOOL_SCHEMAS[name] as never,
      handler: (args: unknown) => requestTool(promptId, name, args)
    })
  }
  return functions as ChatSessionModelFunctions
}

async function disposeModel(): Promise<void> {
  endSession()
  sequence = null
  await context?.dispose().catch(() => undefined)
  context = null
  await model?.dispose().catch(() => undefined)
  model = null
}

async function load(request: MainToEngine & { type: 'load' }): Promise<void> {
  await disposeModel()

  // The backend is chosen when the llama instance is built, so switching it
  // means building a new one. Worth doing explicitly: the automatic pick lands
  // on whatever initialises, which on a card with no matching CUDA prebuilt is
  // Vulkan — it works, so nothing looks wrong, and the user never finds out
  // there was a choice.
  const wanted = request.request.backend
  if (llama && wanted !== activeBackend) llama = null
  if (!llama) {
    // Only the prebuilt binaries are ever used: never a build from source, which
    // would fetch llama.cpp from the network. A purely local app stays local.
    llama = await getLlama({
      build: 'never',
      skipDownload: true,
      ...(wanted === 'auto' ? {} : { gpu: wanted === 'cpu' ? false : wanted })
    })
    activeBackend = wanted
  }

  let lastReported = -1
  model = await llama.loadModel({
    modelPath: request.request.modelPath,
    gpuLayers: request.request.gpuLayers ?? 'auto',
    onLoadProgress(fraction) {
      // Loading a 9 GB file is slow enough that silence reads as a hang. Only
      // whole percents are forwarded so the channel is not flooded.
      const percent = Math.floor(fraction * 100)
      if (percent !== lastReported) {
        lastReported = percent
        send({ type: 'loadProgress', fraction })
      }
    }
  })

  // The automatic context size accounts for the cache type, so 8-bit gets the
  // larger window by itself.
  context = await model.createContext({
    ...(request.request.contextSize === null ? {} : { contextSize: request.request.contextSize }),
    ...kvCacheOptions(request.request.kvCache)
  })
  sequence = context.getSequence()

  const info: ModelInfo = {
    path: request.request.modelPath,
    name: modelDisplayName(model.filename, model.fileInfo.metadata?.general?.name),
    trainContextSize: model.trainContextSize,
    contextSize: context.contextSize,
    gpuLayers: model.gpuLayers,
    backend: llama.gpu === false ? 'cpu' : llama.gpu,
    fileSizeBytes: model.size,
    kvCache: request.request.kvCache
  }
  send({ type: 'loaded', info })
}

function endSession(): void {
  session?.dispose({ disposeSequence: false })
  session = null
  activeChatId = null
  activeSystemPrompt = ''
  activeSystem = []
  turnMeters = []
}

/** A saved history with its system message replaced by the current one. */
function withSystemPrompt(history: ChatHistoryItem[], systemPrompt: string): ChatHistoryItem[] {
  const rest = history[0]?.type === 'system' ? history.slice(1) : history
  return [{ type: 'system', text: systemPrompt }, ...rest]
}

/**
 * The session for `target`, built when the chat or the system prompt changed.
 *
 * Switching chats restores the saved history into a fresh session; the model
 * reads it again on the next prompt, which is why the first reply after a
 * switch is slower. A changed system prompt (rules edited, folders shared) keeps
 * the conversation and only swaps the instructions at its head, so editing a
 * rule no longer costs the chat it was written for.
 */
async function buildSession(system: SystemSection[], target: ChatTarget): Promise<LlamaChatSession> {
  const systemPrompt = joinSystemSections(system)
  const sameChat = session !== null && activeChatId === target.chatId
  if (sameChat && systemPrompt === activeSystemPrompt) return session as LlamaChatSession
  if (!sequence) throw new Error('No model is loaded.')

  const carried: ChatHistoryItem[] | null = sameChat
    ? (session as LlamaChatSession).getChatHistory()
    : ((target.history as ChatHistoryItem[] | undefined) ?? null)
  const keepMeters = sameChat ? turnMeters : []

  // A replaced session is released and the sequence emptied, so the next one
  // starts from a clean window rather than on top of the old one.
  const replacing = session !== null
  endSession()
  if (replacing) await sequence.clearHistory()
  session = new LlamaChatSession({ contextSequence: sequence, systemPrompt })
  if (carried && carried.length > 1) session.setChatHistory(withSystemPrompt(carried, systemPrompt))
  activeChatId = target.chatId
  activeSystemPrompt = systemPrompt
  activeSystem = system
  turnMeters = keepMeters
  return session
}

/**
 * Messages are handled as they arrive, not one after another, so a measurement
 * and a prompt for a chat the engine is switching to can overlap. Building a
 * session awaits the sequence being cleared; a second build starting in that gap
 * would see no session, miss the history the first one carries, and start the
 * chat empty. Builds therefore run one at a time.
 */
let sessionQueue: Promise<unknown> = Promise.resolve()

function ensureSession(system: SystemSection[], target: ChatTarget): Promise<LlamaChatSession> {
  const next = sessionQueue.then(() => buildSession(system, target))
  sessionQueue = next.catch(() => undefined)
  return next
}

function reportUsage(): void {
  if (!session || !model || !context) return
  try {
    send({
      type: 'usage',
      snapshot: measureConversation(session, model, activeSystem, turnMeters, context.contextSize, activeTools)
    })
  } catch (err) {
    // Measurement is a view onto the conversation, never a reason to break it.
    console.warn('Token usage could not be measured:', (err as Error).message)
  }
}

async function runPrompt(message: MainToEngine & { type: 'prompt' }): Promise<void> {
  const controller = new AbortController()
  running.set(message.id, controller)
  let meterStart: TokenMeterState | null = null
  let turn = 0
  try {
    activeTools = message.tools
    const chat = await ensureSession(message.system, message)
    turn = chat.getChatHistory().filter((item) => item.type === 'user').length + 1
    meterStart = chat.sequence.tokenMeter.getState()
    const text = await chat.prompt(message.text, {
      functions: buildFunctions(message.id, activeTools),
      signal: controller.signal,
      stopOnAbortSignal: true,
      onTextChunk(chunk) {
        send({ type: 'chunk', id: message.id, text: chunk })
      }
    })
    send({ type: 'complete', id: message.id, text })
  } catch (err) {
    if (controller.signal.aborted) {
      send({ type: 'aborted', id: message.id })
    } else {
      send({ type: 'error', id: message.id, message: (err as Error).message, fatal: false })
    }
  } finally {
    running.delete(message.id)
    if (meterStart && sequence) {
      const used = sequence.tokenMeter.diff(meterStart)
      turnMeters.push({ turn, inputTokens: used.usedInputTokens, outputTokens: used.usedOutputTokens })
    }
    // Anything still parked for *this* prompt will never be answered now. Calls
    // belonging to another prompt are left alone.
    for (const [callId, pending] of pendingCalls) {
      if (pending.promptId !== message.id) continue
      pendingCalls.delete(callId)
      pending.settle({ ok: false, payload: 'The turn ended before this could run.' })
    }
    if (session && activeChatId === message.chatId) {
      send({ type: 'history', chatId: message.chatId, history: session.getChatHistory() })
    }
    reportUsage()
  }
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
          await disposeModel()
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
          // Never measured mid-reply: the history is changing underneath, and
          // a snapshot follows as soon as the turn ends anyway.
          if (running.size > 0 || !sequence) break
          activeTools = message.tools
          await ensureSession(message.system, message)
          reportUsage()
          break
      }
    } catch (err) {
      const id = 'id' in message ? (message.id as string) : null
      send({ type: 'error', id, message: (err as Error).message, fatal: message.type === 'load' })
    }
  })()
})

send({ type: 'ready' })
