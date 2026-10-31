import { randomUUID } from 'node:crypto'
import {
  type ChatHistoryItem,
  type ChatWrapper,
  defineChatSessionFunction,
  getLlama,
  Llama,
  LlamaChatSession,
  LlamaLogLevel,
  readGgufFileInfo,
  resolveChatWrapper,
  type ChatSessionModelFunctions,
  type LlamaContext,
  type LlamaContextSequence,
  type LlamaModel,
  type TokenMeterState
} from 'node-llama-cpp'
import {
  modelDisplayName,
  type ChatTarget,
  type EngineToMain,
  type MainToEngine,
  type ModelInfo,
  type ModelTemplate
} from '@shared/engine-protocol.js'
import { joinSystemSections, type GpuBackend, type SystemSection } from '@shared/config.js'
import {
  DEFAULT_CONTEXT_CAP,
  DEFAULT_GENERATION,
  recommendedSampling,
  resolveSampling,
  type GenerationPreferences,
  type ModelTraits,
  type ThinkingMode
} from '@shared/generation.js'
import { TOOL_DESCRIPTIONS, TOOL_NAMES, TOOL_SCHEMAS, type ToolName } from '@shared/tools.js'
import type { TurnMeter } from '@shared/usage.js'
import { ReplyClock } from '@shared/reply-stats.js'
import { kvCacheOptions, kvModelOptions } from './kv-cache.js'
import { measureConversation } from './measure.js'
import { discreteDeviceIndices } from '@shared/gpu-devices.js'
import { addonPlacesTensors, planExperts } from './expert-placement.js'
import { describeModel, loadPlan } from './model-traits.js'
import { thoughtBudget, wrapperSettingsFor } from './thinking.js'

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
/** What the loaded model's header says about it; null until a model is loaded. */
let traits: ModelTraits | null = null
/** How replies are generated, as last asked for by the application. */
let generation: GenerationPreferences = DEFAULT_GENERATION
/** Which of the application's chats the session holds. */
let activeChatId: string | null = null
let activeSystemPrompt = ''
let activeSystem: SystemSection[] = []
/** The thinking mode the session's chat wrapper was built for. */
let activeThinking: ThinkingMode | null = null
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

/** Per prompt in flight, the timestamps its speed and thinking time are read from. */
const clocks = new Map<string, ReplyClock>()

function requestTool(promptId: string, name: ToolName, args: unknown): Promise<unknown> {
  const callId = randomUUID()
  // Waiting on approval or a file is not the model writing slowly.
  clocks.get(promptId)?.pause()
  return new Promise((resolve) => {
    pendingCalls.set(callId, {
      promptId,
      // A refusal is handed back to the model as an ordinary result rather than
      // thrown. The model needs to learn that this door is closed so it can try
      // something else; killing the turn would just leave the user staring at a
      // half-finished answer.
      settle: ({ ok, payload }) => {
        clocks.get(promptId)?.resume()
        resolve(ok ? payload : { error: payload })
      }
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
  traits = null
  sequence = null
  await context?.dispose().catch(() => undefined)
  context = null
  await model?.dispose().catch(() => undefined)
  model = null
}

/** llama.cpp's lines about memory placement: "CUDA0 model buffer size = …", "offloaded 41/41 layers to GPU", KV and compute buffers. */
const PLACEMENT_LOG = /(model|KV|compute) buffer size =|offloaded \d+\/\d+ layers to GPU/

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
      ...(wanted === 'auto' ? {} : { gpu: wanted === 'cpu' ? false : wanted }),
      // Warnings and errors as before, plus the few information lines that say
      // where the weights, the context and the scratch memory went, for the
      // engine log. The rest of the information level is a metadata dump.
      logLevel: LlamaLogLevel.info,
      logger(level, message) {
        if (level === LlamaLogLevel.fatal || level === LlamaLogLevel.error || level === LlamaLogLevel.warn) {
          Llama.defaultConsoleLogger(level, message)
        } else if (PLACEMENT_LOG.test(message)) {
          console.log(`[llama] ${message.trim()}`)
        }
      }
    })
    activeBackend = wanted
  }

  // Vulkan on a laptop sees the integrated GPU too, counts its share of system
  // memory as VRAM and spreads the model across both. Which devices Vulkan uses
  // is fixed when the process starts, so the engine asks to be started again
  // with the integrated GPU left out, and loads nothing in this process.
  const devices = llama.gpu === false ? [] : await llama.getGpuDeviceNames()
  // After such a restart, the devices now visible must be the ones asked for.
  // Vulkan numbers devices in the order it enumerates them, which has been seen
  // to change between processes in some setups; a mismatch is refused rather
  // than loading the model onto a GPU nobody chose.
  const expected = process.env['STUDIO_VK_EXPECTED']
  if (llama.gpu === 'vulkan' && expected !== undefined && JSON.stringify(devices) !== expected) {
    throw new Error(
      `The GPU picked for the chat model (${devices.join(', ') || 'none'}) is not the one intended (${(JSON.parse(expected) as string[]).join(', ')}). Restart the studio and load the model again.`
    )
  }
  if (llama.gpu === 'vulkan' && process.env['GGML_VK_VISIBLE_DEVICES'] === undefined) {
    const keep = discreteDeviceIndices(devices)
    if (keep) {
      const left = devices.filter((_, i) => !keep.includes(i))
      console.log(`[engine] vulkan devices ${JSON.stringify(devices)}; restarting with only ${JSON.stringify(keep.map((i) => devices[i]))}`)
      send({
        type: 'restartEngine',
        env: { GGML_VK_VISIBLE_DEVICES: keep.join(','), STUDIO_VK_EXPECTED: JSON.stringify(keep.map((i) => devices[i])) },
        reason: `Using ${keep.map((i) => devices[i]).join(', ')} for the chat model; ${left.join(', ')} shares system memory and was left out.`
      })
      return
    }
  }

  // The header is read first, on its own: the layer plan needs the trained
  // context size before the weights are placed (model-traits.ts, loadPlan).
  const header = await readGgufFileInfo(request.request.modelPath, { readTensorInfo: false })
  const trained = header.architectureMetadata.context_length ?? DEFAULT_CONTEXT_CAP
  const plan = loadPlan(trained, { contextSize: request.request.contextSize, gpuLayers: request.request.gpuLayers })
  const shape = header.architectureMetadata as { expert_count?: number; block_count?: number }
  const experts = planExperts({
    experts: shape.expert_count,
    layers: shape.block_count,
    expertsOnCpu: request.request.server.expertsOnCpu,
    backend: llama.gpu,
    addonPlaces: addonPlacesTensors(llama)
  })
  const threads = request.request.server.threads
  // For the engine log (no path): what was asked of the library, so a failed load can be read back.
  console.log(
    `[engine] load: backend ${llama.gpu === false ? 'cpu' : llama.gpu} ${JSON.stringify(devices)}, architecture ${header.metadata.general.architecture}, trained context ${trained}, context ${plan.contextSize}, gpu layers ${JSON.stringify(experts.kind === 'place' ? (request.request.gpuLayers ?? 'max') : plan.gpuLayers)}, kv ${request.request.kvCache}, experts on CPU ${experts.kind === 'place' ? experts.label : experts.kind}, threads ${threads ?? 'default'}`
  )
  if (experts.kind === 'unavailable') console.log(`[engine] ${experts.reason}`)

  let lastReported = -1
  model = await llama.loadModel({
    modelPath: request.request.modelPath,
    // With the experts kept in system memory, what is left for the GPU is every
    // layer's attention and shared weights. The library's fit cannot plan for
    // that (it sizes whole layers), so its memory checks are skipped and every
    // layer goes to the GPU; if even that does not fit, llama.cpp's own
    // allocation fails the load with an error rather than a guess.
    ...(experts.kind === 'place'
      ? { gpuLayers: request.request.gpuLayers ?? 'max', ignoreMemorySafetyChecks: true, tensorBufferOverrides: experts.overrides }
      : { gpuLayers: plan.gpuLayers }),
    ...kvModelOptions(request.request.kvCache),
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

  console.log(`[engine] weights loaded: ${model.gpuLayers} layers on the GPU, mmap ${model.useMmap}`)
  context = await model.createContext({
    contextSize: plan.contextSize,
    ...kvCacheOptions(request.request.kvCache),
    ...(experts.kind === 'place' ? { ignoreMemorySafetyChecks: true } : {}),
    ...(threads === null ? {} : { threads })
  })
  console.log(`[engine] context ready: ${context.contextSize} tokens`)
  sequence = context.getSequence()

  // The chat template, resolved the way a session will resolve it, is what says
  // whether the model can think; the thinking mode does not change that.
  traits = describeModel(
    {
      architecture: model.fileInfo.metadata.general.architecture,
      architectureMetadata: model.fileInfo.architectureMetadata,
      dominantTensorType: model.fileInsights.dominantTensorType,
      totalParameters: model.fileInsights.totalParameters
    },
    wrapperFor(generation.thinking).settings
  )

  const info: ModelInfo = {
    path: request.request.modelPath,
    name: modelDisplayName(model.filename, model.fileInfo.metadata?.general?.name),
    trainContextSize: model.trainContextSize,
    contextSize: context.contextSize,
    gpuLayers: model.gpuLayers,
    backend: llama.gpu === false ? 'cpu' : llama.gpu,
    devices,
    fileSizeBytes: model.size,
    kvCache: request.request.kvCache,
    engine: 'builtin',
    ...(experts.kind === 'place' ? { expertsOnCpu: experts.label } : {}),
    ...(experts.kind === 'unavailable' ? { expertsOnCpuUnavailable: experts.reason } : {}),
    threads,
    traits,
    recommendedSampling: recommendedSampling(traits),
    template: describeTemplate(wrapperFor(generation.thinking), model.fileInfo.metadata.tokenizer?.chat_template ?? null)
  }
  send({ type: 'loaded', info })
}

/** A fixed little conversation, rendered so the user can see the exact text the model is given. */
const SAMPLE_HISTORY: ChatHistoryItem[] = [
  { type: 'system', text: 'You are a careful assistant.' },
  { type: 'user', text: 'What is the capital of France?' },
  { type: 'model', response: ['Paris.'] },
  { type: 'user', text: 'And of Italy?' }
]

function describeTemplate(wrapper: ChatWrapper, source: string | null): ModelTemplate {
  let rendered: string
  try {
    rendered = wrapper.generateContextState({ chatHistory: SAMPLE_HISTORY }).contextText.toString()
  } catch (err) {
    rendered = `(could not render: ${(err as Error).message})`
  }
  return {
    wrapper: wrapper.wrapperName,
    usesFileTemplate: wrapper.wrapperName === 'JinjaTemplate',
    source,
    rendered: rendered.length > 6000 ? `${rendered.slice(0, 6000)}\n…` : rendered
  }
}

/** The model's chat wrapper, with the thinking switch set where the wrapper has one. */
function wrapperFor(mode: ThinkingMode): ChatWrapper {
  if (!model) throw new Error('No model is loaded.')
  return resolveChatWrapper(model, { customWrapperSettings: wrapperSettingsFor(mode), warningLogs: false })
}

function endSession(): void {
  session?.dispose({ disposeSequence: false })
  session = null
  activeChatId = null
  activeSystemPrompt = ''
  activeSystem = []
  activeThinking = null
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
 * rule no longer costs the chat it was written for. The thinking mode lives in
 * the chat wrapper, so changing it is handled the same way.
 */
async function buildSession(system: SystemSection[], target: ChatTarget): Promise<LlamaChatSession> {
  const systemPrompt = joinSystemSections(system)
  const thinking = generation.thinking
  const sameChat = session !== null && activeChatId === target.chatId
  if (sameChat && systemPrompt === activeSystemPrompt && thinking === activeThinking) return session as LlamaChatSession
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
  session = new LlamaChatSession({ contextSequence: sequence, systemPrompt, chatWrapper: wrapperFor(thinking) })
  if (carried && carried.length > 1) session.setChatHistory(withSystemPrompt(carried, systemPrompt))
  activeChatId = target.chatId
  activeSystemPrompt = systemPrompt
  activeSystem = system
  activeThinking = thinking
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

/** Sent just before the reply's end, so the window and the saved chat have it when the reply settles. */
function sendStats(id: string, clock: ReplyClock, meterStart: TokenMeterState | null): void {
  if (!meterStart || !sequence) return
  const outputTokens = sequence.tokenMeter.diff(meterStart).usedOutputTokens
  send({ type: 'stats', id, stats: clock.stats(outputTokens) })
}

async function runPrompt(message: MainToEngine & { type: 'prompt' }): Promise<void> {
  const controller = new AbortController()
  running.set(message.id, controller)
  const clock = new ReplyClock()
  clocks.set(message.id, clock)
  let meterStart: TokenMeterState | null = null
  let turn = 0
  try {
    activeTools = message.tools
    generation = message.generation
    const chat = await ensureSession(message.system, message)
    turn = chat.getChatHistory().filter((item) => item.type === 'user').length + 1
    meterStart = chat.sequence.tokenMeter.getState()
    // Without these the library samples greedily, and a thinking model run
    // that way tends to repeat itself; the numbers are the model card's.
    const sampling = resolveSampling(generation.sampling, traits ?? { architecture: '', thinking: false })
    const budget = thoughtBudget(generation.thinking, generation.thinkingBudget)
    const text = await chat.prompt(message.text, {
      // No tools (strict mode) means no functions at all: nothing documented
      // in the prompt, nothing in the grammar for the model to call.
      ...(activeTools.length === 0 ? {} : { functions: buildFunctions(message.id, activeTools) }),
      signal: controller.signal,
      stopOnAbortSignal: true,
      temperature: sampling.temperature,
      topP: sampling.topP,
      topK: sampling.topK,
      minP: sampling.minP,
      ...(budget === undefined ? {} : { budgets: { thoughtTokens: budget } }),
      // The thought goes to the window apart from the reply, so it can be shown
      // folded while it is written and the reply is not left looking stalled.
      onResponseChunk(chunk) {
        if (chunk.type === 'segment') {
          clock.thought()
          send({ type: 'thought', id: message.id, text: chunk.text })
        } else {
          clock.text()
          send({ type: 'chunk', id: message.id, text: chunk.text })
        }
      }
    })
    sendStats(message.id, clock, meterStart)
    send({ type: 'complete', id: message.id, text })
  } catch (err) {
    if (controller.signal.aborted) {
      sendStats(message.id, clock, meterStart)
      send({ type: 'aborted', id: message.id })
    } else {
      send({ type: 'error', id: message.id, message: (err as Error).message, fatal: false })
    }
  } finally {
    running.delete(message.id)
    clocks.delete(message.id)
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
      // Also to stderr, which the supervisor keeps in the engine log with the
      // library's own output around it.
      console.error(`[engine] ${message.type} failed:`, (err as Error).stack ?? (err as Error).message)
      send({ type: 'error', id, message: (err as Error).message, fatal: message.type === 'load' })
    }
  })()
})

send({ type: 'ready' })
