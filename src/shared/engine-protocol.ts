import type { ChatEngine, GpuBackend, KvCacheType, ServerPreferences, SystemSection } from './config.js'
import type { GenerationPreferences, ModelTraits, SamplingValues } from './generation.js'
import type { ReplyStats } from './reply-stats.js'
import type { RawToolCall, ToolName } from './tools.js'
import type { UsageSnapshot } from './usage.js'

/**
 * The wire between the application and the process that runs the model.
 *
 * It is deliberately narrow, and deliberately one-directional about capability:
 * the engine can ask for a tool to be run, and it can stream text back. It
 * cannot read a file, write a file, or learn where the user's folders are. Every
 * one of those is a message it sends to somebody else who decides.
 *
 * That split is the reason the engine runs in its own process at all. Isolating
 * a native library that can segfault is worth something on its own, but the
 * larger point is that the component executing untrusted model output holds no
 * authority of its own to misuse.
 */

export interface ModelInfo {
  path: string
  name: string
  /** Size of the model's trained context, in tokens. */
  trainContextSize: number
  /** Context the session was actually opened with, which may be smaller. */
  contextSize: number
  /** How many layers ended up on the GPU. Zero means it is running on the CPU. */
  gpuLayers: number
  /** Name of the compute backend llama.cpp selected, e.g. "cuda" or "cpu". */
  backend: string
  /** The GPUs the backend uses, by name; empty on the CPU. */
  devices: string[]
  fileSizeBytes: number
  /** How the context memory was actually stored for this load. */
  kvCache: KvCacheType
  /** What is running the model. */
  engine: ChatEngine
  /** With a mixture-of-experts model whose experts were kept in system memory: for how many layers ('all' for every layer). */
  expertsOnCpu?: number | 'all'
  /** With a mixture-of-experts model and the built-in engine: why its experts could not be kept in system memory, when they were asked to be. */
  expertsOnCpuUnavailable?: string
  /** The CPU thread count the model was loaded with; null is the engine's default. */
  threads?: number | null
  /** What the file's header says about the model (generation.ts). */
  traits: ModelTraits
  /** The sampling the "recommended" choice resolves to for this model, so the window can show it. */
  recommendedSampling: SamplingValues
  /** How conversations are formatted for this model, shown so the user can see what the model reads. */
  template: ModelTemplate
}

export interface ModelTemplate {
  /** The library's name for the formatting it chose, e.g. "Qwen" or "JinjaTemplate". */
  wrapper: string
  /** True when the file's own chat template is the one being run; false when a built-in one for the family is used instead. */
  usesFileTemplate: boolean
  /** The chat template carried by the file, as text; null when it has none. */
  source: string | null
  /** A short sample conversation rendered the way the model receives it. */
  rendered: string
}

/**
 * What to call a loaded model: its file name, unless the file has no `.gguf`
 * name at all. Some model managers store weights as blobs named only by their
 * hash (`sha256-…`), and the name written in the model's own header says more.
 */
export function modelDisplayName(filename: string | undefined, headerName: string | undefined): string {
  const file = filename?.trim() ?? ''
  const header = headerName?.trim() ?? ''
  if (/\.gguf$/i.test(file)) return file
  return header || file || 'model'
}

export interface LoadRequest {
  modelPath: string
  backend: GpuBackend
  /** Null asks the library to fit as many layers on the GPU as it thinks will hold. */
  gpuLayers: number | null
  contextSize: number | null
  kvCache: KvCacheType
  /** Which engine process this is for; the server host also gets how to run the server. */
  engine: ChatEngine
  server: ServerPreferences
}

/**
 * Which conversation a prompt or a measurement belongs to.
 *
 * The engine holds one conversation at a time. `history` is sent only when the
 * application needs the engine to switch to `chatId`, carrying that chat's saved
 * history (empty for a new chat); while the engine is already on the chat it is
 * left out, so a long conversation is not copied across on every message.
 */
export interface ChatTarget {
  chatId: string
  history?: unknown[]
}

export type MainToEngine =
  | { type: 'load'; request: LoadRequest }
  | { type: 'unload' }
  /** `tools` are the only tools the model is told about for this prompt: the workspace's internet level decides. */
  | ({
      type: 'prompt'
      id: string
      text: string
      system: SystemSection[]
      tools: ToolName[]
      generation: GenerationPreferences
    } & ChatTarget)
  | { type: 'abort'; id: string }
  | { type: 'toolResult'; callId: string; ok: boolean; payload: unknown }
  /**
   * Measure what the conversation occupies. The system sections are sent so the
   * setup cost can be shown before the first message, when no session exists.
   */
  | ({ type: 'measureUsage'; system: SystemSection[]; tools: ToolName[] } & ChatTarget)

export type EngineToMain =
  | { type: 'ready' }
  /**
   * The engine needs to start again with these environment variables before it
   * can load (gpu-devices.ts: the integrated GPU left out). The load in flight is
   * sent again to the new process; nothing is loaded by this one.
   */
  | { type: 'restartEngine'; env: Record<string, string>; reason: string }
  /**
   * The server host started (or stopped: null) a llama-server process. The
   * application keeps the id so the server is ended with the studio even when
   * the engine process is killed outright and cannot clean up.
   */
  | { type: 'sidecar'; pid: number | null }
  | { type: 'loadProgress'; fraction: number }
  | { type: 'loaded'; info: ModelInfo }
  | { type: 'unloaded' }
  | { type: 'chunk'; id: string; text: string }
  /** A piece of the model's chain of thought, streamed apart from the reply so the window can fold it. */
  | { type: 'thought'; id: string; text: string }
  | { type: 'toolCall'; callId: string; promptId: string; call: RawToolCall }
  /** How long the reply thought and how fast it wrote; sent just before `complete` or `aborted`. */
  | { type: 'stats'; id: string; stats: ReplyStats }
  | { type: 'complete'; id: string; text: string }
  | { type: 'aborted'; id: string }
  | { type: 'error'; id: string | null; message: string; fatal: boolean }
  | { type: 'usage'; snapshot: UsageSnapshot }
  /** The conversation's full history after a turn, for the application to save. */
  | { type: 'history'; chatId: string; history: unknown[] }
