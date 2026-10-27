import type { GpuBackend, KvCacheType, SystemSection } from './config.js'
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
  fileSizeBytes: number
  /** How the context memory was actually stored for this load. */
  kvCache: KvCacheType
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
  | ({ type: 'prompt'; id: string; text: string; system: SystemSection[]; tools: ToolName[] } & ChatTarget)
  | { type: 'abort'; id: string }
  | { type: 'toolResult'; callId: string; ok: boolean; payload: unknown }
  /**
   * Measure what the conversation occupies. The system sections are sent so the
   * setup cost can be shown before the first message, when no session exists.
   */
  | ({ type: 'measureUsage'; system: SystemSection[]; tools: ToolName[] } & ChatTarget)

export type EngineToMain =
  | { type: 'ready' }
  | { type: 'loadProgress'; fraction: number }
  | { type: 'loaded'; info: ModelInfo }
  | { type: 'unloaded' }
  | { type: 'chunk'; id: string; text: string }
  | { type: 'toolCall'; callId: string; promptId: string; call: RawToolCall }
  | { type: 'complete'; id: string; text: string }
  | { type: 'aborted'; id: string }
  | { type: 'error'; id: string | null; message: string; fatal: boolean }
  | { type: 'usage'; snapshot: UsageSnapshot }
  /** The conversation's full history after a turn, for the application to save. */
  | { type: 'history'; chatId: string; history: unknown[] }
