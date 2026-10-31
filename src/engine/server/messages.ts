import type { ChatHistoryItem, ChatModelFunctionCall, ChatModelResponse, ChatModelSegment } from 'node-llama-cpp'
import type { SamplingValues, ThinkingMode } from '@shared/generation.js'
import { TOOL_DESCRIPTIONS, TOOL_SCHEMAS, type ToolName } from '@shared/tools.js'

/**
 * The conversation, in the two shapes it has to take.
 *
 * The studio's saved form of a chat is the library's (node-llama-cpp chat
 * history items), whichever engine wrote it, so a conversation begun on one can
 * go on with the other. The server speaks the chat-completions shape. This module
 * converts between them, and gathers a streamed reply back into the saved form.
 */

export interface WireToolCall {
  id: string
  type: 'function'
  function: { name: string; arguments: string }
}

export type WireMessage =
  | { role: 'system' | 'user'; content: string }
  | { role: 'assistant'; content: string | null; reasoning_content?: string; tool_calls?: WireToolCall[] }
  | { role: 'tool'; tool_call_id: string; content: string }

/** The tool list in the server's shape, from the same schemas the broker validates against. */
export function toolDefinitions(names: readonly ToolName[]): object[] {
  return names.map((name) => ({
    type: 'function',
    function: { name, description: TOOL_DESCRIPTIONS[name], parameters: TOOL_SCHEMAS[name] }
  }))
}

/** A system message's text; the library allows a structured form, which is flattened to its strings. */
function systemText(text: ChatHistoryItem & { type: 'system' }): string {
  const raw = text.text
  if (typeof raw === 'string') return raw
  if (Array.isArray(raw)) return raw.map((part) => (typeof part === 'string' ? part : '')).join('')
  return ''
}

let callCounter = 0

/** Ids for tool calls recorded without one (a chat written by the built-in engine). */
function callId(call: ChatModelFunctionCall, index: number): string {
  const raw = (call as { rawCall?: unknown }).rawCall
  const id = raw && typeof raw === 'object' && 'id' in raw ? (raw as { id?: unknown }).id : undefined
  return typeof id === 'string' && id ? id : `call_${index}_${++callCounter}`
}

export function toServerMessages(history: readonly ChatHistoryItem[]): WireMessage[] {
  const out: WireMessage[] = []
  let index = 0
  for (const item of history) {
    if (item.type === 'system') {
      out.push({ role: 'system', content: systemText(item) })
      continue
    }
    if (item.type === 'user') {
      out.push({ role: 'user', content: item.text })
      continue
    }
    // A model turn may call a tool, read its result and go on writing. In the
    // server's shape that is an assistant message carrying the call, a tool
    // message with the result, and a further assistant message for the rest.
    let content = ''
    let reasoning = ''
    const flush = (calls: WireToolCall[]): void => {
      if (!content && !reasoning && calls.length === 0) return
      out.push({
        role: 'assistant',
        content: content || (calls.length ? null : ''),
        ...(reasoning ? { reasoning_content: reasoning } : {}),
        ...(calls.length ? { tool_calls: calls } : {})
      })
      content = ''
      reasoning = ''
    }
    for (const part of item.response) {
      if (typeof part === 'string') {
        content += part
      } else if (part.type === 'segment') {
        if (part.segmentType === 'thought') reasoning += part.text
        else content += part.text
      } else {
        const id = callId(part, index++)
        flush([{ id, type: 'function', function: { name: part.name, arguments: JSON.stringify(part.params ?? {}) } }])
        out.push({ role: 'tool', tool_call_id: id, content: JSON.stringify(part.result ?? null) })
      }
    }
    flush([])
  }
  return out
}

/** One piece of a streamed tool call, as the server sends them: fields arrive across several chunks. */
export interface ToolCallDelta {
  index?: number
  id?: string
  function?: { name?: string; arguments?: string }
}

/**
 * A reply as it streams in, kept in the saved form.
 *
 * Text and thought are appended as they arrive. A tool call is complete when
 * the server says the reply stopped for tool calls; its result is added when
 * the tool has run, and the model's next stretch of text follows it.
 */
export class ReplyAccumulator {
  readonly parts: ChatModelResponse['response'] = []
  private readonly calls = new Map<number, WireToolCall>()

  text(piece: string): void {
    if (!piece) return
    const last = this.parts[this.parts.length - 1]
    if (typeof last === 'string') this.parts[this.parts.length - 1] = last + piece
    else this.parts.push(piece)
  }

  thought(piece: string): void {
    if (!piece) return
    const last = this.parts[this.parts.length - 1]
    if (last && typeof last === 'object' && last.type === 'segment' && last.segmentType === 'thought' && !last.ended) {
      last.text += piece
      return
    }
    const segment: ChatModelSegment = { type: 'segment', segmentType: 'thought', text: piece, ended: false }
    this.parts.push(segment)
  }

  /** The reply moved on from thinking; the open thought is closed. */
  endThought(): void {
    for (const part of this.parts) {
      if (typeof part === 'object' && part.type === 'segment') part.ended = true
    }
  }

  toolCall(delta: ToolCallDelta): void {
    const index = delta.index ?? 0
    const call = this.calls.get(index) ?? { id: '', type: 'function' as const, function: { name: '', arguments: '' } }
    if (delta.id) call.id = delta.id
    if (delta.function?.name) call.function.name += delta.function.name
    if (delta.function?.arguments) call.function.arguments += delta.function.arguments
    this.calls.set(index, call)
  }

  /** The calls gathered so far, in order, and forget them: each is answered once. */
  takeCalls(): WireToolCall[] {
    const calls = [...this.calls.entries()].sort(([a], [b]) => a - b).map(([, call]) => call)
    this.calls.clear()
    return calls.map((call, i) => ({ ...call, id: call.id || `call_${Date.now()}_${i}` }))
  }

  /** Record a call and what came back, in the saved form. */
  result(call: WireToolCall, params: unknown, result: unknown): void {
    this.endThought()
    const part: ChatModelFunctionCall = { type: 'functionCall', name: call.function.name, params, result, rawCall: { id: call.id } as never }
    this.parts.push(part)
  }

  /** The reply's text alone, as the engine reports it when the turn ends. */
  get replyText(): string {
    return this.parts.filter((p): p is string => typeof p === 'string').join('')
  }
}

/** The arguments of a tool call as the model wrote them, or the raw text when they are not JSON. */
export function parseArguments(call: WireToolCall): unknown {
  try {
    return call.function.arguments.trim() ? JSON.parse(call.function.arguments) : {}
  } catch {
    return { _raw: call.function.arguments }
  }
}

export interface CompletionOptions {
  messages: WireMessage[]
  tools: readonly ToolName[]
  sampling: SamplingValues
  thinking: ThinkingMode
  /** Undefined leaves the model's thinking unbounded; zero is the same as off. */
  thinkingBudget: number | undefined
}

/** The body of a chat completion request, streamed. */
export function completionBody(options: CompletionOptions): Record<string, unknown> {
  const off = options.thinking === 'off' || options.thinkingBudget === 0
  return {
    messages: options.messages,
    stream: true,
    stream_options: { include_usage: true },
    temperature: options.sampling.temperature,
    top_p: options.sampling.topP,
    top_k: options.sampling.topK,
    min_p: options.sampling.minP,
    ...(options.tools.length
      ? { tools: toolDefinitions(options.tools), tool_choice: 'auto', parallel_tool_calls: false }
      : {}),
    // The thinking switch the Qwen-family templates read; other templates ignore it.
    ...(off ? { chat_template_kwargs: { enable_thinking: false } } : {}),
    ...(!off && options.thinkingBudget !== undefined ? { reasoning_budget_tokens: options.thinkingBudget } : {})
  }
}

/** What one streamed chunk of a completion carried. */
export interface ChunkReading {
  content: string
  reasoning: string
  toolCalls: ToolCallDelta[]
  finish: string | null
  /** The server's count of what it read and wrote, present on the last chunk. */
  usage: { input: number; output: number } | null
}

/** Read one chunk of the stream. Anything not understood is treated as empty rather than fatal. */
export function readChunk(raw: string): ChunkReading {
  const reading: ChunkReading = { content: '', reasoning: '', toolCalls: [], finish: null, usage: null }
  let json: unknown
  try {
    json = JSON.parse(raw)
  } catch {
    return reading
  }
  if (!json || typeof json !== 'object') return reading
  const chunk = json as {
    choices?: Array<{ delta?: Record<string, unknown>; finish_reason?: string | null }>
    usage?: { prompt_tokens?: number; completion_tokens?: number }
    timings?: { prompt_n?: number; predicted_n?: number }
  }
  const choice = chunk.choices?.[0]
  const delta = choice?.delta
  if (delta) {
    if (typeof delta.content === 'string') reading.content = delta.content
    if (typeof delta.reasoning_content === 'string') reading.reasoning = delta.reasoning_content
    if (Array.isArray(delta.tool_calls)) reading.toolCalls = delta.tool_calls as ToolCallDelta[]
  }
  if (typeof choice?.finish_reason === 'string') reading.finish = choice.finish_reason
  const timings = chunk.timings
  if (timings && (typeof timings.prompt_n === 'number' || typeof timings.predicted_n === 'number')) {
    reading.usage = { input: timings.prompt_n ?? 0, output: timings.predicted_n ?? 0 }
  } else if (chunk.usage && typeof chunk.usage.completion_tokens === 'number') {
    reading.usage = { input: chunk.usage.prompt_tokens ?? 0, output: chunk.usage.completion_tokens }
  }
  return reading
}
