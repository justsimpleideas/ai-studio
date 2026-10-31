import type { ChatHistoryItem } from 'node-llama-cpp'
import type { SystemSection } from '@shared/config.js'
import { TOOL_DESCRIPTIONS, TOOL_SCHEMAS, type ToolName } from '@shared/tools.js'
import type { TurnMeter, UsagePiece, UsageSnapshot } from '@shared/usage.js'
import { toServerMessages, toolDefinitions, type WireMessage } from './messages.js'

/**
 * Measuring the conversation through the server.
 *
 * The same figures as the built-in engine's (engine/measure.ts), from the same
 * two sources: the model's own tokenizer, and the chat template's own rendering
 * of the conversation. Here both live in the server, reached as /tokenize and
 * /apply-template. The arithmetic is kept apart from the transport so it can be
 * tested with a client that answers from a table.
 */

/** What the measurement needs from the server. */
export interface MeasureClient {
  /** How many tokens a piece of text is. */
  tokenCount(text: string): Promise<number>
  /** The conversation as the template renders it for the model. */
  render(messages: WireMessage[], tools: object[] | null): Promise<string>
}

const EXCERPT = 60

function excerpt(text: string): string {
  const flat = text.replace(/\s+/g, ' ').trim()
  return flat.length > EXCERPT ? `${flat.slice(0, EXCERPT - 1)}…` : flat
}

function pathOf(params: unknown): string | null {
  if (params && typeof params === 'object' && 'path' in params) {
    const value = (params as { path: unknown }).path
    if (typeof value === 'string') return value
  }
  return null
}

export async function measureRemote(
  client: MeasureClient,
  history: readonly ChatHistoryItem[],
  system: SystemSection[],
  turns: TurnMeter[],
  contextSize: number,
  tools: readonly ToolName[]
): Promise<UsageSnapshot> {
  const count = (text: string): Promise<number> => (text.length === 0 ? Promise.resolve(0) : client.tokenCount(text))
  const rendered = async (items: readonly ChatHistoryItem[], defs: object[] | null): Promise<number> =>
    count(await client.render(toServerMessages(items), defs))

  const pieces: UsagePiece[] = []
  for (const section of system) {
    pieces.push({ category: 'system', turn: 0, label: section.label, detail: null, tokens: await count(section.text) })
  }

  const systemOnly = history.filter((item) => item.type === 'system')
  const defs = toolDefinitions(tools)
  try {
    const withAll = await rendered(systemOnly, defs)
    const withNone = await rendered(systemOnly, null)
    let accounted = 0
    for (const name of tools) {
      const rest = toolDefinitions(tools.filter((n) => n !== name))
      const marginal = Math.max(0, withAll - (await rendered(systemOnly, rest)))
      accounted += marginal
      pieces.push({ category: 'tools', turn: 0, label: name, detail: null, tokens: marginal })
    }
    const shared = withAll - withNone - accounted
    if (shared > 0) pieces.push({ category: 'tools', turn: 0, label: 'How to call a tool', detail: null, tokens: shared })
  } catch {
    for (const name of tools) {
      pieces.push({
        category: 'tools',
        turn: 0,
        label: name,
        detail: null,
        tokens: await count(TOOL_DESCRIPTIONS[name] + JSON.stringify(TOOL_SCHEMAS[name]))
      })
    }
  }

  let turn = 0
  for (const item of history) {
    if (item.type === 'system') continue
    if (item.type === 'user') {
      turn += 1
      pieces.push({ category: 'user', turn, label: 'Your message', detail: excerpt(item.text), tokens: await count(item.text) })
      continue
    }
    for (const part of item.response) {
      if (typeof part === 'string') {
        pieces.push({ category: 'reply', turn, label: 'Reply', detail: excerpt(part), tokens: await count(part) })
      } else if (part.type === 'segment') {
        pieces.push({
          category: 'thinking',
          turn,
          label: part.segmentType === 'thought' ? 'Thinking' : 'Comment',
          detail: excerpt(part.text),
          tokens: await count(part.text)
        })
      } else {
        const detail = pathOf(part.params)
        pieces.push({
          category: 'toolCall',
          turn,
          label: part.name,
          detail,
          tokens: await count(part.name + JSON.stringify(part.params ?? {}))
        })
        pieces.push({ category: 'toolResult', turn, label: part.name, detail, tokens: await count(JSON.stringify(part.result ?? null)) })
      }
    }
  }

  let renderedTokens: number | null = null
  try {
    renderedTokens = await rendered(history, tools.length ? defs : null)
    const parts = pieces.reduce((sum, p) => sum + p.tokens, 0)
    if (renderedTokens > parts) {
      pieces.push({ category: 'format', turn: null, label: 'Chat template markup', detail: null, tokens: renderedTokens - parts })
    }
  } catch {
    renderedTokens = null
  }

  return { pieces, renderedTokens, contextSize, turns: [...turns], measuredAt: new Date().toISOString() }
}
