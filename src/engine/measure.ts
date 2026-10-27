import type {
  ChatHistoryItem,
  ChatModelFunctions,
  LlamaChatSession,
  LlamaModel,
  LlamaText
} from 'node-llama-cpp'
import type { SystemSection } from '@shared/config.js'
import { TOOL_DESCRIPTIONS, TOOL_NAMES, TOOL_SCHEMAS, type ToolName } from '@shared/tools.js'
import type { TurnMeter, UsagePiece, UsageSnapshot } from '@shared/usage.js'

/**
 * Measuring the conversation with the model's own tokenizer.
 *
 * Every figure here comes from tokenizing real text, not from a characters-per-
 * token guess: the whole point of the view is to be trusted when it says one
 * file ate a third of the window. The total is taken from the chat template's
 * own rendering of the conversation, so it is what the model actually reads.
 * The parts are measured one by one, and whatever the template adds around
 * them (role markers, special tokens) is reported as formatting rather than
 * being quietly spread over everything else.
 */

const EXCERPT = 60

/** The function documentation the model is given for these tools. */
export function functionsFor(names: readonly ToolName[]): ChatModelFunctions {
  return Object.fromEntries(
    names.map((name) => [name, { description: TOOL_DESCRIPTIONS[name], params: TOOL_SCHEMAS[name] as never }])
  )
}

export const TOOL_FUNCTIONS: ChatModelFunctions = functionsFor(TOOL_NAMES)

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

export function measureConversation(
  session: LlamaChatSession,
  model: LlamaModel,
  system: SystemSection[],
  turns: TurnMeter[],
  contextSize: number,
  tools: readonly ToolName[] = TOOL_NAMES
): UsageSnapshot {
  const functions = functionsFor(tools)
  const tokenizer = model.tokenizer
  const wrapper = session.chatWrapper
  const count = (text: string): number => (text.length === 0 ? 0 : model.tokenize(text).length)
  const countText = (text: LlamaText): number => text.tokenize(tokenizer).length

  const render = (history: readonly ChatHistoryItem[], functions: ChatModelFunctions | null): number =>
    countText(
      wrapper.generateContextState(
        functions
          ? { chatHistory: history, availableFunctions: functions, documentFunctionParams: true }
          : { chatHistory: history }
      ).contextText
    )

  const pieces: UsagePiece[] = []

  for (const section of system) {
    pieces.push({ category: 'system', turn: 0, label: section.label, detail: null, tokens: count(section.text) })
  }

  const history = session.getChatHistory()
  const systemOnly = history.filter((item) => item.type === 'system')

  // Tool descriptions are measured as the difference they make to a rendered
  // prompt, which works whichever way a template chooses to write them in. Each
  // tool's share is what removing it would save; what is left over is the
  // template's shared how-to-call-a-function preamble.
  try {
    const withAll = render(systemOnly, functions)
    const withNone = render(systemOnly, null)
    let accounted = 0
    for (const name of tools) {
      const rest = Object.fromEntries(Object.entries(functions).filter(([n]) => n !== name))
      const marginal = Math.max(0, withAll - render(systemOnly, rest))
      accounted += marginal
      pieces.push({ category: 'tools', turn: 0, label: name, detail: null, tokens: marginal })
    }
    const shared = withAll - withNone - accounted
    if (shared > 0) {
      pieces.push({ category: 'tools', turn: 0, label: 'How to call a tool', detail: null, tokens: shared })
    }
  } catch {
    // A template that cannot render functions in isolation still gets an
    // honest if rougher figure: the size of the schemas as text.
    for (const name of tools) {
      pieces.push({
        category: 'tools',
        turn: 0,
        label: name,
        detail: null,
        tokens: count(TOOL_DESCRIPTIONS[name] + JSON.stringify(TOOL_SCHEMAS[name]))
      })
    }
  }

  let turn = 0
  for (const item of history) {
    if (item.type === 'system') continue
    if (item.type === 'user') {
      turn += 1
      pieces.push({ category: 'user', turn, label: 'Your message', detail: excerpt(item.text), tokens: count(item.text) })
      continue
    }
    for (const part of item.response) {
      if (typeof part === 'string') {
        pieces.push({ category: 'reply', turn, label: 'Reply', detail: excerpt(part), tokens: count(part) })
      } else if (part.type === 'segment') {
        pieces.push({
          category: 'thinking',
          turn,
          label: part.segmentType === 'thought' ? 'Thinking' : 'Comment',
          detail: excerpt(part.text),
          tokens: count(part.text)
        })
      } else {
        let callTokens: number
        let resultTokens: number
        try {
          callTokens = countText(wrapper.generateFunctionCall(part.name, part.params))
          resultTokens = countText(wrapper.generateFunctionCallResult(part.name, part.params, part.result))
        } catch {
          callTokens = count(part.name + JSON.stringify(part.params ?? {}))
          resultTokens = count(JSON.stringify(part.result ?? null))
        }
        const detail = pathOf(part.params)
        pieces.push({ category: 'toolCall', turn, label: part.name, detail, tokens: callTokens })
        pieces.push({ category: 'toolResult', turn, label: part.name, detail, tokens: resultTokens })
      }
    }
  }

  let renderedTokens: number | null = null
  try {
    renderedTokens = render(history, functions)
    const parts = pieces.reduce((sum, p) => sum + p.tokens, 0)
    if (renderedTokens > parts) {
      pieces.push({
        category: 'format',
        turn: null,
        label: 'Chat template markup',
        detail: null,
        tokens: renderedTokens - parts
      })
    }
  } catch {
    renderedTokens = null
  }

  return { pieces, renderedTokens, contextSize, turns: [...turns], measuredAt: new Date().toISOString() }
}
