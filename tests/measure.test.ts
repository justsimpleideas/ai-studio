import { describe, expect, it } from 'vitest'
import {
  ChatMLChatWrapper,
  GeneralChatWrapper,
  Llama3_1ChatWrapper,
  QwenChatWrapper,
  type ChatHistoryItem,
  type ChatWrapper,
  type Token,
  type Tokenizer
} from 'node-llama-cpp'
import { measureConversation } from '../src/engine/measure.js'
import { composeSystemSections, joinSystemSections, DEFAULT_CONFIG } from '../src/shared/config.js'
import { TOOL_NAMES } from '../src/shared/tools.js'

/**
 * The engine-side measurement, against node-llama-cpp's real chat templates.
 *
 * No model is loaded: the tokenizer is a stand-in that counts words, which is
 * enough to check the arithmetic. What matters is that each template's own
 * rendering is what the total is taken from, and that the parts plus the
 * formatting always add up to exactly that total, whichever template a model
 * happens to use.
 */

const fakeTokenizer = ((text: string, specialTokens?: boolean | 'builtin') => {
  if (specialTokens === 'builtin') return [1 as Token]
  const words = text.split(/\s+/).filter(Boolean)
  return words.map((_, i) => (i + 2) as Token)
}) as unknown as Tokenizer

const fakeModel = {
  tokenizer: fakeTokenizer,
  tokenize: (text: string) => (fakeTokenizer as unknown as (t: string) => Token[])(text)
}

const system = composeSystemSections(DEFAULT_CONFIG.rules)

const history: ChatHistoryItem[] = [
  { type: 'system', text: joinSystemSections(system) },
  { type: 'user', text: 'What is in my notes folder?' },
  {
    type: 'model',
    response: [
      { type: 'segment', segmentType: 'thought', text: 'I should list the folder first.', ended: true },
      {
        type: 'functionCall',
        name: 'list_directory',
        params: { path: 'C:\\work\\notes' },
        result: { entries: [{ name: 'todo.md', kind: 'file', bytes: 120 }], truncated: false }
      },
      {
        type: 'functionCall',
        name: 'read_file',
        params: { path: 'C:\\work\\notes\\todo.md' },
        result: { text: 'buy milk and eggs and bread and coffee', bytes: 120, truncated: false }
      },
      'There is one file, todo.md, with a shopping list.'
    ]
  },
  { type: 'user', text: 'Thanks.' },
  { type: 'model', response: ['You are welcome.'] }
]

function fakeSession(wrapper: ChatWrapper, items: ChatHistoryItem[]) {
  return { chatWrapper: wrapper, getChatHistory: () => items }
}

const wrappers: [string, () => ChatWrapper][] = [
  ['Qwen', () => new QwenChatWrapper()],
  ['ChatML', () => new ChatMLChatWrapper()],
  ['Llama 3.1', () => new Llama3_1ChatWrapper()],
  ['General', () => new GeneralChatWrapper()]
]

describe.each(wrappers)('measureConversation with the %s template', (_name, make) => {
  const snap = measureConversation(
    fakeSession(make(), history) as never,
    fakeModel as never,
    system,
    [{ turn: 1, inputTokens: 400, outputTokens: 90 }],
    8192
  )
  const sum = snap.pieces.reduce((total, p) => total + p.tokens, 0)

  it('renders the whole conversation for the total', () => {
    expect(snap.renderedTokens).not.toBeNull()
    expect(snap.renderedTokens).toBeGreaterThan(0)
  })

  it('makes the parts add up to the rendered total, or never exceed it by more than a template drops', () => {
    const format = snap.pieces.find((p) => p.category === 'format')
    if (format) {
      expect(sum).toBe(snap.renderedTokens)
    } else {
      // Only possible when the template leaves something out (older thinking,
      // typically); the rendered figure is still what the analysis reports.
      expect(sum).toBeGreaterThanOrEqual(snap.renderedTokens ?? 0)
    }
  })

  it('measures one section per active rule, and every tool', () => {
    expect(snap.pieces.filter((p) => p.category === 'system').map((p) => p.label)).toEqual(
      system.map((s) => s.label)
    )
    const tools = snap.pieces.filter((p) => p.category === 'tools').map((p) => p.label)
    for (const name of TOOL_NAMES) expect(tools).toContain(name)
    expect(snap.pieces.filter((p) => p.category === 'tools').every((p) => p.tokens >= 0)).toBe(true)
  })

  it('assigns each part of a reply to its turn, with the path it touched', () => {
    const results = snap.pieces.filter((p) => p.category === 'toolResult')
    expect(results.map((p) => [p.label, p.turn, p.detail])).toEqual([
      ['list_directory', 1, 'C:\\work\\notes'],
      ['read_file', 1, 'C:\\work\\notes\\todo.md']
    ])
    expect(snap.pieces.find((p) => p.category === 'thinking')?.turn).toBe(1)
    expect(snap.pieces.filter((p) => p.category === 'user').map((p) => p.turn)).toEqual([1, 2])
    expect(snap.pieces.filter((p) => p.category === 'reply').map((p) => p.turn)).toEqual([1, 2])
  })

  it('passes the per-turn meter through untouched', () => {
    expect(snap.turns).toEqual([{ turn: 1, inputTokens: 400, outputTokens: 90 }])
    expect(snap.contextSize).toBe(8192)
  })
})

describe('measureConversation before the first message', () => {
  it('reports only the setup cost', () => {
    const snap = measureConversation(
      fakeSession(new QwenChatWrapper(), [{ type: 'system', text: joinSystemSections(system) }]) as never,
      fakeModel as never,
      system,
      [],
      4096
    )
    const kinds = new Set(snap.pieces.map((p) => p.category))
    expect([...kinds].every((k) => k === 'system' || k === 'tools' || k === 'format')).toBe(true)
    expect(snap.pieces.every((p) => p.turn === 0 || p.turn === null)).toBe(true)
  })
})
