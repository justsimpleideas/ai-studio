import { describe, expect, it } from 'vitest'
import {
  composeSystemPrompt,
  composeSystemSections,
  DEFAULT_CONFIG,
  joinSystemSections,
  type GlobalRule
} from '../src/shared/config.js'
import {
  analyzeUsage,
  formatShare,
  formatTokens,
  type UsagePiece,
  type UsageSnapshot
} from '../src/shared/usage.js'

/**
 * The token usage analysis.
 *
 * The figures come from the model's tokenizer in the engine; what is tested
 * here is everything done with them afterwards, which is where a wrong sum or a
 * misleading sentence would come from.
 */

function piece(overrides: Partial<UsagePiece> & Pick<UsagePiece, 'category' | 'tokens'>): UsagePiece {
  return { turn: 1, label: overrides.category, detail: null, ...overrides }
}

function snapshot(pieces: UsagePiece[], extra: Partial<UsageSnapshot> = {}): UsageSnapshot {
  return {
    pieces,
    renderedTokens: null,
    contextSize: 8192,
    turns: [],
    measuredAt: '2026-01-01T00:00:00.000Z',
    ...extra
  }
}

const SETUP: UsagePiece[] = [
  piece({ category: 'system', turn: 0, tokens: 120, label: 'Base instructions' }),
  piece({ category: 'tools', turn: 0, tokens: 300, label: 'read_file' })
]

describe('system prompt sections', () => {
  const rules: GlobalRule[] = [
    { id: 'a', title: 'First', body: '  Be brief.  ', enabled: true },
    { id: 'b', title: 'Off', body: 'Never shown.', enabled: false },
    { id: 'c', title: '', body: 'Read before editing.', enabled: true },
    { id: 'd', title: 'Blank', body: '   ', enabled: true }
  ]

  it('produces the same prompt text as before sections existed', () => {
    const prompt = composeSystemPrompt(rules)
    const [preamble] = prompt.split('\n')
    expect(prompt).toBe([preamble, '', '- Be brief.', '- Read before editing.'].join('\n'))
  })

  it('is just the preamble when no rule is on', () => {
    const prompt = composeSystemPrompt([])
    expect(prompt).not.toContain('\n')
    expect(prompt.length).toBeGreaterThan(0)
  })

  it('labels one section per active rule, skipping disabled and empty ones', () => {
    const sections = composeSystemSections(rules)
    expect(sections.map((s) => s.label)).toEqual(['Base instructions', 'First', 'Untitled rule'])
    expect(joinSystemSections(sections)).toBe(composeSystemPrompt(rules))
  })

  it('keeps the defaults unchanged', () => {
    expect(composeSystemSections(DEFAULT_CONFIG.rules)).toHaveLength(3)
  })
})

describe('analyzeUsage', () => {
  it('prefers the rendered total over the sum of the parts', () => {
    const report = analyzeUsage(
      snapshot([...SETUP, piece({ category: 'user', tokens: 80 })], { renderedTokens: 540 })
    )
    expect(report.used).toBe(540)
    expect(report.free).toBe(8192 - 540)
  })

  it('falls back to the sum when the template could not be rendered', () => {
    const report = analyzeUsage(snapshot([...SETUP, piece({ category: 'user', tokens: 80 })]))
    expect(report.used).toBe(500)
  })

  it('ranks categories largest first and drops empty ones', () => {
    const report = analyzeUsage(
      snapshot([
        ...SETUP,
        piece({ category: 'toolResult', tokens: 2000, label: 'read_file' }),
        piece({ category: 'reply', tokens: 50 })
      ])
    )
    expect(report.ranked.map((c) => c.category)).toEqual(['toolResult', 'tools', 'system', 'reply'])
    expect(report.categories).toHaveLength(8)
    const shares = report.ranked.reduce((sum, c) => sum + c.share, 0)
    expect(shares).toBeCloseTo(1)
  })

  it('attributes tokens to the turn they arrived in', () => {
    const report = analyzeUsage(
      snapshot(
        [
          ...SETUP,
          piece({ category: 'user', turn: 1, tokens: 10 }),
          piece({ category: 'reply', turn: 1, tokens: 40 }),
          piece({ category: 'user', turn: 2, tokens: 12 }),
          piece({ category: 'toolResult', turn: 2, tokens: 900 }),
          piece({ category: 'format', turn: null, tokens: 30 })
        ],
        { turns: [{ turn: 2, inputTokens: 950, outputTokens: 60 }] }
      )
    )
    expect(report.turns.map((t) => [t.turn, t.total])).toEqual([
      [1, 50],
      [2, 912]
    ])
    // Setup and formatting belong to no single turn.
    expect(report.turns.every((t) => t.byCategory.system === undefined)).toBe(true)
    expect(report.turns[1]?.inputTokens).toBe(950)
    expect(report.turns[0]?.inputTokens).toBeNull()
    expect(report.setupTokens).toBe(420)
  })

  it('groups tool calls and their results by tool', () => {
    const report = analyzeUsage(
      snapshot([
        piece({ category: 'toolCall', tokens: 20, label: 'read_file' }),
        piece({ category: 'toolResult', tokens: 1000, label: 'read_file' }),
        piece({ category: 'toolCall', tokens: 20, label: 'read_file' }),
        piece({ category: 'toolResult', tokens: 500, label: 'read_file' }),
        piece({ category: 'toolCall', tokens: 15, label: 'list_directory' }),
        piece({ category: 'toolResult', tokens: 60, label: 'list_directory' })
      ])
    )
    expect(report.tools).toEqual([
      { tool: 'read_file', calls: 2, tokens: 1540 },
      { tool: 'list_directory', calls: 1, tokens: 75 }
    ])
  })

  it('estimates turns left from the recent growth rate', () => {
    const report = analyzeUsage(
      snapshot(
        [
          piece({ category: 'user', turn: 1, tokens: 100 }),
          piece({ category: 'user', turn: 2, tokens: 200 }),
          piece({ category: 'user', turn: 3, tokens: 300 })
        ],
        { contextSize: 1000 }
      )
    )
    expect(report.growthPerTurn).toBe(200)
    expect(report.turnsLeft).toBe(2)
  })

  it('leaves formatting out of the largest-items list', () => {
    const report = analyzeUsage(
      snapshot([
        piece({ category: 'format', turn: null, tokens: 5000 }),
        piece({ category: 'user', tokens: 10 })
      ])
    )
    expect(report.topItems.map((i) => i.category)).toEqual(['user'])
  })

  it('ignores zero-token pieces', () => {
    const report = analyzeUsage(snapshot([piece({ category: 'reply', tokens: 0 })]))
    expect(report.used).toBe(0)
    expect(report.turns).toEqual([])
  })
})

describe('insights', () => {
  it('says there is nothing to report on an empty conversation', () => {
    const [only] = analyzeUsage(snapshot([])).insights
    expect(only?.title).toBe('Nothing measured yet')
  })

  it('flags an overflowing conversation first, as critical', () => {
    const report = analyzeUsage(
      snapshot([...SETUP, piece({ category: 'toolResult', tokens: 9000 })], { contextSize: 4096 })
    )
    expect(report.overflowing).toBe(true)
    expect(report.insights[0]?.tone).toBe('critical')
    expect(report.insights[0]?.title).toMatch(/no longer fits/)
  })

  it('warns when the window is getting full', () => {
    const report = analyzeUsage(
      snapshot([piece({ category: 'user', tokens: 750 })], { contextSize: 1000 })
    )
    expect(report.insights[0]?.tone).toBe('warn')
    expect(report.insights[0]?.title).toBe('The context window is 75% full')
  })

  it('warns about a window filling fast even while it is half empty', () => {
    const report = analyzeUsage(
      snapshot(
        [
          piece({ category: 'user', turn: 1, tokens: 1000 }),
          piece({ category: 'toolResult', turn: 2, tokens: 3000 })
        ],
        { contextSize: 8192 }
      )
    )
    expect(report.fill).toBeLessThan(0.7)
    expect(report.turnsLeft).toBe(2)
    expect(report.insights[0]).toMatchObject({ tone: 'warn', title: 'Room for about 2 more turns like the last few' })
  })

  it('names the dominant kind and explains it', () => {
    const report = analyzeUsage(
      snapshot([...SETUP, piece({ category: 'toolResult', tokens: 3000, label: 'read_file' })], {
        contextSize: 32768
      })
    )
    const lead = report.insights.find((i) => i.title.startsWith('Most of it is'))
    expect(lead?.title).toMatch(/tool results/)
    expect(lead?.body).toMatch(/file/)
  })

  it('calls out one oversized item with its path and turn', () => {
    const report = analyzeUsage(
      snapshot(
        [
          ...SETUP,
          piece({ category: 'toolResult', turn: 3, tokens: 3000, label: 'read_file', detail: 'C:\\work\\big.log' })
        ],
        { contextSize: 32768 }
      )
    )
    const big = report.insights.find((i) => i.title.startsWith('One item is'))
    expect(big?.body).toContain('C:\\work\\big.log')
    expect(big?.body).toContain('turn 3')
  })

  it('does not call setup an oversized item', () => {
    const report = analyzeUsage(snapshot(SETUP, { contextSize: 32768 }))
    expect(report.insights.some((i) => i.title.startsWith('One item is'))).toBe(false)
  })

  it('warns when setup alone eats a large share of a small window', () => {
    const report = analyzeUsage(snapshot(SETUP, { contextSize: 2048 }))
    expect(report.insights.some((i) => /before you type/.test(i.title))).toBe(true)
  })

  it('notices when most of the output is thinking', () => {
    const report = analyzeUsage(
      snapshot(
        [
          piece({ category: 'thinking', tokens: 800 }),
          piece({ category: 'reply', tokens: 200 })
        ],
        { contextSize: 32768 }
      )
    )
    expect(report.insights.some((i) => /is thinking$/.test(i.title))).toBe(true)
  })

  it('reassures when there is nothing to worry about', () => {
    const report = analyzeUsage(
      snapshot(
        [
          piece({ category: 'system', turn: 0, tokens: 100 }),
          piece({ category: 'user', turn: 1, tokens: 50 }),
          piece({ category: 'reply', turn: 1, tokens: 60 })
        ],
        { contextSize: 32768 }
      )
    )
    expect(report.insights.every((i) => i.tone === 'info')).toBe(true)
    expect(report.insights.at(-1)?.title).toMatch(/^Plenty of room/)
  })
})

describe('formatting', () => {
  it('abbreviates large counts and keeps small ones exact', () => {
    expect(formatTokens(950)).toBe('950')
    expect(formatTokens(9876)).toBe('9,876')
    expect(formatTokens(12_000)).toBe('12k')
    expect(formatTokens(32_768)).toBe('32.8k')
  })

  it('never rounds a real share down to zero', () => {
    expect(formatShare(0.004)).toBe('<1%')
    expect(formatShare(0)).toBe('0%')
    expect(formatShare(0.456)).toBe('46%')
  })
})
