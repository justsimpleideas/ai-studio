/**
 * Where the context window went.
 *
 * On a hosted model tokens are a line on an invoice. Here they are the room the
 * model has to think in, and when it runs out the oldest part of the
 * conversation silently stops being seen. The engine measures every piece of
 * the conversation with the model's own tokenizer; this module turns those
 * measurements into something a person can act on.
 *
 * Everything here is pure so it can be tested without a model loaded.
 */

/**
 * The kinds of thing that occupy the context. The order is the display order,
 * and it is also the colour order: stacked segments sit next to each other in
 * this sequence, and the palette was checked for colour-blind separation
 * pairwise along it.
 */
export const USAGE_CATEGORIES = [
  'system',
  'tools',
  'user',
  'reply',
  'thinking',
  'toolCall',
  'toolResult',
  'format'
] as const

export type UsageCategory = (typeof USAGE_CATEGORIES)[number]

export const CATEGORY_LABEL: Record<UsageCategory, string> = {
  system: 'Instructions & rules',
  tools: 'Tool descriptions',
  user: 'Your messages',
  reply: 'Model replies',
  thinking: 'Model thinking',
  toolCall: 'Tool requests',
  toolResult: 'Tool results',
  format: 'Chat formatting'
}

/** What each kind of token is, and what can be done about it, in plain words. */
export const CATEGORY_EXPLANATION: Record<UsageCategory, string> = {
  system:
    'The base instructions plus every global rule that is switched on. They are paid again at the start of every conversation. Rules that are not visibly changing the model’s behaviour are worth removing.',
  tools:
    'Descriptions of the file tools, written into the prompt so the model knows how to call them. A fixed cost: it does not grow as the conversation does.',
  user: 'What you typed. Pasting a long document into a message costs exactly as much as the model reading it from a file.',
  reply: 'The answers the model wrote for you to read.',
  thinking:
    'Reasoning the model writes before it answers. It is generated one token at a time, so it is the slowest kind of token to produce, and it takes up room like any other text.',
  toolCall:
    'The model’s requests to use a tool. Small, unless it writes a whole file: write_file carries the complete new contents.',
  toolResult:
    'What the tools handed back, mostly file contents. A file the model reads is pasted into the conversation in full, so one large read can outweigh the rest of the chat. Asking it to list a folder first, or to read a smaller file, keeps this down.',
  format:
    'Markup the model’s chat template wraps around every message: role markers and special tokens. It grows a little with every turn.'
}

export interface UsagePiece {
  category: UsageCategory
  /**
   * The user turn this belongs to, counting from 1. Zero is the setup that
   * exists before anyone types (instructions, tool descriptions). Null is
   * spread across the whole conversation and cannot be pinned to one turn.
   */
  turn: number | null
  label: string
  /** A path or a short excerpt, so the person can recognise the item. */
  detail: string | null
  tokens: number
}

/** Work the engine did for one prompt, from llama.cpp's own token meter. */
export interface TurnMeter {
  turn: number
  /** Tokens the model had to read (prompt processing). */
  inputTokens: number
  /** Tokens the model generated. */
  outputTokens: number
}

export interface UsageSnapshot {
  pieces: UsagePiece[]
  /**
   * The exact size of the conversation as the model's chat template renders
   * it. Null when the template could not be rendered for measurement, in which
   * case the sum of the pieces stands in.
   */
  renderedTokens: number | null
  contextSize: number
  turns: TurnMeter[]
  measuredAt: string
}

export interface CategoryTotal {
  category: UsageCategory
  tokens: number
  /** Share of the conversation, 0–1. */
  share: number
}

export interface TurnBreakdown {
  turn: number
  byCategory: Partial<Record<UsageCategory, number>>
  total: number
  inputTokens: number | null
  outputTokens: number | null
}

export interface ToolTotal {
  tool: string
  calls: number
  tokens: number
}

export type InsightTone = 'info' | 'warn' | 'critical'

export interface Insight {
  tone: InsightTone
  title: string
  body: string
}

export interface UsageReport {
  used: number
  contextSize: number
  free: number
  /** Share of the window in use. Can exceed 1 once the conversation overflows. */
  fill: number
  overflowing: boolean
  categories: CategoryTotal[]
  /** Categories sorted largest first, empty ones dropped. */
  ranked: CategoryTotal[]
  topItems: (UsagePiece & { share: number })[]
  turns: TurnBreakdown[]
  tools: ToolTotal[]
  /** Average tokens a turn has added lately. Null before there is a turn. */
  growthPerTurn: number | null
  /** Rough number of further turns before the window is full. */
  turnsLeft: number | null
  setupTokens: number
  generatedTokens: number
  processedTokens: number
  insights: Insight[]
}

const TOP_ITEMS = 8
/** How many recent turns the growth rate is averaged over. */
const GROWTH_WINDOW = 3

function sumBy<T>(items: readonly T[], pick: (item: T) => number): number {
  let total = 0
  for (const item of items) total += pick(item)
  return total
}

export function formatTokens(n: number): string {
  if (n >= 10_000) return `${(n / 1000).toFixed(1).replace(/\.0$/, '')}k`
  return Math.round(n).toLocaleString('en-US')
}

export function formatShare(share: number): string {
  const percent = share * 100
  if (percent > 0 && percent < 1) return '<1%'
  return `${Math.round(percent)}%`
}

export function analyzeUsage(snapshot: UsageSnapshot): UsageReport {
  const pieces = snapshot.pieces.filter((p) => p.tokens > 0)
  const measured = sumBy(pieces, (p) => p.tokens)
  // The rendered count is the truth when it is available. Any difference from
  // the sum of the parts is the template's own markup, which the engine already
  // reports as a 'format' piece; if the two disagree the other way (a template
  // that drops old thinking, say) the rendered figure still wins.
  const used = snapshot.renderedTokens ?? measured
  const contextSize = Math.max(1, snapshot.contextSize)
  const fill = used / contextSize
  const free = Math.max(0, contextSize - used)

  const categories: CategoryTotal[] = USAGE_CATEGORIES.map((category) => {
    const tokens = sumBy(
      pieces.filter((p) => p.category === category),
      (p) => p.tokens
    )
    return { category, tokens, share: measured > 0 ? tokens / measured : 0 }
  })
  const ranked = categories.filter((c) => c.tokens > 0).sort((a, b) => b.tokens - a.tokens)

  const topItems = [...pieces]
    .filter((p) => p.category !== 'format')
    .sort((a, b) => b.tokens - a.tokens)
    .slice(0, TOP_ITEMS)
    .map((p) => ({ ...p, share: measured > 0 ? p.tokens / measured : 0 }))

  const turnNumbers = new Set<number>()
  for (const p of pieces) if (p.turn !== null && p.turn > 0) turnNumbers.add(p.turn)
  for (const m of snapshot.turns) turnNumbers.add(m.turn)
  const turns: TurnBreakdown[] = [...turnNumbers]
    .sort((a, b) => a - b)
    .map((turn) => {
      const byCategory: Partial<Record<UsageCategory, number>> = {}
      for (const p of pieces) {
        if (p.turn !== turn) continue
        byCategory[p.category] = (byCategory[p.category] ?? 0) + p.tokens
      }
      const meter = snapshot.turns.find((m) => m.turn === turn)
      return {
        turn,
        byCategory,
        total: sumBy(Object.values(byCategory), (n) => n ?? 0),
        inputTokens: meter?.inputTokens ?? null,
        outputTokens: meter?.outputTokens ?? null
      }
    })

  const toolMap = new Map<string, ToolTotal>()
  for (const p of pieces) {
    if (p.category !== 'toolCall' && p.category !== 'toolResult') continue
    const entry = toolMap.get(p.label) ?? { tool: p.label, calls: 0, tokens: 0 }
    if (p.category === 'toolCall') entry.calls += 1
    entry.tokens += p.tokens
    toolMap.set(p.label, entry)
  }
  const tools = [...toolMap.values()].sort((a, b) => b.tokens - a.tokens)

  const recent = turns.slice(-GROWTH_WINDOW)
  const growthPerTurn =
    recent.length > 0 ? sumBy(recent, (t) => t.total) / recent.length : null
  const turnsLeft =
    growthPerTurn !== null && growthPerTurn > 0 ? Math.floor(free / growthPerTurn) : null

  const setupTokens = sumBy(
    pieces.filter((p) => p.turn === 0),
    (p) => p.tokens
  )
  const generatedTokens = sumBy(snapshot.turns, (t) => t.outputTokens)
  const processedTokens = sumBy(snapshot.turns, (t) => t.inputTokens)

  const report: UsageReport = {
    used,
    contextSize,
    free,
    fill,
    overflowing: used > contextSize,
    categories,
    ranked,
    topItems,
    turns,
    tools,
    growthPerTurn,
    turnsLeft,
    setupTokens,
    generatedTokens,
    processedTokens,
    insights: []
  }
  report.insights = explain(report)
  return report
}

/** Thresholds for the plain-language findings. Exported for the tests. */
export const THRESHOLDS = {
  /** A single item this large is worth calling out on its own. */
  bigItem: 0.25,
  /** The window is getting full. */
  fillWarn: 0.7,
  /** The window is nearly full. */
  fillCritical: 0.9,
  /** Filling fast enough to mention even though it is not yet full. */
  fewTurnsLeft: 3,
  /** Setup alone takes this much of the window before anyone types. */
  setupWarn: 0.15,
  /** Hidden reasoning is this much of what the model writes. */
  thinkingHeavy: 0.5
} as const

/**
 * The findings, most important first.
 *
 * Each one says what is happening and what it means, not just a number: the
 * number is already on the chart, and the point of this view is to answer
 * "why is it so full" without the person having to know what a KV cache is.
 */
export function explain(report: UsageReport): Insight[] {
  const out: Insight[] = []
  const pct = (share: number): string => formatShare(share)
  const conversation = sumBy(report.categories, (c) => c.tokens)

  if (conversation === 0) {
    return [
      {
        tone: 'info',
        title: 'Nothing measured yet',
        body: 'Load a model and send a message. The breakdown fills in after every reply.'
      }
    ]
  }

  // Capacity first, because running out is the thing that actually hurts.
  if (report.overflowing) {
    out.push({
      tone: 'critical',
      title: 'The conversation no longer fits',
      body: `It is ${formatTokens(report.used)} tokens long, but the model can only hold ${formatTokens(report.contextSize)}. The oldest part is being dropped from what it sees, so it may forget earlier instructions or files. Starting a new chat gives it a clean window.`
    })
  } else if (report.fill >= THRESHOLDS.fillCritical) {
    out.push({
      tone: 'critical',
      title: `The context window is ${pct(report.fill)} full`,
      body: `Only ${formatTokens(report.free)} tokens are left. Once it fills, the oldest messages stop being seen. Start a new chat before anything long.`
    })
  } else if (report.fill >= THRESHOLDS.fillWarn) {
    out.push({
      tone: 'warn',
      title: `The context window is ${pct(report.fill)} full`,
      body:
        report.turnsLeft !== null
          ? `At the rate of the last few turns, roughly ${report.turnsLeft} more before it fills and older messages start to drop out.`
          : `${formatTokens(report.free)} tokens are left before older messages start to drop out.`
    })
  } else if (report.turnsLeft !== null && report.turnsLeft <= THRESHOLDS.fewTurnsLeft) {
    // Not full yet, but filling fast: a couple of big file reads can take a
    // half-empty window to overflowing in the next two turns.
    out.push({
      tone: 'warn',
      title:
        report.turnsLeft === 0
          ? 'The next turn like the last few will not fit'
          : `Room for about ${report.turnsLeft} more turn${report.turnsLeft === 1 ? '' : 's'} like the last few`,
      body: `The window is only ${pct(report.fill)} full, but recent turns have been adding about ${formatTokens(report.growthPerTurn ?? 0)} tokens each.`
    })
  }

  const top = report.ranked[0]
  if (top) {
    const second = report.ranked[1]
    const lead =
      second && top.share < 0.4
        ? `${CATEGORY_LABEL[top.category]} and ${CATEGORY_LABEL[second.category].toLowerCase()} take up the most room`
        : `Most of it is ${CATEGORY_LABEL[top.category].toLowerCase()}`
    out.push({
      tone: 'info',
      title: `${lead} (${pct(top.share)}${second && top.share < 0.4 ? ` and ${pct(second.share)}` : ''})`,
      body: CATEGORY_EXPLANATION[top.category]
    })
  }

  const biggest = report.topItems[0]
  if (biggest && biggest.share >= THRESHOLDS.bigItem && biggest.turn !== 0) {
    const what = biggest.detail ? `${biggest.label} — ${biggest.detail}` : biggest.label
    out.push({
      tone: 'warn',
      title: `One item is ${pct(biggest.share)} of the conversation on its own`,
      body: `${what}, ${formatTokens(biggest.tokens)} tokens${biggest.turn !== null ? `, in turn ${biggest.turn}` : ''}. It stays in the conversation for every turn after it, whether or not it is still needed.`
    })
  }

  const setupShare = report.setupTokens / report.contextSize
  if (setupShare >= THRESHOLDS.setupWarn) {
    out.push({
      tone: 'warn',
      title: `${pct(setupShare)} of the window is gone before you type`,
      body: `Instructions, global rules and tool descriptions come to ${formatTokens(report.setupTokens)} tokens and are paid in every conversation. Shorter rules usually work better on a local model than thorough ones.`
    })
  }

  const byCat = (c: UsageCategory): number =>
    report.categories.find((x) => x.category === c)?.tokens ?? 0
  const written = byCat('thinking') + byCat('reply') + byCat('toolCall')
  if (written > 0 && byCat('thinking') / written >= THRESHOLDS.thinkingHeavy) {
    out.push({
      tone: 'info',
      title: `${pct(byCat('thinking') / written)} of what the model writes is thinking`,
      body: 'Most of its output is reasoning you do not see in the reply. That is where the waiting goes: generated tokens are far slower than tokens it reads. Models with a switch for it can be told to think less for simple requests.'
    })
  }

  const heaviestTool = report.tools[0]
  if (heaviestTool && heaviestTool.calls > 1 && heaviestTool.tokens / conversation >= 0.2) {
    out.push({
      tone: 'info',
      title: `${heaviestTool.tool} accounts for ${pct(heaviestTool.tokens / conversation)}`,
      body: `${heaviestTool.calls} calls, averaging ${formatTokens(heaviestTool.tokens / heaviestTool.calls)} tokens each including what came back.`
    })
  }

  if (out.every((i) => i.tone === 'info') && report.turnsLeft !== null) {
    out.push({
      tone: 'info',
      title: `Plenty of room: ${pct(1 - report.fill)} free`,
      body: `At the rate of the last few turns there is space for about ${report.turnsLeft} more.`
    })
  }

  return out
}
