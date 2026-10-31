/**
 * How long a reply took, measured in the engine as it streams.
 *
 * Two figures are shown with a reply. How long the model thought, beside the
 * folded thought. And how fast it wrote: tokens written over the time spent
 * writing them. That time starts at the first token, so reading the prompt
 * (which can take a while on a long chat) is not counted as slow writing, and it
 * leaves out the time a tool call spent waiting for the user's approval or for
 * a file, during which the model writes nothing.
 */
export interface ReplyStats {
  /** From the first token of the thought to the first token after it. Absent when the model did not think. */
  thinkingMs?: number
  /** Every token the model wrote for this reply: thought, reply and tool calls. */
  outputTokens: number
  /** From the first token to the last, less the time tool calls were waited on. */
  writingMs: number
}

/** Tokens per second, or null when there is too little to divide. */
export function tokensPerSecond(stats: ReplyStats): number | null {
  if (stats.outputTokens < 2 || stats.writingMs < 50) return null
  return (stats.outputTokens / stats.writingMs) * 1000
}

/** "0.4s", "42s", "1m50s", "12m05s": tenths only under a second. */
export function formatDuration(ms: number): string {
  const s = Math.max(0, ms) / 1000
  if (s < 0.95) return `${s.toFixed(1)}s`
  const whole = Math.round(s)
  if (whole < 60) return `${whole}s`
  return `${Math.floor(whole / 60)}m${String(whole % 60).padStart(2, '0')}s`
}

/** "8.4 tok/s", "31 tok/s". */
export function formatRate(rate: number): string {
  return rate < 10 ? `${rate.toFixed(1)} tok/s` : `${Math.round(rate)} tok/s`
}

/**
 * Timestamps for one reply, fed by the engine's stream callbacks.
 *
 * The clock is passed in so the arithmetic can be tested without waiting.
 */
export class ReplyClock {
  private firstToken: number | null = null
  private lastToken: number | null = null
  /** A model may think again after a tool call; every stretch of thought is added up. */
  private thinkingTotal = 0
  private thoughtSince: number | null = null
  private sawThought = false
  /** Waits that fell between two tokens; only those lie inside the writing time. */
  private paused = 0
  /** A finished wait not yet followed by a token. */
  private waited = 0
  private pauseStart: number | null = null

  constructor(private readonly now: () => number = () => performance.now()) {}

  private mark(): number {
    const t = this.now()
    if (this.firstToken === null) this.firstToken = t
    else this.paused += this.waited
    this.waited = 0
    this.lastToken = t
    return t
  }

  private endThought(at: number): void {
    if (this.thoughtSince === null) return
    this.thinkingTotal += Math.max(0, at - this.thoughtSince)
    this.thoughtSince = null
  }

  /** A piece of the chain of thought arrived. */
  thought(): void {
    const t = this.mark()
    this.sawThought = true
    if (this.thoughtSince === null) this.thoughtSince = t
  }

  /** A piece of the reply (anything that is not thought) arrived. */
  text(): void {
    this.endThought(this.mark())
  }

  /** A tool call is waiting for its result; the model writes nothing meanwhile. */
  pause(): void {
    if (this.pauseStart !== null) return
    this.pauseStart = this.now()
    this.endThought(this.lastToken ?? this.pauseStart)
  }

  resume(): void {
    if (this.pauseStart === null) return
    this.waited += this.now() - this.pauseStart
    this.pauseStart = null
  }

  /** The figures for a reply that wrote `outputTokens`. A thought never followed by a reply ends at its last token. */
  stats(outputTokens: number): ReplyStats {
    this.pauseStart = null
    if (this.lastToken !== null) this.endThought(this.lastToken)
    const writingMs =
      this.firstToken === null || this.lastToken === null ? 0 : Math.max(0, this.lastToken - this.firstToken - this.paused)
    return { ...(this.sawThought ? { thinkingMs: this.thinkingTotal } : {}), outputTokens, writingMs }
  }
}

/** A stats object read back from a saved chat or a message: anything malformed is dropped. */
export function parseReplyStats(raw: unknown): ReplyStats | undefined {
  if (!raw || typeof raw !== 'object') return undefined
  const r = raw as Record<string, unknown>
  const num = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= 0
  if (!num(r.outputTokens) || !num(r.writingMs)) return undefined
  return {
    ...(num(r.thinkingMs) ? { thinkingMs: r.thinkingMs } : {}),
    outputTokens: r.outputTokens,
    writingMs: r.writingMs
  }
}
