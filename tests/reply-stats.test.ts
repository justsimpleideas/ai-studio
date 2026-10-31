import { describe, expect, it } from 'vitest'
import { ChatRecorder } from '../src/main/chat-recorder.js'
import { formatDuration, formatRate, parseReplyStats, ReplyClock, tokensPerSecond } from '../src/shared/reply-stats.js'

/** A clock the test moves by hand. */
function manual(): { clock: ReplyClock; at: (ms: number) => void } {
  let t = 0
  return { clock: new ReplyClock(() => t), at: (ms) => (t = ms) }
}

describe('reply timing', () => {
  it('times the thought from its first token to the first token of the reply', () => {
    const { clock, at } = manual()
    at(1000)
    clock.thought()
    at(5000)
    clock.thought()
    at(13000)
    clock.text()
    at(15000)
    clock.text()
    expect(clock.stats(300)).toEqual({ thinkingMs: 12000, outputTokens: 300, writingMs: 14000 })
  })

  it('counts writing from the first token, so reading the prompt is not slow writing', () => {
    const { clock, at } = manual()
    at(8000) // the prompt took eight seconds to read
    clock.text()
    at(10000)
    clock.text()
    const stats = clock.stats(41)
    expect(stats).toEqual({ outputTokens: 41, writingMs: 2000 })
    expect(stats.thinkingMs).toBeUndefined()
    expect(tokensPerSecond(stats)).toBe(20.5)
  })

  it('leaves out a tool call waiting for approval, and adds up thought on both sides of it', () => {
    const { clock, at } = manual()
    at(0)
    clock.thought()
    at(2000)
    clock.thought()
    clock.pause() // the model asked to read a file; a dialog is open
    at(62000)
    clock.resume()
    at(63000)
    clock.thought()
    at(66000)
    clock.text()
    at(67000)
    clock.text()
    // Thought 0–2s and 63–66s; writing 0–67s less the 60s wait.
    expect(clock.stats(70)).toEqual({ thinkingMs: 5000, outputTokens: 70, writingMs: 7000 })
  })

  it('does not take a wait before the first token or after the last out of the writing time', () => {
    const { clock, at } = manual()
    at(0)
    clock.pause()
    at(30000)
    clock.resume()
    at(31000)
    clock.text()
    at(33000)
    clock.text()
    clock.pause() // stopped while waiting
    at(90000)
    expect(clock.stats(10).writingMs).toBe(2000)
  })

  it('ends a thought that never reached a reply at its last token', () => {
    const { clock, at } = manual()
    at(100)
    clock.thought()
    at(4100)
    clock.thought()
    at(9000)
    expect(clock.stats(50).thinkingMs).toBe(4000)
  })

  it('gives no rate when there is too little to divide', () => {
    expect(tokensPerSecond({ outputTokens: 1, writingMs: 5000 })).toBeNull()
    expect(tokensPerSecond({ outputTokens: 40, writingMs: 10 })).toBeNull()
  })
})

describe('reply timing, shown', () => {
  it('writes durations the short way', () => {
    expect(formatDuration(400)).toBe('0.4s')
    expect(formatDuration(4200)).toBe('4s')
    expect(formatDuration(42000)).toBe('42s')
    expect(formatDuration(110000)).toBe('1m50s')
    expect(formatDuration(725000)).toBe('12m05s')
    expect(formatDuration(59600)).toBe('1m00s')
    expect(formatDuration(-5)).toBe('0.0s')
  })

  it('writes rates with a decimal only while it matters', () => {
    expect(formatRate(8.44)).toBe('8.4 tok/s')
    expect(formatRate(31.4)).toBe('31 tok/s')
  })

  it('reads back saved stats, dropping anything malformed', () => {
    expect(parseReplyStats({ thinkingMs: 5, outputTokens: 10, writingMs: 900 })).toEqual({
      thinkingMs: 5,
      outputTokens: 10,
      writingMs: 900
    })
    expect(parseReplyStats({ thinkingMs: 'x', outputTokens: 10, writingMs: 900 })).toEqual({ outputTokens: 10, writingMs: 900 })
    for (const bad of [null, undefined, 3, 'x', {}, { outputTokens: -1, writingMs: 1 }, { outputTokens: 1, writingMs: Infinity }]) {
      expect(parseReplyStats(bad)).toBeUndefined()
    }
  })
})

describe('saved chats keep the timing', () => {
  it('stores the stats the engine sent with the reply they belong to', () => {
    const rec = ChatRecorder.fresh(() => '2026-01-01T00:00:00.000Z')
    rec.userMessage('p1', 'hello')
    rec.thought('p1', 'The user greets me.')
    rec.chunk('p1', 'Hello!')
    rec.replyStats('p1', { thinkingMs: 3000, outputTokens: 20, writingMs: 4000 })
    rec.complete('p1', 'Hello!')
    expect(rec.chat.turns[1]).toEqual({
      id: 'p1',
      role: 'assistant',
      text: 'Hello!',
      thinking: 'The user greets me.',
      stats: { thinkingMs: 3000, outputTokens: 20, writingMs: 4000 }
    })
  })

  it('ignores stats for a prompt it is not recording', () => {
    const rec = ChatRecorder.fresh()
    rec.replyStats('other', { outputTokens: 1, writingMs: 1 })
    rec.userMessage('p1', 'hi')
    rec.complete('p1', 'Hi.')
    expect(rec.chat.turns[1]).toEqual({ id: 'p1', role: 'assistant', text: 'Hi.' })
  })
})
