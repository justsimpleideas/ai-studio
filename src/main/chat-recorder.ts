import { randomUUID } from 'node:crypto'
import type { ToolOutcome } from '@shared/broker-types.js'
import type { Chat, ChatView } from '@shared/workspace.js'
import { chatTitle } from '@shared/workspace.js'

/**
 * Builds the saved form of a conversation from the same engine events the
 * window renders, so a reopened chat reads exactly as it did live.
 *
 * Turn ids match the ones the window gives them ("u-<prompt id>" for the
 * message, the prompt id for the reply), which is what lets a reopened chat and
 * a live one be the same list.
 */
export class ChatRecorder {
  /** Reply text streamed so far, per prompt still running. */
  private readonly partial = new Map<string, string>()

  constructor(
    readonly chat: Chat,
    private readonly now: () => string = () => new Date().toISOString()
  ) {}

  static fresh(now: () => string = () => new Date().toISOString()): ChatRecorder {
    const at = now()
    return new ChatRecorder(
      {
        meta: { id: randomUUID(), title: 'New chat', createdAt: at, updatedAt: at, messageCount: 0 },
        turns: [],
        outcomes: [],
        history: null
      },
      now
    )
  }

  /** True once the chat has something worth keeping. Empty chats are never written. */
  get worthSaving(): boolean {
    return this.chat.meta.messageCount > 0
  }

  /** Whether a reply to this prompt is being recorded here. */
  owns(promptId: string): boolean {
    return this.partial.has(promptId)
  }

  view(): ChatView {
    return { meta: this.chat.meta, turns: this.chat.turns, outcomes: this.chat.outcomes }
  }

  private touch(): void {
    this.chat.meta.updatedAt = this.now()
  }

  userMessage(promptId: string, text: string): void {
    if (this.chat.meta.messageCount === 0) this.chat.meta.title = chatTitle(text)
    this.chat.meta.messageCount += 1
    this.chat.turns.push({ id: `u-${promptId}`, role: 'user', text })
    this.partial.set(promptId, '')
    this.touch()
  }

  chunk(promptId: string, text: string): void {
    const sofar = this.partial.get(promptId)
    if (sofar !== undefined) this.partial.set(promptId, sofar + text)
  }

  /**
   * The streamed text is what the window showed, so it is what is kept. The
   * engine's final text is only a fallback for a reply that streamed nothing.
   */
  complete(promptId: string, text: string): void {
    const sofar = this.partial.get(promptId)
    this.finish(promptId, sofar ? sofar : text)
  }

  aborted(promptId: string): void {
    const sofar = this.partial.get(promptId)
    if (sofar === undefined) return
    this.finish(promptId, `${sofar}\n\n[stopped]`)
  }

  error(promptId: string | null, message: string): void {
    if (promptId !== null && this.partial.has(promptId)) {
      const sofar = this.partial.get(promptId) as string
      if (sofar) this.finish(promptId, sofar)
      else this.partial.delete(promptId)
    }
    this.chat.turns.push({ id: `err-${randomUUID()}`, role: 'system', text: message })
    this.touch()
  }

  outcome(outcome: ToolOutcome): void {
    this.chat.outcomes.push(outcome)
    this.touch()
  }

  history(history: unknown[]): void {
    this.chat.history = history
  }

  private finish(promptId: string, text: string): void {
    if (!this.partial.has(promptId)) return
    this.partial.delete(promptId)
    this.chat.turns.push({ id: promptId, role: 'assistant', text })
    this.touch()
  }
}
