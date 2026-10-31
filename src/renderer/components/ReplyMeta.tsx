import { useEffect, useState } from 'react'
import { formatDuration, formatRate, tokensPerSecond, type ReplyStats } from '@shared/reply-stats.js'

interface Props {
  /** The chain of thought, '' when the model wrote none. */
  thinking: string
  /** Still streaming, and nothing of the reply itself written yet. */
  stillThinking: boolean
  /** When the window saw the thought begin and end; the live figure until the engine's arrives. */
  thinkingStartedAt?: number | undefined
  thinkingEndedAt?: number | undefined
  stats?: ReplyStats | undefined
}

/**
 * The line above a reply: how long the model thought and how fast it wrote,
 * with the thought itself folded away. It stays folded, even while it is being
 * written; only the arrow opens it.
 */
export function ReplyMeta({ thinking, stillThinking, thinkingStartedAt, thinkingEndedAt, stats }: Props): React.JSX.Element | null {
  const [open, setOpen] = useState(false)
  const [now, setNow] = useState(() => Date.now())

  // A ticking count while the model thinks, so a long wait reads as work.
  useEffect(() => {
    if (!stillThinking) return
    const timer = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [stillThinking])

  const rate = stats ? tokensPerSecond(stats) : null
  const thoughtMs =
    stats?.thinkingMs ??
    (thinkingStartedAt === undefined ? undefined : (thinkingEndedAt ?? (stillThinking ? now : thinkingStartedAt)) - thinkingStartedAt)

  if (!thinking && rate === null) return null

  return (
    <div className="reply-meta">
      <div className="reply-meta-line">
        {thinking ? (
          <>
            <button
              type="button"
              className={`thinking-toggle ${open ? 'open' : ''}`}
              aria-expanded={open}
              aria-label={open ? 'Hide the thinking' : 'Show the thinking'}
              title={open ? 'Hide the thinking' : 'Show the thinking'}
              onClick={() => setOpen((o) => !o)}
            >
              <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true">
                <path d="M6 4l4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
            <span className="reply-meta-label">{stillThinking ? 'Thinking' : 'Thought'}</span>
            {thoughtMs !== undefined ? <span className="reply-meta-time">{formatDuration(thoughtMs)}</span> : null}
          </>
        ) : null}
        {rate !== null ? (
          <span className="reply-meta-rate" title={`${stats?.outputTokens.toLocaleString()} tokens written`}>
            {formatRate(rate)}
          </span>
        ) : null}
      </div>
      {thinking && open ? (
        <div className={`thinking-body ${stillThinking ? 'cursor' : ''}`}>{thinking.trim()}</div>
      ) : null}
    </div>
  )
}
