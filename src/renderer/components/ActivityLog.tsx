import type { ToolOutcome } from '@shared/broker-types.js'

/**
 * Everything the agent did, in order.
 *
 * Actions that ran without a prompt are marked. If someone has turned writes on
 * to automatic, the log is the only place they will find out what that cost
 * them, so an unasked action must never look the same as an approved one.
 */

function summarize(outcome: ToolOutcome): string {
  if (!outcome.ok) return outcome.error
  const result = outcome.result as Record<string, unknown>
  switch (outcome.tool) {
    case 'list_directory':
      return `${(result['entries'] as unknown[]).length} entries`
    case 'read_file':
      return `${Number(result['bytes']).toLocaleString()} bytes${result['truncated'] ? ', truncated' : ''}`
    case 'write_file':
      return result['existed'] ? 'replaced' : 'created'
    case 'edit_file':
      return 'edited'
    case 'delete_file':
      return 'moved to the recycle bin'
    default:
      return 'done'
  }
}

export function ActivityLog({ outcomes }: { outcomes: ToolOutcome[] }): React.JSX.Element {
  if (outcomes.length === 0) {
    return (
      <div className="activity">
        <p className="empty" style={{ margin: '24px 12px' }}>
          Nothing yet. Actions the model takes will appear here as they happen.
        </p>
      </div>
    )
  }

  return (
    <div className="activity">
      {outcomes.map((outcome, index) => {
        const state = outcome.ok ? 'ok' : outcome.refused ? 'refused' : 'failed'
        return (
          <div className={`event ${state}`} key={index}>
            <span className="tool">{outcome.tool}</span>
            {outcome.ok && !outcome.asked ? <span className="auto"> · ran without asking</span> : null}
            {outcome.resolved ? <span className="path">{outcome.resolved}</span> : null}
            <div className="note">{summarize(outcome)}</div>
            {!outcome.ok && outcome.hint ? <div className="note">{outcome.hint}</div> : null}
          </div>
        )
      })}
    </div>
  )
}
