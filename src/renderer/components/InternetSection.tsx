import { useCallback, useEffect, useState } from 'react'
import { INTERNET_LEVELS, SEARCH_HOST, type InternetLevel, type NetworkLogEntry } from '@shared/network.js'
import { cleanError } from '../image/api.js'

/**
 * Internet access for the open workspace: the level, the search key, and the
 * log of every request the agent asked for.
 *
 * Offline is the default, and leaving it asks for confirmation in plain words:
 * a request to the internet is the one action here that can carry something
 * off the computer, so turning it on should be a decision, not a click.
 */

const WARNING: Record<Exclude<InternetLevel, 'offline'>, string> = {
  search:
    'Allow the agent to use the internet in this folder?\n\n' +
    '• Search queries are sent to the search service, and pages are requested from the sites the agent opens.\n' +
    '• Text from your files is never put in a request; requests that would carry it are refused. Every request is shown to you first (unless you set internet requests to run automatically) and written to the internet log.\n' +
    '• Web pages can contain text written to mislead the agent. Your protected paths and the change history still apply.\n\n' +
    'You can switch back to Offline at any time.',
  download:
    'Allow the agent to search the web AND download files into this folder?\n\n' +
    '• Everything about search applies, and files from the web can be saved here as new files (never over existing ones, never programs or scripts, at most 50 MB).\n' +
    '• Downloads are recorded in the change history and can be reverted.\n' +
    '• A file from the internet can be harmful even when it is not a program. Only allow this for sources you trust.\n\n' +
    'You can switch back to Offline at any time.'
}

function when(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

const TOOL_LABEL: Record<NetworkLogEntry['tool'], string> = {
  web_search: 'Search',
  fetch_page: 'Page',
  download_file: 'Download'
}

export function InternetSection({
  level,
  onSetLevel
}: {
  level: InternetLevel
  onSetLevel: (level: InternetLevel) => void
}): React.JSX.Element {
  const [hasKey, setHasKey] = useState<boolean | null>(null)
  const [keyDraft, setKeyDraft] = useState('')
  const [keyError, setKeyError] = useState<string | null>(null)
  const [log, setLog] = useState<NetworkLogEntry[]>([])

  const refreshLog = useCallback(() => void window.api.internet.log().then(setLog), [])

  useEffect(() => {
    void window.api.internet.keyStatus().then((s) => setHasKey(s.hasKey))
    refreshLog()
    return window.api.internet.onLogChanged(refreshLog)
  }, [refreshLog])

  const choose = (next: InternetLevel): void => {
    if (next === level) return
    if (next !== 'offline' && !window.confirm(WARNING[next])) return
    onSetLevel(next)
  }

  const saveKey = async (key: string | null): Promise<void> => {
    setKeyError(null)
    try {
      const status = await window.api.internet.setKey(key)
      setHasKey(status.hasKey)
      setKeyDraft('')
    } catch (err) {
      setKeyError(cleanError(err).message)
    }
  }

  return (
    <section className="internet">
      <h3>Internet</h3>
      <p className="help">
        Whether the agent may use the internet in this folder. Requests are made by the app, never by
        the model itself; each one is checked, shown to you before it is sent, and logged below.
      </p>
      <div className="level-list" role="radiogroup" aria-label="Internet access">
        {INTERNET_LEVELS.map((l) => (
          <label key={l.level} className={`level ${level === l.level ? 'active' : ''} level-${l.level}`}>
            <input
              type="radio"
              name="internet-level"
              checked={level === l.level}
              onChange={() => choose(l.level)}
            />
            <span>
              <strong>{l.title}</strong>
              {l.level === 'offline' ? <span className="badge">default</span> : null}
              <span className="help block">{l.help}</span>
            </span>
          </label>
        ))}
      </div>

      {level !== 'offline' ? (
        <div className="row key-row">
          <div className="grow">
            <strong>Search key</strong>
            <div className="help" style={{ margin: 0 }}>
              Web search sends queries to the search API at {SEARCH_HOST}, which needs a key of your own.
              It is stored encrypted by the operating system and never shown again.{' '}
              {hasKey ? <span className="ok-text">A key is set.</span> : <span className="warn-text">No key set: searches will fail.</span>}
            </div>
            {keyError ? <div className="error-text">{keyError}</div> : null}
          </div>
          <input
            type="password"
            value={keyDraft}
            placeholder={hasKey ? 'Replace key' : 'Paste key'}
            aria-label="Search API key"
            autoComplete="off"
            onChange={(e) => setKeyDraft(e.target.value)}
          />
          <button className="primary" disabled={!keyDraft.trim()} onClick={() => void saveKey(keyDraft)}>
            Save
          </button>
          {hasKey ? <button onClick={() => void saveKey(null)}>Remove</button> : null}
        </div>
      ) : null}

      <h4>Internet log</h4>
      {log.length === 0 ? (
        <p className="help">No requests yet.</p>
      ) : (
        <div className="net-log">
          {log.slice(0, 100).map((e) => (
            <div className={`net-row net-${e.outcome}`} key={e.id}>
              <span className="net-time">{when(e.at)}</span>
              <span className="net-tool">{TOOL_LABEL[e.tool]}</span>
              <code className="net-what" title={e.sent}>
                {e.sent}
              </code>
              <span className="net-outcome">{e.outcome}</span>
              <span className="net-detail" title={e.detail}>
                {e.savedTo ? `→ ${e.savedTo.split(/[\\/]/).pop()} · ` : ''}
                {e.detail}
              </span>
            </div>
          ))}
        </div>
      )}
    </section>
  )
}
