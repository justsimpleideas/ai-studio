import { useCallback, useEffect, useState } from 'react'
import { INTERNET_LEVELS, type InternetLevel, type NetworkLogEntry } from '@shared/network.js'
import { DEFAULT_SEARCH, SEARCH_PROVIDERS, searchProvider, type SearchPreferences, type SearchProviderId } from '@shared/search.js'
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
  const [search, setSearch] = useState<SearchPreferences>(DEFAULT_SEARCH)
  const [urlDraft, setUrlDraft] = useState('')
  const [searchError, setSearchError] = useState<string | null>(null)
  const [log, setLog] = useState<NetworkLogEntry[]>([])

  const refreshLog = useCallback(() => void window.api.internet.log().then(setLog), [])
  const refreshKey = useCallback(() => void window.api.internet.keyStatus().then((s) => setHasKey(s.hasKey)), [])

  useEffect(() => {
    void window.api.getConfig().then((c) => {
      setSearch(c.search)
      setUrlDraft(c.search.searxngUrl ?? '')
    })
    refreshKey()
    refreshLog()
    return window.api.internet.onLogChanged(refreshLog)
  }, [refreshKey, refreshLog])

  useEffect(() => window.api.onConfigChanged((c) => setSearch(c.search)), [])

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

  // Each service keeps its own key, so the status is re-read after a switch.
  const saveSearch = async (prefs: SearchPreferences): Promise<void> => {
    setSearchError(null)
    setKeyError(null)
    try {
      const config = await window.api.internet.setSearch(prefs)
      setSearch(config.search)
      setKeyDraft('')
      refreshKey()
    } catch (err) {
      setSearchError(cleanError(err).message)
    }
  }

  const chooseProvider = (provider: SearchProviderId): void => {
    if (provider === search.provider) return
    // SearXNG is only switched to once its address is known; the choice waits on Save.
    if (provider === 'searxng' && !search.searxngUrl) {
      setSearch({ ...search, provider })
      return
    }
    void saveSearch({ ...search, provider })
  }

  const provider = searchProvider(search.provider)
  const searxngPending = search.provider === 'searxng' && urlDraft.trim() !== (search.searxngUrl ?? '')

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
        <div className="search-setup">
          <div className="row key-row">
            <div className="grow">
              <strong>Search service</strong>
              <div className="help" style={{ margin: 0 }}>
                Where web_search queries go. Only the query is sent, whichever service is chosen; every query is still checked,
                shown to you first and logged. {provider.about}
              </div>
              {searchError ? <div className="error-text">{searchError}</div> : null}
            </div>
            <select
              aria-label="Search service"
              value={search.provider}
              onChange={(e) => chooseProvider(e.target.value as SearchProviderId)}
            >
              {SEARCH_PROVIDERS.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>

          {search.provider === 'searxng' ? (
            <div className="row key-row">
              <div className="grow">
                <strong>SearXNG address</strong>
                <div className="help" style={{ margin: 0 }}>
                  The base address of your instance, for example http://localhost:8080.{' '}
                  {search.searxngUrl ? (
                    <span className="ok-text">Using {search.searxngUrl}</span>
                  ) : (
                    <span className="warn-text">No address set: searches will fail.</span>
                  )}
                </div>
              </div>
              <input
                type="url"
                value={urlDraft}
                placeholder="http://localhost:8080"
                aria-label="SearXNG address"
                autoComplete="off"
                onChange={(e) => setUrlDraft(e.target.value)}
              />
              <button
                className="primary"
                disabled={!urlDraft.trim() || !searxngPending}
                onClick={() => void saveSearch({ provider: 'searxng', searxngUrl: urlDraft.trim() })}
              >
                Save
              </button>
            </div>
          ) : null}

          {provider.needsKey ? (
            <div className="row key-row">
              <div className="grow">
                <strong>{provider.name} key</strong>
                <div className="help" style={{ margin: 0 }}>
                  Stored encrypted by the operating system and never shown again.{' '}
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
