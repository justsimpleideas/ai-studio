import { useEffect, useState } from 'react'
import type { EngineModelKind } from '@shared/image.js'
import { IDLE_DOWNLOAD, MODEL_CATALOG, type CatalogModel, type ModelDownloadState, type ModelId } from '@shared/setup.js'
import { api } from '../api.js'
import { downloadSummary } from '../download.js'

/**
 * The supported models of one kind, each downloadable from its own repository
 * after the user has seen where it comes from and under which license.
 */

export function useDownloadState(): ModelDownloadState {
  const [state, setState] = useState<ModelDownloadState>(IDLE_DOWNLOAD)
  useEffect(() => {
    const off = window.api.image.onDownloadState(setState)
    void window.api.image.downloadState().then(setState)
    return off
  }, [])
  return state
}

interface Props {
  kind: EngineModelKind
  /** False while the engine is not installed: downloads need its Python. */
  enabled: boolean
  /** Called once a download has finished and the engine was pointed at it. */
  onSelected: () => void
}

export function ModelDownloadPanel({ kind, enabled, onSelected }: Props): React.JSX.Element {
  const state = useDownloadState()
  const [open, setOpen] = useState<ModelId | null>(null)
  const [error, setError] = useState<string | null>(null)
  // Parts other models need are downloaded by Import a model file…, not chosen here.
  const models = MODEL_CATALOG.filter((m) => m.kind === kind && !m.part)
  const busy = state.phase === 'starting' || state.phase === 'downloading'

  useEffect(() => {
    if (state.phase === 'done' && models.some((m) => m.id === state.id)) onSelected()
    // Only when a download of this kind finishes.
  }, [state.phase, state.id])

  const start = async (id: ModelId): Promise<void> => {
    setOpen(null)
    setError(null)
    try {
      await api.downloadModel(id)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <div className="field">
      <span className="field-label">Download a model</span>
      <div className="download-list">
        {models.map((model) => {
          const mine = state.id === model.id && state.phase !== 'idle'
          return (
            <div key={model.id} className="download-row">
              <div className="model-row">
                <div>
                  <div className="strong">
                    {model.label}
                    {model.recommended && <span className="tag">Recommended</span>}
                  </div>
                  <div className="muted small">
                    {(model.bytes / 1e9).toFixed(1)} GB · {model.detail}
                  </div>
                </div>
                {!(mine && busy) && open !== model.id && (
                  <button disabled={!enabled || busy} onClick={() => setOpen(model.id)}>
                    {mine && (state.phase === 'cancelled' || state.phase === 'failed') ? 'Resume…' : 'Download…'}
                  </button>
                )}
              </div>
              {open === model.id && <Confirm model={model} onStart={() => void start(model.id)} onCancel={() => setOpen(null)} />}
              {mine && <Progress state={state} />}
            </div>
          )
        })}
      </div>
      {!enabled && <span className="hint">Install the engine first; downloads run with its Python.</span>}
      {error && state.phase !== 'failed' && <span className="error-text small">{error}</span>}
    </div>
  )
}

function Confirm({ model, onStart, onCancel }: { model: CatalogModel; onStart: () => void; onCancel: () => void }): React.JSX.Element {
  const [accepted, setAccepted] = useState(false)
  const [folder, setFolder] = useState<string | null>(null)

  useEffect(() => {
    void api.modelsFolder().then(setFolder)
  }, [])

  return (
    <div className="setup-box">
      <dl className="facts">
        <dt>Source</dt>
        <dd>
          <span className="mono">{model.repo}</span> <span className="muted">@ {model.revision.slice(0, 7)}</span>{' '}
          <button className="link-button" onClick={() => void api.openModelLink(model.id, 'source')}>
            Open page
          </button>
        </dd>
        <dt>License</dt>
        <dd>
          {model.license}{' '}
          <button className="link-button" onClick={() => void api.openModelLink(model.id, 'license')}>
            Read
          </button>
        </dd>
        <dt>Saved to</dt>
        <dd>
          <span className="mono path">{folder ? `${folder}${folder.includes('\\') ? '\\' : '/'}${model.folder}` : '…'}</span>{' '}
          <button
            className="link-button"
            onClick={() =>
              void api.chooseModelsFolder().then((picked) => {
                if (picked) setFolder(picked)
              })
            }
          >
            Change folder…
          </button>
        </dd>
      </dl>
      <p className="muted small">
        The weights are not part of this app. They are downloaded from the model&rsquo;s own repository, pinned to the
        commit above, and every file is checked against its published hash; only data files are accepted. Stopping keeps
        what is done, and downloading again resumes. Files shared with a video model already downloaded are linked, not
        downloaded twice.
      </p>
      <label className="toggle wrap">
        <input type="checkbox" checked={accepted} onChange={(e) => setAccepted(e.target.checked)} />
        <span>I have read the model&rsquo;s license and accept its terms.</span>
      </label>
      <div className="row-gap">
        <button className="primary" disabled={!accepted} onClick={onStart}>
          Start download ({(model.bytes / 1e9).toFixed(1)} GB)
        </button>
        <button className="ghost" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </div>
  )
}

function Progress({ state }: { state: ModelDownloadState }): React.JSX.Element {
  const busy = state.phase === 'starting' || state.phase === 'downloading'
  const percent = state.totalBytes > 0 ? (100 * state.doneBytes) / state.totalBytes : 0
  const summary = downloadSummary(state)
  return (
    <div className="download-progress">
      {busy && (
        <div className={`bar ${state.phase === 'starting' ? 'indeterminate' : ''}`}>
          <div style={state.phase === 'starting' ? undefined : { width: `${percent.toFixed(1)}%` }} />
        </div>
      )}
      <div className="progress-foot">
        <span className={state.phase === 'failed' ? 'error-text small pre' : state.phase === 'done' ? 'ok-text small' : 'muted small tabular'}>
          {summary}
        </span>
        {busy && <button onClick={() => void api.cancelDownload()}>Stop</button>}
      </div>
    </div>
  )
}
