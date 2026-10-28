import { useCallback, useEffect, useState } from 'react'
import type { SubjectField } from '@shared/subject.js'
import {
  describeFlag,
  studioTrainingPhotoUrl,
  taskOf,
  type CaptionItem,
  type CaptionView,
  type TrainingProject
} from '@shared/training.js'
import { trainingApi } from '../api.js'
import type { Job } from '../types.js'
import { TaskProgress } from './TrainPanel.js'

const PAGE = 60
const VIEWS: { id: CaptionView; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'review', label: 'To review' },
  { id: 'excluded', label: 'Excluded' }
]

/**
 * The photos and their captions. Flagged ones come first in "To review", with the
 * reason in words; a random audit sample of unflagged ones is mixed in so the
 * error rate of what was not flagged can be measured (docs/training.md, "Review").
 */
export function CaptionsPanel({ project, version, captioning, task, onChanged }: {
  project: TrainingProject
  /** Changes when the project's captions may have changed elsewhere (a scan finished). */
  version: string
  /** A captioning model is installed. */
  captioning: boolean
  /** The project's task that is queued or running, if any. */
  task: Job | null
  onChanged: () => void
}): React.JSX.Element {
  const [view, setView] = useState<CaptionView>('all')
  const [items, setItems] = useState<CaptionItem[]>([])
  const [total, setTotal] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [editing, setEditing] = useState<string | null>(null)

  const load = useCallback(async (offset: number) => {
    try {
      const page = await trainingApi.captions(project.id, view, offset, PAGE)
      setTotal(page.total)
      setItems((prev) => (offset === 0 ? page.items : [...prev, ...page.items]))
      setError(null)
    } catch (e) {
      setError((e as Error).message)
    }
  }, [project.id, view])

  useEffect(() => {
    setItems([])
    void load(0)
  }, [load, version])

  const act = async (item: CaptionItem, action: 'accept' | 'edit' | 'exclude', edits?: Record<string, string>) => {
    try {
      const updated = await trainingApi.review(project.id, item.item, action, edits)
      // Leaving the current view (excluded from "All" stays; reviewed leaves "To review").
      const leaves = (view === 'review' && action !== 'edit') || (view === 'excluded' && action !== 'exclude')
      setItems((list) => (leaves ? list.filter((i) => i.item !== item.item) : list.map((i) => (i.item === item.item ? updated : i))))
      if (leaves) setTotal((t) => t - 1)
      setEditing(null)
      onChanged()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const describe = async () => {
    setError(null)
    try {
      await trainingApi.caption(project.id)
      onChanged()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const caps = project.captions
  const describing = task !== null && taskOf(task)?.task === 'training.caption'
  const left = caps ? caps.total - caps.described : 0
  return (
    <section className="train-section">
      <div className="train-section-head">
        <h3>Photos and captions</h3>
        <div className="segmented" role="radiogroup" aria-label="Which photos">
          {VIEWS.map((v) => (
            <button key={v.id} type="button" role="radio" aria-checked={view === v.id}
              className={view === v.id ? 'active' : ''} onClick={() => setView(v.id)}>
              {v.label}
              {v.id === 'review' && caps ? ` (${caps.to_review})` : v.id === 'excluded' && caps ? ` (${caps.excluded})` : ''}
            </button>
          ))}
        </div>
      </div>
      <div className="caption-actions">
        <button type="button" disabled={!captioning || task !== null || !caps || left === 0} onClick={() => void describe()}>
          {caps && caps.described > 0 && left > 0 ? `Describe the other ${left} photos` : 'Describe photos'}
        </button>
        {caps && caps.described > 0 && (
          <span className="muted small">
            {caps.described} of {caps.total} described · {caps.off_subject} not showing the subject, left out until you include them
          </span>
        )}
      </div>
      {describing && task && <TaskProgress job={task} label="Describing photos" />}
      {!captioning && (
        <p className="hint">
          Describing photos (and finding the ones that do not really show the subject) needs the captioning model:
          download it in Settings, under "Captioning model". Until then captions come from the folder names, and you can
          add details by hand with Edit.
        </p>
      )}
      {captioning && caps && caps.described === 0 && (
        <p className="hint">
          The captioning model looks at every photo: whether it shows the real subject, as the main thing in it, and then
          each description field. It needs the GPU to itself and takes a few seconds per photo; stopping keeps what is done.
        </p>
      )}
      {caps?.audit && (
        <p className="small">
          Audit: {caps.audit.wrong} of {caps.audit.checked} unflagged captions needed a fix — about{' '}
          {Math.round(caps.audit.rate * 100)}% (likely {Math.round(caps.audit.low * 100)}–{Math.round(caps.audit.high * 100)}%).
        </p>
      )}
      {error && <div className="error-text" role="alert">{error}</div>}
      {items.length === 0 && !error && <p className="muted small">{view === 'all' ? 'No photos yet. Scan the folder first.' : 'Nothing here.'}</p>}
      <div className="caption-grid">
        {items.map((item) => (
          <article key={item.item} className={`caption-card ${item.status} ${item.excluded ? 'excluded' : ''}`}>
            <img src={studioTrainingPhotoUrl(project.id, item.item)} alt="" loading="lazy" />
            <div className="caption-body">
              <p className="caption-text">{item.caption}</p>
              <p className="muted small truncate" title={item.path}>{item.path}</p>
              <div className="caption-tags">
                {item.status !== 'auto' && <span className="tag">{item.status}</span>}
                {item.holdout && <span className="tag" title="Never trained on; used to compare against">held out</span>}
                {item.audit && <span className="tag">audit</span>}
                {item.relevance && item.relevance.subject_shown !== 'the real thing, whole' && (
                  <span className="tag" title="What the captioning model saw">{item.relevance.subject_shown}</span>
                )}
                {item.relevance && item.relevance.subject_size !== 'the main subject' && (
                  <span className="tag" title="What the captioning model saw">{item.relevance.subject_size}</span>
                )}
                {item.flags.map((f) => <span key={f} className="tag warn-text">{describeFlag(f)}</span>)}
              </div>
              {editing === item.item ? (
                <CaptionEditor item={item} fields={project.subject.fields} onSave={(edits) => void act(item, 'edit', edits)} onCancel={() => setEditing(null)} />
              ) : (
                <div className="caption-actions">
                  {item.status !== 'accepted' && (
                    <button type="button" onClick={() => void act(item, 'accept')}>
                      {item.excluded ? 'Include' : 'Accept'}
                    </button>
                  )}
                  {project.subject.fields.length > 0 && (
                    <button type="button" onClick={() => setEditing(item.item)}>Edit</button>
                  )}
                  {!item.excluded && (
                    <button type="button" className="danger-text" onClick={() => void act(item, 'exclude')}>Exclude</button>
                  )}
                </div>
              )}
            </div>
          </article>
        ))}
      </div>
      {items.length < total && (
        <button type="button" className="link-button" onClick={() => void load(items.length)}>
          Show more ({total - items.length} left)
        </button>
      )}
    </section>
  )
}

function CaptionEditor({ item, fields, onSave, onCancel }: {
  item: CaptionItem
  fields: SubjectField[]
  onSave: (edits: Record<string, string>) => void
  onCancel: () => void
}): React.JSX.Element {
  const [draft, setDraft] = useState<Record<string, string>>(() => ({ ...item.description }))
  return (
    <form className="caption-editor" onSubmit={(e) => { e.preventDefault(); onSave(draft) }}>
      {fields.map((f) => (
        <label key={f.key} className="number-input">
          <span>{f.label}</span>
          {f.choices.length > 0 ? (
            <select className="input" value={draft[f.key] ?? 'unknown'}
              onChange={(e) => setDraft({ ...draft, [f.key]: e.target.value })}>
              <option value="unknown">unknown</option>
              {f.choices.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          ) : (
            <input className="input" maxLength={40} value={draft[f.key] === 'unknown' ? '' : draft[f.key] ?? ''}
              placeholder="unknown" onChange={(e) => setDraft({ ...draft, [f.key]: e.target.value || 'unknown' })} />
          )}
        </label>
      ))}
      <div className="caption-actions">
        <button type="submit" className="primary">Save</button>
        <button type="button" onClick={onCancel}>Cancel</button>
      </div>
    </form>
  )
}
