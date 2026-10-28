import { useCallback, useEffect, useMemo, useState } from 'react'
import { folderLayout, previewCaption, type Subject } from '@shared/subject.js'
import { taskOf, type LoraItem, type TrainingProject } from '@shared/training.js'
import { trainingApi } from '../api.js'
import type { Job } from '../types.js'
import { CaptionsPanel } from './CaptionsPanel.js'
import { CollectPanel } from './CollectPanel.js'
import { SubjectEditor, presetLabel } from './SubjectEditor.js'
import { TaskProgress, TrainPanel } from './TrainPanel.js'

const ISSUE_LABEL: Record<string, string> = {
  unreadable: 'unreadable',
  too_small: 'too small (under 512 px)',
  duplicate_of: 'duplicates',
  missing_facts: 'missing a required folder level'
}

/**
 * The Train tab: projects on the left, the chosen one on the right — its photo
 * folder, the scan's findings, the captions to review, and training runs.
 */
export function TrainView({ jobs, connected, captioning, library, onLibraryChanged }: {
  jobs: Record<string, Job>
  connected: boolean
  /** A captioning model is installed. */
  captioning: boolean
  library: LoraItem[]
  onLibraryChanged: () => void
}): React.JSX.Element {
  const [projects, setProjects] = useState<TrainingProject[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [newName, setNewName] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [collecting, setCollecting] = useState(false)
  const [presets, setPresets] = useState<Subject[]>([])
  const [newPreset, setNewPreset] = useState('general')

  const refresh = useCallback(async () => {
    try {
      const list = await trainingApi.projects()
      setProjects(list)
      setSelected((s) => (s && list.some((p) => p.id === s) ? s : list[0]?.id ?? null))
    } catch (e) {
      setError((e as Error).message)
    }
  }, [])

  // Tasks per project, and a signature that changes whenever one of them finishes.
  const tasks = useMemo(() => Object.values(jobs).filter((j) => taskOf(j) !== null), [jobs])
  const finished = tasks.filter((j) => ['done', 'failed', 'cancelled'].includes(j.status)).map((j) => `${j.id}:${j.status}`).join(',')
  useEffect(() => {
    if (connected) void refresh()
  }, [connected, refresh, finished])
  useEffect(() => {
    if (connected) void trainingApi.subjectPresets().then(setPresets).catch((e: Error) => setError(e.message))
  }, [connected])

  const project = projects.find((p) => p.id === selected) ?? null
  const active = project
    ? tasks.filter((j) => taskOf(j)?.project === project.id && (j.status === 'queued' || j.status === 'running'))
        .sort((a, b) => a.created - b.created)[0] ?? null
    : null
  const lastFailed = project
    ? tasks.filter((j) => taskOf(j)?.project === project.id && j.status === 'failed').sort((a, b) => (b.finished ?? 0) - (a.finished ?? 0))[0]
    : undefined

  const create = async () => {
    setError(null)
    try {
      const subject = presets.find((p) => p.preset === newPreset) ?? presets[0]
      if (!subject) throw new Error('The subject presets are not loaded yet.')
      const made = await trainingApi.createProject(newName.trim(), subject)
      if (!made) return
      setNewName('')
      await refresh()
      setSelected(made.id)
      await trainingApi.scan(made.id)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const run = async (action: () => Promise<unknown>) => {
    setError(null)
    try {
      await action()
      await refresh()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  if (!connected) return <div className="empty-gallery muted">Training opens once the image engine is running.</div>

  return (
    <div className="train-view">
      <aside className="panel train-projects">
        <span className="field-label">Projects</span>
        <ul className="folder-list">
          {projects.map((p) => (
            <li key={p.id}>
              <button type="button" className={`folder-item ${p.id === selected && !collecting ? 'active' : ''}`}
                onClick={() => { setSelected(p.id); setCollecting(false) }}>
                <span className="folder-name truncate">{p.name}</span>
                <span className="folder-count">{p.dataset?.usable ?? '–'}</span>
              </button>
            </li>
          ))}
        </ul>
        <form className="train-new" onSubmit={(e) => { e.preventDefault(); if (newName.trim()) void create() }}>
          <input className="input" maxLength={60} placeholder="New project name" value={newName}
            onChange={(e) => setNewName(e.target.value)} aria-label="New project name" />
          <select className="input" value={newPreset} aria-label="Subject to start from"
            onChange={(e) => setNewPreset(e.target.value)}>
            {presets.map((p) => <option key={p.preset} value={p.preset}>About: {presetLabel(p.preset)}</option>)}
          </select>
          <button type="submit" disabled={!newName.trim() || presets.length === 0}>Choose photo folder…</button>
        </form>
        <button type="button" className={`folder-item ${collecting ? 'active' : ''}`} onClick={() => setCollecting(true)}>
          <span className="folder-name">Collect photos from the web…</span>
        </button>
        <p className="hint">
          Put photos in folders, one level per name, e.g. <span className="mono">Brand/Model/photo.jpg</span>. What the
          levels are called, and what each photo is described by, is up to you: edit the project's subject. Photos are
          only read, never changed.
        </p>
        <LoraLibrary library={library} onChanged={onLibraryChanged} />
      </aside>

      <section className="train-main">
        {error && <div className="error-text" role="alert">{error}</div>}
        {collecting && (
          <CollectPanel presets={presets} onProject={(id) => {
            setCollecting(false)
            setSelected(id)
            void refresh()
          }} />
        )}
        {!collecting && !project && <p className="muted">Create a project: give it a name, then choose the folder with your photos.</p>}
        {!collecting && project && (
          <>
            <div className="train-section">
              <div className="train-section-head">
                <h2 className="truncate">{project.name}</h2>
                <span className="grow" />
                <button type="button" disabled={active !== null} onClick={() => void run(() => trainingApi.scan(project.id))}>
                  Rescan folder
                </button>
                <button type="button" className="danger-text" disabled={active !== null}
                  onClick={() => void run(() => trainingApi.deleteProject(project.id))}>
                  Delete project
                </button>
              </div>
              <p className="muted small path" title={project.photo_root}>{project.photo_root}</p>
              {project.dataset ? (
                <p className="small">
                  {project.dataset.total} photos found, <strong>{project.dataset.usable} usable</strong> ({project.dataset.train} to
                  train on, {project.dataset.holdout} held out to compare against).
                  {Object.entries(project.dataset.issues).map(([k, n]) => ` ${n} ${ISSUE_LABEL[k] ?? k}.`).join('')}
                </p>
              ) : (
                <p className="muted small">Not scanned yet.</p>
              )}
              {project.warnings.length > 0 && (
                <ul className="small warn-text">{project.warnings.slice(0, 5).map((w) => <li key={w}>{w}</li>)}</ul>
              )}
              {active && taskOf(active)?.task === 'training.scan' && <TaskProgress job={active} label="Scanning photos" />}
              {lastFailed && !active && <div className="error-text">Last task failed: {lastFailed.error}</div>}
            </div>
            <SubjectSection key={project.id} project={project} presets={presets} busy={active !== null}
              onSaved={() => void run(() => trainingApi.scan(project.id))} />
            <TrainPanel project={project} task={active} onChanged={() => void refresh()} onLibraryChanged={onLibraryChanged} />
            <CaptionsPanel project={project} version={finished} captioning={captioning} task={active}
              onChanged={() => void refresh()} />
          </>
        )}
      </section>
    </div>
  )
}

/** What the project is about; editing it rescans, since the folder levels give the facts. */
function SubjectSection({ project, presets, busy, onSaved }: {
  project: TrainingProject
  presets: Subject[]
  busy: boolean
  onSaved: () => void
}): React.JSX.Element {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState<{ subject: Subject | null; error: string | null }>({ subject: project.subject, error: null })
  const [error, setError] = useState<string | null>(null)
  const s = project.subject

  const save = async () => {
    if (!draft.subject) return
    setError(null)
    try {
      await trainingApi.setSubject(project.id, draft.subject)
      setEditing(false)
      onSaved()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <div className="train-section subject-section">
      <div className="train-section-head">
        <h3>Subject</h3>
        <span className="grow" />
        {!editing && <button type="button" disabled={busy} onClick={() => setEditing(true)}>Edit subject</button>}
      </div>
      {editing ? (
        <>
          <SubjectEditor initial={s} presets={presets} onChange={(subject, err) => setDraft({ subject, error: err })} />
          {error && <div className="error-text" role="alert">{error}</div>}
          <div className="caption-actions">
            <button type="button" className="primary" disabled={!draft.subject || busy} onClick={() => void save()}>
              Save and rescan
            </button>
            <button type="button" onClick={() => setEditing(false)}>Cancel</button>
          </div>
        </>
      ) : (
        <>
          <p className="small">
            {s.noun ? <strong>{s.noun}</strong> : <span className="warn-text">Not said what the photos show yet</span>}
            {' · '}folders <span className="mono">{folderLayout(s)}</span>
            {' · '}{s.fields.length} description field{s.fields.length === 1 ? '' : 's'}
            {s.trigger && <> · trigger word <span className="mono">{s.trigger}</span></>}
          </p>
          <p className="hint">Caption example: <span className="mono">{previewCaption(s)}</span></p>
        </>
      )}
    </div>
  )
}

function LoraLibrary({ library, onChanged }: { library: LoraItem[]; onChanged: () => void }): React.JSX.Element {
  const [error, setError] = useState<string | null>(null)
  const [renaming, setRenaming] = useState<{ id: string; name: string } | null>(null)
  const act = async (fn: () => Promise<unknown>) => {
    setError(null)
    try {
      await fn()
      onChanged()
    } catch (e) {
      setError((e as Error).message)
    }
  }
  return (
    <div className="lora-library">
      <span className="field-label">LoRA library</span>
      {library.length === 0 && <p className="hint">Empty. After training, choose a checkpoint and “Use as LoRA”.</p>}
      <ul className="folder-list">
        {library.map((l) => (
          <li key={l.id} className="folder-row" title={`rank ${l.rank} · ${(l.size / 2 ** 20).toFixed(0)} MB · ${l.sha256.slice(0, 12)}${
            l.origin.photo_licences?.['cc-by-sa'] ? ` · trained on ${l.origin.photo_licences['cc-by-sa']} CC BY-SA photos` : ''}`}>
            {renaming?.id === l.id ? (
              <form className="grow" onSubmit={(e) => {
                e.preventDefault()
                const name = renaming.name.trim()
                setRenaming(null)
                if (name && name !== l.name) void act(() => trainingApi.renameLora(l.id, name))
              }}>
                <input className="input folder-name-input" autoFocus maxLength={60} value={renaming.name} aria-label="LoRA name"
                  onChange={(e) => setRenaming({ id: l.id, name: e.target.value })}
                  onKeyDown={(e) => e.key === 'Escape' && setRenaming(null)} />
              </form>
            ) : (
              <>
                <span className="folder-name truncate">{l.name}</span>
                <span className="folder-count">{l.origin.step ? `step ${l.origin.step}` : ''}</span>
              </>
            )}
            <button type="button" className="icon-button" title="Rename" onClick={() => setRenaming({ id: l.id, name: l.name })}>✎</button>
            <button type="button" className="icon-button danger-text" title="Remove from the library"
              onClick={() => void act(() => trainingApi.deleteLora(l.id))}>×</button>
          </li>
        ))}
      </ul>
      {error && <div className="error-text">{error}</div>}
    </div>
  )
}
