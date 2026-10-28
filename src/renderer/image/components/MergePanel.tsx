import { useEffect, useRef, useState } from 'react'
import type { MergePlan, MergeSource, MergeState } from '@shared/image.js'
import { api } from '../api.js'
import { formatBytes } from '../format.js'

/**
 * Merging other galleries (their "outputs" folders, from a backup or another computer) into
 * this one: choose them, see what would come and what clashes, rename clashing folders (or
 * have names suggested), then copy everything in. The other folders are left as they are.
 */

const plural = (n: number, one: string, many = `${one}s`): string => `${n} ${n === 1 ? one : many}`

export function MergePanel({ enabled, state }: { enabled: boolean; state: MergeState }): React.JSX.Element {
  const [sources, setSources] = useState<MergeSource[]>([])
  const [renames, setRenames] = useState<Record<string, string>>({})
  const [plan, setPlan] = useState<MergePlan | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const request = useRef(0)
  const copying = state.state === 'copying'

  // Look again whenever the list or a name changes; a name being typed waits a moment.
  useEffect(() => {
    if (!enabled || sources.length === 0) {
      setPlan(null)
      return
    }
    const id = ++request.current
    const timer = window.setTimeout(() => {
      api
        .inspectMerge(
          sources.map((s) => s.token),
          renames
        )
        .then((p) => {
          if (id !== request.current) return
          setPlan(p)
          setError(null)
        })
        .catch((e: Error) => {
          if (id === request.current) setError(e.message)
        })
    }, 300)
    return () => window.clearTimeout(timer)
  }, [enabled, sources, renames])

  // After a merge, what was merged is here: the list starts over.
  useEffect(() => {
    if (state.state === 'done') {
      setSources([])
      setRenames({})
    }
  }, [state.state])

  const add = async (): Promise<void> => {
    setBusy(true)
    try {
      const picked = await api.pickMergeSource()
      if (picked && !sources.some((s) => s.path === picked.path)) setSources([...sources, picked])
      setError(null)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const remove = (token: string): void => {
    setSources(sources.filter((s) => s.token !== token))
    // Source keys are positions in the list; they shift, so their names start over.
    setRenames(Object.fromEntries(Object.entries(renames).filter(([key]) => key.startsWith('target:'))))
  }

  const suggest = (): void => {
    if (!plan) return
    const next = { ...renames }
    for (const group of plan.conflicts) {
      for (const entry of group.entries) if (entry.suggested !== entry.name) next[entry.key] = entry.suggested
    }
    setRenames(next)
  }

  const [cancelling, setCancelling] = useState(false)
  useEffect(() => {
    if (!copying) setCancelling(false)
  }, [copying])
  const cancel = async (): Promise<void> => {
    setCancelling(true)
    try {
      await api.cancelMerge()
    } catch (e) {
      setError((e as Error).message)
      setCancelling(false)
    }
  }

  const merge = async (): Promise<void> => {
    setBusy(true)
    try {
      await api.startMerge(
        sources.map((s) => s.token),
        renames
      )
      setError(null)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const notes: string[] = []
  if (plan) {
    const unreadable = plan.sources.reduce((n, s) => n + s.unreadable, 0)
    const noFolder = plan.sources.reduce((n, s) => n + s.no_folder, 0)
    const noLast = plan.sources.reduce((n, s) => n + s.no_last_frame, 0)
    const thumbs = plan.sources.reduce((n, s) => n + s.new_thumbs, 0)
    const unknown = [...new Set(plan.sources.flatMap((s) => s.unknown_columns))]
    if (plan.duplicates) {
      notes.push(
        `${plural(plan.duplicates, 'item is', 'items are')} already here (the same file and record), so skipped.` +
          (plan.duplicates_differing
            ? ` ${plural(plan.duplicates_differing, 'of them is', 'of them are')} a favourite or in a folder only in the other gallery; this gallery's own stays.`
            : '')
      )
    }
    for (const j of plan.joined) {
      notes.push(
        `“${j.name}” from ${j.source} is the same folder as one here: what it holds joins it` +
          (j.settings_differ ? ', and this gallery’s PIN and hidden setting stay.' : '.')
      )
    }
    if (unreadable) notes.push(`${plural(unreadable, 'item')} cannot be read (a file is missing, or its record is damaged), so left out.`)
    if (noFolder) notes.push(`${plural(noFolder, 'item')} named a folder its gallery no longer has: they come in no folder.`)
    if (noLast) notes.push(`${plural(noLast, 'video')} lack their last frame: they come, but cannot be continued.`)
    if (thumbs) notes.push(`${plural(thumbs, 'thumbnail')} are missing and will be made again.`)
    if (plan.missing_loras) {
      notes.push(`${plural(plan.missing_loras, 'item was', 'items were')} made with a LoRA this studio does not have: their settings cannot redo them exactly.`)
    }
    if (unknown.length) notes.push(`From a newer version, these are left out: ${unknown.join(', ')}.`)
  }

  return (
    <div className="card merge">
      <h2>Merge galleries</h2>
      <span className="hint">
        Bring in another gallery: an “outputs” folder from a backup, an older install or another computer. Everything is
        copied into this gallery, with its prompts and settings, favourites, folders and their PINs and hidden settings.
        The other folder is left as it is. An item is skipped only when it is exactly one already here (the same id, file and record); anything less and both are kept. No file here is ever replaced.
      </span>
      <span className="hint">
        Hidden folders are left out of the name check while they are hidden, so a folder coming in can end up with the
        same name as one of them. To have those clashes shown too, show hidden folders first (the Hidden folders card
        below).
      </span>

      {sources.map((s, i) => {
        const info = plan?.sources[i]
        return (
          <div className="merge-source" key={s.token}>
            <span className="merge-source-name">
              <strong>{info?.label ?? s.label}</strong>
              <span className="muted small truncate" title={s.path}>
                {s.path}
              </span>
              {info && (
                <span className="muted small">
                  {plural(info.images, 'image')}, {plural(info.videos, 'video')}, {plural(info.favorites, 'favourite')},{' '}
                  {plural(info.folders.length, 'folder')} · {info.new_items} new
                </span>
              )}
            </span>
            <button className="ghost small" disabled={copying} aria-label={`Remove ${s.label}`} onClick={() => remove(s.token)}>
              Remove
            </button>
          </div>
        )
      })}

      <div className="row-gap">
        <button className="ghost small" disabled={!enabled || busy || copying} onClick={() => void add()}>
          Add a gallery…
        </button>
      </div>

      {plan && plan.conflicts.length > 0 && (
        <div className="merge-conflicts" role="group" aria-label="Folders with the same name">
          <p className="error-text small">
            {plural(plan.conflicts.length, 'folder name is', 'folder names are')} in more than one gallery (names count the
            same whatever their capitals). Rename one in each, or let the studio suggest names.
          </p>
          {plan.conflicts.map((group) => (
            <div className="merge-conflict" key={group.entries.map((e) => e.key).join()}>
              <span className="field-label">“{group.name}”</span>
              {group.entries.map((entry) => (
                <label className="merge-conflict-entry" key={entry.key}>
                  <span className="muted small">
                    {entry.where}
                    {entry.hidden ? ' · hidden' : ''}
                    {entry.locked ? ' · locked' : ''}
                  </span>
                  <input
                    className="input"
                    aria-label={`Name for “${entry.name}” from ${entry.where}`}
                    value={renames[entry.key] ?? entry.name}
                    disabled={entry.locked || copying}
                    title={entry.locked ? 'Unlock it in the gallery to rename it' : undefined}
                    maxLength={60}
                    onChange={(e) => setRenames({ ...renames, [entry.key]: e.target.value })}
                  />
                </label>
              ))}
            </div>
          ))}
          <button className="ghost small" disabled={copying} onClick={suggest}>
            Rename them for me
          </button>
        </div>
      )}

      {plan && (
        <div className="merge-summary">
          <p>
            {plan.new_items
              ? `${plural(plan.new_items, 'item')} and ${plural(plan.new_folders, 'folder')} to bring in: ${formatBytes(plan.bytes)} to copy, ${formatBytes(plan.free_bytes)} free.`
              : 'Nothing new to bring in.'}
          </p>
          {notes.map((n) => (
            <p className="muted small" key={n}>
              {n}
            </p>
          ))}
        </div>
      )}

      {copying && (
        <div className="progress-card" role="status" aria-live="polite">
          <div className="progress-head">
            <span>{cancelling ? 'Stopping…' : 'Copying…'}</span>
            <span className="muted tabular">
              {state.done} / {state.total}
            </span>
          </div>
          <div className="bar">
            <div style={{ width: `${(state.done / Math.max(1, state.total)) * 100}%` }} />
          </div>
          <div className="progress-foot">
            <span className="muted small">Stopping removes what was copied so far; nothing is added.</span>
            <button className="ghost small" disabled={cancelling} onClick={() => void cancel()}>
              Cancel
            </button>
          </div>
        </div>
      )}
      {state.state === 'done' && (
        <p className="ok-text merge-result" role="status">
          Merged {plural(state.result.added, 'item')} ({plural(state.result.images, 'image')},{' '}
          {plural(state.result.videos, 'video')}, {plural(state.result.favorites, 'favourite')}) and{' '}
          {plural(state.result.folders, 'new folder')}.
          {state.result.duplicates ? ` ${plural(state.result.duplicates, 'item was', 'items were')} already here.` : ''}
          {state.result.unreadable ? ` ${plural(state.result.unreadable, 'item')} could not be read.` : ''}
        </p>
      )}
      {state.state === 'failed' && (
        <p className="error-text" role="alert">
          {state.message}
        </p>
      )}
      {plan && plan.problems.length > 0 && plan.conflicts.length === 0 && (
        <p className="error-text">{plan.problems.join(' ')}</p>
      )}
      {error && (
        <p className="error-text" role="alert">
          {error}
        </p>
      )}

      {sources.length > 0 && (
        <div className="row-gap">
          <button
            className="primary"
            disabled={!plan || plan.problems.length > 0 || plan.new_items + plan.new_folders === 0 || busy || copying}
            onClick={() => void merge()}
          >
            {plan && plan.new_items ? `Merge ${plural(plan.new_items, 'item')}` : 'Merge'}
          </button>
        </div>
      )}
    </div>
  )
}
