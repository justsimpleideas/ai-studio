import { useState } from 'react'
import { studioImageUrl } from '@shared/image.js'
import {
  COMPARE_SIZES,
  DEFAULT_TRAIN_SETTINGS,
  MAX_COMPARE_CHECKPOINTS,
  MAX_COMPARE_PROMPTS,
  TARGET_SIDES,
  type LossPoint,
  type TrainSettings,
  type TrainingProject,
  type TrainingRun
} from '@shared/training.js'
import { api, trainingApi } from '../api.js'
import type { Job } from '../types.js'
import { LossChart } from './LossChart.js'

const SETTINGS_KEY = 'lis.train.v1'

function loadSettings(): TrainSettings {
  try {
    const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}') as Partial<TrainSettings>
    const out = { ...DEFAULT_TRAIN_SETTINGS }
    for (const key of Object.keys(DEFAULT_TRAIN_SETTINGS) as (keyof TrainSettings)[]) {
      if (typeof saved[key] === typeof DEFAULT_TRAIN_SETTINGS[key]) (out as Record<string, unknown>)[key] = saved[key]
    }
    return out
  } catch {
    return { ...DEFAULT_TRAIN_SETTINGS }
  }
}

const GiB = 2 ** 30

/** Settings, the running task's progress, and every run with its checkpoints. */
export function TrainPanel({ project, task, onChanged, onLibraryChanged }: {
  project: TrainingProject
  /** This project's queued or running task, if any. */
  task: Job | null
  onChanged: () => void
  onLibraryChanged: () => void
}): React.JSX.Element {
  const [settings, setSettings] = useState<TrainSettings>(loadSettings)
  const [error, setError] = useState<string | null>(null)
  const set = <K extends keyof TrainSettings>(key: K, value: TrainSettings[K]) => {
    const next = { ...settings, [key]: value }
    setSettings(next)
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(next))
    } catch {
      // not persisted; fine
    }
  }

  const start = async (s: TrainSettings) => {
    setError(null)
    try {
      await trainingApi.train(project.id, s)
      onChanged()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const trainable = project.dataset?.train ?? 0
  const busy = task !== null
  const kind = (task?.request as { task?: string } | undefined)?.task

  return (
    <section className="train-section">
      <div className="train-section-head"><h3>Train</h3></div>
      <div className="train-settings">
        <Num label="Steps" value={settings.steps} min={1} max={20000} onChange={(v) => set('steps', v)} />
        <Num label="Save a checkpoint every" value={settings.checkpoint_every} min={1} max={20000} onChange={(v) => set('checkpoint_every', v)} />
        <Num label="Rank" value={settings.rank} min={1} max={128} onChange={(v) => set('rank', v)} />
        <label className="number-input">
          <span>Learning rate</span>
          <select className="input" value={settings.learning_rate} onChange={(e) => set('learning_rate', Number(e.target.value))}>
            {[2e-5, 5e-5, 1e-4, 2e-4, 5e-4].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label className="number-input">
          <span>Resolution</span>
          <select className="input" value={settings.target_side}
            onChange={(e) => set('target_side', Number(e.target.value) as TrainSettings['target_side'])}>
            {TARGET_SIDES.map((v) => <option key={v} value={v}>{v} px</option>)}
          </select>
        </label>
        <label className="number-input">
          <span>Caption dropout</span>
          <select className="input" value={settings.caption_dropout} onChange={(e) => set('caption_dropout', Number(e.target.value))}>
            {[0, 0.05, 0.1, 0.2].map((v) => <option key={v} value={v}>{Math.round(v * 100)}%</option>)}
          </select>
        </label>
        <Num label="Seed" value={settings.seed} min={0} max={2 ** 31 - 1} onChange={(v) => set('seed', v)} />
      </div>
      <p className="hint">
        Trains on {trainable > 0 ? `up to ${trainable}` : 'no'} photos: excluded ones are skipped, and held-out ones are kept for comparing. The image model
        must be on the GPU the whole time, so the chat model is unloaded first. Stopping keeps a checkpoint; a stopped
        run can be resumed later with the same result as an uninterrupted one.
      </p>
      {error && <div className="error-text" role="alert">{error}</div>}
      <div className="caption-actions">
        <button type="button" className="primary" disabled={busy || trainable <= 0} onClick={() => void start(settings)}>
          Start a new run
        </button>
        <button type="button" onClick={() => setSettings({ ...DEFAULT_TRAIN_SETTINGS })}>Defaults</button>
      </div>

      {task && kind !== 'training.scan' && <TaskProgress job={task} label={kind === 'training.compare' ? 'Comparing checkpoints' : 'Training'} />}

      {project.runs.length > 0 && <h3 className="runs-title">Runs</h3>}
      {[...project.runs].reverse().map((run) => (
        <RunCard key={run.id} project={project} run={run} busy={busy} onResume={(s) => void start(s)}
          onChanged={onChanged} onLibraryChanged={onLibraryChanged} />
      ))}
    </section>
  )
}

function Num({ label, value, min, max, onChange }: {
  label: string; value: number; min: number; max: number; onChange: (v: number) => void
}): React.JSX.Element {
  return (
    <label className="number-input">
      <span>{label}</span>
      <input className="input" type="number" min={min} max={max} value={value}
        onChange={(e) => {
          const v = Number(e.target.value)
          if (Number.isInteger(v) && v >= min && v <= max) onChange(v)
        }} />
    </label>
  )
}

export function TaskProgress({ job, label }: { job: Job; label: string }): React.JSX.Element {
  const pct = job.steps > 0 ? Math.min(100, (job.step / job.steps) * 100) : 0
  return (
    <div className="progress-card train-progress" role="status">
      <div className="progress-head">
        <strong>{label}</strong>
        <span className="muted small">{job.status === 'queued' ? 'waiting for the GPU' : job.stage}</span>
        <span className="grow" />
        {job.steps > 0 && <span className="tabular small">{job.step} / {job.steps}</span>}
        {job.detail?.loss !== undefined && <span className="tabular small">loss {job.detail.loss.toFixed(4)}</span>}
        <button type="button" className="danger-text" onClick={() => void api.cancel(job.id).catch(() => undefined)}>
          {job.stage === 'training' ? 'Stop (keeps a checkpoint)' : 'Stop'}
        </button>
      </div>
      <div className="bar"><div style={{ width: `${pct}%` }} /></div>
    </div>
  )
}

function RunCard({ project, run, busy, onResume, onChanged, onLibraryChanged }: {
  project: TrainingProject
  run: TrainingRun
  busy: boolean
  onResume: (settings: TrainSettings) => void
  onChanged: () => void
  onLibraryChanged: () => void
}): React.JSX.Element {
  const [log, setLog] = useState<LossPoint[] | null>(null)
  const [picked, setPicked] = useState<number[]>(() => run.checkpoints.slice(-MAX_COMPARE_CHECKPOINTS))
  const [prompts, setPrompts] = useState('')
  const [size, setSize] = useState<(typeof COMPARE_SIZES)[number]>(768)
  const [extra, setExtra] = useState(500)
  const [loraStep, setLoraStep] = useState<number | null>(run.checkpoints[run.checkpoints.length - 1] ?? null)
  const [loraName, setLoraName] = useState(`${project.name} ${new Date(run.created * 1000).toLocaleDateString()}`)
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null)

  const fail = (e: unknown) => setMessage({ ok: false, text: (e as Error).message })
  const config: TrainSettings = {
    rank: run.config.rank, learning_rate: run.config.learning_rate, steps: run.config.steps,
    checkpoint_every: run.config.checkpoint_every, caption_dropout: run.config.caption_dropout,
    seed: run.config.seed, target_side: run.target_side as TrainSettings['target_side'], run: run.id
  }
  const toggleLog = async () => {
    if (log) return setLog(null)
    try {
      setLog(await trainingApi.log(project.id, run.id))
    } catch (e) {
      fail(e)
    }
  }
  const compare = async () => {
    setMessage(null)
    try {
      const list = prompts.split('\n').map((p) => p.trim()).filter(Boolean).slice(0, MAX_COMPARE_PROMPTS)
      await trainingApi.compare(project.id, { run: run.id, steps: picked, prompts: list, size, seed: 1 })
      onChanged()
    } catch (e) {
      fail(e)
    }
  }
  const useAsLora = async () => {
    if (loraStep === null) return
    try {
      const item = await trainingApi.useCheckpoint(project.id, run.id, loraStep, loraName)
      setMessage({ ok: true, text: `Added “${item.name}” to the LoRA library. Choose it in the Image tab.` })
      onLibraryChanged()
    } catch (e) {
      fail(e)
    }
  }

  const finished = run.step >= run.steps
  return (
    <article className="run-card">
      <div className="progress-head">
        <strong>{new Date(run.created * 1000).toLocaleString()}</strong>
        <span className="tag">{run.status}</span>
        <span className="tabular small">step {run.step} / {run.steps}</span>
        <span className="muted small">{run.photos} photos · {run.target_side} px · rank {run.config.rank} · lr {run.config.learning_rate}</span>
        {(run.photo_licences?.['cc-by-sa'] ?? 0) > 0 && (
          <span className="warn-text small" title="Sharing a LoRA or images made from these photos may have to be under CC BY-SA too">
            {run.photo_licences!['cc-by-sa']} CC BY-SA photos
          </span>
        )}
        <span className="grow" />
        {run.loss_avg !== undefined && <span className="tabular small">loss {run.loss_avg.toFixed(4)}</span>}
        {run.peak_vram_bytes ? <span className="tabular small">{(run.peak_vram_bytes / GiB).toFixed(1)} GiB</span> : null}
      </div>
      {run.error && <div className="error-text">{run.error}</div>}
      <div className="caption-actions">
        {!finished && <button type="button" disabled={busy} onClick={() => onResume(config)}>Resume</button>}
        {finished && (
          <>
            <button type="button" disabled={busy} onClick={() => onResume({ ...config, steps: run.steps + extra })}>
              Train {extra} more steps
            </button>
            <input className="input extra-steps" type="number" min={1} max={20000 - run.steps} value={extra}
              aria-label="More steps" onChange={(e) => setExtra(Math.max(1, Math.min(20000 - run.steps, Number(e.target.value) || 1)))} />
          </>
        )}
        <button type="button" onClick={() => void toggleLog()}>{log ? 'Hide loss' : 'Show loss'}</button>
      </div>
      {log && <LossChart points={log} />}

      {run.checkpoints.length > 0 && (
        <div className="run-block">
          <span className="field-label">Checkpoints</span>
          <div className="chips">
            {run.checkpoints.map((s) => {
              const on = picked.includes(s)
              return (
                <button key={s} type="button" className={`chip ${on ? 'active' : ''}`} aria-pressed={on}
                  disabled={!on && picked.length >= MAX_COMPARE_CHECKPOINTS}
                  onClick={() => setPicked(on ? picked.filter((p) => p !== s) : [...picked, s].sort((a, b) => a - b))}>
                  {s}
                </button>
              )
            })}
          </div>
          <label className="field">
            <span className="field-label">Compare with prompts (one per line; empty uses held-out captions)</span>
            <textarea className="input" rows={2} value={prompts} onChange={(e) => setPrompts(e.target.value)} />
          </label>
          <div className="caption-actions">
            <select className="input" value={size} aria-label="Comparison size"
              onChange={(e) => setSize(Number(e.target.value) as (typeof COMPARE_SIZES)[number])}>
              {COMPARE_SIZES.map((s) => <option key={s} value={s}>{s} px</option>)}
            </select>
            <button type="button" disabled={busy || picked.length === 0} onClick={() => void compare()}>
              Compare {picked.length} checkpoint{picked.length === 1 ? '' : 's'} with the base model
            </button>
          </div>
        </div>
      )}

      {run.compare && (
        <div className="compare-grid" style={{ gridTemplateColumns: `minmax(120px, 1.2fr) repeat(${run.compare.columns.length}, 1fr)` }}>
          <span />
          {run.compare.columns.map((c) => <span key={c} className="field-label">{c === 'base' ? 'Without LoRA' : `Step ${c}`}</span>)}
          {run.compare.prompts.map((p, row) => (
            <div key={p} className="compare-row">
              <span className="small compare-prompt" title={p}>{p}</span>
              {(run.compare!.images[row] ?? []).map((id, col) => (
                <img key={id} src={studioImageUrl(id, 'thumb')} alt={`${p} — ${run.compare!.columns[col]}`} loading="lazy" />
              ))}
            </div>
          ))}
        </div>
      )}

      {run.checkpoints.length > 0 && (
        <div className="run-block caption-actions">
          <select className="input" value={loraStep ?? ''} aria-label="Checkpoint to use"
            onChange={(e) => setLoraStep(Number(e.target.value))}>
            {run.checkpoints.map((s) => <option key={s} value={s}>step {s}</option>)}
          </select>
          <input className="input grow" maxLength={60} value={loraName} aria-label="LoRA name"
            onChange={(e) => setLoraName(e.target.value)} />
          <button type="button" onClick={() => void useAsLora()}>Use as LoRA</button>
        </div>
      )}
      {message && <div className={message.ok ? 'ok-text' : 'error-text'} role="status">{message.text}</div>}
    </article>
  )
}
