import { useEffect, useState } from 'react'
import { cleanSubject, type Subject } from '@shared/subject.js'
import {
  COLLECT_HOSTS,
  COLLECT_LICENSES,
  COLLECT_CONTACT,
  COLLECT_TOTAL_CHOICES_GB,
  collectUserAgent,
  DEFAULT_COLLECT_COUNT,
  DEFAULT_COLLECT_TOTAL_GB,
  IDLE_COLLECT,
  MAX_COLLECT_COUNT,
  MAX_COLLECT_TARGETS,
  targetLabel,
  type CollectCheck,
  type CollectPlan,
  type CollectState,
  type SourceCheck,
  type CollectTarget
} from '@shared/training.js'
import { trainingApi } from '../api.js'
import { SubjectEditor } from './SubjectEditor.js'

const STORAGE_KEY = 'lis.collect.v2'
const SKIP_LABEL: Record<string, string> = {
  licence: 'licence not allowed',
  'personality rights': 'identifiable people',
  'not a photo': 'not a photo',
  'too small': 'too small',
  'file too large': 'file too large',
  duplicate: 'duplicates',
  'already from Commons': 'already found on Commons',
  'not a readable picture': 'not a readable picture',
  'download failed': 'download failed',
  'openverse limit reached': 'Openverse asked to slow down',
  'source unavailable': 'a source did not answer'
}

interface Form {
  project: string
  /** The last valid definition; null until the presets have loaded. */
  subject: Subject | null
  targets: CollectTarget[]
  totalGb: number
  commons: boolean
  openverse: boolean
  /** Also keep CC BY-SA photos. */
  shareAlike: boolean
  /** The user's contact for the User-Agent; empty means the project's page. */
  contact: string
}

const emptyTarget = (levels: number): CollectTarget =>
  ({ values: Array<string>(levels).fill(''), query: '', category: '', count: DEFAULT_COLLECT_COUNT })
const DEFAULT_FORM: Form = { project: '', subject: null, targets: [], totalGb: DEFAULT_COLLECT_TOTAL_GB, commons: true, openverse: false,
  shareAlike: false, contact: '' }

/** One name per level: keep the names by position when levels are added or removed. */
function fitTargets(targets: CollectTarget[], levels: number): CollectTarget[] {
  const fitted = targets.map((t) => ({ ...t, values: Array.from({ length: levels }, (_, i) => t.values[i] ?? '') }))
  return fitted.length > 0 ? fitted : [emptyTarget(levels)]
}

function loadForm(): Form {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? 'null') as Partial<Form> | null
    if (!saved) return DEFAULT_FORM
    const subject = saved.subject ? cleanSubject(saved.subject) : null
    const targets = Array.isArray(saved.targets) ? saved.targets.slice(0, MAX_COLLECT_TARGETS)
      .filter((t) => t && Array.isArray(t.values))
      .map((t) => ({ ...emptyTarget(0), ...t, values: t.values.map((v) => (typeof v === 'string' ? v : '')) })) : []
    return { ...DEFAULT_FORM, ...saved, subject, targets: subject ? fitTargets(targets, subject.levels.length) : [] }
  } catch {
    return DEFAULT_FORM
  }
}

const GB = 1024 ** 3
const size = (bytes: number) => (bytes >= GB ? `${(bytes / GB).toFixed(2)} GB` : `${Math.round(bytes / 1024 ** 2)} MB`)

/**
 * Collect openly licensed photos from the web into a folder, then make a training
 * project from it. The subject is defined first: its folder levels are the columns
 * to fill in, and the project is made with it. Says what is sent where before
 * anything is (docs/training.md).
 */
export function CollectPanel({ presets, onProject }: {
  presets: Subject[]
  onProject: (id: string) => void
}): React.JSX.Element {
  const [form, setForm] = useState<Form>(loadForm)
  const [subjectError, setSubjectError] = useState<string | null>(null)
  const [state, setState] = useState<CollectState>(IDLE_COLLECT)
  const [error, setError] = useState<string | null>(null)
  const [checks, setChecks] = useState<CollectCheck[] | null>(null)
  const [checking, setChecking] = useState(false)

  useEffect(() => {
    void trainingApi.collectState().then(setState).catch(() => undefined)
    return window.api.training.onCollectState(setState)
  }, [])
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(form))
    } catch {
      // not kept; fine
    }
  }, [form])
  // A first visit starts from the general preset once it has arrived.
  useEffect(() => {
    const general = presets.find((p) => p.preset === 'general') ?? presets[0]
    if (!form.subject && general) setForm((f) => ({ ...f, subject: general, targets: fitTargets(f.targets, general.levels.length) }))
  }, [presets, form.subject])

  const setTarget = (i: number, change: Partial<CollectTarget>) =>
    setForm((f) => ({ ...f, targets: f.targets.map((t, j) => (j === i ? { ...t, ...change } : t)) }))
  const setValue = (i: number, level: number, value: string) =>
    setForm((f) => ({ ...f, targets: f.targets.map((t, j) => (j === i ? { ...t, values: t.values.map((v, k) => (k === level ? value : v)) } : t)) }))

  const subject = form.subject
  const plan = (s: Subject): CollectPlan => ({
    project: form.project.trim() || form.targets.map(targetLabel).filter(Boolean).join(', ').slice(0, 60) || 'Collected photos',
    subject: s,
    targets: form.targets,
    max_total_bytes: form.totalGb * GB,
    sources: [...(form.commons ? ['commons' as const] : []), ...(form.openverse ? ['openverse' as const] : [])],
    share_alike: form.shareAlike,
    contact: form.contact.trim()
  })
  const start = async () => {
    if (!subject) return
    setError(null)
    try {
      await trainingApi.collect(plan(subject))
    } catch (e) {
      setError((e as Error).message)
    }
  }
  const check = async () => {
    if (!subject) return
    setError(null)
    setChecking(true)
    try {
      setChecks(await trainingApi.checkCollect(plan(subject)))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setChecking(false)
    }
  }

  if (!subject) return <section className="train-section collect-panel"><p className="muted">Loading…</p></section>

  const running = state.phase === 'running'
  const levels = subject.levels
  const contactOk = COLLECT_CONTACT.test(form.contact.trim())
  const ready = contactOk && !subjectError && form.targets.every((t) => levels.every((l, i) => !l.required || t.values[i]?.trim())) &&
    (form.commons || form.openverse)
  const want = form.targets.reduce((n, t) => n + t.count, 0)
  const columns = { gridTemplateColumns: `repeat(${levels.length}, minmax(90px, 1fr)) minmax(120px, 1.3fr) minmax(140px, 1.4fr) 80px 32px` }

  return (
    <section className="train-section collect-panel">
      <div className="train-section-head"><h2>Collect photos from the web</h2></div>
      <div className="collect-notice small">
        <p>
          The names or search words below are searched on <span className="mono">{COLLECT_HOSTS.join(', ')}</span>; photos
          download from Wikimedia and from the sites Openverse links to. Only <strong>{COLLECT_LICENSES}</strong>{form.shareAlike ? <>, and <strong>CC BY-SA</strong></> : ''} are kept.
          What is sent is the words you type; like any web request, the services see this computer's internet address.
          The photos stay in the folder you choose.
        </p>
        <details className="collect-privacy">
          <summary>Who can see what</summary>
          <ul>
            <li>
              <strong>Wikimedia and Openverse</strong> see this computer's internet address, the names, search words or
              category you give, which photos are fetched and when, and that the request comes from this app
              (<span className="mono">{collectUserAgent(form.contact)}</span>). No account, cookies or referrer are sent.
            </li>
            <li>
              <strong>The sites Openverse links to</strong> (often Flickr's image servers) see the address, the same
              app name and which photo is fetched, not what was searched for.
            </li>
            <li>
              <strong>Your internet provider</strong> (and a VPN provider, if one is on) sees which of these sites are
              contacted, when, and roughly how much is downloaded. Not the search words or the photos: everything goes
              over HTTPS, and a photo offered only over plain HTTP is skipped.
            </li>
            <li>Nothing about your projects, your files or other photos is sent. Checking what is available sends the same
              searches and fetches no photos.</li>
          </ul>
        </details>
        <p className="muted">
          A free licence covers the photo, not what is in it: brand logos and product designs can still be protected, and
          photos may show people or licence plates. A search can also find photos of something else that mention the
          same words. Look through them before training on them or sharing anything made from them.
        </p>
        <p className="collect-responsibility">
          <strong>Collecting is your responsibility.</strong> You run it from your own computer, under your own name: follow
          Wikimedia's policies and each photo's licence, comply with the laws where you live, and collect only what you
          need, sparingly and wisely.
        </p>
        <label className="field">
          <span className="field-label">Your contact for Wikimedia</span>
          <input className="input" maxLength={100} value={form.contact} disabled={running} aria-label="Contact for Wikimedia"
            placeholder="e.g. you@example.com or User:YourName"
            onChange={(e) => setForm({ ...form, contact: e.target.value })} />
          <span className="hint">
            Wikimedia's User-Agent policy asks every tool for a way to contact the person running it, and may block tools
            that give none. If you collect, enter your own contact: an email address, a website, or a Wikimedia user page
            (<span className="mono">User:YourName</span>; a separate account keeps your email private). It goes only into
            the requests to these services. Left empty, the app sends its project page instead, which reaches the app's
            author, not you.
          </span>
          {!contactOk && <span className="error-text">Up to 100 letters, digits, spaces and @ . : / _ + ~ -</span>}
        </label>
      </div>

      <label className="field">
        <span className="field-label">Project name</span>
        <input className="input" maxLength={60} value={form.project} placeholder="e.g. Dive watches" disabled={running}
          aria-label="Project name" onChange={(e) => setForm({ ...form, project: e.target.value })} />
      </label>

      <div className="train-section">
        <span className="field-label">Subject</span>
        <SubjectEditor initial={subject} presets={presets} disabled={running} onChange={(next, err) => {
          setSubjectError(err)
          if (next) setForm((f) => ({ ...f, subject: next, targets: fitTargets(f.targets, next.levels.length) }))
        }} />
      </div>

      <div className="collect-targets" role="table" aria-label="What to collect">
        <div className="collect-row head" role="row" style={columns}>
          {levels.map((l) => <span key={l.key}>{l.label}{l.required ? '' : ' (optional)'}</span>)}
          <span title="What is searched for; empty means the names joined">Search words</span>
          <span title="The most precise source, e.g. Category:Brand Model. Empty: a search.">Commons category</span>
          <span>Photos</span><span />
        </div>
        {form.targets.map((t, i) => (
          <div key={i} className="collect-row" role="row" style={columns}>
            {levels.map((l, k) => (
              <input key={l.key} className="input" maxLength={60} value={t.values[k] ?? ''} disabled={running}
                aria-label={`Row ${i + 1} ${l.label}`} placeholder={l.required ? 'required' : 'optional'}
                onChange={(e) => setValue(i, k, e.target.value)} />
            ))}
            <input className="input" maxLength={60} value={t.query} disabled={running} aria-label={`Row ${i + 1} search words`}
              placeholder={targetLabel(t) || 'the names'} onChange={(e) => setTarget(i, { query: e.target.value })} />
            <input className="input" maxLength={60} value={t.category} disabled={running} aria-label={`Row ${i + 1} category`}
              placeholder="optional" onChange={(e) => setTarget(i, { category: e.target.value })} />
            <input className="input" type="number" min={1} max={MAX_COLLECT_COUNT} value={t.count} disabled={running}
              aria-label={`Row ${i + 1} photos`}
              onChange={(e) => {
                const v = Number(e.target.value)
                if (Number.isInteger(v) && v >= 1 && v <= MAX_COLLECT_COUNT) setTarget(i, { count: v })
              }} />
            <button type="button" className="icon-button" title="Remove" disabled={running || form.targets.length === 1}
              onClick={() => setForm({ ...form, targets: form.targets.filter((_, j) => j !== i) })}>×</button>
          </div>
        ))}
      </div>
      <div className="caption-actions">
        <button type="button" disabled={running || form.targets.length >= MAX_COLLECT_TARGETS}
          onClick={() => setForm({ ...form, targets: [...form.targets, emptyTarget(levels.length)] })}>Add a row</button>
        <label className="number-input">
          <span>Size limit</span>
          <select className="input" value={form.totalGb} disabled={running}
            onChange={(e) => setForm({ ...form, totalGb: Number(e.target.value) })}>
            {COLLECT_TOTAL_CHOICES_GB.map((g) => <option key={g} value={g}>{g} GB in total</option>)}
          </select>
        </label>
        <label className="toggle"><input type="checkbox" checked={form.commons} disabled={running}
          onChange={(e) => setForm({ ...form, commons: e.target.checked })} /><span>Wikimedia Commons</span></label>
        <label className="toggle"><input type="checkbox" checked={form.openverse} disabled={running}
          onChange={(e) => setForm({ ...form, openverse: e.target.checked })} />
          <span title="Used only when Commons has too few. It often answers slowly, and shows at most 240 results of a search.">
            Openverse too (rarely helps)
          </span></label>
        <label className="toggle"><input type="checkbox" checked={form.shareAlike} disabled={running} aria-label="Also share-alike photos"
          onChange={(e) => setForm({ ...form, shareAlike: e.target.checked })} /><span>Also CC BY-SA (share-alike)</span></label>
      </div>
      {form.shareAlike && (
        <p className="hint warn-text">
          Most photos on Commons are CC BY-SA, so this finds many more. Keeping them for yourself is fine; if you share
          what you make from them (a LoRA, or images), the licence may ask you to share it under CC BY-SA too, and
          whether a LoRA counts is not settled. Runs and LoRAs trained on them say so.
        </p>
      )}
      <p className="hint">
        Up to {want} photos, each scaled to about 1600 px wide and at most 8 MB. They go in folders by the levels above.
        Running it again into the same folder continues where it stopped.
      </p>
      {error && <div className="error-text" role="alert">{error}</div>}
      <div className="caption-actions">
        {running ? (
          <button type="button" className="danger-text" onClick={() => void trainingApi.cancelCollect()}>Stop (keeps what is saved)</button>
        ) : (
          <>
            <button type="button" disabled={!ready || checking} onClick={() => void check()}>
              {checking ? 'Checking…' : 'Check what is available'}
            </button>
            {checking && state.walking && (
              <span className="muted small">{state.walking.label}: looking through categories ({state.walking.categories})</span>
            )}
            <button type="button" className="primary" disabled={!ready || checking} onClick={() => void start()}>Choose a folder and collect</button>
          </>
        )}
      </div>
      {checks && <CheckResults checks={checks} />}

      {state.phase !== 'idle' && (
        <div className="progress-card train-progress collect-progress" role="status">
          <div className="progress-head">
            <strong>{running ? 'Collecting' : state.phase === 'done' ? 'Done' : state.phase === 'cancelled' ? 'Stopped' : 'Failed'}</strong>
            <span className="muted small truncate" title={state.folder ?? ''}>{state.folder}</span>
            <span className="grow" />
            <span className="tabular small">{state.photos} photos · {size(state.bytes)}{state.maxBytes ? ` of ${size(state.maxBytes)}` : ''}</span>
          </div>
          {running && state.walking && (
            <p className="muted small">{state.walking.label}: looking through Commons categories ({state.walking.categories} so far)</p>
          )}
          {state.maxBytes > 0 && <div className="bar"><div style={{ width: `${Math.min(100, (state.bytes / state.maxBytes) * 100)}%` }} /></div>}
          <ul className="collect-list small">
            {state.targets.map((t, i) => (
              <li key={t.label} className={state.current === i ? 'strong' : ''}>
                <span className="truncate">{t.label}</span>
                <span className="tabular">{t.have} / {t.want}</span>
              </li>
            ))}
          </ul>
          {Object.keys(state.skipped).length > 0 && (
            <p className="muted small">
              Skipped: {Object.entries(state.skipped).map(([k, n]) => `${n} ${SKIP_LABEL[k] ?? k}`).join(', ')}.
            </p>
          )}
          {state.message && <p className={state.phase === 'failed' ? 'error-text' : 'small'}>{state.message}</p>}
          {state.phase === 'done' && state.projectId && (
            <button type="button" className="primary" onClick={() => onProject(state.projectId!)}>Open the project</button>
          )}
        </div>
      )}
    </section>
  )
}

/** One source's answer for one row, in words. */
function describeSource(name: string, c: SourceCheck): string {
  if (c.error) return `${name}: ${c.error}`
  if (c.missing) return `${name}: there is no such category`
  const where = c.mode === 'category'
    ? `${c.found} files in ${c.categories ?? 1} categor${c.categories === 1 ? 'y' : 'ies'}`
    : c.reachable !== undefined && c.reachable < c.found
      ? `${c.found} found, ${c.reachable} shown to this app`
      : `${c.found} found`
  return `${name}: about ${c.estimate} usable (${where}; ${c.kept} of ${c.sampled} sampled have a licence and size that are kept)`
}

/** What "Check what is available" found, row by row, without downloading anything. */
function CheckResults({ checks }: { checks: CollectCheck[] }): React.JSX.Element {
  return (
    <div className="collect-check small" role="status">
      {checks.map((c) => {
        const total = (c.commons?.estimate ?? 0) + (c.openverse?.estimate ?? 0)
        return (
          <div key={c.label} className="collect-check-row">
            <p>
              <strong>{c.label}</strong>{' '}
              <span className={total >= c.want ? 'muted' : 'warn-text'}>
                about {total} usable for {c.want} wanted{total < c.want ? ' — fewer than wanted' : ''}
              </span>
            </p>
            <ul>
              {c.commons && <li>{describeSource('Commons', c.commons)}</li>}
              {c.commons?.skipped && c.commons.skipped.length > 0 && (
                <li className="muted">Left out: {c.commons.skipped.map((t) => t.replace(/^Category:/, '')).join(', ')}</li>
              )}
              {c.commons?.truncated && <li className="muted">Only the first 150 categories are looked at.</li>}
              {c.openverse && <li>{describeSource('Openverse', c.openverse)}</li>}
            </ul>
          </div>
        )
      })}
      <p className="hint">Estimates from a sample; the counts include photos that only mention the words, so expect fewer
        that really show the subject.</p>
    </div>
  )
}
