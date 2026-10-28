/**
 * The Train tab: types shared by the main process and the window, and the pure
 * checks the main process applies before anything reaches the engine
 * (image_engine/training/service.py has the authoritative ones).
 *
 * As with images, the window never names a path: a project's photo folder comes
 * from a native folder dialog in the main process, and photos are shown through
 * the private `studio-image:` protocol by id.
 */

import { IMAGE_SCHEME, type ImageJob } from './image.js'
import { cleanSubject, type Subject } from './subject.js'

const PROJECT_ID = /^[0-9a-f]{16}$/
const RUN_ID = /^[0-9a-f]{12}$/

export const TRAINING_NAME_MAX = 60
export const TARGET_SIDES = [512, 768, 1024] as const
export const COMPARE_SIZES = [512, 768, 1024] as const
export const MAX_COMPARE_CHECKPOINTS = 4
export const MAX_COMPARE_PROMPTS = 4

export interface TrainSettings {
  rank: number
  learning_rate: number
  steps: number
  checkpoint_every: number
  caption_dropout: number
  seed: number
  target_side: (typeof TARGET_SIDES)[number]
  /** Continue (or extend) this run instead of starting a new one. */
  run?: string
}

export const DEFAULT_TRAIN_SETTINGS: TrainSettings = {
  rank: 16,
  learning_rate: 1e-4,
  steps: 2000,
  checkpoint_every: 250,
  caption_dropout: 0.1,
  seed: 0,
  target_side: 768
}

export interface DatasetSummary {
  total: number
  usable: number
  train: number
  holdout: number
  /** Issue kind -> how many photos have it (unreadable, too_small, duplicate_of, missing_facts). */
  issues: Record<string, number>
  warnings: number
}

export interface AuditEstimate {
  checked: number
  wrong: number
  rate: number
  low: number
  high: number
}

export interface CaptionSummary {
  total: number
  flagged: number
  to_review: number
  excluded: number
  /** Photos the captioning model found not to show the subject (toy, picture, absent, too small). */
  off_subject: number
  /** Photos the captioning model has described. */
  described: number
  edited: number
  audit: AuditEstimate | null
}

export interface CompareGrid {
  /** "base" (no LoRA), then checkpoint steps. */
  columns: string[]
  prompts: string[]
  /** Gallery ids, one row per prompt, one entry per column. */
  images: string[][]
  size: number
  seed: number
  created: number
}

export interface TrainingRun {
  id: string
  created: number
  status: 'running' | 'finished' | 'stopped' | 'failed' | 'broken'
  photos: number
  target_side: number
  config: TrainSettings & { shift: number; alpha: number | null; grad_clip: number }
  /** Licence -> photos, when the photos were collected by the studio. */
  photo_licences?: Record<string, number>
  step: number
  steps: number
  loss_avg?: number
  seconds?: number
  peak_vram_bytes?: number
  error?: string
  checkpoints: number[]
  compare: CompareGrid | null
}

export interface TrainingProject {
  id: string
  name: string
  photo_root: string
  subject: Subject
  created: number
  captioner: string
  dataset: DatasetSummary | null
  warnings: string[]
  captions: CaptionSummary | null
  runs: TrainingRun[]
}

export type CaptionStatus = 'auto' | 'accepted' | 'edited' | 'excluded'
export type CaptionView = 'all' | 'review' | 'excluded'

export interface CaptionItem {
  item: string
  path: string
  facts: Record<string, string>
  holdout: boolean
  width: number
  height: number
  caption: string
  description: Record<string, string>
  flags: string[]
  status: CaptionStatus
  audit: boolean
  /** Left out of training: by the user, or as off subject until included. */
  excluded: boolean
  /** What the captioning model said about the photo showing the subject; null before it looked. */
  relevance: { subject_shown: string; subject_size: string } | null
}

export interface CaptionPage {
  total: number
  items: CaptionItem[]
}

export interface LossPoint {
  step: number
  loss: number
  sigma: number
  sample: number
  dropped: boolean
}

export interface LoraItem {
  id: string
  name: string
  created: number
  size: number
  sha256: string
  rank: number
  origin: { project?: string; project_name?: string; run?: string; step?: number; photo_licences?: Record<string, number> }
}

export interface CompareRequest {
  run: string
  steps: number[]
  /** Empty: captions of held-out photos. */
  prompts: string[]
  size: (typeof COMPARE_SIZES)[number]
  seed: number
}

export interface TrainingApi {
  projects(): Promise<TrainingProject[]>
  /** The starting points for a subject definition ("general", "cars"). */
  subjectPresets(): Promise<Subject[]>
  /** Opens a folder picker for the photos; null if cancelled. */
  createProject(name: string, subject: Subject): Promise<TrainingProject | null>
  project(id: string): Promise<TrainingProject>
  /** Replace the subject definition; scan again afterwards. */
  setSubject(id: string, subject: Subject): Promise<TrainingProject>
  deleteProject(id: string): Promise<void>
  scan(id: string): Promise<ImageJob>
  /** Describe the photos with the captioning model (a task; carries on where it stopped). */
  caption(id: string): Promise<ImageJob>
  captions(id: string, view: CaptionView, offset: number, limit: number): Promise<CaptionPage>
  review(id: string, item: string, action: 'accept' | 'edit' | 'exclude', edits?: Record<string, string>): Promise<CaptionItem>
  train(id: string, settings: TrainSettings): Promise<ImageJob>
  compare(id: string, request: CompareRequest): Promise<ImageJob>
  log(id: string, run: string): Promise<LossPoint[]>
  useCheckpoint(id: string, run: string, step: number, name: string): Promise<LoraItem>
  loras(): Promise<LoraItem[]>
  renameLora(id: string, name: string): Promise<LoraItem>
  deleteLora(id: string): Promise<void>

  /** Collecting openly licensed photos (image-engine/scripts/collect_photos.py). */
  collectState(): Promise<CollectState>
  /** Opens a folder picker for where the photos go, then collects; resolves when done or stopped. */
  collect(plan: CollectPlan): Promise<void>
  /** Ask the sources how many photos each row would likely get; downloads nothing. */
  checkCollect(plan: CollectPlan): Promise<CollectCheck[]>
  cancelCollect(): Promise<void>
  onCollectState(handler: (state: CollectState) => void): () => void
}

// ---------------------------------------------------------------------------
// Pure helpers (tested in tests/training-bridge.test.ts)

export function isProjectId(value: unknown): value is string {
  return typeof value === 'string' && PROJECT_ID.test(value)
}

export function isRunId(value: unknown): value is string {
  return typeof value === 'string' && RUN_ID.test(value)
}

/** The window-facing address of a training photo's thumbnail. */
export function studioTrainingPhotoUrl(project: string, item: string): string {
  return `${IMAGE_SCHEME}://training/${project}/${item}.webp`
}

function int(value: unknown, lo: number, hi: number, name: string): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < lo || value > hi) {
    throw new Error(`${name} must be a whole number from ${lo} to ${hi}.`)
  }
  return value
}

function num(value: unknown, lo: number, hi: number, name: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < lo || value > hi) {
    throw new Error(`${name} must be a number from ${lo} to ${hi}.`)
  }
  return value
}

/** Rebuild training settings field by field; anything unexpected is refused. */
export function cleanTrainSettings(raw: unknown): TrainSettings {
  if (!raw || typeof raw !== 'object') throw new Error('Invalid training settings.')
  const s = raw as Record<string, unknown>
  const side = s.target_side
  if (!TARGET_SIDES.includes(side as (typeof TARGET_SIDES)[number])) {
    throw new Error(`Resolution must be one of ${TARGET_SIDES.join(', ')}.`)
  }
  const out: TrainSettings = {
    rank: int(s.rank, 1, 128, 'Rank'),
    learning_rate: num(s.learning_rate, 1e-6, 1e-2, 'Learning rate'),
    steps: int(s.steps, 1, 20000, 'Steps'),
    checkpoint_every: int(s.checkpoint_every, 1, 20000, 'Checkpoint interval'),
    caption_dropout: num(s.caption_dropout, 0, 0.5, 'Caption dropout'),
    seed: int(s.seed, 0, 2 ** 31 - 1, 'Seed'),
    target_side: side as TrainSettings['target_side']
  }
  if (s.run !== undefined) {
    if (!isRunId(s.run)) throw new Error('Invalid run.')
    out.run = s.run
  }
  return out
}

export function cleanCompareRequest(raw: unknown): CompareRequest {
  if (!raw || typeof raw !== 'object') throw new Error('Invalid comparison.')
  const r = raw as Record<string, unknown>
  if (!isRunId(r.run)) throw new Error('Invalid run.')
  if (!Array.isArray(r.steps) || r.steps.length < 1 || r.steps.length > MAX_COMPARE_CHECKPOINTS) {
    throw new Error(`Choose 1 to ${MAX_COMPARE_CHECKPOINTS} checkpoints.`)
  }
  const steps = r.steps.map((s) => int(s, 1, 20000, 'Checkpoint'))
  if (!Array.isArray(r.prompts) || r.prompts.length > MAX_COMPARE_PROMPTS) {
    throw new Error(`Give at most ${MAX_COMPARE_PROMPTS} prompts.`)
  }
  const prompts = r.prompts.map((p) => {
    if (typeof p !== 'string' || !p.trim() || p.length > 1000) throw new Error('Each prompt must be 1 to 1000 characters.')
    return p.trim()
  })
  if (!COMPARE_SIZES.includes(r.size as (typeof COMPARE_SIZES)[number])) throw new Error('Invalid size.')
  return { run: r.run, steps, prompts, size: r.size as CompareRequest['size'], seed: int(r.seed, 0, 2 ** 32 - 1, 'Seed') }
}

/** Caption edits: field keys to answers. Which fields exist is the project's business (the engine checks). */
export function cleanEdits(raw: unknown): Record<string, string> | undefined {
  if (raw === undefined) return undefined
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error('Invalid edits.')
  const entries = Object.entries(raw)
  if (entries.length > 10) throw new Error('Invalid edits.')
  const out: Record<string, string> = {}
  for (const [key, value] of entries) {
    if (!/^[\p{L}\p{N}_]{1,30}$/u.test(key) || typeof value !== 'string' || value.length > 60) throw new Error(`Invalid edit: ${key}`)
    out[key] = value
  }
  return out
}

export function cleanName(raw: unknown, what: string): string {
  if (typeof raw !== 'string') throw new Error(`${what} must be text.`)
  const name = raw.split(/\s+/).filter(Boolean).join(' ')
  if (!name || name.length > TRAINING_NAME_MAX) throw new Error(`${what} must be 1 to ${TRAINING_NAME_MAX} characters.`)
  return name
}

/** A training task's job: is it this project's, and what does it do. */
export function taskOf(job: ImageJob): { task: string; project: string } | null {
  const r = job.request as { kind?: string; task?: unknown; project?: unknown }
  return job.kind === 'task' && typeof r.task === 'string' && typeof r.project === 'string'
    ? { task: r.task, project: r.project }
    : null
}

/** Human words for a caption flag, e.g. "disagree:view" -> "two answers differ on view". */
export function describeFlag(flag: string): string {
  const [kind, field = ''] = flag.split(':', 2)
  const f = field.replace('_', ' ')
  switch (kind) {
    case 'disagree': return `two answers differ on ${f}`
    case 'low_confidence': return `unsure about ${f}`
    case 'unknown': return `no answer for ${f}`
    case 'conflict': return `${f} contradicts labels.json`
    case 'outlier': return `${f} differs from most photos of this subject`
    case 'off_subject': return field === 'subject_size' ? 'subject too small in the photo' : 'not the real subject'
    case 'partial': return 'only part of the subject, or its inside'
    case 'single_pass': return 'only one answer to check'
    case 'unparseable': return 'no usable answer'
    default: return flag
  }
}

// ---------------------------------------------------------------------------
// Collecting photos (docs/training.md, "Collecting photos")

/** Where collected photos come from, in this order; shown to the user before anything is sent. */
export const COLLECT_HOSTS = ['commons.wikimedia.org', 'upload.wikimedia.org', 'api.openverse.org']
export const COLLECT_LICENSES = 'CC0, public domain and CC BY (with credits written next to the photos)'
/** The project's page: the User-Agent's contact when the user gives none (as in collect_photos.py; kept equal by a test). */
export const COLLECT_PROJECT_URL = 'https://github.com/justsimpleideas/ai-studio'
/** What a contact may hold: it goes into an HTTP header. */
export const COLLECT_CONTACT = /^[A-Za-z0-9@.:/_+~ -]{0,100}$/

/** How collect_photos.py names itself to the services (Wikimedia asks tools for a name and a contact). */
export function collectUserAgent(contact = ''): string {
  return `LocalAIStudio-collect/1.0 (${contact.trim() || COLLECT_PROJECT_URL}; dataset tool for personal training)`
}
export const MAX_COLLECT_TARGETS = 10
export const MAX_COLLECT_COUNT = 1000
export const COLLECT_TOTAL_CHOICES_GB = [1, 2, 5, 10, 20] as const
export const COLLECT_FILE_LIMIT_BYTES = 8 * 1024 ** 2
export const DEFAULT_COLLECT_COUNT = 200
export const DEFAULT_COLLECT_TOTAL_GB = 2

export interface CollectTarget {
  /** One name per level of the subject, in order; "" for an optional level left out. */
  values: string[]
  /** Search words; empty means the names joined. */
  query: string
  /** A Wikimedia Commons category, the most precise source; empty means a search. */
  category: string
  count: number
}

export interface CollectPlan {
  /** The training project made from the photos. */
  project: string
  /** Its subject: the levels name the folders, and the project is made with it. */
  subject: Subject
  targets: CollectTarget[]
  max_total_bytes: number
  sources: ('commons' | 'openverse')[]
  /** Also keep CC BY-SA photos (off unless the user turns it on). */
  share_alike: boolean
  /** The user's contact for the User-Agent; empty means the project's page. */
  contact: string
}

/** What one source said about one row (collect_photos.py --check). */
export interface SourceCheck {
  /** "category" (with its subcategories) or "search"; Commons only. */
  mode?: 'category' | 'search'
  /** Files the source has for the row. */
  found: number
  /** Openverse: how many of them it shows an anonymous client. */
  reachable?: number
  /** A sample of them, and how many of the sample have a licence and size that would be kept. */
  sampled: number
  kept: number
  /** found (or reachable) scaled by kept / sampled. */
  estimate: number
  categories?: number
  /** Subcategories left out because their names say they hold logos, drawings, toys... */
  skipped?: string[]
  truncated?: boolean
  /** The Commons category does not exist. */
  missing?: boolean
  error?: string
}

export interface CollectCheck {
  label: string
  want: number
  commons?: SourceCheck
  openverse?: SourceCheck
}

const asNumber = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0)
const asText = (v: unknown): string => (typeof v === 'string' ? v : '')

function sourceCheck(raw: unknown): SourceCheck | undefined {
  if (!raw || typeof raw !== 'object') return undefined
  const r = raw as Record<string, unknown>
  const out: SourceCheck = { found: asNumber(r.found), sampled: asNumber(r.sampled), kept: asNumber(r.kept), estimate: asNumber(r.estimate) }
  if (r.mode === 'category' || r.mode === 'search') out.mode = r.mode
  if (typeof r.reachable === 'number') out.reachable = r.reachable
  if (typeof r.categories === 'number') out.categories = r.categories
  if (Array.isArray(r.skipped)) out.skipped = r.skipped.filter((x): x is string => typeof x === 'string')
  if (r.truncated === true) out.truncated = true
  if (r.missing === true) out.missing = true
  if (typeof r.error === 'string') out.error = r.error
  return out
}

/** One "check" event from collect_photos.py --check --json, or null. */
export function parseCheckEvent(event: Record<string, unknown>): CollectCheck | null {
  if (event.event !== 'check') return null
  const out: CollectCheck = { label: asText(event.label), want: asNumber(event.want) }
  const commons = sourceCheck(event.commons)
  const openverse = sourceCheck(event.openverse)
  if (commons) out.commons = commons
  if (openverse) out.openverse = openverse
  return out
}

export interface CollectTargetProgress {
  label: string
  have: number
  want: number
}

export interface CollectState {
  phase: 'idle' | 'running' | 'done' | 'failed' | 'cancelled'
  folder: string | null
  targets: CollectTargetProgress[]
  current: number | null
  /** Commons categories being looked through for a row (while collecting or checking). */
  walking: { label: string; categories: number } | null
  photos: number
  bytes: number
  maxBytes: number
  skipped: Record<string, number>
  sizeLimitReached: boolean
  /** The project made from the folder once collection finished. */
  projectId: string | null
  message: string | null
}

export const IDLE_COLLECT: CollectState = {
  phase: 'idle', folder: null, targets: [], current: null, walking: null, photos: 0, bytes: 0, maxBytes: 0, skipped: {},
  sizeLimitReached: false, projectId: null, message: null
}

function text(value: unknown, what: string, required: boolean): string {
  if (value === undefined || value === null) value = ''
  if (typeof value !== 'string') throw new Error(`${what} must be text.`)
  const clean = value.split(/\s+/).filter(Boolean).join(' ')
  if (required && !clean) throw new Error(`${what} is required.`)
  if (clean.length > TRAINING_NAME_MAX) throw new Error(`${what} must be at most ${TRAINING_NAME_MAX} characters.`)
  return clean
}

/** What a collect target is called while collecting: its names joined. */
export function targetLabel(target: CollectTarget): string {
  return target.values.filter(Boolean).join(' ')
}

/** Rebuild a collection plan field by field (the script checks it again). */
export function cleanCollectPlan(raw: unknown): CollectPlan {
  if (!raw || typeof raw !== 'object') throw new Error('Invalid collection plan.')
  const r = raw as Record<string, unknown>
  const subject = cleanSubject(r.subject)
  if (!Array.isArray(r.targets) || r.targets.length < 1 || r.targets.length > MAX_COLLECT_TARGETS) {
    throw new Error(`Give 1 to ${MAX_COLLECT_TARGETS} rows to collect.`)
  }
  const targets = r.targets.map((t, i): CollectTarget => {
    if (!t || typeof t !== 'object') throw new Error('Invalid row.')
    const o = t as Record<string, unknown>
    if (!Array.isArray(o.values) || o.values.length !== subject.levels.length) {
      throw new Error(`Row ${i + 1}: give one name per folder level.`)
    }
    const values = subject.levels.map((level, j) => text((o.values as unknown[])[j], `Row ${i + 1}: ${level.label}`, level.required))
    return {
      values,
      query: text(o.query, `Row ${i + 1}: search words`, false),
      category: text(o.category, `Row ${i + 1}: category`, false),
      count: int(o.count, 1, MAX_COLLECT_COUNT, `Row ${i + 1}: photos`)
    }
  })
  const labels = targets.map((t) => targetLabel(t).toLowerCase())
  if (new Set(labels).size !== labels.length) throw new Error('Two rows have the same names.')
  const gb = COLLECT_TOTAL_CHOICES_GB.find((g) => g * 1024 ** 3 === r.max_total_bytes)
  if (!gb) throw new Error('Choose a total size limit.')
  const sources = r.sources
  if (!Array.isArray(sources) || sources.length === 0 || sources.some((s) => s !== 'commons' && s !== 'openverse')) {
    throw new Error('Choose at least one source.')
  }
  if (typeof r.share_alike !== 'boolean') throw new Error('Choose whether share-alike photos are kept.')
  const contact = typeof r.contact === 'string' ? r.contact.trim() : ''
  if (r.contact !== undefined && typeof r.contact !== 'string') throw new Error('Invalid contact.')
  if (!COLLECT_CONTACT.test(contact)) throw new Error('A contact may hold up to 100 letters, digits, spaces and @ . : / _ + ~ -')
  return {
    project: cleanName(r.project, 'A project name'),
    subject,
    share_alike: r.share_alike,
    contact,
    targets,
    max_total_bytes: gb * 1024 ** 3,
    sources: [...new Set(sources as ('commons' | 'openverse')[])]
  }
}

/** One JSON line from collect_photos.py --json, or null for anything else. */
export function parseCollectEvent(line: string): Record<string, unknown> | null {
  if (!line.startsWith('{')) return null
  try {
    const value = JSON.parse(line) as unknown
    return value && typeof value === 'object' && typeof (value as { event?: unknown }).event === 'string'
      ? (value as Record<string, unknown>)
      : null
  } catch {
    return null
  }
}


/** Fold the script's events into the state the window shows. */
export function feedCollect(state: CollectState, event: Record<string, unknown>): CollectState {
  switch (event.event) {
    case 'start': {
      const labels = Array.isArray(event.targets) ? event.targets.filter((l): l is string => typeof l === 'string') : []
      return { ...state, phase: 'running', targets: labels.map((label) => ({ label, have: 0, want: 0 })),
        bytes: asNumber(event.bytes), maxBytes: asNumber(event.max_total_bytes) }
    }
    case 'walking':
      return { ...state, walking: { label: typeof event.label === 'string' ? event.label : '', categories: asNumber(event.categories) } }
    case 'target':
    case 'target_done':
    case 'saved': {
      const index = asNumber(event.index)
      const targets = state.targets.map((t, i) => (i === index ? { ...t, have: asNumber(event.have), want: asNumber(event.want) } : t))
      return { ...state, targets, walking: null, current: event.event === 'target_done' ? null : index,
        bytes: event.event === 'saved' ? asNumber(event.bytes) : state.bytes,
        photos: targets.reduce((n, t) => n + t.have, 0) }
    }
    case 'done': {
      const skipped: Record<string, number> = {}
      if (event.skipped && typeof event.skipped === 'object') {
        for (const [k, v] of Object.entries(event.skipped)) if (typeof v === 'number') skipped[k] = v
      }
      return { ...state, phase: 'done', current: null, photos: asNumber(event.photos), bytes: asNumber(event.bytes), skipped,
        sizeLimitReached: event.size_limit_reached === true }
    }
    case 'error':
      return { ...state, phase: 'failed', current: null, message: typeof event.message === 'string' ? event.message : 'Collection failed.' }
    default:
      return state
  }
}
