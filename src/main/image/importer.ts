import fs from 'node:fs/promises'
import path from 'node:path'
import {
  IDLE_IMPORT,
  fewStepProblem,
  folderNameFor,
  isTwoExperts,
  isValidFolderName,
  parseImportEvent,
  targetOf,
  type ImportOptions,
  type ImportProbe,
  type ImportTarget,
  type ModelImportState
} from '@shared/importing.js'
import { MODEL_CATALOG, catalogModel, type ModelId } from '@shared/setup.js'
import type { ModelDownloads } from './downloads.js'
import { runProcess, type Runner } from './setup.js'

/**
 * Importing a model published as a single transformer file (shared/importing.ts).
 *
 * The file path always comes from a native dialog the main process showed; the window
 * only chooses a variant, a folder name and, optionally, the publisher's hash.
 * image-engine/scripts/import_model.py does the checking and assembling, with the
 * engine's Python. Parts the model needs and that are not downloaded yet are fetched
 * first with the same verified downloads as the model cards, once the user has
 * accepted their licenses.
 */

/** Configured folders the engine reports, used to find parts already downloaded. */
export interface EngineFolders {
  model_dir?: string
  caption_model_dir?: string
  video_model_dir?: string
}

export interface ModelImporterOptions {
  python: () => string
  script: string
  run?: Runner
  modelsFolder: () => string
  engineFolders: () => Promise<EngineFolders | null>
  /** Verified folders seen before (ImagePreferences.partFolders), and a way to add to them. */
  knownFolders: () => string[]
  rememberFolders: (folders: string[]) => Promise<void>
  downloads: ModelDownloads
  /** Point the engine at the assembled folder, as its image or video model. */
  selectFolder: (dest: string, target: ImportTarget) => Promise<void>
  onState: (state: ModelImportState) => void
}

type Need = { id: ModelId; flag: string; configured?: keyof EngineFolders; also: ModelId[] }

/** What each model needs around its transformer file, and where to look for it first. */
const NEEDS: Record<string, Need[]> = {
  // The full 6B download holds every part too.
  diffusion: [{ id: 'image-parts', flag: '--parts-from', configured: 'model_dir', also: ['image'] }],
  large: [
    { id: 'caption', flag: '--text-encoder-from', configured: 'caption_model_dir', also: [] },
    { id: 'large-vae', flag: '--vae-from', also: [] }
  ],
  // Likewise the full 5B video download (the original's; the distilled one is another repository).
  '5b': [{ id: 'video-parts', flag: '--parts-from', configured: 'video_model_dir', also: ['video'] }],
  '14b-text': [{ id: 'video-14b-text-parts', flag: '--parts-from', also: [] }],
  '14b-image': [{ id: 'video-14b-image-parts', flag: '--parts-from', also: [] }]
}

function needsOf(probe: ImportProbe): Need[] {
  return NEEDS[probe.family === 'video' ? (probe.video_kind ?? '5b') : probe.family] ?? []
}

/** File names that say which expert a file is, as the published ones do. */
const HIGH_NOISE = /high[\s._-]?noise/i
const LOW_NOISE = /low[\s._-]?noise/i

const WHAT: Record<ImportTarget, string> = { image: 'an image', video: 'a video' }

/** The download folders an import must never be assembled into. */
const RESERVED = MODEL_CATALOG.map((m) => m.folder.toLowerCase())

/** The pinned commits of the catalog, the only sources a folder is remembered for. */
const PINNED = new Set(MODEL_CATALOG.map((m) => `${m.repo}@${m.revision}`))

/** How many folders next to a configured one are looked into, so a crowded drive root stays cheap. */
const MAX_SIBLINGS = 200

async function manifestSource(folder: string): Promise<string | null> {
  try {
    const manifest = JSON.parse(await fs.readFile(path.join(folder, 'manifest.json'), 'utf8')) as Record<string, unknown>
    return typeof manifest['repo'] === 'string' && typeof manifest['revision'] === 'string'
      ? `${manifest['repo']}@${manifest['revision']}`
      : null
  } catch {
    return null
  }
}

/** The folders next to `folder`, the downloads usually kept side by side with it. */
async function siblings(folder: string): Promise<string[]> {
  const parent = path.dirname(folder)
  if (parent === folder) return []
  try {
    const entries = await fs.readdir(parent, { withFileTypes: true })
    return entries
      .filter((e) => e.isDirectory())
      .slice(0, MAX_SIBLINGS)
      .map((e) => path.join(parent, e.name))
  } catch {
    return []
  }
}

export class ModelImporter {
  private current: ModelImportState = IDLE_IMPORT
  private file: string | null = null
  private secondFile: string | null = null
  private controller: AbortController | null = null
  private readonly run: Runner

  constructor(private readonly options: ModelImporterOptions) {
    this.run = options.run ?? runProcess
  }

  get state(): ModelImportState {
    return this.current
  }

  get running(): boolean {
    return this.controller !== null
  }

  cancel(): void {
    if (this.current.phase === 'parts') this.options.downloads.cancel()
    this.controller?.abort()
  }

  clear(): void {
    if (this.running) throw new Error('An import is running. Stop it first.')
    this.file = null
    this.secondFile = null
    this.set(IDLE_IMPORT)
  }

  /** Read what the chosen file is, and which parts it still needs. `target`: the card it was chosen from. */
  async pick(file: string, target: ImportTarget = 'image'): Promise<void> {
    if (this.running) throw new Error('An import is already running.')
    this.file = null
    this.secondFile = null
    // A 14B expert's name without "high noise" or "low noise", so it names the model rather than the one file.
    const stem = path.basename(file).replace(/\.safetensors$/i, '').replace(/[\s._-]*(high|low)[\s._-]?noise[\s._-]*/i, '-')
    this.set({ ...IDLE_IMPORT, phase: 'probing', target, suggestedName: folderNameFor(stem) })
    let state = this.current
    const { probe, message } = await this.probe(file)
    if (!probe) {
      this.set({ ...state, phase: 'failed', message })
      return
    }
    state = { ...state, probe }
    if (targetOf(probe.family) !== target) {
      const other = targetOf(probe.family) === 'video' ? 'Video model' : 'Image model'
      this.set({ ...state, phase: 'failed', message: `This is the ${probe.label}, not ${WHAT[target]} model. Import it under ${other}.` })
      return
    }
    this.file = file
    const needs = (await this.sources(probe)).filter((s) => s.folder === null).map((s) => s.id)
    this.set({ ...state, phase: 'ready', needs })
  }

  /**
   * A 14B video model's other expert. The two are put in order by their names when the names
   * say which is which (high noise first); `swap` changes the order otherwise.
   */
  async pickSecond(file: string): Promise<void> {
    if (this.running) throw new Error('An import is already running.')
    const first = this.current.probe
    if (!this.file || !first || this.current.phase !== 'ready' || !isTwoExperts(first)) {
      throw new Error('Choose the first expert’s file first.')
    }
    if (path.resolve(file) === path.resolve(this.file)) throw new Error('That is the file already chosen; choose the other expert.')
    const { probe, message } = await this.probe(file)
    if (!probe) throw new Error(message)
    if (probe.video_kind !== first.video_kind) {
      throw new Error(`This is the ${probe.label}; the other expert of the ${first.label} is needed.`)
    }
    this.secondFile = file
    let state: ModelImportState = { ...this.current, second: probe, message: '' }
    if (LOW_NOISE.test(path.basename(this.file)) && HIGH_NOISE.test(path.basename(file))) state = this.swapped(state)
    this.set(state)
  }

  /** Exchange the two experts' order (which is the first, high-noise one). */
  swap(): void {
    if (this.running || !this.secondFile || !this.current.second) throw new Error('Choose both experts’ files first.')
    this.set(this.swapped(this.current))
  }

  private swapped(state: ModelImportState): ModelImportState {
    ;[this.file, this.secondFile] = [this.secondFile, this.file]
    return { ...state, probe: state.second, second: state.probe }
  }

  private async probe(file: string): Promise<{ probe: ImportProbe | null; message: string }> {
    let probe: ImportProbe | null = null
    let message = ''
    const { code, stderrTail } = await this.script(['--transformer', file, '--probe'], (event) => {
      if (event.event === 'probe') probe = event.probe
      if (event.event === 'error') message = event.message
    })
    if (message || !probe) return { probe: null, message: message || this.failure(code, stderrTail, 'The file could not be read.') }
    return { probe, message: '' }
  }

  /** Download what is missing, assemble the folder, and select it. */
  async start(options: ImportOptions): Promise<void> {
    const probe = this.current.probe
    if (this.running) throw new Error('An import is already running.')
    if (!this.file || !probe || this.current.phase !== 'ready') throw new Error('Choose a file to import first.')
    if (options.variant !== 'base' && options.variant !== 'distilled') throw new Error('Choose whether the model is distilled or a base model.')
    const sha256 = options.sha256.trim().toLowerCase()
    const sha256_2 = (options.sha256_2 ?? '').trim().toLowerCase()
    for (const hash of [sha256, sha256_2]) {
      if (hash && !/^[0-9a-f]{64}$/.test(hash)) throw new Error('A SHA-256 is 64 hexadecimal characters.')
    }
    const two = isTwoExperts(probe)
    if (two && !this.secondFile) throw new Error('Choose the second expert’s file first.')
    if (two && options.variant === 'distilled') {
      const problem = fewStepProblem(options.steps, options.shift)
      if (problem) throw new Error(problem)
    }
    if (!isValidFolderName(options.name, RESERVED)) {
      throw new Error('Use a folder name of letters, digits, dots, dashes and underscores (up to 64), not one of the download folders.')
    }
    if (this.current.needs.length && !options.acceptLicenses) throw new Error('Accept the licenses of the parts to download first.')

    const file = this.file
    const dest = path.join(this.options.modelsFolder(), options.name)
    const controller = new AbortController()
    this.controller = controller
    let state: ModelImportState = { ...this.current, message: '', dest: null }
    try {
      for (const id of this.current.needs) {
        this.set((state = { ...state, phase: 'parts', message: catalogModel(id)?.label ?? id }))
        await this.options.downloads.download(id, this.options.modelsFolder())
        if (controller.signal.aborted) break
      }
      if (controller.signal.aborted) {
        this.set({ ...state, phase: 'cancelled', message: 'Stopped. Importing again resumes the downloads.' })
        return
      }
      const sources = await this.sources(probe)
      const missing = sources.find((s) => s.folder === null)
      if (missing) throw new Error(`${catalogModel(missing.id)?.label ?? missing.id} is still missing.`)
      const args = ['--transformer', file, '--variant', options.variant, '--dest', dest]
      for (const s of sources) args.push(s.flag, s.folder as string)
      if (sha256) args.push('--sha256', sha256)
      if (two) {
        args.push('--transformer-2', this.secondFile as string)
        if (sha256_2) args.push('--sha256-2', sha256_2)
        if (options.variant === 'distilled') args.push('--steps', String(options.steps), '--shift', String(options.shift))
      }

      this.set((state = { ...state, phase: 'checking', done: 0, size: probe.size }))
      let finished = false
      const { code, stderrTail } = await this.script(args, (event) => {
        if (event.event === 'progress') state = { ...state, phase: event.stage, done: event.done, size: event.size }
        else if (event.event === 'done') finished = true
        else if (event.event === 'error') state = { ...state, phase: 'failed', message: event.message }
        else return
        this.set(state)
      }, controller.signal)
      if (controller.signal.aborted) {
        this.set({ ...state, phase: 'cancelled', message: 'Stopped. Files linked so far stay in the folder; importing again finishes it.' })
        return
      }
      if (!finished) {
        this.set({ ...state, phase: 'failed', message: state.phase === 'failed' ? state.message : this.failure(code, stderrTail, 'The import stopped.') })
        return
      }
      const target = targetOf(probe.family)
      await this.options.selectFolder(dest, target)
      this.set({ ...state, phase: 'done', dest, message: `Imported and selected. It loads with the next ${target}.` })
    } catch (err) {
      this.set({ ...this.current, phase: 'failed', message: (err as Error).message })
    } finally {
      this.controller = null
    }
  }

  /**
   * Each part the family needs, with a folder already holding it (verified at the pinned
   * commit), or null. Looked for in the folder configured for it, the download folder,
   * every verified folder seen before, and the folders next to any configured one; a
   * download kept outside the models folder is still found after the image model
   * folder has moved on to an imported model.
   */
  private async sources(probe: ImportProbe): Promise<{ id: ModelId; flag: string; folder: string | null }[]> {
    const configured = (await this.options.engineFolders().catch(() => null)) ?? {}
    const engineFolders = [configured.model_dir, configured.caption_model_dir, configured.video_model_dir].filter(
      (f): f is string => typeof f === 'string' && path.isAbsolute(f)
    )
    const known = this.options.knownFolders()
    const nearby = (await Promise.all(engineFolders.map(siblings))).flat()

    const seen = new Map<string, string | null>()
    const source = async (folder: string): Promise<string | null> => {
      if (!seen.has(folder)) seen.set(folder, await manifestSource(folder))
      return seen.get(folder) ?? null
    }

    const out = []
    for (const need of needsOf(probe)) {
      const model = catalogModel(need.id)
      const wanted = model ? `${model.repo}@${model.revision}` : ''
      const candidates = [
        need.configured ? configured[need.configured] : undefined,
        ...[need.id, ...need.also].map((id) => {
          const m = catalogModel(id)
          return m ? path.join(this.options.modelsFolder(), m.folder) : undefined
        }),
        ...known,
        ...nearby
      ]
      let folder: string | null = null
      for (const candidate of candidates) {
        if (candidate && (await source(candidate)) === wanted) {
          folder = candidate
          break
        }
      }
      out.push({ id: need.id, flag: need.flag, folder })
    }

    // Remember the verified downloads found, so switching folders later cannot hide them.
    const verified: string[] = []
    for (const folder of new Set([...engineFolders, ...out.map((s) => s.folder)])) {
      if (folder && !known.includes(folder) && PINNED.has((await source(folder)) ?? '')) verified.push(folder)
    }
    if (verified.length) await this.options.rememberFolders(verified).catch(() => undefined)
    return out
  }

  private async script(
    args: string[],
    onEvent: (event: NonNullable<ReturnType<typeof parseImportEvent>>) => void,
    signal?: AbortSignal
  ): Promise<{ code: number | null; stderrTail: string }> {
    let buffered = ''
    let stderrTail = ''
    const result = await this.run(this.options.python(), ['-I', '-u', this.options.script, ...args, '--json'], {
      ...(signal ? { signal } : {}),
      onOutput: (text, stream) => {
        if (stream === 'stderr') {
          const last = text.trim().split(/\r?\n/).pop()?.trim()
          if (last) stderrTail = last
          return
        }
        buffered += text
        const lines = buffered.split(/\r?\n/)
        buffered = lines.pop() ?? ''
        for (const line of lines) {
          const event = parseImportEvent(line)
          if (event) onEvent(event)
        }
      }
    })
    return { code: result.code, stderrTail }
  }

  private failure(code: number | null, stderrTail: string, fallback: string): string {
    if (code === null) return 'Python could not be started. Install the image engine first.'
    return stderrTail.replace(/^IMPORT FAILED:\s*/, '') || fallback
  }

  private set(state: ModelImportState): void {
    this.current = state
    this.options.onState(state)
  }
}
