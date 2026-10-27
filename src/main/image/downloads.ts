import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import {
  DownloadTracker,
  IDLE_DOWNLOAD,
  catalogModel,
  parseFetchEvent,
  type CatalogModel,
  type ModelDownloadState,
  type ModelId
} from '@shared/setup.js'
import { runProcess, type Runner } from './setup.js'

/**
 * Downloading a supported model from inside the studio.
 *
 * The work is done by image-engine/scripts/fetch_model.py, the same script the
 * README documents: one pinned commit, data formats only, every file checked
 * against its published hash. This only starts it, reads its JSON progress, and
 * points the engine at the folder once everything has been verified. Stopping
 * leaves the partial files in place, so starting again resumes.
 */

export interface ModelDownloadsOptions {
  python: () => string
  script: string
  run?: Runner
  /** Free bytes on the drive holding `folder`, or null if unknown. */
  freeBytes?: (folder: string) => Promise<number | null>
  /** Another folder holding a model whose identical files can be linked instead of downloaded. */
  reuseSource: (model: CatalogModel, dest: string) => Promise<string | null>
  /** Point the engine at the verified folder. */
  applyFolder: (model: CatalogModel, dest: string) => Promise<void>
  onState: (state: ModelDownloadState) => void
}

export async function freeBytesOn(folder: string): Promise<number | null> {
  // statfs needs an existing path; walk up to the nearest one.
  let probe = path.resolve(folder)
  while (!fs.existsSync(probe)) {
    const parent = path.dirname(probe)
    if (parent === probe) return null
    probe = parent
  }
  try {
    const stats = await fsp.statfs(probe)
    return stats.bavail * stats.bsize
  } catch {
    return null
  }
}

/** Bytes already in a folder (a partial download), counting hard links like any file. */
export async function bytesIn(folder: string): Promise<number> {
  let total = 0
  const walk = async (dir: string): Promise<void> => {
    let entries: fs.Dirent[]
    try {
      entries = await fsp.readdir(dir, { withFileTypes: true })
    } catch {
      return
    }
    for (const entry of entries) {
      const full = path.join(dir, entry.name)
      if (entry.isDirectory()) await walk(full)
      else if (entry.isFile()) total += (await fsp.stat(full)).size
    }
  }
  await walk(folder)
  return total
}

/** A little headroom on top of the model, for the manifest and the drive itself. */
const SPACE_MARGIN = 1024 ** 3

export class ModelDownloads {
  private current: ModelDownloadState = IDLE_DOWNLOAD
  private controller: AbortController | null = null
  private readonly run: Runner

  constructor(private readonly options: ModelDownloadsOptions) {
    this.run = options.run ?? runProcess
  }

  get state(): ModelDownloadState {
    return this.current
  }

  get running(): boolean {
    return this.controller !== null
  }

  cancel(): void {
    this.controller?.abort()
  }

  async download(id: ModelId, parentFolder: string): Promise<void> {
    if (this.controller) throw new Error('A model is already being downloaded.')
    const model = catalogModel(id)
    if (!model) throw new Error('Unknown model.')
    const dest = path.join(parentFolder, model.folder)
    const controller = new AbortController()
    this.controller = controller
    const tracker = new DownloadTracker(model.id, dest, model.bytes)
    this.set(tracker.state)
    try {
      const reuse = await this.options.reuseSource(model, dest)
      if (!reuse) await this.checkSpace(model, dest)

      const args = ['-I', '-u', this.options.script, '--model', model.id, '--dest', dest, '--json']
      if (reuse) args.push('--reuse-from', reuse)
      let buffered = ''
      let stderrTail = ''
      const result = await this.run(this.options.python(), args, {
        signal: controller.signal,
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
            const event = parseFetchEvent(line)
            if (event) this.set(tracker.feed(event))
          }
        }
      })
      if (controller.signal.aborted) {
        this.set({ ...tracker.state, phase: 'cancelled', activity: null, bytesPerSecond: null, message: 'Stopped. Downloading again resumes from here.' })
        return
      }
      if (result.code !== 0 || tracker.state.phase !== 'done') {
        const message =
          tracker.state.phase === 'failed'
            ? tracker.state.message
            : result.code === null
              ? 'Python could not be started. Install the image engine first.'
              : stderrTail || `The download stopped (exit code ${result.code}).`
        this.set({ ...tracker.state, phase: 'failed', activity: null, bytesPerSecond: null, message })
        throw new Error(message)
      }
      await this.options.applyFolder(model, dest)
      this.set({ ...tracker.state, message: 'Downloaded, verified, and selected.' })
    } catch (err) {
      if (this.current.phase !== 'failed') {
        this.set({ ...this.current, phase: 'failed', activity: null, bytesPerSecond: null, message: (err as Error).message })
      }
      throw err
    } finally {
      this.controller = null
    }
  }

  private async checkSpace(model: CatalogModel, dest: string): Promise<void> {
    const free = await (this.options.freeBytes ?? freeBytesOn)(dest)
    if (free === null) return
    const needed = model.bytes - (await bytesIn(dest)) + SPACE_MARGIN
    if (free < needed) {
      const gb = (n: number): string => `${(n / 1e9).toFixed(1)} GB`
      throw new Error(`Not enough free space: this model needs ${gb(needed)} more, and the drive has ${gb(free)}. Choose another folder.`)
    }
  }

  private set(state: ModelDownloadState): void {
    this.current = state
    this.options.onState(state)
  }
}
