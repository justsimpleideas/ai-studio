import fsp from 'node:fs/promises'
import path from 'node:path'
import type { Subject } from '@shared/subject.js'
import {
  IDLE_COLLECT,
  feedCollect,
  parseCheckEvent,
  parseCollectEvent,
  targetLabel,
  type CollectCheck,
  type CollectPlan,
  type CollectState
} from '@shared/training.js'
import { freeBytesOn } from './downloads.js'
import { runProcess, type Runner } from './setup.js'

/**
 * Collecting openly licensed photos for a training project, on the user's request.
 *
 * The work is done by image-engine/scripts/collect_photos.py, run in the engine's
 * Python the way fetch_model.py is: this writes the plan, starts the script, reads
 * its JSON progress, and once it is done makes a training project from the folder.
 * The engine's server itself never reaches the network. Stopping keeps what was
 * saved; collecting again into the same folder resumes.
 */

export interface PhotoCollectorOptions {
  python: () => string
  script: string
  /** Where plans are written for the script to read. */
  workDir: string
  run?: Runner
  freeBytes?: (folder: string) => Promise<number | null>
  /** Test hook: the end-to-end test points these at a local stand-in for the real services. */
  apis?: { commons?: string; openverse?: string }
  /** Make (or find) the training project for the folder and scan it; returns its id. */
  makeProject: (name: string, folder: string, subject: Subject) => Promise<string>
  onState: (state: CollectState) => void
}

export class PhotoCollector {
  private current: CollectState = IDLE_COLLECT
  private controller: AbortController | null = null
  private readonly run: Runner

  constructor(private readonly options: PhotoCollectorOptions) {
    this.run = options.run ?? runProcess
  }

  get state(): CollectState {
    return this.current
  }

  get running(): boolean {
    return this.controller !== null
  }

  cancel(): void {
    this.controller?.abort()
  }

  /** How many photos each row would likely get: the script asks the sources and downloads nothing. */
  async check(plan: CollectPlan): Promise<CollectCheck[]> {
    if (this.controller) throw new Error('Photos are being collected; check again when that is done.')
    const controller = new AbortController()
    this.controller = controller
    try {
      const planFile = await this.writePlan(plan, 'check-plan.json')
      const results: CollectCheck[] = []
      let buffered = ''
      let failure: string | null = null
      let stderrTail = ''
      const args = ['-I', '-u', this.options.script, '--plan', planFile, '--check', '--json', ...this.extraArgs()]
      const result = await this.run(this.options.python(), args, {
        signal: controller.signal,
        onOutput: (text, stream) => {
          if (stream === 'stderr') {
            stderrTail = text.trim().split(/\r?\n/).pop()?.trim() || stderrTail
            return
          }
          buffered += text
          const lines = buffered.split(/\r?\n/)
          buffered = lines.pop() ?? ''
          for (const line of lines) {
            const event = parseCollectEvent(line)
            if (!event) continue
            if (event.event === 'error') failure = typeof event.message === 'string' ? event.message : 'The check failed.'
            if (event.event === 'walking') this.set(feedCollect(this.current, event))
            const found = parseCheckEvent(event)
            if (found) results.push(found)
          }
        }
      })
      if (failure) throw new Error(failure)
      if (result.code !== 0) {
        throw new Error(result.code === null
          ? 'Python could not be started. Install the image engine first.'
          : stderrTail || `The check stopped (exit code ${result.code}).`)
      }
      return results
    } finally {
      this.controller = null
      if (this.current.walking) this.set({ ...this.current, walking: null })
    }
  }

  private async writePlan(plan: CollectPlan, name: string): Promise<string> {
    await fsp.mkdir(this.options.workDir, { recursive: true })
    const planFile = path.join(this.options.workDir, name)
    await fsp.writeFile(planFile, JSON.stringify({
      levels: plan.subject.levels.map((l) => ({ key: l.key, required: l.required })),
      targets: plan.targets,
      max_total_bytes: plan.max_total_bytes,
      sources: plan.sources,
      share_alike: plan.share_alike,
      contact: plan.contact
    }), 'utf8')
    return planFile
  }

  /** The test services when set, and a cache of walked category trees shared by checks and collections. */
  private extraArgs(): string[] {
    const args = ['--cache', path.join(this.options.workDir, 'trees')]
    if (this.options.apis?.commons) args.push('--commons-api', this.options.apis.commons)
    if (this.options.apis?.openverse) args.push('--openverse-api', this.options.apis.openverse)
    return args
  }

  async collect(plan: CollectPlan, folder: string): Promise<void> {
    if (this.controller) throw new Error('Photos are already being collected.')
    const controller = new AbortController()
    this.controller = controller
    this.set({ ...IDLE_COLLECT, phase: 'running', folder, maxBytes: plan.max_total_bytes,
      targets: plan.targets.map((t) => ({ label: targetLabel(t), have: 0, want: t.count })) })
    try {
      const free = await (this.options.freeBytes ?? freeBytesOn)(folder)
      if (free !== null && free < plan.max_total_bytes + 512 * 1024 ** 2) {
        throw new Error(`Not enough free space for a ${plan.max_total_bytes / 1024 ** 3} GB limit: the drive has ${(free / 1024 ** 3).toFixed(1)} GB. Choose a smaller limit or another folder.`)
      }
      const planFile = await this.writePlan(plan, 'collect-plan.json')
      const args = ['-I', '-u', this.options.script, '--plan', planFile, '--dest', folder, '--json', ...this.extraArgs()]
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
            const event = parseCollectEvent(line)
            if (event) this.set(feedCollect(this.current, event))
          }
        }
      })
      if (controller.signal.aborted) {
        this.set({ ...this.current, phase: 'cancelled', current: null, message: 'Stopped. Collecting into the same folder again resumes from here.' })
        return
      }
      if (result.code !== 0 || this.current.phase !== 'done') {
        const message = this.current.phase === 'failed' && this.current.message
          ? this.current.message
          : result.code === null
            ? 'Python could not be started. Install the image engine first.'
            : stderrTail || `Collecting stopped (exit code ${result.code}).`
        throw new Error(message)
      }
      const projectId = await this.options.makeProject(plan.project, folder, plan.subject)
      const done = this.current
      this.set({ ...done, projectId, message: done.photos === 0
        ? 'No photos with a suitable licence were found. Try a Commons category, or other names.'
        : `${done.photos} photos collected${done.sizeLimitReached ? ' (the size limit was reached)' : ''}. The training project is ready.` })
    } catch (err) {
      this.set({ ...this.current, phase: 'failed', current: null, message: (err as Error).message })
      throw err
    } finally {
      this.controller = null
    }
  }

  private set(state: CollectState): void {
    this.current = state
    this.options.onState(state)
  }
}
