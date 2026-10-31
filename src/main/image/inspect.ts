import path from 'node:path'
import { IDLE_INSPECT, feedInspect, parseInspectEvent, type ModelInspectState } from '@shared/inspect.js'
import { inspectGgufFile, isGgufFile } from '../gguf.js'
import { runProcess, type Runner } from './setup.js'

/**
 * Inspecting a model file the user picked (shared/inspect.ts).
 *
 * A GGUF file is read here, by the studio's own strict reader, with no Python
 * involved. Any other file goes to image-engine/scripts/inspect_model.py, run with
 * the engine's Python, on this machine only; this starts it, reads its JSON lines,
 * and keeps the latest state for the window. The path always comes from a native
 * dialog the main process showed, or is the chat model's own.
 */

export interface ModelInspectorOptions {
  python: () => string
  script: string
  run?: Runner
  onState: (state: ModelInspectState) => void
}

export class ModelInspector {
  private current: ModelInspectState = IDLE_INSPECT
  private controller: AbortController | null = null
  private readonly run: Runner

  constructor(private readonly options: ModelInspectorOptions) {
    this.run = options.run ?? runProcess
  }

  get state(): ModelInspectState {
    return this.current
  }

  get running(): boolean {
    return this.controller !== null
  }

  cancel(): void {
    this.controller?.abort()
  }

  async inspect(file: string): Promise<void> {
    if (this.controller) throw new Error('A file is already being inspected.')
    const controller = new AbortController()
    this.controller = controller
    let state: ModelInspectState = { ...IDLE_INSPECT, phase: 'header', file: path.basename(file) }
    this.set(state)
    try {
      if (await isGgufFile(file)) {
        try {
          const report = await inspectGgufFile(file, {
            signal: controller.signal,
            onProgress: (stage, done, size) => {
              state = feedInspect(state, { event: 'progress', stage, done, size })
              this.set(state)
            }
          })
          this.set(feedInspect(state, { event: 'report', report }))
        } catch (err) {
          this.set({ ...state, phase: controller.signal.aborted ? 'cancelled' : 'failed', message: controller.signal.aborted ? 'Stopped.' : (err as Error).message })
        }
        return
      }
      let buffered = ''
      let stderrTail = ''
      const result = await this.run(this.options.python(), ['-I', '-u', this.options.script, '--file', file, '--json'], {
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
            const event = parseInspectEvent(line)
            if (event) {
              state = feedInspect(state, event)
              this.set(state)
            }
          }
        }
      })
      if (controller.signal.aborted) {
        this.set({ ...state, phase: 'cancelled', message: 'Stopped.' })
        return
      }
      if (state.phase !== 'done') {
        const message =
          state.phase === 'failed'
            ? state.message
            : result.code === null
              ? 'Python could not be started. Install the image engine first.'
              : stderrTail || `The inspection stopped (exit code ${result.code}).`
        this.set({ ...state, phase: 'failed', message })
      }
    } finally {
      this.controller = null
    }
  }

  private set(state: ModelInspectState): void {
    this.current = state
    this.options.onState(state)
  }
}
