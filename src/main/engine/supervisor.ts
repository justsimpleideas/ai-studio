import path from 'node:path'
import { utilityProcess, type UtilityProcess } from 'electron'
import type { EngineToMain, MainToEngine } from '@shared/engine-protocol.js'

/**
 * Owning the model process.
 *
 * The supervisor exists so that the rest of the application can treat the
 * engine as a thing that is either there or not, rather than as a native
 * library that might take everything down with it. A GGUF built for a newer
 * llama.cpp, a context size that does not fit in VRAM, a driver that gives up
 * mid-load: all of these end as a process exit here, reported as an error the
 * user can read and recover from.
 */
export class EngineSupervisor {
  private child: UtilityProcess | null = null
  private ready = false
  private queue: MainToEngine[] = []

  constructor(
    private readonly onMessage: (message: EngineToMain) => void,
    private readonly onExit: (code: number) => void
  ) {}

  get isRunning(): boolean {
    return this.child !== null
  }

  start(): void {
    if (this.child) return
    const entry = path.join(import.meta.dirname, 'engineHost.js')
    const child = utilityProcess.fork(entry, [], {
      serviceName: 'local-llm-agent-engine',
      stdio: 'inherit'
    })
    this.child = child

    child.on('message', (message: EngineToMain) => {
      if (message.type === 'ready') {
        this.ready = true
        for (const queued of this.queue.splice(0)) child.postMessage(queued)
        return
      }
      this.onMessage(message)
    })

    child.on('exit', (code) => {
      this.child = null
      this.ready = false
      this.queue = []
      this.onExit(code)
    })
  }

  send(message: MainToEngine): void {
    if (!this.child) {
      this.start()
    }
    if (!this.ready) {
      // Messages sent while the process is still coming up are held rather than
      // dropped, so "load the model I used last time" at startup works without
      // the caller having to know about process lifecycle.
      this.queue.push(message)
      return
    }
    this.child?.postMessage(message)
  }

  /** Kill the engine outright. Used on quit and when recovering from a wedge. */
  stop(): void {
    this.child?.kill()
    this.child = null
    this.ready = false
    this.queue = []
  }
}
