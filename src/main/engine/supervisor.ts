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
  /**
   * The engine's last lines of output. llama.cpp says why a load failed on
   * stderr ("failed to allocate…", "unknown model architecture"), and in a
   * packaged app nothing shows stderr, so it is kept here for the error the user
   * sees and passed on for the log.
   */
  private recent: string[] = []
  /** Extra environment for the engine process; kept for every later start too. */
  private env: Record<string, string> = {}
  /** Set while the engine is being restarted on purpose, so its exit is not reported as a failure. */
  private restarting = false
  /** Which host runs: the built-in library (engineHost.js) or the llama-server client (serverHost.js). */
  private entryName = 'engineHost.js'

  constructor(
    private readonly onMessage: (message: EngineToMain) => void,
    private readonly onExit: (code: number) => void,
    private readonly onOutput: (line: string) => void = () => undefined
  ) {}

  get entry(): string {
    return this.entryName
  }

  /**
   * Switch to the other host. A running process of the old kind is ended and
   * the new one started; whatever is sent meanwhile waits for it.
   */
  useEntry(entry: string): void {
    if (entry === this.entryName) return
    this.entryName = entry
    if (!this.child) return
    this.restarting = true
    this.ready = false
    this.child.kill()
  }

  /** The engine's last few lines of output, newest last. */
  lastOutput(lines = 6): string[] {
    return this.recent.slice(-lines)
  }

  private take(chunk: Buffer | string): void {
    for (const raw of chunk.toString().split(/\r?\n/)) {
      const line = raw.trimEnd()
      if (!line) continue
      this.recent.push(line)
      if (this.recent.length > 200) this.recent.splice(0, this.recent.length - 200)
      this.onOutput(line)
    }
  }

  get isRunning(): boolean {
    return this.child !== null
  }

  start(): void {
    if (this.child) return
    const entry = path.join(import.meta.dirname, this.entryName)
    const child = utilityProcess.fork(entry, [], {
      serviceName: 'local-llm-agent-engine',
      stdio: 'pipe',
      env: { ...process.env, ...this.env }
    })
    this.child = child
    this.recent = []
    child.stdout?.on('data', (chunk: Buffer) => this.take(chunk))
    child.stderr?.on('data', (chunk: Buffer) => this.take(chunk))

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
      if (this.restarting) {
        // Messages sent meanwhile are kept for the new process.
        this.restarting = false
        this.start()
        return
      }
      this.queue = []
      this.onExit(code)
    })
  }

  /**
   * Start the engine again with more environment, keeping it for later starts.
   * Whatever is sent before the new process is ready waits for it.
   */
  restartWith(env: Record<string, string>): void {
    this.env = { ...this.env, ...env }
    if (!this.child) {
      this.start()
      return
    }
    this.restarting = true
    this.ready = false
    this.child.kill()
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
