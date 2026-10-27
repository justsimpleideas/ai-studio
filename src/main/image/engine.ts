import { spawn, type ChildProcess } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import {
  IMAGE_TOKEN_HEADER,
  SseParser,
  rewriteEvent,
  type ImageEngineEvent,
  type ImageEngineState,
  type ImageJob,
  type ImageModelState,
  type ImageStatus,
  type MediaKind
} from '@shared/image.js'

/**
 * Owning the Python image engine.
 *
 * Started on demand, with a token generated for this launch and handed over
 * through the environment (never the command line, which other programs can
 * read). The engine exits on its own when its stdin closes, so it cannot outlive
 * the studio even if the studio is killed.
 */

export interface ImageEngineOptions {
  python: () => string
  /** Folder containing the `image_engine` package. */
  engineDir: string
  dataDir: string
  port: number
  /** `image_engine`, or `tests.ui_harness` for end-to-end tests. */
  module: string
  /** The studio's version, recorded in every image the engine makes. */
  appVersion: string
  onEvent: (event: ImageEngineEvent) => void
  onState: (state: ImageEngineState) => void
}

const START_TIMEOUT_MS = 90_000
const STDERR_KEEP = 4000
const ALLOWED_MODULES = new Set(['image_engine', 'tests.ui_harness'])

export class ImageEngineError extends Error {
  constructor(message: string, readonly status: number) {
    super(message)
  }
}

export class ImageEngine {
  private child: ChildProcess | null = null
  private token = ''
  private current: ImageEngineState = { state: 'stopped' }
  private starting: Promise<void> | null = null
  private stderrTail = ''
  private events: AbortController | null = null
  private heldFor: string | null = null

  /** Mirrors of engine state, kept from its event stream for the GPU arbiter. */
  models: Record<MediaKind, ImageModelState> = { image: 'unloaded', video: 'unloaded' }
  private jobs = new Map<string, ImageJob>()

  constructor(private readonly options: ImageEngineOptions) {
    if (!ALLOWED_MODULES.has(options.module)) throw new Error(`Unknown image engine module: ${options.module}`)
  }

  get state(): ImageEngineState {
    return this.current
  }

  /** The image model's state (what there was before video). */
  get modelState(): ImageModelState {
    return this.models.image
  }

  get busyReason(): string | null {
    if (this.models.image === 'loading') return 'The image model is loading.'
    if (this.models.video === 'loading') return 'The video model is loading.'
    const active = [...this.jobs.values()].filter((j) => j.status === 'queued' || j.status === 'running')
    if (active.length > 0) {
      const what = active.some((j) => j.kind === 'video') ? 'Video generation' : 'Image generation'
      return `${what} is in progress (${active.length} job${active.length > 1 ? 's' : ''}).`
    }
    return null
  }

  /** The model on the GPU right now, if any (the engine keeps at most one). */
  get loadedKind(): MediaKind | null {
    for (const kind of ['video', 'image'] as const) {
      if (this.models[kind] === 'ready' || this.models[kind] === 'loading') return kind
    }
    return null
  }

  get modelLoaded(): boolean {
    return this.loadedKind !== null
  }

  private resetModels(): void {
    this.models = { image: 'unloaded', video: 'unloaded' }
  }

  private takeStatus(status: Pick<ImageStatus, 'model' | 'models'>): void {
    this.models = {
      image: status.models?.image?.state ?? status.model.state,
      video: status.models?.video?.state ?? 'unloaded'
    }
  }

  /**
   * Stop the engine and keep it from starting again until release(): its Python
   * environment is being replaced, and a running engine holds files pip has to
   * change (Windows locks them). Resolves once the process has exited.
   */
  async hold(reason: string): Promise<void> {
    this.heldFor = reason
    await this.stop()
  }

  release(): void {
    this.heldFor = null
  }

  start(): Promise<void> {
    if (this.heldFor) return Promise.reject(new Error(this.heldFor))
    if (this.current.state === 'running') return Promise.resolve()
    if (this.starting) return this.starting
    this.starting = this.launch().finally(() => {
      this.starting = null
    })
    return this.starting
  }

  private async launch(): Promise<void> {
    this.token = randomBytes(32).toString('hex')
    this.stderrTail = ''
    this.setState({ state: 'starting' })

    const { python, engineDir, dataDir, port, module, appVersion } = this.options
    let child: ChildProcess
    try {
      child = spawn(python(), ['-m', module, '--data-dir', dataDir, '--port', String(port), '--watch-stdin'], {
        cwd: engineDir,
        env: {
          ...process.env,
          STUDIO_IMAGE_TOKEN: this.token,
          STUDIO_APP_VERSION: appVersion,
          PYTHONUNBUFFERED: '1',
          PYTHONIOENCODING: 'utf-8'
        },
        stdio: ['pipe', 'pipe', 'pipe'],
        windowsHide: true
      })
    } catch (err) {
      return this.fail(`Could not start Python: ${(err as Error).message}`)
    }
    this.child = child

    let exited: string | null = null
    child.stdout?.on('data', () => undefined)
    child.stderr?.on('data', (chunk: Buffer) => {
      this.stderrTail = (this.stderrTail + chunk.toString('utf8')).slice(-STDERR_KEEP)
    })
    child.on('error', (err) => {
      exited = `Could not start Python (${err.message}). Choose the Python environment in the image side's Settings.`
    })
    child.on('exit', (code) => {
      exited ??= this.describeExit(code)
      if (this.child === child) {
        this.child = null
        this.events?.abort()
        this.jobs.clear()
        this.resetModels()
        if (this.current.state === 'running') this.setState({ state: 'failed', message: exited })
      }
    })

    const deadline = Date.now() + START_TIMEOUT_MS
    while (Date.now() < deadline) {
      if (exited) return this.fail(exited)
      try {
        // Bounded too: an engine that accepts the connection but never answers must still time out.
        const signal = AbortSignal.timeout(Math.max(1, deadline - Date.now()))
        const status = (await this.request('GET', '/api/status', undefined, signal)) as ImageStatus
        this.takeStatus(status)
        this.setState({ state: 'running' })
        this.listen()
        return
      } catch {
        await new Promise((r) => setTimeout(r, 300))
      }
    }
    this.stop()
    return this.fail('The image engine did not answer within 90 seconds.')
  }

  private describeExit(code: number | null): string {
    const tail = this.stderrTail.trim().split('\n').slice(-6).join('\n')
    if (/No module named ['"]?(torch|diffusers|PIL)/.test(tail)) {
      return `The Python environment is missing packages the image engine needs.\n${tail}`
    }
    if (/Cannot listen/.test(tail)) {
      return `Port ${this.options.port} is already in use, probably by another copy of the studio.`
    }
    return `The image engine stopped (exit code ${code ?? 'none'}).${tail ? `\n${tail}` : ''}`
  }

  private fail(message: string): Promise<never> {
    this.setState({ state: 'failed', message })
    return Promise.reject(new Error(message))
  }

  private setState(state: ImageEngineState): void {
    this.current = state
    this.options.onState(state)
  }

  /** Call the engine's API. Throws ImageEngineError with the engine's own message. */
  async request(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<unknown> {
    return this.parse(await this.raw(method, path, body, signal))
  }

  /** POST raw bytes (a start picture) with their content type; parsed like request(). */
  async upload(path: string, bytes: Uint8Array, contentType: string): Promise<unknown> {
    const response = await fetch(`http://127.0.0.1:${this.options.port}${path}`, {
      method: 'POST',
      headers: { [IMAGE_TOKEN_HEADER]: this.token, 'Content-Type': contentType },
      body: bytes
    })
    return this.parse(response)
  }

  private async parse(response: Response): Promise<unknown> {
    const text = await response.text()
    const data: unknown = text ? JSON.parse(text) : null
    if (!response.ok) {
      const message =
        data && typeof data === 'object' && 'error' in data ? String((data as { error: unknown }).error) : `HTTP ${response.status}`
      throw new ImageEngineError(message, response.status)
    }
    return data
  }

  /** A raw response, used to stream image and video bytes to the window. */
  raw(method: string, path: string, body?: unknown, signal?: AbortSignal, extraHeaders?: Record<string, string>): Promise<Response> {
    const headers: Record<string, string> = { ...extraHeaders, [IMAGE_TOKEN_HEADER]: this.token }
    if (body !== undefined) headers['Content-Type'] = 'application/json'
    return fetch(`http://127.0.0.1:${this.options.port}${path}`, {
      method,
      headers,
      body: body === undefined ? null : JSON.stringify(body),
      signal: signal ?? null
    })
  }

  /** Follow the engine's event stream for as long as it runs, reconnecting if it drops. */
  private listen(): void {
    const controller = new AbortController()
    this.events = controller
    const run = async (): Promise<void> => {
      while (!controller.signal.aborted && this.current.state === 'running') {
        try {
          const response = await this.raw('GET', '/api/events', undefined, controller.signal)
          if (!response.ok || !response.body) throw new Error(`events: HTTP ${response.status}`)
          const parser = new SseParser()
          const decoder = new TextDecoder()
          for await (const chunk of response.body) {
            for (const data of parser.feed(decoder.decode(chunk as Uint8Array, { stream: true }))) {
              this.handle(JSON.parse(data) as ImageEngineEvent)
            }
          }
        } catch {
          if (controller.signal.aborted) return
        }
        await new Promise((r) => setTimeout(r, 1000))
      }
    }
    void run()
  }

  private handle(event: ImageEngineEvent): void {
    switch (event.type) {
      case 'hello':
        this.takeStatus(event.status)
        this.jobs = new Map(event.jobs.map((j) => [j.id, j]))
        break
      case 'job':
        this.jobs.set(event.job.id, event.job)
        break
      case 'model':
        this.models = { ...this.models, [event.kind ?? 'image']: event.state }
        break
    }
    this.options.onEvent(rewriteEvent(event))
  }

  /**
   * Stop the engine: close its stdin so it shuts down cleanly, then make sure.
   * Resolves once the process has exited (callers that do not care need not wait).
   */
  stop(): Promise<void> {
    const child = this.child
    this.events?.abort()
    this.events = null
    this.child = null
    this.jobs.clear()
    this.resetModels()
    if (this.current.state !== 'stopped') this.setState({ state: 'stopped' })
    if (!child || child.exitCode !== null || child.signalCode !== null) return Promise.resolve()
    return new Promise((resolve) => {
      const timer = setTimeout(() => child.kill(), 3000)
      child.once('exit', () => {
        clearTimeout(timer)
        resolve()
      })
      child.stdin?.end()
    })
  }
}
