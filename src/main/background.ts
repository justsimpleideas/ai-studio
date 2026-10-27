import type { ModelSlot } from '@shared/config.js'

/**
 * Unloading models while the window is closed to the background.
 *
 * Each model the user chose not to keep loaded is unloaded as soon as it is
 * idle, never in the middle of work: a video that takes ten minutes keeps
 * going after the window closes, and its model goes once the queue is empty.
 * While the window stays hidden the check repeats, because queued work can
 * load a model again (an image job queued behind a video, for one).
 */

export interface UnloadTarget {
  slot: ModelSlot
  isLoaded(): boolean
  /** Why it cannot be unloaded right now (loading, answering, generating), or null. */
  busyReason(): string | null
  unload(): Promise<void>
}

export interface Timers {
  setInterval(run: () => void, ms: number): unknown
  clearInterval(handle: unknown): void
}

export const BACKGROUND_CHECK_MS = 3000

export class BackgroundUnloader {
  private handle: unknown = null
  private running = false

  constructor(
    private readonly targets: UnloadTarget[],
    /** Read on every check, so a changed setting applies at once. */
    private readonly keepLoaded: () => Record<ModelSlot, boolean>,
    private readonly onUnloaded: (slot: ModelSlot) => void = () => undefined,
    private readonly intervalMs = BACKGROUND_CHECK_MS,
    private readonly timers: Timers = {
      setInterval: (run, ms) => setInterval(run, ms),
      clearInterval: (handle) => clearInterval(handle as ReturnType<typeof setInterval>)
    }
  ) {}

  get active(): boolean {
    return this.handle !== null
  }

  /** The window went to the background: check now, then keep checking until stop(). */
  start(): void {
    if (this.handle !== null) return
    this.handle = this.timers.setInterval(() => void this.check(), this.intervalMs)
    void this.check()
  }

  /** The window is back: models that are still loaded stay loaded. */
  stop(): void {
    if (this.handle === null) return
    this.timers.clearInterval(this.handle)
    this.handle = null
  }

  /** One pass. Returns the slots it unloaded. Overlapping passes are skipped. */
  async check(): Promise<ModelSlot[]> {
    if (this.running) return []
    this.running = true
    const unloaded: ModelSlot[] = []
    try {
      const keep = this.keepLoaded()
      for (const target of this.targets) {
        if (this.handle === null) break // shown again while this pass was running
        if (keep[target.slot] || !target.isLoaded() || target.busyReason() !== null) continue
        try {
          await target.unload()
          unloaded.push(target.slot)
          this.onUnloaded(target.slot)
        } catch {
          // The next pass tries again (an engine that is restarting, for one).
        }
      }
    } finally {
      this.running = false
    }
    return unloaded
  }
}
