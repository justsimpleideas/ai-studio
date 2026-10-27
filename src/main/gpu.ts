/**
 * One model on the GPU at a time (decisions S4).
 *
 * The chat model and the image model do not fit on a 12 GB card together, and
 * the studio does not try to run them side by side. Before one side loads, the
 * other is unloaded. If the other side is in the middle of something — a reply
 * streaming, an image generating, a model loading — the switch is refused with
 * a reason instead of interrupting that work.
 */

export type GpuOwner = 'chat' | 'image'

export interface GpuSide {
  /** Human name used in messages, e.g. "chat model". */
  label: string
  isLoaded(): boolean
  /** Why this side cannot be unloaded right now, or null if it can. */
  busyReason(): string | null
  unload(): Promise<void>
}

export class GpuBusyError extends Error {}

export class GpuArbiter {
  private chain: Promise<unknown> = Promise.resolve()

  constructor(
    private readonly sides: Record<GpuOwner, GpuSide>,
    private readonly notify: (message: string) => void
  ) {}

  /**
   * Make room for `owner`. Resolves when the other side is unloaded (or was not
   * loaded), rejects with GpuBusyError if it is busy. Claims are serialized, so
   * two quick clicks cannot both decide that the GPU is free.
   */
  claim(owner: GpuOwner): Promise<void> {
    const run = this.chain.then(() => this.makeRoom(owner))
    this.chain = run.catch(() => undefined)
    return run
  }

  private async makeRoom(owner: GpuOwner): Promise<void> {
    const other = this.sides[owner === 'chat' ? 'image' : 'chat']
    if (!other.isLoaded()) return
    const reason = other.busyReason()
    if (reason) {
      throw new GpuBusyError(`${reason} Wait for it to finish, or stop it, before loading the ${this.sides[owner].label}.`)
    }
    const label = other.label // read first: a side may name what it holds, which unloading changes
    await other.unload()
    this.notify(`Unloaded the ${label} to make room for the ${this.sides[owner].label}.`)
  }
}
