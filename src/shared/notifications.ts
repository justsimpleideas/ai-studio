import type { ImageJob, ImageJobStatus } from './image.js'

/**
 * What the tray says when something finishes while the window is hidden.
 *
 * The notice names what is done, never what it is about: it shows on the
 * desktop, over whatever else is there, so no prompt, reply or file name goes
 * in it.
 */

export const CHAT_REPLY_NOTICE = 'A chat reply is ready. Click to open the studio.'

/**
 * The notice for an image engine job that has just finished, or null. A job is
 * announced once, on the update that moves it to "done"; training tasks and
 * failed or cancelled jobs are not announced.
 */
export function mediaNotice(job: ImageJob, previous: ImageJobStatus | undefined): string | null {
  if (job.status !== 'done' || previous === 'done' || job.kind === 'task') return null
  const count = job.images.length > 0 ? job.images.length : job.count
  const what = job.kind === 'video' ? (count > 1 ? `${count} videos are` : 'Your video is') : count > 1 ? `${count} images are` : 'Your image is'
  return `${what} ready. Click to open the studio.`
}

/**
 * Follows job statuses from the engine's event stream, so a job is announced
 * only when it changes to done. The list the engine sends on (re)connecting is
 * taken as already known: whatever finished before then is not news.
 */
export class JobWatcher {
  private readonly statuses = new Map<string, ImageJobStatus>()

  reset(jobs: readonly ImageJob[]): void {
    this.statuses.clear()
    for (const job of jobs) this.statuses.set(job.id, job.status)
  }

  /** The notice this update calls for, if any. */
  update(job: ImageJob): string | null {
    const notice = mediaNotice(job, this.statuses.get(job.id))
    this.statuses.set(job.id, job.status)
    return notice
  }
}
