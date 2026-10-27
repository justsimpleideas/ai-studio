import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import type { NetworkLogEntry } from '@shared/network.js'

/**
 * A workspace's internet log: one JSON line per request the agent asked for,
 * appended and never rewritten. Refused and declined requests are logged too,
 * so the log answers "what did it try to send", not only "what went out".
 */
export class NetworkLog {
  private queue: Promise<void> = Promise.resolve()

  constructor(
    private readonly file: string,
    private readonly onChange: () => void = () => undefined
  ) {}

  append(entry: NetworkLogEntry): Promise<void> {
    this.queue = this.queue
      .catch(() => undefined)
      .then(async () => {
        await fsp.mkdir(path.dirname(this.file), { recursive: true })
        await fsp.appendFile(this.file, JSON.stringify(entry) + '\n', 'utf8')
      })
      .finally(() => this.onChange())
    return this.queue
  }

  /** The most recent entries, newest first. Unreadable lines are skipped. */
  recent(limit = 200): NetworkLogEntry[] {
    let text: string
    try {
      text = fs.readFileSync(this.file, 'utf8')
    } catch {
      return []
    }
    const out: NetworkLogEntry[] = []
    const lines = text.split('\n')
    for (let i = lines.length - 1; i >= 0 && out.length < limit; i--) {
      const line = lines[i]?.trim()
      if (!line) continue
      try {
        out.push(JSON.parse(line) as NetworkLogEntry)
      } catch {
        // A line cut short by a crash; the rest of the log is still good.
      }
    }
    return out
  }
}
