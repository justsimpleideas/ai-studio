import fs from 'node:fs'
import path from 'node:path'
import { comparablePath, isContained } from './broker/paths.js'
import type { DataLocation, DataLocationKey, DataLocationsView } from '@shared/locations.js'

/**
 * The one file that says where the studio's own folders are.
 *
 * It cannot live in the folder it names: the chats folder holds config.json,
 * so the setting that moves that folder has to sit outside it. It goes next to
 * the default folders, in the application data folder, which never moves.
 *
 * Only new data follows a change. Nothing is copied and nothing is deleted:
 * choosing a folder means "write here from now on", and choosing the old one
 * back brings everything in it back into view (docs/decisions.md S18).
 */

export const LOCATION_FILE = 'data-location.json'

/** Where each folder goes when nothing has been chosen, under the application data folder. */
export function defaultLocations(appData: string): Record<DataLocationKey, string> {
  return {
    userData: path.join(appData, 'userData'),
    outputs: path.join(appData, 'image', 'outputs')
  }
}

/**
 * Why `folder` cannot be used for `key`, or null when it can.
 *
 * The two folders may not be the same, nor sit inside one another: the gallery
 * puts a database and megabytes of media where it is pointed, and the chats
 * folder is read whole at startup. Neither may go in the program folder, which
 * an update or an uninstall empties.
 */
export function locationProblem(
  key: DataLocationKey,
  folder: string,
  others: { other: string; programFolders: string[] }
): string | null {
  if (!folder || !path.isAbsolute(folder)) return 'Choose a folder on a drive.'
  const name = key === 'userData' ? 'chats' : 'gallery'
  const otherName = key === 'userData' ? 'gallery' : 'chats'
  if (comparablePath(folder) === comparablePath(others.other) || isContained(folder, others.other) || isContained(others.other, folder)) {
    return `The ${name} folder and the ${otherName} folder cannot be the same, or one inside the other.`
  }
  for (const program of others.programFolders) {
    if (program && isContained(folder, program)) {
      return 'That folder is inside the program itself, which an update or an uninstall empties. Choose one outside it.'
    }
  }
  return null
}

interface Chosen {
  userData: string | null
  outputs: string | null
}

export class DataLocationStore {
  private readonly file: string
  private readonly fallbacks: Record<DataLocationKey, string>
  private chosen: Chosen
  /** The chats folder the running studio actually opened, which only a restart changes. */
  private readonly opened: string

  constructor(
    private readonly appData: string,
    private readonly programFolders: string[] = []
  ) {
    this.file = path.join(appData, LOCATION_FILE)
    this.fallbacks = defaultLocations(appData)
    this.chosen = this.read()
    this.opened = this.get('userData')
  }

  /**
   * Read at startup, before anything else opens a folder. A file that does not
   * parse, or names something that is not a usable path, means the defaults:
   * the studio has to start even when this file has been edited by hand.
   */
  private read(): Chosen {
    const empty: Chosen = { userData: null, outputs: null }
    let parsed: Record<string, unknown>
    try {
      parsed = JSON.parse(fs.readFileSync(this.file, 'utf8')) as Record<string, unknown>
    } catch {
      return empty
    }
    const keep = (key: DataLocationKey): string | null => {
      const value = parsed[key]
      if (typeof value !== 'string' || !path.isAbsolute(value)) return null
      return comparablePath(value) === comparablePath(this.fallbacks[key]) ? null : value
    }
    return { userData: keep('userData'), outputs: keep('outputs') }
  }

  /** The folder in use for `key`: the chosen one, or the default. */
  get(key: DataLocationKey): string {
    return this.chosen[key] ?? this.fallbacks[key]
  }

  /** The folder chosen for `key`, or null when it is left at the default. */
  choice(key: DataLocationKey): string | null {
    return this.chosen[key]
  }

  private describe(key: DataLocationKey): DataLocation {
    return { path: this.get(key), fallback: this.fallbacks[key], isDefault: this.chosen[key] === null }
  }

  view(): DataLocationsView {
    return {
      userData: this.describe('userData'),
      outputs: this.describe('outputs'),
      restartNeeded: comparablePath(this.get('userData')) !== comparablePath(this.opened)
    }
  }

  /**
   * Point `key` at `folder`, or back at the default with null. The folder is
   * made and written to first: a path that cannot be used is refused here,
   * while the studio is still running on the old one.
   */
  set(key: DataLocationKey, folder: string | null): DataLocationsView {
    if (folder !== null) {
      const other = this.get(key === 'userData' ? 'outputs' : 'userData')
      const problem = locationProblem(key, folder, { other, programFolders: this.programFolders })
      if (problem) throw new Error(problem)
      try {
        fs.mkdirSync(folder, { recursive: true })
        const probe = path.join(folder, '.studio-write-test')
        fs.writeFileSync(probe, '')
        fs.rmSync(probe, { force: true })
      } catch (err) {
        throw new Error(`That folder cannot be written to: ${(err as Error).message}`)
      }
    }
    const next: Chosen = {
      ...this.chosen,
      [key]: folder !== null && comparablePath(folder) === comparablePath(this.fallbacks[key]) ? null : folder
    }
    this.write(next)
    this.chosen = next
    return this.view()
  }

  private write(chosen: Chosen): void {
    // The defaults are written as nothing at all, so a studio installed elsewhere
    // (a different user account, say) reads the file as "wherever you are".
    const body: Record<string, string> = {}
    for (const key of ['userData', 'outputs'] as const) {
      const value = chosen[key]
      if (value !== null) body[key] = value
    }
    fs.mkdirSync(this.appData, { recursive: true })
    const temp = `${this.file}.${process.pid}.tmp`
    fs.writeFileSync(temp, JSON.stringify(body, null, 2), 'utf8')
    fs.renameSync(temp, this.file)
  }
}
