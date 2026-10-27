import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import { createHash, randomUUID } from 'node:crypto'
import type { ChangeGroup, ChangeKind, RevertResult } from '@shared/changes.js'

/**
 * Every change the agent makes, kept so it can be walked backwards.
 *
 * Undo is what makes the automatic modes defensible. Someone will only turn off
 * confirmation for writes if a wrong one is a nuisance rather than a loss, and
 * that is a property of this journal existing, not of the model being careful.
 *
 * Before a file is changed its old contents are copied into the backup folder;
 * after, a hash of the new contents is noted. Changes are grouped by the reply
 * that made them, and a group can be put back as a whole. A revert checks each
 * file still holds what the agent left there: a file edited since, by the user
 * or by a later reply, is reported as a conflict and only overwritten when asked
 * to. A revert is itself recorded (with a backup of what it replaced) as a
 * group of its own, so it can be reverted too.
 *
 * With a file to keep it in, the journal survives restarts; without one (tests)
 * it lives in memory.
 */
export interface JournalEntry {
  id: string
  at: string
  kind: ChangeKind
  path: string
  /** Where the previous contents were parked. Null when the file did not exist. */
  backup: string | null
  group: string
  /** Hash of the file after the change; null when the change left no file. Unset until settled. */
  afterHash?: string | null
  /** The revert group that put this change back. */
  revertedBy?: string
}

interface GroupInfo {
  label: string
  chatId: string | null
  at: string
  revertOf: string | null
}

interface Saved {
  version: 1
  entries: JournalEntry[]
  groups: Record<string, GroupInfo>
}

async function hashOf(target: string): Promise<string | null> {
  try {
    return createHash('sha256').update(await fsp.readFile(target)).digest('hex')
  } catch {
    return null
  }
}

export class Journal {
  private entries: JournalEntry[] = []
  private groupInfo: Record<string, GroupInfo> = {}
  private saving: Promise<void> = Promise.resolve()

  constructor(
    private readonly backupDir: string,
    private readonly file: string | null = null,
    private readonly onChange: () => void = () => undefined
  ) {
    if (file) {
      try {
        const saved = JSON.parse(fs.readFileSync(file, 'utf8')) as Saved
        if (Array.isArray(saved.entries)) this.entries = saved.entries
        if (saved.groups && typeof saved.groups === 'object') this.groupInfo = saved.groups
      } catch {
        // No journal yet, or an unreadable one: start empty. Backups already on
        // disk are left where they are.
      }
    }
  }

  private persist(): Promise<void> {
    this.onChange()
    const file = this.file
    if (!file) return Promise.resolve()
    const snapshot: Saved = { version: 1, entries: this.entries, groups: this.groupInfo }
    const text = JSON.stringify(snapshot, null, 1)
    this.saving = this.saving
      .catch(() => undefined)
      .then(async () => {
        await fsp.mkdir(path.dirname(file), { recursive: true })
        const temp = `${file}.${process.pid}.tmp`
        await fsp.writeFile(temp, text, 'utf8')
        await fsp.rename(temp, file)
      })
    return this.saving
  }

  /** Name a group, typically after the message whose reply is about to make changes. */
  label(group: string, label: string, chatId: string | null): void {
    this.groupInfo[group] = { label, chatId, at: new Date().toISOString(), revertOf: null }
  }

  async record(kind: ChangeKind, target: string, group = 'manual'): Promise<JournalEntry> {
    await fsp.mkdir(this.backupDir, { recursive: true })
    let backup: string | null = null
    if (fs.existsSync(target)) {
      backup = path.join(this.backupDir, `${Date.now()}-${randomUUID()}${path.extname(target)}`)
      await fsp.copyFile(target, backup)
    }
    const entry: JournalEntry = { id: randomUUID(), at: new Date().toISOString(), kind, path: target, backup, group }
    this.entries.push(entry)
    this.groupInfo[group] ??= { label: 'Changes', chatId: null, at: entry.at, revertOf: null }
    await this.persist()
    return entry
  }

  /** Note what the change left behind, once it has happened. */
  async settle(entry: JournalEntry): Promise<void> {
    entry.afterHash = await hashOf(entry.path)
    await this.persist()
  }

  list(): readonly JournalEntry[] {
    return this.entries
  }

  /** Groups with at least one change, newest first. */
  groups(limit = 100): ChangeGroup[] {
    const byGroup = new Map<string, JournalEntry[]>()
    for (const e of this.entries) {
      const list = byGroup.get(e.group) ?? []
      list.push(e)
      byGroup.set(e.group, list)
    }
    const out: ChangeGroup[] = []
    for (const [id, list] of byGroup) {
      const info = this.groupInfo[id]
      const seen = new Map<string, { path: string; kind: ChangeKind; existedBefore: boolean }>()
      for (const e of list) {
        const prior = seen.get(e.path)
        if (prior) prior.kind = e.kind
        else seen.set(e.path, { path: e.path, kind: e.kind, existedBefore: e.backup !== null })
      }
      out.push({
        id,
        label: info?.label ?? 'Changes',
        chatId: info?.chatId ?? null,
        at: info?.at ?? (list[0] as JournalEntry).at,
        files: [...seen.values()],
        reverted: list.every((e) => e.revertedBy !== undefined),
        revertOf: info?.revertOf ?? null
      })
    }
    return out.sort((a, b) => b.at.localeCompare(a.at)).slice(0, limit)
  }

  /**
   * Put back every change in `group`, newest first.
   *
   * Without `force`, nothing is touched if any file no longer holds what the
   * group left in it; the conflicts are returned for the user to decide on.
   */
  async revert(group: string, force = false): Promise<RevertResult> {
    const pending = this.entries.filter((e) => e.group === group && e.revertedBy === undefined)
    if (pending.length === 0) return { done: true, restored: [], conflicts: [] }

    // Only the last change to each file in the group says what should be there now.
    const latest = new Map<string, JournalEntry>()
    for (const e of pending) latest.set(e.path, e)
    const conflicts: string[] = []
    for (const e of latest.values()) {
      if (e.afterHash === undefined) continue
      if ((await hashOf(e.path)) !== e.afterHash) conflicts.push(e.path)
    }
    if (conflicts.length > 0 && !force) return { done: false, restored: [], conflicts }

    const revertId = `revert-${randomUUID()}`
    const original = this.groupInfo[group]
    this.groupInfo[revertId] = {
      label: `Reverted: ${original?.label ?? 'changes'}`,
      chatId: original?.chatId ?? null,
      at: new Date().toISOString(),
      revertOf: group
    }
    const restored: string[] = []
    for (const e of [...pending].reverse()) {
      const undo = await this.record('restore', e.path, revertId)
      if (e.backup === null) {
        await fsp.rm(e.path, { force: true })
      } else {
        await fsp.mkdir(path.dirname(e.path), { recursive: true })
        await fsp.copyFile(e.backup, e.path)
      }
      await this.settle(undo)
      e.revertedBy = revertId
      if (!restored.includes(e.path)) restored.push(e.path)
    }
    // Reverting a revert redoes the original, which then counts as not reverted.
    const redone = original?.revertOf
    if (redone) {
      for (const e of this.entries) if (e.group === redone && e.revertedBy === group) delete e.revertedBy
    }
    await this.persist()
    return { done: true, restored, conflicts }
  }

  /** Put back the most recent group that has not been put back yet. */
  async undoLast(force = false): Promise<RevertResult & { group: string | null }> {
    const last = [...this.entries].reverse().find((e) => e.revertedBy === undefined)
    if (!last) return { done: true, restored: [], conflicts: [], group: null }
    return { ...(await this.revert(last.group, force)), group: last.group }
  }
}
