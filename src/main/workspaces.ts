import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import { createHash } from 'node:crypto'
import type { Chat, ChatMeta, Workspace, WorkspaceSummary } from '@shared/workspace.js'
import { workspaceSlug } from '@shared/workspace.js'
import { withDefaultProtections } from '@shared/permissions.js'
import { isInternetLevel } from '@shared/network.js'
import { comparablePath } from './broker/paths.js'
import type { FileSealer } from './sealer.js'

/**
 * Workspaces and their chats on disk, in the application's data folder:
 *
 *   <dir>/<workspace id>/workspace.json     the folder, its rules and grants
 *   <dir>/<workspace id>/chats.json         chat list (titles, dates), newest first
 *   <dir>/<workspace id>/chats/<chat>.json  one conversation each
 *   <dir>/<workspace id>/backups/           copies taken before the agent changes a file
 *   <dir>/<workspace id>/journal.json       what changed, by which reply, for reverting
 *   <dir>/<workspace id>/network-log.jsonl  every internet request the agent asked for
 *
 * The id is derived from the folder's canonical path, so opening the same folder
 * again finds the same workspace, and the folder name is kept in front of it so
 * the directory listing is readable. Ids that reach here from the window are
 * checked against ID_SHAPE before they are used to build a path.
 *
 * Writes are atomic (temp file, then rename) and queued per file, so two saves
 * landing at once cannot interleave and a crash leaves the previous version
 * rather than half of a new one. Workspaces and chat lists are cached once read,
 * and the cache is updated before the write is queued, so a read straight after
 * a save never sees the file as it was before. Anything unreadable is treated
 * as absent.
 *
 * The conversations (chats/<chat>.json) and the chat list, whose titles are the
 * first words of each chat, are encrypted when a sealer is given (sealer.ts).
 * The rest is settings and records of what changed, kept readable.
 */

const ID_SHAPE = /^[a-z0-9][a-z0-9-]{0,80}$/i

export function isSafeId(id: unknown): id is string {
  return typeof id === 'string' && ID_SHAPE.test(id)
}

export function workspaceIdFor(canonicalPath: string): string {
  const hash = createHash('sha256').update(comparablePath(canonicalPath)).digest('hex').slice(0, 12)
  return `${workspaceSlug(path.basename(canonicalPath))}-${hash}`
}

function readJson<T>(file: string): T | null {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8')) as T
  } catch {
    return null
  }
}

export class WorkspaceStore {
  private readonly queues = new Map<string, Promise<void>>()
  private readonly workspaces = new Map<string, Workspace>()
  private readonly chatLists = new Map<string, ChatMeta[]>()

  constructor(
    private readonly dir: string,
    private readonly sealer: FileSealer | null = null
  ) {}

  /** What a sealed file is bound to, so it opens only in its own place. */
  private static place(id: string, name: string): string {
    return `${id}/${name}`
  }

  /**
   * A conversation file's contents. A plain one, from before encryption, is
   * sealed again in the background; one that does not open counts as absent.
   */
  private readPrivate<T>(file: string, place: string): T | null {
    let text: string
    try {
      text = fs.readFileSync(file, 'utf8')
    } catch {
      return null
    }
    try {
      if (!this.sealer) return JSON.parse(text) as T
      const { value, sealed } = this.sealer.open(text, place)
      if (!sealed && this.sealer.sealing) void this.write(file, value, place).catch(() => undefined)
      return value as T
    } catch (err) {
      console.warn(`A saved chat file could not be read: ${(err as Error).message}`)
      return null
    }
  }

  private folder(id: string): string {
    if (!isSafeId(id)) throw new Error('Not a workspace id.')
    return path.join(this.dir, id)
  }

  private chatFile(id: string, chatId: string): string {
    if (!isSafeId(chatId)) throw new Error('Not a chat id.')
    return path.join(this.folder(id), 'chats', `${chatId}.json`)
  }

  backupsDir(id: string): string {
    return path.join(this.folder(id), 'backups')
  }

  journalFile(id: string): string {
    return path.join(this.folder(id), 'journal.json')
  }

  networkLogFile(id: string): string {
    return path.join(this.folder(id), 'network-log.jsonl')
  }

  /** Queue a write of `value`; sealed first when `place` is given and there is a sealer. */
  private write(file: string, value: unknown, place?: string): Promise<void> {
    const previous = this.queues.get(file) ?? Promise.resolve()
    const next = previous
      .catch(() => undefined)
      .then(async () => {
        const json = JSON.stringify(value, null, 2)
        // Sealed before anything reaches the disk, temp file included.
        const text = place !== undefined && this.sealer ? this.sealer.seal(json, place) : json
        await fsp.mkdir(path.dirname(file), { recursive: true })
        const temp = `${file}.${process.pid}.tmp`
        await fsp.writeFile(temp, text, 'utf8')
        await fsp.rename(temp, file)
      })
    this.queues.set(file, next)
    return next
  }

  get(id: string): Workspace | null {
    if (!isSafeId(id)) return null
    const cached = this.workspaces.get(id)
    if (cached) return cached
    const ws = readJson<Workspace>(path.join(this.folder(id), 'workspace.json'))
    if (!ws || ws.id !== id || typeof ws.path !== 'string') return null
    const clean: Workspace = {
      ...ws,
      rules: Array.isArray(ws.rules) ? ws.rules : [],
      extraRoots: Array.isArray(ws.extraRoots) ? ws.extraRoots : [],
      policyRules: Array.isArray(ws.policyRules) ? ws.policyRules : [],
      protectedPaths: withDefaultProtections(ws.protectedPaths),
      // Anything unreadable falls back to the restrictive default.
      internet: isInternetLevel(ws.internet) ? ws.internet : 'offline',
      strict: ws.strict === true
    }
    this.workspaces.set(id, clean)
    return clean
  }

  private saveWorkspace(ws: Workspace): Promise<void> {
    this.workspaces.set(ws.id, ws)
    return this.write(path.join(this.folder(ws.id), 'workspace.json'), ws)
  }

  private saveChatList(id: string, list: ChatMeta[]): Promise<void> {
    this.chatLists.set(id, list)
    return this.write(path.join(this.folder(id), 'chats.json'), list, WorkspaceStore.place(id, 'chats.json'))
  }

  list(): WorkspaceSummary[] {
    let names: string[]
    try {
      names = fs.readdirSync(this.dir)
    } catch {
      return []
    }
    return names
      .map((name) => this.get(name))
      .filter((ws): ws is Workspace => ws !== null)
      .map((ws) => ({
        id: ws.id,
        path: ws.path,
        name: ws.name,
        lastOpenedAt: ws.lastOpenedAt,
        exists: fs.existsSync(ws.path)
      }))
      .sort((a, b) => b.lastOpenedAt.localeCompare(a.lastOpenedAt))
  }

  /** Find the workspace for a folder, creating it the first time, and mark it as just opened. */
  async open(canonicalPath: string): Promise<Workspace> {
    const id = workspaceIdFor(canonicalPath)
    const now = new Date().toISOString()
    const existing = this.get(id)
    const ws: Workspace = existing
      ? { ...existing, path: canonicalPath, lastOpenedAt: now }
      : {
          id,
          path: canonicalPath,
          name: path.basename(canonicalPath) || canonicalPath,
          createdAt: now,
          lastOpenedAt: now,
          rules: [],
          extraRoots: [],
          policyRules: [],
          protectedPaths: withDefaultProtections([]),
          internet: 'offline',
          strict: false
        }
    await this.saveWorkspace(ws)
    return ws
  }

  async update(id: string, mutate: (ws: Workspace) => void): Promise<Workspace> {
    const current = this.get(id)
    if (!current) throw new Error('That workspace no longer exists.')
    const next = structuredClone(current)
    mutate(next)
    next.id = current.id
    await this.saveWorkspace(next)
    return next
  }

  /** Remove a workspace and everything kept for it. The folder itself is not touched. */
  async forget(id: string): Promise<void> {
    const folder = this.folder(id)
    await Promise.all([...this.queues.entries()].filter(([f]) => f.startsWith(folder)).map(([, p]) => p.catch(() => undefined)))
    this.workspaces.delete(id)
    this.chatLists.delete(id)
    await fsp.rm(folder, { recursive: true, force: true })
  }

  listChats(id: string): ChatMeta[] {
    const cached = this.chatLists.get(id)
    if (cached) return cached
    const index = this.readPrivate<ChatMeta[]>(path.join(this.folder(id), 'chats.json'), WorkspaceStore.place(id, 'chats.json'))
    const clean = Array.isArray(index) ? index.filter((m) => isSafeId(m?.id)) : []
    this.chatLists.set(id, clean)
    return clean
  }

  loadChat(id: string, chatId: string): Chat | null {
    const chat = this.readPrivate<Chat>(this.chatFile(id, chatId), WorkspaceStore.place(id, `chats/${chatId}.json`))
    if (!chat || chat.meta?.id !== chatId) return null
    return {
      meta: chat.meta,
      turns: Array.isArray(chat.turns) ? chat.turns : [],
      outcomes: Array.isArray(chat.outcomes) ? chat.outcomes : [],
      history: Array.isArray(chat.history) ? chat.history : null
    }
  }

  async saveChat(id: string, chat: Chat): Promise<ChatMeta[]> {
    const index = [chat.meta, ...this.listChats(id).filter((m) => m.id !== chat.meta.id)].sort((a, b) =>
      b.updatedAt.localeCompare(a.updatedAt)
    )
    await Promise.all([
      this.write(this.chatFile(id, chat.meta.id), chat, WorkspaceStore.place(id, `chats/${chat.meta.id}.json`)),
      this.saveChatList(id, index)
    ])
    return index
  }

  /**
   * Seal every conversation file still in plain text, from before encryption.
   * Returns how many were sealed. Files that do not parse are left alone.
   */
  async sealPlainFiles(): Promise<number> {
    if (!this.sealer?.sealing) return 0
    let ids: string[]
    try {
      ids = fs.readdirSync(this.dir).filter(isSafeId)
    } catch {
      return 0
    }
    const pending: Promise<void>[] = []
    const reseal = (file: string, place: string): void => {
      let text: string
      try {
        text = fs.readFileSync(file, 'utf8')
      } catch {
        return
      }
      try {
        const { value, sealed } = (this.sealer as FileSealer).open(text, place)
        if (!sealed) pending.push(this.write(file, value, place))
      } catch {
        // Unreadable or sealed elsewhere: not this function's to fix.
      }
    }
    for (const id of ids) {
      reseal(path.join(this.dir, id, 'chats.json'), WorkspaceStore.place(id, 'chats.json'))
      let files: string[]
      try {
        files = fs.readdirSync(path.join(this.dir, id, 'chats'))
      } catch {
        continue
      }
      for (const name of files) {
        const chatId = name.endsWith('.json') ? name.slice(0, -'.json'.length) : ''
        if (isSafeId(chatId)) reseal(path.join(this.dir, id, 'chats', name), WorkspaceStore.place(id, `chats/${chatId}.json`))
      }
    }
    await Promise.all(pending)
    return pending.length
  }

  async deleteChat(id: string, chatId: string): Promise<ChatMeta[]> {
    const file = this.chatFile(id, chatId)
    await (this.queues.get(file) ?? Promise.resolve()).catch(() => undefined)
    await fsp.rm(file, { force: true })
    const index = this.listChats(id).filter((m) => m.id !== chatId)
    await this.saveChatList(id, index)
    return index
  }
}
