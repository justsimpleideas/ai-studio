import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import { createHash } from 'node:crypto'
import type { Chat, ChatMeta, Workspace, WorkspaceSummary } from '@shared/workspace.js'
import { workspaceSlug } from '@shared/workspace.js'
import { withDefaultProtections } from '@shared/permissions.js'
import { isInternetLevel } from '@shared/network.js'
import { comparablePath } from './broker/paths.js'

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

  constructor(private readonly dir: string) {}

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

  private write(file: string, value: unknown): Promise<void> {
    const previous = this.queues.get(file) ?? Promise.resolve()
    const next = previous
      .catch(() => undefined)
      .then(async () => {
        await fsp.mkdir(path.dirname(file), { recursive: true })
        const temp = `${file}.${process.pid}.tmp`
        await fsp.writeFile(temp, JSON.stringify(value, null, 2), 'utf8')
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
      internet: isInternetLevel(ws.internet) ? ws.internet : 'offline'
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
    return this.write(path.join(this.folder(id), 'chats.json'), list)
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
          internet: 'offline'
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
    const index = readJson<ChatMeta[]>(path.join(this.folder(id), 'chats.json'))
    const clean = Array.isArray(index) ? index.filter((m) => isSafeId(m?.id)) : []
    this.chatLists.set(id, clean)
    return clean
  }

  loadChat(id: string, chatId: string): Chat | null {
    const chat = readJson<Chat>(this.chatFile(id, chatId))
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
    await Promise.all([this.write(this.chatFile(id, chat.meta.id), chat), this.saveChatList(id, index)])
    return index
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
