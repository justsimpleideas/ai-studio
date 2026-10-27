import type { GlobalRule } from './config.js'
import type { PolicyRule, ProtectedPath, WorkspaceRoot } from './permissions.js'
import type { ToolOutcome } from './broker-types.js'
import type { InternetLevel } from './network.js'

/**
 * A workspace is a folder the user opened, and everything the agent keeps for it.
 *
 * Opening a folder makes it the agent's working folder: relative paths resolve
 * against it, it is shared without being asked for, and the agent is told where
 * it is. Each workspace keeps its own chats, its own project rules, its own
 * extra folders and its own "always allow" grants, so what was agreed for one
 * project never leaks into another. Rules in config.rules apply everywhere.
 *
 * All of it lives in the application's data folder, never in the opened folder
 * itself: opening a project must not write anything into it.
 */
export interface Workspace {
  /** Stable for a given folder: derived from its canonical path. */
  id: string
  /** Canonical, real absolute path of the working folder. */
  path: string
  name: string
  createdAt: string
  lastOpenedAt: string
  /** Instructions that apply only in this workspace, after the global rules. */
  rules: GlobalRule[]
  /** More folders the agent may reach while working here, besides the working folder. */
  extraRoots: WorkspaceRoot[]
  /** "Always allow" decisions made while working here. */
  policyRules: PolicyRule[]
  /** Paths inside the shared folders that are hidden or read-only. Starts with the defaults. */
  protectedPaths: ProtectedPath[]
  /** How much internet the agent may use here. 'offline' unless the user changes it. */
  internet: InternetLevel
}

export interface WorkspaceSummary {
  id: string
  path: string
  name: string
  lastOpenedAt: string
  /** False when the folder has been moved or deleted since; its chats are still readable. */
  exists: boolean
}

export interface ChatMeta {
  id: string
  title: string
  createdAt: string
  updatedAt: string
  messageCount: number
}

/** One line of the transcript as the user saw it. */
export interface ChatTurn {
  id: string
  role: 'user' | 'assistant' | 'system'
  text: string
}

/**
 * A saved conversation.
 *
 * `turns` and `outcomes` are what the window shows. `history` is the model's own
 * record (node-llama-cpp chat history items, kept opaque here), which is what
 * lets a chat be continued after a restart rather than only read.
 */
export interface Chat {
  meta: ChatMeta
  turns: ChatTurn[]
  outcomes: ToolOutcome[]
  history: unknown[] | null
}

/** What the window gets about a chat: everything except the model's own record. */
export type ChatView = Omit<Chat, 'history'>

export interface WorkspaceState {
  workspaces: WorkspaceSummary[]
  active: Workspace | null
  /** Saved chats of the active workspace, newest first. */
  chats: ChatMeta[]
  /** The open chat. It is not in `chats` until its first message is sent. */
  chatId: string | null
}

/**
 * Returned by anything that may change which chat is open. The transcript comes
 * with it only then; ordinary state updates leave the window's live transcript
 * alone.
 */
export interface WorkspaceSnapshot {
  state: WorkspaceState
  chat: ChatView | null
}

const TITLE_LIMIT = 60

/** A chat is named after its first message, like most chat apps do. */
export function chatTitle(firstMessage: string): string {
  const line = firstMessage.trim().split(/\r?\n/)[0]?.trim() ?? ''
  if (!line) return 'New chat'
  return line.length > TITLE_LIMIT ? `${line.slice(0, TITLE_LIMIT - 1).trimEnd()}…` : line
}

/**
 * Folder-name part of a workspace id: lower-case ASCII letters, digits and
 * dashes, so the id is safe as a directory name on every platform.
 */
export function workspaceSlug(folderName: string): string {
  const slug = folderName
    .normalize('NFKD')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 40)
    .replace(/-+$/g, '')
  return slug || 'folder'
}
