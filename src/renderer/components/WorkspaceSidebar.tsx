import { useState } from 'react'
import type { InternetLevel } from '@shared/network.js'
import type { WorkspaceState } from '@shared/workspace.js'

/**
 * The open folder and its chats.
 *
 * One folder is open at a time, the way a project is in an editor; the others
 * are a click away under "Switch folder". Everything here is disabled while a
 * reply is running, because a reply belongs to the folder and chat it started in.
 */

export function internetLabel(level: InternetLevel): string {
  return level === 'offline' ? 'Offline' : level === 'search' ? 'Internet: search' : 'Internet: search + download'
}

function when(iso: string): string {
  const then = new Date(iso)
  const days = Math.floor((Date.now() - then.getTime()) / 86_400_000)
  if (days < 1) return then.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
  if (days < 7) return then.toLocaleDateString(undefined, { weekday: 'short' })
  return then.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

export function WorkspaceSidebar({
  state,
  busy,
  onOpenFolder,
  onSwitch,
  onForget,
  onNewChat,
  onOpenChat,
  onDeleteChat
}: {
  state: WorkspaceState
  busy: boolean
  onOpenFolder: () => void
  onSwitch: (id: string) => void
  onForget: (id: string) => void
  onNewChat: () => void
  onOpenChat: (id: string) => void
  onDeleteChat: (id: string) => void
}): React.JSX.Element {
  const [switching, setSwitching] = useState(false)
  const { active, workspaces, chats, chatId } = state
  const others = workspaces.filter((w) => w.id !== active?.id)

  return (
    <aside className="ws-pane" aria-label="Workspace">
      <div className="ws-head">
        <div className="label-text">Folder</div>
        {active ? (
          <div className="ws-current" title={active.path}>
            <strong className="truncate">{active.name}</strong>
            <code className="truncate">{active.path}</code>
          </div>
        ) : (
          <p className="ws-none">No folder open.</p>
        )}
        <div className="ws-head-actions">
          <button onClick={onOpenFolder} disabled={busy}>
            Open folder…
          </button>
          {others.length > 0 ? (
            <button
              className={switching ? 'active' : ''}
              aria-expanded={switching}
              onClick={() => setSwitching((s) => !s)}
              disabled={busy}
            >
              Switch
            </button>
          ) : null}
        </div>
        {switching && others.length > 0 ? (
          <ul className="ws-list">
            {others.map((w) => (
              <li key={w.id}>
                <button
                  className="ws-item"
                  disabled={busy || !w.exists}
                  title={w.exists ? w.path : `${w.path} is no longer there`}
                  onClick={() => {
                    setSwitching(false)
                    onSwitch(w.id)
                  }}
                >
                  <span className="truncate">{w.name}</span>
                  <small className="truncate">{w.exists ? w.path : 'Folder missing'}</small>
                </button>
                <button
                  className="ws-remove"
                  aria-label={`Remove ${w.name} from the list`}
                  title="Remove from the list"
                  disabled={busy}
                  onClick={() => onForget(w.id)}
                >
                  ✕
                </button>
              </li>
            ))}
          </ul>
        ) : null}
      </div>

      {active ? (
        <>
          <div className="ws-chats-head">
            <div className="label-text">Chats</div>
            <button className="primary small" onClick={onNewChat} disabled={busy}>
              New chat
            </button>
          </div>
          <ul className="chat-list">
            {chatId && !chats.some((c) => c.id === chatId) ? (
              <li>
                <div className="chat-item active draft">
                  <span className="truncate">New chat</span>
                </div>
              </li>
            ) : null}
            {chats.map((c) => (
              <li key={c.id}>
                <button
                  className={`chat-item ${c.id === chatId ? 'active' : ''}`}
                  aria-current={c.id === chatId ? 'true' : undefined}
                  disabled={busy && c.id !== chatId}
                  onClick={() => c.id !== chatId && onOpenChat(c.id)}
                >
                  <span className="truncate">{c.title}</span>
                  <small>{when(c.updatedAt)}</small>
                </button>
                <button
                  className="ws-remove"
                  aria-label={`Delete chat ${c.title}`}
                  title="Delete chat"
                  disabled={busy}
                  onClick={() => onDeleteChat(c.id)}
                >
                  ✕
                </button>
              </li>
            ))}
          </ul>
        </>
      ) : null}

      {active ? (
        <div className="ws-foot">
          {active.rules.filter((r) => r.enabled).length} project rule
          {active.rules.filter((r) => r.enabled).length === 1 ? '' : 's'} ·{' '}
          {1 + active.extraRoots.filter((r) => r.enabled).length} shared folder
          {active.extraRoots.filter((r) => r.enabled).length === 0 ? '' : 's'}
          <div className={`ws-net ws-net-${active.internet}`}>{internetLabel(active.internet)}</div>
        </div>
      ) : null}
    </aside>
  )
}
