import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { AppConfig, GlobalRule, KvCacheType, WorkingContext } from '@shared/config.js'
import type { ModelInfo } from '@shared/engine-protocol.js'
import type { ApprovalRequest, ApprovalResponse, ToolOutcome } from '@shared/broker-types.js'
import type { ClassMode } from '@shared/permissions.js'
import type { RiskClass } from '@shared/tools.js'
import type { UsageSnapshot } from '@shared/usage.js'
import type { WorkspaceSnapshot, WorkspaceState } from '@shared/workspace.js'
import type { ChangeGroup } from '@shared/changes.js'
import { matchesShortcut } from '@shared/shortcuts.js'
import { useShortcuts } from './useShortcuts.js'
import { ActivityLog } from './components/ActivityLog.js'
import { ApprovalDialog } from './components/ApprovalDialog.js'
import { PermissionsPanel } from './components/PermissionsPanel.js'
import { PromptPreview, RulesPanel } from './components/RulesPanel.js'
import { ContextMeter, UsagePanel } from './components/UsagePanel.js'
import { internetLabel, WorkspaceSidebar } from './components/WorkspaceSidebar.js'
import { cleanError } from './image/api.js'
import { formatBytes } from './image/format.js'

type Tab = 'chat' | 'usage' | 'permissions' | 'rules'

const TAB_LABEL: Record<Tab, string> = {
  chat: 'Chat',
  usage: 'Token usage',
  permissions: 'Permissions',
  rules: 'Rules'
}

/** Shorter labels the top bar switches to when the window is narrow. */
const TAB_SHORT: Partial<Record<Tab, string>> = {
  usage: 'Usage'
}

const SIDEBAR_KEY = 'studio.chatSidebar'

interface Turn {
  id: string
  role: 'user' | 'assistant' | 'system'
  text: string
  streaming: boolean
}

/**
 * The chat side. A folder is opened as a workspace; the agent works in it, and
 * its chats, project rules and permissions are kept for it. The main process
 * owns all of that and saves it; this view shows the open chat live and asks
 * for the rest.
 */
export function ChatStudio({ visible }: { visible: boolean }): React.JSX.Element {
  const [config, setConfig] = useState<AppConfig | null>(null)
  const [ws, setWs] = useState<WorkspaceState | null>(null)
  const [tab, setTab] = useState<Tab>('chat')
  const [showSidebar, setShowSidebar] = useState(() => localStorage.getItem(SIDEBAR_KEY) !== 'hidden')
  const [model, setModel] = useState<ModelInfo | null>(null)
  const [loadProgress, setLoadProgress] = useState<number | null>(null)
  const [turns, setTurns] = useState<Turn[]>([])
  const [activePrompt, setActivePrompt] = useState<string | null>(null)
  const [approval, setApproval] = useState<ApprovalRequest | null>(null)
  const [queue, setQueue] = useState<ApprovalRequest[]>([])
  const [outcomes, setOutcomes] = useState<ToolOutcome[]>([])
  const [sessionGrants, setSessionGrants] = useState(0)
  const [input, setInput] = useState('')
  const [usage, setUsage] = useState<UsageSnapshot | null>(null)
  const [changes, setChanges] = useState<ChangeGroup[]>([])

  const transcriptRef = useRef<HTMLDivElement>(null)
  const modelMenuRef = useRef<HTMLDivElement>(null)

  useEffect(() => localStorage.setItem(SIDEBAR_KEY, showSidebar ? 'shown' : 'hidden'), [showSidebar])
  const toggleSidebar = useCallback(() => setShowSidebar((shown) => !shown), [])

  // Ctrl+B by default (Settings can change it), as in most editors, while the chat tab is in front.
  const shortcuts = useShortcuts()
  useEffect(() => {
    if (!visible || tab !== 'chat') return
    const onKey = (e: KeyboardEvent): void => {
      if (matchesShortcut(e, shortcuts.toggleSidebar)) {
        e.preventDefault()
        toggleSidebar()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [visible, tab, toggleSidebar, shortcuts.toggleSidebar])

  const say = useCallback((text: string) => {
    setTurns((prev) => [...prev, { id: `note-${Date.now()}`, role: 'system', text, streaming: false }])
  }, [])

  const refreshChanges = useCallback(() => {
    void window.api.listChanges().then(setChanges)
  }, [])

  /**
   * Put back one reply's changes. Asked twice when a file has changed since the
   * agent touched it; the current version is backed up either way.
   */
  const revert = useCallback(
    async (group: ChangeGroup) => {
      const names = group.files.map((f) => f.path.split(/[\\/]/).pop()).join('\n')
      const what = group.revertOf ? 'Undo this revert, putting the agent’s changes back?' : 'Put back what this reply changed?'
      if (!window.confirm(`${what}\n\n${names}`)) return
      try {
        let result = await window.api.revertChanges(group.id, false)
        if (!result.done) {
          const changed = result.conflicts.map((p) => p.split(/[\\/]/).pop()).join('\n')
          const ok = window.confirm(
            `These files were changed after the agent touched them, by you or by a later reply:\n\n${changed}\n\nPut them back anyway? Their current versions are backed up first, so this can be undone.`
          )
          if (!ok) return
          result = await window.api.revertChanges(group.id, true)
        }
        say(`Put back ${result.restored.length} file${result.restored.length === 1 ? '' : 's'}.`)
      } catch (err) {
        say(cleanError(err).message)
      } finally {
        refreshChanges()
      }
    },
    [say, refreshChanges]
  )

  /** Take on a snapshot from the main process: workspace state and, when it changed, the open chat. */
  const apply = useCallback((snap: WorkspaceSnapshot) => {
    setWs((prev) => {
      if (prev?.active?.id !== snap.state.active?.id) setSessionGrants(0)
      return snap.state
    })
    setTurns((snap.chat?.turns ?? []).map((t) => ({ ...t, streaming: false })))
    setOutcomes(snap.chat?.outcomes ?? [])
    setUsage(null)
  }, [])

  /** Run a workspace action, showing a refusal (e.g. "wait for the reply") in the transcript. */
  const act = useCallback(
    async (action: () => Promise<WorkspaceSnapshot>) => {
      try {
        apply(await action())
      } catch (err) {
        say(cleanError(err).message)
      }
    },
    [apply, say]
  )

  const update = useCallback(
    async (action: () => Promise<WorkspaceState>) => {
      try {
        setWs(await action())
      } catch (err) {
        say(cleanError(err).message)
      }
    },
    [say]
  )

  useEffect(() => {
    void window.api.getConfig().then(setConfig)
    void window.api.workspace.get().then(apply)
  }, [apply])

  useEffect(() => window.api.onConfigChanged(setConfig), [])
  useEffect(() => window.api.workspace.onChanged(setWs), [])

  // The change history follows the open folder, and every change the agent makes.
  const activeId = ws?.active?.id ?? null
  useEffect(() => refreshChanges(), [activeId, refreshChanges])
  useEffect(() => {
    let timer: number | undefined
    const off = window.api.onChangesUpdated(() => {
      window.clearTimeout(timer)
      timer = window.setTimeout(refreshChanges, 150)
    })
    return () => {
      off()
      window.clearTimeout(timer)
    }
  }, [refreshChanges])

  useEffect(() => window.api.onToolOutcome((o) => setOutcomes((prev) => [...prev, o])), [])

  useEffect(
    () =>
      window.api.onApproval((request) => {
        // Only one dialog is shown at a time. A second request arriving while
        // the first is open is queued rather than stacked, so an approval is
        // never given to a prompt the user could not see.
        setApproval((current) => {
          if (current) {
            setQueue((q) => [...q, request])
            return current
          }
          return request
        })
      }),
    []
  )

  useEffect(
    () =>
      window.api.onEngineEvent((event) => {
        switch (event.type) {
          case 'loadProgress':
            setLoadProgress(event.fraction)
            break
          case 'loaded':
            setModel(event.info)
            setLoadProgress(null)
            break
          case 'unloaded':
            setModel(null)
            setUsage(null)
            break
          case 'usage':
            setUsage(event.snapshot)
            break
          case 'chunk':
            setTurns((prev) =>
              prev.map((t) => (t.id === event.id ? { ...t, text: t.text + event.text } : t))
            )
            break
          case 'complete':
            setTurns((prev) =>
              prev.map((t) => (t.id === event.id ? { ...t, streaming: false } : t))
            )
            setActivePrompt(null)
            break
          case 'aborted':
            setTurns((prev) =>
              prev.map((t) =>
                t.id === event.id
                  ? { ...t, streaming: false, text: `${t.text}\n\n[stopped]` }
                  : t
              )
            )
            setActivePrompt(null)
            break
          case 'error':
            if (event.fatal) {
              setModel(null)
              setLoadProgress(null)
              setUsage(null)
            }
            setActivePrompt(null)
            setTurns((prev) => [
              // A reply that never produced any text is dropped rather than left as an empty bubble.
              ...prev
                .filter((t) => !(t.streaming && t.text === ''))
                .map((t) => ({ ...t, streaming: false })),
              {
                id: `err-${Date.now()}`,
                role: 'system',
                text: event.message,
                streaming: false
              }
            ])
            break
        }
      }),
    []
  )

  useEffect(() => {
    transcriptRef.current?.scrollTo({ top: transcriptRef.current.scrollHeight })
  }, [turns])

  const respond = useCallback((id: string, response: ApprovalResponse) => {
    void window.api.respondToApproval(id, response)
    if (response.grant?.scope === 'session') setSessionGrants((n) => n + 1)
    setQueue((q) => {
      const [next, ...rest] = q
      setApproval(next ?? null)
      return rest
    })
  }, [])

  const active = ws?.active ?? null

  const send = useCallback(async () => {
    const text = input.trim()
    if (!text || !model || !active || activePrompt) return
    setInput('')
    try {
      const id = await window.api.sendMessage(text)
      setTurns((prev) => [
        ...prev,
        { id: `u-${id}`, role: 'user', text, streaming: false },
        { id, role: 'assistant', text: '', streaming: true }
      ])
      setActivePrompt(id)
    } catch (err) {
      setInput(text)
      say(cleanError(err).message)
    }
  }, [input, model, active, activePrompt, say])

  const loadPath = useCallback(
    async (modelPath: string) => {
      setLoadProgress(0)
      try {
        await window.api.loadModel(modelPath)
      } catch (err) {
        setLoadProgress(null)
        say(cleanError(err).message)
      }
    },
    [say]
  )

  const pickAndLoad = useCallback(async () => {
    const picked = await window.api.pickModel()
    if (picked) await loadPath(picked)
  }, [loadPath])

  const setKvCache = useCallback(
    (kv: KvCacheType) => {
      void window.api.setKvCache(kv).then(setConfig, (err) => say(cleanError(err).message))
    },
    [say]
  )

  const newChat = useCallback(() => void act(() => window.api.workspace.newChat()), [act])

  const context = useMemo<WorkingContext | undefined>(
    () =>
      active
        ? {
            workingFolder: active.path,
            extraFolders: active.extraRoots.filter((r) => r.enabled).map((r) => r.path),
            projectRules: active.rules
          }
        : undefined,
    [active]
  )

  if (!config || !ws) return <div className="empty">Loading…</div>

  const busy = activePrompt !== null
  const chatTitle = ws.chats.find((c) => c.id === ws.chatId)?.title ?? 'New chat'
  const fromMenu = (action: () => void) => () => {
    modelMenuRef.current?.hidePopover()
    action()
  }

  return (
    <div className="app">
      <div className="topbar">
        {/* Hidden rather than removed on the other tabs, so the tabs never shift. */}
        <button
          className={`icon-button sidebar-toggle ${tab === 'chat' ? '' : 'invisible'}`}
          aria-label="Folder and chats"
          aria-pressed={showSidebar}
          title={`${showSidebar ? 'Hide' : 'Show'} folder and chats (${shortcuts.toggleSidebar})`}
          tabIndex={tab === 'chat' ? undefined : -1}
          onClick={toggleSidebar}
        >
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden>
            <rect x="3" y="4.5" width="18" height="15" />
            <path d="M9 4.5v15" />
            {showSidebar ? <rect x="3" y="4.5" width="6" height="15" fill="currentColor" opacity="0.45" stroke="none" /> : null}
          </svg>
        </button>

        <div className="tabs">
          {(Object.keys(TAB_LABEL) as Tab[]).map((t) => (
            <button key={t} aria-selected={tab === t} onClick={() => setTab(t)}>
              <span className={TAB_SHORT[t] ? 'tab-long' : undefined}>{TAB_LABEL[t]}</span>
              {TAB_SHORT[t] ? <span className="tab-short">{TAB_SHORT[t]}</span> : null}
            </button>
          ))}
        </div>

        <div className="spacer" />

        {model && loadProgress === null ? <ContextMeter snapshot={usage} onOpen={() => setTab('usage')} /> : null}

        {loadProgress !== null ? (
          <div className="model-chip idle">Loading… {Math.round(loadProgress * 100)}%</div>
        ) : model ? (
          <>
            <button className="model-chip" popoverTarget="model-menu" title={`${model.name}\n${model.path}`}>
              <strong>{model.name}</strong>
              <span className={`badge ${model.gpuLayers > 0 ? 'gpu' : 'cpu'}`}>
                {model.gpuLayers > 0 ? (
                  <>
                    {model.backend}
                    <span className="badge-more"> · {model.gpuLayers} layers</span>
                  </>
                ) : (
                  'cpu'
                )}
              </span>
              <span className="caret" aria-hidden>
                ▾
              </span>
            </button>
            <div className="model-menu" id="model-menu" popover="auto" ref={modelMenuRef}>
              <div className="mm-name">{model.name}</div>
              <code className="mm-path">{model.path}</code>
              <dl className="mm-facts">
                <dt>Runs on</dt>
                <dd>{model.gpuLayers > 0 ? `${model.backend}, ${model.gpuLayers} layers on the GPU` : 'CPU only'}</dd>
                <dt>Context</dt>
                <dd>
                  {model.contextSize.toLocaleString()} tokens
                  {model.trainContextSize > model.contextSize
                    ? ` (trained for ${model.trainContextSize.toLocaleString()})`
                    : ''}
                  , {model.kvCache === 'q8_0' ? '8-bit' : '16-bit'} memory
                </dd>
                <dt>File</dt>
                <dd>{formatBytes(model.fileSizeBytes)}</dd>
              </dl>
              <div className="mm-kv">
                <div className="mm-kv-row">
                  <span>Context memory</span>
                  <div className="seg" role="group" aria-label="Context memory">
                    <button aria-pressed={config.model.kvCache === 'f16'} onClick={() => setKvCache('f16')}>
                      16-bit
                    </button>
                    <button aria-pressed={config.model.kvCache === 'q8_0'} onClick={() => setKvCache('q8_0')}>
                      8-bit
                    </button>
                  </div>
                </div>
                <p className="help">
                  8-bit fits about twice as many tokens in the same GPU memory, at a small cost in precision.
                  Experimental: if it fails to load, the studio switches back to 16-bit by itself.
                </p>
                {config.model.kvCache !== model.kvCache ? (
                  <button className="primary small" disabled={busy} onClick={fromMenu(() => void loadPath(model.path))}>
                    Reload with {config.model.kvCache === 'q8_0' ? '8-bit' : '16-bit'} memory
                  </button>
                ) : null}
              </div>
              <div className="mm-actions">
                <button onClick={fromMenu(() => void pickAndLoad())}>Change model…</button>
                <button onClick={fromMenu(() => void window.api.unloadModel())}>Unload</button>
              </div>
            </div>
          </>
        ) : (
          <>
            <div className="model-chip idle">No model loaded</div>
            <button onClick={pickAndLoad}>Load a GGUF model…</button>
          </>
        )}
      </div>

      {tab === 'chat' ? (
        <div className={`body with-workspace ${showSidebar ? '' : 'no-sidebar'}`}>
          {showSidebar ? (
            <WorkspaceSidebar
              state={ws}
              busy={busy}
              onOpenFolder={() => void act(() => window.api.workspace.openFolder())}
              onSwitch={(id) => void act(() => window.api.workspace.switchTo(id))}
              onForget={(id) => void act(() => window.api.workspace.forget(id))}
              onNewChat={newChat}
              onOpenChat={(id) => void act(() => window.api.workspace.openChat(id))}
              onDeleteChat={(id) => {
                const title = ws.chats.find((c) => c.id === id)?.title ?? 'this chat'
                if (window.confirm(`Delete “${title}”? Its messages cannot be brought back.`)) {
                  void act(() => window.api.workspace.deleteChat(id))
                }
              }}
            />
          ) : null}
          <div className="main-pane">
            {/* With the sidebar hidden, where the agent is working and whether it
                can reach the internet stay in view. */}
            {!showSidebar && active ? (
              <div className="chat-head">
                <strong className="truncate" title={active.path}>
                  {active.name}
                </strong>
                <span className="chat-head-title truncate">{chatTitle}</span>
                <span className={`ws-net ws-net-${active.internet}`}>{internetLabel(active.internet)}</span>
                <div className="spacer" />
                <button className="small" onClick={newChat} disabled={busy}>
                  New chat
                </button>
              </div>
            ) : null}
            <div className="transcript" ref={transcriptRef}>
              {turns.length === 0 ? (
                <div className="empty">
                  {!active ? (
                    <>
                      <p>
                        Open a folder to start. It becomes the agent&rsquo;s working folder, and the chats
                        you have there are kept for it.
                      </p>
                      <button className="primary" onClick={() => void act(() => window.api.workspace.openFolder())}>
                        Open folder…
                      </button>
                    </>
                  ) : !model ? (
                    active.internet === 'offline'
                      ? 'Load a GGUF model to begin. This folder is offline: nothing leaves this machine.'
                      : 'Load a GGUF model to begin. Internet access is on for this folder; every request is shown first and logged.'
                  ) : (
                    `Working in ${active.name}. Ask for something; every action it takes is listed on the right.`
                  )}
                </div>
              ) : (
                turns.map((turn) => (
                  <div className={`turn ${turn.role}`} key={turn.id}>
                    <div className="who">
                      {turn.role === 'user' ? 'You' : turn.role === 'system' ? 'App' : 'Model'}
                    </div>
                    {/* Leading blank lines are dropped: some models open every reply with
                        the newlines left behind by an empty thinking block. */}
                    <div className={`bubble ${turn.streaming ? 'cursor' : ''}`}>{turn.text.trimStart()}</div>
                    {turn.role === 'assistant' && !turn.streaming ? (
                      <ChangeBar group={changes.find((g) => g.id === turn.id)} onRevert={revert} disabled={busy} />
                    ) : null}
                  </div>
                ))
              )}
            </div>

            <div className="composer">
              <textarea
                value={input}
                placeholder={
                  !active ? 'Open a folder first' : model ? askIn(active.name) : 'Load a model first'
                }
                disabled={!model || !active}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    void send()
                  }
                }}
              />
              {activePrompt ? (
                <button className="danger" onClick={() => void window.api.abort(activePrompt)}>
                  Stop
                </button>
              ) : (
                <button
                  className="primary"
                  disabled={!model || !active || !input.trim()}
                  onClick={() => void send()}
                >
                  Send
                </button>
              )}
            </div>
          </div>

          <div className="side-pane">
            <h2>Activity</h2>
            <ActivityLog outcomes={outcomes} />
          </div>
        </div>
      ) : tab === 'usage' ? (
        <UsagePanel
          snapshot={usage}
          model={model}
          onRefresh={() => void window.api.refreshUsage()}
          onNewChat={newChat}
        />
      ) : tab === 'permissions' ? (
        <PermissionsPanel
          config={config}
          workspace={active}
          sessionGrantCount={sessionGrants}
          onAddRoot={() => void update(() => window.api.workspace.addFolder())}
          onToggleRoot={(id, enabled) => void update(() => window.api.workspace.setFolderEnabled(id, enabled))}
          onRemoveRoot={(id) => void update(() => window.api.workspace.removeFolder(id))}
          onRemoveRule={(id) => void update(() => window.api.workspace.removeGrant(id))}
          onSetMode={(risk: RiskClass, mode: ClassMode) =>
            void window.api.setClassMode(risk, mode).then(setConfig)
          }
          onSetConfirmEverything={(value) =>
            void window.api.setConfirmEverything(value).then(setConfig)
          }
          onClearSessionGrants={() => {
            void window.api.clearSessionGrants()
            setSessionGrants(0)
          }}
          onSetProtected={(list) => void update(() => window.api.workspace.setProtected(list))}
          onSetInternet={(level) => void update(() => window.api.workspace.setInternet(level))}
          changes={changes}
          onRevert={(group) => void revert(group)}
        />
      ) : (
        <div className="panel">
          {active ? (
            <RulesPanel
              heading={`Project rules · ${active.name}`}
              help="Followed only in this folder, after the global rules. Put what is particular to this project here: its conventions, what to leave alone, how to test."
              rules={active.rules}
              model={model}
              onSave={(rules: GlobalRule[]) => void update(() => window.api.workspace.saveRules(rules))}
            />
          ) : (
            <section>
              <h3>Project rules</h3>
              <p className="help">Open a folder to give it rules of its own.</p>
            </section>
          )}
          <RulesPanel
            heading="Global rules"
            help="Followed in every folder."
            rules={config.rules}
            model={model}
            onSave={(rules: GlobalRule[]) => void window.api.saveRules(rules).then(setConfig)}
          />
          <PromptPreview rules={config.rules} context={context} />
        </div>
      )}

      {approval ? <ApprovalDialog request={approval} onRespond={respond} /> : null}
    </div>
  )
}

/** The composer's placeholder. Placeholders wrap rather than truncate, so a long folder name is cut here. */
function askIn(folder: string): string {
  return `Ask for something in ${folder.length > 32 ? folder.slice(0, 31) : folder}…`
}

/** Under a reply that changed files: what it changed, and a way to put it all back. */
function ChangeBar({
  group,
  onRevert,
  disabled
}: {
  group: ChangeGroup | undefined
  onRevert: (group: ChangeGroup) => void
  disabled: boolean
}): React.JSX.Element | null {
  if (!group || group.files.length === 0) return null
  const count = group.files.length
  return (
    <div className={`change-bar ${group.reverted ? 'reverted' : ''}`}>
      <span title={group.files.map((f) => f.path).join('\n')}>
        {group.reverted ? 'Put back: ' : 'Changed '}
        {count} file{count === 1 ? '' : 's'}
      </span>
      {group.reverted ? null : (
        <button className="ghost small" disabled={disabled} onClick={() => onRevert(group)}>
          ↺ Revert this reply
        </button>
      )}
    </div>
  )
}
