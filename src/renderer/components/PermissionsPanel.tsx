import { useState } from 'react'
import type { AppConfig } from '@shared/config.js'
import type { ClassMode, ProtectedPath, ProtectionLevel } from '@shared/permissions.js'
import type { ChangeGroup } from '@shared/changes.js'
import type { RiskClass } from '@shared/tools.js'
import type { Workspace } from '@shared/workspace.js'
import type { InternetLevel } from '@shared/network.js'
import { InternetSection } from './InternetSection.js'

/**
 * Where the boundary is set.
 *
 * Folders and saved grants belong to the open workspace; the modes and the
 * confirm-everything switch apply in every workspace.
 *
 * The two halves are deliberately different in kind. Shared folders are the
 * boundary and cannot be argued with: a path outside them is refused whatever
 * the modes and rules say. Modes and rules only decide how much is asked about
 * inside that boundary. Presenting them as one list of settings would blur a
 * distinction the whole design depends on.
 */

const CLASS_COPY: Record<RiskClass, { title: string; help: string }> = {
  read: {
    title: 'Reading',
    help: 'Listing folders and reading files. Nothing is changed, but the contents do go into the model’s context.'
  },
  write: {
    title: 'Writing',
    help: 'Creating files and changing existing ones. Every change is backed up first, so it can be undone.'
  },
  destructive: {
    title: 'Deleting',
    help: 'Files go to the recycle bin rather than being erased, so a mistake here is recoverable.'
  },
  network: {
    title: 'Internet requests',
    help: 'Searches, page requests and downloads, when the folder’s internet level allows them. Asking shows exactly what will be sent and where.'
  }
}

function ModeToggle({
  risk,
  mode,
  disabled,
  onChange
}: {
  risk: RiskClass
  mode: ClassMode
  disabled: boolean
  onChange: (risk: RiskClass, mode: ClassMode) => void
}): React.JSX.Element {
  return (
    <div className="seg">
      <button
        aria-pressed={mode === 'ask'}
        disabled={disabled}
        onClick={() => onChange(risk, 'ask')}
      >
        Ask every time
      </button>
      <button
        aria-pressed={mode === 'auto'}
        disabled={disabled}
        onClick={() => onChange(risk, 'auto')}
      >
        Run automatically
      </button>
    </div>
  )
}

export function PermissionsPanel({
  config,
  workspace,
  sessionGrantCount,
  onAddRoot,
  onToggleRoot,
  onRemoveRoot,
  onRemoveRule,
  onSetMode,
  onSetConfirmEverything,
  onClearSessionGrants,
  onSetProtected,
  onSetInternet,
  onSetStrict,
  changes,
  onRevert
}: {
  config: AppConfig
  workspace: Workspace | null
  sessionGrantCount: number
  onAddRoot: () => void
  onToggleRoot: (id: string, enabled: boolean) => void
  onRemoveRoot: (id: string) => void
  onRemoveRule: (id: string) => void
  onSetMode: (risk: RiskClass, mode: ClassMode) => void
  onSetConfirmEverything: (value: boolean) => void
  onClearSessionGrants: () => void
  onSetProtected: (list: ProtectedPath[]) => void
  onSetInternet: (level: InternetLevel) => void
  onSetStrict: (strict: boolean) => void
  changes: ChangeGroup[]
  onRevert: (group: ChangeGroup) => void
}): React.JSX.Element {
  const { modes, confirmEverything } = config.permissions
  const roots = workspace?.extraRoots ?? []
  const rules = workspace?.policyRules ?? []

  return (
    <div className="panel">
      {workspace ? (
        <section className="strict-section">
          <h3>Strict mode</h3>
          <label className="toggle-row">
            <input type="checkbox" checked={workspace.strict} onChange={(e) => onSetStrict(e.target.checked)} />
            <span>
              <strong>Conversation only: the model gets no tools at all</strong>
              <span className="help block">
                No reading, listing or changing files, no internet, whatever the settings below say. The model is not told
                any tool exists, its output has no way to call one, and a call that arrives anyway is refused. For talking
                with a model you do not yet trust, or when you want nothing touched.
              </span>
            </span>
          </label>
        </section>
      ) : null}

      <section>
        <h3>Shared folders{workspace ? ` · ${workspace.name}` : ''}</h3>
        <p className="help">
          The agent can only reach paths inside these folders. Everything else is refused before any
          rule is considered, so a folder you have not shared cannot be opened by a rule, a setting,
          or a mistake. Shortcuts and junctions are followed to where they really point. Each
          workspace has its own list.
        </p>
        {!workspace ? (
          <p className="help" style={{ color: 'var(--warn)' }}>
            No folder is open. Open one from the Chat tab; it becomes the agent&rsquo;s working folder.
          </p>
        ) : (
          <div className="row">
            <input type="checkbox" checked disabled aria-label="Working folder, always shared" />
            <div className="grow">
              <div>
                {workspace.name} <span className="badge">working folder</span>
              </div>
              <code>{workspace.path}</code>
            </div>
          </div>
        )}
        {roots.map((root) => (
            <div className="row" key={root.id}>
              <input
                type="checkbox"
                checked={root.enabled}
                onChange={(e) => onToggleRoot(root.id, e.target.checked)}
                aria-label={`Enable ${root.label}`}
              />
              <div className="grow">
                <div>{root.label}</div>
                <code>{root.path}</code>
              </div>
              <button className="danger" onClick={() => onRemoveRoot(root.id)}>
                Remove
              </button>
            </div>
        ))}
        <div style={{ marginTop: 12 }}>
          <button onClick={onAddRoot} disabled={!workspace}>
            Share another folder…
          </button>
        </div>
      </section>

      {workspace ? <ProtectedPaths list={workspace.protectedPaths} onChange={onSetProtected} /> : null}

      {workspace ? <InternetSection level={workspace.internet} onSetLevel={onSetInternet} /> : null}

      <section>
        <h3>What gets asked about</h3>
        <p className="help">
          These apply inside the shared folders only. Turning something to automatic means you will
          not see it happen until you read the activity log.
        </p>
        <div className="mode-grid">
          {(Object.keys(CLASS_COPY) as RiskClass[]).map((risk) => (
            <div key={risk} style={{ display: 'contents' }}>
              <div>
                <strong>{CLASS_COPY[risk].title}</strong>
              </div>
              <div>
                <ModeToggle
                  risk={risk}
                  mode={modes[risk]}
                  disabled={confirmEverything}
                  onChange={onSetMode}
                />
                <p className="help" style={{ margin: '6px 0 0' }}>
                  {CLASS_COPY[risk].help}
                </p>
              </div>
            </div>
          ))}
        </div>

        <div className="row" style={{ marginTop: 14 }}>
          <input
            type="checkbox"
            id="confirm-everything"
            checked={confirmEverything}
            onChange={(e) => onSetConfirmEverything(e.target.checked)}
          />
          <label className="grow" htmlFor="confirm-everything">
            <strong>Confirm everything</strong>
            <p className="help" style={{ margin: '2px 0 0' }}>
              Ask about every action regardless of the settings above and any rules you have saved.
              A rule that denies something still denies it.
            </p>
          </label>
        </div>
      </section>

      <section>
        <h3>Saved rules</h3>
        <p className="help">
          Created when you choose “Always allow” in an approval prompt while working in this folder.
          Removing one puts that decision back to being asked about.
        </p>
        {rules.length === 0 ? (
          <p className="help">None yet.</p>
        ) : (
          rules.map((rule) => (
            <div className="row" key={rule.id}>
              <div className="grow">
                <code>{rule.pattern}</code>
                <div className="help" style={{ margin: 0 }}>
                  {rule.decision === 'allow' ? 'Allow' : 'Deny'}{' '}
                  {rule.applies === '*'
                    ? 'every action'
                    : Array.isArray(rule.applies)
                      ? rule.applies.join(', ')
                      : `${rule.applies} actions`}
                </div>
              </div>
              <button className="danger" onClick={() => onRemoveRule(rule.id)}>
                Remove
              </button>
            </div>
          ))
        )}
      </section>

      {workspace ? <ChangeHistory changes={changes} onRevert={onRevert} /> : null}

      <section>
        <h3>This session</h3>
        <p className="help">Grants you gave in this folder for the session only.</p>
        <div className="row">
          <div className="grow">
            {sessionGrantCount === 0
              ? 'No session-only grants.'
              : `${sessionGrantCount} session-only grant${sessionGrantCount === 1 ? '' : 's'} in effect.`}
          </div>
          <button disabled={sessionGrantCount === 0} onClick={onClearSessionGrants}>
            Revoke them
          </button>
        </div>
      </section>
    </div>
  )
}

const LEVEL_COPY: Record<ProtectionLevel, string> = {
  hidden: 'No access',
  'read-only': 'Read-only'
}

/**
 * Paths inside the shared folders that the agent cannot touch. Enforced by the
 * broker on every call, unlike the text rules on the Rules tab, which the model
 * can ignore.
 */
function ProtectedPaths({
  list,
  onChange
}: {
  list: ProtectedPath[]
  onChange: (list: ProtectedPath[]) => void
}): React.JSX.Element {
  const [pattern, setPattern] = useState('')
  const [level, setLevel] = useState<ProtectionLevel>('hidden')
  const set = (id: string, change: Partial<ProtectedPath>): void =>
    onChange(list.map((p) => (p.id === id ? { ...p, ...change } : p)))
  const add = (): void => {
    const trimmed = pattern.trim()
    if (!trimmed) return
    onChange([
      ...list,
      { id: `user:${Date.now()}`, pattern: trimmed, level, enabled: true, builtin: false }
    ])
    setPattern('')
  }

  return (
    <section>
      <h3>Protected paths</h3>
      <p className="help">
        Enforced on every action, whatever the model was told and whatever you approve. <b>No access</b>{' '}
        hides a path completely: it cannot be read, changed or even seen in a folder listing.{' '}
        <b>Read-only</b> can be read but never changed or deleted. A name like <code>.env</code> or{' '}
        <code>*.pem</code> matches at any depth; a path like <code>data/raw</code> is taken from the top
        of the folder. Rules written on the Rules tab are advice to the model; these are not.
      </p>
      {(() => {
        const defaults = list.filter((p) => p.builtin)
        const own = list.filter((p) => !p.builtin)
        const on = (level: ProtectionLevel): number => defaults.filter((p) => p.enabled && p.level === level).length
        return (
          <>
            {own.length > 0 ? <div className="protect-list">{own.map(row)}</div> : null}
            <details className="protect-defaults">
              <summary>
                Defaults: {on('hidden')} no access (secrets, keys, credentials) · {on('read-only')} read-only
                (.git, node_modules, virtual environments)
                {defaults.some((p) => !p.enabled) ? ` · ${defaults.filter((p) => !p.enabled).length} switched off` : ''}
              </summary>
              <div className="protect-list">{defaults.map(row)}</div>
            </details>
          </>
        )
      })()}
      <div className="row protect-add">
        <input
          type="text"
          value={pattern}
          placeholder="e.g. data/raw, *.sqlite, secrets"
          aria-label="Path to protect"
          onChange={(e) => setPattern(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') add()
          }}
        />
        <select value={level} aria-label="Protection" onChange={(e) => setLevel(e.target.value as ProtectionLevel)}>
          <option value="hidden">{LEVEL_COPY.hidden}</option>
          <option value="read-only">{LEVEL_COPY['read-only']}</option>
        </select>
        <button className="primary" disabled={!pattern.trim()} onClick={add}>
          Protect
        </button>
      </div>
    </section>
  )

  function row(p: ProtectedPath): React.JSX.Element {
    return (
          <div className={`protect-row ${p.enabled ? '' : 'off'}`} key={p.id}>
            <input
              type="checkbox"
              checked={p.enabled}
              aria-label={`Protect ${p.pattern}`}
              onChange={(e) => set(p.id, { enabled: e.target.checked })}
            />
            <code className="protect-pattern">{p.pattern}</code>
            <select
              value={p.level}
              aria-label={`Protection for ${p.pattern}`}
              onChange={(e) => set(p.id, { level: e.target.value as ProtectionLevel })}
            >
              <option value="hidden">{LEVEL_COPY.hidden}</option>
              <option value="read-only">{LEVEL_COPY['read-only']}</option>
            </select>
            {p.builtin ? (
              <span className="badge">default</span>
            ) : (
              <button className="ghost small" onClick={() => onChange(list.filter((x) => x.id !== p.id))}>
                Remove
              </button>
            )}
          </div>
    )
  }
}

function when(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

const basename = (p: string): string => p.split(/[\\/]/).pop() ?? p

/** Everything the agent changed here, by reply, each one revertible. Survives restarts. */
function ChangeHistory({
  changes,
  onRevert
}: {
  changes: ChangeGroup[]
  onRevert: (group: ChangeGroup) => void
}): React.JSX.Element {
  return (
    <section>
      <h3>Change history</h3>
      <p className="help">
        Every file the agent wrote, edited or deleted in this folder, grouped by the message that led to
        it. Old contents are backed up before each change, so a whole reply can be put back, including
        after a restart. Putting back is recorded too, and can itself be undone.
      </p>
      {changes.length === 0 ? (
        <p className="help">No changes yet.</p>
      ) : (
        changes.slice(0, 40).map((g) => (
          <div className="row change-row" key={g.id}>
            <div className="grow">
              <div className="truncate">
                {g.label} <span className="help">· {when(g.at)}</span>
              </div>
              <div className="change-files">
                {g.files.map((f) => (
                  <code key={f.path} title={f.path} className={`change-${f.kind}`}>
                    {f.kind === 'delete' ? '−' : f.existedBefore ? '~' : '+'} {basename(f.path)}
                  </code>
                ))}
              </div>
            </div>
            {g.reverted ? (
              <span className="badge">{g.revertOf ? 'undone' : 'put back'}</span>
            ) : (
              <button onClick={() => onRevert(g)}>{g.revertOf ? 'Undo this revert' : 'Revert'}</button>
            )}
          </div>
        ))
      )}
    </section>
  )
}
