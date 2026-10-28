import { useEffect, useState } from 'react'
import {
  DEFAULT_BACKGROUND,
  DEFAULT_FOLDER_PRIVACY,
  type BackgroundPreferences,
  type CloseBehavior,
  type FolderPrivacyPreferences,
  type ModelSlot,
  type Theme
} from '@shared/config.js'
import {
  DEFAULT_SHORTCUTS,
  FIXED_SHORTCUTS,
  SHORTCUTS,
  shortcutOf,
  shortcutProblem,
  type ShortcutId,
  type ShortcutMap
} from '@shared/shortcuts.js'

/**
 * Settings for the studio as a whole (the chat and image sides keep their own).
 *
 * What closing the window does: quitting is the default because it is the only
 * choice that gives the GPU back for certain; the background option is for keeping
 * long work going, or a model warm, with the window out of the way. Then when open
 * gallery folders lock again, and the keyboard shortcuts.
 */

const BEHAVIORS: { value: CloseBehavior; label: string; detail: string }[] = [
  {
    value: 'quit',
    label: 'Quit',
    detail: 'Closing the window stops the studio and unloads every model, which frees the GPU. Work still running is stopped.'
  },
  {
    value: 'background',
    label: 'Keep running in the background',
    detail:
      'Closing the window only hides it. The studio stays in the tray (next to the clock): click the icon to open it again, or right-click it to quit. Work in progress carries on.'
  }
]

const MODELS: { slot: ModelSlot; label: string }[] = [
  { slot: 'chat', label: 'Chat model' },
  { slot: 'image', label: 'Image model' },
  { slot: 'video', label: 'Video model' }
]

const PRIVACY: { key: keyof FolderPrivacyPreferences; label: string; detail?: string }[] = [
  {
    key: 'lockOnBlur',
    label: 'When another window is chosen',
    detail: 'Another program or the desktop clicked, or Alt+Tab. Not the studio’s own file pickers.'
  },
  { key: 'lockOnMinimize', label: 'When the window is minimized' },
  { key: 'lockOnBackground', label: 'When the window closes to the background' }
]

const cleanError = (e: unknown): string =>
  (e as Error).message.replace(/^Error invoking remote method '[^']+': (?:\w*Error: )?/, '')

export function AppSettings(): React.JSX.Element {
  const [prefs, setPrefs] = useState<BackgroundPreferences>(DEFAULT_BACKGROUND)
  const [privacy, setPrivacy] = useState<FolderPrivacyPreferences>(DEFAULT_FOLDER_PRIVACY)
  const [shortcuts, setShortcuts] = useState<ShortcutMap>(DEFAULT_SHORTCUTS)
  const [theme, setTheme] = useState<Theme>('dark')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const apply = (c: Awaited<ReturnType<typeof window.api.getConfig>>): void => {
      setPrefs(c.background)
      setPrivacy(c.folderPrivacy)
      setShortcuts(c.shortcuts)
      setTheme(c.theme)
    }
    void window.api.getConfig().then(apply)
    return window.api.onConfigChanged(apply)
  }, [])

  const saveTheme = async (next: Theme): Promise<void> => {
    setTheme(next)
    try {
      setTheme((await window.api.setTheme(next)).theme)
    } catch (e) {
      setError(cleanError(e))
    }
  }

  const savePrivacy = async (next: FolderPrivacyPreferences): Promise<void> => {
    setPrivacy(next)
    try {
      setPrivacy((await window.api.setFolderPrivacy(next)).folderPrivacy)
    } catch (e) {
      setError(cleanError(e))
    }
  }

  const save = async (next: BackgroundPreferences): Promise<void> => {
    setPrefs(next)
    try {
      setPrefs((await window.api.setBackground(next)).background)
      setError(null)
    } catch (e) {
      setError(cleanError(e))
    }
  }

  const background = prefs.closeBehavior === 'background'

  return (
    <section className="app-settings">
      <header className="topbar">
        <h1 className="app-settings-title">Settings</h1>
      </header>
      <div className="app-settings-body">
        <div className="settings-card">
          <h2>Appearance</h2>
          <div className="keep-row">
            <span className="keep-label">Theme</span>
            <div className="segmented-pair" role="radiogroup" aria-label="Theme">
              {(
                [
                  ['dark', 'Dark'],
                  ['light', 'Light']
                ] as const
              ).map(([value, label]) => (
                <button
                  key={value}
                  role="radio"
                  aria-checked={theme === value}
                  className={theme === value ? 'active' : ''}
                  onClick={() => void saveTheme(value)}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="settings-card">
          <h2>When the window is closed</h2>
          <div className="choice-list" role="radiogroup" aria-label="When the window is closed">
            {BEHAVIORS.map((b) => (
              <label key={b.value} className={`choice ${prefs.closeBehavior === b.value ? 'active' : ''}`}>
                <input
                  type="radio"
                  name="close-behavior"
                  value={b.value}
                  checked={prefs.closeBehavior === b.value}
                  onChange={() => void save({ ...prefs, closeBehavior: b.value })}
                />
                <span>
                  <span className="choice-label">{b.label}</span>
                  <span className="choice-detail">{b.detail}</span>
                </span>
              </label>
            ))}
          </div>

          <div className={`keep-models ${background ? '' : 'off'}`} aria-disabled={!background}>
            <h3>In the background, each model…</h3>
            {MODELS.map((m) => (
              <div className="keep-row" key={m.slot}>
                <span className="keep-label">{m.label}</span>
                <div className="segmented-pair" role="radiogroup" aria-label={m.label}>
                  {([true, false] as const).map((keep) => (
                    <button
                      key={String(keep)}
                      role="radio"
                      aria-checked={prefs.keepLoaded[m.slot] === keep}
                      className={prefs.keepLoaded[m.slot] === keep ? 'active' : ''}
                      disabled={!background}
                      onClick={() => void save({ ...prefs, keepLoaded: { ...prefs.keepLoaded, [m.slot]: keep } })}
                    >
                      {keep ? 'Stays on the GPU' : 'Is unloaded'}
                    </button>
                  ))}
                </div>
              </div>
            ))}
            <p className="choice-detail">
              {background
                ? 'Staying on the GPU means no wait when you come back, but its memory stays taken from games and other apps. A model being unloaded finishes its current work first. The studio holds one model on the GPU at a time either way.'
                : 'Applies when closing the window keeps the studio running in the background.'}
            </p>
          </div>
          {error && <p className="error-text">{error}</p>}
        </div>

        <div className="settings-card">
          <h2>Locked gallery folders</h2>
          <p className="choice-detail">
            A folder opened with its PIN stays open until the studio quits, which always locks it again (and hides
            again any hidden folders shown for now). These can do the same sooner.
          </p>
          {PRIVACY.map((row) => {
            const off = row.key === 'lockOnBackground' && !background
            return (
              <div className={`keep-row ${off ? 'off' : ''}`} key={row.key}>
                <span className="keep-label">
                  {row.label}
                  {row.detail && <span className="choice-detail">{row.detail}</span>}
                </span>
                <div className="segmented-pair" role="radiogroup" aria-label={row.label}>
                  {([true, false] as const).map((lock) => (
                    <button
                      key={String(lock)}
                      role="radio"
                      aria-checked={privacy[row.key] === lock}
                      className={privacy[row.key] === lock ? 'active' : ''}
                      disabled={off}
                      onClick={() => void savePrivacy({ ...privacy, [row.key]: lock })}
                    >
                      {lock ? 'Lock them again' : 'Leave them open'}
                    </button>
                  ))}
                </div>
              </div>
            )
          })}
          {!background && (
            <p className="choice-detail">Closing the window quits the studio now, which locks them in any case.</p>
          )}
        </div>

        <ShortcutSettings shortcuts={shortcuts} onSaved={setShortcuts} />
      </div>
    </section>
  )
}

/**
 * Each changeable shortcut with the keys it takes: press "Change", then the new keys
 * (Esc cancels). A binding another shortcut has, or one that would get in the way of
 * typing, is refused with the reason. Below them, the keys that cannot be changed.
 */
function ShortcutSettings({ shortcuts, onSaved }: { shortcuts: ShortcutMap; onSaved: (s: ShortcutMap) => void }): React.JSX.Element {
  const [recording, setRecording] = useState<ShortcutId | null>(null)
  const [problem, setProblem] = useState<{ id: ShortcutId; text: string } | null>(null)

  const save = async (id: ShortcutId, next: ShortcutMap): Promise<void> => {
    try {
      onSaved((await window.api.setShortcuts(next)).shortcuts)
      setProblem(null)
    } catch (e) {
      setProblem({ id, text: cleanError(e) })
    }
  }

  // While recording, the next key press is the binding; nothing else in the studio sees it.
  useEffect(() => {
    if (!recording) return
    const onKey = (e: KeyboardEvent): void => {
      e.preventDefault()
      e.stopImmediatePropagation()
      if (e.key === 'Escape') {
        setRecording(null)
        return
      }
      const binding = shortcutOf(e)
      if (!binding) return // only modifiers so far
      const text = shortcutProblem(recording, binding, shortcuts)
      if (text) {
        setProblem({ id: recording, text })
        return
      }
      setRecording(null)
      void save(recording, { ...shortcuts, [recording]: binding })
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  })

  const changed = SHORTCUTS.some((s) => shortcuts[s.id] !== DEFAULT_SHORTCUTS[s.id])

  return (
    <div className="settings-card shortcuts">
      <h2>Keyboard shortcuts</h2>
      {SHORTCUTS.map((s) => (
        <div className="shortcut-row" key={s.id}>
          <span>
            <span className="choice-label">{s.label}</span>
            <span className="choice-detail">{s.detail}</span>
            {problem?.id === s.id && (
              <span className="error-text shortcut-problem" role="alert">
                {problem.text}
              </span>
            )}
          </span>
          <span className="shortcut-keys">
            {recording === s.id ? <kbd className="recording">Press the new keys… (Esc cancels)</kbd> : <kbd>{shortcuts[s.id]}</kbd>}
            <button
              className="ghost small"
              aria-label={`Change the keys for ${s.label}`}
              onClick={() => {
                setProblem(null)
                setRecording(recording === s.id ? null : s.id)
              }}
            >
              {recording === s.id ? 'Cancel' : 'Change'}
            </button>
            {shortcuts[s.id] !== DEFAULT_SHORTCUTS[s.id] && recording !== s.id && (
              <button
                className="ghost small"
                aria-label={`Put back ${DEFAULT_SHORTCUTS[s.id]} for ${s.label}`}
                onClick={() => {
                  const next = { ...shortcuts, [s.id]: DEFAULT_SHORTCUTS[s.id] }
                  const text = shortcutProblem(s.id, DEFAULT_SHORTCUTS[s.id], next)
                  if (text) setProblem({ id: s.id, text })
                  else void save(s.id, next)
                }}
              >
                Default
              </button>
            )}
          </span>
        </div>
      ))}
      {changed && (
        <button className="ghost small reset-shortcuts" onClick={() => void save(SHORTCUTS[0]!.id, { ...DEFAULT_SHORTCUTS })}>
          Put every shortcut back to its default
        </button>
      )}
      <h3>Always</h3>
      {FIXED_SHORTCUTS.map((f) => (
        <div className="shortcut-row fixed" key={f.keys}>
          <span className="choice-detail">{f.label}</span>
          <kbd>{f.keys}</kbd>
        </div>
      ))}
    </div>
  )
}
