import { useEffect, useState } from 'react'
import { DEFAULT_BACKGROUND, type BackgroundPreferences, type CloseBehavior, type ModelSlot } from '@shared/config.js'

/**
 * Settings for the studio as a whole (the chat and image sides keep their own).
 *
 * For now: what closing the window does. Quitting is the default because it is
 * the only choice that gives the GPU back for certain; the background option is
 * for keeping long work going, or a model warm, with the window out of the way.
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

export function AppSettings(): React.JSX.Element {
  const [prefs, setPrefs] = useState<BackgroundPreferences>(DEFAULT_BACKGROUND)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    void window.api.getConfig().then((c) => setPrefs(c.background))
    return window.api.onConfigChanged((c) => setPrefs(c.background))
  }, [])

  const save = async (next: BackgroundPreferences): Promise<void> => {
    setPrefs(next)
    try {
      setPrefs((await window.api.setBackground(next)).background)
      setError(null)
    } catch (e) {
      setError((e as Error).message.replace(/^Error invoking remote method '[^']+': (?:\w*Error: )?/, ''))
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
      </div>
    </section>
  )
}
