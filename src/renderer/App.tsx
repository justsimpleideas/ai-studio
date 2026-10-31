import { useEffect, useState } from 'react'
import { ChatStudio } from './ChatStudio.js'
import { AppSettings } from './components/AppSettings.js'
import { ImageStudio } from './image/ImageStudio.js'

type Side = 'chat' | 'image' | 'settings'

const SIDE_KEY = 'studio.side'

/**
 * The studio shell: a rail to switch between the chat side, the image side and
 * the studio's own settings.
 *
 * Both sides stay mounted, so a conversation, a running batch or an open gallery
 * survives switching back and forth. Only one model is on the GPU at a time; the
 * main process handles that, and its notices appear here.
 */
export function App(): React.JSX.Element {
  const [side, setSide] = useState<Side>(() => {
    const stored = localStorage.getItem(SIDE_KEY)
    return stored === 'image' || stored === 'settings' ? stored : 'chat'
  })
  const [notice, setNotice] = useState<string | null>(null)

  useEffect(() => localStorage.setItem(SIDE_KEY, side), [side])

  // Dark or light, from the config; changed in Settings and applied at once.
  useEffect(() => {
    const apply = (theme: string): void => {
      if (theme === 'light') document.documentElement.dataset['theme'] = 'light'
      else delete document.documentElement.dataset['theme']
    }
    void window.api.getConfig().then((c) => apply(c.theme))
    return window.api.onConfigChanged((c) => apply(c.theme))
  }, [])

  // A file dropped anywhere but the image side's drop zone does nothing, rather
  // than the window opening it in place of the studio. (The main process blocks
  // that navigation as well; this keeps the cursor honest.)
  useEffect(() => {
    const refuse = (e: DragEvent): void => {
      if (e.defaultPrevented) return
      e.preventDefault()
      if (e.dataTransfer) e.dataTransfer.dropEffect = 'none'
    }
    window.addEventListener('dragover', refuse)
    window.addEventListener('drop', refuse)
    return () => {
      window.removeEventListener('dragover', refuse)
      window.removeEventListener('drop', refuse)
    }
  }, [])

  useEffect(() => {
    let timer: number | undefined
    const off = window.api.onNotice((message) => {
      setNotice(message)
      window.clearTimeout(timer)
      // A failure stays until it is dismissed: reading why a model did not load
      // should not be a race against a timer.
      if (!/could not be loaded|stopped unexpectedly/.test(message)) timer = window.setTimeout(() => setNotice(null), 7000)
    })
    return () => {
      off()
      window.clearTimeout(timer)
    }
  }, [])

  return (
    <div className="studio">
      <nav className="rail" aria-label="Studio">
        <RailButton label="Chat" active={side === 'chat'} onClick={() => setSide('chat')}>
          <path d="M5 6h14a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-7l-4 3v-3H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2z" />
        </RailButton>
        <RailButton label="Image" active={side === 'image'} onClick={() => setSide('image')}>
          <rect x="3" y="4" width="18" height="16" rx="2" />
          <circle cx="9" cy="10" r="2" />
          <path d="M3 17l5-5 4 4 3-3 6 6" />
        </RailButton>
        <RailButton label="Settings" active={side === 'settings'} onClick={() => setSide('settings')} bottom>
          <circle cx="12" cy="12" r="3" />
          <path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M5.3 5.3l2.1 2.1M16.6 16.6l2.1 2.1M5.3 18.7l2.1-2.1M16.6 7.4l2.1-2.1" />
        </RailButton>
      </nav>
      <div className="studio-pane" hidden={side !== 'chat'}>
        <ChatStudio visible={side === 'chat'} />
      </div>
      <div className="studio-pane" hidden={side !== 'image'}>
        <ImageStudio active={side === 'image'} />
      </div>
      <div className="studio-pane" hidden={side !== 'settings'}>
        <AppSettings />
      </div>
      {notice && (
        <div className="studio-notice" role="status">
          {notice}
          <button className="ghost" onClick={() => setNotice(null)} aria-label="Dismiss">
            ✕
          </button>
        </div>
      )}
    </div>
  )
}

function RailButton({
  label,
  active,
  onClick,
  bottom = false,
  children
}: {
  label: string
  active: boolean
  onClick: () => void
  /** Sits at the foot of the rail, apart from the two sides. */
  bottom?: boolean
  children: React.ReactNode
}): React.JSX.Element {
  return (
    <button
      className={`rail-button ${active ? 'active' : ''} ${bottom ? 'rail-bottom' : ''}`}
      aria-current={active ? 'page' : undefined}
      onClick={onClick}
    >
      <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" aria-hidden>
        {children}
      </svg>
      <span>{label}</span>
    </button>
  )
}
