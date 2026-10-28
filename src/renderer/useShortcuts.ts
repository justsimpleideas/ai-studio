import { useEffect, useRef, useState } from 'react'
import { DEFAULT_SHORTCUTS, type ShortcutMap } from '@shared/shortcuts.js'

// One copy of the bindings for the whole window, kept current from the config.
let current: ShortcutMap = DEFAULT_SHORTCUTS
const listeners = new Set<(shortcuts: ShortcutMap) => void>()
let started = false

function publish(shortcuts: ShortcutMap | undefined): void {
  if (!shortcuts) return
  current = shortcuts
  for (const listener of listeners) listener(current)
}

function start(): void {
  if (started || typeof window === 'undefined' || !window.api) return
  started = true
  void window.api.getConfig().then((c) => publish(c.shortcuts))
  window.api.onConfigChanged((c) => publish(c.shortcuts))
}

/** The keyboard shortcuts as set in Settings; re-renders when they change. */
export function useShortcuts(): ShortcutMap {
  const [shortcuts, setShortcuts] = useState(current)
  useEffect(() => {
    start()
    listeners.add(setShortcuts)
    setShortcuts(current)
    return () => {
      listeners.delete(setShortcuts)
    }
  }, [])
  return shortcuts
}

/** The same, through a ref, for key listeners added once. */
export function useShortcutsRef(): React.MutableRefObject<ShortcutMap> {
  const shortcuts = useShortcuts()
  const ref = useRef(shortcuts)
  ref.current = shortcuts
  return ref
}
