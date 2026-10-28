/**
 * Keyboard shortcuts that can be changed in Settings. A binding is written as its keys
 * joined with "+", modifiers first in a fixed order: "Ctrl+Shift+L", "Alt+F2", "Ctrl+Enter".
 * The Mac's Command key counts as Ctrl, as it does everywhere else in the studio.
 */

export type ShortcutId = 'toggleSidebar' | 'lockFolders' | 'generate'
export type ShortcutMap = Record<ShortcutId, string>

export const SHORTCUTS: readonly { id: ShortcutId; label: string; detail: string }[] = [
  {
    id: 'toggleSidebar',
    label: 'Show or hide the sidebar',
    detail: 'The folder and chats on the chat side; the folder list in the gallery.'
  },
  {
    id: 'lockFolders',
    label: 'Lock every open folder',
    detail: 'From anywhere in the studio. A gallery view of one of them goes to Favorites.'
  },
  { id: 'generate', label: 'Generate', detail: 'From the prompt of the Create and Video tabs.' }
]

export const DEFAULT_SHORTCUTS: ShortcutMap = {
  toggleSidebar: 'Ctrl+B',
  lockFolders: 'Ctrl+Shift+L',
  generate: 'Ctrl+Enter'
}

/** Keys that do one thing wherever they are pressed, listed so Settings shows every shortcut. */
export const FIXED_SHORTCUTS: readonly { keys: string; label: string }[] = [
  { keys: 'Enter', label: 'Send a chat message (Shift+Enter starts a new line)' },
  { keys: 'Shift+F10, or the Menu key', label: 'Open the menu of the gallery item in focus' },
  { keys: 'Right-click', label: 'Open the menu of a gallery item' },
  { keys: 'Double-click', label: 'Rename a gallery folder' },
  { keys: 'Esc', label: 'Close a menu, or clear the settings search' }
]

const MODIFIERS = ['Ctrl', 'Alt', 'Shift'] as const
type Modifier = (typeof MODIFIERS)[number]

// Editing keys every text box needs, and closing the window.
const RESERVED = new Set(['Ctrl+A', 'Ctrl+C', 'Ctrl+V', 'Ctrl+X', 'Ctrl+Y', 'Ctrl+Z', 'Ctrl+Shift+Z', 'Alt+F4'])

const NAMED_KEYS = new Set(['Enter', 'Space', 'Tab', 'Backspace', 'Delete', 'Insert', 'Home', 'End', 'PageUp', 'PageDown',
  'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'])

function isKeyName(key: string): boolean {
  return /^[A-Z0-9]$/.test(key) || /^F([1-9]|1[0-2])$/.test(key) || NAMED_KEYS.has(key)
}

/** A binding in its one written form, or null if it is not one. */
export function normalizeShortcut(binding: unknown): string | null {
  if (typeof binding !== 'string') return null
  const parts = binding.split('+').map((p) => p.trim()).filter(Boolean)
  if (parts.length === 0) return null
  const raw = parts.at(-1)!
  const key = raw.length === 1 ? raw.toUpperCase() : raw
  const mods = new Set<Modifier>()
  for (const part of parts.slice(0, -1)) {
    const mod = MODIFIERS.find((m) => m.toLowerCase() === part.toLowerCase())
    if (!mod || mods.has(mod)) return null
    mods.add(mod)
  }
  if (!isKeyName(key)) return null
  return [...MODIFIERS.filter((m) => mods.has(m)), key].join('+')
}

export interface KeyLike {
  key: string
  code?: string
  ctrlKey: boolean
  altKey: boolean
  shiftKey: boolean
  metaKey: boolean
}

/** The binding a key press makes, or null while only modifiers are down. */
export function shortcutOf(e: KeyLike): string | null {
  // The physical key, so Shift and the keyboard layout do not change which key it is.
  const code = e.code ?? ''
  let key =
    /^Key([A-Z])$/.exec(code)?.[1] ??
    /^Digit([0-9])$/.exec(code)?.[1] ??
    (code === 'NumpadEnter' ? 'Enter' : null)
  if (!key) {
    key = e.key === ' ' ? 'Space' : e.key.length === 1 ? e.key.toUpperCase() : e.key
  }
  if (!isKeyName(key)) return null
  const mods: Modifier[] = []
  if (e.ctrlKey || e.metaKey) mods.push('Ctrl')
  if (e.altKey) mods.push('Alt')
  if (e.shiftKey) mods.push('Shift')
  return [...mods, key].join('+')
}

export function matchesShortcut(e: KeyLike, binding: string): boolean {
  return shortcutOf(e) === binding
}

/** What is wrong with giving `id` this binding, or null if it can have it. */
export function shortcutProblem(id: ShortcutId, binding: string, all: ShortcutMap): string | null {
  const normal = normalizeShortcut(binding)
  if (!normal) return 'That is not a key this shortcut can use.'
  const key = normal.split('+').at(-1)!
  const hasCtrlOrAlt = normal.startsWith('Ctrl+') || normal.includes('Alt+')
  if (!hasCtrlOrAlt && !/^F([1-9]|1[0-2])$/.test(key)) {
    return 'Use Ctrl or Alt with it (or an F key), so it does not get in the way of typing.'
  }
  if (RESERVED.has(normal)) return `${normal} is kept for editing text and closing windows.`
  const taken = SHORTCUTS.find((s) => s.id !== id && all[s.id] === normal)
  if (taken) return `${normal} already does “${taken.label}”.`
  return null
}

/** Read bindings defensively: anything unusable, or already taken, falls back to the default. */
export function parseShortcuts(value: unknown): ShortcutMap {
  const raw = (value && typeof value === 'object' ? value : {}) as Record<string, unknown>
  const out = { ...DEFAULT_SHORTCUTS }
  for (const { id } of SHORTCUTS) {
    const normal = normalizeShortcut(raw[id])
    if (normal) out[id] = normal
  }
  for (const { id } of SHORTCUTS) {
    if (shortcutProblem(id, out[id], out)) out[id] = DEFAULT_SHORTCUTS[id]
  }
  // A default can collide with another's custom binding; then both go back to defaults.
  if (SHORTCUTS.some(({ id }) => shortcutProblem(id, out[id], out))) return { ...DEFAULT_SHORTCUTS }
  return out
}
