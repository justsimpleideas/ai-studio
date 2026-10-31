import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { DEFAULT_FOLDER_PRIVACY, parseFolderPrivacy } from '@shared/config.js'
import {
  DEFAULT_SHORTCUTS,
  SHORTCUTS,
  matchesShortcut,
  normalizeShortcut,
  parseShortcuts,
  shortcutOf,
  shortcutProblem,
  type KeyLike
} from '@shared/shortcuts.js'
import { ConfigStore } from '../src/main/store.js'

const press = (key: string, mods: Partial<KeyLike> = {}): KeyLike => ({
  key,
  ctrlKey: false,
  altKey: false,
  shiftKey: false,
  metaKey: false,
  ...mods
})

describe('keyboard shortcuts', () => {
  it('starts with the keys the studio always had', () => {
    expect(DEFAULT_SHORTCUTS).toEqual({ toggleSidebar: 'Ctrl+B', lockFolders: 'Ctrl+Shift+L', generate: 'Ctrl+Enter', pickFolder: 'Ctrl+M' })
    for (const { id } of SHORTCUTS) expect(shortcutProblem(id, DEFAULT_SHORTCUTS[id], DEFAULT_SHORTCUTS)).toBeNull()
  })

  it('writes a binding one way, modifiers first in a fixed order', () => {
    expect(normalizeShortcut('shift+ctrl+l')).toBe('Ctrl+Shift+L')
    expect(normalizeShortcut('Alt + f2')).toBeNull() // "f2" is not a key name; F2 is
    expect(normalizeShortcut('Alt+F2')).toBe('Alt+F2')
    expect(normalizeShortcut('Ctrl+Ctrl+B')).toBeNull()
    expect(normalizeShortcut('Ctrl+Hyper+B')).toBeNull()
    expect(normalizeShortcut('Ctrl+')).toBeNull()
    expect(normalizeShortcut(42)).toBeNull()
  })

  it('reads a key press by its physical key, with Command counting as Ctrl', () => {
    expect(shortcutOf(press('L', { code: 'KeyL', ctrlKey: true, shiftKey: true }))).toBe('Ctrl+Shift+L')
    // Shift turns "1" into "!", and another layout moves letters; the physical key does not change.
    expect(shortcutOf(press('!', { code: 'Digit1', ctrlKey: true, shiftKey: true }))).toBe('Ctrl+Shift+1')
    expect(shortcutOf(press('b', { metaKey: true }))).toBe('Ctrl+B')
    expect(shortcutOf(press('Enter', { code: 'NumpadEnter', ctrlKey: true }))).toBe('Ctrl+Enter')
    expect(shortcutOf(press('Control', { ctrlKey: true }))).toBeNull()
    expect(shortcutOf(press('Shift', { shiftKey: true }))).toBeNull()
  })

  it('matches only the exact keys', () => {
    expect(matchesShortcut(press('b', { ctrlKey: true }), 'Ctrl+B')).toBe(true)
    expect(matchesShortcut(press('b', { ctrlKey: true, shiftKey: true }), 'Ctrl+B')).toBe(false)
    expect(matchesShortcut(press('b'), 'Ctrl+B')).toBe(false)
  })

  it('refuses keys that would get in the way of typing, editing keys, and keys already taken', () => {
    expect(shortcutProblem('lockFolders', 'Shift+H', DEFAULT_SHORTCUTS)).toMatch(/Ctrl or Alt/)
    expect(shortcutProblem('lockFolders', 'H', DEFAULT_SHORTCUTS)).toMatch(/Ctrl or Alt/)
    expect(shortcutProblem('lockFolders', 'F9', DEFAULT_SHORTCUTS)).toBeNull()
    expect(shortcutProblem('lockFolders', 'Ctrl+C', DEFAULT_SHORTCUTS)).toMatch(/editing text/)
    expect(shortcutProblem('lockFolders', 'Alt+F4', DEFAULT_SHORTCUTS)).toMatch(/closing windows/)
    expect(shortcutProblem('lockFolders', 'Ctrl+B', DEFAULT_SHORTCUTS)).toMatch(/Show or hide the sidebar/)
    expect(shortcutProblem('lockFolders', 'Ctrl+Alt+K', DEFAULT_SHORTCUTS)).toBeNull()
  })

  it('reads saved bindings defensively', () => {
    expect(parseShortcuts(undefined)).toEqual(DEFAULT_SHORTCUTS)
    expect(parseShortcuts({ lockFolders: 'alt+k' })).toEqual({ ...DEFAULT_SHORTCUTS, lockFolders: 'Alt+K' })
    expect(parseShortcuts({ lockFolders: 'Ctrl+C', generate: 7 })).toEqual(DEFAULT_SHORTCUTS)
    // Two shortcuts on the same keys: one of them goes back to its default.
    expect(parseShortcuts({ toggleSidebar: 'Alt+K', lockFolders: 'Alt+K' })).toEqual({ ...DEFAULT_SHORTCUTS, lockFolders: 'Alt+K' })
    // A custom binding on another's default: when the defaults cannot all fit, all go back.
    const tangled = parseShortcuts({ toggleSidebar: 'Ctrl+Shift+L', lockFolders: 'Ctrl+Shift+L' })
    for (const { id } of SHORTCUTS) expect(shortcutProblem(id, tangled[id], tangled)).toBeNull()
  })
})

describe('folder privacy preferences', () => {
  it('locks on going to the background, and not on switching away or minimizing, by default', () => {
    expect(DEFAULT_FOLDER_PRIVACY).toEqual({ lockOnBlur: false, lockOnMinimize: false, lockOnBackground: true })
    expect(parseFolderPrivacy(null)).toEqual(DEFAULT_FOLDER_PRIVACY)
    expect(parseFolderPrivacy({ lockOnBlur: true, lockOnMinimize: true, lockOnBackground: 'no' })).toEqual({
      lockOnBlur: true,
      lockOnMinimize: true,
      lockOnBackground: true
    })
  })
})

describe('both on disk', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'shortcuts-test-'))
  afterAll(() => fs.rmSync(dir, { recursive: true, force: true }))

  it('come back as saved, and as defaults from an older or broken file', async () => {
    const file = path.join(dir, 'config.json')
    fs.writeFileSync(file, JSON.stringify({}))
    const store = new ConfigStore(file)
    expect(store.get().shortcuts).toEqual(DEFAULT_SHORTCUTS)
    expect(store.get().folderPrivacy).toEqual(DEFAULT_FOLDER_PRIVACY)
    await store.update((c) => {
      c.shortcuts = { ...c.shortcuts, lockFolders: 'Alt+K' }
      c.folderPrivacy = { lockOnBlur: true, lockOnMinimize: true, lockOnBackground: false }
    })
    const again = new ConfigStore(file)
    expect(again.get().shortcuts.lockFolders).toBe('Alt+K')
    expect(again.get().folderPrivacy).toEqual({ lockOnBlur: true, lockOnMinimize: true, lockOnBackground: false })
    fs.writeFileSync(file, JSON.stringify({ shortcuts: { generate: 'Ctrl+V' }, folderPrivacy: { lockOnMinimize: 1 } }))
    const broken = new ConfigStore(file)
    expect(broken.get().shortcuts).toEqual(DEFAULT_SHORTCUTS)
    expect(broken.get().folderPrivacy).toEqual(DEFAULT_FOLDER_PRIVACY)
  })
})
