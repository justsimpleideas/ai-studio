import { useCallback, useEffect, useState } from 'react'
import { api } from './api.js'
import type { GalleryFolder, ImageRecord } from './types.js'

export interface Folders {
  list: GalleryFolder[]
  /** Increments whenever folder membership changed, so gallery views know to reload. */
  version: number
  refresh(): Promise<void>
  create(name: string): Promise<GalleryFolder>
  rename(id: string, name: string): Promise<GalleryFolder>
  remove(id: string): Promise<void>
  move(itemId: string, folderId: string | null): Promise<ImageRecord>
  setHidden(id: string, hidden: boolean): Promise<GalleryFolder>
  /** Set, change or (pin null) remove a PIN; `current` is needed once the folder has one. */
  setPin(id: string, pin: string | null, current: string | null): Promise<GalleryFolder>
  unlock(id: string, pin: string): Promise<GalleryFolder>
  lock(id: string): Promise<GalleryFolder>
}

/**
 * The gallery's folders, shared by the gallery view and the detail window. Hidden folders
 * are in the list only while shown; the engine decides that per session (gallery.py), and
 * `privacyVersion` (an engine event) brings the list up to date when it changes.
 */
export function useFolders(enabled: boolean, privacyVersion = 0): Folders {
  const [list, setList] = useState<GalleryFolder[]>([])
  const [version, setVersion] = useState(0)

  const refresh = useCallback(async () => {
    try {
      setList(await api.folders())
    } catch {
      // The engine is not running yet; the list fills in once it is.
    }
  }, [])

  useEffect(() => {
    if (enabled) void refresh()
  }, [enabled, refresh, privacyVersion])

  // The list is refreshed before the change resolves, so a view switched to a new or just
  // unlocked folder finds it there.
  const after = <T,>(run: () => Promise<T>) => async (): Promise<T> => {
    const result = await run()
    setVersion((v) => v + 1)
    await refresh()
    return result
  }

  return {
    list,
    version,
    refresh,
    create: (name) => after(() => api.createFolder(name))(),
    rename: (id, name) => after(() => api.renameFolder(id, name))(),
    remove: async (id) => {
      await after(() => api.deleteFolder(id))()
    },
    move: (itemId, folderId) => after(() => api.moveToFolder(itemId, folderId))(),
    setHidden: (id, hidden) => after(() => api.setFolderHidden(id, hidden))(),
    setPin: (id, pin, current) => after(() => api.setFolderPin(id, pin, current))(),
    unlock: (id, pin) => after(() => api.unlockFolder(id, pin))(),
    lock: (id) => after(() => api.lockFolder(id))()
  }
}
