import { useCallback, useEffect, useState } from 'react'
import { api } from './api.js'
import type { GalleryFolder, ImageRecord } from './types.js'

export interface Folders {
  list: GalleryFolder[]
  /** The list has been read from the engine at least once. */
  loaded: boolean
  /** Increments whenever folder membership changed, so gallery views know to reload. */
  version: number
  refresh(): Promise<void>
  create(name: string): Promise<GalleryFolder>
  rename(id: string, name: string): Promise<GalleryFolder>
  remove(id: string): Promise<void>
  move(itemId: string, folderId: string | null): Promise<ImageRecord>
  /** Several items at once (the gallery's multi-select); resolves to the ones moved. */
  moveMany(itemIds: string[], folderId: string | null): Promise<ImageRecord[]>
  /**
   * Put a folder just before `before` in the list (last with null). The list is
   * shown in its new order at once, and the engine keeps it (gallery.py).
   */
  reorder(id: string, before: string | null): Promise<void>
  setHidden(id: string, hidden: boolean): Promise<GalleryFolder>
  /** Set, change or (pin null) remove a PIN; `current` is needed once the folder has one. */
  setPin(id: string, pin: string | null, current: string | null): Promise<GalleryFolder>
  unlock(id: string, pin: string): Promise<GalleryFolder>
  lock(id: string): Promise<GalleryFolder>
  /** Lock every folder with a PIN that is open this session; resolves to how many were. */
  lockAll(): Promise<number>
}

/**
 * The gallery's folders, shared by the gallery view and the detail window. Hidden folders
 * are in the list only while shown; the engine decides that per session (gallery.py), and
 * `privacyVersion` (an engine event) brings the list up to date when it changes.
 */
export function useFolders(enabled: boolean, privacyVersion = 0): Folders {
  const [list, setList] = useState<GalleryFolder[]>([])
  const [loaded, setLoaded] = useState(false)
  const [version, setVersion] = useState(0)
  const refresh = useCallback(async () => {
    try {
      setList(await api.folders())
      setLoaded(true)
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
    loaded,
    version,
    refresh,
    create: (name) => after(() => api.createFolder(name))(),
    rename: (id, name) => after(() => api.renameFolder(id, name))(),
    remove: async (id) => {
      await after(() => api.deleteFolder(id))()
    },
    move: (itemId, folderId) => after(() => api.moveToFolder(itemId, folderId))(),
    moveMany: (itemIds, folderId) => after(() => api.moveManyToFolder(itemIds, folderId))(),
    // The answer is the list in its new order, so the move is on screen at once;
    // then a read, as every other action ends in, has the last word. `version` is
    // left alone: no item changed folder, so no gallery view needs reloading.
    reorder: async (id, before) => {
      setList(await api.reorderFolder(id, before))
      await refresh()
    },
    setHidden: (id, hidden) => after(() => api.setFolderHidden(id, hidden))(),
    setPin: (id, pin, current) => after(() => api.setFolderPin(id, pin, current))(),
    unlock: (id, pin) => after(() => api.unlockFolder(id, pin))(),
    lock: (id) => after(() => api.lockFolder(id))(),
    // Asks the engine which are open rather than trusting the list, which may be behind.
    lockAll: () =>
      after(async () => {
        const open = (await api.folders()).filter((f) => f.unlocked)
        await Promise.all(open.map((f) => api.lockFolder(f.id)))
        return open.length
      })()
  }
}
