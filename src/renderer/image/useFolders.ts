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
}

/** The gallery's folders, shared by the gallery view and the detail window. */
export function useFolders(enabled: boolean): Folders {
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
  }, [enabled, refresh])

  const changed = useCallback(() => {
    setVersion((v) => v + 1)
    void refresh()
  }, [refresh])

  return {
    list,
    version,
    refresh,
    create: async (name) => {
      const folder = await api.createFolder(name)
      changed()
      return folder
    },
    rename: async (id, name) => {
      const folder = await api.renameFolder(id, name)
      changed()
      return folder
    },
    remove: async (id) => {
      await api.deleteFolder(id)
      changed()
    },
    move: async (itemId, folderId) => {
      const record = await api.moveToFolder(itemId, folderId)
      changed()
      return record
    }
  }
}
