import { useCallback, useEffect, useState } from 'react'
import type { LoraItem } from '@shared/training.js'
import { trainingApi } from './api.js'

/** The LoRA library, loaded once the engine is up and refreshed on demand. */
export function useLoras(enabled: boolean): { list: LoraItem[]; refresh: () => Promise<void> } {
  const [list, setList] = useState<LoraItem[]>([])
  const refresh = useCallback(async () => {
    try {
      setList(await trainingApi.loras())
    } catch {
      // the engine went away; the list is refreshed when it is back
    }
  }, [])
  useEffect(() => {
    if (enabled) void refresh()
  }, [enabled, refresh])
  return { list, refresh }
}
