import fs from 'node:fs'
import path from 'node:path'

/**
 * Where the "Choose a GGUF model" dialog opens: on the last model loaded, so
 * opening it again is one click, or in its folder if the file has gone since.
 * Undefined (the system's own choice) when neither is there any more.
 */
export function modelPickerStart(lastModelPath: string | null, exists: (p: string) => boolean = fs.existsSync): string | undefined {
  if (!lastModelPath) return undefined
  if (exists(lastModelPath)) return lastModelPath
  const folder = path.dirname(lastModelPath)
  return folder !== lastModelPath && exists(folder) ? folder : undefined
}
