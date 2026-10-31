import fsp from 'node:fs/promises'
import path from 'node:path'
import { isEngineId, type ImageRecord, type SaveManyOptions } from '@shared/image.js'

/**
 * Saving several gallery items into a folder the user picked (the gallery's
 * multi-select). Nothing already in that folder is ever overwritten: a name
 * that is taken gets " (2)", " (3)" and so on.
 */

/** The most items one multi-select action can name (the engine's MAX_BULK). */
export const MAX_BULK = 2000

export function requireIds(ids: unknown): string[] {
  if (!Array.isArray(ids) || ids.length === 0 || ids.length > MAX_BULK || !ids.every(isEngineId)) {
    throw new Error('Invalid selection.')
  }
  return [...new Set(ids as string[])]
}

export function parseSaveManyOptions(value: unknown): SaveManyOptions {
  const v = (value && typeof value === 'object' ? value : {}) as Record<string, unknown>
  if (typeof v['includeSettings'] !== 'boolean' || typeof v['promptFiles'] !== 'boolean') {
    throw new Error('Invalid save options.')
  }
  return { includeSettings: v['includeSettings'], promptFiles: v['promptFiles'] }
}

/** A plain file name from the engine's, or a made-up one if it is not usable as one. */
export function safeFileName(name: string, fallback: string): string {
  const base = path.basename(name.replaceAll('\\', '/'))
  return /^[\w .()-]+\.[a-z0-9]{2,5}$/i.test(base) && !base.startsWith('.') ? base : fallback
}

/** Write `data` as `name` in `folder`, or as "name (2).ext" and so on if that is taken. The path written. */
export async function writeUnique(folder: string, name: string, data: Uint8Array | string): Promise<string> {
  const ext = path.extname(name)
  const stem = name.slice(0, name.length - ext.length)
  for (let n = 1; n < 10_000; n++) {
    const target = path.join(folder, n === 1 ? name : `${stem} (${n})${ext}`)
    try {
      await fsp.writeFile(target, data, { flag: 'wx' })
      return target
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code !== 'EEXIST') throw e
    }
  }
  throw new Error(`Too many files named like ${name}.`)
}

/** What a prompt file beside a saved item says: the prompt, then the settings that made it. */
export function promptText(record: ImageRecord): string {
  const lines = [record.prompt, '']
  if (record.negative_prompt) lines.push(`Negative prompt: ${record.negative_prompt}`)
  lines.push(
    `Size: ${record.width} x ${record.height}`,
    `Seed: ${record.seed}`,
    `Steps: ${record.steps}`,
    `Guidance: ${record.guidance}`
  )
  if (record.media === 'video' && record.frames) lines.push(`Frames: ${record.frames} at ${record.fps} fps`)
  lines.push(`Model: ${record.model}`)
  return lines.join('\n') + '\n'
}
