import type { ModelId } from './setup.js'

/**
 * Importing a model published as a single transformer file (Settings → Image model, or
 * Video model → Import a model file…).
 *
 * image-engine/scripts/import_model.py does the work: --probe says from the file's
 * header which model it is, then the import checks it tensor by tensor, hashes it,
 * and links it together with the parts that model needs (downloaded first if they are
 * missing) into a model folder, which the engine is then pointed at. The studio starts
 * those steps and reports what they say; nothing is loaded or run.
 */

export type ImportFamily = 'diffusion' | 'large' | 'video'

/** Which Settings card an import was started from, and which model folder it sets. */
export type ImportTarget = 'image' | 'video'

export function targetOf(family: ImportFamily): ImportTarget {
  return family === 'video' ? 'video' : 'image'
}

/** Which video model a video file is (single_file.VIDEO_KINDS): the 14B ones are two files, one per expert. */
export type VideoKind = '5b' | '14b-text' | '14b-image'

/** What --probe found in the file's header. */
export interface ImportProbe {
  file: string
  size: number
  family: ImportFamily
  /** "6B image model", "12B image model", "5B video model", "14B text-to-video model (one expert of two)"... */
  label: string
  parameters: number
  /** "fp8", or the dtype most tensors use. */
  storage: string
  layers: number
  fp8_weights: number
  /** For a video file; null otherwise. */
  video_kind: VideoKind | null
}

export function isTwoExperts(probe: ImportProbe | null): boolean {
  return probe?.video_kind === '14b-text' || probe?.video_kind === '14b-image'
}

export type ImportPhase =
  | 'idle'
  | 'probing'
  | 'ready'
  | 'parts'
  | 'checking'
  | 'hashing'
  | 'assembling'
  | 'done'
  | 'failed'
  | 'cancelled'

export interface ModelImportState {
  phase: ImportPhase
  /** The card the import was started from (null when idle). */
  target: ImportTarget | null
  /** The file read (for a 14B video model, the first expert: high noise). */
  probe: ImportProbe | null
  /** A 14B video model's second expert (low noise), once chosen. */
  second: ImportProbe | null
  /** Parts the model needs that are not downloaded yet; they are downloaded first, after the user accepts their licenses. */
  needs: ModelId[]
  /** A folder name for the import, from the file's name. */
  suggestedName: string
  /** Bytes done and total of the current step (hashing). */
  done: number
  size: number
  message: string
  /** The assembled folder, once done. */
  dest: string | null
}

export const IDLE_IMPORT: ModelImportState = {
  phase: 'idle',
  target: null,
  probe: null,
  second: null,
  needs: [],
  suggestedName: '',
  done: 0,
  size: 0,
  message: '',
  dest: null
}

export interface ImportOptions {
  /** distilled: few steps, no guidance; base: more steps, with guidance. */
  variant: 'base' | 'distilled'
  /** The publisher's SHA-256, or "" to only record the file's. */
  sha256: string
  /** The new folder's name, inside the models folder. */
  name: string
  /** The user accepted the licenses of the parts in `needs`. */
  acceptLicenses: boolean
  /** A 14B video model: the second expert's published SHA-256, or "". */
  sha256_2?: string
  /** A distilled 14B video model: the steps and schedule shift its download page gives. */
  steps?: number | null
  shift?: number | null
}

/** The limits on a distilled 14B model's settings (video_models.py). */
export const MAX_FEW_STEPS = 16
export const MAX_SHIFT = 20

/** What each model is, in the window's words, and its own defaults per variant. */
export const IMPORT_VARIANTS: Record<ImportFamily, Record<ImportOptions['variant'], string>> = {
  diffusion: { distilled: 'Distilled: 8 steps, no guidance', base: 'Base: 30 steps, guidance 4' },
  large: { distilled: 'Distilled: 8 steps, no guidance', base: 'Base: 28 steps, guidance 4.5' },
  video: { distilled: 'Distilled: 3 fixed steps, no guidance', base: 'Original: 50 steps, guidance 5' }
}

const TWO_EXPERT_VARIANTS: Record<ImportOptions['variant'], string> = {
  distilled: 'Distilled: the few steps and shift its page gives, no guidance',
  base: 'Original: 40 steps, with guidance'
}

export function variantLabels(probe: ImportProbe): Record<ImportOptions['variant'], string> {
  return isTwoExperts(probe) ? TWO_EXPERT_VARIANTS : IMPORT_VARIANTS[probe.family]
}

/** Checks a distilled 14B model's settings; null when fine, otherwise what is wrong. */
export function fewStepProblem(steps: number | null | undefined, shift: number | null | undefined): string | null {
  if (steps == null || !Number.isInteger(steps) || steps < 1 || steps > MAX_FEW_STEPS) {
    return `Steps must be a whole number from 1 to ${MAX_FEW_STEPS}.`
  }
  if (shift == null || !Number.isFinite(shift) || shift <= 0 || shift > MAX_SHIFT) return `Shift must be above 0 and at most ${MAX_SHIFT}.`
  return null
}

export type ImportEvent =
  | { event: 'probe'; probe: ImportProbe }
  | { event: 'progress'; stage: 'checking' | 'hashing' | 'assembling'; done: number; size: number }
  | { event: 'done'; dest: string }
  | { event: 'error'; message: string }

/** One line of import_model.py --json, or null for anything else. */
export function parseImportEvent(line: string): ImportEvent | null {
  let parsed: unknown
  try {
    parsed = JSON.parse(line)
  } catch {
    return null
  }
  if (!parsed || typeof parsed !== 'object') return null
  const e = parsed as Record<string, unknown>
  if (e['event'] === 'probe' && (e['family'] === 'diffusion' || e['family'] === 'large' || e['family'] === 'video')
      && typeof e['file'] === 'string') {
    const num = (k: string): number => (typeof e[k] === 'number' ? (e[k] as number) : 0)
    return {
      event: 'probe',
      probe: {
        file: e['file'],
        size: num('size'),
        family: e['family'],
        label: typeof e['label'] === 'string' ? e['label'] : '',
        parameters: num('parameters'),
        storage: typeof e['storage'] === 'string' ? e['storage'] : '?',
        layers: num('layers'),
        fp8_weights: num('fp8_weights'),
        video_kind: e['video_kind'] === '5b' || e['video_kind'] === '14b-text' || e['video_kind'] === '14b-image' ? e['video_kind'] : null
      }
    }
  }
  if (e['event'] === 'progress' && (e['stage'] === 'checking' || e['stage'] === 'hashing' || e['stage'] === 'assembling')
      && typeof e['done'] === 'number' && typeof e['size'] === 'number') {
    return { event: 'progress', stage: e['stage'], done: e['done'], size: e['size'] }
  }
  if (e['event'] === 'done' && typeof e['dest'] === 'string') return { event: 'done', dest: e['dest'] }
  if (e['event'] === 'error' && typeof e['message'] === 'string') return { event: 'error', message: e['message'] }
  return null
}

/** Folder names the import may create: plain, short, and not one of the downloads' own folders. */
const NAME = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/

export function isValidFolderName(name: string, reserved: readonly string[] = []): boolean {
  return NAME.test(name) && !name.endsWith('.') && !reserved.includes(name.toLowerCase())
}

/** A folder name from a file name: its stem, reduced to the characters a folder name may use. */
export function folderNameFor(file: string): string {
  const stem = file.replace(/\.safetensors$/i, '')
  const cleaned = stem.replace(/[^A-Za-z0-9._-]+/g, '-').replace(/^[^A-Za-z0-9]+/, '').replace(/[.-]+$/, '').slice(0, 64)
  return cleaned || 'imported-model'
}
