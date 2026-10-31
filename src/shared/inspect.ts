import type { GgufDetails } from './gguf.js'

/**
 * Inspecting a downloaded model file (Settings → Inspect a model file, and the
 * chat side's Model check).
 *
 * A GGUF file is read by the studio's own strict reader (shared/gguf.ts) in the
 * main process. Anything else is handed to image-engine/scripts/inspect_model.py,
 * run with the engine's Python on this machine. Either way the file's header is
 * read and the file hashed, and none of it is loaded or run. Scanning the file
 * for malware is left to the user's own antivirus.
 */

export type FindingLevel = 'ok' | 'info' | 'warn' | 'danger'

export interface InspectFinding {
  level: FindingLevel
  title: string
  detail: string
}

export interface InspectReport {
  file: string
  folder: string
  size: number
  /** From the file's first bytes, not its name. */
  format: 'safetensors' | 'gguf' | 'zip' | 'pickle' | 'unknown'
  sha256: string
  /** unsafe: something that can run code or hide a file; warnings: worth a look; clean: nothing found. */
  verdict: 'clean' | 'warnings' | 'unsafe'
  findings: InspectFinding[]
  metadata: { key: string; chars: number; preview: string }[]
  summary: {
    tensors: number
    parameters: number
    dtypes: Record<string, number>
    metadata_keys: number
    structure: string
  } | null
  /** For a GGUF file whose header could be read: what it says about the model, and its chat template. */
  gguf?: GgufDetails
}

export type InspectPhase = 'idle' | 'header' | 'hashing' | 'done' | 'failed' | 'cancelled'

export interface ModelInspectState {
  phase: InspectPhase
  /** The file being (or last) inspected, as a name for display. */
  file: string | null
  done: number
  size: number
  message: string
  report: InspectReport | null
}

export const IDLE_INSPECT: ModelInspectState = { phase: 'idle', file: null, done: 0, size: 0, message: '', report: null }

export type InspectEvent =
  | { event: 'progress'; stage: 'header' | 'hashing'; done: number; size: number }
  | { event: 'report'; report: InspectReport }
  | { event: 'error'; message: string }

/** One line of the script's --json output, or null for anything else. */
export function parseInspectEvent(line: string): InspectEvent | null {
  let parsed: unknown
  try {
    parsed = JSON.parse(line)
  } catch {
    return null
  }
  if (!parsed || typeof parsed !== 'object') return null
  const e = parsed as Record<string, unknown>
  if (e['event'] === 'progress' && (e['stage'] === 'header' || e['stage'] === 'hashing')
      && typeof e['done'] === 'number' && typeof e['size'] === 'number') {
    return { event: 'progress', stage: e['stage'], done: e['done'], size: e['size'] }
  }
  if (e['event'] === 'report' && e['report'] && typeof e['report'] === 'object') {
    const r = e['report'] as Record<string, unknown>
    if (typeof r['sha256'] === 'string' && Array.isArray(r['findings'])) return { event: 'report', report: r as unknown as InspectReport }
  }
  if (e['event'] === 'error' && typeof e['message'] === 'string') return { event: 'error', message: e['message'] }
  return null
}

/** The state after one event. */
export function feedInspect(state: ModelInspectState, event: InspectEvent): ModelInspectState {
  if (event.event === 'progress') return { ...state, phase: event.stage, done: event.done, size: event.size }
  if (event.event === 'report') {
    return { ...state, phase: 'done', done: event.report.size, size: event.report.size, report: event.report, message: '' }
  }
  return { ...state, phase: 'failed', message: event.message }
}

/**
 * A pasted hash as 64 lowercase hex digits, or null. Spaces, line breaks and a
 * "sha256:" prefix (as some pages show it) are ignored.
 */
export function normalizeSha256(text: string): string | null {
  const cleaned = text.trim().replace(/^sha-?256\s*[:=]\s*/i, '').replace(/\s+/g, '').toLowerCase()
  return /^[0-9a-f]{64}$/.test(cleaned) ? cleaned : null
}

export type HashComparison = 'empty' | 'invalid' | 'match' | 'mismatch'

export function compareSha256(actual: string, pasted: string): HashComparison {
  if (!pasted.trim()) return 'empty'
  const expected = normalizeSha256(pasted)
  if (!expected) return 'invalid'
  return expected === actual.toLowerCase() ? 'match' : 'mismatch'
}
