import fs from 'node:fs'
import path from 'node:path'

/**
 * Turning a string the model wrote into a location on disk we are willing to
 * reason about.
 *
 * Everything the permission system claims rests on one property: that the path
 * we check is the path we later act on. On Windows that is genuinely hard,
 * because a great many different strings name the same file. Written with
 * single backslashes, the shapes that matter are:
 *
 *   C:\work\..\work\notes.txt     a lexical detour
 *   C:\work\link\notes.txt        a directory junction pointing anywhere at all
 *   C:\PROGRA~1\notes.txt         an 8.3 short name
 *   \\?\C:\work\notes.txt         the extended-length prefix
 *   C:work\notes.txt              drive-relative, depends on a per-drive cwd
 *   C:\work\notes.txt.            a trailing dot, silently stripped by the OS
 *   C:\work\notes.txt:hidden      an alternate data stream
 *
 * A containment check written against the raw string is defeated by every one
 * of these. So we refuse the forms that have no honest use, and resolve the
 * rest against the real filesystem before any policy is consulted.
 */

const WIN = process.platform === 'win32'

/** Device names Windows reserves in every directory. */
const RESERVED_SEGMENT = /^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\.|$)/i

export type ResolveResult =
  | { ok: true; path: string; existed: boolean }
  | { ok: false; kind: 'invalid-path' | 'unresolvable'; detail: string }

/** Strip the extended-length prefixes so the rest of the code sees one shape. */
function stripExtendedPrefix(input: string): string {
  if (input.startsWith('\\\\?\\UNC\\')) return '\\\\' + input.slice(8)
  if (input.startsWith('\\\\?\\')) return input.slice(4)
  if (input.startsWith('\\\\.\\')) return input.slice(4)
  return input
}

function rejectableSegment(seg: string): string | null {
  if (seg.includes(':')) return `"${seg}" contains a colon (alternate data stream)`
  if (RESERVED_SEGMENT.test(seg)) return `"${seg}" is a reserved device name`
  if (WIN && /[ .]$/.test(seg)) {
    // Windows strips these when it opens the file, so the name we checked and
    // the name it opens would differ.
    return `"${seg}" ends in a space or a dot`
  }
  if (WIN && /["*<>?|]/.test(seg)) return `"${seg}" contains a character illegal in Windows names`
  return null
}

/**
 * Resolve `input` to a single canonical absolute path, following links.
 *
 * Non-existent paths are allowed, because creating a file is a legitimate thing
 * for the agent to ask for, but only the part that does not exist may be
 * missing. Every existing ancestor is resolved for real, so a junction anywhere
 * along the way is followed to where it actually points rather than taken at
 * its word.
 */
export function canonicalize(input: unknown): ResolveResult {
  if (typeof input !== 'string' || input.length === 0) {
    return { ok: false, kind: 'invalid-path', detail: 'path must be a non-empty string' }
  }
  if (input.includes('\0')) {
    return { ok: false, kind: 'invalid-path', detail: 'path contains a null byte' }
  }

  const p = WIN ? stripExtendedPrefix(input) : input
  const lib = WIN ? path.win32 : path.posix

  if (!lib.isAbsolute(p)) {
    // Catches "C:notes.txt" and plain "notes.txt" alike. Both depend on a
    // current directory the user never sees, so neither is something we should
    // be guessing about. (A relative path from the model has already been
    // anchored to the working folder by anchorToFolder by the time it gets here.)
    return { ok: false, kind: 'invalid-path', detail: 'path must be absolute' }
  }

  const parsed = lib.parse(p)
  const rootPrefix = parsed.root
  if (WIN && /^[\\/]+$/.test(rootPrefix)) {
    // "\notes.txt" is rooted on whichever drive happens to be current.
    return { ok: false, kind: 'invalid-path', detail: 'path must name a drive or a share' }
  }

  const rest = p.slice(rootPrefix.length)
  const segments = rest.split(WIN ? /[\\/]+/ : '/')

  for (const seg of segments) {
    if (!seg || seg === '.' || seg === '..') continue
    const problem = rejectableSegment(seg)
    if (problem) return { ok: false, kind: 'invalid-path', detail: problem }
  }

  // Walk the path one segment at a time, resolving each existing level for real
  // before stepping into the next. Resolving link by link is what makes ".."
  // behave correctly: "root/junction/.." must land in the parent of the
  // junction's *target*, which is not where lexical normalization would put it.
  let current = rootPrefix
  let existed = true

  for (const seg of segments) {
    if (!seg || seg === '.') continue
    if (seg === '..') {
      current = lib.dirname(current)
      continue
    }
    const next = lib.join(current, seg)
    if (!existed) {
      current = next
      continue
    }
    try {
      current = fs.realpathSync.native(next)
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code
      if (code === 'ENOENT') {
        current = next
        existed = false
        continue
      }
      return {
        ok: false,
        kind: 'unresolvable',
        detail: `${code ?? 'error'} while resolving ${seg}`
      }
    }
  }

  return { ok: true, path: current, existed }
}

/**
 * Anchor a relative path the model wrote to the workspace's working folder.
 *
 * Only plain relative paths ("notes.txt", "src/../README.md") are anchored.
 * Everything else is passed through untouched for canonicalize to judge, which
 * matters for the two Windows shapes that look relative but are not: "C:notes"
 * depends on the current directory of drive C, and "\notes" on the current
 * drive. Joining either onto the working folder would quietly turn a path that
 * names some other place into one that names a place inside the workspace.
 * Where a joined path ends up is still decided by canonicalize and the root
 * check, so "../elsewhere" is resolved and then refused like any other path.
 */
export function anchorToFolder(workingFolder: string | null, input: unknown): unknown {
  if (workingFolder === null || typeof input !== 'string' || input.length === 0) return input
  const lib = WIN ? path.win32 : path.posix
  if (lib.isAbsolute(input)) return input
  if (WIN && /^[a-zA-Z]:/.test(input)) return input
  return lib.join(workingFolder, input)
}

/**
 * The form two paths are compared in.
 *
 * Windows filesystems are case-insensitive in practice, so a check that is not
 * would be sidestepped by changing case. Separators are unified so a root
 * stored with backslashes still matches a path built with forward ones.
 */
export function comparablePath(p: string): string {
  const unified = WIN ? p.replace(/\\/g, '/') : p
  const trimmed = unified.length > 1 ? unified.replace(/\/+$/, '') : unified
  return WIN ? trimmed.toLowerCase() : trimmed
}

/**
 * Is `child` the root itself, or somewhere beneath it?
 *
 * The separator in the prefix test is what stops "C:\work" from swallowing
 * "C:\work-secrets". A prefix comparison on the bare string would call that one
 * contained.
 */
export function isContained(child: string, root: string): boolean {
  const c = comparablePath(child)
  const r = comparablePath(root)
  if (c === r) return true
  return c.startsWith(r.endsWith('/') ? r : r + '/')
}
