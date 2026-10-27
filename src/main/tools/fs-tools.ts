import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import { randomUUID } from 'node:crypto'
import type { DirEntry, ToolArgs, ToolResults } from '@shared/tools.js'
import type { Journal } from './journal.js'

/**
 * The only code in the application that touches the files of the user.
 *
 * Nothing here consults policy. By the time a function in this module runs, the
 * broker has already ruled on the call. Keeping the two apart means the check
 * can be tested without side effects, and the side effects can be read without
 * having to hold the policy in your head at the same time.
 */

export const MAX_READ_BYTES = 1_000_000
export const MAX_DIR_ENTRIES = 1_000

export class ToolError extends Error {
  constructor(
    message: string,
    readonly hint?: string
  ) {
    super(message)
    this.name = 'ToolError'
  }
}

export { Journal, type JournalEntry } from './journal.js'

function describeErrno(err: unknown, target: string): ToolError {
  const code = (err as NodeJS.ErrnoException).code
  switch (code) {
    case 'ENOENT':
      return new ToolError(`No such file: ${target}`)
    case 'EACCES':
    case 'EPERM':
      return new ToolError(`Permission denied by the operating system: ${target}`)
    case 'EISDIR':
      return new ToolError(`That path is a directory, not a file: ${target}`)
    case 'ENOTDIR':
      return new ToolError(`Part of that path is not a directory: ${target}`)
    case 'EBUSY':
      return new ToolError(`That file is in use by another program: ${target}`)
    default:
      return new ToolError(`${code ?? 'Error'} while accessing ${target}`)
  }
}

export async function listDirectory(
  args: ToolArgs['list_directory']
): Promise<ToolResults['list_directory']> {
  let dirents
  try {
    dirents = await fsp.readdir(args.path, { withFileTypes: true })
  } catch (err) {
    throw describeErrno(err, args.path)
  }

  const truncated = dirents.length > MAX_DIR_ENTRIES
  const slice = truncated ? dirents.slice(0, MAX_DIR_ENTRIES) : dirents

  const entries: DirEntry[] = await Promise.all(
    slice.map(async (d) => {
      const kind: DirEntry['kind'] = d.isFile()
        ? 'file'
        : d.isDirectory()
          ? 'directory'
          : 'other'
      let bytes: number | null = null
      if (kind === 'file') {
        try {
          bytes = (await fsp.stat(path.join(args.path, d.name))).size
        } catch {
          bytes = null
        }
      }
      return { name: d.name, kind, bytes }
    })
  )

  entries.sort((a, b) =>
    a.kind === b.kind ? a.name.localeCompare(b.name) : a.kind === 'directory' ? -1 : 1
  )
  return { entries, truncated }
}

export async function readFile(args: ToolArgs['read_file']): Promise<ToolResults['read_file']> {
  let stat
  try {
    stat = await fsp.stat(args.path)
  } catch (err) {
    throw describeErrno(err, args.path)
  }
  if (stat.isDirectory()) {
    throw new ToolError(`That path is a directory: ${args.path}`, 'Use list_directory instead.')
  }

  const truncated = stat.size > MAX_READ_BYTES
  const handle = await fsp.open(args.path, 'r')
  try {
    const length = truncated ? MAX_READ_BYTES : stat.size
    const buffer = Buffer.alloc(length)
    await handle.read(buffer, 0, length, 0)

    // A null byte in the first stretch means this is not text. Feeding the model
    // a megabyte of mojibake burns the context window it needs for the actual
    // task, so refuse plainly instead.
    if (buffer.subarray(0, Math.min(8192, length)).includes(0)) {
      throw new ToolError(
        `${path.basename(args.path)} looks like a binary file.`,
        'Only text files can be read.'
      )
    }
    return { text: buffer.toString('utf8'), bytes: stat.size, truncated }
  } finally {
    await handle.close()
  }
}

/**
 * Replace the contents of a file without ever leaving it half written.
 *
 * The write lands in a sibling temp file and is then renamed over the target,
 * which is atomic within a volume. A crash mid-write therefore leaves either the
 * old file or the new one, never a truncated hybrid.
 */
export async function writeFile(
  args: ToolArgs['write_file'],
  journal: Journal,
  group?: string
): Promise<ToolResults['write_file']> {
  const existed = fs.existsSync(args.path)
  const entry = await journal.record('write', args.path, group)

  const dir = path.dirname(args.path)
  try {
    await fsp.mkdir(dir, { recursive: true })
  } catch (err) {
    throw describeErrno(err, dir)
  }

  const temp = path.join(dir, `.${path.basename(args.path)}.${randomUUID()}.tmp`)
  try {
    await fsp.writeFile(temp, args.content, 'utf8')
    await fsp.rename(temp, args.path)
  } catch (err) {
    await fsp.rm(temp, { force: true })
    throw describeErrno(err, args.path)
  } finally {
    await journal.settle(entry)
  }
  return { bytes: Buffer.byteLength(args.content, 'utf8'), existed }
}

/**
 * Replace one exact run of text.
 *
 * An ambiguous match is refused rather than guessed at. A small model asked to
 * change "the second one" will cheerfully produce a snippet that matches four
 * places, and silently rewriting all four is exactly the kind of quiet damage
 * the approval prompt cannot catch, because the call looked fine when it was
 * approved.
 */
export async function editFile(
  args: ToolArgs['edit_file'],
  journal: Journal,
  group?: string
): Promise<ToolResults['edit_file']> {
  if (args.old_text.length === 0) {
    throw new ToolError('old_text is empty.', 'Use write_file to create or replace a whole file.')
  }
  if (args.old_text === args.new_text) {
    throw new ToolError('old_text and new_text are identical, so this edit would do nothing.')
  }

  let original: string
  try {
    original = await fsp.readFile(args.path, 'utf8')
  } catch (err) {
    throw describeErrno(err, args.path)
  }

  const occurrences = original.split(args.old_text).length - 1
  if (occurrences === 0) {
    throw new ToolError(
      `That text does not appear in ${path.basename(args.path)}.`,
      'Read the file again, it may not say what you expect.'
    )
  }
  if (occurrences > 1) {
    throw new ToolError(
      `That text appears ${occurrences} times in ${path.basename(args.path)}.`,
      'Include enough surrounding text to name exactly one place.'
    )
  }

  const entry = await journal.record('edit', args.path, group)
  const updated = original.replace(args.old_text, args.new_text)

  const dir = path.dirname(args.path)
  const temp = path.join(dir, `.${path.basename(args.path)}.${randomUUID()}.tmp`)
  try {
    await fsp.writeFile(temp, updated, 'utf8')
    await fsp.rename(temp, args.path)
  } catch (err) {
    await fsp.rm(temp, { force: true })
    throw describeErrno(err, args.path)
  } finally {
    await journal.settle(entry)
  }
  return { replacements: 1, bytes: Buffer.byteLength(updated, 'utf8') }
}

/**
 * Delete by moving to the recycle bin.
 *
 * `trash` is injected rather than imported so this module stays testable outside
 * Electron. The application passes shell.trashItem.
 */
export async function deleteFile(
  args: ToolArgs['delete_file'],
  journal: Journal,
  trash: (target: string) => Promise<void>,
  group?: string
): Promise<ToolResults['delete_file']> {
  if (!fs.existsSync(args.path)) {
    throw describeErrno({ code: 'ENOENT' }, args.path)
  }
  const stat = await fsp.stat(args.path)
  if (stat.isDirectory()) {
    throw new ToolError(
      `${path.basename(args.path)} is a folder.`,
      'This agent only deletes individual files.'
    )
  }
  const entry = await journal.record('delete', args.path, group)
  try {
    await trash(args.path)
  } catch (err) {
    throw new ToolError(
      `Could not move ${path.basename(args.path)} to the recycle bin: ${(err as Error).message}`
    )
  } finally {
    await journal.settle(entry)
  }
  return { recycled: true }
}
