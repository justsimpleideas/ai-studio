import { createHash } from 'node:crypto'
import fsp from 'node:fs/promises'
import path from 'node:path'
import { describeGguf, GgufError, metadataFindings, metadataRows, readGgufHeader, verdictOf, type ByteSource } from '@shared/gguf.js'
import type { InspectFinding, InspectReport } from '@shared/inspect.js'

const MAGIC = Buffer.from('GGUF', 'ascii')

/**
 * Whether a file is a GGUF model, judged by its first four bytes, not its name.
 *
 * Some model managers store models as plain GGUF files named only by their hash
 * (`blobs/sha256-…`, no extension), so the extension cannot be relied on. Checking
 * the magic also turns "picked the wrong file" into a clear message here instead of
 * an obscure failure inside llama.cpp.
 */
export async function isGgufFile(filePath: string): Promise<boolean> {
  let handle: fsp.FileHandle | null = null
  try {
    handle = await fsp.open(filePath, 'r')
    const head = Buffer.alloc(4)
    const { bytesRead } = await handle.read(head, 0, 4, 0)
    return bytesRead === 4 && head.equals(MAGIC)
  } catch {
    return false
  } finally {
    await handle?.close()
  }
}

/** A file on disk, read at offsets; the strict reader (shared/gguf.ts) pulls the header through this. */
class FileSource implements ByteSource {
  constructor(
    private readonly handle: fsp.FileHandle,
    readonly size: number
  ) {}

  async read(offset: number, length: number): Promise<Uint8Array> {
    const buffer = Buffer.alloc(length)
    const { bytesRead } = await this.handle.read(buffer, 0, length, offset)
    return buffer.subarray(0, bytesRead)
  }
}

export type GgufCheck = { ok: true } | { ok: false; reason: string }

/**
 * The gate a model file passes before llama.cpp sees it: the whole header read
 * and checked by the studio's own strict reader. A malformed header is refused
 * here, with a reason, instead of reaching a native loader that trusts it.
 */
export async function checkGgufFile(filePath: string): Promise<GgufCheck> {
  let handle: fsp.FileHandle | null = null
  try {
    handle = await fsp.open(filePath, 'r')
    const { size } = await handle.stat()
    await readGgufHeader(new FileSource(handle, size))
    return { ok: true }
  } catch (err) {
    if (err instanceof GgufError) return { ok: false, reason: `This file was not loaded: ${err.message}.` }
    return { ok: false, reason: `This file could not be read: ${(err as Error).message}` }
  } finally {
    await handle?.close()
  }
}

export interface InspectGgufOptions {
  onProgress?: (stage: 'header' | 'hashing', done: number, size: number) => void
  signal?: AbortSignal
}

/**
 * The full report for a GGUF file: the strict header check, what the metadata
 * says (including the chat template and anything odd in it), and the SHA-256 of
 * the whole file. Nothing is loaded or run.
 */
export async function inspectGgufFile(filePath: string, options: InspectGgufOptions = {}): Promise<InspectReport> {
  const handle = await fsp.open(filePath, 'r')
  try {
    const { size } = await handle.stat()
    options.onProgress?.('header', 0, size)
    const findings: InspectFinding[] = []
    let summary: InspectReport['summary'] = null
    let gguf: InspectReport['gguf'] = undefined
    let metadata: InspectReport['metadata'] = []
    try {
      const { header, findings: structural } = await readGgufHeader(new FileSource(handle, size))
      const details = describeGguf(header)
      findings.push(...structural, ...metadataFindings(header, details))
      metadata = metadataRows(header)
      const dtypes: Record<string, number> = {}
      let parameters = 0
      for (const t of header.tensors) {
        dtypes[t.typeName] = (dtypes[t.typeName] ?? 0) + 1
        parameters += t.dims.reduce((a, b) => a * b, 1)
      }
      summary = {
        tensors: header.tensors.length,
        parameters,
        dtypes,
        metadata_keys: header.metadata.size,
        structure: details.architecture ? `${details.architecture}${details.layers ? `, ${details.layers} layers` : ''}` : 'unknown'
      }
      gguf = details
    } catch (err) {
      if (!(err instanceof GgufError)) throw err
      findings.push({ level: 'danger', title: 'The header is not well-formed', detail: `${err.message}. The studio will not load this file.` })
    }

    const hash = createHash('sha256')
    const chunk = Buffer.alloc(8 * 1024 * 1024)
    let done = 0
    let lastReport = 0
    while (done < size) {
      if (options.signal?.aborted) throw new Error('Stopped.')
      const { bytesRead } = await handle.read(chunk, 0, chunk.length, done)
      if (bytesRead === 0) break
      hash.update(chunk.subarray(0, bytesRead))
      done += bytesRead
      const now = Date.now()
      if (now - lastReport > 200 || done === size) {
        lastReport = now
        options.onProgress?.('hashing', done, size)
      }
    }
    return {
      file: path.basename(filePath),
      folder: path.dirname(filePath),
      size,
      format: 'gguf',
      sha256: hash.digest('hex'),
      verdict: verdictOf(findings),
      findings,
      metadata,
      summary,
      ...(gguf ? { gguf } : {})
    }
  } finally {
    await handle.close()
  }
}
