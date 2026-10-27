import fsp from 'node:fs/promises'

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
