/**
 * Just enough of the Windows PE format to build against a DLL without its
 * import library: the names it exports (and whether each is code or data), and
 * the DLLs a file imports. Headers and tables only; nothing is loaded or run.
 */

const IMAGE_SCN_MEM_EXECUTE = 0x20000000

function headers(buf) {
  if (buf.readUInt16LE(0) !== 0x5a4d) throw new Error('Not a PE file (no MZ header)')
  const pe = buf.readUInt32LE(0x3c)
  if (buf.readUInt32LE(pe) !== 0x00004550) throw new Error('Not a PE file (no PE signature)')
  const sectionCount = buf.readUInt16LE(pe + 6)
  const optionalSize = buf.readUInt16LE(pe + 20)
  const optional = pe + 24
  const magic = buf.readUInt16LE(optional)
  if (magic !== 0x20b) throw new Error('Only 64-bit (PE32+) files are supported')
  const dataDirectories = optional + 112
  const sections = []
  for (let i = 0; i < sectionCount; i++) {
    const s = optional + optionalSize + i * 40
    sections.push({
      virtualSize: buf.readUInt32LE(s + 8),
      virtualAddress: buf.readUInt32LE(s + 12),
      rawSize: buf.readUInt32LE(s + 16),
      rawPointer: buf.readUInt32LE(s + 20),
      characteristics: buf.readUInt32LE(s + 36)
    })
  }
  const section = (rva) =>
    sections.find((s) => rva >= s.virtualAddress && rva < s.virtualAddress + Math.max(s.virtualSize, s.rawSize)) ?? null
  const offset = (rva) => {
    const s = section(rva)
    if (!s) throw new Error(`RVA 0x${rva.toString(16)} is outside every section`)
    return rva - s.virtualAddress + s.rawPointer
  }
  const directory = (index) => ({
    rva: buf.readUInt32LE(dataDirectories + index * 8),
    size: buf.readUInt32LE(dataDirectories + index * 8 + 4)
  })
  return { section, offset, directory }
}

function cString(buf, at) {
  const end = buf.indexOf(0, at)
  if (end === -1) throw new Error('Unterminated string')
  return buf.toString('latin1', at, end)
}

/** Exported names, each marked data when it points outside executable code. Forwarders are skipped. */
export function readExports(buf) {
  const { section, offset, directory } = headers(buf)
  const dir = directory(0)
  if (dir.rva === 0) return []
  const e = offset(dir.rva)
  const functionCount = buf.readUInt32LE(e + 20)
  const nameCount = buf.readUInt32LE(e + 24)
  const functions = offset(buf.readUInt32LE(e + 28))
  const names = offset(buf.readUInt32LE(e + 32))
  const ordinals = offset(buf.readUInt32LE(e + 36))
  const out = []
  for (let i = 0; i < nameCount; i++) {
    const name = cString(buf, offset(buf.readUInt32LE(names + i * 4)))
    const ordinal = buf.readUInt16LE(ordinals + i * 2)
    if (ordinal >= functionCount) throw new Error(`Export ${name} has an ordinal outside the table`)
    const rva = buf.readUInt32LE(functions + ordinal * 4)
    // A forwarder's address points back into the export directory, at a "dll.name" string.
    if (rva >= dir.rva && rva < dir.rva + dir.size) continue
    const s = section(rva)
    out.push({ name, data: !s || (s.characteristics & IMAGE_SCN_MEM_EXECUTE) === 0 })
  }
  return out
}

/** DLL names a file imports, normally and delay-loaded. */
export function readImports(buf) {
  const { offset, directory } = headers(buf)
  const normal = []
  const delayed = []
  const imp = directory(1)
  if (imp.rva !== 0) {
    for (let at = offset(imp.rva); ; at += 20) {
      const nameRva = buf.readUInt32LE(at + 12)
      if (nameRva === 0) break
      normal.push(cString(buf, offset(nameRva)))
    }
  }
  const delay = directory(13)
  if (delay.rva !== 0) {
    for (let at = offset(delay.rva); ; at += 32) {
      const nameRva = buf.readUInt32LE(at + 4)
      if (nameRva === 0) break
      delayed.push(cString(buf, offset(nameRva)))
    }
  }
  return { normal, delayed }
}

/** A module-definition file for `lib /def`, so the linker can bind to the DLL's exports. */
export function moduleDefinition(dllName, exports) {
  const lines = [`LIBRARY "${dllName}"`, 'EXPORTS']
  for (const { name, data } of exports) lines.push(data ? `    ${name} DATA` : `    ${name}`)
  return lines.join('\r\n') + '\r\n'
}
