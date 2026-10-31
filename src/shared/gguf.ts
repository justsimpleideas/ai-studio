import type { InspectFinding } from './inspect.js'

/**
 * A strict reader for the GGUF container, independent of llama.cpp.
 *
 * A GGUF file holds numbers (the weights) and a header of named values (the
 * metadata, including the chat template). It cannot carry code the way a pickle
 * checkpoint can. What it can carry is a malformed header: counts, lengths and
 * offsets chosen to trip the loader that trusts them. This reader trusts
 * nothing. Every count has a ceiling, every length is checked against the file,
 * every tensor must fit, align and not overlap another, and anything the loader
 * would not understand is a reason to refuse rather than to guess. It runs
 * before llama.cpp ever opens the file (the load path) and again, with the hash,
 * for the report the user reads (the inspect path).
 *
 * Pure: it reads through a `ByteSource`, so tests hand it bytes in memory.
 */

export interface ByteSource {
  readonly size: number
  /** Up to `length` bytes at `offset`; fewer only at the end of the file. */
  read(offset: number, length: number): Promise<Uint8Array>
}

/** Ceilings well above any real model and well below anything that could exhaust memory. */
export const GGUF_LIMITS = {
  tensors: 200_000,
  metadataEntries: 65_536,
  stringBytes: 64 * 1024 * 1024,
  arrayLength: 16_000_000,
  headerBytes: 512 * 1024 * 1024,
  tensorNameBytes: 64, // GGML_MAX_NAME
  dims: 4, // GGML_MAX_DIMS
  alignmentMax: 1024 * 1024
}

/** GGUF metadata value types (the container's own numbering). */
const enum ValueType {
  Uint8 = 0,
  Int8 = 1,
  Uint16 = 2,
  Int16 = 3,
  Uint32 = 4,
  Int32 = 5,
  Float32 = 6,
  Bool = 7,
  String = 8,
  Array = 9,
  Uint64 = 10,
  Int64 = 11,
  Float64 = 12
}

const SCALAR_BYTES: Record<number, number> = {
  [ValueType.Uint8]: 1,
  [ValueType.Int8]: 1,
  [ValueType.Uint16]: 2,
  [ValueType.Int16]: 2,
  [ValueType.Uint32]: 4,
  [ValueType.Int32]: 4,
  [ValueType.Float32]: 4,
  [ValueType.Bool]: 1,
  [ValueType.Uint64]: 8,
  [ValueType.Int64]: 8,
  [ValueType.Float64]: 8
}

/**
 * ggml tensor types: name, elements per block, bytes per block. From ggml's
 * type traits; a type not listed here is one this reader cannot size, which is
 * reported rather than assumed.
 */
export const GGML_TYPES: Record<number, { name: string; block: number; bytes: number }> = {
  0: { name: 'F32', block: 1, bytes: 4 },
  1: { name: 'F16', block: 1, bytes: 2 },
  2: { name: 'Q4_0', block: 32, bytes: 18 },
  3: { name: 'Q4_1', block: 32, bytes: 20 },
  6: { name: 'Q5_0', block: 32, bytes: 22 },
  7: { name: 'Q5_1', block: 32, bytes: 24 },
  8: { name: 'Q8_0', block: 32, bytes: 34 },
  9: { name: 'Q8_1', block: 32, bytes: 36 },
  10: { name: 'Q2_K', block: 256, bytes: 84 },
  11: { name: 'Q3_K', block: 256, bytes: 110 },
  12: { name: 'Q4_K', block: 256, bytes: 144 },
  13: { name: 'Q5_K', block: 256, bytes: 176 },
  14: { name: 'Q6_K', block: 256, bytes: 210 },
  15: { name: 'Q8_K', block: 256, bytes: 292 },
  16: { name: 'IQ2_XXS', block: 256, bytes: 66 },
  17: { name: 'IQ2_XS', block: 256, bytes: 74 },
  18: { name: 'IQ3_XXS', block: 256, bytes: 98 },
  19: { name: 'IQ1_S', block: 256, bytes: 50 },
  20: { name: 'IQ4_NL', block: 32, bytes: 18 },
  21: { name: 'IQ3_S', block: 256, bytes: 110 },
  22: { name: 'IQ2_S', block: 256, bytes: 82 },
  23: { name: 'IQ4_XS', block: 256, bytes: 136 },
  24: { name: 'I8', block: 1, bytes: 1 },
  25: { name: 'I16', block: 1, bytes: 2 },
  26: { name: 'I32', block: 1, bytes: 4 },
  27: { name: 'I64', block: 1, bytes: 8 },
  28: { name: 'F64', block: 1, bytes: 8 },
  29: { name: 'IQ1_M', block: 256, bytes: 56 },
  30: { name: 'BF16', block: 1, bytes: 2 },
  34: { name: 'TQ1_0', block: 256, bytes: 54 },
  35: { name: 'TQ2_0', block: 256, bytes: 66 },
  39: { name: 'MXFP4', block: 32, bytes: 17 }
}

/** `general.file_type` values, as llama.cpp names them. */
const FILE_TYPES: Record<number, string> = {
  0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 7: 'Q8_0', 8: 'Q5_0', 9: 'Q5_1', 10: 'Q2_K', 11: 'Q3_K_S',
  12: 'Q3_K_M', 13: 'Q3_K_L', 14: 'Q4_K_S', 15: 'Q4_K_M', 16: 'Q5_K_S', 17: 'Q5_K_M', 18: 'Q6_K',
  19: 'IQ2_XXS', 20: 'IQ2_XS', 21: 'Q2_K_S', 22: 'IQ3_XS', 23: 'IQ3_XXS', 24: 'IQ1_S', 25: 'IQ4_NL',
  26: 'IQ3_S', 27: 'IQ3_M', 28: 'IQ2_S', 29: 'IQ2_M', 30: 'IQ4_XS', 31: 'IQ1_M', 32: 'BF16'
}

export type MetadataValue =
  | { kind: 'number'; value: number }
  | { kind: 'bool'; value: boolean }
  | { kind: 'string'; value: string }
  /** Arrays are summarised: their length, and the first few elements when they are strings or numbers. */
  | { kind: 'array'; length: number; sample: (string | number | boolean)[] }

export interface GgufTensor {
  name: string
  dims: number[]
  type: number
  typeName: string
  offset: number
  /** Bytes the tensor occupies; null when the type is not one this reader can size. */
  bytes: number | null
}

export interface GgufHeader {
  version: number
  alignment: number
  /** Where the tensor data begins (the header, padded to the alignment). */
  dataOffset: number
  metadata: Map<string, MetadataValue>
  tensors: GgufTensor[]
}

export class GgufError extends Error {}

/** Reads the header through a ByteSource in chunks, so a multi-megabyte token table is not read byte by byte. */
class Cursor {
  private chunk: Uint8Array = new Uint8Array(0)
  private chunkStart = 0
  pos = 0

  constructor(
    private readonly source: ByteSource,
    private readonly chunkSize = 1024 * 1024
  ) {}

  async bytes(length: number): Promise<Uint8Array> {
    if (length < 0) throw new GgufError('negative length')
    if (this.pos + length > this.source.size) {
      throw new GgufError(`the header runs past the end of the file (needs ${this.pos + length} bytes, the file has ${this.source.size})`)
    }
    if (this.pos + length > GGUF_LIMITS.headerBytes) throw new GgufError(`the header is larger than ${GGUF_LIMITS.headerBytes} bytes`)
    const within = this.pos - this.chunkStart
    if (within >= 0 && within + length <= this.chunk.length) {
      const out = this.chunk.subarray(within, within + length)
      this.pos += length
      return out
    }
    if (length > this.chunkSize) {
      const out = await this.source.read(this.pos, length)
      if (out.length < length) throw new GgufError('the file ended inside the header')
      this.pos += length
      return out
    }
    this.chunk = await this.source.read(this.pos, Math.min(this.chunkSize, this.source.size - this.pos))
    this.chunkStart = this.pos
    if (this.chunk.length < length) throw new GgufError('the file ended inside the header')
    const out = this.chunk.subarray(0, length)
    this.pos += length
    return out
  }

  async u32(): Promise<number> {
    const b = await this.bytes(4)
    return new DataView(b.buffer, b.byteOffset, 4).getUint32(0, true)
  }

  async u64(): Promise<number> {
    const b = await this.bytes(8)
    const v = new DataView(b.buffer, b.byteOffset, 8).getBigUint64(0, true)
    if (v > BigInt(Number.MAX_SAFE_INTEGER)) throw new GgufError(`a 64-bit value in the header (${v}) is beyond anything a file could hold`)
    return Number(v)
  }

  async scalar(type: number): Promise<number | boolean> {
    const size = SCALAR_BYTES[type]
    if (size === undefined) throw new GgufError(`unknown metadata value type ${type}`)
    const b = await this.bytes(size)
    const v = new DataView(b.buffer, b.byteOffset, size)
    switch (type) {
      case ValueType.Uint8: return v.getUint8(0)
      case ValueType.Int8: return v.getInt8(0)
      case ValueType.Uint16: return v.getUint16(0, true)
      case ValueType.Int16: return v.getInt16(0, true)
      case ValueType.Uint32: return v.getUint32(0, true)
      case ValueType.Int32: return v.getInt32(0, true)
      case ValueType.Float32: return v.getFloat32(0, true)
      case ValueType.Bool: return v.getUint8(0) !== 0
      case ValueType.Uint64: return Number(v.getBigUint64(0, true))
      case ValueType.Int64: return Number(v.getBigInt64(0, true))
      case ValueType.Float64: return v.getFloat64(0, true)
      default: throw new GgufError(`unknown metadata value type ${type}`)
    }
  }

  async string(what: string): Promise<string> {
    const length = await this.u64()
    if (length > GGUF_LIMITS.stringBytes) throw new GgufError(`${what} claims to be ${length} bytes long`)
    return new TextDecoder('utf-8', { fatal: false }).decode(await this.bytes(length))
  }
}

const SAMPLE = 8

/**
 * Read and check the header. Throws GgufError at the first thing that would
 * make the file unsafe to hand to a loader; structural doubts that are not
 * fatal (gaps, trailing bytes, unsized types) are returned as findings.
 */
export async function readGgufHeader(source: ByteSource): Promise<{ header: GgufHeader; findings: InspectFinding[] }> {
  const findings: InspectFinding[] = []
  const c = new Cursor(source)
  const magic = await c.bytes(4)
  if (String.fromCharCode(...magic) !== 'GGUF') throw new GgufError('not a GGUF file: the first bytes are not "GGUF"')
  const version = await c.u32()
  if (version !== 2 && version !== 3) throw new GgufError(`GGUF version ${version}; only versions 2 and 3 can be loaded`)
  const tensorCount = await c.u64()
  const kvCount = await c.u64()
  if (tensorCount > GGUF_LIMITS.tensors) throw new GgufError(`the header claims ${tensorCount} tensors (limit ${GGUF_LIMITS.tensors})`)
  if (kvCount > GGUF_LIMITS.metadataEntries) throw new GgufError(`the header claims ${kvCount} metadata entries (limit ${GGUF_LIMITS.metadataEntries})`)

  const metadata = new Map<string, MetadataValue>()
  for (let i = 0; i < kvCount; i++) {
    const key = await c.string(`metadata key ${i}`)
    if (!key || key.length > 1024) throw new GgufError(`metadata key ${i} is empty or absurdly long`)
    if (metadata.has(key)) throw new GgufError(`metadata key "${key}" appears twice`)
    const type = await c.u32()
    if (type === ValueType.String) {
      metadata.set(key, { kind: 'string', value: await c.string(`metadata "${key}"`) })
    } else if (type === ValueType.Array) {
      const elementType = await c.u32()
      const length = await c.u64()
      if (length > GGUF_LIMITS.arrayLength) throw new GgufError(`metadata "${key}" is an array of ${length} elements (limit ${GGUF_LIMITS.arrayLength})`)
      if (elementType === ValueType.Array) throw new GgufError(`metadata "${key}" is an array of arrays, which no loader reads`)
      const sample: (string | number | boolean)[] = []
      if (elementType === ValueType.String) {
        for (let j = 0; j < length; j++) {
          const s = await c.string(`metadata "${key}"[${j}]`)
          if (j < SAMPLE) sample.push(s)
        }
      } else {
        const size = SCALAR_BYTES[elementType]
        if (size === undefined) throw new GgufError(`metadata "${key}" has elements of unknown type ${elementType}`)
        for (let j = 0; j < Math.min(length, SAMPLE); j++) sample.push(await c.scalar(elementType))
        if (length > SAMPLE) await c.bytes((length - SAMPLE) * size)
      }
      metadata.set(key, { kind: 'array', length, sample })
    } else {
      const value = await c.scalar(type)
      metadata.set(key, typeof value === 'boolean' ? { kind: 'bool', value } : { kind: 'number', value })
    }
  }

  const alignmentValue = metadata.get('general.alignment')
  const alignment = alignmentValue?.kind === 'number' ? alignmentValue.value : 32
  if (!Number.isInteger(alignment) || alignment < 8 || alignment > GGUF_LIMITS.alignmentMax || (alignment & (alignment - 1)) !== 0) {
    throw new GgufError(`general.alignment is ${alignment}; it must be a power of two from 8 to ${GGUF_LIMITS.alignmentMax}`)
  }

  const tensors: GgufTensor[] = []
  const names = new Set<string>()
  for (let i = 0; i < tensorCount; i++) {
    const name = await c.string(`tensor ${i}'s name`)
    const nameBytes = new TextEncoder().encode(name).length
    if (nameBytes === 0 || nameBytes >= GGUF_LIMITS.tensorNameBytes) throw new GgufError(`tensor ${i}'s name is empty or ${nameBytes} bytes long (limit ${GGUF_LIMITS.tensorNameBytes - 1})`)
    if (/[\u0000-\u001f\u007f]/.test(name)) throw new GgufError(`tensor ${i}'s name contains control characters`)
    if (names.has(name)) throw new GgufError(`tensor name "${name}" appears twice`)
    names.add(name)
    const nDims = await c.u32()
    if (nDims < 1 || nDims > GGUF_LIMITS.dims) throw new GgufError(`tensor "${name}" has ${nDims} dimensions (1 to ${GGUF_LIMITS.dims})`)
    const dims: number[] = []
    let elements = 1
    for (let d = 0; d < nDims; d++) {
      const dim = await c.u64()
      if (dim < 1) throw new GgufError(`tensor "${name}" has a dimension of ${dim}`)
      elements *= dim
      if (elements > Number.MAX_SAFE_INTEGER) throw new GgufError(`tensor "${name}" has more elements than a file could hold`)
      dims.push(dim)
    }
    const type = await c.u32()
    const offset = await c.u64()
    const info = GGML_TYPES[type]
    let bytes: number | null = null
    if (info) {
      if ((dims[0] as number) % info.block !== 0) throw new GgufError(`tensor "${name}" (${info.name}) has a first dimension of ${dims[0]}, not a multiple of its block size ${info.block}`)
      bytes = (elements / info.block) * info.bytes
    }
    tensors.push({ name, dims, type, typeName: info?.name ?? `type ${type}`, offset, bytes })
  }

  const dataOffset = Math.ceil(c.pos / alignment) * alignment
  if (dataOffset > source.size) throw new GgufError('the header fills the whole file; there is no room for tensor data')
  const dataSize = source.size - dataOffset

  const unsized = tensors.filter((t) => t.bytes === null)
  if (unsized.length > 0) {
    throw new GgufError(`${unsized.length} tensor${unsized.length > 1 ? 's use' : ' uses'} a weight type this studio does not know (${[...new Set(unsized.map((t) => t.typeName))].join(', ')})`)
  }
  const ordered = [...tensors].sort((a, b) => a.offset - b.offset)
  let expected = 0
  let gaps = 0
  for (const t of ordered) {
    if (t.offset % alignment !== 0) throw new GgufError(`tensor "${t.name}" starts at ${t.offset}, which is not aligned to ${alignment}`)
    if (t.offset < expected) throw new GgufError(`tensor "${t.name}" overlaps the tensor before it`)
    if (t.offset + (t.bytes as number) > dataSize) throw new GgufError(`tensor "${t.name}" runs past the end of the file`)
    if (t.offset > expected) gaps += 1
    expected = Math.ceil((t.offset + (t.bytes as number)) / alignment) * alignment
  }
  const trailing = dataSize - Math.min(dataSize, expected)
  if (gaps > 0) findings.push({ level: 'warn', title: `${gaps} unexplained gap${gaps > 1 ? 's' : ''} between tensors`, detail: 'Bytes the model never reads. A plain conversion leaves none.' })
  if (trailing >= alignment) {
    findings.push({ level: 'warn', title: 'Data after the last tensor', detail: `${trailing.toLocaleString()} bytes the model never reads. A plain conversion leaves none.` })
  }
  findings.unshift({
    level: 'ok',
    title: 'The header is well-formed',
    detail: `GGUF v${version}; ${tensors.length.toLocaleString()} tensors of known types, all inside the file, aligned to ${alignment}, none overlapping; ${metadata.size} metadata entries within limits.`
  })
  return { header: { version, alignment, dataOffset, metadata, tensors }, findings }
}

// ---------------------------------------------------------------- what the header says

export interface TemplateFlag {
  level: 'info' | 'warn' | 'danger'
  title: string
  detail: string
}

/** The words around a match, on one line, so a finding can quote what it found. */
function excerpt(text: string, index: number, length: number, around = 70): string {
  const start = Math.max(0, index - around)
  const end = Math.min(text.length, index + length + around)
  const body = text.slice(start, end).replace(/\s+/g, ' ').trim()
  return `${start > 0 ? '...' : ''}${body}${end < text.length ? '...' : ''}`
}

/** Characters that are invisible in an editor but reach the model. */
const INVISIBLE = /[\u200b-\u200f\u2028-\u202e\u2060-\u2064\ufeff\u00ad]/

/**
 * Things worth a look in a chat template. A template is text the model reads
 * on every turn before the user's words; one that tells the model to hide
 * something, to reach for a tool, or that carries text the user cannot see,
 * is a template to read before trusting. None of this proves intent; all of it
 * is shown so the user can judge.
 */
export function templateFlags(template: string): TemplateFlag[] {
  const flags: TemplateFlag[] = []
  const lower = template.toLowerCase()
  if (template.length > 50_000) flags.push({ level: 'warn', title: 'Unusually long', detail: `${template.length.toLocaleString()} characters; most templates are a few thousand.` })
  if (INVISIBLE.test(template)) flags.push({ level: 'danger', title: 'Contains invisible characters', detail: 'Zero-width or direction-control characters: text the model reads that an editor does not show.' })
  if (/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(template)) flags.push({ level: 'warn', title: 'Contains control characters', detail: 'Text meant for a model has no reason to.' })
  // Overriding instructions or hiding from the user outright: no formatting template needs either.
  const override = /ignore (all |any )?(previous|prior|above|earlier) instructions|disregard (all |any )?(previous|prior|above|earlier) instructions|keep (this|it) (a )?(secret|hidden) from|without (the user'?s? )?(knowledge|noticing)/.exec(lower)
  if (override) {
    flags.push({
      level: 'danger',
      title: 'Tells the model to override its instructions or to act behind the user',
      detail: `"${excerpt(template, override.index, override[0].length)}"`
    })
  }
  // "Do not tell the user …" depends entirely on what follows. Templates that
  // support tools commonly tell the model not to talk about the call format
  // ("do not tell the user about function calls"); that is formatting, and is
  // shown as such. Anything else is quoted for the user to read and judge.
  const quiet = /(do not|don't|never)\s+(tell|reveal|mention|disclose|inform)\s+(the\s+)?user(s)?\b/g
  let quietMatch: RegExpExecArray | null
  let quietFormat = 0
  const quietOther: string[] = []
  while ((quietMatch = quiet.exec(lower)) !== null) {
    const after = lower.slice(quietMatch.index + quietMatch[0].length, quietMatch.index + quietMatch[0].length + 40)
    if (/^\s*(about|of)\s+(the\s+|any\s+)?(function|tool)s?(\s|-)?(call|calling|use|usage|format)?/.test(after)) quietFormat += 1
    else quietOther.push(excerpt(template, quietMatch.index, quietMatch[0].length))
  }
  if (quietOther.length > 0) {
    flags.push({
      level: 'warn',
      title: 'Tells the model not to tell the user something',
      detail: `Read the sentence and judge: in a template this is usually about formatting, but it can also be an instruction to hide something.\n${quietOther
        .slice(0, 3)
        .map((q) => `"${q}"`)
        .join('\n')}`
    })
  }
  if (quietFormat > 0) {
    flags.push({
      level: 'info',
      title: 'Tells the model not to talk about its tool-call format',
      detail: 'Usual in templates that support tools ("do not tell the user about function calls"): it keeps the call syntax out of replies.'
    })
  }
  const urls = template.match(/https?:\/\/[^\s"'<>{}]+/g) ?? []
  if (urls.length > 0) flags.push({ level: 'warn', title: `Mentions ${urls.length} web address${urls.length > 1 ? 'es' : ''}`, detail: urls.slice(0, 3).join('\n') })
  const ourTools = ['read_file', 'write_file', 'edit_file', 'delete_file', 'list_directory', 'web_search', 'fetch_page', 'download_file'].filter((t) => lower.includes(t))
  if (ourTools.length > 0) flags.push({ level: 'danger', title: "Names this studio's own tools", detail: `${ourTools.join(', ')}: a template from a model's publisher would not know them.` })
  if (/__class__|__globals__|__import__|__subclasses__|\bsubprocess\b|\bos\.(system|popen)|\beval\(|\bexec\(/.test(template)) {
    flags.push({ level: 'danger', title: 'Contains code-like escape attempts', detail: 'Harmless in this studio, which renders templates without any such access, but a sign of a file written to attack a loader.' })
  }
  const blob = template.match(/[A-Za-z0-9+/]{200,}={0,2}/)
  if (blob) flags.push({ level: 'warn', title: 'Holds a long encoded blob', detail: `${blob[0].length.toLocaleString()} characters of base64-like text.` })
  return flags
}

export interface GgufDetails {
  version: number
  architecture: string | null
  name: string | null
  fileType: string | null
  quantizedBy: string | null
  license: string | null
  sourceUrl: string | null
  contextLength: number | null
  layers: number | null
  experts: { total: number; active: number } | null
  tokens: number | null
  /** The chat template carried by the file; null when the file has none. */
  template: { text: string; chars: number; flags: TemplateFlag[] } | null
}

function str(h: GgufHeader, key: string): string | null {
  const v = h.metadata.get(key)
  return v?.kind === 'string' && v.value.trim() ? v.value : null
}

function num(h: GgufHeader, key: string): number | null {
  const v = h.metadata.get(key)
  return v?.kind === 'number' ? v.value : null
}

export function describeGguf(h: GgufHeader): GgufDetails {
  const arch = str(h, 'general.architecture')
  const fileType = num(h, 'general.file_type')
  const template = str(h, 'tokenizer.chat_template')
  const tokens = h.metadata.get('tokenizer.ggml.tokens')
  const total = arch ? num(h, `${arch}.expert_count`) : null
  const active = arch ? num(h, `${arch}.expert_used_count`) : null
  return {
    version: h.version,
    architecture: arch,
    name: str(h, 'general.name'),
    fileType: fileType === null ? null : (FILE_TYPES[fileType] ?? `type ${fileType}`),
    quantizedBy: str(h, 'general.quantized_by'),
    license: str(h, 'general.license'),
    sourceUrl: str(h, 'general.source.url') ?? str(h, 'general.url') ?? str(h, 'general.source.huggingface.repository'),
    contextLength: arch ? num(h, `${arch}.context_length`) : null,
    layers: arch ? num(h, `${arch}.block_count`) : null,
    experts: total !== null && total > 1 ? { total, active: active ?? 1 } : null,
    tokens: tokens?.kind === 'array' ? tokens.length : null,
    template: template ? { text: template, chars: template.length, flags: templateFlags(template) } : null
  }
}

/** The metadata as the report lists it: every key, how long its value is, and the start of it. */
export function metadataRows(h: GgufHeader): { key: string; chars: number; preview: string }[] {
  const rows: { key: string; chars: number; preview: string }[] = []
  for (const [key, v] of h.metadata) {
    let text: string
    if (v.kind === 'string') text = v.value
    else if (v.kind === 'array') text = `[${v.length} × ${v.sample.map((s) => JSON.stringify(s)).join(', ')}${v.length > v.sample.length ? ', …' : ''}]`
    else text = String(v.value)
    rows.push({ key, chars: text.length, preview: text.length > 120 ? `${text.slice(0, 117)}…` : text })
  }
  return rows
}

/** Findings about what the metadata says, beyond its shape. */
export function metadataFindings(h: GgufHeader, d: GgufDetails): InspectFinding[] {
  const out: InspectFinding[] = []
  if (!d.architecture) out.push({ level: 'danger', title: 'No architecture named', detail: 'general.architecture is missing; no loader could run this file.' })
  if (d.tokens === null) out.push({ level: 'warn', title: 'No tokenizer vocabulary', detail: 'tokenizer.ggml.tokens is missing, so the file cannot be used for chat on its own.' })
  if (d.template) {
    for (const f of d.template.flags) out.push({ level: f.level, title: `Chat template: ${f.title}`, detail: f.detail })
    if (!d.template.flags.some((f) => f.level !== 'info')) out.push({ level: 'ok', title: 'Chat template reads like a formatting template', detail: `${d.template.chars.toLocaleString()} characters; no hidden text, web addresses, tool names or code-like escapes.` })
  } else {
    out.push({ level: 'info', title: 'No chat template in the file', detail: 'The studio formats conversations with a built-in template for the model family, if it knows one.' })
  }
  let odd = 0
  for (const [key, v] of h.metadata) {
    if (v.kind !== 'string' || key === 'tokenizer.chat_template') continue
    if (INVISIBLE.test(v.value) || /[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(v.value)) odd += 1
  }
  if (odd > 0) out.push({ level: 'warn', title: `${odd} metadata value${odd > 1 ? 's' : ''} contain hidden or control characters`, detail: 'Descriptive text has no reason to.' })
  const pieces = [d.name, d.quantizedBy ? `quantized by ${d.quantizedBy}` : null, d.license ? `license ${d.license}` : null, d.sourceUrl ? `source ${d.sourceUrl}` : null].filter(Boolean)
  out.push({ level: 'info', title: 'What the file says about itself', detail: pieces.length ? pieces.join(' · ') : 'Nothing: no name, publisher, license or source recorded.' })
  return out
}

/** The report's verdict from its findings. */
export function verdictOf(findings: InspectFinding[]): 'clean' | 'warnings' | 'unsafe' {
  if (findings.some((f) => f.level === 'danger')) return 'unsafe'
  if (findings.some((f) => f.level === 'warn')) return 'warnings'
  return 'clean'
}
