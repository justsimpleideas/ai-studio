/**
 * A tiny mixture-of-experts model with made-up weights, written as a GGUF file.
 *
 * For checking the addon build without a real model: two llama-architecture
 * layers of four experts each, a byte-level vocabulary, about 1 MB. Its output
 * is noise, but deterministic noise, which is what a comparison between two
 * builds of the addon needs. The weights come from a fixed-seed generator, so
 * every run writes the same file.
 */
import fs from 'node:fs'

const F32 = 0

const T_UINT32 = 4
const T_FLOAT32 = 6
const T_STRING = 8
const T_ARRAY = 9
const T_INT32 = 5

export const TINY_MOE = {
  layers: 2,
  embedding: 64,
  feedForward: 128,
  heads: 4,
  kvHeads: 2,
  experts: 4,
  expertsUsed: 2,
  context: 256
}

class Writer {
  parts = []
  u32(v) {
    const b = Buffer.alloc(4)
    b.writeUInt32LE(v)
    this.parts.push(b)
  }
  i32(v) {
    const b = Buffer.alloc(4)
    b.writeInt32LE(v)
    this.parts.push(b)
  }
  u64(v) {
    const b = Buffer.alloc(8)
    b.writeBigUInt64LE(BigInt(v))
    this.parts.push(b)
  }
  f32(v) {
    const b = Buffer.alloc(4)
    b.writeFloatLE(v)
    this.parts.push(b)
  }
  str(s) {
    const b = Buffer.from(s, 'utf8')
    this.u64(b.length)
    this.parts.push(b)
  }
  raw(b) {
    this.parts.push(b)
  }
  get length() {
    return this.parts.reduce((n, p) => n + p.length, 0)
  }
  buffer() {
    return Buffer.concat(this.parts)
  }
}

function vocabulary() {
  const tokens = ['<unk>', '<s>', '</s>', '▁']
  const types = [2, 3, 3, 1]
  for (let i = 0; i < 256; i++) {
    tokens.push(`<0x${i.toString(16).toUpperCase().padStart(2, '0')}>`)
    types.push(6)
  }
  return { tokens, types }
}

/** Every tensor of the model, with ggml's dimension order (fastest-varying first). */
export function tinyMoeTensors(c = TINY_MOE) {
  const vocab = vocabulary().tokens.length
  const headDim = c.embedding / c.heads
  const kv = headDim * c.kvHeads
  const t = [
    ['token_embd.weight', [c.embedding, vocab]],
    ['output_norm.weight', [c.embedding]],
    ['output.weight', [c.embedding, vocab]]
  ]
  for (let i = 0; i < c.layers; i++) {
    t.push(
      [`blk.${i}.attn_norm.weight`, [c.embedding]],
      [`blk.${i}.attn_q.weight`, [c.embedding, c.embedding]],
      [`blk.${i}.attn_k.weight`, [c.embedding, kv]],
      [`blk.${i}.attn_v.weight`, [c.embedding, kv]],
      [`blk.${i}.attn_output.weight`, [c.embedding, c.embedding]],
      [`blk.${i}.ffn_norm.weight`, [c.embedding]],
      [`blk.${i}.ffn_gate_inp.weight`, [c.embedding, c.experts]],
      [`blk.${i}.ffn_gate_exps.weight`, [c.embedding, c.feedForward, c.experts]],
      [`blk.${i}.ffn_down_exps.weight`, [c.feedForward, c.embedding, c.experts]],
      [`blk.${i}.ffn_up_exps.weight`, [c.embedding, c.feedForward, c.experts]]
    )
  }
  return t
}

export function writeTinyMoeGguf(file, c = TINY_MOE) {
  const { tokens, types } = vocabulary()
  const meta = [
    ['general.architecture', T_STRING, 'llama'],
    ['general.name', T_STRING, 'tiny-moe-test'],
    ['general.alignment', T_UINT32, 32],
    ['llama.block_count', T_UINT32, c.layers],
    ['llama.context_length', T_UINT32, c.context],
    ['llama.embedding_length', T_UINT32, c.embedding],
    ['llama.feed_forward_length', T_UINT32, c.feedForward],
    ['llama.attention.head_count', T_UINT32, c.heads],
    ['llama.attention.head_count_kv', T_UINT32, c.kvHeads],
    ['llama.rope.dimension_count', T_UINT32, c.embedding / c.heads],
    ['llama.rope.freq_base', T_FLOAT32, 10000],
    ['llama.attention.layer_norm_rms_epsilon', T_FLOAT32, 1e-5],
    ['llama.expert_count', T_UINT32, c.experts],
    ['llama.expert_used_count', T_UINT32, c.expertsUsed],
    ['tokenizer.ggml.model', T_STRING, 'llama'],
    ['tokenizer.ggml.tokens', T_ARRAY, { type: T_STRING, values: tokens }],
    ['tokenizer.ggml.scores', T_ARRAY, { type: T_FLOAT32, values: tokens.map(() => 0) }],
    ['tokenizer.ggml.token_type', T_ARRAY, { type: T_INT32, values: types }],
    ['tokenizer.ggml.unknown_token_id', T_UINT32, 0],
    ['tokenizer.ggml.bos_token_id', T_UINT32, 1],
    ['tokenizer.ggml.eos_token_id', T_UINT32, 2]
  ]
  const tensors = tinyMoeTensors(c)
  const w = new Writer()
  w.raw(Buffer.from('GGUF'))
  w.u32(3)
  w.u64(tensors.length)
  w.u64(meta.length)
  const value = (type, v) => {
    if (type === T_UINT32) w.u32(v)
    else if (type === T_INT32) w.i32(v)
    else if (type === T_FLOAT32) w.f32(v)
    else if (type === T_STRING) w.str(v)
    else if (type === T_ARRAY) {
      w.u32(v.type)
      w.u64(v.values.length)
      for (const item of v.values) value(v.type, item)
    }
  }
  for (const [key, type, v] of meta) {
    w.str(key)
    w.u32(type)
    value(type, v)
  }
  let offset = 0
  const align = (n) => Math.ceil(n / 32) * 32
  for (const [name, dims] of tensors) {
    w.str(name)
    w.u32(dims.length)
    for (const d of dims) w.u64(d)
    w.u32(F32)
    w.u64(offset)
    offset = align(offset + dims.reduce((a, b) => a * b, 1) * 4)
  }
  w.raw(Buffer.alloc(align(w.length) - w.length))

  // A fixed-seed generator (mulberry32): the same weights every time.
  let seed = 0x5eed
  const random = () => {
    seed = (seed + 0x6d2b79f5) | 0
    let x = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296
  }
  for (const [name, dims] of tensors) {
    const count = dims.reduce((a, b) => a * b, 1)
    const data = Buffer.alloc(align(count * 4))
    const norm = name.endsWith('norm.weight')
    for (let i = 0; i < count; i++) data.writeFloatLE(norm ? 1 : (random() - 0.5) * 0.2, i * 4)
    w.raw(data)
  }
  fs.writeFileSync(file, w.buffer())
  return file
}
