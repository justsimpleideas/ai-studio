import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { kvCacheOptions, kvModelOptions } from '../src/engine/kv-cache.js'
import { ConfigStore } from '../src/main/store.js'
import { isKvCacheType } from '../src/shared/config.js'

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kv-cache-test-'))
afterAll(() => fs.rmSync(dir, { recursive: true, force: true }))

function storeWith(model: unknown): ConfigStore {
  const file = path.join(dir, `${Math.random().toString(16).slice(2)}.json`)
  fs.writeFileSync(file, JSON.stringify({ model }))
  return new ConfigStore(file)
}

describe('context memory (KV cache) type', () => {
  it('adds nothing for 16-bit, the library default', () => {
    expect(kvCacheOptions('f16')).toEqual({})
  })

  it('sets both halves of the cache to 8-bit, with the flash attention it requires', () => {
    expect(kvCacheOptions('q8_0')).toEqual({
      flashAttention: true,
      experimentalKvCacheKeyType: 'Q8_0',
      experimentalKvCacheValueType: 'Q8_0'
    })
  })

  it('gives the model load the same choice, so the GPU layer fit plans for it', () => {
    expect(kvModelOptions('f16')).toEqual({})
    expect(kvModelOptions('q8_0')).toEqual({
      defaultContextFlashAttention: true,
      experimentalDefaultContextKvCacheKeyType: 'Q8_0',
      experimentalDefaultContextKvCacheValueType: 'Q8_0'
    })
  })

  it('accepts only the two known types', () => {
    expect(isKvCacheType('f16')).toBe(true)
    expect(isKvCacheType('q8_0')).toBe(true)
    for (const other of ['q4_0', 'F16', '', null, undefined, 8]) expect(isKvCacheType(other)).toBe(false)
  })

  it('defaults to 8-bit for a config saved before the setting existed', () => {
    expect(storeWith({ backend: 'auto' }).get().model.kvCache).toBe('q8_0')
  })

  it('keeps a saved choice, and turns an unknown one into the default (8-bit)', () => {
    expect(storeWith({ kvCache: 'q8_0' }).get().model.kvCache).toBe('q8_0')
    expect(storeWith({ kvCache: 'f16' }).get().model.kvCache).toBe('f16')
    expect(storeWith({ kvCache: 'q2_k' }).get().model.kvCache).toBe('q8_0')
  })
})
