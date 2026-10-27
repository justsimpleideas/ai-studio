import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { isGgufFile } from '../src/main/gguf.js'
import { modelDisplayName } from '../src/shared/engine-protocol.js'

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'gguf-test-'))
const file = (name: string, bytes: Buffer) => {
  const p = path.join(dir, name)
  fs.writeFileSync(p, bytes)
  return p
}

afterAll(() => fs.rmSync(dir, { recursive: true, force: true }))

describe('isGgufFile', () => {
  it('accepts a GGUF file whatever its name, like a hash-named blob', () =>
    Promise.all([
      expect(isGgufFile(file('model.gguf', Buffer.from('GGUF\x03\x00\x00\x00rest')))).resolves.toBe(true),
      expect(isGgufFile(file('sha256-0123456789ab', Buffer.from('GGUF\x03\x00\x00\x00')))).resolves.toBe(true)
    ]))

  it('rejects other files, short files and missing files', () =>
    Promise.all([
      expect(isGgufFile(file('fake.gguf', Buffer.from('PK\x03\x04zip')))).resolves.toBe(false),
      expect(isGgufFile(file('params', Buffer.from('{"stop":[]}')))).resolves.toBe(false),
      expect(isGgufFile(file('short', Buffer.from('GGU')))).resolves.toBe(false),
      expect(isGgufFile(file('empty', Buffer.alloc(0)))).resolves.toBe(false),
      expect(isGgufFile(path.join(dir, 'missing.gguf'))).resolves.toBe(false)
    ]))
})

describe('modelDisplayName', () => {
  it('keeps a .gguf file name, which usually carries the quantization', () => {
    expect(modelDisplayName('Some-Model-9B-Q4_K_M.gguf', 'Some Model 9B')).toBe('Some-Model-9B-Q4_K_M.gguf')
  })

  it('uses the header name for a blob named only by its hash', () => {
    expect(modelDisplayName(`sha256-${'ab12'.repeat(16)}`, 'Some Model 9B')).toBe('Some Model 9B')
  })

  it('falls back to whatever there is', () => {
    expect(modelDisplayName('weights', undefined)).toBe('weights')
    expect(modelDisplayName(undefined, '  ')).toBe('model')
  })
})
