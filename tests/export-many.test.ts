import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { parseSaveManyOptions, promptText, requireIds, safeFileName, writeUnique } from '../src/main/image/export-many.js'
import type { ImageRecord } from '../src/shared/image.js'

let folder: string
beforeEach(() => {
  folder = fs.mkdtempSync(path.join(os.tmpdir(), 'save-many-'))
})
afterEach(() => fs.rmSync(folder, { recursive: true, force: true }))

describe('saving several items', () => {
  it('never overwrites: a taken name gets a number', async () => {
    fs.writeFileSync(path.join(folder, 'a.png'), 'mine')
    expect(path.basename(await writeUnique(folder, 'a.png', 'one'))).toBe('a (2).png')
    expect(path.basename(await writeUnique(folder, 'a.png', 'two'))).toBe('a (3).png')
    expect(fs.readFileSync(path.join(folder, 'a.png'), 'utf8')).toBe('mine')
  })

  it('keeps only a plain file name from the engine', () => {
    expect(safeFileName('20261031-abc.png', 'x.png')).toBe('20261031-abc.png')
    expect(safeFileName('../../evil.png', 'x.png')).toBe('evil.png')
    expect(safeFileName('..\\..\\evil.png', 'x.png')).toBe('evil.png')
    expect(safeFileName('', 'x.png')).toBe('x.png')
    expect(safeFileName('.hidden', 'x.png')).toBe('x.png')
    expect(safeFileName('no-extension', 'x.png')).toBe('x.png')
  })

  it('checks the selection and the options', () => {
    expect(requireIds(['0123456789abcdef', '0123456789abcdef'])).toEqual(['0123456789abcdef'])
    for (const bad of [[], ['../x'], 'abc', Array(2001).fill('0123456789abcdef')]) expect(() => requireIds(bad)).toThrow()
    expect(parseSaveManyOptions({ includeSettings: true, promptFiles: false })).toEqual({ includeSettings: true, promptFiles: false })
    expect(() => parseSaveManyOptions({ includeSettings: 'yes', promptFiles: false })).toThrow()
    expect(() => parseSaveManyOptions(null)).toThrow()
  })

  it('a prompt file holds the prompt, then the settings', () => {
    const record = {
      prompt: 'A lighthouse', negative_prompt: 'blurry', width: 1024, height: 768, seed: 9, steps: 8, guidance: 0, model: 'example/model',
      media: 'image'
    } as ImageRecord
    expect(promptText(record)).toBe('A lighthouse\n\nNegative prompt: blurry\nSize: 1024 x 768\nSeed: 9\nSteps: 8\nGuidance: 0\nModel: example/model\n')
  })
})
