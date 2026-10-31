import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { ConfigStore } from '../src/main/store.js'
import {
  contextChoicesFor,
  DEFAULT_CONTEXT_CAP,
  DEFAULT_GENERATION,
  defaultContextSize,
  isSamplingChoice,
  parseContextSize,
  parseGeneration,
  recommendedSampling,
  resolveSampling
} from '../src/shared/generation.js'

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'generation-test-'))
afterAll(() => fs.rmSync(dir, { recursive: true, force: true }))

function storeWith(model: unknown): ConfigStore {
  const file = path.join(dir, `${Math.random().toString(16).slice(2)}.json`)
  fs.writeFileSync(file, JSON.stringify({ model }))
  return new ConfigStore(file)
}

describe('recommended sampling, by what the header says', () => {
  it('gives a thinking Qwen the model card’s precise-coding row, and a plain one the non-thinking row', () => {
    expect(recommendedSampling({ architecture: 'qwen35moe', thinking: true })).toEqual({
      temperature: 0.6,
      topP: 0.95,
      topK: 20,
      minP: 0
    })
    expect(recommendedSampling({ architecture: 'qwen3', thinking: false })).toEqual({
      temperature: 0.7,
      topP: 0.8,
      topK: 20,
      minP: 0
    })
  })

  it('knows the other common families and is not fooled by case', () => {
    expect(recommendedSampling({ architecture: 'Gemma4', thinking: true }).topK).toBe(64)
    expect(recommendedSampling({ architecture: 'llama', thinking: false }).temperature).toBe(0.6)
    expect(recommendedSampling({ architecture: 'mistral3', thinking: false }).temperature).toBe(0.15)
  })

  it('never samples greedily for a family it does not know', () => {
    const generic = recommendedSampling({ architecture: 'somethingnew', thinking: false })
    expect(generic.temperature).toBeGreaterThan(0)
    expect(generic.topP).toBeLessThanOrEqual(1)
  })

  it('resolves "recommended" through the table and "custom" to the numbers given, nothing else', () => {
    const traits = { architecture: 'qwen35', thinking: true }
    expect(resolveSampling({ kind: 'recommended' }, traits)).toEqual(recommendedSampling(traits))
    expect(resolveSampling({ kind: 'custom', temperature: 1.1, topP: 0.5, topK: 7, minP: 0.1 }, traits)).toEqual({
      temperature: 1.1,
      topP: 0.5,
      topK: 7,
      minP: 0.1
    })
  })
})

describe('context size defaults', () => {
  it('opens the trained size up to the cap', () => {
    expect(defaultContextSize(8_192)).toBe(8_192)
    expect(defaultContextSize(262_144)).toBe(DEFAULT_CONTEXT_CAP)
  })

  it('offers the standard sizes below the trained one, then the trained one', () => {
    expect(contextChoicesFor(262_144)).toEqual([8_192, 16_384, 32_768, 65_536, 131_072, 262_144])
    expect(contextChoicesFor(40_960)).toEqual([8_192, 16_384, 32_768, 40_960])
    expect(contextChoicesFor(4_096)).toEqual([4_096])
  })

  it('accepts a saved whole number of tokens and nothing odd', () => {
    expect(parseContextSize(16_384)).toBe(16_384)
    for (const bad of [null, undefined, '16384', 16_384.5, 100, -1, NaN]) expect(parseContextSize(bad)).toBeNull()
  })
})

describe('generation preferences', () => {
  it('fills a missing or broken setting with the default', () => {
    expect(parseGeneration(undefined)).toEqual(DEFAULT_GENERATION)
    expect(parseGeneration({ thinking: 'always', thinkingBudget: -5, sampling: { kind: 'custom', temperature: 9 } })).toEqual(
      DEFAULT_GENERATION
    )
  })

  it('keeps a complete custom setting and an explicit budget', () => {
    const custom = { kind: 'custom' as const, temperature: 0.3, topP: 0.9, topK: 50, minP: 0.02 }
    expect(parseGeneration({ thinking: 'off', thinkingBudget: 2048, sampling: { ...custom, extra: 'dropped' } })).toEqual({
      thinking: 'off',
      thinkingBudget: 2048,
      sampling: custom
    })
  })

  it('refuses custom numbers out of range or non-integer top-k', () => {
    const base = { kind: 'custom', temperature: 0.5, topP: 0.9, topK: 40, minP: 0 }
    expect(isSamplingChoice(base)).toBe(true)
    expect(isSamplingChoice({ ...base, temperature: 2.5 })).toBe(false)
    expect(isSamplingChoice({ ...base, topP: 1.2 })).toBe(false)
    expect(isSamplingChoice({ ...base, topK: 2.5 })).toBe(false)
    expect(isSamplingChoice({ ...base, minP: -0.1 })).toBe(false)
    expect(isSamplingChoice({ kind: 'recommended' })).toBe(true)
    expect(isSamplingChoice({ kind: 'other' })).toBe(false)
  })

  it('is the default for a config saved before the setting existed, and survives a save', () => {
    expect(storeWith({ backend: 'auto' }).get().model.generation).toEqual(DEFAULT_GENERATION)
    const saved = storeWith({ generation: { thinking: 'model', thinkingBudget: 8192, sampling: { kind: 'recommended' } } })
    expect(saved.get().model.generation).toEqual({ thinking: 'model', thinkingBudget: 8192, sampling: { kind: 'recommended' } })
  })

  it('turns a saved context size that is not a number back into the default', () => {
    expect(storeWith({ contextSize: 'lots' }).get().model.contextSize).toBeNull()
    expect(storeWith({ contextSize: 65_536 }).get().model.contextSize).toBe(65_536)
  })
})
