import { describe, expect, it } from 'vitest'
import { DEFAULT_SERVER as DEFAULTS } from '../../src/shared/config.js'
import type { ModelInfo } from '../../src/shared/engine-protocol.js'
import { loadedDiffers, loadedExpertsOnCpu } from '../../src/renderer/components/EngineSettings.js'

// The model below was loaded with the engine's own thread count.
const DEFAULT_SERVER = { ...DEFAULTS, threads: null }

const model = (patch: Partial<ModelInfo>): ModelInfo =>
  ({
    path: 'C:/Users/You/models/a.gguf',
    name: 'a',
    trainContextSize: 32768,
    contextSize: 32768,
    gpuLayers: 40,
    backend: 'cuda',
    devices: [],
    fileSizeBytes: 1,
    kvCache: 'f16',
    engine: 'builtin',
    threads: null,
    traits: { architecture: 'qwen3moe', experts: { total: 128, active: 8 }, thinking: true, quantization: null, parameters: 1 },
    ...patch
  }) as ModelInfo

describe('offering a reload from the model menu', () => {
  it('reads where the loaded model’s experts are', () => {
    expect(loadedExpertsOnCpu(model({ expertsOnCpu: 'all' }))).toBe('all')
    expect(loadedExpertsOnCpu(model({ expertsOnCpu: 24 }))).toBe(24)
    // Split by layer, as asked ("None").
    expect(loadedExpertsOnCpu(model({}))).toBe(0)
    expect(loadedExpertsOnCpu(model({ traits: { architecture: 'llama', experts: null, thinking: false, quantization: null, parameters: 1 } }))).toBeNull()
    expect(loadedExpertsOnCpu(model({ expertsOnCpuUnavailable: 'Needs CUDA' }))).toBeNull()
    expect(loadedExpertsOnCpu(model({ backend: 'cpu' }))).toBeNull()
  })

  it('offers one when the expert placement or the threads changed', () => {
    const loaded = model({ expertsOnCpu: 'all' })
    expect(loadedDiffers(loaded, 'builtin', DEFAULT_SERVER)).toBe(false)
    expect(loadedDiffers(loaded, 'builtin', { ...DEFAULT_SERVER, expertsOnCpu: 16 })).toBe(true)
    expect(loadedDiffers(loaded, 'builtin', { ...DEFAULT_SERVER, threads: 8 })).toBe(true)
    expect(loadedDiffers(loaded, 'server', DEFAULT_SERVER)).toBe(true)
  })

  it('does not offer one that would change nothing', () => {
    // The experts could not be placed (Vulkan): another count would not help.
    const vulkan = model({ backend: 'vulkan', expertsOnCpuUnavailable: 'Needs CUDA' })
    expect(loadedDiffers(vulkan, 'builtin', { ...DEFAULT_SERVER, expertsOnCpu: 16 })).toBe(false)
    expect(loadedDiffers(model({ expertsOnCpu: 0 }), 'builtin', { ...DEFAULT_SERVER, expertsOnCpu: 0 })).toBe(false)
  })
})
