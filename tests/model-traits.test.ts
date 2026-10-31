import { describe, expect, it } from 'vitest'
import { GeneralChatWrapper, GgmlType, QwenChatWrapper } from 'node-llama-cpp'
import { describeModel, loadPlan, type ModelHeader } from '../src/engine/model-traits.js'
import { ChatRecorder } from '../src/main/chat-recorder.js'
import { thoughtBudget, wrapperSettingsFor } from '../src/engine/thinking.js'
import { DEFAULT_CONTEXT_CAP } from '../src/shared/generation.js'

/**
 * What the engine learns about a model from its header and chat template, with
 * no model loaded: the header is a plain object and the templates are the
 * library's real ones.
 */

const dense: ModelHeader = {
  architecture: 'llama',
  architectureMetadata: {},
  dominantTensorType: GgmlType.Q4_K,
  totalParameters: 8_000_000_000
}

const moe: ModelHeader = {
  architecture: 'qwen35moe',
  architectureMetadata: { expert_count: 256, expert_used_count: 8 },
  dominantTensorType: GgmlType.Q4_K,
  totalParameters: 35_000_000_000
}

describe('describing a model from its header', () => {
  it('reads the expert layout of a mixture-of-experts model, and none for a dense one', () => {
    expect(describeModel(moe, new QwenChatWrapper().settings).experts).toEqual({ total: 256, active: 8 })
    expect(describeModel(dense, new GeneralChatWrapper().settings).experts).toBeNull()
  })

  it('takes thinking from the chat template, not the name', () => {
    expect(describeModel(moe, new QwenChatWrapper().settings).thinking).toBe(true)
    expect(describeModel(moe, new GeneralChatWrapper().settings).thinking).toBe(false)
    // Discouraging thought keeps the segment in the template, so the trait holds whatever the mode.
    expect(describeModel(moe, new QwenChatWrapper({ thoughts: 'discourage' }).settings).thinking).toBe(true)
  })

  it('names the weight format and keeps the architecture and size', () => {
    const traits = describeModel(moe, new QwenChatWrapper().settings)
    expect(traits).toMatchObject({ architecture: 'qwen35moe', quantization: 'Q4_K', parameters: 35_000_000_000 })
    expect(describeModel({ ...dense, dominantTensorType: undefined }, new GeneralChatWrapper().settings).quantization).toBeNull()
  })
})

describe('planning a load', () => {
  it('caps the context by default and asks the layers to leave room for exactly that', () => {
    expect(loadPlan(262_144, { contextSize: null, gpuLayers: null })).toEqual({
      contextSize: DEFAULT_CONTEXT_CAP,
      gpuLayers: { fitContext: { contextSize: DEFAULT_CONTEXT_CAP } }
    })
    expect(loadPlan(8_192, { contextSize: null, gpuLayers: null }).contextSize).toBe(8_192)
  })

  it('never opens more context than the model was trained for', () => {
    expect(loadPlan(32_768, { contextSize: 131_072, gpuLayers: null }).contextSize).toBe(32_768)
  })

  it('passes an explicit layer count through untouched', () => {
    expect(loadPlan(262_144, { contextSize: 16_384, gpuLayers: 20 })).toEqual({ contextSize: 16_384, gpuLayers: 20 })
  })
})

describe('thinking mode, expressed to the library', () => {
  it('maps the three modes onto the wrapper switch', () => {
    expect(wrapperSettingsFor('on')).toEqual({ qwen: { thoughts: 'auto' } })
    expect(wrapperSettingsFor('model')).toEqual({ qwen: { thoughts: 'modelInitiated' } })
    expect(wrapperSettingsFor('off')).toEqual({ qwen: { thoughts: 'discourage' } })
  })

  it('cuts a thought at once when thinking is off, whatever the wrapper did', () => {
    expect(thoughtBudget('off', null)).toBe(0)
    expect(thoughtBudget('off', 8192)).toBe(0)
  })

  it('otherwise uses the user’s cap, or leaves it to the library', () => {
    expect(thoughtBudget('on', 2048)).toBe(2048)
    expect(thoughtBudget('model', null)).toBeUndefined()
  })
})

describe('a recorded chain of thought', () => {
  it('is kept with the reply it led to, trimmed, and only when there was one', () => {
    const rec = ChatRecorder.fresh()
    rec.userMessage('p1', 'Why is the sky blue?')
    rec.thought('p1', 'Rayleigh ')
    rec.thought('p1', 'scattering.\n')
    rec.chunk('p1', 'Because of scattering.')
    rec.complete('p1', 'Because of scattering.')
    rec.userMessage('p2', 'Thanks')
    rec.chunk('p2', 'Any time.')
    rec.complete('p2', 'Any time.')
    expect(rec.chat.turns[1]).toEqual({ id: 'p1', role: 'assistant', text: 'Because of scattering.', thinking: 'Rayleigh scattering.' })
    expect(rec.chat.turns[3]).toEqual({ id: 'p2', role: 'assistant', text: 'Any time.' })
  })

  it('ignores a thought for a prompt it is not recording', () => {
    const rec = ChatRecorder.fresh()
    rec.thought('ghost', 'hmm')
    rec.userMessage('p1', 'Hi')
    rec.complete('p1', 'Hello')
    expect(rec.chat.turns[1]).toEqual({ id: 'p1', role: 'assistant', text: 'Hello' })
  })
})
