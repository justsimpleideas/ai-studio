import type { ChatWrapperSettings, LlamaModelOptions } from 'node-llama-cpp'
import { GgmlType } from 'node-llama-cpp'
import { defaultContextSize, type ModelTraits } from '@shared/generation.js'

/**
 * The parts of a loaded model that describe it, narrowed so a test can hand in
 * a header without loading anything.
 */
export interface ModelHeader {
  architecture: string
  /** The architecture's own metadata block, where a mixture-of-experts model counts its experts. */
  architectureMetadata: { expert_count?: number; expert_used_count?: number }
  dominantTensorType: GgmlType | undefined
  totalParameters: number
}

/**
 * What the file says about the model, read from its header rather than its
 * name. A chat template is the authority on thinking: if the library's wrapper
 * for this model has a thought segment, the model can write one.
 */
export function describeModel(header: ModelHeader, wrapperSettings: ChatWrapperSettings): ModelTraits {
  const { expert_count: total, expert_used_count: active } = header.architectureMetadata
  const experts = typeof total === 'number' && total > 1 ? { total, active: typeof active === 'number' ? active : 1 } : null
  const tensorType = header.dominantTensorType
  return {
    architecture: header.architecture,
    experts,
    thinking: wrapperSettings.segments?.thought != null,
    quantization: tensorType === undefined ? null : (GgmlType[tensorType] ?? null),
    parameters: header.totalParameters
  }
}

/**
 * How to open the model: the context to ask for, and how to place the layers.
 *
 * With no context size chosen, the library would plan for the trained size and
 * fit the layers around that, which on a model trained for 262k tokens means
 * most of the weights off the GPU. Instead the context is the trained size up
 * to a cap, and the layer count is asked to leave room for exactly that
 * context. An explicit layer count is passed through untouched.
 */
export function loadPlan(
  trainContextSize: number,
  requested: { contextSize: number | null; gpuLayers: number | null }
): { contextSize: number; gpuLayers: NonNullable<LlamaModelOptions['gpuLayers']> } {
  const contextSize = Math.min(requested.contextSize ?? defaultContextSize(trainContextSize), trainContextSize)
  return {
    contextSize,
    gpuLayers: requested.gpuLayers ?? { fitContext: { contextSize } }
  }
}
