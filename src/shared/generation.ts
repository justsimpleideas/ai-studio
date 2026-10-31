/**
 * How a reply is generated: sampling, thinking, and how much context to open.
 *
 * None of this is tied to one model. What the file carries in its header
 * (architecture, whether its chat template has a thinking segment, the context
 * it was trained for) decides the defaults; the user can override any of them.
 */

/** What the model's header says about it, read at load time. */
export interface ModelTraits {
  /** The architecture name from the header, e.g. "qwen35moe" or "llama". */
  architecture: string
  /** Set for a mixture-of-experts model: how many experts it has and how many run per token. */
  experts: { total: number; active: number } | null
  /** Whether the chat template has a thinking segment the model can write into. */
  thinking: boolean
  /** The dominant weight format, e.g. "Q4_K", as the library describes it. */
  quantization: string | null
  parameters: number
}

/**
 * Whether the model writes a chain of thought before its reply.
 * - `on`: opened at the start of every reply, the way a thinking model is trained.
 * - `model`: only when the model starts one itself.
 * - `off`: discouraged, and cut at once if it starts anyway.
 * Ignored by a model whose template has no thinking segment.
 */
export type ThinkingMode = 'on' | 'model' | 'off'

export const THINKING_MODES: readonly ThinkingMode[] = ['on', 'model', 'off']

export function isThinkingMode(value: unknown): value is ThinkingMode {
  return (THINKING_MODES as readonly unknown[]).includes(value)
}

export interface SamplingValues {
  temperature: number
  topP: number
  topK: number
  minP: number
}

/** A sampling setting: the publisher's recommendation for the model, or numbers of the user's own. */
export type SamplingChoice = { kind: 'recommended' } | ({ kind: 'custom' } & SamplingValues)

export interface GenerationPreferences {
  thinking: ThinkingMode
  /** Most tokens a chain of thought may take; null leaves it to the library (most of the context). */
  thinkingBudget: number | null
  sampling: SamplingChoice
}

export const DEFAULT_GENERATION: GenerationPreferences = {
  // Off by default: a thinking model answers far sooner, and it can be turned on per need.
  thinking: 'off',
  thinkingBudget: null,
  sampling: { kind: 'recommended' }
}

/** Thinking budgets offered in the UI; null is "no limit". */
export const THINKING_BUDGETS: readonly (number | null)[] = [2048, 8192, 32768, null]

/**
 * How much context to open when the user has not said: the trained size, up to
 * this. A model trained for 262,144 tokens would otherwise claim memory for all
 * of it and push layers off the GPU; a 32k window holds a long conversation and
 * leaves the GPU for the weights.
 */
export const DEFAULT_CONTEXT_CAP = 32_768

/** Context sizes the UI offers, pruned to what the model was trained for. */
export const CONTEXT_CHOICES: readonly number[] = [8_192, 16_384, 32_768, 65_536, 131_072, 262_144]

export function defaultContextSize(trainContextSize: number): number {
  return Math.min(trainContextSize, DEFAULT_CONTEXT_CAP)
}

/**
 * The context sizes worth offering for a model: each standard choice below the
 * trained size, then the trained size itself.
 */
export function contextChoicesFor(trainContextSize: number): number[] {
  const below = CONTEXT_CHOICES.filter((n) => n < trainContextSize)
  return [...below, trainContextSize]
}

const GENERIC: SamplingValues = { temperature: 0.7, topP: 0.9, topK: 40, minP: 0.05 }

/**
 * The sampling the model's publisher recommends, by architecture family.
 *
 * Each row is what the model card says; the fallback is a middle-of-the-road
 * setting that no common model misbehaves on. With no temperature at all (the
 * library's default) a thinking model tends to repeat itself, which is why this
 * exists.
 */
export function recommendedSampling(traits: Pick<ModelTraits, 'architecture' | 'thinking'>): SamplingValues {
  const arch = traits.architecture.toLowerCase()
  if (arch.startsWith('qwen')) {
    // Qwen3 family: the "thinking, precise coding" and "non-thinking" rows of the model card.
    return traits.thinking
      ? { temperature: 0.6, topP: 0.95, topK: 20, minP: 0 }
      : { temperature: 0.7, topP: 0.8, topK: 20, minP: 0 }
  }
  if (arch.startsWith('gemma')) return { temperature: 1.0, topP: 0.95, topK: 64, minP: 0 }
  if (arch.startsWith('llama')) return { temperature: 0.6, topP: 0.9, topK: 40, minP: 0 }
  if (arch.startsWith('deepseek')) return { temperature: 0.6, topP: 0.95, topK: 40, minP: 0 }
  if (arch.startsWith('mistral')) return { temperature: 0.15, topP: 1.0, topK: 40, minP: 0 }
  if (arch.startsWith('gpt-oss') || arch.startsWith('gptoss')) return { temperature: 1.0, topP: 1.0, topK: 0, minP: 0 }
  return GENERIC
}

/** The numbers a prompt actually runs with. */
export function resolveSampling(choice: SamplingChoice, traits: Pick<ModelTraits, 'architecture' | 'thinking'>): SamplingValues {
  if (choice.kind === 'custom') {
    const { temperature, topP, topK, minP } = choice
    return { temperature, topP, topK, minP }
  }
  return recommendedSampling(traits)
}

function finite(value: unknown, min: number, max: number): number | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null
  return value >= min && value <= max ? value : null
}

export function isSamplingChoice(value: unknown): value is SamplingChoice {
  if (!value || typeof value !== 'object') return false
  const raw = value as Record<string, unknown>
  if (raw.kind === 'recommended') return true
  if (raw.kind !== 'custom') return false
  return (
    finite(raw.temperature, 0, 2) !== null &&
    finite(raw.topP, 0, 1) !== null &&
    finite(raw.topK, 0, 1000) !== null &&
    Number.isInteger(raw.topK) &&
    finite(raw.minP, 0, 1) !== null
  )
}

/** A saved or submitted generation setting, with anything unusable replaced by the default. */
export function parseGeneration(value: unknown): GenerationPreferences {
  const raw = (value && typeof value === 'object' ? value : {}) as Record<string, unknown>
  const budget = raw.thinkingBudget
  return {
    thinking: isThinkingMode(raw.thinking) ? raw.thinking : DEFAULT_GENERATION.thinking,
    thinkingBudget: typeof budget === 'number' && Number.isInteger(budget) && budget >= 0 ? budget : null,
    sampling: isSamplingChoice(raw.sampling)
      ? raw.sampling.kind === 'recommended'
        ? { kind: 'recommended' }
        : {
            kind: 'custom',
            temperature: raw.sampling.temperature,
            topP: raw.sampling.topP,
            topK: raw.sampling.topK,
            minP: raw.sampling.minP
          }
      : { kind: 'recommended' }
  }
}

/** A saved context size: a whole number of tokens, or null for the default. */
export function parseContextSize(value: unknown): number | null {
  return typeof value === 'number' && Number.isInteger(value) && value >= 512 ? value : null
}
