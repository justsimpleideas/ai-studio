import type { ResolveChatWrapperWithModelOptions } from 'node-llama-cpp'
import type { ThinkingMode } from '@shared/generation.js'

/**
 * The thinking mode, expressed to the library.
 *
 * Two levers, used together. The chat wrapper decides whether a thought
 * segment is opened at the start of a reply; only some wrappers expose that
 * switch, so it is set where it exists. The budget caps how long a thought may
 * run once open, and works on every wrapper that has a thought segment, so "off"
 * also sets it to zero: a model that opens a thought anyway has it closed at
 * once, and the reply begins.
 */
export function wrapperSettingsFor(mode: ThinkingMode): NonNullable<ResolveChatWrapperWithModelOptions['customWrapperSettings']> {
  const thoughts = mode === 'on' ? 'auto' : mode === 'model' ? 'modelInitiated' : 'discourage'
  return { qwen: { thoughts } }
}

/** The thought budget for a prompt: the user's cap, zero when thinking is off, or the library's default. */
export function thoughtBudget(mode: ThinkingMode, budget: number | null): number | undefined {
  if (mode === 'off') return 0
  return budget ?? undefined
}
