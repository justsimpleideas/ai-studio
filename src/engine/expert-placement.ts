/**
 * Keeping a mixture-of-experts model's experts in system memory.
 *
 * Splitting a model by layer, the library's only way, puts whole layers on the
 * CPU once the GPU is full, and every token then waits on the CPU for those
 * layers' attention as well as their experts. Split by tensor instead, the GPU
 * holds every layer's attention and shared weights (small, and used by every
 * token) and system memory holds the experts (most of the file, and only a few
 * used per token). For a model bigger than the GPU that is several times faster.
 *
 * The library's stock addon has no option for it; the studio's build of the
 * addon adds one (scripts/native/addon-patch.mjs), and only that build is told
 * to use it.
 */

/** llama.cpp's own pattern for a layer's expert weights (common/common.h, LLM_FFN_EXPS_REGEX). */
export const EXPERT_TENSORS = '\\.ffn_(up|down|gate|gate_up)_(ch|)exps'

/** One entry of the addon's `tensorBufferOverrides` option. */
export interface TensorBufferOverride {
  /** A regular expression, matched (std::regex_search) against each tensor's name. */
  pattern: string
  target: 'cpu'
}

/**
 * The overrides that keep experts on the CPU: every layer's (null), or the
 * first `count` layers', as llama-server's --cpu-moe and --n-cpu-moe do. Zero
 * is none at all.
 */
export function expertOverrides(count: number | null, layers: number): TensorBufferOverride[] {
  if (count === null || count >= layers) return [{ pattern: EXPERT_TENSORS, target: 'cpu' }]
  return Array.from({ length: Math.max(0, count) }, (_, i) => ({ pattern: `blk\\.${i}${EXPERT_TENSORS}`, target: 'cpu' }))
}

/** The feature the studio's addon build reports (scripts/native/addon-patch.mjs, ADDON_FEATURE). */
export const ADDON_FEATURE = 'tensorBufferOverrides'

/**
 * Whether the loaded addon is the studio's build. The stock addon ignores an
 * option it does not know, which here would mean loading every expert onto
 * the GPU, so the option is only ever passed after this says yes.
 */
export function addonPlacesTensors(llama: unknown): boolean {
  const bindings = (llama as { _bindings?: { getStudioFeatures?: () => unknown } })._bindings
  if (typeof bindings?.getStudioFeatures !== 'function') return false
  const features = bindings.getStudioFeatures()
  return Array.isArray(features) && features.includes(ADDON_FEATURE)
}

export type ExpertPlan =
  | { kind: 'none' }
  | { kind: 'place'; overrides: TensorBufferOverride[]; label: number | 'all' }
  | { kind: 'unavailable'; reason: string }

/**
 * What to do with a model's experts: nothing (a dense model, or the user asked
 * for them all on the GPU), place them, or explain why they cannot be.
 */
export function planExperts(options: {
  experts: number | undefined
  layers: number | undefined
  expertsOnCpu: number | null
  backend: string | false
  addonPlaces: boolean
}): ExpertPlan {
  const { experts, layers, expertsOnCpu, backend, addonPlaces } = options
  if (typeof experts !== 'number' || experts <= 1 || expertsOnCpu === 0) return { kind: 'none' }
  if (backend === false) return { kind: 'none' }
  if (backend !== 'cuda') return { kind: 'unavailable', reason: `Experts on the CPU needs the CUDA backend; this model is running on ${backend}.` }
  if (!addonPlaces) {
    return { kind: 'unavailable', reason: 'Experts on the CPU needs the studio’s build of the llama.cpp addon, which this copy does not have.' }
  }
  if (typeof layers !== 'number' || layers <= 0) return { kind: 'unavailable', reason: 'The file does not say how many layers the model has.' }
  return {
    kind: 'place',
    overrides: expertOverrides(expertsOnCpu, layers),
    label: expertsOnCpu === null || expertsOnCpu >= layers ? 'all' : expertsOnCpu
  }
}
