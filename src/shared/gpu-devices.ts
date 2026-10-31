/**
 * Which GPUs the chat engine should use when Vulkan sees more than one.
 *
 * A laptop exposes its integrated GPU to Vulkan next to the discrete one, and
 * llama.cpp then counts the integrated GPU's share of system memory as VRAM and
 * spreads the model across both. Planned that way, a model that does not fit
 * the discrete card is placed as if it had twice the memory, and the load fails
 * with an out-of-device-memory error partway through. When a discrete GPU is
 * present, the integrated one is left out (GGML_VK_VISIBLE_DEVICES); when there
 * is none, nothing changes, since the integrated GPU is then the best there is.
 *
 * Judged by the names Vulkan reports, the only description of a device the
 * inference library exposes.
 */

/** True for a GPU that shares system memory (or renders in software) rather than having its own. */
export function isIntegratedGpu(name: string): boolean {
  const n = name.toLowerCase()
  if (/llvmpipe|lavapipe|swiftshader|microsoft basic render/.test(n)) return true
  if (n.includes('nvidia')) return false
  // Intel's discrete cards carry a model number (Arc A770, Arc B580); its integrated GPUs do not.
  if (n.includes('intel')) return !/\barc(\(tm\))?\s+[ab]\d{3}/.test(n)
  // AMD's discrete cards are RX or Pro; "Radeon Graphics", "Radeon 780M" and "Vega 8" are integrated.
  if (/\bamd\b|radeon/.test(n)) return !/\brx\b|\bpro\b/.test(n)
  return false
}

/**
 * The device indices to keep visible, or null to leave the engine as it is:
 * when every device is discrete, when every device is integrated, or when
 * there is only one.
 */
export function discreteDeviceIndices(names: readonly string[]): number[] | null {
  const keep = names.flatMap((name, i) => (isIntegratedGpu(name) ? [] : [i]))
  if (keep.length === 0 || keep.length === names.length) return null
  return keep
}
