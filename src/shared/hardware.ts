import { MODEL_CATALOG, type CatalogModel, type GpuInfo, type ModelId } from './setup.js'

/**
 * Which downloadable models suit this computer: the detected GPU and RAM held
 * against what each model needs, so the download list recommends what runs well
 * here rather than what ran well on the machine the catalog was written on.
 *
 * The VRAM thresholds are the engine's own Automatic choice (image-engine/
 * image_engine/memory.py, checked by tests/hardware.test.ts), so the mode named
 * here is the one the engine will pick. The RAM figures add up what that mode keeps
 * in system memory, from the component sizes listed in memory.py, plus the
 * headroom memory.py leaves for the system.
 */

const GIB = 2 ** 30

export type PlanName = 'full' | 'quality' | 'balanced' | 'low'

/** As the Settings view names the modes. */
export const PLAN_LABELS: Record<PlanName, string> = {
  full: 'Everything on GPU',
  quality: 'Full precision',
  balanced: 'Balanced',
  low: 'Low VRAM'
}

/** memory.py AUTO_THRESHOLDS, in GiB: the fastest plan whose minimum the GPU meets. */
export const IMAGE_AUTO_GIB: readonly (readonly [PlanName, number])[] = [
  ['full', 23],
  ['quality', 15.5],
  ['balanced', 9.5]
]

/** memory.py VIDEO_AUTO_THRESHOLDS, in GiB. */
export const VIDEO_AUTO_GIB: readonly (readonly [PlanName, number])[] = [
  ['full', 30],
  ['quality', 18],
  ['balanced', 9.5]
]

/** memory.py RAM_HEADROOM: left to the system and the rest of the engine. */
export const RAM_HEADROOM_GIB = 8

interface Needs {
  /** Automatic's thresholds, fastest first; below the last one, Low VRAM. Empty: no modes. */
  auto: readonly (readonly [PlanName, number])[]
  /** The least VRAM the model runs in at all, in its slowest mode. */
  minVramGiB: number
  /** What each mode keeps in system memory, in GiB. */
  ramGiB: Record<PlanName, number>
}

const VIDEO_NEEDS: Needs = {
  auto: VIDEO_AUTO_GIB,
  minVramGiB: 8,
  // Below 30 GB the 11.4 GB text encoder lives in RAM. Balanced parks the fp8 transformer
  // (5 GB) there while the decoder runs; Low VRAM streams the bf16 one (10 GB) from it and
  // keeps the 2.8 GB decoder there between clips.
  ramGiB: { full: 0, quality: 11.4, balanced: 16.4, low: 24.2 }
}

const NEEDS: Partial<Record<ModelId, Needs>> = {
  image: {
    auto: IMAGE_AUTO_GIB,
    minVramGiB: 6,
    // The 8 GB text encoder waits in RAM in the swapping modes; Low VRAM keeps every part
    // there (fp8 transformer 6.2 GB, text encoder 8 GB, VAE 0.2 GB).
    ramGiB: { full: 0, quality: 8, balanced: 8, low: 14.4 }
  },
  'video-fast': VIDEO_NEEDS,
  video: VIDEO_NEEDS,
  // Loaded whole onto the GPU in bf16 (8.3 GiB of weights) with room for the photos it reads.
  caption: { auto: [], minVramGiB: 10, ramGiB: { full: 0, quality: 0, balanced: 0, low: 0 } }
}

export interface HardwareInfo {
  /** The GPU the engine would use (the largest NVIDIA one), or null if nvidia-smi did not answer. */
  gpu: GpuInfo | null
  ramBytes: number
}

export type Fit =
  /** Runs in a mode faster than Low VRAM, with the RAM it wants. */
  | 'good'
  /** Runs, but only in Low VRAM mode or short of RAM. */
  | 'slow'
  /** Does not fit this GPU, or there is no NVIDIA GPU. */
  | 'no'

export interface ModelFit {
  fit: Fit
  /** The mode Automatic will pick here; null if the model has no modes or does not fit. */
  plan: PlanName | null
  /** One sentence for the download list. */
  note: string
}

const gib = (bytes: number): number => bytes / GIB
const roundGb = (value: number): string => String(Math.round(value))

export function vramGiB(gpu: GpuInfo): number {
  return gpu.memoryMiB / 1024
}

/** The mode memory.py's Automatic picks for this much VRAM. */
export function autoPlan(thresholds: Needs['auto'], vram: number): PlanName {
  for (const [name, minimum] of thresholds) if (vram >= minimum) return name
  return 'low'
}

/** How a model would run here; null for models with no known needs (parts). */
export function modelFit(model: CatalogModel, hw: HardwareInfo): ModelFit | null {
  const needs = NEEDS[model.id]
  if (!needs) return null
  if (!hw.gpu) return { fit: 'no', plan: null, note: 'Needs an NVIDIA GPU, and none was found.' }

  const vram = vramGiB(hw.gpu)
  // Cards and PCs report a little less than the size they are sold as (an 8 GB card shows
  // 8,188 MiB), so the minimums are compared as the nominal, rounded sizes.
  if (Math.round(vram) < needs.minVramGiB) {
    return {
      fit: 'no',
      plan: null,
      note: `Needs a GPU with at least ${needs.minVramGiB} GB of VRAM${needs.auto.length ? ', even in Low VRAM mode' : ''}; this one has ${roundGb(vram)} GB.`
    }
  }

  const plan = needs.auto.length ? autoPlan(needs.auto, vram) : null
  const ramWanted = Math.round((plan ? needs.ramGiB[plan] : 0) + RAM_HEADROOM_GIB)
  const ram = Math.round(gib(hw.ramBytes))
  const ramShort = ram < ramWanted

  let note = plan
    ? plan === 'low'
      ? 'Runs here only in Low VRAM mode, the slowest.'
      : `Runs here in ${PLAN_LABELS[plan]} mode.`
    : 'Fits on this GPU.'
  if (ramShort) {
    note += ` It wants about ${ramWanted} GB of RAM and this PC has ${ram} GB, so expect Windows to swap to disk.`
  }
  return { fit: plan === 'low' || ramShort ? 'slow' : 'good', plan, note }
}

/**
 * Whether to mark a model Recommended. Of the models the catalog recommends, only
 * those that run well here; with the hardware unknown, the catalog's choice as it is.
 */
export function isRecommendedHere(model: CatalogModel, fit: ModelFit | null): boolean {
  if (!model.recommended) return false
  return fit === null || fit.fit === 'good'
}

/** "NVIDIA GeForce RTX 4070 · 12 GB VRAM · 32 GB RAM". */
export function describeHardware(hw: HardwareInfo): string {
  const ram = `${roundGb(gib(hw.ramBytes))} GB RAM`
  return hw.gpu ? `${hw.gpu.name} · ${roundGb(vramGiB(hw.gpu))} GB VRAM · ${ram}` : `No NVIDIA GPU found · ${ram}`
}

/** The downloadable models that have known needs (every one but the parts). */
export function modelsWithNeeds(): CatalogModel[] {
  return MODEL_CATALOG.filter((m) => !m.part && NEEDS[m.id])
}
