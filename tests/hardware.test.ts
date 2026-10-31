import fs from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'
import {
  IMAGE_AUTO_GIB,
  RAM_HEADROOM_GIB,
  VIDEO_AUTO_GIB,
  describeHardware,
  isRecommendedHere,
  modelFit,
  modelsWithNeeds,
  type HardwareInfo
} from '../src/shared/hardware.js'
import { MODEL_CATALOG, catalogModel, type GpuInfo, type ModelId } from '../src/shared/setup.js'
import { HardwareProbe, type Runner } from '../src/main/image/setup.js'

const MEMORY_PY = fs.readFileSync(path.join(import.meta.dirname, '..', 'image-engine', 'image_engine', 'memory.py'), 'utf8')
const GIB = 2 ** 30

/** `NAME = (("full", 23 * GIB), ...)` from memory.py, as [plan, GiB] pairs. */
function pyThresholds(name: string): [string, number][] {
  const match = new RegExp(`^${name} = \\((.*)\\)\\r?$`, 'm').exec(MEMORY_PY)
  if (!match?.[1]) throw new Error(`${name} not found in memory.py`)
  return [...match[1].matchAll(/\("(\w+)", ([\d.]+) \* GIB\)/g)].map((m) => [m[1]!, Number(m[2])])
}

// What nvidia-smi and the OS report for cards and PCs sold as these sizes.
const gpu = (memoryMiB: number): GpuInfo => ({ name: 'NVIDIA Test GPU', memoryMiB, driver: '580.97' })
const CARD = { 4: 4096, 6: 6138, 8: 8188, 10: 10240, 12: 12282, 16: 16376, 24: 24564 } as const
const RAM = { 16: 15.7 * GIB, 32: 31.7 * GIB, 64: 63.8 * GIB } as const
const pc = (card: keyof typeof CARD | null, ram: keyof typeof RAM): HardwareInfo => ({
  gpu: card === null ? null : gpu(CARD[card]),
  ramBytes: RAM[ram]
})
const fit = (id: ModelId, hw: HardwareInfo) => modelFit(catalogModel(id)!, hw)

describe('hardware fit', () => {
  it('uses the same Automatic thresholds as the engine', () => {
    expect(pyThresholds('AUTO_THRESHOLDS')).toEqual(IMAGE_AUTO_GIB)
    expect(pyThresholds('VIDEO_AUTO_THRESHOLDS')).toEqual(VIDEO_AUTO_GIB)
    expect(MEMORY_PY).toMatch(new RegExp(`^RAM_HEADROOM = ${RAM_HEADROOM_GIB} \\* GIB\\r?$`, 'm'))
  })

  it('knows what every downloadable model needs', () => {
    expect(modelsWithNeeds().map((m) => m.id)).toEqual(MODEL_CATALOG.filter((m) => !m.part).map((m) => m.id))
  })

  it('names the mode Automatic picks on a 12 GB card with 32 GB of RAM', () => {
    const hw = pc(12, 32)
    expect(fit('image', hw)).toMatchObject({ fit: 'good', plan: 'balanced', note: 'Runs here in Balanced mode.' })
    expect(fit('video-fast', hw)).toMatchObject({ fit: 'good', plan: 'balanced' })
    expect(fit('caption', hw)).toMatchObject({ fit: 'good', plan: null, note: 'Fits on this GPU.' })
  })

  it('picks the faster modes on bigger cards', () => {
    expect(fit('image', pc(16, 32))?.plan).toBe('quality')
    expect(fit('image', pc(24, 64))?.plan).toBe('full')
    expect(fit('video', pc(24, 64))?.plan).toBe('quality')
  })

  it('calls Low VRAM slow, and an 8 GB card still runs video in it', () => {
    expect(fit('video-fast', pc(8, 64))).toMatchObject({ fit: 'slow', plan: 'low', note: 'Runs here only in Low VRAM mode, the slowest.' })
    expect(fit('image', pc(8, 32))).toMatchObject({ fit: 'slow', plan: 'low' })
    expect(fit('image', pc(6, 32))).toMatchObject({ fit: 'slow', plan: 'low' })
  })

  it('says what does not fit, and why', () => {
    expect(fit('video-fast', pc(6, 32))).toMatchObject({ fit: 'no', plan: null })
    expect(fit('video-fast', pc(6, 32))?.note).toBe('Needs a GPU with at least 8 GB of VRAM, even in Low VRAM mode; this one has 6 GB.')
    expect(fit('image', pc(4, 32))?.fit).toBe('no')
    expect(fit('caption', pc(8, 32))?.note).toBe('Needs a GPU with at least 10 GB of VRAM; this one has 8 GB.')
    expect(fit('caption', pc(10, 32))?.fit).toBe('good')
    expect(fit('image', pc(null, 32))).toEqual({ fit: 'no', plan: null, note: 'Needs an NVIDIA GPU, and none was found.' })
  })

  it('warns when the mode wants more RAM than there is', () => {
    // Video in Balanced keeps the text encoder and the parked transformer in RAM: about 24 GB with headroom.
    const short = fit('video-fast', pc(12, 16))
    expect(short).toMatchObject({ fit: 'slow', plan: 'balanced' })
    expect(short?.note).toBe('Runs here in Balanced mode. It wants about 24 GB of RAM and this PC has 16 GB, so expect Windows to swap to disk.')
    // A PC sold as 16 GB is enough for the image model in Balanced (8 GB text encoder plus headroom).
    expect(fit('image', pc(12, 16))?.fit).toBe('good')
  })

  it('recommends the catalog picks only where they run well', () => {
    const image = catalogModel('image')!
    const original = catalogModel('video')!
    expect(isRecommendedHere(image, fit('image', pc(12, 32)))).toBe(true)
    expect(isRecommendedHere(image, fit('image', pc(8, 32)))).toBe(false)
    expect(isRecommendedHere(image, null)).toBe(true)
    // The 50-step model is never the recommendation, even where it fits well.
    expect(isRecommendedHere(original, fit('video', pc(24, 64)))).toBe(false)
  })

  it('describes the computer in one line', () => {
    expect(describeHardware(pc(12, 32))).toBe('NVIDIA Test GPU · 12 GB VRAM · 32 GB RAM')
    expect(describeHardware(pc(null, 16))).toBe('No NVIDIA GPU found · 16 GB RAM')
  })
})

describe('hardware probe', () => {
  it('asks nvidia-smi for the largest GPU and keeps it once found', async () => {
    let calls = 0
    const run: Runner = async () => {
      calls++
      return { code: 0, stdout: 'NVIDIA Small, 4096, 580.97\nNVIDIA Large, 12282, 580.97\n' }
    }
    const probe = new HardwareProbe(run, () => 32 * GIB)
    expect(await probe.info()).toEqual({ gpu: { name: 'NVIDIA Large', memoryMiB: 12282, driver: '580.97' }, ramBytes: 32 * GIB })
    await probe.info()
    expect(calls).toBe(1)
  })

  it('asks again while no GPU has been found', async () => {
    let calls = 0
    const run: Runner = async () => {
      calls++
      return { code: null, stdout: '' }
    }
    const probe = new HardwareProbe(run, () => 16 * GIB)
    expect((await probe.info()).gpu).toBeNull()
    await probe.info()
    expect(calls).toBe(2)
  })
})
