import { describe, expect, it } from 'vitest'
import { discreteDeviceIndices, isIntegratedGpu } from '../src/shared/gpu-devices.js'

describe('telling integrated GPUs from discrete ones by name', () => {
  it('knows the integrated GPUs that share system memory', () => {
    for (const name of [
      'Intel(R) Graphics',
      'Intel(R) UHD Graphics 770',
      'Intel(R) Iris(R) Xe Graphics',
      'Intel(R) Arc(TM) Graphics',
      'AMD Radeon(TM) Graphics',
      'AMD Radeon 780M Graphics',
      'AMD Radeon Vega 8 Graphics',
      'llvmpipe (LLVM 17.0.6, 256 bits)',
      'Microsoft Basic Render Driver'
    ]) {
      expect(isIntegratedGpu(name), name).toBe(true)
    }
  })

  it('knows the discrete ones, including Intel and AMD cards and laptop parts', () => {
    for (const name of [
      'NVIDIA GeForce RTX 4060 Laptop GPU',
      'NVIDIA GeForce GTX 1660',
      'Intel(R) Arc(TM) A770 Graphics',
      'Intel(R) Arc(TM) B580 Graphics',
      'AMD Radeon RX 7900 XTX',
      'AMD Radeon RX 7600M XT',
      'AMD Radeon Pro W7800'
    ]) {
      expect(isIntegratedGpu(name), name).toBe(false)
    }
  })
})

describe('which devices to keep', () => {
  it('leaves the integrated GPU out when a discrete one is present, as on most laptops', () => {
    expect(discreteDeviceIndices(['Intel(R) Graphics', 'NVIDIA GeForce RTX 4060 Laptop GPU'])).toEqual([1])
    expect(discreteDeviceIndices(['NVIDIA GeForce RTX 4090', 'AMD Radeon(TM) Graphics', 'NVIDIA GeForce RTX 4090'])).toEqual([0, 2])
  })

  it('changes nothing when all are discrete, all are integrated, or there is one', () => {
    expect(discreteDeviceIndices(['NVIDIA GeForce RTX 4090', 'NVIDIA GeForce RTX 3090'])).toBeNull()
    expect(discreteDeviceIndices(['Intel(R) UHD Graphics 620'])).toBeNull()
    expect(discreteDeviceIndices(['NVIDIA GeForce RTX 4060 Laptop GPU'])).toBeNull()
    expect(discreteDeviceIndices([])).toBeNull()
  })
})
