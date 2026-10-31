import fs from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'
import {
  ADDON_FEATURE,
  JS_MARKER,
  NODE_LLAMA_CPP_VERSION,
  patchAddonEntry,
  patchAddonModelHeader,
  patchAddonModelSource,
  patchLlamaModelJs
} from '../scripts/native/addon-patch.mjs'
import { moduleDefinition, readExports, readImports } from '../scripts/native/pe.mjs'
import { ADDON_FEATURE as ENGINE_FEATURE, addonPlacesTensors, EXPERT_TENSORS, expertOverrides, planExperts } from '../src/engine/expert-placement.js'

const NLC = path.join(import.meta.dirname, '..', 'node_modules', 'node-llama-cpp')
const read = (...parts: string[]): string => fs.readFileSync(path.join(NLC, ...parts), 'utf8')
const installed = JSON.parse(read('package.json')).version as string

describe('the patch to node-llama-cpp', () => {
  it('is written for the installed release', () => {
    expect(installed).toBe(NODE_LLAMA_CPP_VERSION)
  })

  it('adds the option to the addon at the anchors it expects', () => {
    const header = patchAddonModelHeader(read('llama', 'addon', 'AddonModel.h'))
    expect(header).toContain('std::vector<llama_model_tensor_buft_override> tensor_buft_overrides;')
    const source = patchAddonModelSource(read('llama', 'addon', 'AddonModel.cpp'))
    expect(source).toContain('options.Has("tensorBufferOverrides")')
    expect(source).toContain('ggml_backend_cpu_buffer_type()')
    // The list is ended with an empty entry, as llama.cpp reads it up to a null pattern.
    expect(source).toContain('tensor_buft_overrides.push_back({ nullptr, nullptr });')
    // Read before the progress callback, inside the options block.
    expect(source.indexOf('tensorBufferOverrides')).toBeLessThan(source.indexOf('options.Has("onLoadProgress")'))
    const entry = patchAddonEntry(read('llama', 'addon', 'addon.cpp'))
    expect(entry).toContain(`"${ADDON_FEATURE}"`)
    expect(entry).toContain('Napi::PropertyDescriptor::Function("getStudioFeatures", addonGetStudioFeatures)')
  })

  it('passes the option through the library, once', () => {
    const js = read('dist', 'evaluator', 'LlamaModel', 'LlamaModel.js')
    // The file may already be patched in this checkout (scripts/native/addon.mjs install).
    const stock = js.includes(JS_MARKER) ? null : js
    if (stock !== null) {
      const patched = patchLlamaModelJs(stock)
      expect(patched).toContain('metadataOverrides, tensorBufferOverrides }')
      expect(patched).toContain(`${JS_MARKER}\n            tensorBufferOverrides,`)
      expect(patchLlamaModelJs(patched)).toBe(patched)
    } else {
      expect(patchLlamaModelJs(js)).toBe(js)
    }
  })

  it('refuses code that moved rather than patching the wrong place', () => {
    expect(() => patchAddonModelSource('int main() {}')).toThrow(/anchor not found/)
    const twice = '        if (options.Has("onLoadProgress")) {\n'.repeat(2)
    expect(() => patchAddonModelSource(twice)).toThrow(/more than once/)
  })

  it('names the feature the engine looks for', () => {
    expect(ENGINE_FEATURE).toBe(ADDON_FEATURE)
  })
})

describe.skipIf(process.platform !== 'win32')('reading a DLL’s exports and imports', () => {
  const node = fs.readFileSync(process.execPath)

  it('finds N-API in node.exe as code', () => {
    const exports = readExports(node)
    const napi = exports.find((e) => e.name === 'napi_create_object')
    expect(napi).toEqual({ name: 'napi_create_object', data: false })
  })

  it('lists imported DLLs', () => {
    expect(readImports(node).normal.map((n) => n.toUpperCase())).toContain('KERNEL32.DLL')
  })

  it('writes a definition file the librarian reads, data marked as such', () => {
    const def = moduleDefinition('ggml-base.dll', [
      { name: 'ggml_init', data: false },
      { name: '?table@@3PAHA', data: true }
    ])
    expect(def).toBe('LIBRARY "ggml-base.dll"\r\nEXPORTS\r\n    ggml_init\r\n    ?table@@3PAHA DATA\r\n')
  })

  it('rejects a file that is not a PE image', () => {
    expect(() => readExports(Buffer.from('not a dll at all, not even close'))).toThrow(/Not a PE file/)
  })
})

describe('keeping experts on the CPU', () => {
  it('uses llama.cpp’s own pattern for every layer', () => {
    expect(expertOverrides(null, 40)).toEqual([{ pattern: EXPERT_TENSORS, target: 'cpu' }])
    expect(expertOverrides(64, 40)).toEqual([{ pattern: EXPERT_TENSORS, target: 'cpu' }])
    const pattern = new RegExp(EXPERT_TENSORS)
    for (const name of ['blk.0.ffn_up_exps.weight', 'blk.9.ffn_down_exps.weight', 'blk.3.ffn_gate_exps.weight', 'blk.1.ffn_gate_up_exps.weight', 'blk.2.ffn_up_chexps.weight']) {
      expect(pattern.test(name), name).toBe(true)
    }
    for (const name of ['blk.0.ffn_up.weight', 'blk.0.ffn_up_shexp.weight', 'blk.0.ffn_gate_inp.weight', 'blk.0.attn_q.weight']) {
      expect(pattern.test(name), name).toBe(false)
    }
  })

  it('counts the first layers only, as --n-cpu-moe does', () => {
    const overrides = expertOverrides(2, 40)
    expect(overrides).toHaveLength(2)
    const matches = (name: string): boolean => overrides.some((o) => new RegExp(o.pattern).test(name))
    expect(matches('blk.0.ffn_up_exps.weight')).toBe(true)
    expect(matches('blk.1.ffn_down_exps.weight')).toBe(true)
    // Layer 10 and 21 must not be caught by the patterns for layers 1 and 2.
    expect(matches('blk.10.ffn_up_exps.weight')).toBe(false)
    expect(matches('blk.21.ffn_up_exps.weight')).toBe(false)
    expect(matches('blk.2.ffn_up_exps.weight')).toBe(false)
    expect(expertOverrides(0, 40)).toEqual([])
  })

  const moe = { experts: 128, layers: 40, expertsOnCpu: null, backend: 'cuda' as const, addonPlaces: true }

  it('places them on CUDA with the studio’s addon', () => {
    expect(planExperts(moe)).toMatchObject({ kind: 'place', label: 'all' })
    expect(planExperts({ ...moe, expertsOnCpu: 24 })).toMatchObject({ kind: 'place', label: 24 })
  })

  it('leaves a dense model, a CPU-only load and "none" alone', () => {
    expect(planExperts({ ...moe, experts: undefined })).toEqual({ kind: 'none' })
    expect(planExperts({ ...moe, experts: 1 })).toEqual({ kind: 'none' })
    expect(planExperts({ ...moe, expertsOnCpu: 0 })).toEqual({ kind: 'none' })
    expect(planExperts({ ...moe, backend: false })).toEqual({ kind: 'none' })
  })

  it('says why when it cannot, instead of loading every expert onto the GPU', () => {
    expect(planExperts({ ...moe, backend: 'vulkan' })).toMatchObject({ kind: 'unavailable', reason: expect.stringMatching(/CUDA/) })
    expect(planExperts({ ...moe, addonPlaces: false })).toMatchObject({ kind: 'unavailable', reason: expect.stringMatching(/addon/) })
    expect(planExperts({ ...moe, layers: undefined })).toMatchObject({ kind: 'unavailable' })
  })

  it('tells the studio’s addon from the stock one', () => {
    expect(addonPlacesTensors({ _bindings: {} })).toBe(false)
    expect(addonPlacesTensors({})).toBe(false)
    expect(addonPlacesTensors({ _bindings: { getStudioFeatures: () => ['somethingElse'] } })).toBe(false)
    expect(addonPlacesTensors({ _bindings: { getStudioFeatures: () => [ADDON_FEATURE] } })).toBe(true)
  })
})
