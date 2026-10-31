#!/usr/bin/env node
/**
 * Loads the tiny made-up MoE model (tiny-moe-gguf.mjs) through node-llama-cpp
 * and prints, as JSON, what the addon did with it: whether it is the studio's
 * build, where llama.cpp put the weights, and the first tokens it generates
 * greedily. Run against the stock addon and the patched one, the tokens must
 * match; with --experts cpu, the expert tensors must be the ones moved to system
 * memory.
 *
 *   node scripts/native/addon-smoke.mjs [--gpu cuda|false] [--experts gpu|cpu]
 *
 * No real model is involved: the file is written to the temp folder and is a
 * few hundred kilobytes of fixed-seed noise.
 */
import os from 'node:os'
import path from 'node:path'
import { getLlama, LlamaLogLevel } from 'node-llama-cpp'
import { writeTinyMoeGguf } from './tiny-moe-gguf.mjs'

const arg = (name, fallback) => {
  const i = process.argv.indexOf(`--${name}`)
  return i === -1 ? fallback : process.argv[i + 1]
}
const gpu = arg('gpu', 'cuda')
const experts = arg('experts', 'gpu')

const file = writeTinyMoeGguf(path.join(os.tmpdir(), 'local-ai-studio-tiny-moe.gguf'))
const lines = []
const llama = await getLlama({
  build: 'never',
  skipDownload: true,
  gpu: gpu === 'false' ? false : gpu,
  logLevel: LlamaLogLevel.debug,
  logger: (_level, message) => lines.push(message)
})
const features = typeof llama._bindings.getStudioFeatures === 'function' ? llama._bindings.getStudioFeatures() : []

const model = await llama.loadModel({
  modelPath: file,
  gpuLayers: 'max',
  ignoreMemorySafetyChecks: true,
  ...(experts === 'cpu' ? { tensorBufferOverrides: [{ pattern: '\\.ffn_(up|down|gate|gate_up)_(ch|)exps', target: 'cpu' }] } : {})
})
const context = await model.createContext({ contextSize: 128, ignoreMemorySafetyChecks: true })
const sequence = context.getSequence()
const tokens = []
for await (const token of sequence.evaluate(model.tokenize('hello world'), { temperature: 0 })) {
  tokens.push(token)
  if (tokens.length >= 24) break
}
await context.dispose()
await model.dispose()

const text = lines.join('\n')
const result = {
  gpu: llama.gpu,
  features,
  buffers: [...text.matchAll(/^\S*\s*(\S+) model buffer size =\s*([\d.]+) MiB/gm)].map((m) => `${m[1]} ${m[2]} MiB`),
  overridden: [...text.matchAll(/tensor (\S+) \(.*\) buffer type overridden to (\S+)/g)].map((m) => `${m[1]} -> ${m[2]}`),
  offloaded: /offloaded (\d+\/\d+) layers to GPU/.exec(text)?.[1] ?? null,
  tokens
}
console.log(JSON.stringify(result, null, 2))
await llama.dispose()
