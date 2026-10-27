/**
 * Load a GGUF and make it call one tool, outside the application.
 *
 * This answers three questions the UI cannot answer quickly, and answers them
 * in the order they can fail:
 *
 *   1. Does the bundled llama.cpp recognise this architecture at all?
 *   2. Did the weights land on the GPU, or silently fall back to the CPU?
 *   3. Will the model emit a well-formed tool call under a GBNF grammar?
 *
 * The third is the one that decides whether this whole project is usable. A
 * model that chats well and cannot reliably call a function is of no use here.
 *
 *   node scripts/smoke-model.mjs <model.gguf> [contextSize] [gpu]
 *
 * `gpu` forces a backend ("cuda", "vulkan", "false" for CPU). Left off, the
 * library picks, which is not always the fastest choice available.
 */
import { fileURLToPath } from 'node:url'
import {
  defineChatSessionFunction,
  getLlama,
  LlamaChatSession
} from 'node-llama-cpp'

const modelPath = process.argv[2]
const contextSize = Number(process.argv[3] ?? 8192)
const forcedGpu = process.argv[4]

if (!modelPath) {
  console.error('usage: node scripts/smoke-model.mjs <model.gguf> [contextSize] [gpu]')
  process.exit(1)
}

const started = Date.now()
const since = () => `${((Date.now() - started) / 1000).toFixed(1)}s`

console.log(`[${since()}] initialising llama…`)
const llama = await getLlama(
  forcedGpu === undefined
    ? {}
    : { gpu: forcedGpu === 'false' ? false : forcedGpu }
)
console.log(`[${since()}] gpu backend: ${llama.gpu === false ? 'none (cpu)' : llama.gpu}`)

console.log(`[${since()}] loading ${modelPath}`)
let lastPercent = -1
const model = await llama.loadModel({
  modelPath,
  onLoadProgress(fraction) {
    const percent = Math.floor(fraction * 100)
    if (percent >= lastPercent + 25) {
      lastPercent = percent
      console.log(`[${since()}]   ${percent}%`)
    }
  }
})

console.log(`[${since()}] loaded`)
console.log(`  name        : ${model.filename}`)
console.log(`  layers      : ${model.gpuLayers} of ${model.fileInfo.metadata[model.fileInfo.metadata.general.architecture]?.block_count ?? '?'} on the GPU`)
console.log(`  train ctx   : ${model.trainContextSize}`)
console.log(`  size        : ${(model.size / 1024 ** 3).toFixed(2)} GB`)

if (model.gpuLayers === 0) {
  console.log('\n  ⚠ Nothing was offloaded to the GPU. Inference will be slow.')
}

const context = await model.createContext({ contextSize })
console.log(`[${since()}] context: ${context.contextSize} tokens`)

// One tool, deliberately trivial, so a failure is unambiguously about the
// model's ability to call it rather than about the task being hard.
let called = null
const session = new LlamaChatSession({
  contextSequence: context.getSequence(),
  systemPrompt:
    'You are a careful assistant with access to the user\u2019s files. Use the tools available to you rather than guessing.'
})

const functions = {
  list_directory: defineChatSessionFunction({
    description: 'List the files and folders directly inside a directory.',
    params: {
      type: 'object',
      properties: { path: { type: 'string', description: 'Absolute path of the directory.' } },
      required: ['path'],
      additionalProperties: false
    },
    handler({ path }) {
      called = path
      console.log(`[${since()}] >>> model called list_directory(${JSON.stringify(path)})`)
      return { entries: [{ name: 'report.txt', kind: 'file', bytes: 120 }], truncated: false }
    }
  })
}

console.log(`[${since()}] prompting…`)
const promptStarted = Date.now()
const answer = await session.prompt('What files are in C:\\work? Use your tool to find out.', {
  functions,
  maxTokens: 400
})

const promptSeconds = (Date.now() - promptStarted) / 1000
console.log(`\n[${since()}] answer, ${promptSeconds.toFixed(1)}s after the prompt:\n${answer}\n`)
console.log(called ? `✅ tool was called with: ${called}` : '❌ the model never called the tool')

await context.dispose()
await model.dispose()
process.exit(called ? 0 : 2)

// Keep the module-detection happy when imported rather than run.
export const __main = fileURLToPath(import.meta.url)
