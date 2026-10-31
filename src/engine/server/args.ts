import type { KvCacheType } from '@shared/config.js'

/**
 * How llama-server is started.
 *
 * Everything about where it listens is fixed here, not left to defaults: the
 * loopback address only, a port the studio picked as free, its web page off.
 * The key the studio uses to talk to it goes in the environment (LLAMA_API_KEY),
 * not on the command line where any process on the machine could list it.
 */
export interface ServerLaunch {
  modelPath: string
  port: number
  contextSize: number
  kvCache: KvCacheType
  /** Null puts every layer on the GPU and lets the server fit what it can. */
  gpuLayers: number | null
  threads: number | null
  /** Whether the model is a mixture of experts; the expert placement flags only make sense then. */
  mixtureOfExperts: boolean
  /** Layers whose experts stay on the CPU; null is all of them. */
  expertsOnCpu: number | null
}

export function serverArgs(launch: ServerLaunch): string[] {
  const args = [
    '--model',
    launch.modelPath,
    '--host',
    '127.0.0.1',
    '--port',
    String(launch.port),
    '--no-webui',
    // No network of its own: never fetches a model or anything else.
    '--offline',
    // Nothing for a web page in a browser to reach with the user's credentials,
    // and no slot monitoring; the key is what lets the studio in.
    '--no-cors-credentials',
    '--no-slots',
    // The file's own chat template, which is what makes tool calls and the
    // thinking switch work for the model families that have them.
    '--jinja',
    // The thought comes back apart from the reply, as reasoning_content.
    '--reasoning-format',
    'deepseek',
    '--ctx-size',
    String(launch.contextSize),
    '--n-gpu-layers',
    launch.gpuLayers === null ? 'all' : String(launch.gpuLayers),
    '--flash-attn',
    'on'
  ]
  if (launch.kvCache === 'q8_0') args.push('--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0')
  if (launch.threads !== null) args.push('--threads', String(launch.threads))
  if (launch.mixtureOfExperts) {
    if (launch.expertsOnCpu === null) args.push('--cpu-moe')
    else if (launch.expertsOnCpu > 0) args.push('--n-cpu-moe', String(launch.expertsOnCpu))
  }
  return args
}

/**
 * The server's environment: the studio's, less anything llama.cpp would read
 * as a setting. llama-server takes every option from a LLAMA_ARG_* variable as
 * well as from its command line, and some of those (a model repository, a
 * model URL) make it download. Only the key the studio made is added; the
 * command line already says --offline.
 */
export function serverEnv(base: NodeJS.ProcessEnv, apiKey: string): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {}
  for (const [key, value] of Object.entries(base)) {
    if (/^(LLAMA_|HF_|HUGGINGFACE_)/i.test(key)) continue
    env[key] = value
  }
  env.LLAMA_API_KEY = apiKey
  env.HF_HUB_OFFLINE = '1'
  return env
}

/**
 * The devices `llama-server --list-devices` reports, which it prints without
 * loading anything:
 *
 *   Available devices:
 *     CUDA0: NVIDIA GeForce RTX 4070 (12282 MiB, 11200 MiB free)
 */
export function parseDeviceList(text: string): { backend: string | null; devices: string[] } {
  const devices: string[] = []
  let backend: string | null = null
  for (const line of text.split(/\r?\n/)) {
    const m = /^\s*([A-Za-z]+)\d+:\s*(.+?)\s*(?:\(\d+ MiB.*\))?\s*$/.exec(line)
    if (!m || !m[1] || !m[2]) continue
    backend ??= m[1].toLowerCase()
    devices.push(m[2])
  }
  return { backend, devices }
}

/** What the server says about itself on stderr, read for the model menu. */
export interface ServerLog {
  backend: string | null
  devices: string[]
  /** "offloaded 49/49 layers to GPU" */
  gpuLayers: number | null
  totalLayers: number | null
}

export function emptyServerLog(): ServerLog {
  return { backend: null, devices: [], gpuLayers: null, totalLayers: null }
}

/** Fold one line of the server's output into what is known so far. */
export function readServerLine(log: ServerLog, line: string): void {
  if (log.backend === null) {
    if (/ggml_cuda_init/.test(line)) log.backend = 'cuda'
    else if (/ggml_vulkan/.test(line)) log.backend = 'vulkan'
    else if (/ggml_metal/.test(line)) log.backend = 'metal'
  }
  // "  Device 0: NVIDIA GeForce RTX 4070, compute capability 8.9, VMM: yes"
  const device = /^\s*Device \d+: ([^,]+?)(?:,|$)/.exec(line)
  if (device && device[1] && !log.devices.includes(device[1].trim())) log.devices.push(device[1].trim())
  const layers = /offloaded (\d+)\/(\d+) layers to GPU/.exec(line)
  if (layers) {
    log.gpuLayers = Number(layers[1])
    log.totalLayers = Number(layers[2])
  }
}
