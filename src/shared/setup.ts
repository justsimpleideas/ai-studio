import type { MediaKind } from './image.js'

/**
 * Setting up the image side from inside the studio: the Python environment the
 * engine runs in, and the model weights it loads.
 *
 * Nothing here is bundled with the app. The engine's packages are installed by
 * pip on the user's machine, from the official PyTorch and PyPI indexes, pinned
 * to the audited versions; the weights come straight from each model's own
 * repository, pinned to a commit and checked file by file. The studio only
 * starts those programs and reports what they say.
 */

// ---------------------------------------------------------------------------
// Supported models

export type ModelId = 'image' | 'video-fast' | 'video'

export interface CatalogModel {
  id: ModelId
  kind: MediaKind
  label: string
  /** One line on what to expect, shown under the label. */
  detail: string
  /** Where the weights come from, shown before anything is downloaded. */
  repo: string
  revision: string
  license: string
  licenseUrl: string
  /** Total size of the files fetch_model.py downloads. */
  bytes: number
  /** Folder name created inside the chosen models folder. */
  folder: string
  recommended?: boolean
}

const APACHE_2 = { license: 'Apache License 2.0', licenseUrl: 'https://www.apache.org/licenses/LICENSE-2.0' }

/** Must agree with image-engine/scripts/fetch_model.py (checked by tests/setup.test.ts). */
export const MODEL_CATALOG: readonly CatalogModel[] = [
  {
    id: 'image',
    kind: 'image',
    label: 'Image model · 6B, 8 steps',
    detail: 'About 16 s per 1360×768 image on a 12 GB laptop GPU in Balanced mode.',
    repo: 'Tongyi-MAI/Z-Image-Turbo',
    revision: 'f332072aa78be7aecdf3ee76d5c247082da564a6',
    ...APACHE_2,
    bytes: 32_848_305_533,
    folder: 'image-model',
    recommended: true
  },
  {
    id: 'video-fast',
    kind: 'video',
    label: 'Video model · 5B, 3-step distilled',
    detail: 'About 4½ minutes for 5 s of 720p video on a 12 GB laptop GPU.',
    repo: 'FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers',
    revision: '3e187042a324f6f5fb68fd22110a78725253de8f',
    ...APACHE_2,
    bytes: 24_201_770_562,
    folder: 'video-model-fast',
    recommended: true
  },
  {
    id: 'video',
    kind: 'video',
    label: 'Video model · 5B, 50 steps',
    detail: 'The original model: about ten times slower than the distilled one (44 minutes for 5 s of 720p on the same GPU).',
    repo: 'Wan-AI/Wan2.2-TI2V-5B-Diffusers',
    revision: 'b8fff7315c768468a5333511427288870b2e9635',
    ...APACHE_2,
    bytes: 34_201_418_400,
    folder: 'video-model'
  }
]

export function catalogModel(id: unknown): CatalogModel | null {
  return MODEL_CATALOG.find((m) => m.id === id) ?? null
}

// ---------------------------------------------------------------------------
// Engine setup

export type SetupPhase = 'idle' | 'checking' | 'creating' | 'torch' | 'packages' | 'verifying' | 'done' | 'failed' | 'cancelled'

export interface EngineSetupState {
  phase: SetupPhase
  /** What is happening, or why it stopped. */
  message: string
  /** The newest lines of pip's output, for anyone who wants to see them. */
  log: string[]
  /** The environment's interpreter, once created. */
  python: string | null
  /** What the last check found, e.g. "NVIDIA GeForce RTX 4070 · 12 GB · driver 580.97". */
  gpu: string | null
}

export const SETUP_STEPS: readonly { phase: SetupPhase; label: string }[] = [
  { phase: 'checking', label: 'Check Python and the GPU' },
  { phase: 'creating', label: 'Create the environment' },
  { phase: 'torch', label: 'Install PyTorch (about 3 GB)' },
  { phase: 'packages', label: 'Install the other packages' },
  { phase: 'verifying', label: 'Check that the GPU works' }
]

/** Python versions the pinned packages publish wheels for. */
export const SUPPORTED_PYTHON_MINORS: readonly number[] = [12, 13, 14]

export const PYTORCH_INDEX = 'https://download.pytorch.org/whl/cu128'

/** Printed by every candidate interpreter; parsed by parsePythonProbe. */
export const PYTHON_PROBE =
  'import sys, json; print(json.dumps({"major": sys.version_info[0], "minor": sys.version_info[1], "executable": sys.executable, "bits64": sys.maxsize > 2**32}))'

export interface PythonProbe {
  major: number
  minor: number
  executable: string
  bits64: boolean
}

export function parsePythonProbe(stdout: string): PythonProbe | null {
  const line = stdout.trim().split(/\r?\n/).pop()
  if (!line) return null
  try {
    const v = JSON.parse(line) as Partial<PythonProbe>
    if (typeof v.major !== 'number' || typeof v.minor !== 'number' || typeof v.executable !== 'string' || typeof v.bits64 !== 'boolean') {
      return null
    }
    return { major: v.major, minor: v.minor, executable: v.executable, bits64: v.bits64 }
  } catch {
    return null
  }
}

/** Null if this interpreter can host the engine, otherwise why not. */
export function pythonProblem(p: PythonProbe): string | null {
  if (!p.bits64) return `Python ${p.major}.${p.minor} is 32-bit; PyTorch needs 64-bit Python.`
  if (p.major !== 3 || !SUPPORTED_PYTHON_MINORS.includes(p.minor)) {
    return `Python ${p.major}.${p.minor} is not supported; use 3.${SUPPORTED_PYTHON_MINORS[0]} to 3.${SUPPORTED_PYTHON_MINORS.at(-1)}.`
  }
  return null
}

export interface GpuInfo {
  name: string
  memoryMiB: number
  driver: string
}

/** Output of `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits`. */
export function parseNvidiaSmi(stdout: string): GpuInfo[] {
  const gpus: GpuInfo[] = []
  for (const line of stdout.split(/\r?\n/)) {
    const parts = line.split(',').map((s) => s.trim())
    if (parts.length < 3) continue
    const [name, memory, driver] = parts as [string, string, string]
    const memoryMiB = Number(memory)
    if (!name || !Number.isFinite(memoryMiB)) continue
    gpus.push({ name, memoryMiB, driver })
  }
  return gpus
}

export function describeGpu(gpu: GpuInfo): string {
  return `${gpu.name} · ${Math.round(gpu.memoryMiB / 1024)} GB · driver ${gpu.driver}`
}

export interface PipSource {
  /** Install from a folder of wheels instead of the internet (tests, offline machines). */
  wheelhouse?: string | null
}

const PIP_COMMON = ['-m', 'pip', 'install', '--disable-pip-version-check', '--no-input', '--progress-bar', 'off', '--only-binary=:all:']

/**
 * PyTorch alone, without its dependencies, from the PyTorch index: its CUDA builds
 * are published there and not on PyPI. Everything else comes from PyPI in the next
 * step, so no package is ever looked up in two places.
 */
export function torchInstallArgs(constraints: string, source: PipSource = {}): string[] {
  const where = source.wheelhouse ? ['--no-index', '--find-links', source.wheelhouse] : ['--index-url', PYTORCH_INDEX]
  return [...PIP_COMMON, ...where, '--no-deps', '-c', constraints, 'torch']
}

/** The engine's requirements, plus PyTorch's own dependencies, from PyPI. */
export function packagesInstallArgs(requirements: string, constraints: string, source: PipSource = {}): string[] {
  const where = source.wheelhouse ? ['--no-index', '--find-links', source.wheelhouse] : []
  return [...PIP_COMMON, ...where, '-c', constraints, '-r', requirements, 'torch']
}

/** Printed by the finished environment; parsed by parseVerify. */
export const VERIFY_SCRIPT =
  'import json, torch, diffusers; c = torch.cuda.is_available(); print(json.dumps({"torch": torch.__version__, "diffusers": diffusers.__version__, "cuda": c, "device": torch.cuda.get_device_name(0) if c else None}))'

export interface VerifyResult {
  torch: string
  diffusers: string
  cuda: boolean
  device: string | null
}

export function parseVerify(stdout: string): VerifyResult | null {
  const line = stdout.trim().split(/\r?\n/).pop()
  if (!line) return null
  try {
    const v = JSON.parse(line) as Partial<VerifyResult>
    if (typeof v.torch !== 'string' || typeof v.diffusers !== 'string' || typeof v.cuda !== 'boolean') return null
    return { torch: v.torch, diffusers: v.diffusers, cuda: v.cuda, device: typeof v.device === 'string' ? v.device : null }
  } catch {
    return null
  }
}

export const SETUP_LOG_LINES = 300

export function appendLog(log: string[], text: string): string[] {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0)
  if (lines.length === 0) return log
  return [...log, ...lines].slice(-SETUP_LOG_LINES)
}

// ---------------------------------------------------------------------------
// Model downloads

export type DownloadPhase = 'idle' | 'starting' | 'downloading' | 'done' | 'failed' | 'cancelled'

export interface ModelDownloadState {
  phase: DownloadPhase
  id: ModelId | null
  /** The folder being filled. */
  dest: string | null
  /** Bytes in place and verified, plus the bytes of the file in flight. */
  doneBytes: number
  totalBytes: number
  /** The file being worked on and what is happening to it. */
  file: string | null
  activity: 'listing' | 'downloading' | 'verifying' | 'linking' | null
  bytesPerSecond: number | null
  message: string
}

export const IDLE_DOWNLOAD: ModelDownloadState = {
  phase: 'idle',
  id: null,
  dest: null,
  doneBytes: 0,
  totalBytes: 0,
  file: null,
  activity: null,
  bytesPerSecond: null,
  message: ''
}

/** One line of `fetch_model.py --json`. */
export type FetchEvent =
  | { event: 'plan'; repo: string; revision: string; files: number; bytes: number }
  | { event: 'file'; path: string; size: number; action: 'fetch' | 'ok' | 'stale' | 'linked' | 'copied' }
  | { event: 'progress'; path: string; done: number; size: number }
  | { event: 'verify'; path: string; size: number }
  | { event: 'retry'; path: string; message: string; wait: number }
  | { event: 'done'; files: number; bytes: number }
  | { event: 'error'; message: string }

export function parseFetchEvent(line: string): FetchEvent | null {
  try {
    const value = JSON.parse(line) as { event?: unknown }
    return value && typeof value === 'object' && typeof value.event === 'string' ? (value as FetchEvent) : null
  } catch {
    return null
  }
}

/**
 * Turns fetch_model.py's events into a download state. Files are verified (or
 * fetched) one after another, so the total done is every file marked ok plus the
 * one in flight.
 */
export class DownloadTracker {
  private verified = 0
  private inFlight = 0
  private samples: { at: number; bytes: number }[] = []
  state: ModelDownloadState

  constructor(id: ModelId, dest: string, totalBytes: number) {
    this.state = { ...IDLE_DOWNLOAD, phase: 'starting', id, dest, totalBytes, activity: 'listing', message: 'Listing the files…' }
  }

  feed(e: FetchEvent, now = Date.now()): ModelDownloadState {
    const s = this.state
    switch (e.event) {
      case 'plan':
        this.state = { ...s, phase: 'downloading', totalBytes: e.bytes, message: `${e.files} files` }
        break
      case 'file':
        if (e.action === 'ok') {
          this.verified += e.size
          this.inFlight = 0
        } else if (e.action === 'fetch' || e.action === 'stale') {
          this.inFlight = 0
        }
        this.state = {
          ...s,
          file: e.path,
          activity: e.action === 'linked' || e.action === 'copied' ? 'linking' : e.action === 'ok' ? s.activity : 'downloading'
        }
        break
      case 'progress':
        this.inFlight = e.done
        this.samples = [...this.samples, { at: now, bytes: this.verified + e.done }].filter((x) => now - x.at <= 10_000)
        this.state = { ...s, file: e.path, activity: 'downloading', bytesPerSecond: this.rate(), message: '' }
        break
      case 'verify':
        this.inFlight = e.size
        this.state = { ...s, file: e.path, activity: 'verifying', bytesPerSecond: null }
        break
      case 'retry':
        this.state = { ...s, message: `Connection problem (${e.message}); retrying in ${e.wait} s.` }
        break
      case 'done':
        this.verified = e.bytes
        this.inFlight = 0
        this.state = { ...s, phase: 'done', file: null, activity: null, bytesPerSecond: null, message: 'Downloaded and verified.' }
        break
      case 'error':
        this.state = { ...s, phase: 'failed', activity: null, bytesPerSecond: null, message: e.message }
        break
    }
    this.state = { ...this.state, doneBytes: Math.min(this.verified + this.inFlight, this.state.totalBytes || Infinity) }
    return this.state
  }

  private rate(): number | null {
    const first = this.samples[0]
    const last = this.samples.at(-1)
    if (!first || !last || last.at - first.at < 2000) return null
    return (last.bytes - first.bytes) / ((last.at - first.at) / 1000)
  }
}
