import { spawn } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import {
  PYTHON_PROBE,
  VERIFY_SCRIPT,
  appendLog,
  describeGpu,
  packagesInstallArgs,
  parseNvidiaSmi,
  parsePythonProbe,
  parseVerify,
  pythonProblem,
  torchInstallArgs,
  type EngineSetupState,
  type PipSource,
  type PythonProbe
} from '@shared/setup.js'

/**
 * Installing the image engine's Python environment from inside the studio.
 *
 * A virtual environment in the studio's data folder, made with a Python already
 * on the machine, then pip: PyTorch's CUDA build from the PyTorch index, the rest
 * from PyPI, every package pinned by image-engine/constraints.txt and binary
 * wheels only (no package runs a build script). Nothing is installed outside
 * that folder, and running it again repairs it.
 */

export interface RunResult {
  code: number | null
  stdout: string
}

export interface RunOptions {
  /** Called with each chunk of stdout and stderr as it arrives. */
  onOutput?: (text: string, stream: 'stdout' | 'stderr') => void
  signal?: AbortSignal
}

export type Runner = (command: string, args: string[], options?: RunOptions) => Promise<RunResult>

export const runProcess: Runner = (command, args, options = {}) =>
  new Promise((resolve) => {
    let stdout = ''
    let child: ReturnType<typeof spawn>
    try {
      child = spawn(command, args, {
        windowsHide: true,
        stdio: ['ignore', 'pipe', 'pipe'],
        env: { ...process.env, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8', PIP_NO_INPUT: '1' }
      })
    } catch {
      resolve({ code: null, stdout: '' })
      return
    }
    const onAbort = (): void => {
      child.kill()
    }
    options.signal?.addEventListener('abort', onAbort, { once: true })
    child.stdout?.on('data', (chunk: Buffer) => {
      const text = chunk.toString('utf8')
      stdout += text
      options.onOutput?.(text, 'stdout')
    })
    child.stderr?.on('data', (chunk: Buffer) => options.onOutput?.(chunk.toString('utf8'), 'stderr'))
    child.on('error', () => resolve({ code: null, stdout }))
    child.on('close', (code) => {
      options.signal?.removeEventListener('abort', onAbort)
      resolve({ code, stdout })
    })
  })

export interface EngineSetupOptions {
  /** Where the environment is created. */
  venvDir: string
  requirements: string
  constraints: string
  source?: PipSource
  run?: Runner
  /** Interpreters to try, in order; the platform's usual ones by default. */
  candidates?: [string, string[]][]
  onState: (state: EngineSetupState) => void
}

export function venvPython(venvDir: string, platform = process.platform): string {
  return platform === 'win32' ? path.join(venvDir, 'Scripts', 'python.exe') : path.join(venvDir, 'bin', 'python')
}

export function pythonCandidates(platform = process.platform): [string, string[]][] {
  if (platform === 'win32') {
    // The py launcher knows every installed version; a bare "python" may be the Store stub.
    return [['py', ['-3.14']], ['py', ['-3.13']], ['py', ['-3.12']], ['python', []], ['python3', []]]
  }
  return [['python3.14', []], ['python3.13', []], ['python3.12', []], ['python3', []], ['python', []]]
}

class SetupStopped extends Error {}

export class EngineSetup {
  private current: EngineSetupState = { phase: 'idle', message: '', log: [], python: null, gpu: null }
  private controller: AbortController | null = null
  private readonly run: Runner

  constructor(private readonly options: EngineSetupOptions) {
    this.run = options.run ?? runProcess
  }

  get state(): EngineSetupState {
    return this.current
  }

  get running(): boolean {
    return this.controller !== null
  }

  cancel(): void {
    this.controller?.abort()
  }

  /** Runs every step; resolves with the environment's interpreter. */
  async install(): Promise<string> {
    if (this.controller) throw new Error('The engine is already being installed.')
    const controller = new AbortController()
    this.controller = controller
    this.set({ phase: 'checking', message: 'Looking for Python 3.12 to 3.14…', log: [], python: null, gpu: null })
    try {
      const python = await this.steps(controller.signal)
      this.set({ phase: 'done', message: 'The image engine is installed.' })
      return python
    } catch (err) {
      if (controller.signal.aborted) {
        this.set({ phase: 'cancelled', message: 'Stopped. Running the install again picks up where it left off.' })
      } else {
        this.set({ phase: 'failed', message: (err as Error).message })
      }
      throw err instanceof SetupStopped || controller.signal.aborted ? new Error(this.current.message) : err
    } finally {
      this.controller = null
    }
  }

  private async steps(signal: AbortSignal): Promise<string> {
    const { venvDir, requirements, constraints, source } = this.options

    const base = await this.findPython(signal)
    const gpu = await this.findGpu(signal)
    this.set({ gpu: gpu ? describeGpu(gpu) : null })
    if (!gpu) {
      throw new SetupStopped(
        'No NVIDIA GPU was found (nvidia-smi did not answer). Image and video generation need an NVIDIA GPU with a current driver.'
      )
    }

    this.set({ phase: 'creating', message: `Creating the environment with Python ${base.major}.${base.minor}…` })
    const python = venvPython(venvDir)
    if (!(await this.works(python, signal))) {
      fs.mkdirSync(path.dirname(venvDir), { recursive: true })
      await this.step(base.executable, ['-m', 'venv', '--clear', venvDir], signal, 'Could not create the Python environment.')
    }
    this.set({ python })

    this.set({ phase: 'torch', message: 'Installing PyTorch with CUDA (about 3 GB)…' })
    await this.step(python, torchInstallArgs(constraints, source), signal, 'PyTorch could not be installed.')

    this.set({ phase: 'packages', message: 'Installing diffusers and the other packages…' })
    await this.step(python, packagesInstallArgs(requirements, constraints, source), signal, 'The packages could not be installed.')
    await this.step(python, ['-m', 'pip', 'check', '--disable-pip-version-check'], signal, 'The installed packages do not fit together.')

    this.set({ phase: 'verifying', message: 'Checking that PyTorch can use the GPU…' })
    const verify = await this.run(python, ['-c', VERIFY_SCRIPT], { signal, onOutput: (t) => this.log(t) })
    if (signal.aborted) throw new SetupStopped()
    const result = parseVerify(verify.stdout)
    if (verify.code !== 0 || !result) throw new SetupStopped('The new environment could not import PyTorch and diffusers.')
    if (!result.cuda) {
      throw new SetupStopped(
        `PyTorch ${result.torch} is installed but cannot use the GPU. Update the NVIDIA driver, then run the install again.`
      )
    }
    this.log(`PyTorch ${result.torch}, diffusers ${result.diffusers}, ${result.device ?? 'GPU'}`)
    return python
  }

  private async findPython(signal: AbortSignal): Promise<PythonProbe> {
    const found: string[] = []
    for (const [command, args] of this.options.candidates ?? pythonCandidates()) {
      const result = await this.run(command, [...args, '-c', PYTHON_PROBE], { signal })
      if (signal.aborted) throw new SetupStopped()
      const probe = result.code === 0 ? parsePythonProbe(result.stdout) : null
      if (!probe) continue
      const problem = pythonProblem(probe)
      if (!problem) {
        this.log(`Using Python ${probe.major}.${probe.minor}: ${probe.executable}`)
        return probe
      }
      found.push(problem)
    }
    const seen = found.length ? ` Found: ${[...new Set(found)].join(' ')}` : ''
    throw new SetupStopped(
      `Python 3.12 to 3.14 (64-bit) was not found.${seen} Install it from python.org, or with "winget install Python.Python.3.13", then run the install again.`
    )
  }

  private async findGpu(signal: AbortSignal) {
    const result = await this.run(
      'nvidia-smi',
      ['--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'],
      { signal }
    )
    if (signal.aborted) throw new SetupStopped()
    if (result.code !== 0) return null
    const gpus = parseNvidiaSmi(result.stdout)
    // The engine uses the first CUDA device; with several, the largest is the likeliest one meant.
    return gpus.sort((a, b) => b.memoryMiB - a.memoryMiB)[0] ?? null
  }

  private async works(python: string, signal: AbortSignal): Promise<boolean> {
    if (!fs.existsSync(python)) return false
    const result = await this.run(python, ['-c', PYTHON_PROBE], { signal })
    const probe = result.code === 0 ? parsePythonProbe(result.stdout) : null
    return probe !== null && pythonProblem(probe) === null
  }

  private async step(command: string, args: string[], signal: AbortSignal, failure: string): Promise<void> {
    this.log(`> ${[path.basename(command), ...args].join(' ')}`)
    const result = await this.run(command, args, { signal, onOutput: (text) => this.log(text) })
    if (signal.aborted) throw new SetupStopped()
    if (result.code !== 0) {
      const last = this.current.log.at(-1)
      throw new SetupStopped(`${failure}${last ? ` ${last}` : ''}`)
    }
  }

  private log(text: string): void {
    this.set({ log: appendLog(this.current.log, text) })
  }

  private set(patch: Partial<EngineSetupState>): void {
    this.current = { ...this.current, ...patch }
    this.options.onState(this.current)
  }
}
