import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import {
  DownloadTracker,
  MODEL_CATALOG,
  PYTORCH_INDEX,
  catalogModel,
  packagesInstallArgs,
  parseFetchEvent,
  parseNvidiaSmi,
  parsePythonProbe,
  parseVerify,
  pythonProblem,
  torchInstallArgs,
  type EngineSetupState,
  type FetchEvent,
  type ModelDownloadState
} from '@shared/setup.js'
import { ModelDownloads } from '../src/main/image/downloads.js'
import { ImageEngine } from '../src/main/image/engine.js'
import { EngineSetup, venvPython, type RunOptions, type RunResult, type Runner } from '../src/main/image/setup.js'

const FETCH_SCRIPT = fs.readFileSync(path.join(import.meta.dirname, '..', 'image-engine', 'scripts', 'fetch_model.py'), 'utf8')

function pyConstant(name: string): string {
  const match = new RegExp(`^${name} = "([^"]+)"`, 'm').exec(FETCH_SCRIPT)
  if (!match?.[1]) throw new Error(`${name} not found in fetch_model.py`)
  return match[1]
}

describe('model catalog', () => {
  it('names the same repositories and commits as fetch_model.py', () => {
    const script = {
      image: [pyConstant('DEFAULT_REPO'), pyConstant('DEFAULT_REVISION')],
      video: [pyConstant('VIDEO_REPO'), pyConstant('VIDEO_REVISION')],
      'video-fast': [pyConstant('FAST_VIDEO_REPO'), pyConstant('FAST_VIDEO_REVISION')],
      caption: [pyConstant('CAPTION_REPO'), pyConstant('CAPTION_REVISION')],
      'image-parts': [pyConstant('DEFAULT_REPO'), pyConstant('DEFAULT_REVISION')],
      'large-vae': [pyConstant('LARGE_VAE_REPO'), pyConstant('LARGE_VAE_REVISION')],
      'video-parts': [pyConstant('VIDEO_REPO'), pyConstant('VIDEO_REVISION')],
      'video-14b-text-parts': [pyConstant('VIDEO_14B_TEXT_REPO'), pyConstant('VIDEO_14B_TEXT_REVISION')],
      'video-14b-image-parts': [pyConstant('VIDEO_14B_IMAGE_REPO'), pyConstant('VIDEO_14B_IMAGE_REVISION')]
    }
    for (const model of MODEL_CATALOG) {
      expect([model.repo, model.revision], model.id).toEqual(script[model.id])
      expect(model.revision).toMatch(/^[0-9a-f]{40}$/)
      expect(model.bytes).toBeGreaterThan(model.part ? 1e8 : 1e9)
    }
    expect(new Set(MODEL_CATALOG.map((m) => m.folder)).size).toBe(MODEL_CATALOG.length)
  })

  it('looks models up by id only', () => {
    expect(catalogModel('video-fast')?.kind).toBe('video')
    expect(catalogModel('../video')).toBeNull()
    expect(catalogModel(undefined)).toBeNull()
  })
})

describe('checking Python and the GPU', () => {
  const probe = (major: number, minor: number, bits64 = true) =>
    JSON.stringify({ major, minor, executable: 'C:\\Python\\python.exe', bits64 })

  it('reads the probe from the last line', () => {
    expect(parsePythonProbe(`warning\n${probe(3, 13)}\n`)).toEqual({ major: 3, minor: 13, executable: 'C:\\Python\\python.exe', bits64: true })
    expect(parsePythonProbe('Python was not found; run without arguments to install from the Microsoft Store')).toBeNull()
    expect(parsePythonProbe('')).toBeNull()
  })

  it('accepts 64-bit 3.12 to 3.14 only', () => {
    expect(pythonProblem(parsePythonProbe(probe(3, 12))!)).toBeNull()
    expect(pythonProblem(parsePythonProbe(probe(3, 14))!)).toBeNull()
    expect(pythonProblem(parsePythonProbe(probe(3, 11))!)).toMatch(/3\.11 is not supported/)
    expect(pythonProblem(parsePythonProbe(probe(3, 15))!)).toMatch(/3\.15 is not supported/)
    expect(pythonProblem(parsePythonProbe(probe(3, 13, false))!)).toMatch(/32-bit/)
  })

  it('reads every GPU nvidia-smi lists', () => {
    expect(parseNvidiaSmi('NVIDIA GeForce RTX 4070, 12282, 580.97\r\nNVIDIA RTX A2000, 6138, 580.97\r\n')).toEqual([
      { name: 'NVIDIA GeForce RTX 4070', memoryMiB: 12282, driver: '580.97' },
      { name: 'NVIDIA RTX A2000', memoryMiB: 6138, driver: '580.97' }
    ])
    expect(parseNvidiaSmi('NVIDIA-SMI has failed because it could not communicate with the NVIDIA driver.')).toEqual([])
  })

  it('reads the final check', () => {
    expect(parseVerify('{"torch": "2.11.0+cu128", "diffusers": "0.42.0", "cuda": true, "device": "GPU"}')).toEqual({
      torch: '2.11.0+cu128',
      diffusers: '0.42.0',
      cuda: true,
      device: 'GPU'
    })
    expect(parseVerify('Traceback (most recent call last):')).toBeNull()
  })
})

describe('pip commands', () => {
  it('takes only PyTorch and torchvision from the PyTorch index, without their dependencies', () => {
    const args = torchInstallArgs('c.txt')
    expect(args).toContain('--only-binary=:all:')
    expect(args.slice(args.indexOf('--index-url'), args.indexOf('--index-url') + 2)).toEqual(['--index-url', PYTORCH_INDEX])
    expect(args).toContain('--no-deps')
    expect(args.slice(-4)).toEqual(['-c', 'c.txt', 'torch', 'torchvision'])
    expect(args).not.toContain('--extra-index-url')
  })

  it('takes everything else from PyPI, constrained to the audited versions', () => {
    const args = packagesInstallArgs('r.txt', 'c.txt')
    expect(args).not.toContain('--index-url')
    expect(args).not.toContain('--extra-index-url')
    expect(args).toContain('--only-binary=:all:')
    expect(args.join(' ')).toContain('-c c.txt -r r.txt torch torchvision')
  })

  it('can install from a folder of wheels with no index at all', () => {
    for (const args of [torchInstallArgs('c', { wheelhouse: 'W' }), packagesInstallArgs('r', 'c', { wheelhouse: 'W' })]) {
      expect(args.join(' ')).toContain('--no-index --find-links W')
      expect(args).not.toContain('--index-url')
    }
  })

  it('pins every requirement in the constraints file', () => {
    const dir = path.join(import.meta.dirname, '..', 'image-engine')
    const pinned = new Map(
      fs
        .readFileSync(path.join(dir, 'constraints.txt'), 'utf8')
        .split(/\r?\n/)
        .filter((l) => l && !l.startsWith('#'))
        .map((l) => {
          const [name, version] = l.split('==')
          return [name!.toLowerCase().replace(/_/g, '-'), version] as const
        })
    )
    expect(pinned.get('torch')).toMatch(/^2\.\d+\.\d+\+cu128$/)
    for (const line of fs.readFileSync(path.join(dir, 'requirements.txt'), 'utf8').split(/\r?\n/)) {
      if (!line || line.startsWith('#')) continue
      const [name, version] = line.split('==')
      expect(pinned.get(name!.toLowerCase().replace(/_/g, '-')), name).toBe(version)
    }
  })
})

// ---------------------------------------------------------------------------
// The setup steps, against a fake machine.

interface Call {
  command: string
  args: string[]
}

function fakeMachine(answers: (call: Call) => Partial<RunResult> & { output?: string }) {
  const calls: Call[] = []
  const run: Runner = async (command: string, args: string[], options?: RunOptions) => {
    const call = { command, args }
    calls.push(call)
    const answer = answers(call)
    if (answer.output) options?.onOutput?.(answer.output, 'stdout')
    return { code: answer.code ?? 0, stdout: answer.stdout ?? '' }
  }
  return { run, calls }
}

const PROBE = (minor: number) => JSON.stringify({ major: 3, minor, executable: `C:\\Py3${minor}\\python.exe`, bits64: true })
const SMI = 'NVIDIA GeForce RTX 4070, 12282, 580.97'
const VERIFY_OK = '{"torch": "2.11.0+cu128", "diffusers": "0.42.0", "cuda": true, "device": "RTX 4070"}'

let tmp: string | null = null
afterEach(() => {
  if (tmp) fs.rmSync(tmp, { recursive: true, force: true })
  tmp = null
})

function setupWith(answers: Parameters<typeof fakeMachine>[0]) {
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'setup-test-'))
  const machine = fakeMachine(answers)
  const states: EngineSetupState[] = []
  const setup = new EngineSetup({
    venvDir: path.join(tmp, 'image-python'),
    requirements: 'requirements.txt',
    constraints: 'constraints.txt',
    run: machine.run,
    candidates: [['py', ['-3.14']], ['py', ['-3.13']], ['python', []]],
    onState: (s) => states.push(s)
  })
  return { setup, calls: machine.calls, states, venv: path.join(tmp, 'image-python') }
}

/** The usual healthy machine: 3.13 through the launcher, a GPU, pip succeeding. */
function healthy(call: Call): Partial<RunResult> & { output?: string } {
  if (call.command === 'py' && call.args[0] === '-3.14') return { code: 103 }
  if (call.command === 'py' && call.args[0] === '-3.13') return { stdout: PROBE(13) }
  if (call.command === 'nvidia-smi') return { stdout: SMI }
  if (call.args[0] === '-c' && call.args[1]?.includes('torch.cuda')) return { stdout: VERIFY_OK }
  if (call.args.includes('install')) return { output: 'Successfully installed\n' }
  return {}
}

describe('installing the engine', () => {
  it('finds a supported Python, makes the environment, installs PyTorch then the rest, and checks the GPU', async () => {
    const { setup, calls, states, venv } = setupWith(healthy)
    const python = await setup.install()
    expect(python).toBe(venvPython(venv))
    const shown = calls.map((c) => `${path.basename(c.command)} ${c.args.slice(0, 4).join(' ')}`)
    expect(shown[0]).toMatch(/^py -3\.14 -c/)
    expect(shown[1]).toMatch(/^py -3\.13 -c/)
    expect(shown[2]).toMatch(/^nvidia-smi/)
    expect(calls[3]).toEqual({ command: 'C:\\Py313\\python.exe', args: ['-m', 'venv', '--clear', venv] })
    expect(calls[4]!.args).toContain(PYTORCH_INDEX)
    expect(calls[5]!.args).toContain('-r')
    expect(calls[6]!.args.slice(0, 3)).toEqual(['-m', 'pip', 'check'])
    expect(calls[7]!.args[1]).toContain('torch.cuda.is_available()')
    expect(states.map((s) => s.phase)).toEqual(expect.arrayContaining(['checking', 'creating', 'torch', 'packages', 'verifying', 'done']))
    expect(setup.state.gpu).toBe('NVIDIA GeForce RTX 4070 · 12 GB · driver 580.97')
    expect(setup.state.log.join('\n')).toContain('Successfully installed')
  })

  it('keeps a working environment instead of making it again (repair)', async () => {
    const { setup, calls, venv } = setupWith((call) => {
      if (call.command === venvPython(venv) && call.args[1]?.includes('sys.version_info')) return { stdout: PROBE(13) }
      return healthy(call)
    })
    fs.mkdirSync(path.dirname(venvPython(venv)), { recursive: true })
    fs.writeFileSync(venvPython(venv), '')
    await setup.install()
    expect(calls.some((c) => c.args.includes('venv'))).toBe(false)
  })

  it('says which Pythons it found and how to get a supported one', async () => {
    const { setup } = setupWith((call) => (call.command === 'python' ? { stdout: PROBE(11) } : { code: 103 }))
    await expect(setup.install()).rejects.toThrow(/Python 3\.12 to 3\.14 \(64-bit\) was not found\. Found: Python 3\.11 is not supported/)
    expect(setup.state.phase).toBe('failed')
    expect(setup.state.message).toContain('winget install Python.Python.3.13')
  })

  it('stops before installing anything without an NVIDIA GPU', async () => {
    const { setup, calls } = setupWith((call) => (call.command === 'nvidia-smi' ? { code: null } : healthy(call)))
    await expect(setup.install()).rejects.toThrow(/No NVIDIA GPU was found/)
    expect(calls.some((c) => c.args.includes('install') || c.args.includes('venv'))).toBe(false)
  })

  it('fails with pip’s last words when a step fails', async () => {
    const { setup } = setupWith((call) =>
      call.args.includes(PYTORCH_INDEX) ? { code: 1, output: 'ERROR: No matching distribution found for torch\n' } : healthy(call)
    )
    await expect(setup.install()).rejects.toThrow('PyTorch could not be installed. ERROR: No matching distribution found for torch')
    expect(setup.state.phase).toBe('failed')
  })

  it('says so when PyTorch is installed but cannot see the GPU', async () => {
    const { setup, venv } = setupWith((call) =>
      call.command === venvPython(venv) && call.args[1]?.includes('torch.cuda')
        ? { stdout: '{"torch": "2.11.0+cu128", "diffusers": "0.42.0", "cuda": false, "device": null}' }
        : healthy(call)
    )
    await expect(setup.install()).rejects.toThrow(/cannot use the GPU\. Update the NVIDIA driver/)
  })

  it('can be stopped, and says the next run picks up where it left off', async () => {
    let setupRef: EngineSetup | null = null
    const { setup } = setupWith((call) => {
      if (call.args.includes(PYTORCH_INDEX)) setupRef?.cancel()
      return healthy(call)
    })
    setupRef = setup
    await expect(setup.install()).rejects.toThrow(/picks up where it left off/)
    expect(setup.state.phase).toBe('cancelled')
    expect(setup.running).toBe(false)
  })

  it('runs one install at a time', async () => {
    let release: () => void = () => undefined
    const gate = new Promise<void>((r) => (release = r))
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'setup-test-'))
    const setup = new EngineSetup({
      venvDir: path.join(tmp, 'v'),
      requirements: 'r',
      constraints: 'c',
      candidates: [['py', []]],
      run: async () => {
        await gate
        return { code: 1, stdout: '' }
      },
      onState: () => undefined
    })
    const first = setup.install().catch(() => undefined)
    await expect(setup.install()).rejects.toThrow(/already being installed/)
    release()
    await first
  })
})

// ---------------------------------------------------------------------------
// Downloads

describe('download progress', () => {
  const events: FetchEvent[] = [
    { event: 'plan', repo: 'r', revision: 'x', files: 3, bytes: 1000 },
    { event: 'verify', path: 'model_index.json', size: 100 },
    { event: 'file', path: 'model_index.json', size: 100, action: 'ok' },
    { event: 'file', path: 'vae/w.safetensors', size: 300, action: 'linked' },
    { event: 'verify', path: 'vae/w.safetensors', size: 300 },
    { event: 'file', path: 'vae/w.safetensors', size: 300, action: 'ok' },
    { event: 'file', path: 'transformer/t.safetensors', size: 600, action: 'fetch' }
  ]

  it('counts verified files plus the one in flight', () => {
    const tracker = new DownloadTracker('video-fast', 'D', 999)
    expect(tracker.state.phase).toBe('starting')
    let state: ModelDownloadState = tracker.state
    for (const e of events) state = tracker.feed(e, 0)
    expect(state.totalBytes).toBe(1000)
    expect(state.doneBytes).toBe(400)
    state = tracker.feed({ event: 'progress', path: 'transformer/t.safetensors', done: 250, size: 600 }, 0)
    expect(state.doneBytes).toBe(650)
    expect(state.activity).toBe('downloading')
    state = tracker.feed({ event: 'progress', path: 'transformer/t.safetensors', done: 450, size: 600 }, 4000)
    expect(state.bytesPerSecond).toBe(50)
    state = tracker.feed({ event: 'done', files: 3, bytes: 1000 }, 5000)
    expect(state).toMatchObject({ phase: 'done', doneBytes: 1000, activity: null })
  })

  it('a restarted file does not count twice', () => {
    const tracker = new DownloadTracker('image', 'D', 100)
    tracker.feed({ event: 'plan', repo: 'r', revision: 'x', files: 1, bytes: 100 })
    tracker.feed({ event: 'progress', path: 'a', done: 80, size: 100 })
    const state = tracker.feed({ event: 'file', path: 'a', size: 100, action: 'stale' })
    expect(state.doneBytes).toBe(0)
  })

  it('ignores lines that are not events', () => {
    expect(parseFetchEvent('  ok      vae/config.json')).toBeNull()
    expect(parseFetchEvent('{"no": "event"}')).toBeNull()
    expect(parseFetchEvent('{"event": "done", "files": 1, "bytes": 2}')).toEqual({ event: 'done', files: 1, bytes: 2 })
  })
})

describe('downloading a model', () => {
  function downloadsWith(lines: string[], init: { code?: number | null; free?: number | null; reuse?: string | null } = {}) {
    const calls: Call[] = []
    const applied: string[] = []
    const states: ModelDownloadState[] = []
    const downloads = new ModelDownloads({
      python: () => 'venv-python',
      script: 'fetch_model.py',
      run: async (command, args, options) => {
        calls.push({ command, args })
        for (const line of lines) options?.onOutput?.(`${line}\n`, 'stdout')
        options?.onOutput?.('a message for people\n', 'stderr')
        return { code: init.code ?? 0, stdout: '' }
      },
      freeBytes: async () => init.free ?? null,
      reuseSource: async () => init.reuse ?? null,
      applyFolder: async (model, dest) => {
        applied.push(`${model.kind}:${dest}`)
      },
      onState: (s) => states.push(s)
    })
    return { downloads, calls, applied, states }
  }

  const DONE = [
    '{"event": "plan", "repo": "r", "revision": "x", "files": 1, "bytes": 5}',
    '{"event": "file", "path": "a", "size": 5, "action": "fetch"}',
    '{"event": "progress", "path": "a", "done": 2, "size": 5}',
    '{"event": "verify", "path": "a", "size": 5}',
    '{"event": "file", "path": "a", "size": 5, "action": "ok"}',
    '{"event": "done", "files": 1, "bytes": 5}'
  ]

  it('runs fetch_model.py isolated with JSON progress, then selects the verified folder', async () => {
    const { downloads, calls, applied } = downloadsWith(DONE)
    await downloads.download('video-fast', 'M')
    const dest = path.join('M', 'video-model-fast')
    expect(calls[0]).toEqual({
      command: 'venv-python',
      args: ['-I', '-u', 'fetch_model.py', '--model', 'video-fast', '--dest', dest, '--json']
    })
    expect(applied).toEqual([`video:${dest}`])
    expect(downloads.state).toMatchObject({ phase: 'done', doneBytes: 5, message: 'Downloaded, verified, and selected.' })
  })

  it('links the files another video model already has', async () => {
    const { downloads, calls } = downloadsWith(DONE, { reuse: 'M/video-model' })
    await downloads.download('video-fast', 'M')
    expect(calls[0]!.args.slice(-2)).toEqual(['--reuse-from', 'M/video-model'])
  })

  it('refuses up front when the drive is too small', async () => {
    const { downloads, calls } = downloadsWith(DONE, { free: 5e9 })
    await expect(downloads.download('image', path.join(os.tmpdir(), 'no-such-models-folder'))).rejects.toThrow(/Not enough free space/)
    expect(calls).toHaveLength(0)
    expect(downloads.state.phase).toBe('failed')
  })

  it('reports the script’s own error and selects nothing', async () => {
    const { downloads, applied } = downloadsWith(
      ['{"event": "plan", "repo": "r", "revision": "x", "files": 1, "bytes": 5}', '{"event": "error", "message": "Verification failed: a: SHA-256 mismatch"}'],
      { code: 1 }
    )
    await expect(downloads.download('image', 'M')).rejects.toThrow('Verification failed: a: SHA-256 mismatch')
    expect(applied).toEqual([])
    expect(downloads.state.phase).toBe('failed')
  })

  it('falls back to what the script printed for people when it ends without an event', async () => {
    const { downloads } = downloadsWith([], { code: 2 })
    await expect(downloads.download('image', 'M')).rejects.toThrow('a message for people')
  })

  it('rejects unknown models and a second download at once', async () => {
    const { downloads } = downloadsWith(DONE)
    await expect(downloads.download('../../x' as never, 'M')).rejects.toThrow('Unknown model.')
    const first = downloads.download('image', 'M')
    await expect(downloads.download('video', 'M')).rejects.toThrow(/already being downloaded/)
    await first
  })
})

describe('the engine while its environment is installed', () => {
  it('cannot be started until released, so the window cannot restart it under pip', async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'engine-hold-'))
    const states: string[] = []
    const engine = new ImageEngine({
      python: () => 'no-such-python-for-this-test',
      engineDir: dir,
      dataDir: dir,
      outputsDir: () => null,
      port: 1,
      module: 'image_engine',
      appVersion: '0',
      onEvent: () => undefined,
      onState: (s) => states.push(s.state)
    })
    await engine.hold('The engine is being installed.')
    await expect(engine.start()).rejects.toThrow('The engine is being installed.')
    expect(states).toEqual([])
    engine.release()
    await expect(engine.start()).rejects.toThrow(/Could not start Python/)
    fs.rmSync(dir, { recursive: true, force: true })
  })
})
