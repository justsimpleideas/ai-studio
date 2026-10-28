import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import {
  fewStepProblem,
  folderNameFor,
  isTwoExperts,
  isValidFolderName,
  parseImportEvent,
  type ModelImportState
} from '../src/shared/importing.js'
import { catalogModel, type ModelId } from '../src/shared/setup.js'
import type { ModelDownloads } from '../src/main/image/downloads.js'
import { ModelImporter } from '../src/main/image/importer.js'

describe('import events and names', () => {
  it('reads the probe, progress, done and error lines and nothing else', () => {
    const probe = parseImportEvent('{"event": "probe", "file": "m.safetensors", "size": 9, "family": "large", "label": "12B image model", "parameters": 12.2e9, "storage": "fp8", "layers": 28, "fp8_weights": 200}')
    expect(probe).toMatchObject({ event: 'probe', probe: { family: 'large', storage: 'fp8', layers: 28 } })
    expect(parseImportEvent('{"event": "probe", "file": "m", "family": "other"}')).toBeNull()
    expect(parseImportEvent('{"event": "probe", "file": "v", "family": "video", "label": "5B video model"}')).toMatchObject({
      probe: { family: 'video', label: '5B video model', video_kind: null }
    })
    const expert = parseImportEvent('{"event": "probe", "file": "v", "family": "video", "video_kind": "14b-image"}')
    expect(expert?.event === 'probe' && isTwoExperts(expert.probe)).toBe(true)
    expect(fewStepProblem(4, 5)).toBeNull()
    expect(fewStepProblem(4.5, 5)).toMatch(/Steps/)
    expect(fewStepProblem(4, 0)).toMatch(/Shift/)
    expect(parseImportEvent('{"event": "progress", "stage": "hashing", "done": 1, "size": 2}')).toEqual({ event: 'progress', stage: 'hashing', done: 1, size: 2 })
    expect(parseImportEvent('{"event": "done", "dest": "D"}')).toEqual({ event: 'done', dest: 'D' })
    expect(parseImportEvent('{"event": "error", "message": "no"}')).toEqual({ event: 'error', message: 'no' })
    expect(parseImportEvent('not json')).toBeNull()
  })

  it('makes a folder name from a file name, and refuses the download folders', () => {
    expect(folderNameFor('my model (fp8) v2.safetensors')).toBe('my-model-fp8-v2')
    expect(folderNameFor('...safetensors')).toBe('imported-model')
    expect(isValidFolderName('my-model_fp8.v2')).toBe(true)
    expect(isValidFolderName('../escape')).toBe(false)
    expect(isValidFolderName('a/b')).toBe(false)
    expect(isValidFolderName('trailing.')).toBe(false)
    expect(isValidFolderName('caption-model', ['caption-model'])).toBe(false)
  })
})

describe('importing a model file', () => {
  let models: string
  let calls: string[][]
  let downloaded: ModelId[]
  let selected: string[]
  let targets: string[]
  let states: ModelImportState[]
  let known: string[]

  beforeEach(() => {
    models = fs.mkdtempSync(path.join(os.tmpdir(), 'import-test-'))
    calls = []
    downloaded = []
    selected = []
    targets = []
    states = []
    known = []
  })
  afterEach(() => fs.rmSync(models, { recursive: true, force: true }))

  /** A folder fetch_model.py verified: just its manifest, naming the catalog's pinned commit. */
  function verified(id: ModelId, folder = path.join(models, catalogModel(id)!.folder)): string {
    const model = catalogModel(id)!
    fs.mkdirSync(folder, { recursive: true })
    fs.writeFileSync(path.join(folder, 'manifest.json'), JSON.stringify({ repo: model.repo, revision: model.revision, files: [] }))
    return folder
  }

  function importer(
    family: 'large' | 'diffusion' | 'video',
    script: { probe?: string[]; run?: string[]; code?: number; kind?: string; kinds?: Record<string, string> } = {},
    configured = {}
  ) {
    const downloads = {
      download: async (id: ModelId) => {
        downloaded.push(id)
        verified(id)
      },
      cancel: () => undefined
    } as unknown as ModelDownloads
    // The probe line for a file; `kinds` gives a video kind per file name (the 5B model's by default).
    const probeOf = (file: string) => {
      const kind = script.kinds?.[path.basename(file)] ?? script.kind ?? (family === 'video' ? '5b' : null)
      return `{"event": "probe", "file": "${path.basename(file)}", "size": 100, "family": "${family}", "label": "x", "parameters": 1, "storage": "fp8", "layers": 2, "fp8_weights": 3, "video_kind": ${kind ? `"${kind}"` : 'null'}}`
    }
    return new ModelImporter({
      python: () => 'venv-python',
      script: 'import_model.py',
      modelsFolder: () => models,
      engineFolders: async () => configured,
      knownFolders: () => known,
      rememberFolders: async (folders) => {
        known = [...new Set([...folders, ...known])]
      },
      downloads,
      selectFolder: async (dest, target) => {
        selected.push(dest)
        targets.push(target)
      },
      run: async (_command, args, options) => {
        calls.push(args)
        const lines = args.includes('--probe')
          ? (script.probe ?? [probeOf(args[args.indexOf('--transformer') + 1]!)])
          : (script.run ?? ['{"event": "progress", "stage": "hashing", "done": 100, "size": 100}', '{"event": "done", "dest": "D"}'])
        for (const line of lines) options?.onOutput?.(`${line}\n`, 'stdout')
        return { code: script.code ?? 0, stdout: '' }
      },
      onState: (s) => states.push(s)
    })
  }

  it('reads the file, downloads the missing VAE once accepted, imports, and selects the 12B folder', async () => {
    const caption = verified('caption', path.join(models, 'somewhere-else'))
    const imp = importer('large', {}, { caption_model_dir: caption })
    await imp.pick('C:/downloads/m.safetensors')
    expect(calls[0]).toEqual(['-I', '-u', 'import_model.py', '--transformer', 'C:/downloads/m.safetensors', '--probe', '--json'])
    expect(imp.state).toMatchObject({ phase: 'ready', needs: ['large-vae'], suggestedName: 'm' })

    await expect(imp.start({ variant: 'base', sha256: '', name: 'my-model', acceptLicenses: false })).rejects.toThrow(/Accept the licenses/)
    await imp.start({ variant: 'base', sha256: 'A'.repeat(64), name: 'my-model', acceptLicenses: true })
    expect(downloaded).toEqual(['large-vae'])
    const args = calls[1]!
    expect(args).toEqual(expect.arrayContaining(['--variant', 'base', '--dest', path.join(models, 'my-model'), '--sha256', 'a'.repeat(64)]))
    expect(args[args.indexOf('--text-encoder-from') + 1]).toBe(caption)
    expect(args[args.indexOf('--vae-from') + 1]).toBe(path.join(models, 'large-vae'))
    expect(selected).toEqual([path.join(models, 'my-model')])
    expect(imp.state).toMatchObject({ phase: 'done', dest: path.join(models, 'my-model') })
    expect(states.some((s) => s.phase === 'parts')).toBe(true)
  })

  it('takes the 6B parts from the downloaded 6B model when it is there', async () => {
    const full = verified('image')
    const imp = importer('diffusion')
    await imp.pick('m.safetensors')
    expect(imp.state.needs).toEqual([])
    await imp.start({ variant: 'distilled', sha256: '', name: 'six-b', acceptLicenses: false })
    const args = calls[1]!
    expect(args[args.indexOf('--parts-from') + 1]).toBe(full)
    expect(args).not.toContain('--sha256')
    expect(downloaded).toEqual([])
  })

  it('finds parts kept outside the models folder after the image model folder moved to an import', async () => {
    // The 6B download and the captioning model live together elsewhere; the image
    // model folder already points at an imported 12B model, as after a first import.
    const elsewhere = path.join(models, 'elsewhere')
    const full = verified('image', path.join(elsewhere, 'six-b-download'))
    const caption = verified('caption', path.join(elsewhere, 'caption-model'))
    const imported = path.join(models, 'my-12b')
    fs.mkdirSync(imported)
    fs.writeFileSync(path.join(imported, 'manifest.json'), JSON.stringify({ repo: 'local/m.safetensors', revision: 'f'.repeat(64) }))

    const imp = importer('diffusion', {}, { model_dir: imported, caption_model_dir: caption, memory_mode: 'auto' })
    await imp.pick('m.safetensors')
    expect(imp.state.needs).toEqual([])
    await imp.start({ variant: 'distilled', sha256: '', name: 'six-b', acceptLicenses: false })
    const args = calls[1]!
    expect(args[args.indexOf('--parts-from') + 1]).toBe(full)
    expect(downloaded).toEqual([])
    // Remembered, the imported folder is not: its source is a local file, not a pinned commit.
    expect(known).toEqual(expect.arrayContaining([full, caption]))
    expect(known).not.toContain(imported)

    // With the engine stopped, the remembered folders alone are enough.
    const later = importer('diffusion')
    await later.pick('m.safetensors')
    expect(later.state.needs).toEqual([])
  })

  it('imports a video model file from the Video model card and selects it as the video model', async () => {
    const imp = importer('video')
    await imp.pick('C:/downloads/v.safetensors', 'video')
    expect(imp.state).toMatchObject({ phase: 'ready', target: 'video', needs: ['video-parts'] })
    await imp.start({ variant: 'distilled', sha256: '', name: 'my-video', acceptLicenses: true })
    expect(downloaded).toEqual(['video-parts'])
    const args = calls[1]!
    expect(args[args.indexOf('--parts-from') + 1]).toBe(path.join(models, 'video-parts'))
    expect(args).toEqual(expect.arrayContaining(['--variant', 'distilled']))
    expect(selected).toEqual([path.join(models, 'my-video')])
    expect(targets).toEqual(['video'])
    expect(imp.state.message).toMatch(/next video/)
  })

  it('takes the video parts from the full original download, not from the distilled one', async () => {
    verified('video-fast')
    const full = verified('video')
    const imp = importer('video', {}, { video_model_dir: path.join(models, 'video-model-fast') })
    await imp.pick('v.safetensors', 'video')
    expect(imp.state.needs).toEqual([])
    await imp.start({ variant: 'base', sha256: '', name: 'my-video', acceptLicenses: false })
    const args = calls[1]!
    expect(args[args.indexOf('--parts-from') + 1]).toBe(full)
    expect(downloaded).toEqual([])
  })

  it('sends a file read under the wrong card to the other one', async () => {
    const video = importer('video')
    await video.pick('v.safetensors', 'image')
    expect(video.state).toMatchObject({ phase: 'failed', target: 'image' })
    expect(video.state.message).toMatch(/Import it under Video model/)
    await expect(video.start({ variant: 'base', sha256: '', name: 'x', acceptLicenses: true })).rejects.toThrow(/Choose a file/)

    const image = importer('diffusion')
    await image.pick('m.safetensors', 'video')
    expect(image.state.message).toMatch(/Import it under Image model/)
    expect(selected).toEqual([])
  })

  it('imports a 14B model from its two experts, put in order by their names, with a few-step setting', async () => {
    verified('video-parts')
    const kinds = { 'wan_low_noise.safetensors': '14b-text', 'wan_high_noise.safetensors': '14b-text', 'v5.safetensors': '5b' }
    const imp = importer('video', { kinds })
    await imp.pick('C:/Users/You/Downloads/wan_low_noise.safetensors', 'video')
    expect(imp.state).toMatchObject({ phase: 'ready', needs: ['video-14b-text-parts'], suggestedName: 'wan' })
    // Not yet: the second expert is missing.
    await expect(imp.start({ variant: 'base', sha256: '', name: 'w', acceptLicenses: true })).rejects.toThrow(/second expert/)
    await expect(imp.pickSecond('C:/Users/You/Downloads/v5.safetensors')).rejects.toThrow(/other expert of the/)
    await imp.pickSecond('C:/Users/You/Downloads/wan_high_noise.safetensors')
    // Chosen low noise first: put in order by the names.
    expect([imp.state.probe?.file, imp.state.second?.file]).toEqual(['wan_high_noise.safetensors', 'wan_low_noise.safetensors'])

    await expect(imp.start({ variant: 'distilled', sha256: '', name: 'w', acceptLicenses: true, steps: 0, shift: 5 }))
      .rejects.toThrow(/Steps must be/)
    await imp.start({ variant: 'distilled', sha256: '', name: 'w', acceptLicenses: true, steps: 4, shift: 5, sha256_2: 'B'.repeat(64) })
    expect(downloaded).toEqual(['video-14b-text-parts'])
    const args = calls.at(-1)!
    expect(args[args.indexOf('--transformer') + 1]).toBe('C:/Users/You/Downloads/wan_high_noise.safetensors')
    expect(args[args.indexOf('--transformer-2') + 1]).toBe('C:/Users/You/Downloads/wan_low_noise.safetensors')
    expect(args).toEqual(expect.arrayContaining(['--steps', '4', '--shift', '5', '--sha256-2', 'b'.repeat(64)]))
    expect(targets).toEqual(['video'])
  })

  it('swaps the experts on request, and a 5B file takes no second', async () => {
    const kinds = { 'a.safetensors': '14b-image', 'b.safetensors': '14b-image' }
    const imp = importer('video', { kinds })
    await imp.pick('a.safetensors', 'video')
    expect(imp.state.needs).toEqual(['video-14b-image-parts'])
    await imp.pickSecond('b.safetensors')
    expect(imp.state.probe?.file).toBe('a.safetensors')
    imp.swap()
    expect([imp.state.probe?.file, imp.state.second?.file]).toEqual(['b.safetensors', 'a.safetensors'])
    await imp.start({ variant: 'base', sha256: '', name: 'i', acceptLicenses: true })
    const args = calls.at(-1)!
    expect(args[args.indexOf('--transformer') + 1]).toBe('b.safetensors')
    expect(args).not.toContain('--steps')

    const five = importer('video')
    await five.pick('v.safetensors', 'video')
    await expect(five.pickSecond('w.safetensors')).rejects.toThrow(/first expert/)
  })

  it('refuses a bad folder name, a bad hash, and a folder of the downloads', async () => {
    verified('image-parts')
    const imp = importer('diffusion')
    await imp.pick('m.safetensors')
    for (const options of [
      { name: '../x', sha256: '' },
      { name: 'ok', sha256: 'abc' },
      { name: 'image-parts', sha256: '' }
    ]) {
      await expect(imp.start({ variant: 'base', acceptLicenses: false, ...options })).rejects.toThrow()
    }
    expect(calls).toHaveLength(1)
  })

  it('reports a file it cannot read, and an import the script refuses, without selecting anything', async () => {
    const bad = importer('large', { probe: ['{"event": "error", "message": "This is not a model the engine supports"}'], code: 1 })
    await bad.pick('x.safetensors')
    expect(bad.state).toMatchObject({ phase: 'failed', message: 'This is not a model the engine supports' })
    await expect(bad.start({ variant: 'base', sha256: '', name: 'x', acceptLicenses: true })).rejects.toThrow(/Choose a file/)

    verified('image-parts')
    const refused = importer('diffusion', { run: ['{"event": "error", "message": "SHA-256 mismatch"}'], code: 1 })
    await refused.pick('m.safetensors')
    await refused.start({ variant: 'base', sha256: '', name: 'x', acceptLicenses: false })
    expect(refused.state).toMatchObject({ phase: 'failed', message: 'SHA-256 mismatch' })
    expect(selected).toEqual([])
  })
})
