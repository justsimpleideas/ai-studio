import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import { randomUUID } from 'node:crypto'
import type { BrowserWindow, Dialog, IpcMain, OpenDialogOptions } from 'electron'
import {
  FOLDER_PIN,
  MAX_START_IMAGE_BYTES,
  engineImagePath,
  gifFileName,
  isEngineId,
  isVideo,
  parseGifOptions,
  rewriteRecord,
  sniffImageType,
  type GalleryFolder,
  type GalleryPage,
  type GalleryQuery,
  type GenerationRequest,
  type ImageJob,
  type ImageRecord,
  type ImageSettings,
  type MediaKind,
  type StartImage,
  type VideoRequest
} from '@shared/image.js'
import { catalogModel } from '@shared/setup.js'
import type { GpuArbiter } from '../gpu.js'
import type { ConfigStore } from '../store.js'
import type { ModelDownloads } from './downloads.js'
import type { ModelImporter } from './importer.js'
import type { ModelInspector } from './inspect.js'
import type { ImageEngine } from './engine.js'
import { withoutPngMetadata } from './png-metadata.js'
import type { EngineSetup } from './setup.js'

/**
 * The window's image and video operations, each a named handler.
 *
 * Arguments from the window are checked here before they reach the engine:
 * ids must look like engine ids, queries are rebuilt field by field, and the
 * only paths ever used come from native dialogs the user answered. The engine
 * validates generation requests again on its own side. A start picture travels
 * as bytes whose real type is checked here and again by the engine.
 */

interface Deps {
  ipcMain: IpcMain
  dialog: Dialog
  store: ConfigStore
  imageEngine: ImageEngine
  gpu: GpuArbiter
  getWindow: () => BrowserWindow | null
  setup: EngineSetup
  downloads: ModelDownloads
  inspector: ModelInspector
  importer: ModelImporter
  /** Where downloads go when no folder has been chosen. */
  defaultModelsFolder: string
  openExternal: (url: string) => Promise<void>
}

const SETTING_KEYS = new Set([
  'model_dir',
  'memory_mode',
  'load_on_start',
  'video_model_dir',
  'video_memory_mode',
  'ffmpeg_path',
  'caption_model_dir'
])

function requireId(id: unknown): string {
  if (!isEngineId(id)) throw new Error('Invalid id.')
  return id
}

function optionalKind(kind: unknown): MediaKind | undefined {
  if (kind === undefined || kind === null) return undefined
  if (kind !== 'image' && kind !== 'video') throw new Error('Invalid model kind.')
  return kind
}

/** Store picture bytes as a start picture, after checking what they really are. */
async function uploadStartImage(imageEngine: ImageEngine, bytes: Uint8Array): Promise<StartImage> {
  if (bytes.byteLength === 0 || bytes.byteLength > MAX_START_IMAGE_BYTES) {
    throw new Error(`A start picture must be smaller than ${MAX_START_IMAGE_BYTES / 2 ** 20} MB.`)
  }
  const type = sniffImageType(bytes)
  if (!type) throw new Error('A start picture must be a PNG, JPEG or WebP image.')
  await imageEngine.start()
  return (await imageEngine.upload('/api/inputs', bytes, type)) as StartImage
}

export function registerImageHandlers({
  ipcMain,
  dialog,
  store,
  imageEngine,
  gpu,
  getWindow,
  setup,
  downloads,
  inspector,
  importer,
  defaultModelsFolder,
  openExternal
}: Deps): void {
  const api = async (method: string, route: string, body?: unknown): Promise<unknown> => {
    await imageEngine.start()
    return imageEngine.request(method, route, body)
  }

  ipcMain.handle('image:engineState', () => imageEngine.state)
  ipcMain.handle('image:start', () => imageEngine.start())
  ipcMain.handle('image:status', () => api('GET', '/api/status'))
  ipcMain.handle('image:settings', () => api('GET', '/api/settings'))

  ipcMain.handle('image:saveSettings', (_e, changes: unknown) => {
    if (!changes || typeof changes !== 'object') throw new Error('Invalid settings.')
    const clean: Record<string, unknown> = {}
    for (const [key, value] of Object.entries(changes)) {
      if (!SETTING_KEYS.has(key)) throw new Error(`Unknown setting: ${key}`)
      clean[key] = value
    }
    return api('PUT', '/api/settings', clean)
  })

  ipcMain.handle('image:loadModel', async (_e, kind: unknown) => {
    const which = optionalKind(kind)
    await imageEngine.start()
    await gpu.claim('image')
    await imageEngine.request('POST', '/api/model/load', which ? { kind: which } : {})
  })

  ipcMain.handle('image:unloadModel', async (_e, kind: unknown) => {
    const which = optionalKind(kind)
    await api('POST', '/api/model/unload', which ? { kind: which } : {})
  })

  ipcMain.handle('image:submit', async (_e, request: GenerationRequest) => {
    await imageEngine.start()
    // Generating loads the image model, so the chat model has to make room first.
    await gpu.claim('image')
    return imageEngine.request('POST', '/api/jobs', request)
  })

  ipcMain.handle('image:submitVideo', async (_e, request: VideoRequest) => {
    if (!request || typeof request !== 'object') throw new Error('Invalid request.')
    await imageEngine.start()
    // The video model lives in the same engine and needs the same room on the GPU.
    await gpu.claim('image')
    return imageEngine.request('POST', '/api/jobs', { ...request, kind: 'video' })
  })

  ipcMain.handle('image:pickStartImage', async () => {
    const picked = await pickOpenPath('STUDIO_PICK_START_IMAGE', {
      title: 'Choose a picture for the video to start from',
      properties: ['openFile'],
      filters: [{ name: 'Pictures', extensions: ['png', 'jpg', 'jpeg', 'webp'] }]
    })
    if (!picked) return null
    const stat = await fsp.stat(picked)
    if (stat.size > MAX_START_IMAGE_BYTES) throw new Error(`That picture is larger than ${MAX_START_IMAGE_BYTES / 2 ** 20} MB.`)
    return uploadStartImage(imageEngine, await fsp.readFile(picked))
  })

  ipcMain.handle('image:uploadStartImage', (_e, data: unknown) => {
    if (!(data instanceof Uint8Array)) throw new Error('Invalid picture.')
    return uploadStartImage(imageEngine, data)
  })

  ipcMain.handle('image:startImageFrom', (_e, id: unknown, frame: unknown) => {
    if (frame !== 'image' && frame !== 'last') throw new Error('Invalid frame.')
    return api('POST', '/api/inputs/from-gallery', { id: requireId(id), frame })
  })

  ipcMain.handle('image:cancel', (_e, id: unknown) => api('POST', `/api/jobs/${requireId(id)}/cancel`, {}))

  ipcMain.handle('image:jobs', async () => ((await api('GET', '/api/jobs')) as { items: ImageJob[] }).items)

  ipcMain.handle('image:gallery', async (_e, query: GalleryQuery) => {
    const params = new URLSearchParams()
    const offset = Number(query?.offset ?? 0)
    const limit = Number(query?.limit ?? 60)
    if (Number.isInteger(offset) && offset > 0) params.set('offset', String(offset))
    if (Number.isInteger(limit) && limit > 0) params.set('limit', String(Math.min(limit, 200)))
    if (typeof query?.q === 'string' && query.q.trim()) params.set('q', query.q.slice(0, 200))
    if (query?.favorites === true) params.set('favorites', '1')
    if (query?.media === 'image' || query?.media === 'video') params.set('media', query.media)
    if (query?.folder !== undefined) params.set('folder', requireId(query.folder))
    const qs = params.toString()
    const page = (await api('GET', `/api/gallery${qs ? `?${qs}` : ''}`)) as GalleryPage
    return { total: page.total, items: page.items.map(rewriteRecord) }
  })

  ipcMain.handle('image:setFavorite', async (_e, id: unknown, favorite: unknown) => {
    if (typeof favorite !== 'boolean') throw new Error('Invalid value.')
    return rewriteRecord((await api('PATCH', `/api/gallery/${requireId(id)}`, { favorite })) as ImageRecord)
  })

  ipcMain.handle('image:delete', async (_e, id: unknown) => {
    await api('DELETE', `/api/gallery/${requireId(id)}`)
  })

  // Folder names are checked by the engine (length, characters, duplicates); here only
  // that they are text of a sane size, and that ids look like ids.
  const folderName = (name: unknown): string => {
    if (typeof name !== 'string' || name.length > 1000) throw new Error('Invalid folder name.')
    return name
  }

  ipcMain.handle('image:folders', async () => ((await api('GET', '/api/folders')) as { items: GalleryFolder[] }).items)

  ipcMain.handle('image:createFolder', (_e, name: unknown) => api('POST', '/api/folders', { name: folderName(name) }))

  ipcMain.handle('image:renameFolder', (_e, id: unknown, name: unknown) =>
    api('PATCH', `/api/folders/${requireId(id)}`, { name: folderName(name) })
  )

  ipcMain.handle('image:deleteFolder', (_e, id: unknown) => api('DELETE', `/api/folders/${requireId(id)}`))

  ipcMain.handle('image:moveToFolder', async (_e, id: unknown, folderId: unknown) => {
    const folder = folderId === null ? null : requireId(folderId)
    return rewriteRecord((await api('PATCH', `/api/gallery/${requireId(id)}`, { folder })) as ImageRecord)
  })

  // Hidden and locked folders. The engine keeps what this session has shown or unlocked in
  // memory only, and the PIN is checked there; here only that values have the right shape.
  const folderPin = (pin: unknown): string => {
    if (typeof pin !== 'string' || !FOLDER_PIN.test(pin)) throw new Error('A PIN is six digits.')
    return pin
  }

  ipcMain.handle('image:setFolderHidden', (_e, id: unknown, hidden: unknown) => {
    if (typeof hidden !== 'boolean') throw new Error('Invalid value.')
    return api('PATCH', `/api/folders/${requireId(id)}`, { hidden })
  })

  ipcMain.handle('image:setFolderPin', (_e, id: unknown, pin: unknown, current: unknown) =>
    api('PUT', `/api/folders/${requireId(id)}/pin`, {
      pin: pin === null ? null : folderPin(pin),
      ...(current === null ? {} : { current: folderPin(current) })
    })
  )

  ipcMain.handle('image:unlockFolder', (_e, id: unknown, pin: unknown) =>
    api('POST', `/api/folders/${requireId(id)}/unlock`, { pin: folderPin(pin) })
  )

  ipcMain.handle('image:lockFolder', (_e, id: unknown) => api('POST', `/api/folders/${requireId(id)}/lock`, {}))

  // Merging other galleries. The window holds tokens; the paths stay here, chosen in a dialog.
  const mergeSources = new Map<string, string>()
  let mergePicks = 0
  ipcMain.handle('image:pickMergeSource', async () => {
    // For the end-to-end test: folders named up front, one per pick in turn.
    const preset = process.env['STUDIO_PICK_MERGE_SOURCES']?.split(path.delimiter).filter(Boolean)
    const picked = preset?.length
      ? preset[mergePicks++ % preset.length]!
      : await pickOpenPath('STUDIO_PICK_MERGE_SOURCE', {
          title: 'Choose a gallery to merge: an outputs folder',
          properties: ['openDirectory'],
          buttonLabel: 'Merge this gallery'
        })
    if (!picked) return null
    // The outputs folder itself, or the folder holding it (the studio's image folder, say).
    const folder = fs.existsSync(path.join(picked, 'gallery.db'))
      ? picked
      : fs.existsSync(path.join(picked, 'outputs', 'gallery.db'))
        ? path.join(picked, 'outputs')
        : null
    if (!folder) throw new Error('That folder is not a studio gallery. Choose an “outputs” folder: the one that holds gallery.db.')
    const token = randomUUID()
    mergeSources.set(token, folder)
    return { token, label: path.basename(folder), path: folder }
  })

  const mergeBody = (tokens: unknown, renames: unknown): { sources: string[]; renames: Record<string, string> } => {
    if (!Array.isArray(tokens) || tokens.length === 0) throw new Error('Choose a gallery to merge first.')
    const sources = tokens.map((t) => {
      const folder = typeof t === 'string' ? mergeSources.get(t) : undefined
      if (!folder) throw new Error('That gallery is no longer chosen. Add it again.')
      return folder
    })
    const clean: Record<string, string> = {}
    if (renames && typeof renames === 'object') {
      for (const [key, name] of Object.entries(renames)) {
        if (/^(target|\d{1,2}):[0-9a-f]{16}$/.test(key) && typeof name === 'string' && name.length <= 200) clean[key] = name
      }
    }
    return { sources, renames: clean }
  }

  ipcMain.handle('image:inspectMerge', (_e, tokens: unknown, renames: unknown) =>
    api('POST', '/api/gallery/merge/inspect', mergeBody(tokens, renames))
  )
  ipcMain.handle('image:startMerge', (_e, tokens: unknown, renames: unknown) =>
    api('POST', '/api/gallery/merge', mergeBody(tokens, renames))
  )
  ipcMain.handle('image:mergeStatus', () => api('GET', '/api/gallery/merge'))
  ipcMain.handle('image:cancelMerge', async () => {
    await api('POST', '/api/gallery/merge/cancel', {})
  })

  ipcMain.handle('image:showHiddenFolders', async () =>
    ((await api('GET', '/api/folders')) as { show_hidden: boolean }).show_hidden
  )

  ipcMain.handle('image:setShowHiddenFolders', async (_e, show: unknown) => {
    if (typeof show !== 'boolean') throw new Error('Invalid value.')
    return ((await api('PUT', '/api/gallery/privacy', { show_hidden: show })) as { show_hidden: boolean }).show_hidden
  })

  ipcMain.handle('image:saveAs', async (_e, id: unknown) => {
    const imageId = requireId(id)
    const record = (await api('GET', `/api/gallery/${imageId}`)) as ImageRecord
    if (isVideo(record)) {
      // Videos carry no settings inside the file, so there is nothing to strip.
      const mp4 = record.mime === 'video/mp4'
      const target = await pickSavePath(record.file_name, mp4 ? 'MP4 video' : 'Animated WebP', mp4 ? 'mp4' : 'webp')
      if (!target) return null
      const response = await imageEngine.raw('GET', `/videos/${imageId}`)
      if (!response.ok) throw new Error('The video could not be read.')
      await fsp.writeFile(target, Buffer.from(await response.arrayBuffer()))
      return target
    }
    const target = await pickSavePath(record.file_name, 'PNG image', 'png')
    if (!target) return null
    const response = await imageEngine.raw('GET', `/images/${imageId}.png`)
    if (!response.ok) throw new Error('The image could not be read.')
    const png = Buffer.from(await response.arrayBuffer())
    // The gallery's copy keeps its settings; the exported one only if the user said so.
    const out = store.get().image.includeSettingsInExports ? png : withoutPngMetadata(png)
    await fsp.writeFile(target, out)
    return target
  })

  ipcMain.handle('image:saveAsGif', async (_e, id: unknown, options: unknown) => {
    const videoId = requireId(id)
    const gif = parseGifOptions(options)
    const record = (await api('GET', `/api/gallery/${videoId}`)) as ImageRecord
    if (!isVideo(record)) throw new Error('Only a video can be saved as a GIF.')
    const target = await pickSavePath(gifFileName(record.file_name), 'GIF animation', 'gif')
    if (!target) return null
    // The engine makes the GIF on request and sends it back; it keeps no copy.
    const response = await imageEngine.raw('POST', `/api/gallery/${videoId}/gif`, gif)
    if (!response.ok) {
      const data = (await response.json().catch(() => null)) as { error?: unknown } | null
      throw new Error(typeof data?.error === 'string' ? data.error : `The GIF could not be made (HTTP ${response.status}).`)
    }
    await fsp.writeFile(target, Buffer.from(await response.arrayBuffer()))
    return target
  })

  ipcMain.handle('image:includeSettingsInExports', () => store.get().image.includeSettingsInExports)

  ipcMain.handle('image:setIncludeSettingsInExports', async (_e, value: unknown) => {
    if (typeof value !== 'boolean') throw new Error('Invalid value.')
    const config = await store.update((c) => {
      c.image.includeSettingsInExports = value
    })
    return config.image.includeSettingsInExports
  })

  /** A native save dialog, or for the end-to-end test a file named up front. */
  async function pickSavePath(defaultName: string, typeName: string, extension: string): Promise<string | null> {
    const preset = process.env['STUDIO_SAVE_AS']
    if (preset) return preset
    const window = getWindow()
    const options = {
      title: extension === 'png' ? 'Save image' : extension === 'gif' ? 'Save as GIF' : 'Save video',
      defaultPath: defaultName,
      filters: [{ name: typeName, extensions: [extension] }]
    }
    const result = window ? await dialog.showSaveDialog(window, options) : await dialog.showSaveDialog(options)
    return result.canceled || !result.filePath ? null : result.filePath
  }

  /** A native open dialog, or for the end-to-end test a path named up front in `envName`. */
  async function pickOpenPath(envName: string, options: OpenDialogOptions): Promise<string | null> {
    const preset = process.env[envName]
    if (preset) return preset
    const window = getWindow()
    const result = window ? await dialog.showOpenDialog(window, options) : await dialog.showOpenDialog(options)
    const picked = result.filePaths[0]
    return result.canceled || !picked ? null : picked
  }

  ipcMain.handle('image:pickModelFolder', async (_e, kind: unknown) => {
    const which = kind === 'caption' ? 'caption' : optionalKind(kind) ?? 'image'
    const picked = await pickOpenPath('STUDIO_PICK_MODEL_FOLDER', {
      title: `Choose the ${which === 'caption' ? 'captioning' : which} model folder`,
      properties: ['openDirectory'],
      buttonLabel: 'Use this folder'
    })
    if (!picked) return null
    const key = { image: 'model_dir', video: 'video_model_dir', caption: 'caption_model_dir' }[which]
    return (await api('PUT', '/api/settings', { [key]: picked })) as ImageSettings
  })

  ipcMain.handle('image:pickFfmpeg', async () => {
    const picked = await pickOpenPath('STUDIO_PICK_FFMPEG', {
      title: 'Choose ffmpeg',
      properties: ['openFile'],
      filters: process.platform === 'win32' ? [{ name: 'ffmpeg', extensions: ['exe'] }] : []
    })
    if (!picked) return null
    if (!/^ffmpeg(\.exe)?$/i.test(path.basename(picked))) throw new Error('That does not look like ffmpeg (ffmpeg.exe).')
    return (await api('PUT', '/api/settings', { ffmpeg_path: picked })) as ImageSettings
  })

  ipcMain.handle('image:pickPython', async () => {
    const result = await dialog.showOpenDialog({
      title: 'Choose the Python interpreter for the image engine',
      properties: ['openFile'],
      filters: process.platform === 'win32' ? [{ name: 'Python', extensions: ['exe'] }] : []
    })
    const picked = result.filePaths[0]
    if (result.canceled || !picked) return null
    if (!/^python(3(\.\d+)?)?(w)?(\.exe)?$/i.test(path.basename(picked))) {
      throw new Error('That does not look like a Python interpreter (python.exe).')
    }
    await store.update((c) => {
      c.image.pythonPath = picked
    })
    // A different interpreter means a different engine process; the old one must have
    // let go of the port first.
    await imageEngine.stop()
    void imageEngine.start().catch(() => undefined)
    return picked
  })

  // ---- Installing the engine and downloading models (shared/setup.ts)

  ipcMain.handle('image:setupState', () => setup.state)

  ipcMain.handle('image:installEngine', async () => {
    const busy = imageEngine.busyReason
    if (busy) throw new Error(`${busy} Wait for it to finish first.`)
    if (downloads.running) throw new Error('A model is being downloaded with this Python. Wait for it to finish first.')
    // pip cannot replace files the running engine has loaded (Windows locks them), so the
    // engine is stopped, waited for, and kept from restarting until the install is over.
    await imageEngine.hold('The engine is being installed.')
    let python: string
    try {
      python = await setup.install()
      await store.update((c) => {
        c.image.pythonPath = python
      })
    } finally {
      imageEngine.release()
    }
    void imageEngine.start().catch(() => undefined)
    return python
  })

  ipcMain.handle('image:cancelInstall', () => setup.cancel())

  ipcMain.handle('image:downloadState', () => downloads.state)

  const modelsFolder = (): string => store.get().image.modelsFolder ?? defaultModelsFolder

  ipcMain.handle('image:modelsFolder', () => modelsFolder())

  ipcMain.handle('image:chooseModelsFolder', async () => {
    const picked = await pickOpenPath('STUDIO_PICK_MODELS_FOLDER', {
      title: 'Choose where to keep downloaded models',
      properties: ['openDirectory', 'createDirectory'],
      buttonLabel: 'Keep models here'
    })
    if (!picked) return null
    await store.update((c) => {
      c.image.modelsFolder = picked
    })
    return picked
  })

  ipcMain.handle('image:downloadModel', async (_e, id: unknown) => {
    const model = catalogModel(id)
    if (!model) throw new Error('Unknown model.')
    if (setup.running) throw new Error('The engine is being installed. Wait for it to finish first.')
    try {
      await imageEngine.start()
    } catch {
      throw new Error('Install the image engine first (Settings → Engine), then download the model.')
    }
    await downloads.download(model.id, modelsFolder())
  })

  ipcMain.handle('image:cancelDownload', () => downloads.cancel())

  // ---- Inspecting a model file (shared/inspect.ts): header and hash, nothing loaded or run

  ipcMain.handle('image:inspectState', () => inspector.state)

  ipcMain.handle('image:inspectModelFile', async () => {
    if (inspector.running) throw new Error('A file is already being inspected.')
    if (setup.running) throw new Error('The engine is being installed. Wait for it to finish first.')
    const picked = await pickOpenPath('STUDIO_PICK_INSPECT_FILE', {
      title: 'Choose a model file to inspect',
      properties: ['openFile'],
      filters: [
        { name: 'Model files', extensions: ['safetensors', 'gguf', 'ckpt', 'pt', 'pth', 'bin'] },
        { name: 'All files', extensions: ['*'] }
      ],
      buttonLabel: 'Inspect'
    })
    if (!picked) return false
    await inspector.inspect(picked)
    return true
  })

  ipcMain.handle('image:cancelInspect', () => inspector.cancel())

  // ---- Importing a model from a single transformer file (shared/importing.ts)

  ipcMain.handle('image:importState', () => importer.state)

  ipcMain.handle('image:pickImportFile', async (_e, target: unknown, slot: unknown = 'first') => {
    if (target !== 'image' && target !== 'video') throw new Error('Invalid import target.')
    if (slot !== 'first' && slot !== 'second') throw new Error('Invalid import file.')
    if (importer.running) throw new Error('An import is already running.')
    if (setup.running) throw new Error('The engine is being installed. Wait for it to finish first.')
    const picked = await pickOpenPath(slot === 'second' ? 'STUDIO_PICK_IMPORT_FILE_2' : 'STUDIO_PICK_IMPORT_FILE', {
      title: slot === 'second' ? 'Choose the other expert’s file' : 'Choose a model file to import',
      properties: ['openFile'],
      filters: [{ name: 'safetensors', extensions: ['safetensors'] }],
      buttonLabel: 'Choose'
    })
    if (!picked) return false
    if (slot === 'second') await importer.pickSecond(picked)
    else await importer.pick(picked, target)
    return true
  })

  ipcMain.handle('image:swapImportExperts', () => importer.swap())

  ipcMain.handle('image:startImport', async (_e, options: unknown) => {
    if (!options || typeof options !== 'object') throw new Error('Invalid import options.')
    const o = options as Record<string, unknown>
    const optionalNumber = (v: unknown): v is number | null | undefined => v == null || typeof v === 'number'
    if (typeof o['sha256'] !== 'string' || typeof o['name'] !== 'string' || typeof o['acceptLicenses'] !== 'boolean'
        || (o['variant'] !== 'base' && o['variant'] !== 'distilled')
        || !(o['sha256_2'] == null || typeof o['sha256_2'] === 'string') || !optionalNumber(o['steps']) || !optionalNumber(o['shift'])) {
      throw new Error('Invalid import options.')
    }
    if (setup.running) throw new Error('The engine is being installed. Wait for it to finish first.')
    await importer.start({
      variant: o['variant'],
      sha256: o['sha256'],
      name: o['name'],
      acceptLicenses: o['acceptLicenses'],
      sha256_2: typeof o['sha256_2'] === 'string' ? o['sha256_2'] : '',
      steps: typeof o['steps'] === 'number' ? o['steps'] : null,
      shift: typeof o['shift'] === 'number' ? o['shift'] : null
    })
  })

  ipcMain.handle('image:cancelImport', () => importer.cancel())
  ipcMain.handle('image:clearImport', () => importer.clear())

  ipcMain.handle('image:openModelLink', async (_e, id: unknown, which: unknown) => {
    const model = catalogModel(id)
    if (!model || (which !== 'source' && which !== 'license')) throw new Error('Invalid link.')
    // Only the two addresses in the catalog, never one the window names.
    await openExternal(which === 'source' ? `https://huggingface.co/${model.repo}` : model.licenseUrl)
  })
}

const PASSED_HEADERS = ['Content-Type', 'Content-Length', 'Content-Range', 'Accept-Ranges']

/**
 * Serve gallery images, videos, last frames and start pictures (the `studio-image:`
 * URLs in shared/image.ts) from the engine, and nothing else. A byte range asked for
 * by the window is passed on, so a video player can seek.
 */
export async function serveImage(imageEngine: ImageEngine, url: string, range?: string | null): Promise<Response> {
  const route = engineImagePath(url)
  if (!route || imageEngine.state.state !== 'running') return new Response(null, { status: 404 })
  try {
    const asked = range && /^bytes=\d*-\d*$/.test(range) ? { Range: range } : undefined
    const upstream = await imageEngine.raw('GET', route, undefined, undefined, asked)
    if (!upstream.ok) return new Response(null, { status: upstream.status })
    const headers: Record<string, string> = {}
    for (const name of PASSED_HEADERS) {
      const value = upstream.headers.get(name)
      if (value) headers[name] = value
    }
    headers['Content-Type'] ??= 'application/octet-stream'
    return new Response(upstream.body, { status: upstream.status, headers })
  } catch {
    return new Response(null, { status: 502 })
  }
}
