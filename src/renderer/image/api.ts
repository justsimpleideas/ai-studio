import type { GalleryQuery, GenerationRequest, GifOptions, ImageSettings, MediaKind, VideoRequest } from '@shared/image.js'
import type { ModelId } from '@shared/setup.js'

/**
 * The image UI's calls, routed through the main process.
 *
 * Electron prefixes errors thrown in a handler with "Error invoking remote
 * method …"; that is stripped so the engine's own message reaches the user.
 */
export function cleanError(err: unknown): Error {
  const raw = err instanceof Error ? err.message : String(err)
  return new Error(raw.replace(/^Error invoking remote method '[^']+': (?:\w*Error: )?/, ''))
}

async function call<T>(run: () => Promise<T>): Promise<T> {
  try {
    return await run()
  } catch (err) {
    throw cleanError(err)
  }
}

const image = () => window.api.image

export const api = {
  status: () => call(() => image().status()),
  settings: () => call(() => image().settings()),
  saveSettings: (changes: Partial<ImageSettings>) => call(() => image().saveSettings(changes)),
  loadModel: (kind?: MediaKind) => call(() => image().loadModel(kind)),
  unloadModel: (kind?: MediaKind) => call(() => image().unloadModel(kind)),
  submit: (request: GenerationRequest) => call(() => image().submit(request)),
  submitVideo: (request: VideoRequest) => call(() => image().submitVideo(request)),
  pickStartImage: () => call(() => image().pickStartImage()),
  uploadStartImage: (data: Uint8Array) => call(() => image().uploadStartImage(data)),
  startImageFrom: (id: string, frame: 'image' | 'last') => call(() => image().startImageFrom(id, frame)),
  cancel: (id: string) => call(() => image().cancel(id)),
  jobs: () => call(() => image().jobs()),
  gallery: (query: GalleryQuery) => call(() => image().gallery(query)),
  setFavorite: (id: string, favorite: boolean) => call(() => image().setFavorite(id, favorite)),
  deleteImage: (id: string) => call(() => image().deleteImage(id)),
  folders: () => call(() => image().folders()),
  createFolder: (name: string) => call(() => image().createFolder(name)),
  renameFolder: (id: string, name: string) => call(() => image().renameFolder(id, name)),
  deleteFolder: (id: string) => call(() => image().deleteFolder(id)),
  moveToFolder: (id: string, folderId: string | null) => call(() => image().moveToFolder(id, folderId)),
  saveImageAs: (id: string) => call(() => image().saveImageAs(id)),
  saveAsGif: (id: string, options: GifOptions) => call(() => image().saveAsGif(id, options)),
  includeSettingsInExports: () => call(() => image().includeSettingsInExports()),
  setIncludeSettingsInExports: (value: boolean) => call(() => image().setIncludeSettingsInExports(value)),
  pickModelFolder: (kind?: MediaKind) => call(() => image().pickModelFolder(kind)),
  pickFfmpeg: () => call(() => image().pickFfmpeg()),
  pickPython: () => call(() => image().pickPython()),
  start: () => call(() => image().start()),
  installEngine: () => call(() => image().installEngine()),
  cancelInstall: () => call(() => image().cancelInstall()),
  modelsFolder: () => call(() => image().modelsFolder()),
  chooseModelsFolder: () => call(() => image().chooseModelsFolder()),
  downloadModel: (id: ModelId) => call(() => image().downloadModel(id)),
  cancelDownload: () => call(() => image().cancelDownload()),
  openModelLink: (id: ModelId, which: 'source' | 'license') => call(() => image().openModelLink(id, which))
}
