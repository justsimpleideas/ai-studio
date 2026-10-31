import type { EngineModelKind, GalleryQuery, GenerationRequest, GifOptions, ImageSettings, MediaKind, SaveManyOptions, VideoRequest } from '@shared/image.js'
import type { ImportOptions, ImportTarget } from '@shared/importing.js'
import type { ModelId } from '@shared/setup.js'
import type { Subject } from '@shared/subject.js'
import type { CaptionView, CollectPlan, CompareRequest, TrainSettings } from '@shared/training.js'

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
  updateJob: (id: string, request: GenerationRequest | VideoRequest) => call(() => image().updateJob(id, request)),
  jobs: () => call(() => image().jobs()),
  gallery: (query: GalleryQuery) => call(() => image().gallery(query)),
  setFavorite: (id: string, favorite: boolean) => call(() => image().setFavorite(id, favorite)),
  deleteImage: (id: string) => call(() => image().deleteImage(id)),
  folders: () => call(() => image().folders()),
  createFolder: (name: string) => call(() => image().createFolder(name)),
  renameFolder: (id: string, name: string) => call(() => image().renameFolder(id, name)),
  deleteFolder: (id: string) => call(() => image().deleteFolder(id)),
  moveToFolder: (id: string, folderId: string | null) => call(() => image().moveToFolder(id, folderId)),
  moveManyToFolder: (ids: string[], folderId: string | null) => call(() => image().moveManyToFolder(ids, folderId)),
  reorderFolder: (id: string, before: string | null) => call(() => image().reorderFolder(id, before)),
  setFolderHidden: (id: string, hidden: boolean) => call(() => image().setFolderHidden(id, hidden)),
  setFolderPin: (id: string, pin: string | null, current: string | null) =>
    call(() => image().setFolderPin(id, pin, current)),
  unlockFolder: (id: string, pin: string) => call(() => image().unlockFolder(id, pin)),
  lockFolder: (id: string) => call(() => image().lockFolder(id)),
  pickMergeSource: () => call(() => image().pickMergeSource()),
  inspectMerge: (tokens: string[], renames: Record<string, string>) => call(() => image().inspectMerge(tokens, renames)),
  startMerge: (tokens: string[], renames: Record<string, string>) => call(() => image().startMerge(tokens, renames)),
  mergeStatus: () => call(() => image().mergeStatus()),
  cancelMerge: () => call(() => image().cancelMerge()),
  showHiddenFolders: () => call(() => image().showHiddenFolders()),
  setShowHiddenFolders: (show: boolean) => call(() => image().setShowHiddenFolders(show)),
  saveImageAs: (id: string) => call(() => image().saveImageAs(id)),
  saveManyTo: (ids: string[], options: SaveManyOptions) => call(() => image().saveManyTo(ids, options)),
  saveAsGif: (id: string, options: GifOptions) => call(() => image().saveAsGif(id, options)),
  includeSettingsInExports: () => call(() => image().includeSettingsInExports()),
  setIncludeSettingsInExports: (value: boolean) => call(() => image().setIncludeSettingsInExports(value)),
  pickModelFolder: (kind?: EngineModelKind) => call(() => image().pickModelFolder(kind)),
  pickFfmpeg: () => call(() => image().pickFfmpeg()),
  pickPython: () => call(() => image().pickPython()),
  start: () => call(() => image().start()),
  installEngine: () => call(() => image().installEngine()),
  cancelInstall: () => call(() => image().cancelInstall()),
  hardware: () => call(() => image().hardware()),
  modelsFolder: () => call(() => image().modelsFolder()),
  chooseModelsFolder: () => call(() => image().chooseModelsFolder()),
  downloadModel: (id: ModelId) => call(() => image().downloadModel(id)),
  cancelDownload: () => call(() => image().cancelDownload()),
  openModelLink: (id: ModelId, which: 'source' | 'license') => call(() => image().openModelLink(id, which)),
  inspectState: () => call(() => image().inspectState()),
  inspectModelFile: () => call(() => image().inspectModelFile()),
  cancelInspect: () => call(() => image().cancelInspect()),
  pickImportFile: (target: ImportTarget, slot: 'first' | 'second' = 'first') => call(() => image().pickImportFile(target, slot)),
  swapImportExperts: () => call(() => image().swapImportExperts()),
  startImport: (options: ImportOptions) => call(() => image().startImport(options)),
  cancelImport: () => call(() => image().cancelImport()),
  clearImport: () => call(() => image().clearImport())
}

const training = () => window.api.training

export const trainingApi = {
  projects: () => call(() => training().projects()),
  subjectPresets: () => call(() => training().subjectPresets()),
  createProject: (name: string, subject: Subject) => call(() => training().createProject(name, subject)),
  project: (id: string) => call(() => training().project(id)),
  setSubject: (id: string, subject: Subject) => call(() => training().setSubject(id, subject)),
  deleteProject: (id: string) => call(() => training().deleteProject(id)),
  scan: (id: string) => call(() => training().scan(id)),
  caption: (id: string) => call(() => training().caption(id)),
  captions: (id: string, view: CaptionView, offset: number, limit: number) =>
    call(() => training().captions(id, view, offset, limit)),
  review: (id: string, item: string, action: 'accept' | 'edit' | 'exclude', edits?: Record<string, string>) =>
    call(() => training().review(id, item, action, edits)),
  train: (id: string, settings: TrainSettings) => call(() => training().train(id, settings)),
  compare: (id: string, request: CompareRequest) => call(() => training().compare(id, request)),
  log: (id: string, run: string) => call(() => training().log(id, run)),
  useCheckpoint: (id: string, run: string, step: number, name: string) =>
    call(() => training().useCheckpoint(id, run, step, name)),
  loras: () => call(() => training().loras()),
  renameLora: (id: string, name: string) => call(() => training().renameLora(id, name)),
  deleteLora: (id: string) => call(() => training().deleteLora(id)),
  collectState: () => call(() => training().collectState()),
  collect: (plan: CollectPlan) => call(() => training().collect(plan)),
  checkCollect: (plan: CollectPlan) => call(() => training().checkCollect(plan)),
  cancelCollect: () => call(() => training().cancelCollect())
}
