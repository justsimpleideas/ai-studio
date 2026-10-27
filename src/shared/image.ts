/**
 * The image side of the studio: types shared by the main process and the window,
 * and the small pure pieces of the bridge to the Python image engine. The same
 * engine makes videos, so they live here too.
 *
 * The window never talks to the engine. It calls named operations on the main
 * process, which forwards them over loopback HTTP with a per-launch token, and it
 * shows images and videos through the private `studio-image:` protocol (decisions S3).
 */

import type { EngineSetupState, ModelDownloadState, ModelId } from './setup.js'

export const IMAGE_ENGINE_PORT = 5198
export const IMAGE_SCHEME = 'studio-image'
export const IMAGE_TOKEN_HEADER = 'X-Studio-Token'
/** The largest start picture the window or main process will send to the engine (the engine's own limit). */
export const MAX_START_IMAGE_BYTES = 40 * 1024 * 1024

export type ImageModelState = 'unloaded' | 'loading' | 'ready' | 'error'
export type ImageJobStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled'
export type MemoryMode = 'auto' | 'full' | 'quality' | 'balanced' | 'low'
export type MediaKind = 'image' | 'video'

export interface GenerationRequest {
  prompt: string
  negative_prompt: string
  width: number
  height: number
  steps: number
  guidance: number
  /** -1 asks the engine for a random seed. */
  seed: number
  count: number
}

export interface VideoRequest {
  kind: 'video'
  prompt: string
  negative_prompt: string
  width: number
  height: number
  /** 4k + 1 frames, at the model's fixed frame rate (VideoInfo.limits.fps). */
  frames: number
  steps: number
  guidance: number
  /** -1 asks the engine for a random seed. */
  seed: number
  count: number
  /** An engine input id (StartImage.id): the video opens on that picture. */
  start_image: string | null
}

/** A picture stored in the engine for a video to start from. */
export interface StartImage {
  id: string
  width: number
  height: number
}

/**
 * A video model the engine supports (image_engine/video_models.py): "standard" is
 * sampled in 50 steps with guidance; "fast" is a 3-step distillation whose steps and
 * guidance are fixed by the model.
 */
export interface VideoProfile {
  name: 'standard' | 'fast'
  label: string
  steps: number
  guidance: number
  /** Steps and guidance are the model's own; the engine ignores those in a request. */
  fixed: boolean
  /** The clip length it was trained on. */
  trained_frames: number
}

export interface VideoInfo {
  limits: {
    min_side: number
    max_side: number
    side_multiple: number
    max_pixels: number
    min_frames: number
    max_frames: number
    frame_step: number
    fps: number
    max_steps: number
    max_count: number
  }
  defaults: { width: number; height: number; frames: number; steps: number; guidance: number; count: number }
  /** Which model the video folder holds; null until a supported one is chosen. Missing from older engines. */
  profile?: VideoProfile | null
  /** "webp" when no ffmpeg was found: videos are then saved as animated WebP. */
  encoder: 'mp4' | 'webp'
  /** The ffmpeg the engine would use right now, or null. */
  ffmpeg: string | null
}

export interface ModelInfo {
  id: string
  state: ImageModelState
  message: string
}

export interface DeviceInfo {
  name: string
  cuda: boolean
  total_bytes: number
  used_bytes: number
  memory_plan?: string
  /** False when the Python environment has no PyTorch at all. */
  torch?: boolean
}

export interface ImageStatus {
  version: string
  /** The image model. */
  model: ModelInfo
  /** Every model the engine has; "video" is missing from an engine without video support. */
  models?: Partial<Record<MediaKind, ModelInfo>>
  device: DeviceInfo
  queue: { queued: number; running: number }
  limits: { min_side: number; max_side: number; side_multiple: number; max_steps: number; max_count: number }
  defaults: { width: number; height: number; steps: number; guidance: number; count: number }
  video?: VideoInfo
}

export interface ImageJob {
  id: string
  /** Missing on jobs from an engine without video support: those are images. */
  kind?: MediaKind
  request: GenerationRequest | (VideoRequest & { fps: number })
  created: number
  status: ImageJobStatus
  stage: string
  index: number
  count: number
  step: number
  steps: number
  images: string[]
  error: string | null
  started: number | null
  finished: number | null
}

export interface ImageRecord {
  id: string
  created: number
  job_id: string
  model: string
  prompt: string
  negative_prompt: string
  width: number
  height: number
  steps: number
  guidance: number
  seed: number
  duration_ms: number
  favorite: boolean
  /** `encoder` says how a video file was made (MP4 through ffmpeg, or animated WebP). */
  stats: { peak_vram_bytes?: number; memory_plan?: string; encoder?: string }
  /** What made it, when known: app version, exact model revision ("repo@commit"), library versions. */
  provenance: { app?: string; model_revision?: string; libraries?: Record<string, string> }
  /** For a video, a still (its thumbnail). */
  image_url: string
  thumb_url: string
  file_name: string
  /** Missing on records from an engine without video support: those are images. */
  media?: MediaKind
  mime?: string
  frames?: number
  fps?: number
  start_image?: string | null
  /** Videos only. */
  video_url?: string
  /** Videos only: the last frame as a PNG, what "continue" starts from. */
  last_frame_url?: string
  /** The folder it sits in, or null for none. Missing from engines without folders. */
  folder_id?: string | null
}

/** A user-made folder in the gallery. Folders group the index only; files never move. */
export interface GalleryFolder {
  id: string
  name: string
  created: number
  /** How many images and videos it holds. */
  count: number
}

/** The longest folder name the engine accepts. */
export const FOLDER_NAME_MAX = 60

/** How a video is turned into a GIF: at most `width` wide (never enlarged), at `fps`. */
export interface GifOptions {
  width: number
  fps: number
}

/** The limits the engine accepts for a GIF export (image_engine/video_io.py). */
export const GIF_MIN_WIDTH = 64
export const GIF_MAX_WIDTH = 1280
export const GIF_MAX_FPS = 24
/** Small and a little slower than the video: GIFs get large quickly. */
export const DEFAULT_GIF_OPTIONS: GifOptions = { width: 480, fps: 12 }

export interface ImageSettings {
  model_dir: string
  memory_mode: MemoryMode
  load_on_start: boolean
  video_model_dir: string
  video_memory_mode: MemoryMode
  /** Empty: found automatically (PATH, then where winget installs it). */
  ffmpeg_path: string
}

export interface GalleryQuery {
  offset?: number
  limit?: number
  q?: string
  /** Every favourite, whatever folder it is in (unless `folder` is given too). */
  favorites?: boolean
  media?: MediaKind
  /** One folder's items only. */
  folder?: string
}

export interface GalleryPage {
  total: number
  items: ImageRecord[]
}

export type ImageEngineEvent =
  | { type: 'hello'; status: ImageStatus; jobs: ImageJob[] }
  | { type: 'job'; job: ImageJob }
  | { type: 'image'; job_id: string; image: ImageRecord }
  /** `kind` is missing from an engine without video support: that is the image model. */
  | { type: 'model'; kind?: MediaKind; state: ImageModelState; message: string }

/** The engine process itself, as opposed to the model inside it. */
export type ImageEngineState =
  | { state: 'stopped' }
  | { state: 'starting' }
  | { state: 'running' }
  | { state: 'failed'; message: string }

/** What the window may ask of the image side. */
export interface ImageApi {
  engineState(): Promise<ImageEngineState>
  /** Start the engine if it is not running. Resolves once it answers, or rejects with the reason. */
  start(): Promise<void>
  status(): Promise<ImageStatus>
  settings(): Promise<ImageSettings>
  saveSettings(changes: Partial<ImageSettings>): Promise<ImageSettings>
  /** Without a kind: the image model. */
  loadModel(kind?: MediaKind): Promise<void>
  /** Without a kind: every model, which frees the GPU. */
  unloadModel(kind?: MediaKind): Promise<void>
  submit(request: GenerationRequest): Promise<ImageJob>
  submitVideo(request: VideoRequest): Promise<ImageJob>
  /** Opens a file picker for a start picture and stores it in the engine; null if cancelled. */
  pickStartImage(): Promise<StartImage | null>
  /** Stores a picture the window already has (dropped on it) as a start picture. */
  uploadStartImage(data: Uint8Array): Promise<StartImage>
  /** A gallery image, or a video's last frame, as a start picture. */
  startImageFrom(id: string, frame: 'image' | 'last'): Promise<StartImage>
  cancel(jobId: string): Promise<ImageJob>
  jobs(): Promise<ImageJob[]>
  gallery(query: GalleryQuery): Promise<GalleryPage>
  setFavorite(id: string, favorite: boolean): Promise<ImageRecord>
  deleteImage(id: string): Promise<void>
  folders(): Promise<GalleryFolder[]>
  createFolder(name: string): Promise<GalleryFolder>
  renameFolder(id: string, name: string): Promise<GalleryFolder>
  /** Removes the folder only: what was in it stays in the gallery, in no folder. */
  deleteFolder(id: string): Promise<{ moved: number }>
  /** Put an image or video in a folder, or (null) in none. */
  moveToFolder(id: string, folderId: string | null): Promise<ImageRecord>
  /**
   * Opens a save dialog in the main process; resolves to the saved path or null
   * if cancelled. The prompt and settings are left out of the saved file unless
   * includeSettingsInExports is on.
   */
  saveImageAs(id: string): Promise<string | null>
  /** Opens a save dialog, then has the engine turn the video into a GIF; the saved path, or null if cancelled. */
  saveAsGif(id: string, options: GifOptions): Promise<string | null>
  includeSettingsInExports(): Promise<boolean>
  setIncludeSettingsInExports(value: boolean): Promise<boolean>
  /** Opens a folder picker for the image (or video) model folder and saves the choice. */
  pickModelFolder(kind?: MediaKind): Promise<ImageSettings | null>
  /** Opens a file picker for ffmpeg and saves the choice. */
  pickFfmpeg(): Promise<ImageSettings | null>
  /** Opens a file picker for the Python interpreter and restarts the engine with it. */
  pickPython(): Promise<string | null>
  onEvent(handler: (event: ImageEngineEvent) => void): () => void
  onEngineState(handler: (state: ImageEngineState) => void): () => void

  /** Installing the engine's Python environment (shared/setup.ts). */
  setupState(): Promise<EngineSetupState>
  /** Installs or repairs the environment, then restarts the engine with it. Resolves with its interpreter. */
  installEngine(): Promise<string>
  cancelInstall(): Promise<void>
  onSetupState(handler: (state: EngineSetupState) => void): () => void

  /** Downloading a supported model into the models folder. */
  downloadState(): Promise<ModelDownloadState>
  /** Where downloads go: the chosen folder, or the default one. */
  modelsFolder(): Promise<string>
  /** Opens a folder picker for where downloads go; null if cancelled. */
  chooseModelsFolder(): Promise<string | null>
  /** Downloads, verifies and selects the model. Resolves when done or stopped. */
  downloadModel(id: ModelId): Promise<void>
  cancelDownload(): Promise<void>
  /** Opens the model's source page or its license in the browser. */
  openModelLink(id: ModelId, which: 'source' | 'license'): Promise<void>
  onDownloadState(handler: (state: ModelDownloadState) => void): () => void
}

// ---------------------------------------------------------------------------
// Pure helpers (tested in tests/image-bridge.test.ts)

const ID = /^[0-9a-f]{16}$/

export function isEngineId(value: unknown): value is string {
  return typeof value === 'string' && ID.test(value)
}

/** The window-facing address of an image or its thumbnail. */
export function studioImageUrl(id: string, kind: 'image' | 'thumb'): string {
  return `${IMAGE_SCHEME}://gallery/${id}.${kind === 'image' ? 'png' : 'webp'}`
}

/** The window-facing address of a video, a video's last frame, or a start picture. */
export function studioMediaUrl(id: string, kind: 'video' | 'last' | 'input'): string {
  if (kind === 'video') return `${IMAGE_SCHEME}://video/${id}`
  if (kind === 'last') return `${IMAGE_SCHEME}://frame/${id}-last.png`
  return `${IMAGE_SCHEME}://input/${id}.png`
}

const MEDIA_ROUTES: [RegExp, (m: RegExpExecArray) => string][] = [
  [/^studio-image:\/\/gallery\/([0-9a-f]{16})\.(png|webp)$/, (m) => `/images/${m[1]}.${m[2]}`],
  [/^studio-image:\/\/video\/([0-9a-f]{16})$/, (m) => `/videos/${m[1]}`],
  [/^studio-image:\/\/frame\/([0-9a-f]{16})-last\.png$/, (m) => `/frames/${m[1]}-last.png`],
  [/^studio-image:\/\/input\/([0-9a-f]{16})\.png$/, (m) => `/inputs/${m[1]}.png`]
]

/** Parse a `studio-image:` URL back to an engine path, refusing anything else. */
export function engineImagePath(url: string): string | null {
  for (const [pattern, route] of MEDIA_ROUTES) {
    const match = pattern.exec(url)
    if (match) return route(match)
  }
  return null
}

export function isVideo(record: Pick<ImageRecord, 'media'>): boolean {
  return record.media === 'video'
}

/** Point an engine record's URLs at the private protocol. */
export function rewriteRecord(record: ImageRecord): ImageRecord {
  const thumb = studioImageUrl(record.id, 'thumb')
  if (isVideo(record)) {
    return {
      ...record,
      image_url: thumb,
      thumb_url: thumb,
      video_url: studioMediaUrl(record.id, 'video'),
      last_frame_url: studioMediaUrl(record.id, 'last')
    }
  }
  return { ...record, image_url: studioImageUrl(record.id, 'image'), thumb_url: thumb }
}

/** What a start picture's bytes really are, from their first bytes; null for anything else. */
export function sniffImageType(bytes: Uint8Array): 'image/png' | 'image/jpeg' | 'image/webp' | null {
  const at = (offset: number, ...values: number[]): boolean => values.every((v, i) => bytes[offset + i] === v)
  if (at(0, 0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a)) return 'image/png'
  if (at(0, 0xff, 0xd8, 0xff)) return 'image/jpeg'
  if (at(0, 0x52, 0x49, 0x46, 0x46) && at(8, 0x57, 0x45, 0x42, 0x50)) return 'image/webp' // RIFF....WEBP
  return null
}

/** GIF options from the window, checked field by field; throws on anything else. */
export function parseGifOptions(value: unknown): GifOptions {
  const ok = (n: unknown, low: number, high: number): n is number => Number.isInteger(n) && (n as number) >= low && (n as number) <= high
  if (!value || typeof value !== 'object') throw new Error('Invalid GIF options.')
  const { width, fps } = value as Record<string, unknown>
  if (!ok(width, GIF_MIN_WIDTH, GIF_MAX_WIDTH) || !ok(fps, 1, GIF_MAX_FPS)) throw new Error('Invalid GIF options.')
  return { width, fps }
}

/** The GIF's size for a video of `width` x `height` (as the engine computes it). */
export function gifSize(width: number, height: number, asked: number): { width: number; height: number } {
  const w = Math.min(asked, width)
  return { width: w, height: Math.max(1, Math.round((height * w) / width)) }
}

/** "2026-10-26_1234.mp4" -> "2026-10-26_1234.gif". */
export function gifFileName(fileName: string): string {
  const dot = fileName.lastIndexOf('.')
  return `${dot > 0 ? fileName.slice(0, dot) : fileName || 'video'}.gif`
}

export function rewriteEvent(event: ImageEngineEvent): ImageEngineEvent {
  return event.type === 'image' ? { ...event, image: rewriteRecord(event.image) } : event
}

/**
 * Incremental Server-Sent Events parser for the engine's event stream.
 *
 * Feed it raw text as it arrives; it returns the complete `data:` payloads and
 * keeps any partial event for the next chunk. Comment lines (keep-alives) are
 * skipped.
 */
export class SseParser {
  private buffer = ''

  feed(chunk: string): string[] {
    this.buffer += chunk.replace(/\r\n/g, '\n')
    const out: string[] = []
    let boundary: number
    while ((boundary = this.buffer.indexOf('\n\n')) >= 0) {
      const block = this.buffer.slice(0, boundary)
      this.buffer = this.buffer.slice(boundary + 2)
      const data = block
        .split('\n')
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(line.startsWith('data: ') ? 6 : 5))
      if (data.length > 0) out.push(data.join('\n'))
    }
    return out
  }
}
