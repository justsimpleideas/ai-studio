import { useCallback, useEffect, useRef, useState } from 'react'
import { MAX_START_IMAGE_BYTES } from '@shared/image.js'
import { api } from './api.js'
import { GalleryView } from './components/GalleryView.js'
import { GeneratePanel } from './components/GeneratePanel.js'
import { ImageDetail } from './components/ImageDetail.js'
import { Preview } from './components/Preview.js'
import { SettingsView } from './components/SettingsView.js'
import { StatusChip } from './components/StatusChip.js'
import { VideoPanel } from './components/VideoPanel.js'
import { VideoPreview } from './components/VideoPreview.js'
import { fromImage, loadForm, saveForm, toRequest, type FormState } from './form.js'
import { MAX_PNG_BYTES, settingsFromPng } from './png-settings.js'
import type { ImageRecord, StartImage } from './types.js'
import { activeJob, jobKind, queuedCount, useEngine } from './useEngine.js'
import { useFolders } from './useFolders.js'
import { fromVideo, loadVideoForm, saveVideoForm, toVideoRequest, withStart, type VideoFormState } from './videoForm.js'
import './styles.css'

type Tab = 'generate' | 'video' | 'gallery' | 'settings'
type GalleryUpdate = ImageRecord | { id: string; deleted: true }
type Note = { ok: boolean; text: string } | null

const TAB_LABEL: Record<Tab, string> = { generate: 'Image', video: 'Video', gallery: 'Gallery', settings: 'Settings' }
const PICTURE_TYPES = ['image/png', 'image/jpeg', 'image/webp']

export function ImageStudio({ active }: { active: boolean }): React.JSX.Element {
  const [engine, dispatch] = useEngine(active)
  const [tab, setTab] = useState<Tab>('generate')
  const [form, setForm] = useState<FormState>(() => loadForm())
  const [videoForm, setVideoForm] = useState<VideoFormState>(() => loadVideoForm())
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [videoError, setVideoError] = useState<string | null>(null)
  const [detail, setDetail] = useState<{ image: ImageRecord; list: ImageRecord[] } | null>(null)
  const [note, setNote] = useState<Note>(null)
  const [dragging, setDragging] = useState(false)
  const galleryUpdater = useRef<((u: GalleryUpdate) => void) | null>(null)
  const folders = useFolders(engine.connected)

  useEffect(() => saveForm(form), [form])
  useEffect(() => saveVideoForm(videoForm), [videoForm])

  // A success note fades on its own; a refusal stays until dismissed.
  useEffect(() => {
    if (!note?.ok) return
    const timer = window.setTimeout(() => setNote(null), 9000)
    return () => window.clearTimeout(timer)
  }, [note])

  const job = activeJob(engine.jobs, 'image')
  const videoJob = activeJob(engine.jobs, 'video')
  const pending = queuedCount(engine.jobs)
  const lastFailed = (kind: 'image' | 'video') =>
    Object.values(engine.jobs)
      .filter((j) => j.status === 'failed' && jobKind(j) === kind)
      .sort((a, b) => (b.finished ?? 0) - (a.finished ?? 0))[0]
  const failed = lastFailed('image')
  const failedVideo = lastFailed('video')
  const sessionImages = engine.session.filter((i) => i.media !== 'video')
  const sessionVideos = engine.session.filter((i) => i.media === 'video')

  const generate = async (): Promise<void> => {
    setSubmitError(null)
    try {
      await api.submit(toRequest(form))
    } catch (e) {
      setSubmitError((e as Error).message)
    }
  }

  const generateVideo = async (): Promise<void> => {
    setVideoError(null)
    try {
      await api.submitVideo(toVideoRequest(videoForm))
    } catch (e) {
      setVideoError((e as Error).message)
    }
  }

  const updateEverywhere = (update: GalleryUpdate): void => {
    galleryUpdater.current?.(update)
    if ('deleted' in update) dispatch({ type: 'forget-image', id: update.id })
    else dispatch({ type: 'replace-image', image: update })
  }

  const toggleFavorite = async (image: ImageRecord): Promise<void> => {
    const updated = await api.setFavorite(image.id, !image.favorite)
    updateEverywhere(updated)
    setDetail((d) => d && { image: updated, list: d.list.map((i) => (i.id === updated.id ? updated : i)) })
  }

  const moveToFolder = async (image: ImageRecord, folderId: string | null): Promise<void> => {
    try {
      const updated = await folders.move(image.id, folderId)
      dispatch({ type: 'replace-image', image: updated })
      setDetail((d) => d && { image: updated, list: d.list.map((i) => (i.id === updated.id ? updated : i)) })
    } catch (e) {
      setNote({ ok: false, text: (e as Error).message })
    }
  }

  const remove = async (image: ImageRecord): Promise<void> => {
    await api.deleteImage(image.id)
    updateEverywhere({ id: image.id, deleted: true })
    if (image.folder_id) void folders.refresh() // its folder's count changed
    setDetail((d) => {
      if (!d) return d
      const index = d.list.findIndex((i) => i.id === image.id)
      const list = d.list.filter((i) => i.id !== image.id)
      const next = list[Math.min(index, list.length - 1)]
      return next ? { image: next, list } : null
    })
  }

  const reuse = (record: ImageRecord): void => {
    if (record.media === 'video') {
      setVideoForm((f) => fromVideo(record, f))
      setTab('video')
    } else {
      setForm((f) => fromImage(record, f))
      setTab('generate')
    }
    setDetail(null)
  }

  /** Put a start picture into the video form and show the Video tab. */
  const applyStart = (start: StartImage, change?: (f: VideoFormState) => Partial<VideoFormState>): void => {
    setVideoForm((f) => ({ ...withStart(f, start), ...change?.(f) }))
    setDetail(null)
    setTab('video')
  }

  const startAction = async (run: () => Promise<StartImage | null>, done?: (start: StartImage) => void): Promise<void> => {
    try {
      const start = await run()
      if (start) done ? done(start) : applyStart(start)
    } catch (e) {
      setNote({ ok: false, text: (e as Error).message })
    }
  }

  const animate = (image: ImageRecord): void =>
    void startAction(
      () => api.startImageFrom(image.id, 'image'),
      // The image's prompt describes the scene; it is a starting point if the video prompt is empty.
      (start) => applyStart(start, (f) => (f.prompt.trim() ? {} : { prompt: image.prompt }))
    )

  const continueVideo = (video: ImageRecord): void =>
    void startAction(
      () => api.startImageFrom(video.id, 'last'),
      // Same size and prompt, so the new clip joins the old one; a fresh seed.
      (start) => applyStart(start, () => ({ prompt: video.prompt, width: video.width, height: video.height, randomSeed: true }))
    )

  /** Take the settings saved inside a PNG (dropped, or chosen with the button). */
  const importPng = async (file: File): Promise<void> => {
    const result =
      file.size > MAX_PNG_BYTES
        ? ({ ok: false, reason: 'That file is too large to be an image from this studio.' } as const)
        : await settingsFromPng(new Uint8Array(await file.arrayBuffer()))
    if (!result.ok) {
      setNote({ ok: false, text: result.reason })
      return
    }
    setForm((f) => ({ ...f, ...result.settings }))
    setDetail(null)
    setTab('generate')
    setNote({
      ok: true,
      text:
        result.settings.randomSeed === false
          ? `Settings taken from ${file.name}. The seed is locked, so Generate makes the same image again.`
          : `Settings taken from ${file.name}. It had no seed saved, so a new one will be used.`
    })
  }

  /** A picture dropped on the Video tab becomes the start picture. */
  const dropStart = async (file: File): Promise<void> => {
    if (!PICTURE_TYPES.includes(file.type)) {
      setNote({ ok: false, text: 'A start picture must be a PNG, JPEG or WebP image.' })
      return
    }
    if (file.size > MAX_START_IMAGE_BYTES) {
      setNote({ ok: false, text: `That picture is larger than ${MAX_START_IMAGE_BYTES / 2 ** 20} MB.` })
      return
    }
    const bytes = new Uint8Array(await file.arrayBuffer())
    await startAction(() => api.uploadStartImage(bytes))
  }

  const hasFiles = (e: React.DragEvent): boolean => Array.from(e.dataTransfer.types).includes('Files')
  const dragOver = (e: React.DragEvent): void => {
    if (!hasFiles(e)) return
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
    setDragging(true)
  }

  const registerUpdater = useCallback((fn: (u: GalleryUpdate) => void) => {
    galleryUpdater.current = fn
  }, [])

  const engineDown = engine.engine.state === 'failed' || engine.engine.state === 'stopped'

  // No environment chosen or installed yet: a first run, not a fault. Either Python
  // could not start the engine at all, or a Python on PATH runs it without PyTorch.
  const [notInstalled, setNotInstalled] = useState(false)
  useEffect(() => {
    void window.api.getConfig().then((c) => setNotInstalled(c.image.pythonPath === null))
  }, [engine.engine.state])
  const needsSetup =
    (engine.engine.state === 'failed' && notInstalled) ||
    (engine.engine.state === 'running' && engine.status?.device.torch === false)
  const onVideo = tab === 'video'

  return (
    <div
      className="image-studio"
      onDragEnter={dragOver}
      onDragOver={dragOver}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDragging(false)
      }}
      onDrop={(e) => {
        if (!hasFiles(e)) return
        e.preventDefault()
        setDragging(false)
        const file = e.dataTransfer.files[0]
        if (!file) return
        void (onVideo ? dropStart(file) : importPng(file))
      }}
    >
      {dragging && (
        <div className="drop-overlay" aria-hidden>
          <div>{onVideo ? 'Drop a picture to start the video from it' : 'Drop a PNG to use its settings'}</div>
        </div>
      )}
      <header className="topbar">
        <nav className="tabs" aria-label="Image sections">
          {(Object.keys(TAB_LABEL) as Tab[]).map((t) => (
            <button
              key={t}
              className={tab === t ? 'active' : ''}
              aria-current={tab === t ? 'page' : undefined}
              onClick={() => setTab(t)}
            >
              {TAB_LABEL[t]}
            </button>
          ))}
        </nav>
        <StatusChip engine={engine.engine} connected={engine.connected} status={engine.status} />
      </header>

      {(engine.engine.state === 'failed' || needsSetup) && (
        <div className="banner" role="alert">
          {needsSetup ? (
            <>
              <strong>The image and video side needs a one-time setup:</strong> install its engine, then download a model.
            </>
          ) : (
            <>
              <strong>The image engine is not running.</strong> <span className="pre">{engine.engine.state === 'failed' ? engine.engine.message : ''}</span>
            </>
          )}{' '}
          <button className="link-button" onClick={() => setTab('settings')}>
            {needsSetup ? 'Set it up' : 'Open settings'}
          </button>
        </div>
      )}

      {note && (
        <div className={`banner ${note.ok ? 'ok' : ''}`} role="status">
          {note.text}{' '}
          <button className="link-button" onClick={() => setNote(null)}>
            Dismiss
          </button>
        </div>
      )}

      <main className={`content ${tab === 'video' ? 'generate video' : tab}`}>
        {tab === 'generate' && (
          <>
            <GeneratePanel
              form={form}
              onChange={setForm}
              onGenerate={() => void generate()}
              canGenerate={engine.connected}
              pending={pending}
              error={submitError ?? (failed && !job ? `Last job failed: ${failed.error ?? 'unknown error'}` : null)}
              onImport={(file) => void importPng(file)}
            />
            <Preview
              job={job}
              images={sessionImages}
              starting={engine.engine.state === 'starting'}
              modelMessage={engine.status?.model.message ?? ''}
              onOpen={(image, list) => setDetail({ image, list })}
              onCancel={(j) => void api.cancel(j.id).catch(() => undefined)}
            />
          </>
        )}
        {tab === 'video' && (
          <>
            <VideoPanel
              form={videoForm}
              onChange={setVideoForm}
              onGenerate={() => void generateVideo()}
              canGenerate={engine.connected}
              pending={pending}
              error={videoError ?? (failedVideo && !videoJob ? `Last video failed: ${failedVideo.error ?? 'unknown error'}` : null)}
              video={engine.status?.video}
              onPickStart={() => void startAction(() => api.pickStartImage())}
            />
            <VideoPreview
              job={videoJob}
              videos={sessionVideos}
              starting={engine.engine.state === 'starting'}
              modelMessage={engine.status?.models?.video?.message ?? ''}
              onOpen={(video, list) => setDetail({ image: video, list })}
              onCancel={(j) => void api.cancel(j.id).catch(() => undefined)}
            />
          </>
        )}
        {tab === 'gallery' &&
          (engineDown ? (
            <div className="empty-gallery muted">The gallery opens once the image engine is running.</div>
          ) : (
            <GalleryView
              version={`${engine.galleryVersion}:${folders.version}`}
              folders={folders}
              onOpen={(image, list) => setDetail({ image, list })}
              registerUpdater={registerUpdater}
              onError={(text) => setNote({ ok: false, text })}
            />
          ))}
        {tab === 'settings' && <SettingsView status={engine.status} engine={engine.engine} />}
      </main>

      {detail && (
        <ImageDetail
          image={detail.image}
          list={detail.list}
          onNavigate={(image) => setDetail((d) => d && { ...d, image })}
          onClose={() => setDetail(null)}
          onReuse={reuse}
          onFavorite={(img) => void toggleFavorite(img).catch(() => undefined)}
          onDelete={(img) => void remove(img).catch(() => undefined)}
          onSaveAs={(img) => api.saveImageAs(img.id)}
          onSaveAsGif={(video, options) => api.saveAsGif(video.id, options)}
          onAnimate={animate}
          onContinue={continueVideo}
          folders={folders.list}
          onMove={(img, folderId) => void moveToFolder(img, folderId)}
        />
      )}
    </div>
  )
}
