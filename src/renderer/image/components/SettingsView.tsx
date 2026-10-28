import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import type { ImageEngineState } from '@shared/image.js'
import { api } from '../api.js'
import { formatBytes } from '../format.js'
import type { MemoryMode, Settings, Status } from '../types.js'
import { EngineSetupPanel } from './EngineSetupPanel.js'
import { ModelDownloadPanel } from './ModelDownloadPanel.js'
import { ImportModelPanel } from './ImportModelPanel.js'
import { ModelInspectPanel } from './ModelInspectPanel.js'

const MODES: { value: MemoryMode; label: string; detail: string }[] = [
  { value: 'auto', label: 'Automatic', detail: 'Picks the fastest plan that fits this GPU.' },
  { value: 'full', label: 'Everything on GPU', detail: 'Fastest. Needs about 24 GB of VRAM.' },
  { value: 'quality', label: 'Full precision', detail: 'Image model in bf16, text encoder swapped in per prompt. About 16 GB.' },
  { value: 'balanced', label: 'Balanced', detail: 'Image model stored in fp8, text encoder swapped in per prompt. About 10 GB.' },
  { value: 'low', label: 'Low VRAM', detail: 'Moves each part to the GPU only while it runs. Slowest, about 6 GB.' }
]

// The same modes for the 12B image model, whose weights stay in fp8 in every one (memory.py, LARGE_PLANS).
const LARGE_MODES: { value: MemoryMode; label: string; detail: string }[] = [
  { value: 'auto', label: 'Automatic', detail: 'Picks the fastest plan that fits this GPU.' },
  { value: 'full', label: 'Everything on GPU', detail: 'Fastest. Needs about 26 GB of VRAM.' },
  { value: 'quality', label: 'Model on GPU', detail: 'Transformer resident in fp8, text encoder on the CPU. About 17 GB.' },
  {
    value: 'balanced',
    label: 'Balanced',
    detail: 'Transformer streamed from RAM block by block, text encoder swapped in per prompt. About 11 GB, plus about 22 GB of RAM.'
  },
  { value: 'low', label: 'Low VRAM', detail: 'Transformer streamed, text encoder on the CPU. Slowest, about 4 GB.' }
]

const VIDEO_MODES: { value: MemoryMode; label: string; detail: string }[] = [
  { value: 'auto', label: 'Automatic', detail: 'Picks the fastest plan that fits this GPU.' },
  { value: 'full', label: 'Everything on GPU', detail: 'Fastest. Needs about 32 GB of VRAM.' },
  { value: 'quality', label: 'Full precision', detail: 'Video model in bf16 on the GPU, text encoder on the CPU. About 18 GB.' },
  {
    value: 'balanced',
    label: 'Balanced',
    detail: 'Video model stored in fp8, text encoder on the CPU, decoder swapped in at the end. About 10 GB.'
  },
  { value: 'low', label: 'Low VRAM', detail: 'Streams the video model from RAM one block at a time. Slowest, for 8 GB cards.' }
]

// The 14B video models: two experts of ~14 GB each in fp8, one running at a time (memory.py).
const TWO_EXPERT_MODES: { value: MemoryMode; label: string; detail: string }[] = [
  { value: 'auto', label: 'Automatic', detail: 'Picks the fastest plan that fits this GPU.' },
  { value: 'full', label: 'Everything on GPU', detail: 'Fastest. Needs about 40 GB of VRAM.' },
  { value: 'quality', label: 'Experts on GPU', detail: 'The running expert resident in fp8, text encoder on the CPU. About 22 GB.' },
  {
    value: 'balanced',
    label: 'Balanced',
    detail:
      'Experts streamed from RAM block by block, text encoder on the CPU. Fits 12 GB cards. Both experts stay in RAM ' +
      'when there is room (about 48 GB); otherwise each is read from disk when it takes over.'
  },
  { value: 'low', label: 'Low VRAM', detail: 'As Balanced, but never more than one expert in RAM: for machines short of memory.' }
]

/**
 * Settings search: the cards and the settings in them whose text has every word of the query
 * (labels, values, hints and buttons alike). A card whose title matches stays whole. A hint
 * standing on its own after a setting goes with that setting. Done on the rendered cards, so
 * a setting added later is found without being listed anywhere.
 */
export function filterSettings(root: HTMLElement, empty: HTMLElement, query: string): void {
  const words = query.trim().toLowerCase().split(/\s+/).filter(Boolean)
  const has = (el: Element): boolean => {
    const text = `${el.textContent ?? ''} ${Array.from(el.querySelectorAll('input')).map((i) => i.placeholder).join(' ')}`.toLowerCase()
    return words.every((w) => text.includes(w))
  }
  let shown = 0
  for (const card of Array.from(root.children).filter((el) => el.classList.contains('card'))) {
    const title = card.querySelector(':scope > h2')
    // Each setting with the hints right after it.
    const groups: Element[][] = []
    for (const child of Array.from(card.children)) {
      if (child === title) continue
      if (child.classList.contains('hint') && groups.length) groups.at(-1)!.push(child)
      else groups.push([child])
    }
    const whole = words.length === 0 || (title !== null && has(title))
    let any = whole
    for (const group of groups) {
      const match = whole || group.some(has)
      any ||= match
      for (const el of group) el.classList.toggle('search-miss', !match)
    }
    card.classList.toggle('search-miss', !any)
    if (any) shown++
  }
  empty.hidden = words.length === 0 || shown > 0
}

interface Props {
  status: Status | null
  engine: ImageEngineState
  /** Changes when folders were hidden, shown, locked or unlocked (an engine event). */
  privacyVersion?: number
}

type Message = { kind: 'ok' | 'error'; text: string } | null

export function SettingsView({ status, engine, privacyVersion = 0 }: Props): React.JSX.Element {
  const [settings, setSettings] = useState<Settings | null>(null)
  const [python, setPython] = useState<string | null>(null)
  const [message, setMessage] = useState<Message>(null)
  const [includeSettings, setIncludeSettings] = useState(false)
  const [showHidden, setShowHidden] = useState(false)
  const running = engine.state === 'running'

  useEffect(() => {
    if (running) void api.showHiddenFolders().then(setShowHidden).catch(() => undefined)
  }, [running, privacyVersion])

  useEffect(() => {
    void window.api.getConfig().then((c) => {
      setPython(c.image.pythonPath)
      setIncludeSettings(c.image.includeSettingsInExports)
    })
  }, [])

  const reload = (): void => {
    api.settings().then(setSettings).catch((e: Error) => setMessage({ kind: 'error', text: e.message }))
  }

  useEffect(() => {
    if (running) reload()
  }, [running])

  const act = async (run: () => Promise<unknown>, ok: string): Promise<void> => {
    try {
      await run()
      setMessage({ kind: 'ok', text: ok })
    } catch (e) {
      setMessage({ kind: 'error', text: (e as Error).message })
    }
  }

  const save = (changes: Partial<Settings>): Promise<void> =>
    act(async () => setSettings(await api.saveSettings(changes)), 'Saved. Changes apply the next time the model loads.')

  const model = status?.model
  const imageProfile = status?.image?.profile ?? null
  const videoModel = status?.models?.video
  const video = status?.video
  const device = status?.device

  const [query, setQuery] = useState('')
  const sectionRef = useRef<HTMLElement>(null)
  const emptyRef = useRef<HTMLParagraphElement>(null)
  // After every render, since what the cards hold changes with the engine's state.
  useLayoutEffect(() => {
    if (sectionRef.current && emptyRef.current) filterSettings(sectionRef.current, emptyRef.current, query)
  })

  return (
    <section className="settings" ref={sectionRef}>
      <div className="settings-search">
        <input
          className="input"
          type="search"
          placeholder="Search settings"
          aria-label="Search settings"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Escape') setQuery('')
          }}
        />
        <p className="muted small search-empty" ref={emptyRef} hidden>
          No setting matches “{query.trim()}”.
        </p>
      </div>

      <div className="card">
        <h2>Engine</h2>
        <div className="model-row">
          <div>
            <div className="strong">
              {engine.state === 'running' ? 'Running' : engine.state === 'starting' ? 'Starting…' : engine.state === 'failed' ? 'Not running' : 'Stopped'}
            </div>
            {engine.state === 'failed' && <div className="error-text small pre">{engine.message}</div>}
          </div>
          <button onClick={() => void act(() => api.start(), 'Image engine started.')} disabled={running || engine.state === 'starting'}>
            Start
          </button>
        </div>
        <div className="field">
          <span className="field-label">Python environment</span>
          <div className="row-gap">
            <span className="mono grow path">{python ?? 'Not installed yet (trying python from PATH)'}</span>
            <button
              className="ghost"
              onClick={() =>
                void act(async () => {
                  const picked = await api.pickPython()
                  if (picked) setPython(picked)
                }, 'Restarting the image engine with this Python.')
              }
            >
              Use another Python…
            </button>
          </div>
          <EngineSetupPanel python={python} onInstalled={setPython} />
          <span className="hint">
            The image and video side runs in its own Python environment with PyTorch and diffusers. Install it here, or
            choose the <code>python.exe</code> of an environment you set up yourself (see the README).
          </span>
        </div>
        <div className="field">
          <span className="field-label">GPU</span>
          <span className="muted">
            {!device
              ? 'Shown once the engine is running.'
              : device.cuda
                ? `${device.name} · ${formatBytes(device.total_bytes)} VRAM${device.memory_plan ? ` · in use: ${device.memory_plan}` : ''}`
                : device.torch === false
                  ? device.name
                  : 'No CUDA GPU detected. Generation needs an NVIDIA GPU.'}
          </span>
          <span className="hint">The chat, image and video models never share the GPU. Loading one unloads the others.</span>
        </div>
      </div>

      <div className="card">
        <h2>Image model</h2>
        <div className="model-row">
          <div>
            <div className="strong">{model?.id ?? 'Image model'}</div>
            {imageProfile && <div className="small video-profile">{imageProfile.label}</div>}
            <div className="muted small">
              {model?.state === 'ready' && 'Loaded and ready'}
              {model?.state === 'loading' && (model.message || 'Loading…')}
              {model?.state === 'unloaded' && 'Not loaded. It loads automatically on the first image.'}
              {model?.state === 'error' && <span className="error-text">{model.message}</span>}
              {!model && 'Available once the engine is running.'}
            </div>
          </div>
          <div className="row-gap">
            <button
              onClick={() => void act(() => api.loadModel(), 'Loading the image model.')}
              disabled={!running || model?.state === 'loading' || model?.state === 'ready'}
            >
              Load now
            </button>
            <button onClick={() => void act(() => api.unloadModel(), 'Image model unloaded.')} disabled={!running || model?.state !== 'ready'}>
              Unload
            </button>
          </div>
        </div>

        <div className="field">
          <span className="field-label">Model folder</span>
          <div className="row-gap">
            <span className="mono grow path">{settings?.model_dir ?? '—'}</span>
            <button
              disabled={!running}
              onClick={() =>
                void act(async () => {
                  const next = await api.pickModelFolder()
                  if (next) setSettings(next)
                }, 'Saved. The folder is checked when the model loads.')
              }
            >
              Choose folder…
            </button>
          </div>
          <span className="hint">
            A folder downloaded below, one made with <em>Import a model file…</em> from a single transformer file, or
            one made with <code>image-engine/scripts/fetch_model.py</code> or <code>import_model.py</code>. Only
            safetensors weights are accepted, and the engine never runs code shipped with a model.
          </span>
          {settings?.model_dir && status?.image && status.image.profile === null && (
            <span className="hint warn-text">This folder holds no supported image model.</span>
          )}
          {imageProfile && !imageProfile.loras && (
            <span className="hint">This model takes no LoRAs and cannot be trained; the Train tab needs the 6B model.</span>
          )}
          {imageProfile && imageProfile.loras && !imageProfile.training && (
            <span className="hint">This model takes LoRAs; training them needs the distilled 6B model.</span>
          )}
        </div>

        <ModelDownloadPanel kind="image" enabled={running} onSelected={reload} />

        <ImportModelPanel target="image" enabled={running} onSelected={reload} />

        <label className="toggle">
          <input
            type="checkbox"
            checked={settings?.load_on_start ?? false}
            disabled={!settings}
            onChange={(e) => void save({ load_on_start: e.target.checked })}
          />
          <span>Load the image model when the engine starts</span>
        </label>

        <div className="field">
          <span className="field-label">Memory</span>
          <div className="mode-list" role="radiogroup" aria-label="Image memory mode">
            {(imageProfile?.name.startsWith('large') ? LARGE_MODES : MODES).map((m) => (
              <label key={m.value} className={`mode ${settings?.memory_mode === m.value ? 'active' : ''}`}>
                <input
                  type="radio"
                  name="memory_mode"
                  value={m.value}
                  checked={settings?.memory_mode === m.value}
                  disabled={!settings}
                  onChange={() => void save({ memory_mode: m.value })}
                />
                <span>
                  <span className="strong">{m.label}</span>
                  <span className="muted small block">{m.detail}</span>
                </span>
              </label>
            ))}
          </div>
        </div>
      </div>

      <div className="card">
        <h2>Video model</h2>
        {!videoModel ? (
          <p className="muted">{running ? 'This engine has no video model.' : 'Available once the engine is running.'}</p>
        ) : (
          <>
            <div className="model-row">
              <div>
                <div className="strong">{videoModel.id}</div>
                {video?.profile && <div className="small video-profile">{video.profile.label}</div>}
                <div className="muted small">
                  {videoModel.state === 'ready' && 'Loaded and ready'}
                  {videoModel.state === 'loading' && (videoModel.message || 'Loading…')}
                  {videoModel.state === 'unloaded' && 'Not loaded. It loads automatically on the first video, in place of the image model.'}
                  {videoModel.state === 'error' && <span className="error-text">{videoModel.message}</span>}
                </div>
              </div>
              <div className="row-gap">
                <button
                  onClick={() => void act(() => api.loadModel('video'), 'Loading the video model.')}
                  disabled={!running || videoModel.state === 'loading' || videoModel.state === 'ready'}
                >
                  Load now
                </button>
                <button
                  onClick={() => void act(() => api.unloadModel('video'), 'Video model unloaded.')}
                  disabled={!running || videoModel.state !== 'ready'}
                >
                  Unload
                </button>
              </div>
            </div>

            <div className="field">
              <span className="field-label">Model folder</span>
              <div className="row-gap">
                <span className="mono grow path">{settings?.video_model_dir ?? '—'}</span>
                <button
                  disabled={!running}
                  onClick={() =>
                    void act(async () => {
                      const next = await api.pickModelFolder('video')
                      if (next) setSettings(next)
                    }, 'Saved. The folder is checked when the video model loads.')
                  }
                >
                  Choose folder…
                </button>
              </div>
              <span className="hint">
                A folder downloaded below, one made with <em>Import a model file…</em> from a single transformer file, or
                one made with <code>image-engine/scripts/fetch_model.py --model video-fast</code> (or{' '}
                <code>--model video</code>) or <code>import_model.py</code>. Only safetensors weights are accepted.
              </span>
              {settings?.video_model_dir && video && video.profile === null && (
                <span className="hint warn-text">This folder holds no supported video model yet.</span>
              )}
            </div>

            <ModelDownloadPanel kind="video" enabled={running} onSelected={reload} />

            <ImportModelPanel target="video" enabled={running} onSelected={reload} />

            <div className="field">
              <span className="field-label">Memory</span>
              <div className="mode-list" role="radiogroup" aria-label="Video memory mode">
                {(video?.profile?.experts === 2 ? TWO_EXPERT_MODES : VIDEO_MODES).map((m) => (
                  <label key={m.value} className={`mode ${settings?.video_memory_mode === m.value ? 'active' : ''}`}>
                    <input
                      type="radio"
                      name="video_memory_mode"
                      value={m.value}
                      checked={settings?.video_memory_mode === m.value}
                      disabled={!settings}
                      onChange={() => void save({ video_memory_mode: m.value })}
                    />
                    <span>
                      <span className="strong">{m.label}</span>
                      <span className="muted small block">{m.detail}</span>
                    </span>
                  </label>
                ))}
              </div>
            </div>

            <div className="field">
              <span className="field-label">ffmpeg</span>
              <div className="row-gap">
                <span className="mono grow path">
                  {video?.ffmpeg ?? (settings?.ffmpeg_path ? `${settings.ffmpeg_path} (not found)` : 'Not found')}
                </span>
                <button
                  disabled={!running}
                  onClick={() =>
                    void act(async () => {
                      const next = await api.pickFfmpeg()
                      if (next) setSettings(next)
                    }, 'Saved. The next video uses this ffmpeg.')
                  }
                >
                  Choose…
                </button>
                {settings?.ffmpeg_path && (
                  <button className="ghost" onClick={() => void save({ ffmpeg_path: '' })}>
                    Find automatically
                  </button>
                )}
              </div>
              <span className="hint">
                {video?.encoder === 'mp4'
                  ? 'Videos are saved as MP4 (H.264).'
                  : 'Without ffmpeg, videos are saved as animated WebP: larger files that play as pictures.'}{' '}
                ffmpeg is not bundled; it is found on PATH or where winget installs it.
              </span>
            </div>
          </>
        )}
      </div>

      <div className="card">
        <h2>Captioning model</h2>
        <p className="hint">
          Used only by the Train tab: it describes training photos and finds the ones that do not really show the
          subject. {status?.captioning ? 'Installed.' : 'Not installed.'}
        </p>
        <div className="field">
          <span className="field-label">Model folder</span>
          <div className="row-gap">
            <span className="mono grow path">{settings?.caption_model_dir ?? '—'}</span>
            <button
              disabled={!running}
              onClick={() =>
                void act(async () => {
                  const next = await api.pickModelFolder('caption')
                  if (next) setSettings(next)
                }, 'Saved. The folder is checked before photos are described.')
              }
            >
              Choose folder…
            </button>
          </div>
          <span className="hint">
            A folder downloaded below, or with <code>image-engine/scripts/fetch_model.py --model caption</code>. Only
            safetensors weights are accepted, and the model's own code is never run.
          </span>
        </div>
        <ModelDownloadPanel kind="caption" enabled={running} onSelected={reload} />
      </div>

      <ModelInspectPanel enabled={running} />

      <div className="card">
        <h2>Hidden folders</h2>
        <label className="toggle">
          <input
            type="checkbox"
            checked={showHidden}
            disabled={!running}
            onChange={(e) => {
              const value = e.target.checked
              void act(
                async () => setShowHidden(await api.setShowHiddenFolders(value)),
                value ? 'Hidden folders are shown until the window closes.' : 'Hidden folders are hidden again.'
              )
            }}
          />
          <span>Show hidden folders</span>
        </label>
        <span className="hint">
          Hide a folder from its menu in the gallery: it leaves the folder list, and what it holds leaves every view
          and search. This shows them again for now; closing the window hides them, and locks every unlocked folder,
          again. Hiding and locking keep the studio&rsquo;s own views private. They do not encrypt anything: the files
          stay on disk as they are.
        </span>
      </div>

      <div className="card">
        <h2>Saved copies</h2>
        <label className="toggle">
          <input
            type="checkbox"
            checked={includeSettings}
            onChange={(e) => {
              const value = e.target.checked
              void act(async () => setIncludeSettings(await api.setIncludeSettingsInExports(value)), 'Saved.')
            }}
          />
          <span>Include the prompt and settings in images saved with “Save as…”</span>
        </label>
        <span className="hint">
          Off by default: a PNG can carry its prompt wherever it is shared. With this off, “Save as…” writes the
          picture only. The gallery&rsquo;s own copy always keeps them, so “Use these settings” and dropping an image
          back in keep working.
        </span>
      </div>

      {message && (
        <div className={message.kind === 'error' ? 'error-text' : 'ok-text'} role="status">
          {message.text}
        </div>
      )}
    </section>
  )
}
