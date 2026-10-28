import { useEffect, useState } from 'react'
import {
  IDLE_IMPORT,
  MAX_FEW_STEPS,
  MAX_SHIFT,
  fewStepProblem,
  isTwoExperts,
  variantLabels,
  type ImportOptions,
  type ImportTarget,
  type ModelImportState
} from '@shared/importing.js'
import { catalogModel } from '@shared/setup.js'
import { api } from '../api.js'
import { downloadSummary } from '../download.js'
import { formatBytes } from '../format.js'
import { useDownloadState } from './ModelDownloadPanel.js'
import { HASH_HELP } from './ModelInspectPanel.js'

/**
 * Settings → Image model (or Video model) → Import a model file…: a model shared as a
 * single transformer .safetensors file, put together with the parts it needs into a model
 * folder the engine loads, then selected. The file is checked and hashed, never loaded or
 * run. One import runs at a time; it shows in the card it was started from.
 */

const ABOUT: Record<ImportTarget, string> = {
  image: 'A 6B or 12B image model',
  video: 'A 5B video model (the architecture of the downloads above), or a 14B text-to-video or image-to-video model (two files),'
}

const TWO_EXPERT_HINT =
  'Distilled 14B files (often called lightning, 4-step or few-step) make a clip in a few steps without guidance; ' +
  'their page gives the steps and shift to use. The original runs 40 steps.'

const VARIANT_HINT: Record<ImportTarget, string> = {
  image: 'The download page says which: distilled models are often called turbo, fast or few-step. It only sets the default steps and guidance.',
  video:
    'Distilled means a 3-step distillation like the downloadable distilled model, which runs on a fixed schedule. ' +
    'A file distilled for a different number of steps fits neither choice. The download page says which.'
}

export function useImportState(): ModelImportState {
  const [state, setState] = useState<ModelImportState>(IDLE_IMPORT)
  useEffect(() => {
    const off = window.api.image.onImportState(setState)
    void window.api.image.importState().then(setState)
    return off
  }, [])
  return state
}

const BUSY = new Set(['probing', 'parts', 'checking', 'hashing', 'assembling'])

export function ImportModelPanel({
  target,
  enabled,
  onSelected
}: {
  target: ImportTarget
  enabled: boolean
  onSelected: () => void
}): React.JSX.Element {
  const shared = useImportState()
  // An import started from the other card shows there; here it only holds the button.
  const elsewhere = shared.target !== null && shared.target !== target && shared.phase !== 'idle'
  const state = elsewhere ? IDLE_IMPORT : shared
  const download = useDownloadState()
  const [variant, setVariant] = useState<ImportOptions['variant'] | null>(null)
  const [sha256, setSha256] = useState('')
  const [sha256Second, setSha256Second] = useState('')
  const [steps, setSteps] = useState('4')
  const [shift, setShift] = useState('5')
  const [name, setName] = useState('')
  const [accept, setAccept] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const busy = BUSY.has(state.phase)
  const probe = state.probe
  const second = state.second
  const two = isTwoExperts(probe)

  // A newly read file starts a fresh form (choosing or swapping a 14B model's other expert does not).
  useEffect(() => {
    if (state.phase !== 'ready') return
    setVariant(null)
    setSha256('')
    setSha256Second('')
    setSteps('4')
    setShift('5')
    setAccept(false)
    setName(state.suggestedName)
  }, [state.phase === 'ready' ? state.suggestedName : ''])

  useEffect(() => {
    if (state.phase === 'done') onSelected()
  }, [state.phase])

  const act = async (run: () => Promise<unknown>): Promise<void> => {
    setError(null)
    try {
      await run()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const needs = state.needs.map((id) => catalogModel(id)).filter((m) => m !== null)
  const ready = state.phase === 'ready' && probe !== null
  const fewStep = two && variant === 'distilled'
  const fewStepError = fewStep ? fewStepProblem(Number(steps), Number(shift)) : null
  const canStart =
    ready && variant !== null && name.trim() !== '' && (needs.length === 0 || accept) && (!two || second !== null) && !fewStepError

  return (
    <div className="field import-panel">
      <span className="field-label">Import a model file</span>
      <span className="hint">
        {ABOUT[target]} shared as a single transformer <code>.safetensors</code> file (fp8 or bf16). The file is checked
        and hashed, then linked with the parts it needs into a new model folder, which is selected. Nothing in it is
        loaded or run until you generate.
      </span>
      {!ready && !busy && (
        <div className="row-gap">
          <button disabled={!enabled || elsewhere} onClick={() => void act(() => api.pickImportFile(target))}>
            Import a model file…
          </button>
          {!enabled && <span className="hint">Available once the engine is running: the import uses its Python.</span>}
          {enabled && elsewhere && (
            <span className="hint">An import is open under {target === 'video' ? 'Image model' : 'Video model'}; finish or cancel it first.</span>
          )}
        </div>
      )}

      {state.phase === 'probing' && <span className="muted small">Reading the file's header…</span>}

      {ready && probe && (
        <div className="import-form">
          <div>
            {two && <div className="muted small">First expert (high noise)</div>}
            <div className="strong">{probe.file}</div>
            <div className="muted small">
              The {probe.label} · {(probe.parameters / 1e9).toFixed(1)}B parameters · {probe.storage} · {probe.layers} blocks ·{' '}
              {formatBytes(probe.size)}
            </div>
          </div>

          {two && (
            <div className="field">
              {second ? (
                <div>
                  <div className="muted small">Second expert (low noise)</div>
                  <div className="strong">{second.file}</div>
                  <div className="muted small">
                    {second.storage} · {formatBytes(second.size)}{' '}
                    <button type="button" className="link-button" onClick={() => void act(() => api.swapImportExperts())}>
                      Swap the two
                    </button>
                  </div>
                </div>
              ) : (
                <div className="row-gap">
                  <button type="button" onClick={() => void act(() => api.pickImportFile(target, 'second'))}>
                    Choose the second expert's file…
                  </button>
                </div>
              )}
              <span className="hint">
                A 14B model is two files of the same size: the first expert works on the noisy start of a clip, the second
                on the details at the end. Their names usually say high noise and low noise; when both say so, they are put
                in order by themselves.
              </span>
            </div>
          )}

          <div className="field">
            <span className="field-label">Which version is it?</span>
            <div className="mode-list" role="radiogroup" aria-label="Model version">
              {(['distilled', 'base'] as const).map((v) => (
                <label key={v} className={`mode ${variant === v ? 'active' : ''}`}>
                  <input type="radio" name={`import-variant-${target}`} checked={variant === v} onChange={() => setVariant(v)} />
                  <span>{variantLabels(probe)[v]}</span>
                </label>
              ))}
            </div>
            <span className="hint">{two ? TWO_EXPERT_HINT : VARIANT_HINT[target]}</span>
          </div>

          {fewStep && (
            <div className="field">
              <span className="field-label">Steps and shift, from the download page</span>
              <div className="row-gap">
                <label className="row-gap small">
                  Steps
                  <input className="input narrow" type="number" min={1} max={MAX_FEW_STEPS} step={1} value={steps}
                         onChange={(e) => setSteps(e.target.value)} aria-label="Few-step steps" />
                </label>
                <label className="row-gap small">
                  Shift
                  <input className="input narrow" type="number" min={0.1} max={MAX_SHIFT} step={0.5} value={shift}
                         onChange={(e) => setShift(e.target.value)} aria-label="Few-step shift" />
                </label>
              </div>
              <span className="hint">
                The studio runs plain Euler steps on a schedule shifted this much, with no guidance. 4 steps with shift 5 is
                common, but use the page's values: a wrong shift makes noisy or washed-out clips.
              </span>
              {fewStepError && <span className="error-text small">{fewStepError}</span>}
            </div>
          )}

          <label className="field">
            <span className="field-label">
              Published SHA-256{two ? ' of the first expert' : ''} (optional){' '}
              <span className="info-tip" title={HASH_HELP} tabIndex={0} aria-label={HASH_HELP}>
                ⓘ
              </span>
            </span>
            <input
              className="input mono"
              value={sha256}
              spellCheck={false}
              placeholder="Paste it to have the import refuse a file that differs"
              onChange={(e) => setSha256(e.target.value)}
            />
          </label>
          {two && second && (
            <label className="field">
              <span className="field-label">Published SHA-256 of the second expert (optional)</span>
              <input
                className="input mono"
                value={sha256Second}
                spellCheck={false}
                placeholder="Paste it to have the import refuse a file that differs"
                onChange={(e) => setSha256Second(e.target.value)}
              />
            </label>
          )}

          <label className="field">
            <span className="field-label">Folder name</span>
            <input className="input mono" value={name} spellCheck={false} onChange={(e) => setName(e.target.value)} />
            <span className="hint">Made inside the models folder. Hard links where the disk allows, so no second copy.</span>
          </label>

          {needs.length > 0 && (
            <div className="field">
              <span className="field-label">Also needed, downloaded first</span>
              {needs.map((m) => (
                <div key={m.id} className="small">
                  <span className="strong">{m.label}</span> · {formatBytes(m.bytes)} from{' '}
                  <button type="button" className="link-button" onClick={() => void api.openModelLink(m.id, 'source')}>
                    {m.repo}
                  </button>{' '}
                  at commit <span className="mono">{m.revision.slice(0, 12)}</span> ·{' '}
                  <button type="button" className="link-button" onClick={() => void api.openModelLink(m.id, 'license')}>
                    {m.license}
                  </button>
                </div>
              ))}
              <label className="toggle">
                <input type="checkbox" checked={accept} onChange={(e) => setAccept(e.target.checked)} />
                <span>I accept {needs.length > 1 ? 'these licenses' : 'this license'}</span>
              </label>
            </div>
          )}

          <div className="row-gap">
            <button
              className="primary"
              disabled={!canStart}
              onClick={() =>
                variant &&
                void act(() =>
                  api.startImport({
                    variant,
                    sha256: sha256.trim(),
                    name: name.trim(),
                    acceptLicenses: accept,
                    sha256_2: sha256Second.trim(),
                    steps: fewStep ? Number(steps) : null,
                    shift: fewStep ? Number(shift) : null
                  })
                )
              }
            >
              Import
            </button>
            <button onClick={() => void act(() => api.clearImport())}>Cancel</button>
          </div>
        </div>
      )}

      {busy && state.phase !== 'probing' && <Progress state={state} downloadText={downloadSummary(download)} />}

      {(state.phase === 'failed' || state.phase === 'cancelled' || state.phase === 'done') && (
        <div className="row-gap">
          <span className={state.phase === 'failed' ? 'error-text small pre' : state.phase === 'done' ? 'ok-text small' : 'muted small'}>
            {state.probe ? `${state.probe.file}: ` : ''}
            {state.message}
          </span>
          <button onClick={() => void act(() => api.clearImport())}>OK</button>
        </div>
      )}
      {error && <div className="error-text small">{error}</div>}
    </div>
  )
}

function Progress({ state, downloadText }: { state: ModelImportState; downloadText: string }): React.JSX.Element {
  const hashing = state.phase === 'hashing' && state.size > 0
  const percent = hashing ? (100 * state.done) / state.size : 0
  const text =
    state.phase === 'parts'
      ? `Downloading ${state.message}: ${downloadText}`
      : state.phase === 'hashing'
        ? `Hashing, ${formatBytes(state.done)} of ${formatBytes(state.size)}`
        : state.phase === 'assembling'
          ? 'Linking the files into the model folder…'
          : 'Checking every tensor against the model…'
  return (
    <div className="download-progress">
      <div className={`bar ${hashing ? '' : 'indeterminate'}`}>
        <div style={hashing ? { width: `${percent.toFixed(1)}%` } : undefined} />
      </div>
      <div className="progress-foot">
        <span className="muted small tabular">{text}</span>
        <button onClick={() => void api.cancelImport()}>Stop</button>
      </div>
    </div>
  )
}
