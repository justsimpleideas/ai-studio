import { useEffect, useState } from 'react'
import { IDLE_INSPECT, compareSha256, type FindingLevel, type InspectReport, type ModelInspectState } from '@shared/inspect.js'
import { api } from '../api.js'
import { formatBytes } from '../format.js'

/**
 * Settings → Inspect a model file: the SHA-256 of a downloaded model file, to compare
 * with the publisher's, and what image-engine/scripts/inspect_model.py found in its
 * header and metadata. Nothing is loaded or run, and nothing leaves the computer.
 */

export const HASH_HELP = [
  'Where to find the published SHA-256:',
  "• Hugging Face: open the model's \"Files and versions\" tab and click the file's name. The file's page lists its SHA256 (in the details of the large-file storage). Copy it from there.",
  '• Other sites: look for "SHA256" or "hash" next to the download link.',
  "Take it only from the publisher's own page. Then paste it below, or compare it with the hash shown here."
].join('\n')

const LEVEL_MARK: Record<FindingLevel, string> = { ok: '✓', info: 'i', warn: '!', danger: '✕' }

const VERDICT: Record<InspectReport['verdict'], { className: string; text: string }> = {
  clean: { className: 'ok-text', text: 'No problems found in the header and metadata.' },
  warnings: { className: 'warn-text', text: 'Worth a look: see the warnings below.' },
  unsafe: { className: 'error-text', text: 'Do not use this file: see below.' }
}

export function useInspectState(): ModelInspectState {
  const [state, setState] = useState<ModelInspectState>(IDLE_INSPECT)
  useEffect(() => {
    const off = window.api.image.onInspectState(setState)
    void window.api.image.inspectState().then(setState)
    return off
  }, [])
  return state
}

export function ModelInspectPanel({ enabled }: { enabled: boolean }): React.JSX.Element {
  const state = useInspectState()
  const [error, setError] = useState<string | null>(null)
  const busy = state.phase === 'header' || state.phase === 'hashing'

  const inspect = async (): Promise<void> => {
    setError(null)
    try {
      await api.inspectModelFile()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <div className="card">
      <h2>Inspect a model file</h2>
      <p className="muted">
        Checks a downloaded model file without loading or running it: its SHA-256, to compare with the one its
        publisher lists, and its header and metadata, for anything that does not belong in a model. It runs on this
        computer only. Scanning the file for malware, for example with Microsoft Defender, is up to you.
      </p>
      <div className="row-gap">
        <button disabled={!enabled || busy} onClick={() => void inspect()}>
          Choose a file…
        </button>
        {busy && <button onClick={() => void api.cancelInspect()}>Stop</button>}
        {!enabled && <span className="hint">Available once the engine is running: the check uses its Python.</span>}
      </div>
      {error && <div className="error-text small">{error}</div>}
      {busy && <Progress state={state} />}
      {(state.phase === 'failed' || state.phase === 'cancelled') && (
        <div className={state.phase === 'failed' ? 'error-text small pre' : 'muted small'}>
          {state.file}: {state.message}
        </div>
      )}
      {state.phase === 'done' && state.report && <Report report={state.report} />}
    </div>
  )
}

function Progress({ state }: { state: ModelInspectState }): React.JSX.Element {
  const hashing = state.phase === 'hashing' && state.size > 0
  const percent = hashing ? (100 * state.done) / state.size : 0
  return (
    <div className="download-progress">
      <div className={`bar ${hashing ? '' : 'indeterminate'}`}>
        <div style={hashing ? { width: `${percent.toFixed(1)}%` } : undefined} />
      </div>
      <span className="muted small tabular">
        {state.file}:{' '}
        {hashing ? `hashing, ${formatBytes(state.done)} of ${formatBytes(state.size)}` : 'reading the header…'}
      </span>
    </div>
  )
}

function Report({ report }: { report: InspectReport }): React.JSX.Element {
  const [pasted, setPasted] = useState('')
  const [copied, setCopied] = useState(false)
  const comparison = compareSha256(report.sha256, pasted)
  const verdict = VERDICT[report.verdict]

  useEffect(() => {
    setPasted('')
    setCopied(false)
  }, [report.sha256])

  return (
    <div className="inspect-report">
      <div>
        <div className="strong">{report.file}</div>
        <div className="muted small mono path">{report.folder}</div>
        <div className="muted small">
          {formatBytes(report.size)} · {report.format}
          {report.summary &&
            ` · ${report.summary.tensors.toLocaleString()} tensors · ${(report.summary.parameters / 1e9).toFixed(2)}B parameters · ${Object.entries(report.summary.dtypes)
              .map(([dtype, n]) => `${dtype} ${n}`)
              .join(', ')}`}
        </div>
      </div>
      <div className={`strong ${verdict.className}`}>{verdict.text}</div>

      <div className="field">
        <span className="field-label">
          SHA-256{' '}
          <span className="info-tip" title={HASH_HELP} tabIndex={0} aria-label={HASH_HELP}>
            ⓘ
          </span>
        </span>
        <div className="row-gap">
          <span className="mono grow sha">{report.sha256}</span>
          <button
            onClick={() => {
              void navigator.clipboard.writeText(report.sha256).then(() => setCopied(true))
            }}
          >
            {copied ? 'Copied' : 'Copy'}
          </button>
        </div>
        <input
          className="input mono"
          value={pasted}
          placeholder="Paste the publisher's SHA-256 to compare (optional)"
          spellCheck={false}
          onChange={(e) => setPasted(e.target.value)}
        />
        {comparison === 'match' && <span className="ok-text small">Matches: this is the file the publisher lists.</span>}
        {comparison === 'mismatch' && (
          <span className="error-text small">
            Does not match: this file is not the one the publisher lists. Do not use it; download it again from the
            publisher's own page.
          </span>
        )}
        {comparison === 'invalid' && <span className="warn-text small">A SHA-256 is 64 hexadecimal characters (0–9, a–f).</span>}
      </div>

      <ul className="inspect-findings">
        {report.findings.map((f, i) => (
          <li key={i} className={`finding ${f.level}`}>
            <span className="finding-mark" aria-label={f.level}>
              {LEVEL_MARK[f.level]}
            </span>
            <span>
              <span className="strong">{f.title}</span>
              {f.detail && <span className="muted small block pre">{f.detail}</span>}
            </span>
          </li>
        ))}
      </ul>

      {report.metadata.length > 0 && (
        <details>
          <summary className="small">Metadata ({report.metadata.length})</summary>
          <table className="metadata-table">
            <tbody>
              {report.metadata.map((m) => (
                <tr key={m.key}>
                  <td className="mono">{m.key}</td>
                  <td className="muted small tabular">{m.chars.toLocaleString()} chars</td>
                  <td className="small pre">{m.preview}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
    </div>
  )
}
