import { useEffect, useState } from 'react'
import { SETUP_STEPS, type EngineSetupState, type SetupPhase } from '@shared/setup.js'
import { api } from '../api.js'

/**
 * Installing the engine's Python environment: what will happen, the steps as
 * they run, and pip's own output for anyone who wants it.
 */

const ACTIVE: readonly SetupPhase[] = ['checking', 'creating', 'torch', 'packages', 'verifying']

export function useSetupState(): EngineSetupState | null {
  const [state, setState] = useState<EngineSetupState | null>(null)
  useEffect(() => {
    const off = window.api.image.onSetupState(setState)
    void window.api.image.setupState().then(setState)
    return off
  }, [])
  return state
}

interface Props {
  /** The configured interpreter, or null when none has been set up. */
  python: string | null
  onInstalled: (python: string) => void
}

export function EngineSetupPanel({ python, onInstalled }: Props): React.JSX.Element {
  const state = useSetupState()
  const [confirming, setConfirming] = useState(false)
  const [showLog, setShowLog] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const phase = state?.phase ?? 'idle'
  const running = ACTIVE.includes(phase)

  const install = async (): Promise<void> => {
    setConfirming(false)
    setError(null)
    try {
      onInstalled(await api.installEngine())
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const current = SETUP_STEPS.findIndex((s) => s.phase === phase)

  return (
    <div className="setup">
      {!running && !confirming && (
        <div className="row-gap">
          <button className={python ? '' : 'primary'} onClick={() => setConfirming(true)}>
            {python ? 'Reinstall or repair…' : 'Install the engine…'}
          </button>
          {phase === 'done' && <span className="ok-text small">{state?.message}</span>}
        </div>
      )}

      {confirming && (
        <div className="setup-box">
          <p>
            The studio makes its own Python environment in its data folder and installs PyTorch with CUDA, diffusers and
            their dependencies into it: about <strong>3 GB</strong> to download, <strong>5 GB</strong> on disk. Every
            package comes from the official PyTorch and PyPI indexes, pinned to the versions this release was tested
            with, as prebuilt wheels only. Nothing is installed anywhere else.
          </p>
          <p className="muted small">
            Needs Python 3.12 to 3.14 (64-bit) already installed, from python.org or <code>winget install Python.Python.3.13</code>,
            and an NVIDIA GPU with a current driver. The chat side needs none of this.
          </p>
          <div className="row-gap">
            <button className="primary" onClick={() => void install()}>
              Install
            </button>
            <button className="ghost" onClick={() => setConfirming(false)}>
              Cancel
            </button>
          </div>
        </div>
      )}

      {(running || phase === 'failed' || phase === 'cancelled') && state && (
        <div className="setup-box">
          <ol className="setup-steps">
            {SETUP_STEPS.map((step, i) => {
              const mark = phase === 'failed' || phase === 'cancelled' ? (i < current ? 'done' : i === current ? 'stopped' : '') : i < current ? 'done' : i === current ? 'active' : ''
              return (
                <li key={step.phase} className={mark}>
                  {step.label}
                </li>
              )
            })}
          </ol>
          <div className={phase === 'failed' ? 'error-text small pre' : 'muted small'}>{state.message}</div>
          {state.gpu && <div className="muted small">GPU: {state.gpu}</div>}
          <div className="row-gap">
            {running ? (
              <button onClick={() => void api.cancelInstall()}>Stop</button>
            ) : (
              <button onClick={() => setConfirming(true)}>Try again</button>
            )}
            <button className="ghost" onClick={() => setShowLog((v) => !v)}>
              {showLog ? 'Hide details' : 'Show details'}
            </button>
          </div>
          {showLog && <pre className="setup-log">{state.log.slice(-60).join('\n') || 'Nothing yet.'}</pre>}
        </div>
      )}

      {error && phase !== 'failed' && phase !== 'cancelled' && <div className="error-text small">{error}</div>}
    </div>
  )
}
