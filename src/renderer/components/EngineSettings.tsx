import { EXPERTS_ON_CPU_CHOICES, THREAD_CHOICES, type AppConfig, type ChatEngine, type ServerPreferences } from '@shared/config.js'
import type { ModelInfo } from '@shared/engine-protocol.js'

interface Props {
  config: AppConfig
  /** The loaded model, when there is one; decides whether the expert placement row is shown and whether a reload is offered. */
  model: ModelInfo | null
  busy: boolean
  onChange: (engine: ChatEngine, server: ServerPreferences) => void
  onPickServer: () => void
  onReload: () => void
}

const ENGINE_LABEL: Record<ChatEngine, string> = { builtin: 'Built-in', server: 'llama-server' }

/**
 * Where the loaded model's experts are, in the same terms as the setting
 * (null: not a question for this model, as for a dense model or one whose
 * experts could not be placed).
 */
export function loadedExpertsOnCpu(model: ModelInfo): number | 'all' | null {
  if (model.traits.experts === null || model.expertsOnCpuUnavailable !== undefined) return null
  if (model.engine === 'builtin' && model.backend === 'cpu') return null
  return model.expertsOnCpu ?? 0
}

/** Whether the loaded model was started some other way than the settings now say. */
export function loadedDiffers(model: ModelInfo, engine: ChatEngine, server: ServerPreferences): boolean {
  if (model.engine !== engine) return true
  const experts = loadedExpertsOnCpu(model)
  if (experts !== null && experts !== (server.expertsOnCpu ?? 'all')) return true
  return model.threads !== undefined && model.threads !== server.threads
}

/** The chat engine, how the model is spread over the GPU and the CPU, and for llama-server, which program. Shown in the model menu, and before a model is loaded. */
export function EngineSettings({ config, model, busy, onChange, onPickServer, onReload }: Props): React.JSX.Element {
  const { engine, server } = config.model
  const set = (patch: Partial<ServerPreferences>): void => onChange(engine, { ...server, ...patch })
  const differs = model !== null && loadedDiffers(model, engine, server)
  const showExperts = model === null || model.traits.experts !== null
  const unavailable = model !== null && model.engine === engine ? model.expertsOnCpuUnavailable : undefined

  return (
    <div className="mm-kv">
      <div className="mm-kv-row">
        <span>Engine</span>
        <div className="seg" role="group" aria-label="Engine">
          {(Object.keys(ENGINE_LABEL) as ChatEngine[]).map((choice) => (
            <button key={choice} aria-pressed={engine === choice} onClick={() => onChange(choice, server)}>
              {ENGINE_LABEL[choice]}
            </button>
          ))}
        </div>
      </div>
      {engine === 'server' ? (
        <>
          <div className="mm-kv-row mm-sub">
            <span>Program</span>
            <button className="small" disabled={busy} onClick={onPickServer}>
              {server.path ? 'Change…' : 'Choose llama-server…'}
            </button>
          </div>
          {server.path ? <code className="mm-path">{server.path}</code> : null}
        </>
      ) : null}
      {showExperts ? (
        <div className="mm-kv-row mm-sub">
          <span>Experts on CPU</span>
          <select
            aria-label="Experts on CPU"
            value={server.expertsOnCpu ?? ''}
            onChange={(e) => set({ expertsOnCpu: e.target.value === '' ? null : Number(e.target.value) })}
          >
            {EXPERTS_ON_CPU_CHOICES.map((choice) => (
              <option key={choice ?? 'all'} value={choice ?? ''}>
                {choice === null ? 'All layers' : choice === 0 ? 'None (split by layer)' : `First ${choice} layers`}
              </option>
            ))}
          </select>
        </div>
      ) : null}
      {unavailable ? <p className="help">{unavailable}</p> : null}
      <div className="mm-kv-row mm-sub">
        <span>CPU threads</span>
        <select aria-label="CPU threads" value={server.threads ?? ''} onChange={(e) => set({ threads: e.target.value === '' ? null : Number(e.target.value) })}>
          {THREAD_CHOICES.map((choice) => (
            <option key={choice ?? 'auto'} value={choice ?? ''}>
              {choice === null ? 'Default' : String(choice)}
            </option>
          ))}
        </select>
      </div>
      <p className="help">
        {engine === 'builtin'
          ? 'The library inside the studio. Nothing to install. For a mixture-of-experts model bigger than the GPU, keeping the experts in system memory and the rest on the GPU is much faster than splitting it by layer (CUDA only).'
          : 'A llama-server program you installed, started by the studio on this computer and reached on this computer only. It places experts the same way the built-in engine does.'}
      </p>
      {differs ? (
        <button className="primary small" disabled={busy} onClick={onReload}>
          Reload with {model?.engine === engine ? 'these settings' : ENGINE_LABEL[engine]}
        </button>
      ) : null}
    </div>
  )
}
