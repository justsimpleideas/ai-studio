import { useState } from 'react'
import type { AppConfig } from '@shared/config.js'
import type { ModelInfo } from '@shared/engine-protocol.js'
import type { InternetLevel } from '@shared/network.js'
import { cleanError } from '../image/api.js'
import { Report, useInspectState } from '../image/components/ModelInspectPanel.js'

/**
 * Model check: what can be known about a chat model file, and what cannot.
 *
 * Three parts. The file's report (the strict header check, what the metadata
 * says, the SHA-256 to compare with the publisher's), the chat template as the
 * loaded model actually receives it, and the list of what no model can do here
 * whatever its weights hold. The last part is the honest answer to "is this
 * model safe": its behaviour cannot be read from the file, so the studio is
 * built not to need to know.
 */

export function ModelCheckPanel({
  model,
  config,
  internet,
  strict
}: {
  model: ModelInfo | null
  config: AppConfig
  /** The open folder's internet level, or null with no folder open. */
  internet: InternetLevel | null
  strict: boolean
}): React.JSX.Element {
  const state = useInspectState()
  const [error, setError] = useState<string | null>(null)
  const busy = state.phase === 'header' || state.phase === 'hashing'

  const run = async (action: () => Promise<boolean>): Promise<void> => {
    setError(null)
    try {
      if (!(await action())) setError('No model file is known yet. Load a model, or choose a file.')
    } catch (err) {
      setError(cleanError(err).message)
    }
  }

  return (
    <div className="panel model-check">
      <section>
        <h3>The model file</h3>
        <p className="help">
          Read by the studio’s own strict reader, without loading or running any of it: every count, length and offset in
          the header is checked against the file, the metadata and chat template are read for anything that does not
          belong in a model, and the whole file is hashed to compare with the publisher’s SHA-256. A file that fails the
          header check is refused at load as well. Scanning for malware with your antivirus is still worth doing: this
          checks the file’s shape, not its reputation.
        </p>
        <div className="row-gap">
          <button className="primary" disabled={busy || !model} onClick={() => void run(() => window.api.inspectModel())}>
            Check {model ? model.name : 'the loaded model'}
          </button>
          <button disabled={busy} onClick={() => void run(() => window.api.image.inspectModelFile())}>
            Choose another file…
          </button>
          {busy ? <button onClick={() => void window.api.image.cancelInspect()}>Stop</button> : null}
        </div>
        {error ? <div className="error-text small">{error}</div> : null}
        {busy ? (
          <p className="help">
            {state.file}:{' '}
            {state.phase === 'hashing' && state.size > 0
              ? `hashing, ${Math.round((100 * state.done) / state.size)}%`
              : 'reading the header…'}
          </p>
        ) : null}
        {(state.phase === 'failed' || state.phase === 'cancelled') && (
          <div className={state.phase === 'failed' ? 'error-text small pre' : 'help'}>
            {state.file}: {state.message}
          </div>
        )}
        {state.phase === 'done' && state.report ? <Report report={state.report} /> : null}
      </section>

      <section>
        <h3>What the model reads</h3>
        {model ? (
          <>
            <p className="help">
              {model.template.usesFileTemplate
                ? 'Conversations are formatted with the chat template carried in this file. It is rendered by the studio’s own template engine, which has no access to files, the network or the system; a template can shape the text the model sees, and nothing else.'
                : `Conversations are formatted with the studio’s built-in “${model.template.wrapper}” template for this model family. The template carried in the file is not run.`}
            </p>
            <details className="template-view">
              <summary>A sample conversation, exactly as this model receives it</summary>
              <pre>{model.template.rendered}</pre>
            </details>
            {model.template.source ? (
              <details className="template-view">
                <summary>The chat template in the file ({model.template.source.length.toLocaleString()} characters)</summary>
                <pre>{model.template.source}</pre>
              </details>
            ) : (
              <p className="help">The file carries no chat template of its own.</p>
            )}
          </>
        ) : (
          <p className="help">Load a model to see how conversations are formatted for it.</p>
        )}
      </section>

      <section>
        <h3>What no model can do here</h3>
        <p className="help">
          Whether a model misbehaves cannot be read from its file: its weights are numbers, and a file can look ordinary
          and still hold a model trained to do the wrong thing. So the studio is built not to depend on the model being
          honest. These hold for every model, including one written to break them:
        </p>
        <ul className="guarantees">
          <li>
            <strong>It can only ask.</strong> The model runs in a process with no access to files or the network. The only
            thing it can do is request one of a few named actions; the request goes to another process that decides, and a
            request it was not given the vocabulary for cannot even be formed.
          </li>
          <li>
            <strong>Files: only inside the shared folders.</strong> Every path is resolved against the real filesystem —
            through junctions, links and every spelling Windows allows — before it is judged, and judged again before it is
            acted on. Secrets are unreachable and invisible; version control and dependencies are read-only; a file is not
            changed before it has been read; a change is backed up before it is made and can be put back.
          </li>
          <li>
            <strong>Internet: none unless you turn it on, and then only through one gate.</strong> Offline is the default,
            and offline means the web tools do not exist for the model. When on, every request is checked, refused if it
            carries text from your files, shown to you first unless you chose otherwise, and logged.
          </li>
          <li>
            <strong>Strict mode: nothing at all.</strong> With strict mode on for a folder the model has no tools of any
            kind and can only talk.
          </li>
        </ul>
        <dl className="guarantee-status">
          <dt>This folder</dt>
          <dd>
            {internet === null
              ? 'No folder is open.'
              : `${strict ? 'Strict mode: conversation only. ' : ''}Internet: ${internet === 'offline' ? 'off' : internet === 'search' ? 'search and pages' : 'search, pages and downloads'}.`}
          </dd>
          <dt>Asked about</dt>
          <dd>
            {config.permissions.confirmEverything
              ? 'Everything: confirm-everything is on.'
              : (['read', 'write', 'destructive', 'network'] as const)
                  .map((risk) => `${risk} ${config.permissions.modes[risk] === 'auto' ? 'automatic' : 'asked'}`)
                  .join(' · ')}
          </dd>
        </dl>
        <p className="help">
          The one thing the structure cannot catch on its own: with the internet on and network requests set to automatic,
          a model could put a reworded piece of your files into a search. Keeping network requests asked about means that
          passes in front of you first.
        </p>
      </section>
    </div>
  )
}
