import { useEffect, useMemo, useState } from 'react'
import {
  describeGrant,
  type ApprovalRequest,
  type ApprovalResponse,
  type Grant
} from '@shared/broker-types.js'

/**
 * The approval prompt.
 *
 * The point of this dialog is that the person clicking it knows what they are
 * agreeing to, which takes more than a tool name. It shows the resolved path
 * rather than the one the model typed, it shows what the file will look like
 * afterwards, and when a button would create a standing rule it spells that
 * rule out in a sentence underneath.
 */

const VERBS: Record<string, string> = {
  list_directory: 'list',
  read_file: 'read',
  write_file: 'write to',
  edit_file: 'edit',
  delete_file: 'delete',
  web_search: 'search the web',
  fetch_page: 'open a web page',
  download_file: 'download a file'
}

const CONTEXT_CHARS = 220

function grantPatternFor(request: ApprovalRequest): string {
  const p = request.resolved.replace(/\\/g, '/')
  // Listing a directory is already about the directory; for a file, the useful
  // unit to grant is the folder it lives in.
  if (request.call.name === 'list_directory') return `${p}/**`
  const cut = p.lastIndexOf('/')
  return cut <= 0 ? `${p}/**` : `${p.slice(0, cut)}/**`
}

function Diff({ request }: { request: ApprovalRequest }): React.JSX.Element | null {
  const { preview, call } = request

  if (preview.kind === 'none') return null

  if (preview.kind === 'network') {
    return (
      <div className="net-preview">
        <div className="label-text">Sent to {preview.to}</div>
        <pre className="diff">{preview.sends}</pre>
        {preview.saveTo ? (
          <>
            <div className="label-text">Saved as a new file</div>
            <pre className="diff">
              <span className="ins">{preview.saveTo}</span>
            </pre>
          </>
        ) : null}
      </div>
    )
  }

  if (preview.kind === 'delete') {
    return (
      <pre className="diff">
        <span className="del">
          {preview.bytes.toLocaleString()} bytes will be moved to the recycle bin.
        </span>
      </pre>
    )
  }

  if (preview.kind === 'create') {
    return (
      <pre className="diff">
        <span className="ins">{preview.after}</span>
      </pre>
    )
  }

  if (preview.kind === 'overwrite') {
    return (
      <pre className="diff">
        <span className="del">{preview.before}</span>
        {'\n'}
        <span className="ins">{preview.after}</span>
      </pre>
    )
  }

  // An edit touches one run of text, so showing the whole file would bury the
  // change. Only the neighbourhood of it is rendered.
  if (call.name !== 'edit_file') return null
  const head = preview.before.slice(Math.max(0, preview.at - CONTEXT_CHARS), preview.at)
  const tailStart = preview.at + call.args.old_text.length
  const tail = preview.before.slice(tailStart, tailStart + CONTEXT_CHARS)

  return (
    <pre className="diff">
      {preview.at > CONTEXT_CHARS ? '…' : ''}
      {head}
      <span className="del">{call.args.old_text}</span>
      <span className="ins">{call.args.new_text}</span>
      {tail}
      {tailStart + CONTEXT_CHARS < preview.before.length ? '…' : ''}
    </pre>
  )
}

export function ApprovalDialog({
  request,
  onRespond
}: {
  request: ApprovalRequest
  onRespond: (id: string, response: ApprovalResponse) => void
}): React.JSX.Element {
  const [hovered, setHovered] = useState<Grant | null>(null)

  const pattern = useMemo(() => grantPatternFor(request), [request])
  const sessionGrant: Grant = { scope: 'session', pattern, applies: request.risk }
  const alwaysGrant: Grant = { scope: 'always', pattern, applies: request.risk }

  const respond = (response: ApprovalResponse): void => onRespond(request.id, response)
  // An internet request is approved one at a time. Standing grants are for
  // paths; what a request sends changes every time, so there is nothing safe to remember.
  const network = request.risk === 'network'

  // Escape declines. Refusing is the safe direction, so it is the one that gets
  // the shortcut; there is deliberately no keyboard shortcut for approving.
  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') respond({ allow: false, grant: null })
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  return (
    <div className="scrim" role="dialog" aria-modal="true" aria-labelledby="approval-title">
      <div className="dialog">
        <header>
          <div className="kicker">Approval needed</div>
          <h2 id="approval-title">
            The model wants to <span className={`risk-${request.risk}`}>{VERBS[request.call.name]}</span>
            {network ? '' : request.call.name === 'list_directory' ? ' this folder' : ' this file'}
          </h2>
        </header>

        <div className="target">{request.resolved}</div>
        <div className="why">{request.because}</div>

        <div className="preview">
          <Diff request={request} />
        </div>

        <footer>
          <button className="danger" onClick={() => respond({ allow: false, grant: null })}>
            Deny
          </button>
          <div style={{ flex: 1 }} />
          <button className="primary" onClick={() => respond({ allow: true, grant: null })}>
            {network ? 'Send' : 'Allow once'}
          </button>
          {network ? null : (
          <>
          <button
            onMouseEnter={() => setHovered(sessionGrant)}
            onMouseLeave={() => setHovered(null)}
            onFocus={() => setHovered(sessionGrant)}
            onBlur={() => setHovered(null)}
            onClick={() => respond({ allow: true, grant: sessionGrant })}
          >
            Allow for this session
          </button>
          <button
            onMouseEnter={() => setHovered(alwaysGrant)}
            onMouseLeave={() => setHovered(null)}
            onFocus={() => setHovered(alwaysGrant)}
            onBlur={() => setHovered(null)}
            onClick={() => respond({ allow: true, grant: alwaysGrant })}
          >
            Always allow
          </button>
          </>
          )}
          <div className="grant-note">
            {network
              ? 'Only this request is sent. To stop being asked, set internet requests to run automatically under Permissions.'
              : hovered
                ? describeGrant(hovered)
                : 'Allowing once applies to this action only. The other two create a rule you can remove later.'}
          </div>
        </footer>
      </div>
    </div>
  )
}
