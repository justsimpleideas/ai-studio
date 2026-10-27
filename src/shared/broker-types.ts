import type { GrantScope } from './permissions.js'
import type { RiskClass, ToolCall, ToolName } from './tools.js'

/**
 * The shapes the window and the broker share.
 *
 * They live in shared rather than beside the broker because the approval dialog
 * has to render them, and the renderer must not be able to import anything from
 * the main process. Types crossing that line is fine; code crossing it is how a
 * view ends up holding a file handle.
 */

/** What the user is shown before they decide. */
export type Preview =
  | { kind: 'create'; after: string }
  | { kind: 'overwrite'; before: string; after: string }
  | { kind: 'edit'; before: string; after: string; at: number }
  | { kind: 'delete'; bytes: number }
  /** An internet request: exactly what leaves the computer, and where it goes. */
  | { kind: 'network'; sends: string; to: string; saveTo: string | null }
  | { kind: 'none' }

export interface ApprovalRequest {
  id: string
  call: ToolCall
  risk: RiskClass
  /** The real path on disk, which is what the prompt must show, not what the model typed. */
  resolved: string
  /** Shown verbatim, so the reason a prompt appeared is never a mystery. */
  because: string
  preview: Preview
}

export interface Grant {
  scope: Exclude<GrantScope, 'once'>
  pattern: string
  applies: ToolName[] | RiskClass | '*'
}

export interface ApprovalResponse {
  allow: boolean
  /** Non-null only when the user chose to remember the decision. */
  grant: Grant | null
}

export type ToolOutcome =
  | {
      ok: true
      tool: ToolName
      resolved: string
      result: unknown
      /** False when the call ran without a prompt. Drives the marker in the activity log. */
      asked: boolean
    }
  | {
      ok: false
      tool: ToolName
      resolved: string | null
      error: string
      hint?: string
      /** True when policy or the user refused, as opposed to the action failing. */
      refused: boolean
    }

export interface AuditEntry {
  at: string
  tool: ToolName
  path: string | null
  decision: 'allowed' | 'refused' | 'failed'
  asked: boolean
  reason: string
  detail?: string
}

/**
 * Put the grant a "remember this" choice would create into words.
 *
 * The dialog shows this sentence next to the button, because a person agreeing
 * to a standing rule should be reading the rule, not inferring it from the
 * label on a checkbox.
 */
export function describeGrant(grant: Grant): string {
  const folderPattern = grant.pattern.endsWith('/**')
  const where = folderPattern
    ? `everything in ${grant.pattern.slice(0, -3)}`
    : grant.pattern
  const what =
    grant.applies === '*'
      ? 'every action'
      : Array.isArray(grant.applies)
        ? grant.applies.join(' and ')
        : `${grant.applies} actions`
  const how = grant.scope === 'session' ? 'for this session' : 'from now on'
  return `Allow ${what} on ${where} ${how}.`
}
