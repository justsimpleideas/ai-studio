import type { RiskClass, ToolName } from './tools.js'

/**
 * What the broker decided to do with a tool call.
 *
 * 'ask' is not a failure state — it is the default and the point of the product.
 * 'deny' from a boundary check is final: no rule and no user click can turn it
 * into an allow, because those checks are what make the roots mean anything.
 */
export type Decision = 'allow' | 'ask' | 'deny'

/** How long an approval the user grants stays good for. */
export type GrantScope = 'once' | 'session' | 'always'

/**
 * A root the agent is allowed to operate under. Every path in every tool call is
 * resolved to its real location on disk and then tested against these. A path
 * that resolves outside all of them is refused before any rule is consulted.
 */
export interface WorkspaceRoot {
  id: string
  /** Canonical, real (symlink- and junction-resolved) absolute path. */
  path: string
  label: string
  /** When false the root stays configured but is inert — a quick off switch. */
  enabled: boolean
  addedAt: string
}

/**
 * A standing decision the user has recorded, either from the Settings screen or
 * by picking "always" in an approval prompt.
 */
export interface PolicyRule {
  id: string
  /**
   * Glob matched against the canonical path, using forward slashes and compared
   * case-insensitively on Windows. `**` crosses directory separators, `*` does not.
   */
  pattern: string
  /** Which tools this covers. '*' means every tool; a risk class means every tool in it. */
  applies: ToolName[] | RiskClass | '*'
  decision: 'allow' | 'deny'
  /** Free-text note shown in the rules list, e.g. "granted from approval prompt". */
  note?: string
  createdAt: string
}

/**
 * How firmly a path inside a shared folder is kept from the agent.
 *
 * 'hidden': no tool may touch it, and directory listings leave it out.
 * 'read-only': it can be listed and read, never written, edited or deleted.
 *
 * Both are final. They are checked by the broker on every call, before any rule,
 * mode or approval, so an agent that ignores its instructions, or is talked into
 * something by text it read, still cannot get past them.
 */
export type ProtectionLevel = 'hidden' | 'read-only'

export interface ProtectedPath {
  id: string
  /**
   * Relative to the shared folder, with forward slashes. A pattern without a
   * slash ("*.pem", ".git") matches that name at any depth; one with a slash
   * ("data/raw") is anchored at the folder. A match on a folder covers
   * everything inside it.
   */
  pattern: string
  level: ProtectionLevel
  enabled: boolean
  /** Part of the default set: can be switched off, not removed. */
  builtin: boolean
}

/**
 * Applied to every workspace from the start, so the dangerous cases are covered
 * without the user listing them: secrets are out of reach entirely, and
 * version-control data and installed dependencies cannot be changed.
 */
export const DEFAULT_PROTECTED_PATHS: ProtectedPath[] = [
  ...['.env', '.env.*', '*.pem', '*.key', '*.p12', '*.pfx', 'id_rsa*', 'id_ed25519*', 'id_ecdsa*', '.ssh', '.aws', '.netrc', '.git-credentials', '.npmrc', '.pypirc', '*.kdbx'].map(
    (pattern): ProtectedPath => ({ id: `builtin:${pattern}`, pattern, level: 'hidden', enabled: true, builtin: true })
  ),
  ...['.git', '.hg', '.svn', 'node_modules', '.venv', 'venv'].map(
    (pattern): ProtectedPath => ({ id: `builtin:${pattern}`, pattern, level: 'read-only', enabled: true, builtin: true })
  )
]

/**
 * A workspace's protections: its saved list, with any default that list does
 * not know about yet added (a workspace created before a default existed still
 * gets it, and a default the user switched off stays off).
 */
export function withDefaultProtections(saved: ProtectedPath[] | undefined): ProtectedPath[] {
  const list = Array.isArray(saved) ? saved : []
  const known = new Set(list.map((p) => p.id))
  return [...list, ...DEFAULT_PROTECTED_PATHS.filter((d) => !known.has(d.id))]
}

/**
 * Per-risk-class default for calls that no rule covers.
 *
 * 'auto' is what the user means by "just go ahead" / don't ask me again. It is kept
 * per risk class rather than as one global switch so that the common, sane
 * setting — read freely, confirm every write — is expressible without writing
 * a single rule.
 */
export type ClassMode = 'ask' | 'auto'

export interface PermissionSettings {
  roots: WorkspaceRoot[]
  rules: PolicyRule[]
  modes: Record<RiskClass, ClassMode>
  /**
   * Master switch. While true every call is asked about regardless of modes and
   * allow-rules, so a user who gets nervous mid-session has one thing to click.
   * Deny rules and boundary checks still apply.
   */
  confirmEverything: boolean
  /** Paths inside the roots that are hidden or read-only. Checked before everything but the roots. */
  protectedPaths?: ProtectedPath[]
}

export const DEFAULT_PERMISSION_SETTINGS: PermissionSettings = {
  roots: [],
  rules: [],
  modes: { read: 'ask', write: 'ask', destructive: 'ask', network: 'ask' },
  confirmEverything: false
}

/** Why the broker ruled the way it did — surfaced verbatim in the approval UI and the audit log. */
export type DecisionReason =
  | { kind: 'no-roots' }
  | { kind: 'outside-roots'; resolved: string }
  | { kind: 'unresolvable'; detail: string }
  | { kind: 'invalid-path'; detail: string }
  | { kind: 'rule'; ruleId: string; pattern: string }
  | { kind: 'protected'; pattern: string; level: ProtectionLevel }
  | { kind: 'confirm-everything' }
  | { kind: 'class-mode'; risk: RiskClass; mode: ClassMode }

export interface PolicyVerdict {
  decision: Decision
  reason: DecisionReason
  /** Canonical path the decision was made about. Null when it could not be resolved. */
  resolved: string | null
  /** True when no rule or click can overturn this. Always true for boundary denials. */
  final: boolean
}

export function humanizeReason(reason: DecisionReason): string {
  switch (reason.kind) {
    case 'no-roots':
      return 'No folders have been shared with the agent yet.'
    case 'outside-roots':
      return `That path is outside every shared folder (resolves to ${reason.resolved}).`
    case 'unresolvable':
      return `That path could not be resolved on disk: ${reason.detail}`
    case 'invalid-path':
      return `That path is not usable: ${reason.detail}`
    case 'rule':
      return `Matched your rule for ${reason.pattern}.`
    case 'protected':
      return reason.level === 'hidden'
        ? `That path is protected (${reason.pattern}) and cannot be accessed.`
        : `That path is read-only (${reason.pattern}) and cannot be changed.`
    case 'confirm-everything':
      return 'Confirm-everything is on, so this needs your approval.'
    case 'class-mode':
      return reason.mode === 'auto'
        ? `${reason.risk} actions are set to run automatically.`
        : `${reason.risk} actions are set to ask first.`
  }
}
