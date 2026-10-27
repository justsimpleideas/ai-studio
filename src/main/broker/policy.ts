import type {
  PermissionSettings,
  PolicyRule,
  PolicyVerdict,
  ProtectedPath,
  WorkspaceRoot
} from '@shared/permissions.js'
import { TOOL_RISK, type ToolName } from '@shared/tools.js'
import { canonicalize, comparablePath, isContained } from './paths.js'

/**
 * Deciding what happens to one tool call.
 *
 * Apart from the filesystem lookups inside canonicalize, this module is pure,
 * which is what lets the whole policy be tested exhaustively without an Electron
 * window or a loaded model anywhere near it.
 */

function escapeRegExp(ch: string): string {
  return /[.+^${}()|[\]\\]/.test(ch) ? '\\' + ch : ch
}

/**
 * Translate a glob into a regular expression over comparable paths.
 *
 * A single star stops at a separator and a double star crosses them. A double
 * star followed by a separator is additionally allowed to match nothing at all,
 * so a pattern for "notes.txt anywhere under work" still covers a notes.txt
 * sitting directly in work. Patterns are folded to comparable form first, so on
 * Windows the match is case-insensitive for the same reason containment is.
 */
export function globToRegExp(pattern: string): RegExp {
  const p = comparablePath(pattern)
  let out = '^'
  let i = 0
  while (i < p.length) {
    const c = p[i] as string
    if (c === '*') {
      if (p[i + 1] === '*') {
        if (p[i + 2] === '/') {
          out += '(?:.*/)?'
          i += 3
        } else {
          out += '.*'
          i += 2
        }
      } else {
        out += '[^/]*'
        i += 1
      }
    } else if (c === '?') {
      out += '[^/]'
      i += 1
    } else {
      out += escapeRegExp(c)
      i += 1
    }
  }
  return new RegExp(out + '$')
}

export function ruleCoversTool(rule: PolicyRule, tool: ToolName): boolean {
  if (rule.applies === '*') return true
  if (Array.isArray(rule.applies)) return rule.applies.includes(tool)
  return TOOL_RISK[tool] === rule.applies
}

export function ruleMatches(rule: PolicyRule, tool: ToolName, resolved: string): boolean {
  return ruleCoversTool(rule, tool) && globToRegExp(rule.pattern).test(comparablePath(resolved))
}

function enabledRoots(settings: PermissionSettings): WorkspaceRoot[] {
  return settings.roots.filter((r) => r.enabled)
}

/**
 * The protection covering `resolved`, if any: the strongest one matching the
 * path or any folder above it, measured from the shared folder that contains it.
 * 'hidden' outranks 'read-only'.
 */
export function protectionFor(settings: PermissionSettings, resolved: string): ProtectedPath | null {
  const protections = (settings.protectedPaths ?? []).filter((p) => p.enabled && p.pattern.trim())
  if (protections.length === 0) return null
  const target = comparablePath(resolved)
  let found: ProtectedPath | null = null
  for (const root of enabledRoots(settings)) {
    const base = comparablePath(root.path)
    if (!isContained(resolved, root.path) || target === base) continue
    const segments = target.slice(base.length).replace(/^\/+/, '').split('/')
    const prefixes = segments.map((_, i) => segments.slice(0, i + 1).join('/'))
    for (const p of protections) {
      const pattern = p.pattern.trim().replace(/\\/g, '/').replace(/^\.\//, '').replace(/\/+$/, '')
      const re = globToRegExp(pattern)
      const hit = pattern.includes('/') ? prefixes.some((x) => re.test(x)) : segments.some((x) => re.test(x))
      if (hit && (!found || (p.level === 'hidden' && found.level !== 'hidden'))) found = p
    }
  }
  return found
}

/**
 * The single decision point. Every tool call passes through here, and nothing
 * touches the disk without a verdict of 'allow' coming out of it.
 *
 * The order is the design. Boundary failures are settled before any rule is
 * read, so no rule can widen the reach of the agent past the folders the user
 * shared. Denials are settled before allowances, so the narrow thing always
 * beats the broad thing. And confirm-everything sits above the allow paths but
 * below the denials, because it is a brake, not a gate.
 */
export function evaluate(
  settings: PermissionSettings,
  tool: ToolName,
  rawPath: unknown
): PolicyVerdict {
  const resolution = canonicalize(rawPath)
  if (!resolution.ok) {
    return {
      decision: 'deny',
      reason: { kind: resolution.kind, detail: resolution.detail },
      resolved: null,
      final: true
    }
  }

  const resolved = resolution.path
  const roots = enabledRoots(settings)

  if (roots.length === 0) {
    return { decision: 'deny', reason: { kind: 'no-roots' }, resolved, final: true }
  }

  if (!roots.some((root) => isContained(resolved, root.path))) {
    return {
      decision: 'deny',
      reason: { kind: 'outside-roots', resolved },
      resolved,
      final: true
    }
  }

  // Protections come before every rule and mode, and no click can lift them.
  const protection = protectionFor(settings, resolved)
  if (protection && (protection.level === 'hidden' || TOOL_RISK[tool] !== 'read')) {
    return {
      decision: 'deny',
      reason: { kind: 'protected', pattern: protection.pattern, level: protection.level },
      resolved,
      final: true
    }
  }

  const matching = settings.rules.filter((rule) => ruleMatches(rule, tool, resolved))

  const denial = matching.find((rule) => rule.decision === 'deny')
  if (denial) {
    return {
      decision: 'deny',
      reason: { kind: 'rule', ruleId: denial.id, pattern: denial.pattern },
      resolved,
      final: true
    }
  }

  if (settings.confirmEverything) {
    return {
      decision: 'ask',
      reason: { kind: 'confirm-everything' },
      resolved,
      final: false
    }
  }

  const allowance = matching.find((rule) => rule.decision === 'allow')
  if (allowance) {
    return {
      decision: 'allow',
      reason: { kind: 'rule', ruleId: allowance.id, pattern: allowance.pattern },
      resolved,
      final: false
    }
  }

  const risk = TOOL_RISK[tool]
  const mode = settings.modes[risk]
  return {
    decision: mode === 'auto' ? 'allow' : 'ask',
    reason: { kind: 'class-mode', risk, mode },
    resolved,
    final: false
  }
}
