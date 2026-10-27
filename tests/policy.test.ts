import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { evaluate, globToRegExp, ruleCoversTool } from '../src/main/broker/policy.js'
import type { PermissionSettings, PolicyRule, WorkspaceRoot } from '../src/shared/permissions.js'
import { DEFAULT_PERMISSION_SETTINGS } from '../src/shared/permissions.js'

const WIN = process.platform === 'win32'

function root(p: string, overrides: Partial<WorkspaceRoot> = {}): WorkspaceRoot {
  return {
    id: 'root-1',
    path: p,
    label: 'sandbox',
    enabled: true,
    addedAt: '2026-01-01T00:00:00.000Z',
    ...overrides
  }
}

function rule(overrides: Partial<PolicyRule> & Pick<PolicyRule, 'pattern' | 'decision'>): PolicyRule {
  return {
    id: 'rule-1',
    applies: '*',
    createdAt: '2026-01-01T00:00:00.000Z',
    ...overrides
  }
}

function settings(overrides: Partial<PermissionSettings> = {}): PermissionSettings {
  return { ...DEFAULT_PERMISSION_SETTINGS, ...overrides }
}

describe('globToRegExp', () => {
  it('matches a literal path exactly', () => {
    const re = globToRegExp('C:/work/notes.txt')
    expect(re.test('c:/work/notes.txt')).toBe(true)
    expect(re.test('c:/work/notes.txt.bak')).toBe(false)
  })

  it('does not let a single star cross a separator', () => {
    const re = globToRegExp('C:/work/*.txt')
    expect(re.test('c:/work/notes.txt')).toBe(true)
    expect(re.test('c:/work/sub/notes.txt')).toBe(false)
  })

  it('lets a double star cross separators', () => {
    const re = globToRegExp('C:/work/**')
    expect(re.test('c:/work/notes.txt')).toBe(true)
    expect(re.test('c:/work/a/b/c/notes.txt')).toBe(true)
  })

  it('lets **/ match nothing at all', () => {
    const re = globToRegExp('C:/work/**/notes.txt')
    expect(re.test('c:/work/notes.txt')).toBe(true)
    expect(re.test('c:/work/a/b/notes.txt')).toBe(true)
  })

  it('treats regex metacharacters in the pattern as literals', () => {
    const re = globToRegExp('C:/work/report (final).txt')
    expect(re.test('c:/work/report (final).txt')).toBe(true)
    expect(re.test('c:/work/report final.txt')).toBe(false)
  })

  it('matches a single character with ?', () => {
    const re = globToRegExp('C:/work/log?.txt')
    expect(re.test('c:/work/log1.txt')).toBe(true)
    expect(re.test('c:/work/log12.txt')).toBe(false)
  })
})

describe('ruleCoversTool', () => {
  it('covers everything with *', () => {
    expect(ruleCoversTool(rule({ pattern: '**', decision: 'allow' }), 'delete_file')).toBe(true)
  })

  it('covers a named list of tools', () => {
    const r = rule({ pattern: '**', decision: 'allow', applies: ['read_file', 'list_directory'] })
    expect(ruleCoversTool(r, 'read_file')).toBe(true)
    expect(ruleCoversTool(r, 'write_file')).toBe(false)
  })

  it('covers a whole risk class', () => {
    const r = rule({ pattern: '**', decision: 'allow', applies: 'read' })
    expect(ruleCoversTool(r, 'list_directory')).toBe(true)
    expect(ruleCoversTool(r, 'delete_file')).toBe(false)
  })
})

describe('evaluate', () => {
  let sandbox: string
  let inside: string
  let sibling: string

  beforeAll(() => {
    sandbox = fs.realpathSync.native(fs.mkdtempSync(path.join(os.tmpdir(), 'lla-policy-')))
    inside = path.join(sandbox, 'work')
    sibling = path.join(sandbox, 'work-secrets')
    fs.mkdirSync(inside)
    fs.mkdirSync(sibling)
    fs.writeFileSync(path.join(inside, 'notes.txt'), 'hello')
    fs.writeFileSync(path.join(sibling, 'keys.txt'), 'sensitive')
  })

  afterAll(() => {
    fs.rmSync(sandbox, { recursive: true, force: true })
  })

  it('refuses everything when no folder has been shared', () => {
    const v = evaluate(settings(), 'read_file', path.join(inside, 'notes.txt'))
    expect(v.decision).toBe('deny')
    expect(v.reason.kind).toBe('no-roots')
    expect(v.final).toBe(true)
  })

  it('refuses a path outside every root, and that refusal is final', () => {
    const v = evaluate(
      settings({ roots: [root(inside)] }),
      'read_file',
      path.join(sibling, 'keys.txt')
    )
    expect(v.decision).toBe('deny')
    expect(v.reason.kind).toBe('outside-roots')
    expect(v.final).toBe(true)
  })

  it('does not treat a disabled root as shared', () => {
    const v = evaluate(
      settings({ roots: [root(inside, { enabled: false })] }),
      'read_file',
      path.join(inside, 'notes.txt')
    )
    expect(v.decision).toBe('deny')
    expect(v.reason.kind).toBe('no-roots')
  })

  it('refuses an unusable path before it looks at any rule', () => {
    const v = evaluate(
      settings({
        roots: [root(inside)],
        rules: [rule({ pattern: '**', decision: 'allow' })]
      }),
      'read_file',
      'notes.txt'
    )
    expect(v.decision).toBe('deny')
    expect(v.reason.kind).toBe('invalid-path')
    expect(v.final).toBe(true)
  })

  it('asks by default for a path inside a shared folder', () => {
    const v = evaluate(settings({ roots: [root(inside)] }), 'read_file', path.join(inside, 'notes.txt'))
    expect(v.decision).toBe('ask')
    expect(v.reason.kind).toBe('class-mode')
  })

  it('runs automatically when the risk class is set to auto', () => {
    const v = evaluate(
      settings({ roots: [root(inside)], modes: { read: 'auto', write: 'ask', destructive: 'ask', network: 'ask' } }),
      'read_file',
      path.join(inside, 'notes.txt')
    )
    expect(v.decision).toBe('allow')
  })

  it('keeps risk classes independent of each other', () => {
    const s = settings({
      roots: [root(inside)],
      modes: { read: 'auto', write: 'ask', destructive: 'ask', network: 'ask' }
    })
    expect(evaluate(s, 'list_directory', inside).decision).toBe('allow')
    expect(evaluate(s, 'write_file', path.join(inside, 'notes.txt')).decision).toBe('ask')
    expect(evaluate(s, 'delete_file', path.join(inside, 'notes.txt')).decision).toBe('ask')
  })

  it('honours an allow rule over an ask default', () => {
    const v = evaluate(
      settings({
        roots: [root(inside)],
        rules: [rule({ pattern: `${inside.replace(/\\/g, '/')}/**`, decision: 'allow' })]
      }),
      'write_file',
      path.join(inside, 'notes.txt')
    )
    expect(v.decision).toBe('allow')
    expect(v.reason.kind).toBe('rule')
  })

  it('lets a deny rule beat an allow rule for the same path', () => {
    const v = evaluate(
      settings({
        roots: [root(inside)],
        rules: [
          rule({ id: 'broad', pattern: `${inside.replace(/\\/g, '/')}/**`, decision: 'allow' }),
          rule({ id: 'narrow', pattern: '**/notes.txt', decision: 'deny' })
        ]
      }),
      'write_file',
      path.join(inside, 'notes.txt')
    )
    expect(v.decision).toBe('deny')
    expect(v.final).toBe(true)
    if (v.reason.kind === 'rule') expect(v.reason.ruleId).toBe('narrow')
  })

  it('makes confirm-everything override an allow rule', () => {
    const v = evaluate(
      settings({
        roots: [root(inside)],
        rules: [rule({ pattern: '**', decision: 'allow' })],
        confirmEverything: true
      }),
      'write_file',
      path.join(inside, 'notes.txt')
    )
    expect(v.decision).toBe('ask')
    expect(v.reason.kind).toBe('confirm-everything')
  })

  it('makes confirm-everything override an auto risk class', () => {
    const v = evaluate(
      settings({
        roots: [root(inside)],
        modes: { read: 'auto', write: 'auto', destructive: 'auto', network: 'ask' },
        confirmEverything: true
      }),
      'delete_file',
      path.join(inside, 'notes.txt')
    )
    expect(v.decision).toBe('ask')
  })

  it('still denies outright while confirm-everything is on', () => {
    const v = evaluate(
      settings({
        roots: [root(inside)],
        rules: [rule({ pattern: '**/notes.txt', decision: 'deny' })],
        confirmEverything: true
      }),
      'write_file',
      path.join(inside, 'notes.txt')
    )
    expect(v.decision).toBe('deny')
  })

  it('does not let a rule reach outside the shared folders', () => {
    // The broadest possible allow rule must not widen the boundary.
    const v = evaluate(
      settings({
        roots: [root(inside)],
        rules: [rule({ pattern: '**', decision: 'allow' })],
        modes: { read: 'auto', write: 'auto', destructive: 'auto', network: 'ask' }
      }),
      'read_file',
      path.join(sibling, 'keys.txt')
    )
    expect(v.decision).toBe('deny')
    expect(v.reason.kind).toBe('outside-roots')
  })

  it('does not let a sibling folder sharing a name prefix slip in', () => {
    const v = evaluate(settings({ roots: [root(inside)] }), 'read_file', path.join(sibling, 'keys.txt'))
    expect(v.decision).toBe('deny')
    expect(v.reason.kind).toBe('outside-roots')
  })

  it('refuses a junction that points out of the shared folder', () => {
    const link = path.join(inside, 'escape')
    try {
      fs.symlinkSync(sibling, link, 'junction')
    } catch {
      return // cannot create links here, nothing to prove
    }
    const v = evaluate(
      settings({
        roots: [root(inside)],
        modes: { read: 'auto', write: 'auto', destructive: 'auto', network: 'ask' }
      }),
      'read_file',
      path.join(link, 'keys.txt')
    )
    expect(v.decision).toBe('deny')
    expect(v.reason.kind).toBe('outside-roots')
  })

  it.runIf(WIN)('refuses a drive-relative path that would otherwise look contained', () => {
    const v = evaluate(settings({ roots: [root(inside)] }), 'read_file', 'C:work/notes.txt')
    expect(v.decision).toBe('deny')
    expect(v.reason.kind).toBe('invalid-path')
  })

  it('reports the resolved path it decided about', () => {
    const v = evaluate(
      settings({ roots: [root(inside)] }),
      'read_file',
      path.join(inside, '..', 'work', 'notes.txt')
    )
    expect(v.resolved).not.toBeNull()
    expect(v.resolved?.toLowerCase()).toBe(path.join(inside, 'notes.txt').toLowerCase())
  })
})
