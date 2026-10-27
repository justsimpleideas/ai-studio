import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { canonicalize, comparablePath, isContained } from '../src/main/broker/paths.js'

const WIN = process.platform === 'win32'

/**
 * These tests build real junctions and real files rather than stubbing the
 * filesystem. The bypasses they cover are properties of how Windows resolves
 * names, so a mock that behaved the way we assumed would prove nothing.
 */
describe('canonicalize', () => {
  let sandbox: string
  let realSandbox: string

  beforeAll(() => {
    sandbox = fs.mkdtempSync(path.join(os.tmpdir(), 'lla-paths-'))
    realSandbox = fs.realpathSync.native(sandbox)
    fs.mkdirSync(path.join(realSandbox, 'work'))
    fs.mkdirSync(path.join(realSandbox, 'secrets'))
    fs.writeFileSync(path.join(realSandbox, 'work', 'notes.txt'), 'hello')
    fs.writeFileSync(path.join(realSandbox, 'notes.txt'), 'top level')
    fs.writeFileSync(path.join(realSandbox, 'secrets', 'keys.txt'), 'sensitive')
  })

  afterAll(() => {
    fs.rmSync(sandbox, { recursive: true, force: true })
  })

  const ok = (input: string) => {
    const r = canonicalize(input)
    if (!r.ok) throw new Error(`expected resolvable, got ${r.kind}: ${r.detail}`)
    return r
  }

  it('resolves an ordinary existing file', () => {
    const r = ok(path.join(realSandbox, 'work', 'notes.txt'))
    expect(r.existed).toBe(true)
    expect(comparablePath(r.path)).toBe(
      comparablePath(path.join(realSandbox, 'work', 'notes.txt'))
    )
  })

  it('accepts a file that does not exist yet, and says so', () => {
    const r = ok(path.join(realSandbox, 'work', 'new-file.txt'))
    expect(r.existed).toBe(false)
    expect(comparablePath(r.path)).toBe(
      comparablePath(path.join(realSandbox, 'work', 'new-file.txt'))
    )
  })

  /**
   * Built by concatenation rather than path.join, because join normalizes ".."
   * away before the call is even made. A test that used it would be checking
   * Node's normalizer, not ours.
   */
  const raw = (...parts: string[]) => parts.join(path.sep)

  it('collapses a lexical detour through ..', () => {
    const r = ok(raw(realSandbox, 'work', '..', 'work', 'notes.txt'))
    expect(comparablePath(r.path)).toBe(
      comparablePath(path.join(realSandbox, 'work', 'notes.txt'))
    )
  })

  it('rejects a relative path', () => {
    const r = canonicalize('notes.txt')
    expect(r.ok).toBe(false)
    if (!r.ok) expect(r.kind).toBe('invalid-path')
  })

  it('rejects an empty path and a null byte', () => {
    expect(canonicalize('').ok).toBe(false)
    expect(canonicalize(`${realSandbox}\u0000.txt`).ok).toBe(false)
  })

  it('rejects a non-string', () => {
    expect(canonicalize(undefined).ok).toBe(false)
    expect(canonicalize(42).ok).toBe(false)
  })

  it.runIf(WIN)('rejects a drive-relative path', () => {
    // "C:work" means "work, under whatever directory C: is currently sitting in".
    const r = canonicalize('C:work\\notes.txt')
    expect(r.ok).toBe(false)
    if (!r.ok) expect(r.detail).toMatch(/absolute/)
  })

  it.runIf(WIN)('rejects a path rooted on the current drive', () => {
    const r = canonicalize('\\work\\notes.txt')
    expect(r.ok).toBe(false)
    if (!r.ok) expect(r.detail).toMatch(/drive or a share/)
  })

  it.runIf(WIN)('rejects an alternate data stream', () => {
    const r = canonicalize(path.join(realSandbox, 'work', 'notes.txt:hidden'))
    expect(r.ok).toBe(false)
    if (!r.ok) expect(r.detail).toMatch(/alternate data stream/)
  })

  it.runIf(WIN)('rejects reserved device names', () => {
    for (const name of ['CON', 'nul', 'COM1', 'LPT9', 'aux.txt']) {
      const r = canonicalize(path.join(realSandbox, name))
      expect(r.ok, name).toBe(false)
    }
  })

  it.runIf(WIN)('rejects a segment with a trailing dot or space', () => {
    expect(canonicalize(path.join(realSandbox, 'work', 'notes.txt.')).ok).toBe(false)
    expect(canonicalize(`${path.join(realSandbox, 'work', 'notes.txt')} `).ok).toBe(false)
  })

  it.runIf(WIN)('accepts and normalizes the extended-length prefix', () => {
    const r = ok(`\\\\?\\${path.join(realSandbox, 'work', 'notes.txt')}`)
    expect(r.path.startsWith('\\\\?\\')).toBe(false)
    expect(comparablePath(r.path)).toBe(
      comparablePath(path.join(realSandbox, 'work', 'notes.txt'))
    )
  })

  it('follows a junction to where it actually points', () => {
    const link = path.join(realSandbox, 'work', 'escape')
    try {
      fs.symlinkSync(path.join(realSandbox, 'secrets'), link, 'junction')
    } catch {
      return // no permission to create links here; nothing to assert
    }
    const r = ok(path.join(link, 'keys.txt'))
    expect(comparablePath(r.path)).toBe(
      comparablePath(path.join(realSandbox, 'secrets', 'keys.txt'))
    )
    // The whole point: the resolved path is no longer under work/.
    expect(isContained(r.path, path.join(realSandbox, 'work'))).toBe(false)
  })

  it('resolves .. relative to a junction target, not to the link', () => {
    const link = path.join(realSandbox, 'work', 'escape2')
    try {
      fs.symlinkSync(path.join(realSandbox, 'secrets'), link, 'junction')
    } catch {
      return
    }
    // Lexically this is work/notes.txt. Truthfully it is secrets/../notes.txt.
    const r = ok(raw(link, '..', 'notes.txt'))
    expect(comparablePath(r.path)).toBe(comparablePath(path.join(realSandbox, 'notes.txt')))
  })
})

describe('isContained', () => {
  it('accepts the root itself and anything under it', () => {
    expect(isContained('C:\\work', 'C:\\work')).toBe(true)
    expect(isContained('C:\\work\\a\\b.txt', 'C:\\work')).toBe(true)
  })

  it('does not let a root swallow a sibling that merely shares its prefix', () => {
    expect(isContained('C:\\work-secrets\\keys.txt', 'C:\\work')).toBe(false)
    expect(isContained('C:\\workshop', 'C:\\work')).toBe(false)
  })

  it('rejects a parent of the root', () => {
    expect(isContained('C:\\', 'C:\\work')).toBe(false)
  })

  it.runIf(WIN)('compares case-insensitively and separator-insensitively', () => {
    expect(isContained('c:/WORK/a.txt', 'C:\\work')).toBe(true)
  })

  it('tolerates a trailing separator on the root', () => {
    expect(isContained('C:\\work\\a.txt', 'C:\\work\\')).toBe(true)
  })
})
