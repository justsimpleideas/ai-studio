import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { Broker } from '../src/main/broker/index.js'
import { evaluate } from '../src/main/broker/policy.js'
import { Journal } from '../src/main/tools/journal.js'
import type { ApprovalRequest, ApprovalResponse } from '../src/shared/broker-types.js'
import {
  DEFAULT_PERMISSION_SETTINGS,
  DEFAULT_PROTECTED_PATHS,
  withDefaultProtections,
  type PermissionSettings,
  type ProtectedPath
} from '../src/shared/permissions.js'
import type { FileToolCall as ToolCall } from '../src/shared/tools.js'

/**
 * The guarantees that must hold even when the model ignores its instructions:
 * protected paths, no blind overwrites, a brake on runaway deletes, and a way
 * back from any change.
 */

let sandbox: string
let work: string

beforeEach(() => {
  sandbox = fs.realpathSync.native(fs.mkdtempSync(path.join(os.tmpdir(), 'las-safety-')))
  work = path.join(sandbox, 'work')
  fs.mkdirSync(path.join(work, '.git'), { recursive: true })
  fs.mkdirSync(path.join(work, 'src', 'config'), { recursive: true })
  fs.mkdirSync(path.join(work, 'data', 'raw'), { recursive: true })
  fs.writeFileSync(path.join(work, '.git', 'HEAD'), 'ref: refs/heads/main\n')
  fs.writeFileSync(path.join(work, 'src', 'config', '.env.local'), 'TOKEN=secret')
  fs.writeFileSync(path.join(work, 'data', 'raw', 'survey.csv'), 'a,b\n1,2\n')
  fs.writeFileSync(path.join(work, 'notes.txt'), 'alpha\n')
})

afterEach(() => {
  fs.rmSync(sandbox, { recursive: true, force: true })
})

function settings(protectedPaths: ProtectedPath[] = DEFAULT_PROTECTED_PATHS): PermissionSettings {
  return {
    ...DEFAULT_PERMISSION_SETTINGS,
    // Everything automatic: the protections must hold without anyone asking.
    modes: { read: 'auto', write: 'auto', destructive: 'auto', network: 'ask' },
    roots: [{ id: 'working-folder', path: work, label: 'work', enabled: true, addedAt: '' }],
    protectedPaths
  }
}

const call = (name: ToolCall['name'], args: Record<string, string>): ToolCall =>
  ({ id: 'c', name, args }) as ToolCall

function makeBroker(protectedPaths?: ProtectedPath[]) {
  const prompt = vi.fn(async (_r: ApprovalRequest): Promise<ApprovalResponse> => ({ allow: true, grant: null }))
  const trash = vi.fn(async (target: string) => fs.rmSync(target, { force: true }))
  const journal = new Journal(path.join(sandbox, 'backups'), path.join(sandbox, 'journal.json'))
  const broker = new Broker({
    getSettings: () => settings(protectedPaths),
    saveRule: async () => undefined,
    prompt,
    trash,
    audit: () => undefined,
    journal,
    workingFolder: work
  })
  return { broker, prompt, journal }
}

describe('protected paths', () => {
  const at = (...p: string[]) => path.join(work, ...p)

  it('keep secrets out of reach at any depth, even with every mode on automatic', () => {
    for (const tool of ['read_file', 'write_file', 'delete_file', 'list_directory'] as const) {
      const v = evaluate(settings(), tool, at('src', 'config', '.env.local'))
      expect(v.decision).toBe('deny')
      expect(v.final).toBe(true)
      expect(v.reason.kind).toBe('protected')
    }
  })

  it('let version control be read but never changed, down to every file inside', () => {
    expect(evaluate(settings(), 'read_file', at('.git', 'HEAD')).decision).toBe('allow')
    expect(evaluate(settings(), 'list_directory', at('.git')).decision).toBe('allow')
    for (const tool of ['write_file', 'edit_file', 'delete_file'] as const) {
      expect(evaluate(settings(), tool, at('.git', 'HEAD')).decision).toBe('deny')
      expect(evaluate(settings(), tool, at('.git', 'hooks', 'new-hook')).decision).toBe('deny')
    }
  })

  it('anchor a pattern with a slash at the top of the folder', () => {
    const rules: ProtectedPath[] = [
      { id: 'u', pattern: 'data/raw', level: 'read-only', enabled: true, builtin: false }
    ]
    expect(evaluate(settings(rules), 'write_file', at('data', 'raw', 'survey.csv')).decision).toBe('deny')
    expect(evaluate(settings(rules), 'write_file', at('data', 'clean.csv')).decision).toBe('allow')
    expect(evaluate(settings(rules), 'write_file', at('src', 'data', 'raw', 'x.csv')).decision).toBe('allow')
  })

  it('let the stronger protection win, and ignore one that is switched off', () => {
    const both: ProtectedPath[] = [
      { id: 'a', pattern: 'data', level: 'read-only', enabled: true, builtin: false },
      { id: 'b', pattern: 'raw', level: 'hidden', enabled: true, builtin: false }
    ]
    expect(evaluate(settings(both), 'read_file', at('data', 'raw', 'survey.csv')).decision).toBe('deny')
    const off: ProtectedPath[] = [{ id: 'c', pattern: '.git', level: 'read-only', enabled: false, builtin: true }]
    expect(evaluate(settings(off), 'write_file', at('.git', 'HEAD')).decision).toBe('allow')
  })

  it.runIf(process.platform === 'win32')('ignore case on Windows', () => {
    expect(evaluate(settings(), 'write_file', at('.GIT', 'HEAD')).decision).toBe('deny')
    expect(evaluate(settings(), 'read_file', at('src', 'config', '.ENV.local')).decision).toBe('deny')
  })

  it('are never put to the user: no approval can open them', async () => {
    const { broker, prompt } = makeBroker()
    const outcome = await broker.execute(call('delete_file', { path: '.git/HEAD' }))
    expect(outcome.ok).toBe(false)
    expect(prompt).not.toHaveBeenCalled()
    expect(fs.existsSync(at('.git', 'HEAD'))).toBe(true)
  })

  it('leave hidden paths out of folder listings', async () => {
    const { broker } = makeBroker()
    const outcome = await broker.execute(call('list_directory', { path: 'src/config' }))
    expect(outcome.ok).toBe(true)
    if (outcome.ok) expect((outcome.result as { entries: unknown[] }).entries).toEqual([])
  })

  it('keep the defaults a workspace has not seen yet, and respect the ones switched off', () => {
    const saved: ProtectedPath[] = [{ ...(DEFAULT_PROTECTED_PATHS[0] as ProtectedPath), enabled: false }]
    const merged = withDefaultProtections(saved)
    expect(merged).toHaveLength(DEFAULT_PROTECTED_PATHS.length)
    expect(merged.find((p) => p.id === saved[0]?.id)?.enabled).toBe(false)
  })
})

describe('no blind changes', () => {
  it('refuse to replace or edit an existing file that has not been read', async () => {
    const { broker, prompt } = makeBroker()
    const write = await broker.execute(call('write_file', { path: 'notes.txt', content: 'overwritten' }))
    const edit = await broker.execute(call('edit_file', { path: 'notes.txt', old_text: 'alpha', new_text: 'beta' }))
    expect(write.ok).toBe(false)
    expect(edit.ok).toBe(false)
    if (!write.ok) expect(write.error).toMatch(/Read it with read_file first/)
    expect(prompt).not.toHaveBeenCalled()
    expect(fs.readFileSync(path.join(work, 'notes.txt'), 'utf8')).toBe('alpha\n')
  })

  it('allow it once read, and refuse again if the file changed after reading', async () => {
    const { broker } = makeBroker()
    await broker.execute(call('read_file', { path: 'notes.txt' }))
    expect((await broker.execute(call('edit_file', { path: 'notes.txt', old_text: 'alpha', new_text: 'beta' }))).ok).toBe(true)
    // The agent's own change keeps it current, so a second edit is fine.
    expect((await broker.execute(call('edit_file', { path: 'notes.txt', old_text: 'beta', new_text: 'gamma' }))).ok).toBe(true)

    fs.writeFileSync(path.join(work, 'notes.txt'), 'changed by the user, much longer\n')
    const stale = await broker.execute(call('write_file', { path: 'notes.txt', content: 'x' }))
    expect(stale.ok).toBe(false)
    if (!stale.ok) expect(stale.error).toMatch(/changed since it was read/)
  })

  it('let new files be created without reading anything', async () => {
    const { broker } = makeBroker()
    expect((await broker.execute(call('write_file', { path: 'src/new.ts', content: 'export {}' }))).ok).toBe(true)
  })

  it('forget what was read when the conversation changes', async () => {
    const { broker } = makeBroker()
    await broker.execute(call('read_file', { path: 'notes.txt' }))
    broker.resetConversation()
    expect((await broker.execute(call('write_file', { path: 'notes.txt', content: 'x' }))).ok).toBe(false)
  })
})

describe('runaway changes', () => {
  it('stop to ask after three deletes in one reply, even on automatic', async () => {
    const { broker, prompt } = makeBroker()
    for (let i = 0; i < 5; i++) fs.writeFileSync(path.join(work, `tmp${i}.txt`), 'x')
    for (let i = 0; i < 3; i++) {
      expect((await broker.execute(call('delete_file', { path: `tmp${i}.txt` }), { promptId: 'p1' })).ok).toBe(true)
    }
    expect(prompt).not.toHaveBeenCalled()
    prompt.mockResolvedValueOnce({ allow: false, grant: null })
    const fourth = await broker.execute(call('delete_file', { path: 'tmp3.txt' }), { promptId: 'p1' })
    expect(prompt).toHaveBeenCalledTimes(1)
    expect(prompt.mock.calls[0]?.[0].because).toMatch(/already deleted 3 files/)
    expect(fourth.ok).toBe(false)
    expect(fs.existsSync(path.join(work, 'tmp3.txt'))).toBe(true)

    // A new reply starts its own count.
    expect((await broker.execute(call('delete_file', { path: 'tmp4.txt' }), { promptId: 'p2' })).ok).toBe(true)
  })
})

describe('reverting a reply', () => {
  async function aReply(broker: Broker): Promise<void> {
    await broker.execute(call('read_file', { path: 'notes.txt' }), { promptId: 'r1' })
    await broker.execute(call('edit_file', { path: 'notes.txt', old_text: 'alpha', new_text: 'beta' }), { promptId: 'r1' })
    await broker.execute(call('write_file', { path: 'src/made.ts', content: 'new file' }), { promptId: 'r1' })
    await broker.execute(call('delete_file', { path: 'data/raw/survey.csv' }), { promptId: 'r1' })
  }

  it('puts back every file the reply touched: edited, created and deleted', async () => {
    const { broker, journal } = makeBroker()
    journal.label('r1', 'Tidy the project', 'chat-1')
    await aReply(broker)
    expect(fs.existsSync(path.join(work, 'data', 'raw', 'survey.csv'))).toBe(false)

    const [group] = journal.groups()
    expect(group?.label).toBe('Tidy the project')
    expect(group?.files.map((f) => path.basename(f.path)).sort()).toEqual(['made.ts', 'notes.txt', 'survey.csv'])

    const result = await journal.revert('r1')
    expect(result.done).toBe(true)
    expect(fs.readFileSync(path.join(work, 'notes.txt'), 'utf8')).toBe('alpha\n')
    expect(fs.existsSync(path.join(work, 'src', 'made.ts'))).toBe(false)
    expect(fs.readFileSync(path.join(work, 'data', 'raw', 'survey.csv'), 'utf8')).toBe('a,b\n1,2\n')
    expect(journal.groups().find((g) => g.id === 'r1')?.reverted).toBe(true)
  })

  it('does nothing when a file changed since, until told to go ahead, and that can be undone too', async () => {
    const { broker, journal } = makeBroker()
    await aReply(broker)
    fs.writeFileSync(path.join(work, 'notes.txt'), 'my own edit\n')

    const careful = await journal.revert('r1')
    expect(careful.done).toBe(false)
    expect(careful.conflicts.map((p) => path.basename(p))).toEqual(['notes.txt'])
    expect(fs.existsSync(path.join(work, 'src', 'made.ts'))).toBe(true)

    const forced = await journal.revert('r1', true)
    expect(forced.done).toBe(true)
    expect(fs.readFileSync(path.join(work, 'notes.txt'), 'utf8')).toBe('alpha\n')

    // The user's own edit was backed up by the revert, so undoing the revert brings it back.
    const undo = journal.groups().find((g) => g.revertOf === 'r1')
    expect(undo).toBeDefined()
    await journal.revert(undo!.id, true)
    expect(fs.readFileSync(path.join(work, 'notes.txt'), 'utf8')).toBe('my own edit\n')
    expect(journal.groups().find((g) => g.id === 'r1')?.reverted).toBe(false)
  })

  it('still works after a restart', async () => {
    const { broker } = makeBroker()
    await aReply(broker)
    const reopened = new Journal(path.join(sandbox, 'backups'), path.join(sandbox, 'journal.json'))
    expect(reopened.groups()).toHaveLength(1)
    expect((await reopened.revert('r1')).done).toBe(true)
    expect(fs.readFileSync(path.join(work, 'notes.txt'), 'utf8')).toBe('alpha\n')
  })
})
