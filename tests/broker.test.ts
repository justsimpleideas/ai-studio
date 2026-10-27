import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from 'vitest'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { Broker } from '../src/main/broker/index.js'
import { Journal } from '../src/main/tools/fs-tools.js'
import type { ApprovalRequest, ApprovalResponse, AuditEntry } from '../src/shared/broker-types.js'
import type { PermissionSettings, PolicyRule } from '../src/shared/permissions.js'
import { DEFAULT_PERMISSION_SETTINGS } from '../src/shared/permissions.js'
import type { FileToolCall as ToolCall } from '../src/shared/tools.js'

/**
 * The broker end to end, against a real temp directory.
 *
 * These are the tests that would catch the failure nobody notices: an approval
 * granted for one path and an action carried out on another.
 */
describe('Broker', () => {
  let sandbox: string
  let shared: string
  let outside: string
  let settings: PermissionSettings
  let broker: Broker
  // Typed at the shape the broker expects, rather than cast at the call site —
  // an untyped mock would happily accept a signature change that breaks the
  // real wiring.
  let prompt: Mock<(request: ApprovalRequest) => Promise<ApprovalResponse>>
  let saveRule: Mock<(rule: PolicyRule) => Promise<void>>
  let trash: Mock<(target: string) => Promise<void>>
  let audit: AuditEntry[]

  beforeEach(() => {
    sandbox = fs.realpathSync.native(fs.mkdtempSync(path.join(os.tmpdir(), 'lla-broker-')))
    shared = path.join(sandbox, 'shared')
    outside = path.join(sandbox, 'private')
    fs.mkdirSync(shared)
    fs.mkdirSync(path.join(shared, 'sub'))
    fs.mkdirSync(outside)
    fs.writeFileSync(path.join(shared, 'notes.txt'), 'alpha\nbeta\ngamma\n')
    fs.writeFileSync(path.join(outside, 'secret.txt'), 'do not read me')

    settings = {
      ...DEFAULT_PERMISSION_SETTINGS,
      roots: [
        {
          id: 'r1',
          path: shared,
          label: 'shared',
          enabled: true,
          addedAt: '2026-01-01T00:00:00.000Z'
        }
      ]
    }

    audit = []
    prompt = vi.fn(async (_request: ApprovalRequest): Promise<ApprovalResponse> => ({
      allow: true,
      grant: null
    }))
    saveRule = vi.fn(async (_rule: PolicyRule) => undefined)
    trash = vi.fn(async (target: string) => {
      fs.rmSync(target, { force: true })
    })

    broker = new Broker({
      getSettings: () => settings,
      saveRule,
      prompt,
      trash,
      audit: (entry) => audit.push(entry),
      journal: new Journal(path.join(sandbox, 'backups'))
    })
  })

  afterEach(() => {
    fs.rmSync(sandbox, { recursive: true, force: true })
  })

  const call = (partial: Omit<ToolCall, 'id'>): ToolCall => ({ id: 'call-1', ...partial } as ToolCall)

  it('refuses a path outside the shared folder without ever prompting', async () => {
    const outcome = await broker.execute(
      call({ name: 'read_file', args: { path: path.join(outside, 'secret.txt') } })
    )
    expect(outcome.ok).toBe(false)
    if (!outcome.ok) expect(outcome.refused).toBe(true)
    expect(prompt).not.toHaveBeenCalled()
    expect(audit.at(-1)?.decision).toBe('refused')
  })

  it('asks before reading, and reads once approved', async () => {
    const outcome = await broker.execute(
      call({ name: 'read_file', args: { path: path.join(shared, 'notes.txt') } })
    )
    expect(prompt).toHaveBeenCalledTimes(1)
    expect(outcome.ok).toBe(true)
    if (outcome.ok) expect((outcome.result as { text: string }).text).toContain('alpha')
  })

  it('does nothing when the user declines', async () => {
    prompt.mockResolvedValueOnce({ allow: false, grant: null })
    const target = path.join(shared, 'created.txt')
    const outcome = await broker.execute(
      call({ name: 'write_file', args: { path: target, content: 'x' } })
    )
    expect(outcome.ok).toBe(false)
    expect(fs.existsSync(target)).toBe(false)
  })

  /**
   * The important one. The model names the file by a route that is legal but
   * not canonical; the approval prompt shows the real path; the write has to
   * land on that same real path.
   */
  it('acts on the resolved path, not the string the model wrote', async () => {
    const detour = [shared, 'sub', '..', 'written.txt'].join(path.sep)
    const outcome = await broker.execute(
      call({ name: 'write_file', args: { path: detour, content: 'landed' } })
    )

    expect(outcome.ok).toBe(true)
    const shown = (prompt.mock.calls[0]?.[0] as ApprovalRequest).resolved
    expect(shown).toBe(path.join(shared, 'written.txt'))
    expect(fs.readFileSync(path.join(shared, 'written.txt'), 'utf8')).toBe('landed')
    expect(fs.existsSync(path.join(shared, 'sub', 'written.txt'))).toBe(false)
  })

  it('shows the user what a write will replace', async () => {
    // An existing file has to be read before it can be replaced.
    await broker.execute(call({ name: 'read_file', args: { path: path.join(shared, 'notes.txt') } }))
    await broker.execute(
      call({ name: 'write_file', args: { path: path.join(shared, 'notes.txt'), content: 'new' } })
    )
    const request = prompt.mock.calls[1]?.[0] as ApprovalRequest
    expect(request.preview.kind).toBe('overwrite')
    if (request.preview.kind === 'overwrite') {
      expect(request.preview.before).toContain('alpha')
      expect(request.preview.after).toBe('new')
    }
  })

  it('remembers a session grant and stops asking', async () => {
    prompt.mockResolvedValueOnce({
      allow: true,
      grant: {
        scope: 'session',
        pattern: `${shared.replace(/\\/g, '/')}/**`,
        applies: 'read'
      }
    })

    await broker.execute(call({ name: 'read_file', args: { path: path.join(shared, 'notes.txt') } }))
    expect(prompt).toHaveBeenCalledTimes(1)

    await broker.execute(call({ name: 'read_file', args: { path: path.join(shared, 'notes.txt') } }))
    expect(prompt).toHaveBeenCalledTimes(1)
    expect(broker.listSessionGrants()).toHaveLength(1)

    // A session grant is not written to the config file.
    expect(saveRule).not.toHaveBeenCalled()
  })

  it('does not let a read grant cover a write', async () => {
    prompt.mockResolvedValueOnce({
      allow: true,
      grant: { scope: 'session', pattern: `${shared.replace(/\\/g, '/')}/**`, applies: 'read' }
    })
    await broker.execute(call({ name: 'read_file', args: { path: path.join(shared, 'notes.txt') } }))

    await broker.execute(
      call({ name: 'write_file', args: { path: path.join(shared, 'notes.txt'), content: 'z' } })
    )
    expect(prompt).toHaveBeenCalledTimes(2)
  })

  it('persists an always grant through saveRule', async () => {
    prompt.mockResolvedValueOnce({
      allow: true,
      grant: { scope: 'always', pattern: `${shared.replace(/\\/g, '/')}/**`, applies: 'read' }
    })
    await broker.execute(call({ name: 'read_file', args: { path: path.join(shared, 'notes.txt') } }))
    expect(saveRule).toHaveBeenCalledTimes(1)
    expect(broker.listSessionGrants()).toHaveLength(0)
  })

  it('revoking session grants brings the prompts back', async () => {
    prompt.mockResolvedValueOnce({
      allow: true,
      grant: { scope: 'session', pattern: `${shared.replace(/\\/g, '/')}/**`, applies: 'read' }
    })
    await broker.execute(call({ name: 'read_file', args: { path: path.join(shared, 'notes.txt') } }))
    broker.clearSessionGrants()
    await broker.execute(call({ name: 'read_file', args: { path: path.join(shared, 'notes.txt') } }))
    expect(prompt).toHaveBeenCalledTimes(2)
  })

  it('runs without asking when the risk class is automatic', async () => {
    settings = { ...settings, modes: { read: 'auto', write: 'ask', destructive: 'ask', network: 'ask' } }
    const outcome = await broker.execute(
      call({ name: 'read_file', args: { path: path.join(shared, 'notes.txt') } })
    )
    expect(prompt).not.toHaveBeenCalled()
    expect(outcome.ok).toBe(true)
    if (outcome.ok) expect(outcome.asked).toBe(false)
    expect(audit.at(-1)?.asked).toBe(false)
  })

  it('refuses an ambiguous edit rather than guessing which one was meant', async () => {
    fs.writeFileSync(path.join(shared, 'dup.txt'), 'x\nsame\ny\nsame\n')
    await broker.execute(call({ name: 'read_file', args: { path: path.join(shared, 'dup.txt') } }))
    const outcome = await broker.execute(
      call({
        name: 'edit_file',
        args: { path: path.join(shared, 'dup.txt'), old_text: 'same', new_text: 'changed' }
      })
    )
    expect(outcome.ok).toBe(false)
    if (!outcome.ok) {
      expect(outcome.refused).toBe(false) // the action failed; policy allowed it
      expect(outcome.error).toContain('2 times')
    }
    expect(fs.readFileSync(path.join(shared, 'dup.txt'), 'utf8')).toBe('x\nsame\ny\nsame\n')
  })

  it('sends deletes to the recycle bin rather than unlinking', async () => {
    const target = path.join(shared, 'notes.txt')
    const outcome = await broker.execute(call({ name: 'delete_file', args: { path: target } }))
    expect(outcome.ok).toBe(true)
    expect(trash).toHaveBeenCalledWith(target)
  })

  it('refuses to delete a folder', async () => {
    const outcome = await broker.execute(
      call({ name: 'delete_file', args: { path: path.join(shared, 'sub') } })
    )
    expect(outcome.ok).toBe(false)
    expect(trash).not.toHaveBeenCalled()
  })

  it('reports a failure as a failure, not a refusal', async () => {
    const outcome = await broker.execute(
      call({ name: 'read_file', args: { path: path.join(shared, 'nope.txt') } })
    )
    expect(outcome.ok).toBe(false)
    if (!outcome.ok) expect(outcome.refused).toBe(false)
    expect(audit.at(-1)?.decision).toBe('failed')
  })
})
