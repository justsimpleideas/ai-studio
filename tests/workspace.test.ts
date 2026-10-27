import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { Broker } from '../src/main/broker/index.js'
import { anchorToFolder } from '../src/main/broker/paths.js'
import { ChatRecorder } from '../src/main/chat-recorder.js'
import { Journal } from '../src/main/tools/fs-tools.js'
import { isSafeId, workspaceIdFor, WorkspaceStore } from '../src/main/workspaces.js'
import { composeSystemPrompt, composeSystemSections, DEFAULT_CONFIG } from '../src/shared/config.js'
import { DEFAULT_PERMISSION_SETTINGS } from '../src/shared/permissions.js'
import type { FileToolCall as ToolCall } from '../src/shared/tools.js'
import { chatTitle, workspaceSlug } from '../src/shared/workspace.js'

/**
 * Workspaces: a folder opened is the agent's working folder, and its chats,
 * rules and grants are kept for it alone.
 */

let sandbox: string

beforeEach(() => {
  sandbox = fs.realpathSync.native(fs.mkdtempSync(path.join(os.tmpdir(), 'las-ws-')))
})

afterEach(() => {
  fs.rmSync(sandbox, { recursive: true, force: true })
})

function folder(name: string): string {
  const dir = path.join(sandbox, name)
  fs.mkdirSync(dir, { recursive: true })
  return dir
}

describe('workspace ids', () => {
  it('are stable for a folder and differ between folders of the same name', () => {
    const a = folder('one/project')
    const b = folder('two/project')
    expect(workspaceIdFor(a)).toBe(workspaceIdFor(a))
    expect(workspaceIdFor(a)).not.toBe(workspaceIdFor(b))
    expect(workspaceIdFor(a)).toMatch(/^project-[0-9a-f]{12}$/)
  })

  it.runIf(process.platform === 'win32')('ignore case and separators on Windows', () => {
    const a = folder('Project')
    expect(workspaceIdFor(a.toUpperCase())).toBe(workspaceIdFor(a))
    expect(workspaceIdFor(a.replace(/\\/g, '/'))).toBe(workspaceIdFor(a))
  })

  it('keep only safe characters from the folder name', () => {
    expect(workspaceSlug('My App (v2)')).toBe('my-app-v2')
    expect(workspaceSlug('Προϊόν')).toBe('folder')
    expect(workspaceSlug('..')).toBe('folder')
  })

  it('refuse anything that could step out of the data folder', () => {
    for (const bad of ['../x', '..', 'a/b', 'a\\b', '', 'C:', '.hidden', 42, null]) {
      expect(isSafeId(bad)).toBe(false)
    }
    expect(isSafeId('project-0123456789ab')).toBe(true)
    expect(isSafeId('0b6c0a6e-9c1f-4a43-8d0e-0e7e6a8f9b11')).toBe(true)
  })
})

describe('WorkspaceStore', () => {
  let data: string
  let store: WorkspaceStore

  beforeEach(() => {
    data = path.join(sandbox, 'appdata')
    store = new WorkspaceStore(data)
  })

  it('opens the same workspace when the same folder is opened again', async () => {
    const dir = folder('alpha')
    const first = await store.open(dir)
    await store.update(first.id, (w) => {
      w.rules.push({ id: 'r', title: 'Tabs', body: 'Indent with tabs.', enabled: true })
    })
    const again = await new WorkspaceStore(data).open(dir)
    expect(again.id).toBe(first.id)
    expect(again.rules).toHaveLength(1)
    expect(again.createdAt).toBe(first.createdAt)
  })

  it('writes nothing into the opened folder', async () => {
    const dir = folder('untouched')
    const ws = await store.open(dir)
    await store.saveChat(ws.id, ChatRecorder.fresh().chat)
    expect(fs.readdirSync(dir)).toEqual([])
  })

  it('keeps each folder’s chats apart, newest first, and survives a restart', async () => {
    const a = await store.open(folder('a'))
    const b = await store.open(folder('b'))

    let clock = 0
    const now = () => new Date(Date.UTC(2026, 0, 1, 0, 0, clock++)).toISOString()
    const first = ChatRecorder.fresh(now)
    first.userMessage('p1', 'Rename the config file')
    const second = ChatRecorder.fresh(now)
    second.userMessage('p2', 'Add a test')
    const other = ChatRecorder.fresh(now)
    other.userMessage('p3', 'Something in b')

    await store.saveChat(a.id, first.chat)
    await store.saveChat(a.id, second.chat)
    await store.saveChat(b.id, other.chat)

    const reopened = new WorkspaceStore(data)
    expect(reopened.listChats(a.id).map((c) => c.title)).toEqual(['Add a test', 'Rename the config file'])
    expect(reopened.listChats(b.id).map((c) => c.title)).toEqual(['Something in b'])
    expect(reopened.loadChat(a.id, first.chat.meta.id)?.turns[0]?.text).toBe('Rename the config file')
    expect(reopened.loadChat(b.id, first.chat.meta.id)).toBeNull()
  })

  it('keeps the chat list right when saves land back to back', async () => {
    const ws = await store.open(folder('busy'))
    const chats = Array.from({ length: 5 }, (_, i) => {
      const rec = ChatRecorder.fresh()
      rec.userMessage(`p${i}`, `message ${i}`)
      return rec.chat
    })
    await Promise.all(chats.map((c) => store.saveChat(ws.id, c)))
    expect(new WorkspaceStore(data).listChats(ws.id)).toHaveLength(5)
  })

  it('deletes a chat and forgets a workspace without touching the folder', async () => {
    const dir = folder('gone')
    fs.writeFileSync(path.join(dir, 'keep.txt'), 'mine')
    const ws = await store.open(dir)
    const rec = ChatRecorder.fresh()
    rec.userMessage('p', 'hello')
    await store.saveChat(ws.id, rec.chat)

    expect(await store.deleteChat(ws.id, rec.chat.meta.id)).toEqual([])
    expect(store.loadChat(ws.id, rec.chat.meta.id)).toBeNull()

    await store.forget(ws.id)
    expect(store.get(ws.id)).toBeNull()
    expect(store.list()).toEqual([])
    expect(fs.readFileSync(path.join(dir, 'keep.txt'), 'utf8')).toBe('mine')
  })

  it('reports a folder that has gone missing, and still lists it', async () => {
    const dir = folder('moved')
    await store.open(dir)
    fs.rmSync(dir, { recursive: true })
    expect(store.list()).toEqual([expect.objectContaining({ name: 'moved', exists: false })])
  })

  it('treats a damaged chat file as absent', async () => {
    const ws = await store.open(folder('damaged'))
    const rec = ChatRecorder.fresh()
    rec.userMessage('p', 'hi')
    await store.saveChat(ws.id, rec.chat)
    fs.writeFileSync(path.join(data, ws.id, 'chats', `${rec.chat.meta.id}.json`), '{ not json')
    expect(new WorkspaceStore(data).loadChat(ws.id, rec.chat.meta.id)).toBeNull()
  })
})

describe('ChatRecorder', () => {
  it('records what the window shows, with the same ids', () => {
    const rec = ChatRecorder.fresh()
    expect(rec.worthSaving).toBe(false)
    rec.userMessage('p1', 'List the files\nand say which is largest')
    rec.chunk('p1', 'There are ')
    rec.chunk('p1', 'three.')
    rec.outcome({ ok: false, tool: 'read_file', resolved: null, error: 'x', refused: true })
    rec.complete('p1', 'three.')
    expect(rec.worthSaving).toBe(true)
    expect(rec.chat.meta.title).toBe('List the files')
    expect(rec.chat.turns).toEqual([
      { id: 'u-p1', role: 'user', text: 'List the files\nand say which is largest' },
      { id: 'p1', role: 'assistant', text: 'There are three.' }
    ])
    expect(rec.chat.outcomes).toHaveLength(1)
  })

  it('keeps a stopped reply as far as it got', () => {
    const rec = ChatRecorder.fresh()
    rec.userMessage('p1', 'Write a poem')
    rec.chunk('p1', 'Roses')
    rec.aborted('p1')
    expect(rec.chat.turns.at(-1)).toEqual({ id: 'p1', role: 'assistant', text: 'Roses\n\n[stopped]' })
  })

  it('drops an empty reply on error and keeps the error', () => {
    const rec = ChatRecorder.fresh()
    rec.userMessage('p1', 'Hi')
    expect(rec.owns('p1')).toBe(true)
    rec.error('p1', 'The model ran out of memory.')
    expect(rec.owns('p1')).toBe(false)
    expect(rec.chat.turns.map((t) => t.role)).toEqual(['user', 'system'])
  })

  it('ignores events for prompts it is not recording', () => {
    const rec = ChatRecorder.fresh()
    rec.chunk('elsewhere', 'x')
    rec.complete('elsewhere', 'x')
    expect(rec.chat.turns).toEqual([])
  })

  it('gives the chat the first message as its title, shortened', () => {
    expect(chatTitle('   ')).toBe('New chat')
    expect(chatTitle('a'.repeat(100))).toHaveLength(60)
    expect(chatTitle('a'.repeat(100)).endsWith('…')).toBe(true)
  })
})

describe('system prompt for a workspace', () => {
  const context = {
    workingFolder: 'C:\\Users\\You\\Projects\\site',
    extraFolders: ['C:\\Users\\You\\Shared\\assets'],
    projectRules: [
      { id: 'p1', title: 'Style', body: 'Use two-space indents.', enabled: true },
      { id: 'p2', title: 'Off', body: 'Never shown.', enabled: false }
    ]
  }

  it('says where the agent works, then global rules, then project rules', () => {
    const sections = composeSystemSections(DEFAULT_CONFIG.rules, context)
    expect(sections.map((s) => s.label)).toEqual([
      'Base instructions',
      'Working folder',
      'Say when you do not know',
      'One step at a time',
      'Project: Style'
    ])
    const prompt = composeSystemPrompt(DEFAULT_CONFIG.rules, context)
    expect(prompt).toContain('Your working folder is C:\\Users\\You\\Projects\\site.')
    expect(prompt).toContain('C:\\Users\\You\\Shared\\assets')
    expect(prompt).not.toContain('Never shown.')
    // Project rules come after the global ones.
    expect(prompt.indexOf('Use two-space indents.')).toBeGreaterThan(prompt.indexOf('Take one action at a time'))
  })

  it('is unchanged when there is no workspace', () => {
    expect(composeSystemSections(DEFAULT_CONFIG.rules)).toHaveLength(3)
  })
})

describe('relative paths', () => {
  const root = process.platform === 'win32' ? 'C:\\Work\\site' : '/work/site'

  it('are anchored to the working folder', () => {
    expect(anchorToFolder(root, 'notes.txt')).toBe(path.join(root, 'notes.txt'))
    expect(anchorToFolder(root, 'src/../README.md')).toBe(path.join(root, 'README.md'))
  })

  it('leave absolute paths and non-strings alone, and do nothing without a folder', () => {
    const abs = process.platform === 'win32' ? 'D:\\other\\x.txt' : '/other/x.txt'
    expect(anchorToFolder(root, abs)).toBe(abs)
    expect(anchorToFolder(root, 42)).toBe(42)
    expect(anchorToFolder(root, '')).toBe('')
    expect(anchorToFolder(null, 'notes.txt')).toBe('notes.txt')
  })

  it.runIf(process.platform === 'win32')('do not anchor the Windows shapes that only look relative', () => {
    expect(anchorToFolder(root, 'C:notes.txt')).toBe('C:notes.txt')
    expect(anchorToFolder(root, '\\notes.txt')).toBe('\\notes.txt')
  })
})

describe('Broker in a workspace', () => {
  it('runs a relative path inside the working folder and refuses one that climbs out', async () => {
    const work = folder('work')
    const outside = folder('outside')
    fs.writeFileSync(path.join(work, 'notes.txt'), 'inside')
    fs.writeFileSync(path.join(outside, 'secret.txt'), 'outside')

    const prompt = vi.fn(async () => ({ allow: true, grant: null }))
    const broker = new Broker({
      getSettings: () => ({
        ...DEFAULT_PERMISSION_SETTINGS,
        modes: { read: 'auto', write: 'ask', destructive: 'ask', network: 'ask' },
        roots: [{ id: 'working-folder', path: work, label: 'work', enabled: true, addedAt: '' }]
      }),
      saveRule: async () => undefined,
      prompt,
      trash: async () => undefined,
      audit: () => undefined,
      journal: new Journal(path.join(sandbox, 'backups')),
      workingFolder: work
    })
    const read = (p: string): ToolCall => ({ id: 'c', name: 'read_file', args: { path: p } })

    const inside = await broker.execute(read('notes.txt'))
    expect(inside.ok).toBe(true)
    expect(inside.resolved).toBe(path.join(work, 'notes.txt'))

    const climbed = await broker.execute(read('../outside/secret.txt'))
    expect(climbed.ok).toBe(false)
    expect(prompt).not.toHaveBeenCalled()
  })
})
