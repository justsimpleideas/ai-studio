import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterAll, beforeEach, describe, expect, it } from 'vitest'
import { ChatRecorder } from '../src/main/chat-recorder.js'
import { FileSealer, SEALED_FORMAT } from '../src/main/sealer.js'
import type { Cipher } from '../src/main/secrets.js'
import { WorkspaceStore } from '../src/main/workspaces.js'

/**
 * Stands in for the system store (DPAPI): reversible, and tied to an "account",
 * so a key sealed under one account does not open under another.
 */
function systemStore(account = 'you', available = true): Cipher {
  return {
    isEncryptionAvailable: () => available,
    encryptString: (text) => Buffer.from(`${account}:${text}`, 'utf8'),
    decryptString: (data) => {
      const text = data.toString('utf8')
      if (!text.startsWith(`${account}:`)) throw new Error('not this account')
      return text.slice(account.length + 1)
    }
  }
}

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'chat-encryption-'))
afterAll(() => fs.rmSync(root, { recursive: true, force: true }))

const SECRET = 'my bank password is hunter2'

let data: string
let folder: string
let keyFile: string

beforeEach(() => {
  const base = fs.mkdtempSync(path.join(root, 'case-'))
  data = path.join(base, 'workspaces')
  folder = path.join(base, 'project')
  keyFile = path.join(base, 'chat-key.json')
  fs.mkdirSync(folder)
})

function storeFor(cipher: Cipher = systemStore()): WorkspaceStore {
  return new WorkspaceStore(data, new FileSealer(keyFile, cipher))
}

function aChat(): ChatRecorder {
  const rec = ChatRecorder.fresh()
  rec.userMessage('p1', `Remember this: ${SECRET}`)
  rec.chunk('p1', `Noted: ${SECRET}`)
  rec.complete('p1', `Noted: ${SECRET}`)
  return rec
}

const chatFile = (wsId: string, chatId: string): string => path.join(data, wsId, 'chats', `${chatId}.json`)
const listFile = (wsId: string): string => path.join(data, wsId, 'chats.json')

describe('saved chats on disk', () => {
  it('hold neither the conversation nor its title in readable form', async () => {
    const store = storeFor()
    const ws = await store.open(folder)
    const rec = aChat()
    await store.saveChat(ws.id, rec.chat)
    for (const file of [chatFile(ws.id, rec.chat.meta.id), listFile(ws.id)]) {
      const text = fs.readFileSync(file, 'utf8')
      expect(text).not.toContain('hunter2')
      expect(text).not.toContain('Remember')
      expect(JSON.parse(text)).toMatchObject({ format: SEALED_FORMAT, version: 1 })
    }
    // No temp file left behind, plain or otherwise.
    expect(fs.readdirSync(path.join(data, ws.id, 'chats')).filter((f) => f.endsWith('.tmp'))).toEqual([])
  })

  it('open again after a restart, with the same key', async () => {
    const ws = await storeFor().open(folder)
    const rec = aChat()
    await storeFor().saveChat(ws.id, rec.chat)
    const reopened = storeFor()
    expect(reopened.listChats(ws.id).map((m) => m.title)).toEqual([rec.chat.meta.title])
    expect(reopened.loadChat(ws.id, rec.chat.meta.id)?.turns.map((t) => t.text)).toEqual(rec.chat.turns.map((t) => t.text))
    // One key, made once, and itself only readable through the system store.
    const key = JSON.parse(fs.readFileSync(keyFile, 'utf8')) as { key: string }
    expect(Buffer.from(key.key, 'base64').toString('utf8')).toMatch(/^you:/)
  })

  it('take chats saved before encryption, and seal them', async () => {
    const plain = new WorkspaceStore(data)
    const ws = await plain.open(folder)
    const rec = aChat()
    await plain.saveChat(ws.id, rec.chat)
    expect(fs.readFileSync(chatFile(ws.id, rec.chat.meta.id), 'utf8')).toContain('hunter2')

    const store = storeFor()
    expect(await store.sealPlainFiles()).toBe(2)
    expect(fs.readFileSync(chatFile(ws.id, rec.chat.meta.id), 'utf8')).not.toContain('hunter2')
    expect(fs.readFileSync(listFile(ws.id), 'utf8')).not.toContain('Remember')
    expect(storeFor().loadChat(ws.id, rec.chat.meta.id)?.turns).toHaveLength(2)
    expect(await storeFor().sealPlainFiles()).toBe(0)
  })

  it('refuse a sealed chat moved into another chat’s file, or changed by one byte', async () => {
    const store = storeFor()
    const ws = await store.open(folder)
    const a = aChat()
    const b = aChat()
    await store.saveChat(ws.id, a.chat)
    await store.saveChat(ws.id, b.chat)
    fs.copyFileSync(chatFile(ws.id, a.chat.meta.id), chatFile(ws.id, b.chat.meta.id))
    expect(storeFor().loadChat(ws.id, b.chat.meta.id)).toBeNull()

    const file = chatFile(ws.id, a.chat.meta.id)
    const envelope = JSON.parse(fs.readFileSync(file, 'utf8')) as { data: string }
    const bytes = Buffer.from(envelope.data, 'base64')
    bytes[0] = (bytes[0] ?? 0) ^ 1
    fs.writeFileSync(file, JSON.stringify({ ...envelope, data: bytes.toString('base64') }))
    expect(storeFor().loadChat(ws.id, a.chat.meta.id)).toBeNull()
  })

  it('are neither read nor overwritten when the key cannot be opened, and the key is kept', async () => {
    const ws = await storeFor().open(folder)
    const rec = aChat()
    await storeFor().saveChat(ws.id, rec.chat)
    const keyBefore = fs.readFileSync(keyFile, 'utf8')
    const before = fs.readFileSync(chatFile(ws.id, rec.chat.meta.id), 'utf8')

    const elsewhere = storeFor(systemStore('someone-else'))
    expect(elsewhere.loadChat(ws.id, rec.chat.meta.id)).toBeNull()
    expect(elsewhere.listChats(ws.id)).toEqual([])
    await expect(elsewhere.saveChat(ws.id, aChat().chat)).rejects.toThrow(/key for saved chats could not be opened/)
    expect(fs.readFileSync(keyFile, 'utf8')).toBe(keyBefore)
    expect(fs.readFileSync(chatFile(ws.id, rec.chat.meta.id), 'utf8')).toBe(before)
    // And the right account still reads everything.
    expect(storeFor().loadChat(ws.id, rec.chat.meta.id)).not.toBeNull()
  })

  it('stay as before on a system with no secure store', async () => {
    const store = storeFor(systemStore('you', false))
    const ws = await store.open(folder)
    const rec = aChat()
    await store.saveChat(ws.id, rec.chat)
    expect(JSON.parse(fs.readFileSync(chatFile(ws.id, rec.chat.meta.id), 'utf8'))).toMatchObject({ meta: { id: rec.chat.meta.id } })
    expect(fs.existsSync(keyFile)).toBe(false)
  })

  it('wait to make a key until the system’s own key is on disk, then seal what was written meanwhile', async () => {
    let saved = false
    const store = new WorkspaceStore(data, new FileSealer(keyFile, systemStore(), () => saved))
    const ws = await store.open(folder)
    const rec = aChat()
    await store.saveChat(ws.id, rec.chat)
    // A key made now could be lost with a crash before the system key is written.
    expect(fs.existsSync(keyFile)).toBe(false)
    expect(fs.readFileSync(chatFile(ws.id, rec.chat.meta.id), 'utf8')).toContain('hunter2')
    expect(await store.sealPlainFiles()).toBe(0)

    saved = true
    expect(await store.sealPlainFiles()).toBe(2)
    expect(fs.existsSync(keyFile)).toBe(true)
    expect(fs.readFileSync(chatFile(ws.id, rec.chat.meta.id), 'utf8')).not.toContain('hunter2')
    // Once the key exists it is used whatever the check says, so nothing goes back to plain text.
    saved = false
    await store.saveChat(ws.id, aChat().chat)
    expect(fs.readFileSync(listFile(ws.id), 'utf8')).not.toContain('Remember')
  })

  it('leave the workspace settings readable', async () => {
    const store = storeFor()
    const ws = await store.open(folder)
    await store.saveChat(ws.id, aChat().chat)
    expect(JSON.parse(fs.readFileSync(path.join(data, ws.id, 'workspace.json'), 'utf8'))).toMatchObject({ id: ws.id })
  })
})
