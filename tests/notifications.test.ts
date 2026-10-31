import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { ConfigStore } from '../src/main/store.js'
import { DEFAULT_CONFIG, DEFAULT_NOTIFICATIONS, parseNotifications, parseServer } from '../src/shared/config.js'
import { DEFAULT_GENERATION } from '../src/shared/generation.js'
import type { ImageJob } from '../src/shared/image.js'
import { CHAT_REPLY_NOTICE, JobWatcher, mediaNotice } from '../src/shared/notifications.js'

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'notifications-test-'))
afterAll(() => fs.rmSync(dir, { recursive: true, force: true }))

function storeWith(saved: unknown): ConfigStore {
  const file = path.join(dir, `${Math.random().toString(16).slice(2)}.json`)
  fs.writeFileSync(file, JSON.stringify(saved))
  return new ConfigStore(file)
}

const job = (patch: Partial<ImageJob>): ImageJob =>
  ({
    id: 'j1',
    kind: 'image',
    request: {},
    created: 0,
    status: 'running',
    stage: '',
    index: 0,
    count: 1,
    step: 0,
    steps: 8,
    images: [],
    error: null,
    started: 0,
    finished: null,
    ...patch
  }) as ImageJob

describe('what the tray says when something is done', () => {
  it('names what finished and nothing about it', () => {
    expect(mediaNotice(job({ status: 'done', images: ['a'] }), 'running')).toBe('Your image is ready. Click to open the studio.')
    expect(mediaNotice(job({ status: 'done', images: ['a', 'b', 'c', 'd'], count: 4 }), 'running')).toBe(
      '4 images are ready. Click to open the studio.'
    )
    expect(mediaNotice(job({ kind: 'video', status: 'done', images: ['v'] }), 'running')).toBe('Your video is ready. Click to open the studio.')
    // An engine without video support sends no kind: that is an image.
    const old = job({ status: 'done', images: ['a'] })
    delete old.kind
    expect(mediaNotice(old, 'running')).toMatch(/^Your image/)
    expect(CHAT_REPLY_NOTICE).not.toMatch(/"|:/)
  })

  it('stays quiet for anything but a job that has just finished', () => {
    expect(mediaNotice(job({ status: 'running' }), 'queued')).toBeNull()
    expect(mediaNotice(job({ status: 'failed' }), 'running')).toBeNull()
    expect(mediaNotice(job({ status: 'cancelled' }), 'running')).toBeNull()
    expect(mediaNotice(job({ status: 'done' }), 'done')).toBeNull()
    // Training tasks are not "an image or video".
    expect(mediaNotice(job({ kind: 'task', status: 'done' }), 'running')).toBeNull()
  })

  it('announces each job once, and nothing that was done before the engine connected', () => {
    const watcher = new JobWatcher()
    watcher.reset([job({ id: 'old', status: 'done', images: ['x'] })])
    expect(watcher.update(job({ id: 'old', status: 'done', images: ['x'] }))).toBeNull()
    expect(watcher.update(job({ id: 'new', status: 'queued' }))).toBeNull()
    expect(watcher.update(job({ id: 'new', status: 'running' }))).toBeNull()
    expect(watcher.update(job({ id: 'new', status: 'done', images: ['y'] }))).toMatch(/^Your image is ready/)
    expect(watcher.update(job({ id: 'new', status: 'done', images: ['y'] }))).toBeNull()
  })

  it('reads the setting defensively, both on by default', () => {
    expect(parseNotifications(undefined)).toEqual(DEFAULT_NOTIFICATIONS)
    expect(DEFAULT_NOTIFICATIONS).toEqual({ chatReply: true, mediaDone: true })
    expect(parseNotifications({ chatReply: false, mediaDone: 'no' })).toEqual({ chatReply: false, mediaDone: true })
    expect(storeWith({ notifications: { chatReply: false, mediaDone: false } }).get().notifications).toEqual({ chatReply: false, mediaDone: false })
    expect(storeWith({}).get().notifications).toEqual(DEFAULT_NOTIFICATIONS)
  })
})

describe('the chat model’s defaults', () => {
  it('start on llama-server with every expert on the CPU, 8 threads, 32k, 8-bit memory, thinking off', () => {
    const { model } = DEFAULT_CONFIG
    expect(model.engine).toBe('server')
    expect(model.server).toEqual({ path: null, expertsOnCpu: null, threads: 8 })
    expect(model.contextSize).toBe(32_768)
    expect(model.kvCache).toBe('q8_0')
    expect(model.generation).toEqual({ thinking: 'off', thinkingBudget: null, sampling: { kind: 'recommended' } })
    expect(DEFAULT_GENERATION.thinking).toBe('off')
  })

  it('apply to a fresh profile and to settings never saved', () => {
    expect(storeWith({}).get().model).toMatchObject({ engine: 'server', contextSize: 32_768, kvCache: 'q8_0', server: { threads: 8 } })
    expect(storeWith({ model: { server: { path: null } } }).get().model.server.threads).toBe(8)
  })

  it('leave what the user saved alone, a deliberate "Default" included', () => {
    const saved = storeWith({
      model: { engine: 'builtin', contextSize: null, kvCache: 'f16', server: { path: null, expertsOnCpu: 16, threads: null }, generation: { thinking: 'on' } }
    }).get().model
    expect(saved.engine).toBe('builtin')
    expect(saved.contextSize).toBeNull()
    expect(saved.kvCache).toBe('f16')
    expect(saved.server).toEqual({ path: null, expertsOnCpu: 16, threads: null })
    expect(saved.generation.thinking).toBe('on')
    expect(parseServer({ threads: null }).threads).toBeNull()
  })
})
