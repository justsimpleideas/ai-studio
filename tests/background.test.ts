import { describe, expect, it } from 'vitest'
import { DEFAULT_BACKGROUND, parseBackground, type ModelSlot } from '@shared/config.js'
import { BackgroundUnloader, type Timers, type UnloadTarget } from '../src/main/background.js'

function target(slot: ModelSlot, init: { loaded?: boolean; busy?: string | null; fail?: number } = {}) {
  const t = {
    slot,
    loaded: init.loaded ?? true,
    busy: init.busy ?? null,
    failuresLeft: init.fail ?? 0,
    unloads: 0,
    isLoaded: () => t.loaded,
    busyReason: () => t.busy,
    unload: async () => {
      if (t.failuresLeft > 0) {
        t.failuresLeft--
        throw new Error('engine restarting')
      }
      t.unloads++
      t.loaded = false
    }
  }
  return t satisfies UnloadTarget & Record<string, unknown>
}

/** Timers the test drives by hand: tick() runs one scheduled check. */
function manualTimers() {
  let run: (() => void) | null = null
  const timers: Timers = {
    setInterval: (fn) => ((run = fn), 'handle'),
    clearInterval: () => (run = null)
  }
  return {
    timers,
    get armed() {
      return run !== null
    },
    tick: async () => {
      run?.()
      await new Promise((r) => setTimeout(r, 0))
    }
  }
}

const settle = () => new Promise((r) => setTimeout(r, 0))

describe('background unloading', () => {
  it('unloads an idle model the user chose not to keep, and leaves the others', async () => {
    const chat = target('chat')
    const image = target('image')
    const clock = manualTimers()
    const unloaded: ModelSlot[] = []
    const u = new BackgroundUnloader([chat, image], () => ({ chat: false, image: true, video: true }), (s) => unloaded.push(s), 1000, clock.timers)
    u.start()
    await settle()
    expect(chat.unloads).toBe(1)
    expect(image.unloads).toBe(0)
    expect(unloaded).toEqual(['chat'])
  })

  it('waits for work to finish rather than interrupting it', async () => {
    const video = target('video', { busy: 'Video generation is in progress (1 job).' })
    const clock = manualTimers()
    const u = new BackgroundUnloader([video], () => ({ chat: true, image: true, video: false }), undefined, 1000, clock.timers)
    u.start()
    await settle()
    expect(video.unloads).toBe(0)
    await clock.tick()
    expect(video.unloads).toBe(0)
    video.busy = null // the queue emptied
    await clock.tick()
    expect(video.unloads).toBe(1)
  })

  it('stops when the window comes back, and keeps whatever is still loaded', async () => {
    const image = target('image', { busy: 'loading' })
    const clock = manualTimers()
    const u = new BackgroundUnloader([image], () => ({ chat: true, image: false, video: true }), undefined, 1000, clock.timers)
    u.start()
    await settle()
    u.stop()
    expect(clock.armed).toBe(false)
    image.busy = null
    await clock.tick()
    expect(image.unloads).toBe(0)
    expect(u.active).toBe(false)
  })

  it('reads the setting on every check, so a change applies while hidden', async () => {
    const chat = target('chat')
    const keep = { chat: true, image: true, video: true }
    const clock = manualTimers()
    const u = new BackgroundUnloader([chat], () => keep, undefined, 1000, clock.timers)
    u.start()
    await settle()
    expect(chat.unloads).toBe(0)
    keep.chat = false
    await clock.tick()
    expect(chat.unloads).toBe(1)
  })

  it('catches a model that queued work loaded again', async () => {
    const image = target('image')
    const clock = manualTimers()
    const u = new BackgroundUnloader([image], () => ({ chat: true, image: false, video: true }), undefined, 1000, clock.timers)
    u.start()
    await settle()
    image.loaded = true // an image job queued before the window closed loaded it again
    await clock.tick()
    expect(image.unloads).toBe(2)
  })

  it('tries again after a failed unload', async () => {
    const chat = target('chat', { fail: 1 })
    const clock = manualTimers()
    const u = new BackgroundUnloader([chat], () => ({ chat: false, image: true, video: true }), undefined, 1000, clock.timers)
    u.start()
    await settle()
    expect(chat.unloads).toBe(0)
    await clock.tick()
    expect(chat.unloads).toBe(1)
  })

  it('does nothing for models that are not loaded', async () => {
    const video = target('video', { loaded: false })
    const clock = manualTimers()
    new BackgroundUnloader([video], () => ({ chat: false, image: false, video: false }), undefined, 1000, clock.timers).start()
    await settle()
    expect(video.unloads).toBe(0)
  })
})

describe('background preferences on disk', () => {
  it('defaults to quitting, with every model kept when in the background', () => {
    expect(DEFAULT_BACKGROUND).toEqual({ closeBehavior: 'quit', keepLoaded: { chat: true, image: true, video: true } })
    expect(parseBackground(undefined)).toEqual(DEFAULT_BACKGROUND)
  })

  it('keeps valid choices and drops anything else', () => {
    expect(parseBackground({ closeBehavior: 'background', keepLoaded: { chat: false, image: true, video: false } })).toEqual({
      closeBehavior: 'background',
      keepLoaded: { chat: false, image: true, video: false }
    })
    expect(parseBackground({ closeBehavior: 'minimize', keepLoaded: { chat: 'no', video: 0 } })).toEqual(DEFAULT_BACKGROUND)
    expect(parseBackground('background')).toEqual(DEFAULT_BACKGROUND)
  })
})
