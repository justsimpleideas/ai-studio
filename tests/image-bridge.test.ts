import { describe, expect, it } from 'vitest'
import {
  SseParser,
  engineImagePath,
  gifFileName,
  gifSize,
  isEngineId,
  isVideo,
  parseGifOptions,
  rewriteEvent,
  rewriteRecord,
  sniffImageType,
  studioImageUrl,
  studioMediaUrl,
  type ImageRecord
} from '@shared/image.js'
import { GpuArbiter, GpuBusyError, type GpuSide } from '../src/main/gpu.js'

const ID = '0123456789abcdef'

describe('GIF export options', () => {
  it('accepts whole numbers within the engine limits', () => {
    expect(parseGifOptions({ width: 480, fps: 12 })).toEqual({ width: 480, fps: 12 })
    expect(parseGifOptions({ width: 64, fps: 1, extra: 'dropped' })).toEqual({ width: 64, fps: 1 })
  })

  it('refuses anything else', () => {
    for (const bad of [null, 'x', {}, { width: 480 }, { width: 63, fps: 12 }, { width: 1281, fps: 12 },
      { width: 480, fps: 25 }, { width: 480.5, fps: 12 }, { width: '480', fps: 12 }, { width: 480, fps: 0 }]) {
      expect(() => parseGifOptions(bad), JSON.stringify(bad)).toThrow('Invalid GIF options.')
    }
  })

  it('sizes the GIF like the engine: same shape, never enlarged', () => {
    expect(gifSize(1280, 704, 480)).toEqual({ width: 480, height: 264 })
    expect(gifSize(832, 480, 1280)).toEqual({ width: 832, height: 480 })
    expect(gifSize(704, 1280, 320)).toEqual({ width: 320, height: 582 })
    expect(gifSize(4, 1, 2)).toEqual({ width: 2, height: 1 }) // 0.5 rounds up, as in the engine
  })

  it('names the file after the video', () => {
    expect(gifFileName('2026-10-26_1234.mp4')).toBe('2026-10-26_1234.gif')
    expect(gifFileName('clip.webp')).toBe('clip.gif')
    expect(gifFileName('')).toBe('video.gif')
  })
})

describe('studio-image URLs', () => {
  it('round-trips an image and a thumbnail', () => {
    expect(engineImagePath(studioImageUrl(ID, 'image'))).toBe(`/images/${ID}.png`)
    expect(engineImagePath(studioImageUrl(ID, 'thumb'))).toBe(`/images/${ID}.webp`)
  })

  it('refuses anything that is not exactly a gallery image', () => {
    for (const url of [
      `studio-image://gallery/../api/status`,
      `studio-image://gallery/${ID}.png?x=1`,
      `studio-image://gallery/${ID}.jpg`,
      `studio-image://other/${ID}.png`,
      `studio-image://gallery/${ID.toUpperCase()}.png`,
      `studio-image://gallery/${ID}0.png`,
      `http://127.0.0.1:5198/images/${ID}.png`,
      `studio-image://gallery/%2e%2e/${ID}.png`
    ]) {
      expect(engineImagePath(url), url).toBeNull()
    }
  })

  it('rewrites engine URLs in records and image events only', () => {
    const record = { id: ID, image_url: `/images/${ID}.png`, thumb_url: `/images/${ID}.webp` } as ImageRecord
    expect(rewriteRecord(record).image_url).toBe(`studio-image://gallery/${ID}.png`)
    const event = rewriteEvent({ type: 'image', job_id: ID, image: record })
    expect(event.type === 'image' && event.image.thumb_url).toBe(`studio-image://gallery/${ID}.webp`)
    const model = { type: 'model' as const, state: 'ready' as const, message: '' }
    expect(rewriteEvent(model)).toBe(model)
  })

  it('round-trips videos, last frames and start pictures', () => {
    expect(engineImagePath(studioMediaUrl(ID, 'video'))).toBe(`/videos/${ID}`)
    expect(engineImagePath(studioMediaUrl(ID, 'last'))).toBe(`/frames/${ID}-last.png`)
    expect(engineImagePath(studioMediaUrl(ID, 'input'))).toBe(`/inputs/${ID}.png`)
    for (const url of [
      `studio-image://video/${ID}.mp4`,
      `studio-image://video/../api/status`,
      `studio-image://frame/${ID}.png`,
      `studio-image://frame/${ID}-first.png`,
      `studio-image://input/${ID}.jpg`,
      `studio-image://input/../gallery.db`,
      `studio-image://inputs/${ID}.png`
    ]) {
      expect(engineImagePath(url), url).toBeNull()
    }
  })

  it('rewrites a video record to the video, its poster and its last frame', () => {
    const record = { id: ID, media: 'video', image_url: '/x', thumb_url: '/y', video_url: '/z' } as ImageRecord
    const out = rewriteRecord(record)
    expect(isVideo(out)).toBe(true)
    expect(out.video_url).toBe(`studio-image://video/${ID}`)
    expect(out.last_frame_url).toBe(`studio-image://frame/${ID}-last.png`)
    expect(out.image_url).toBe(out.thumb_url)
    expect(isVideo({})).toBe(false) // records from before video have no media field
  })

  it('validates engine ids', () => {
    expect(isEngineId(ID)).toBe(true)
    for (const bad of ['', 'xyz', `${ID}0`, ID.toUpperCase(), 42, null, '../../etc']) expect(isEngineId(bad)).toBe(false)
  })
})

describe('sniffImageType', () => {
  const bytes = (...values: number[]) => new Uint8Array(values)
  it('knows PNG, JPEG and WebP by their first bytes', () => {
    expect(sniffImageType(bytes(0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0))).toBe('image/png')
    expect(sniffImageType(bytes(0xff, 0xd8, 0xff, 0xe0))).toBe('image/jpeg')
    const webp = new TextEncoder().encode('RIFF\0\0\0\0WEBPVP8 ')
    expect(sniffImageType(webp)).toBe('image/webp')
  })

  it('refuses everything else, whatever the file was called', () => {
    for (const data of [new Uint8Array(), bytes(0x47, 0x49, 0x46, 0x38), new TextEncoder().encode('RIFF\0\0\0\0WAVE'),
                        new TextEncoder().encode('<svg xmlns=')]) {
      expect(sniffImageType(data)).toBeNull()
    }
  })
})

describe('SseParser', () => {
  it('splits events across arbitrary chunk boundaries and skips keep-alives', () => {
    const parser = new SseParser()
    const stream = ': keepalive\n\ndata: {"a":1}\n\ndata: {"b":2}\r\n\r\n: keepalive\n\ndata: {"c":3}\n\n'
    const out: string[] = []
    for (let i = 0; i < stream.length; i += 3) out.push(...parser.feed(stream.slice(i, i + 3)))
    expect(out.map((d) => JSON.parse(d))).toEqual([{ a: 1 }, { b: 2 }, { c: 3 }])
  })

  it('keeps an incomplete event until it is finished', () => {
    const parser = new SseParser()
    expect(parser.feed('data: {"x":')).toEqual([])
    expect(parser.feed('1}\n')).toEqual([])
    expect(parser.feed('\n')).toEqual(['{"x":1}'])
  })

  it('joins multi-line data fields', () => {
    expect(new SseParser().feed('data: line one\ndata: line two\n\n')).toEqual(['line one\nline two'])
  })
})

function side(label: string, init: { loaded?: boolean; busy?: string | null } = {}) {
  const s = {
    label,
    loaded: init.loaded ?? false,
    busy: init.busy ?? null,
    unloads: 0,
    isLoaded: () => s.loaded,
    busyReason: () => s.busy,
    unload: async () => {
      s.unloads++
      await new Promise((r) => setTimeout(r, 5))
      s.loaded = false
    }
  }
  return s satisfies GpuSide & Record<string, unknown>
}

describe('GpuArbiter', () => {
  it('does nothing when the other side is not loaded', async () => {
    const chat = side('chat model')
    const image = side('image model')
    const notes: string[] = []
    await new GpuArbiter({ chat, image }, (m) => notes.push(m)).claim('image')
    expect(chat.unloads).toBe(0)
    expect(notes).toEqual([])
  })

  it('unloads the idle other side and says so', async () => {
    const chat = side('chat model', { loaded: true })
    const image = side('image model')
    const notes: string[] = []
    await new GpuArbiter({ chat, image }, (m) => notes.push(m)).claim('image')
    expect(chat.unloads).toBe(1)
    expect(notes).toEqual(['Unloaded the chat model to make room for the image model.'])
  })

  it('refuses instead of interrupting a busy side', async () => {
    const chat = side('chat model')
    const image = side('image model', { loaded: true, busy: 'Image generation is in progress (1 job).' })
    const arbiter = new GpuArbiter({ chat, image }, () => undefined)
    await expect(arbiter.claim('chat')).rejects.toBeInstanceOf(GpuBusyError)
    await expect(arbiter.claim('chat')).rejects.toThrow(/in progress.*before loading the chat model/)
    expect(image.unloads).toBe(0)
  })

  it('serializes claims so a double click unloads only once', async () => {
    const chat = side('chat model', { loaded: true })
    const image = side('image model')
    const arbiter = new GpuArbiter({ chat, image }, () => undefined)
    await Promise.all([arbiter.claim('image'), arbiter.claim('image'), arbiter.claim('image')])
    expect(chat.unloads).toBe(1)
  })

  it('names what the other side held before it was unloaded', async () => {
    const chat = side('chat model')
    let holding: string | null = 'video model'
    const image: GpuSide = {
      get label() {
        return holding ?? 'image model'
      },
      isLoaded: () => holding !== null,
      busyReason: () => null,
      unload: async () => {
        holding = null
      }
    }
    const notes: string[] = []
    await new GpuArbiter({ chat, image }, (m) => notes.push(m)).claim('chat')
    expect(notes).toEqual(['Unloaded the video model to make room for the chat model.'])
  })

  it('keeps working after a refused claim', async () => {
    const chat = side('chat model', { loaded: true, busy: 'The chat model is answering.' })
    const image = side('image model')
    const arbiter = new GpuArbiter({ chat, image }, () => undefined)
    await expect(arbiter.claim('image')).rejects.toThrow()
    chat.busy = null
    await arbiter.claim('image')
    expect(chat.unloads).toBe(1)
  })
})
