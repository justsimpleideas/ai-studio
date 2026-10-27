import zlib from 'node:zlib'
import { describe, expect, it } from 'vitest'
import { withoutPngMetadata } from '../src/main/image/png-metadata.js'

/**
 * PNG files built chunk by chunk, so each test says exactly what is inside.
 * The pixel data is never decoded by either side, so a token IDAT is enough.
 */
const SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])

function chunk(type: string, data: Buffer | string): Buffer {
  const body = Buffer.isBuffer(data) ? data : Buffer.from(data, 'latin1')
  const head = Buffer.alloc(8)
  head.writeUInt32BE(body.length, 0)
  head.write(type, 4, 'latin1')
  const crc = Buffer.alloc(4)
  crc.writeUInt32BE(zlib.crc32(Buffer.concat([head.subarray(4), body])), 0)
  return Buffer.concat([head, body, crc])
}

const IHDR = chunk('IHDR', Buffer.from([0, 0, 0, 1, 0, 0, 0, 1, 8, 2, 0, 0, 0]))
const IDAT = chunk('IDAT', zlib.deflateSync(Buffer.from([0, 255, 0, 0])))
const IEND = chunk('IEND', Buffer.alloc(0))

const tEXt = (key: string, text: string) => chunk('tEXt', `${key}\0${text}`)
const zTXt = (key: string, text: string) =>
  chunk('zTXt', Buffer.concat([Buffer.from(`${key}\0\0`, 'latin1'), zlib.deflateSync(Buffer.from(text, 'latin1'))]))
const iTXt = (key: string, text: string, compress = false) =>
  chunk(
    'iTXt',
    Buffer.concat([
      Buffer.from(`${key}\0`, 'latin1'),
      Buffer.from([compress ? 1 : 0, 0]),
      Buffer.from('\0\0', 'latin1'),
      compress ? zlib.deflateSync(Buffer.from(text, 'utf8')) : Buffer.from(text, 'utf8')
    ])
  )

const png = (...chunks: Buffer[]) => Buffer.concat([SIGNATURE, IHDR, ...chunks, IDAT, IEND])

describe('exporting without metadata', () => {
  it('drops every text, time and EXIF chunk and keeps the picture byte for byte', () => {
    const withText = png(tEXt('local-ai-studio', '{}'), iTXt('Comment', 'hi'), zTXt('x', 'y'), chunk('tIME', Buffer.alloc(7)), chunk('eXIf', 'MM'))
    const out = withoutPngMetadata(withText)
    expect(out.equals(Buffer.concat([SIGNATURE, IHDR, IDAT, IEND]))).toBe(true)
  })

  it('drops anything hidden after the end of the image', () => {
    const out = withoutPngMetadata(Buffer.concat([png(), Buffer.from('trailing secret')]))
    expect(out.includes(Buffer.from('secret'))).toBe(false)
  })

  it('refuses a file that is not a PNG, or is cut short, rather than writing it out', () => {
    expect(() => withoutPngMetadata(Buffer.from('GIF89a'))).toThrow(/not a PNG/)
    const whole = png(tEXt('local-ai-studio', '{"prompt":"private"}'))
    expect(() => withoutPngMetadata(whole.subarray(0, whole.length - 20))).toThrow(/damaged/)
  })
})
