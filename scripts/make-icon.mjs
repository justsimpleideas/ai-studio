// Draws the app icon: a bevelled graphite tile holding a 4x4 cluster of square
// lamps, lit cyan to red, the same mark as the top of the studio rail.
//
//   node scripts/make-icon.mjs
//
// Writes resources/icon.png (512 px, also used by electron-builder for macOS and
// Linux) and resources/icon.ico (16 to 256 px, PNG-compressed entries). No
// dependencies: every size is drawn on its own grid so the lamps land on whole
// pixels and stay sharp at 16 px, and the PNG and ICO containers are written by
// hand.

import { mkdirSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { deflateSync } from 'node:zlib'

const OUT = resolve(import.meta.dirname, '..', 'resources')

// Same values as src/renderer/styles/tokens.css.
const TILE = [0x13, 0x13, 0x14]
const EDGE = [0x3a, 0x3a, 0x3f]
const CYAN = [0x2d, 0xe2, 0xff]
const SIGNAL = [0xff, 0x2f, 0x6e]

const mix = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t)

function draw(size) {
  const margin = size <= 24 ? 0 : Math.round(size * 0.04)
  const bevel = Math.max(2, Math.round(size * 0.2))
  const edge = Math.max(1, Math.round(size / 64))

  const inner = size - 2 * margin
  const lamp = Math.max(2, Math.round(inner * 0.135))
  const gap = Math.max(1, Math.round(inner * 0.05))
  const span = 4 * lamp + 3 * gap
  const start = Math.round((size - span) / 2)

  const lamps = []
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      lamps.push({
        x: start + col * (lamp + gap),
        y: start + row * (lamp + gap),
        color: mix(CYAN, SIGNAL, (row + col) / 6)
      })
    }
  }

  // Inside an octagon whose box is inset by `inset` and whose corners are cut by
  // lines at 45 degrees, moved inwards along their normal by the same amount.
  const inOctagon = (x, y, inset) => {
    const lo = margin + inset
    const hi = size - margin - inset
    if (x < lo || y < lo || x > hi || y > hi) return false
    const cut = 2 * margin + bevel + inset * Math.SQRT2
    return x + y >= cut && size - x + y >= cut && x + size - y >= cut && 2 * size - x - y >= cut
  }

  // Glow is drawn from a lamp's nearest edge, only where there is room for it.
  const glowRadius = size >= 48 ? lamp * 0.45 : 0

  const px = Buffer.alloc(size * size * 4)
  const SS = 4
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let r = 0
      let g = 0
      let b = 0
      let a = 0
      for (let sy = 0; sy < SS; sy++) {
        for (let sx = 0; sx < SS; sx++) {
          const fx = x + (sx + 0.5) / SS
          const fy = y + (sy + 0.5) / SS
          if (!inOctagon(fx, fy, 0)) continue
          let c = inOctagon(fx, fy, edge) ? TILE : EDGE
          const hit = lamps.find((l) => fx >= l.x && fx < l.x + lamp && fy >= l.y && fy < l.y + lamp)
          if (hit) {
            c = hit.color
          } else if (glowRadius) {
            let best = Infinity
            let near = null
            for (const l of lamps) {
              const dx = Math.max(l.x - fx, 0, fx - (l.x + lamp))
              const dy = Math.max(l.y - fy, 0, fy - (l.y + lamp))
              const d = Math.hypot(dx, dy)
              if (d < best) {
                best = d
                near = l
              }
            }
            const strength = 0.32 * Math.exp(-((best / glowRadius) ** 2))
            c = mix(c, near.color, strength)
          }
          r += c[0]
          g += c[1]
          b += c[2]
          a += 255
        }
      }
      const n = SS * SS
      const i = (y * size + x) * 4
      const cover = a / 255
      // Store straight (not premultiplied) colour.
      px[i] = cover ? Math.round(r / cover) : 0
      px[i + 1] = cover ? Math.round(g / cover) : 0
      px[i + 2] = cover ? Math.round(b / cover) : 0
      px[i + 3] = Math.round(a / n)
    }
  }
  return png(size, px)
}

// ------------------------------------------------------------ containers

const CRC_TABLE = Array.from({ length: 256 }, (_, n) => {
  let c = n
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
  return c >>> 0
})

function crc32(buf) {
  let c = 0xffffffff
  for (const byte of buf) c = CRC_TABLE[(c ^ byte) & 0xff] ^ (c >>> 8)
  return (c ^ 0xffffffff) >>> 0
}

function chunk(type, data) {
  const len = Buffer.alloc(4)
  len.writeUInt32BE(data.length)
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data])
  const crc = Buffer.alloc(4)
  crc.writeUInt32BE(crc32(body))
  return Buffer.concat([len, body, crc])
}

function png(size, rgba) {
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(size, 0)
  ihdr.writeUInt32BE(size, 4)
  ihdr[8] = 8 // bit depth
  ihdr[9] = 6 // RGBA
  const rows = Buffer.alloc(size * (size * 4 + 1))
  for (let y = 0; y < size; y++) {
    rows[y * (size * 4 + 1)] = 0 // filter: none
    rgba.copy(rows, y * (size * 4 + 1) + 1, y * size * 4, (y + 1) * size * 4)
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(rows, { level: 9 })),
    chunk('IEND', Buffer.alloc(0))
  ])
}

function ico(images) {
  const header = Buffer.alloc(6)
  header.writeUInt16LE(1, 2) // type: icon
  header.writeUInt16LE(images.length, 4)
  let offset = 6 + 16 * images.length
  const entries = images.map(({ size, data }) => {
    const e = Buffer.alloc(16)
    e[0] = size >= 256 ? 0 : size
    e[1] = size >= 256 ? 0 : size
    e.writeUInt16LE(1, 4) // colour planes
    e.writeUInt16LE(32, 6) // bits per pixel
    e.writeUInt32LE(data.length, 8)
    e.writeUInt32LE(offset, 12)
    offset += data.length
    return e
  })
  return Buffer.concat([header, ...entries, ...images.map((i) => i.data)])
}

mkdirSync(OUT, { recursive: true })
writeFileSync(join(OUT, 'icon.png'), draw(512))
const sizes = [16, 20, 24, 32, 40, 48, 64, 128, 256]
writeFileSync(join(OUT, 'icon.ico'), ico(sizes.map((size) => ({ size, data: draw(size) }))))
console.log(`Wrote ${join(OUT, 'icon.png')} and icon.ico (${sizes.join(', ')} px)`)
