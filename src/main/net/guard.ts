import net from 'node:net'

/**
 * Checks on what the agent wants to send out, before anything is sent.
 *
 * Pure functions, so the rules can be tested without a network. The gate
 * (gate.ts) applies them to every request; none of them can be switched off by
 * the model, and none depend on the model describing its request honestly.
 */

export const LIMITS = {
  queryChars: 200,
  urlChars: 2048,
  /** Characters after "?" in a page address: enough for real links, too few to carry a file. */
  urlQueryChars: 512,
  pageBytes: 2 * 1024 * 1024,
  pageTextChars: 40_000,
  downloadBytes: 50 * 1024 * 1024,
  redirects: 5,
  timeoutMs: 20_000
}

/** File types a download may never be saved as: anything Windows or another OS would run. */
const RUNNABLE = new Set([
  'exe', 'msi', 'msix', 'msp', 'appx', 'appxbundle', 'bat', 'cmd', 'com', 'scr', 'pif', 'cpl',
  'ps1', 'psm1', 'psd1', 'vbs', 'vbe', 'js', 'jse', 'mjs', 'wsf', 'wsh', 'hta', 'msc', 'jar',
  'dll', 'sys', 'drv', 'ocx', 'lnk', 'url', 'reg', 'inf', 'sh', 'bash', 'zsh', 'command', 'app',
  'dmg', 'pkg', 'deb', 'rpm', 'apk', 'iso', 'img', 'vhd', 'vhdx', 'appimage', 'run', 'bin', 'elf',
  'py', 'pyw', 'pl', 'rb', 'php'
])

export function isRunnableName(fileName: string): boolean {
  const lower = fileName.toLowerCase().replace(/[. ]+$/, '')
  const dot = lower.lastIndexOf('.')
  return dot >= 0 && RUNNABLE.has(lower.slice(dot + 1))
}

export function checkQuery(query: string): string | null {
  const q = query.trim()
  if (!q) return 'The search query is empty.'
  if (/[\r\n\t\u0000-\u001f]/.test(query)) return 'A search query must be a single line of plain text.'
  if (q.length > LIMITS.queryChars) {
    return `The search query is ${q.length} characters long; the limit is ${LIMITS.queryChars}. Search in short, general terms.`
  }
  return null
}

const PRIVATE_V4: Array<[string, number]> = [
  ['0.0.0.0', 8],
  ['10.0.0.0', 8],
  ['100.64.0.0', 10],
  ['127.0.0.0', 8],
  ['169.254.0.0', 16],
  ['172.16.0.0', 12],
  ['192.0.0.0', 24],
  ['192.168.0.0', 16],
  ['198.18.0.0', 15],
  ['224.0.0.0', 4],
  ['240.0.0.0', 4]
]

function v4ToInt(ip: string): number {
  return ip.split('.').reduce((n, part) => (n << 8) + Number(part), 0) >>> 0
}

/**
 * True for any address that is not on the public internet: this machine, the
 * local network, link-local, carrier NAT, multicast and reserved ranges, in
 * IPv4 and IPv6 (including IPv4-mapped IPv6).
 */
export function isPrivateAddress(ip: string): boolean {
  const kind = net.isIP(ip)
  if (kind === 4) {
    const n = v4ToInt(ip)
    return PRIVATE_V4.some(([base, bits]) => {
      const mask = bits === 0 ? 0 : (~0 << (32 - bits)) >>> 0
      return (n & mask) === (v4ToInt(base) & mask)
    })
  }
  if (kind === 6) {
    const lower = ip.toLowerCase()
    const mapped = lower.match(/^::ffff:(\d+\.\d+\.\d+\.\d+)$/)
    if (mapped) return isPrivateAddress(mapped[1] as string)
    if (lower === '::' || lower === '::1') return true
    // fc00::/7 unique local, fe80::/10 link-local, ff00::/8 multicast.
    return /^(fc|fd|fe[89ab]|ff)/.test(lower)
  }
  return true
}

export type UrlCheck = { ok: true; url: URL } | { ok: false; error: string }

/** Everything about an address that can be judged without looking it up. */
export function checkUrl(raw: string): UrlCheck {
  if (raw.length > LIMITS.urlChars) return { ok: false, error: `That address is longer than ${LIMITS.urlChars} characters.` }
  let url: URL
  try {
    url = new URL(raw)
  } catch {
    return { ok: false, error: 'That is not a valid web address.' }
  }
  if (url.protocol !== 'https:' && url.protocol !== 'http:') return { ok: false, error: 'Only http and https addresses can be opened.' }
  if (url.username || url.password) return { ok: false, error: 'Addresses with a user name or password in them are not allowed.' }
  if (url.search.length - 1 > LIMITS.urlQueryChars) {
    return { ok: false, error: `The part of the address after "?" is too long (over ${LIMITS.urlQueryChars} characters).` }
  }
  const host = url.hostname.replace(/^\[|\]$/g, '').toLowerCase()
  if (net.isIP(host)) {
    if (isPrivateAddress(host)) return { ok: false, error: 'Addresses on this computer or the local network cannot be opened.' }
  } else if (!host.includes('.') || /\.(local|localhost|internal|intranet|lan|home|corp|arpa)$/.test(host) || host === 'localhost') {
    return { ok: false, error: 'Addresses on this computer or the local network cannot be opened.' }
  }
  return { ok: true, url }
}

/**
 * Notices when outgoing text repeats text from the workspace.
 *
 * Fed with everything the agent read from files (and wrote into them) in the
 * current conversation. A search query or address that repeats six words in a
 * row from that text, or any long identifier from it (a key, a hash, a unique
 * name), is refused. That stops file contents and code being carried out in a
 * request, deliberately or because a web page talked the model into it, while
 * ordinary searches in general terms pass.
 */
export class LeakDetector {
  static readonly RUN = 6
  static readonly LONG_TOKEN = 20
  static readonly MAX_CHARS = 4 * 1024 * 1024

  private runs = new Set<string>()
  private longTokens = new Set<string>()
  private chars = 0

  static words(text: string): string[] {
    return text.toLowerCase().match(/[\p{L}\p{N}_]+/gu) ?? []
  }

  add(text: string): void {
    if (this.chars >= LeakDetector.MAX_CHARS) return
    this.chars += text.length
    const words = LeakDetector.words(text)
    for (let i = 0; i + LeakDetector.RUN <= words.length; i++) {
      this.runs.add(words.slice(i, i + LeakDetector.RUN).join(' '))
    }
    for (const w of words) if (w.length >= LeakDetector.LONG_TOKEN) this.longTokens.add(w)
  }

  clear(): void {
    this.runs.clear()
    this.longTokens.clear()
    this.chars = 0
  }

  /** The repeated text, if `outgoing` carries any; null when it is clean. */
  find(outgoing: string): string | null {
    let decoded = outgoing
    try {
      decoded = decodeURIComponent(outgoing.replace(/\+/g, ' '))
    } catch {
      // Not URI-encoded; check it as it is.
    }
    const words = LeakDetector.words(decoded)
    for (const w of words) if (this.longTokens.has(w)) return w
    for (let i = 0; i + LeakDetector.RUN <= words.length; i++) {
      const run = words.slice(i, i + LeakDetector.RUN).join(' ')
      if (this.runs.has(run)) return run
    }
    return null
  }
}

const ENTITIES: Record<string, string> = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ' }

function decodeEntities(text: string): string {
  return text.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (whole, code: string) => {
    if (code[0] === '#') {
      const n = code[1]?.toLowerCase() === 'x' ? parseInt(code.slice(2), 16) : parseInt(code.slice(1), 10)
      return Number.isFinite(n) && n > 0 && n < 0x110000 ? String.fromCodePoint(n) : ''
    }
    return ENTITIES[code.toLowerCase()] ?? whole
  })
}

/** The readable text of a page: scripts, styles and markup removed, blocks kept apart. */
export function htmlToText(html: string): { title: string; text: string } {
  const title = decodeEntities((html.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1] ?? '').replace(/\s+/g, ' ').trim())
  const text = decodeEntities(
    html
      .replace(/<(script|style|noscript|template|svg|head)\b[\s\S]*?<\/\1>/gi, ' ')
      .replace(/<!--[\s\S]*?-->/g, ' ')
      .replace(/<(br|\/p|\/div|\/li|\/h[1-6]|\/tr|\/section|\/article|\/header|\/footer)\b[^>]*>/gi, '\n')
      .replace(/<[^>]+>/g, ' ')
  )
    .replace(/[ \t\f\v ]+/g, ' ')
    .replace(/ *\n */g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
  return { title, text }
}
