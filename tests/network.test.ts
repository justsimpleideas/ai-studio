import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { Broker } from '../src/main/broker/index.js'
import { WebGate } from '../src/main/net/gate.js'
import { checkUrl, htmlToText, isPrivateAddress, LeakDetector } from '../src/main/net/guard.js'
import { NetworkLog } from '../src/main/net/log.js'
import { Journal } from '../src/main/tools/journal.js'
import type { ApprovalRequest, ApprovalResponse } from '../src/shared/broker-types.js'
import { composeSystemPrompt } from '../src/shared/config.js'
import { SEARCH_HOST, toolsFor, type InternetLevel } from '../src/shared/network.js'
import { DEFAULT_PERMISSION_SETTINGS, DEFAULT_PROTECTED_PATHS, type ClassMode } from '../src/shared/permissions.js'
import type { FileToolCall, WebToolCall } from '../src/shared/tools.js'

/**
 * The internet gate, against a fake network: nothing in these tests leaves the
 * machine. What is checked is what would be sent, and that nothing is sent when
 * it should not be.
 */

let sandbox: string
let work: string

beforeEach(() => {
  sandbox = fs.realpathSync.native(fs.mkdtempSync(path.join(os.tmpdir(), 'las-net-')))
  work = path.join(sandbox, 'work')
  fs.mkdirSync(path.join(work, '.git'), { recursive: true })
  fs.writeFileSync(
    path.join(work, 'billing.ts'),
    'export function computeInvoiceTotalWithRegionalSurcharge(order) {\n  return order.lines.reduce((sum, line) => sum + line.price * line.quantity, 0)\n}\n'
  )
})

afterEach(() => {
  fs.rmSync(sandbox, { recursive: true, force: true })
})

interface Setup {
  level?: InternetLevel
  network?: ClassMode
  write?: ClassMode
  key?: string | null
  respond?: (url: string) => Response
  addresses?: Record<string, string[]>
  allow?: boolean
}

function setup(opts: Setup = {}) {
  const settings = {
    ...DEFAULT_PERMISSION_SETTINGS,
    modes: { read: 'auto' as ClassMode, write: opts.write ?? 'auto', destructive: 'ask' as ClassMode, network: opts.network ?? 'ask' },
    roots: [{ id: 'working-folder', path: work, label: 'work', enabled: true, addedAt: '' }],
    protectedPaths: DEFAULT_PROTECTED_PATHS
  }
  const journal = new Journal(path.join(sandbox, 'backups'), path.join(sandbox, 'journal.json'))
  const prompt = vi.fn(async (_r: ApprovalRequest): Promise<ApprovalResponse> => ({ allow: opts.allow ?? true, grant: null }))
  const broker = new Broker({
    getSettings: () => settings,
    saveRule: async () => undefined,
    prompt,
    trash: async () => undefined,
    audit: () => undefined,
    journal,
    workingFolder: work
  })
  const fetchMock = vi.fn(async (input: URL | string, _init?: RequestInit) =>
    (opts.respond ?? (() => new Response('<html><title>Hi</title><body><p>Hello</p></body></html>', { headers: { 'content-type': 'text/html' } })))(String(input))
  )
  const log = new NetworkLog(path.join(sandbox, 'network-log.jsonl'))
  const gate = new WebGate({
    level: () => opts.level ?? 'search',
    askFirst: () => settings.modes.network !== 'auto',
    searchKey: () => (opts.key === undefined ? 'test-key-1234' : opts.key),
    prompt,
    log,
    leaks: () => broker.contents,
    files: {
      checkNewFile: (raw) => broker.checkNewFile(raw),
      saveNewFile: (resolved, bytes, ctx) => broker.saveNewFile(resolved, bytes, { promptId: ctx.promptId })
    },
    fetch: fetchMock as unknown as typeof fetch,
    lookup: async (host) => opts.addresses?.[host] ?? ['93.184.215.14']
  })
  return { gate, broker, prompt, fetchMock, log, journal }
}

const search = (query: string): WebToolCall => ({ id: 'c', name: 'web_search', args: { query } })
const page = (url: string): WebToolCall => ({ id: 'c', name: 'fetch_page', args: { url } })
const download = (url: string, p: string): WebToolCall => ({ id: 'c', name: 'download_file', args: { url, path: p } })
const read = (p: string): FileToolCall => ({ id: 'r', name: 'read_file', args: { path: p } })

describe('internet levels', () => {
  it('give the model no web tools at all when offline, which is the default', () => {
    expect(toolsFor('offline')).toEqual(['list_directory', 'read_file', 'write_file', 'edit_file', 'delete_file'])
    expect(toolsFor('search')).toContain('fetch_page')
    expect(toolsFor('search')).not.toContain('download_file')
    expect(toolsFor('download')).toContain('download_file')
  })

  it('refuse every web request when offline, send nothing, and log the attempt', async () => {
    const { gate, fetchMock, prompt, log } = setup({ level: 'offline' })
    for (const call of [search('weather'), page('https://example.com/'), download('https://example.com/a.csv', 'a.csv')]) {
      const outcome = await gate.execute(call)
      expect(outcome.ok).toBe(false)
    }
    expect(fetchMock).not.toHaveBeenCalled()
    expect(prompt).not.toHaveBeenCalled()
    expect(log.recent().map((e) => e.outcome)).toEqual(['refused', 'refused', 'refused'])
  })

  it('refuse downloads at the search-only level', async () => {
    const { gate, fetchMock } = setup({ level: 'search' })
    expect((await gate.execute(download('https://example.com/a.csv', 'a.csv'))).ok).toBe(false)
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('describe the web tools in the prompt only when they are on', () => {
    const base = { workingFolder: 'C:\\Work', extraFolders: [], projectRules: [] }
    expect(composeSystemPrompt([], { ...base, internet: 'offline' })).not.toContain('web_search')
    expect(composeSystemPrompt([], { ...base, internet: 'search' })).toContain('web_search')
    expect(composeSystemPrompt([], { ...base, internet: 'search' })).not.toContain('download_file')
  })
})

describe('searching', () => {
  it('shows the exact query first, then sends only it, with the key, to the search service', async () => {
    const { gate, prompt, fetchMock, log } = setup({
      respond: () =>
        Response.json({ web: { results: [{ title: 'A <b>result</b>', url: 'https://a.example/', description: 'Some &amp; text' }] } })
    })
    const outcome = await gate.execute(search('how to parse dates in typescript'), { promptId: 'p1', chatId: 'c1' })
    expect(prompt).toHaveBeenCalledTimes(1)
    const request = prompt.mock.calls[0]?.[0] as ApprovalRequest
    expect(request.preview).toMatchObject({ kind: 'network', sends: 'how to parse dates in typescript' })
    const [url, init] = fetchMock.mock.calls[0] as [URL, RequestInit]
    expect(String(url)).toMatch(new RegExp(`^https://${SEARCH_HOST.replace(/\./g, '\\.')}/res/v1/web/search\\?q=how\\+to\\+parse`))
    expect((init.headers as Record<string, string>)['X-Subscription-Token']).toBe('test-key-1234')
    expect(init.method ?? 'GET').toBe('GET')
    expect(init.body).toBeUndefined()
    expect(outcome.ok).toBe(true)
    if (outcome.ok) expect(outcome.result).toEqual({ results: [{ title: 'A result', url: 'https://a.example/', snippet: 'Some & text' }] })
    expect(log.recent()[0]).toMatchObject({ outcome: 'sent', sent: 'how to parse dates in typescript', chatId: 'c1', promptId: 'p1' })
  })

  it('refuses queries that are too long or more than one line', async () => {
    const { gate, fetchMock } = setup()
    expect((await gate.execute(search('x'.repeat(201)))).ok).toBe(false)
    expect((await gate.execute(search('first line\nsecond line'))).ok).toBe(false)
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('sends nothing when the user declines, and logs that', async () => {
    const { gate, fetchMock, log } = setup({ allow: false })
    expect((await gate.execute(search('weather tomorrow'))).ok).toBe(false)
    expect(fetchMock).not.toHaveBeenCalled()
    expect(log.recent()[0]?.outcome).toBe('declined')
  })

  it('reports a missing key rather than sending without one', async () => {
    const { gate, fetchMock } = setup({ key: null })
    const outcome = await gate.execute(search('weather'))
    expect(outcome.ok).toBe(false)
    if (!outcome.ok) expect(outcome.error).toMatch(/No search key/)
    expect(fetchMock).not.toHaveBeenCalled()
  })
})

describe('keeping the workspace in', () => {
  it('refuses a query or address that repeats code the agent read', async () => {
    const { gate, broker, fetchMock, prompt } = setup({ respond: () => Response.json({ web: { results: [] } }) })
    await broker.execute(read('billing.ts'))

    // Six words in a row, as they appear in the file (punctuation does not matter).
    const copied = await gate.execute(search('why does return order.lines.reduce((sum, line) fail'))
    expect(copied.ok).toBe(false)
    if (!copied.ok) expect(copied.error).toMatch(/repeats text from files/)

    const identifier = await gate.execute(page('https://example.com/?q=computeInvoiceTotalWithRegionalSurcharge'))
    expect(identifier.ok).toBe(false)

    const encoded = await gate.execute(page(`https://example.com/?q=${encodeURIComponent('return order.lines.reduce((sum, line) => sum + line.price')}`))
    expect(encoded.ok).toBe(false)

    expect(fetchMock).not.toHaveBeenCalled()
    expect(prompt).not.toHaveBeenCalled()

    // General terms still go through.
    expect((await gate.execute(search('typescript reduce sum of array'))).ok).toBe(true)
  })

  it('forgets what was read when the conversation changes', async () => {
    const { gate, broker } = setup()
    await broker.execute(read('billing.ts'))
    broker.resetConversation()
    expect((await gate.execute(page('https://example.com/?q=computeInvoiceTotalWithRegionalSurcharge'))).ok).toBe(true)
  })
})

describe('addresses', () => {
  it('refuses anything that is not a public http(s) address, before sending', async () => {
    const { gate, fetchMock } = setup()
    for (const url of [
      'http://localhost:5198/api/status',
      'http://127.0.0.1/',
      'http://192.168.1.10/admin',
      'http://[::1]/',
      'http://intranet/',
      'http://printer.local/',
      'https://user:secret@example.com/',
      'ftp://example.com/file',
      'file:///C:/Windows/win.ini',
      `https://example.com/?data=${'a'.repeat(600)}`
    ]) {
      expect((await gate.execute(page(url))).ok, url).toBe(false)
    }
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('refuses a public name that resolves to the local network', async () => {
    const { gate, fetchMock } = setup({ addresses: { 'sneaky.example': ['10.0.0.5'] } })
    expect((await gate.execute(page('https://sneaky.example/'))).ok).toBe(false)
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('refuses a redirect to a local address', async () => {
    const { gate } = setup({
      respond: () => new Response(null, { status: 302, headers: { location: 'http://127.0.0.1:5198/api/status' } })
    })
    const outcome = await gate.execute(page('https://example.com/go'))
    expect(outcome.ok).toBe(false)
    if (!outcome.ok) expect(outcome.error).toMatch(/redirected/)
  })

  it('knows the private ranges', () => {
    for (const ip of ['10.1.2.3', '172.16.0.1', '172.31.255.255', '192.168.0.1', '127.0.0.1', '169.254.1.1', '100.64.0.1', '::1', 'fd00::1', 'fe80::1', '::ffff:192.168.1.1']) {
      expect(isPrivateAddress(ip), ip).toBe(true)
    }
    for (const ip of ['93.184.215.14', '172.32.0.1', '8.8.8.8', '2606:4700::1111']) {
      expect(isPrivateAddress(ip), ip).toBe(false)
    }
    expect(checkUrl('https://example.com/path?x=1').ok).toBe(true)
  })
})

describe('reading pages', () => {
  it('returns readable text, marked as coming from the web', async () => {
    const { gate } = setup({
      respond: () =>
        new Response('<html><head><title>T &amp; C</title><script>steal()</script></head><body><h1>Title</h1><p>One</p><style>p{}</style><p>Two &lt;3</p></body></html>', {
          headers: { 'content-type': 'text/html; charset=utf-8' }
        })
    })
    const outcome = await gate.execute(page('https://example.com/'))
    expect(outcome.ok).toBe(true)
    if (outcome.ok) {
      const result = outcome.result as { title: string; text: string; note: string }
      expect(result.title).toBe('T & C')
      expect(result.text).toContain('Two <3')
      expect(result.text).not.toContain('steal')
      expect(result.note).toMatch(/not from the user/)
    }
  })

  it('refuses something that is not a page', async () => {
    const { gate } = setup({ respond: () => new Response(new Uint8Array([1, 2, 3]), { headers: { 'content-type': 'application/zip' } }) })
    expect((await gate.execute(page('https://example.com/a.zip'))).ok).toBe(false)
  })

  it('turns markup into text', () => {
    expect(htmlToText('<p>a&nbsp;b</p><br><div>&#x41;&#66;</div>').text).toBe('a b\n\nAB')
  })
})

describe('downloading', () => {
  const csv = () => new Response('a,b\n1,2\n', { headers: { 'content-type': 'text/csv' } })

  it('saves a new file through the journal, so it can be reverted', async () => {
    const { gate, journal } = setup({ level: 'download', respond: csv })
    const outcome = await gate.execute(download('https://example.com/data.csv', 'data/raw.csv'), { promptId: 'p9' })
    expect(outcome.ok).toBe(true)
    expect(fs.readFileSync(path.join(work, 'data', 'raw.csv'), 'utf8')).toBe('a,b\n1,2\n')
    expect((await journal.revert('p9')).done).toBe(true)
    expect(fs.existsSync(path.join(work, 'data', 'raw.csv'))).toBe(false)
  })

  it('never overwrites, never writes into a protected path, never saves a program', async () => {
    const { gate, fetchMock } = setup({ level: 'download', respond: csv })
    expect((await gate.execute(download('https://example.com/x', 'billing.ts'))).ok).toBe(false)
    expect((await gate.execute(download('https://example.com/x', '.git/hooks/pre-commit.csv'))).ok).toBe(false)
    expect((await gate.execute(download('https://example.com/setup.exe', 'setup.exe'))).ok).toBe(false)
    expect((await gate.execute(download('https://example.com/x', 'run.ps1'))).ok).toBe(false)
    expect((await gate.execute(download('https://example.com/x', '../outside.csv'))).ok).toBe(false)
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('stops at the size limit', async () => {
    const { gate } = setup({
      level: 'download',
      respond: () => new Response('x', { headers: { 'content-length': String(60 * 1024 * 1024) } })
    })
    expect((await gate.execute(download('https://example.com/big.bin', 'big.dat'))).ok).toBe(false)
    expect(fs.existsSync(path.join(work, 'big.dat'))).toBe(false)
  })

  it('asks when writing would be asked about, even with internet requests on automatic', async () => {
    const { gate, prompt } = setup({ level: 'download', network: 'auto', write: 'ask', respond: csv })
    await gate.execute(download('https://example.com/data.csv', 'new.csv'))
    expect(prompt).toHaveBeenCalledTimes(1)
    expect((prompt.mock.calls[0]?.[0] as ApprovalRequest).preview).toMatchObject({ kind: 'network', saveTo: path.join(work, 'new.csv') })
  })
})

describe('the leak detector', () => {
  it('ignores short overlaps and catches six words in a row', () => {
    const d = new LeakDetector()
    d.add('The quick brown fox jumps over the lazy dog near the riverbank.')
    expect(d.find('quick brown fox')).toBeNull()
    expect(d.find('the quick brown fox jumps over')).toBe('the quick brown fox jumps over')
  })
})
