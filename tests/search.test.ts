import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { searchKeyName, SecretStore, type Cipher } from '../src/main/secrets.js'
import { ConfigStore } from '../src/main/store.js'
import {
  buildSearchRequest,
  checkSearxngUrl,
  DEFAULT_SEARCH,
  parseSearch,
  parseSearchResponse,
  SEARCH_PROVIDERS,
  searchEndpoint
} from '../src/shared/search.js'

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'search-test-'))
afterAll(() => fs.rmSync(dir, { recursive: true, force: true }))

/** A cipher that only reverses the text: enough to see that what goes in comes out, and that nothing is stored bare. */
const cipher: Cipher = {
  isEncryptionAvailable: () => true,
  encryptString: (text) => Buffer.from([...text].reverse().join('')),
  decryptString: (data) => [...data.toString()].reverse().join('')
}

describe('choosing a search service', () => {
  it('defaults to the hosted service with a free allowance, and keeps an unknown choice out', () => {
    expect(DEFAULT_SEARCH.provider).toBe('tavily')
    expect(parseSearch(undefined)).toEqual(DEFAULT_SEARCH)
    expect(parseSearch({ provider: 'google' })).toEqual(DEFAULT_SEARCH)
    expect(parseSearch({ provider: 'brave' })).toEqual({ provider: 'brave', searxngUrl: null })
    // The address outlives a switch to another service, so switching back needs no retyping.
    expect(parseSearch({ provider: 'brave', searxngUrl: 'http://x' })).toEqual({ provider: 'brave', searxngUrl: 'http://x/' })
  })

  it('keeps a usable SearXNG address, normalised, and drops a broken one', () => {
    expect(parseSearch({ provider: 'searxng', searxngUrl: 'http://localhost:8080' }).searxngUrl).toBe('http://localhost:8080/')
    expect(parseSearch({ provider: 'searxng', searxngUrl: 'ftp://x' }).searxngUrl).toBeNull()
  })

  it('accepts only a bare http(s) base address for SearXNG', () => {
    expect(checkSearxngUrl('http://localhost:8080').ok).toBe(true)
    expect(checkSearxngUrl('https://search.home.example').ok).toBe(true)
    for (const bad of ['', 'localhost:8080', 'ftp://x/', 'http://user:pw@host/', 'http://host/?q=1', 'http://host/#x']) {
      expect(checkSearxngUrl(bad).ok).toBe(false)
    }
  })

  it('names where a query goes, and refuses when a self-hosted address is missing', () => {
    expect(searchEndpoint({ provider: 'tavily', searxngUrl: null })).toEqual({ ok: true, host: 'api.tavily.com', label: 'Tavily (api.tavily.com)' })
    expect(searchEndpoint({ provider: 'searxng', searxngUrl: 'http://10.0.0.5:8080/' })).toEqual({
      ok: true,
      host: '10.0.0.5:8080',
      label: 'SearXNG (self-hosted) at 10.0.0.5:8080'
    })
    expect(searchEndpoint({ provider: 'searxng', searxngUrl: null }).ok).toBe(false)
  })

  it('knows which services need a key', () => {
    expect(SEARCH_PROVIDERS.map((p) => [p.id, p.needsKey])).toEqual([
      ['tavily', true],
      ['brave', true],
      ['searxng', false]
    ])
  })
})

describe('the request for one query', () => {
  it('carries the query and a count in each service’s shape, and never anything else', () => {
    const tavily = buildSearchRequest({ provider: 'tavily', searxngUrl: null }, 'parse dates', 'k1', 'UA')
    expect(tavily).toEqual({
      url: 'https://api.tavily.com/search',
      method: 'POST',
      headers: { Accept: 'application/json', 'Content-Type': 'application/json', Authorization: 'Bearer k1', 'User-Agent': 'UA' },
      body: JSON.stringify({ query: 'parse dates', max_results: 8 })
    })
    const brave = buildSearchRequest({ provider: 'brave', searxngUrl: null }, 'parse dates', 'k2', 'UA')
    expect(brave.url).toBe('https://api.search.brave.com/res/v1/web/search?q=parse+dates&count=8')
    expect(brave.method).toBe('GET')
    expect(brave.body).toBeUndefined()
    expect(brave.headers['X-Subscription-Token']).toBe('k2')
    const searx = buildSearchRequest({ provider: 'searxng', searxngUrl: 'http://localhost:8080/searx' }, 'parse dates', null, 'UA')
    expect(searx.url).toBe('http://localhost:8080/searx/search?q=parse+dates&format=json')
    expect(Object.keys(searx.headers)).toEqual(['Accept', 'User-Agent'])
  })

  it('reads each service’s reply into the same shape, and shrugs at a malformed one', () => {
    expect(parseSearchResponse('tavily', { results: [{ title: 't', url: 'u', content: 'c', score: 0.9 }] })).toEqual([
      { title: 't', url: 'u', snippet: 'c' }
    ])
    expect(parseSearchResponse('brave', { web: { results: [{ title: 't', url: 'u', description: 'd' }] } })).toEqual([
      { title: 't', url: 'u', snippet: 'd' }
    ])
    expect(parseSearchResponse('searxng', { results: [{ title: 't', url: 'u', content: 'c', engine: 'x' }] })).toEqual([
      { title: 't', url: 'u', snippet: 'c' }
    ])
    expect(parseSearchResponse('tavily', null)).toEqual([])
    expect(parseSearchResponse('tavily', { results: 'nope' })).toEqual([])
    expect(parseSearchResponse('searxng', { results: [null, 7] })).toEqual([{ title: '', url: '', snippet: '' }, { title: '', url: '', snippet: '' }])
  })

  it('never returns more than the count asked for', () => {
    const many = { results: Array.from({ length: 30 }, (_, i) => ({ title: String(i), url: '', content: '' })) }
    expect(parseSearchResponse('tavily', many)).toHaveLength(8)
  })
})

describe('search keys', () => {
  it('keeps one key per service, and reads a key from before services existed as Brave’s', async () => {
    const file = path.join(dir, 'secrets.json')
    const store = new SecretStore(file, cipher)
    await store.set('searchKey', 'old-brave-key')
    expect(store.searchKey('brave')).toBe('old-brave-key')
    expect(store.searchKey('tavily')).toBeNull()
    await store.set(searchKeyName('tavily'), 'tvly-1')
    expect(store.searchKey('tavily')).toBe('tvly-1')
    expect(store.searchKey('brave')).toBe('old-brave-key')
    await store.set(searchKeyName('brave'), 'new-brave')
    expect(store.searchKey('brave')).toBe('new-brave')
    expect(fs.readFileSync(file, 'utf8')).not.toContain('tvly-1')
  })

  it('is the default service for a config saved before the setting existed', () => {
    const file = path.join(dir, 'config.json')
    fs.writeFileSync(file, JSON.stringify({ model: { backend: 'auto' } }))
    expect(new ConfigStore(file).get().search).toEqual(DEFAULT_SEARCH)
  })
})
