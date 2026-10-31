/**
 * Where web_search queries go.
 *
 * One of a few search services, chosen in Settings: two hosted APIs with the
 * user's own key, or a SearXNG instance the user runs themselves. Whichever is
 * chosen, the gate's checks, the approval and the log are the same; only the
 * request's shape and the host differ, and both are described here, as pure
 * functions, so the gate stays one code path and each shape can be tested
 * without a network.
 */

export type SearchProviderId = 'tavily' | 'brave' | 'searxng'

export interface SearchProviderInfo {
  id: SearchProviderId
  name: string
  /** The fixed host of a hosted service; null for SearXNG, whose address the user gives. */
  host: string | null
  needsKey: boolean
  /** Where to get a key, or how to run the service. Shown as plain text, never opened by the studio. */
  about: string
}

export const SEARCH_PROVIDERS: readonly SearchProviderInfo[] = [
  {
    id: 'tavily',
    name: 'Tavily',
    host: 'api.tavily.com',
    needsKey: true,
    about: 'A hosted search API made for assistants, with a free monthly allowance. Sign up at tavily.com for a key.'
  },
  {
    id: 'brave',
    name: 'Brave Search',
    host: 'api.search.brave.com',
    needsKey: true,
    about: 'A hosted web search API, billed per request. Sign up at brave.com/search/api for a key.'
  },
  {
    id: 'searxng',
    name: 'SearXNG (self-hosted)',
    host: null,
    needsKey: false,
    about:
      'A search engine you run yourself, on this computer or another. No key and no account; its JSON output must be enabled (search.formats in settings.yml).'
  }
]

export function isSearchProviderId(value: unknown): value is SearchProviderId {
  return SEARCH_PROVIDERS.some((p) => p.id === value)
}

export function searchProvider(id: SearchProviderId): SearchProviderInfo {
  return SEARCH_PROVIDERS.find((p) => p.id === id) as SearchProviderInfo
}

export interface SearchPreferences {
  provider: SearchProviderId
  /** The SearXNG instance's address, e.g. http://localhost:8080. Only used when the provider is SearXNG. */
  searxngUrl: string | null
}

export const DEFAULT_SEARCH: SearchPreferences = { provider: 'tavily', searxngUrl: null }

export type SearxngUrlCheck = { ok: true; url: URL } | { ok: false; error: string }

/**
 * An address for a SearXNG instance: http(s), no credentials, nothing after
 * the path. A local address is allowed here and nowhere else in the gate: the
 * user typed it to name their own server, and only the query is ever sent to it.
 */
export function checkSearxngUrl(raw: unknown): SearxngUrlCheck {
  if (typeof raw !== 'string' || !raw.trim()) return { ok: false, error: 'Enter the address of your SearXNG instance.' }
  let url: URL
  try {
    url = new URL(raw.trim())
  } catch {
    return { ok: false, error: 'That is not a valid address.' }
  }
  if (url.protocol !== 'https:' && url.protocol !== 'http:') return { ok: false, error: 'The address must start with http:// or https://.' }
  if (url.username || url.password) return { ok: false, error: 'Leave user names and passwords out of the address.' }
  if (url.search || url.hash) return { ok: false, error: 'Give only the base address, without "?" or "#".' }
  return { ok: true, url }
}

export function parseSearch(value: unknown): SearchPreferences {
  const raw = (value && typeof value === 'object' ? value : {}) as Record<string, unknown>
  const checked = checkSearxngUrl(raw.searxngUrl)
  return {
    provider: isSearchProviderId(raw.provider) ? raw.provider : DEFAULT_SEARCH.provider,
    searxngUrl: checked.ok ? checked.url.href : null
  }
}

/** Where a query would go, as shown next to it in the approval and the log. */
export function searchEndpoint(prefs: SearchPreferences): { ok: true; host: string; label: string } | { ok: false; error: string } {
  const info = searchProvider(prefs.provider)
  if (info.host) return { ok: true, host: info.host, label: `${info.name} (${info.host})` }
  const checked = checkSearxngUrl(prefs.searxngUrl)
  if (!checked.ok) return { ok: false, error: 'No SearXNG address is set. Add one under Permissions → Internet.' }
  return { ok: true, host: checked.url.host, label: `${info.name} at ${checked.url.host}` }
}

export interface SearchResult {
  title: string
  url: string
  snippet: string
}

export interface SearchRequest {
  url: string
  method: 'GET' | 'POST'
  headers: Record<string, string>
  /** Only ever the query and a result count; never anything else. */
  body?: string
}

export const SEARCH_RESULT_COUNT = 8

/**
 * The request for one query, in the chosen service's shape. The query is the
 * only text that comes from the model; everything else is fixed here.
 */
export function buildSearchRequest(prefs: SearchPreferences, query: string, key: string | null, userAgent: string): SearchRequest {
  switch (prefs.provider) {
    case 'tavily':
      return {
        url: 'https://api.tavily.com/search',
        method: 'POST',
        headers: { Accept: 'application/json', 'Content-Type': 'application/json', Authorization: `Bearer ${key ?? ''}`, 'User-Agent': userAgent },
        body: JSON.stringify({ query, max_results: SEARCH_RESULT_COUNT })
      }
    case 'brave': {
      const url = new URL('https://api.search.brave.com/res/v1/web/search')
      url.searchParams.set('q', query)
      url.searchParams.set('count', String(SEARCH_RESULT_COUNT))
      return {
        url: url.href,
        method: 'GET',
        headers: { Accept: 'application/json', 'X-Subscription-Token': key ?? '', 'User-Agent': userAgent }
      }
    }
    case 'searxng': {
      const checked = checkSearxngUrl(prefs.searxngUrl)
      if (!checked.ok) throw new Error(checked.error)
      const url = new URL('search', checked.url.href.endsWith('/') ? checked.url.href : `${checked.url.href}/`)
      url.searchParams.set('q', query)
      url.searchParams.set('format', 'json')
      return { url: url.href, method: 'GET', headers: { Accept: 'application/json', 'User-Agent': userAgent } }
    }
  }
}

type Raw = Record<string, unknown>

function str(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

/** The results in a service's reply, reduced to title, address and snippet (still with markup; the gate strips it). */
export function parseSearchResponse(provider: SearchProviderId, data: unknown): SearchResult[] {
  const root = (data && typeof data === 'object' ? data : {}) as Raw
  const list: unknown =
    provider === 'brave' ? ((root.web as Raw | undefined)?.results ?? []) : (root.results ?? [])
  if (!Array.isArray(list)) return []
  return list.slice(0, SEARCH_RESULT_COUNT).map((item) => {
    const r = (item && typeof item === 'object' ? item : {}) as Raw
    return {
      title: str(r.title),
      url: str(r.url),
      snippet: str(provider === 'brave' ? r.description : r.content)
    }
  })
}
