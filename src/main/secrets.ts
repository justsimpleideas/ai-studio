import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'

import type { SearchProviderId } from '@shared/search.js'

/**
 * Secrets the app holds for the user: for now, the web search keys, one per
 * search service, so switching services and back does not lose a key.
 *
 * Kept apart from config.json, which the window can read, and encrypted with
 * the operating system's own store (DPAPI on Windows, the keychain on macOS)
 * through Electron's safeStorage. If that store is not available the key is not
 * kept at all, rather than written down in plain text. The window only ever
 * learns whether a key is set, never the key.
 */
export interface Cipher {
  isEncryptionAvailable(): boolean
  encryptString(text: string): Buffer
  decryptString(data: Buffer): string
}

/** `searchKey` on its own is the name from before services could be chosen, when the only one was Brave. */
type Name = 'searchKey' | `searchKey:${SearchProviderId}`

export function searchKeyName(provider: SearchProviderId): Name {
  return `searchKey:${provider}`
}

export class SecretStore {
  private cache = new Map<Name, string | null>()

  constructor(
    private readonly file: string,
    private readonly cipher: Cipher
  ) {}

  private readAll(): Partial<Record<Name, string>> {
    try {
      return JSON.parse(fs.readFileSync(this.file, 'utf8')) as Partial<Record<Name, string>>
    } catch {
      return {}
    }
  }

  get(name: Name): string | null {
    if (this.cache.has(name)) return this.cache.get(name) ?? null
    const sealed = this.readAll()[name]
    let value: string | null = null
    if (sealed && this.cipher.isEncryptionAvailable()) {
      try {
        value = this.cipher.decryptString(Buffer.from(sealed, 'base64'))
      } catch {
        value = null
      }
    }
    this.cache.set(name, value)
    return value
  }

  has(name: Name): boolean {
    return this.get(name) !== null
  }

  /** The key for a search service; a key saved before services could be chosen counts as Brave's. */
  searchKey(provider: SearchProviderId): string | null {
    const own = this.get(searchKeyName(provider))
    if (own !== null || provider !== 'brave') return own
    return this.get('searchKey')
  }

  async set(name: Name, value: string | null): Promise<void> {
    if (value !== null && !this.cipher.isEncryptionAvailable()) {
      throw new Error('This system offers no secure storage, so the key cannot be saved.')
    }
    const all = this.readAll()
    if (value === null) delete all[name]
    else all[name] = this.cipher.encryptString(value).toString('base64')
    await fsp.mkdir(path.dirname(this.file), { recursive: true })
    const temp = `${this.file}.${process.pid}.tmp`
    await fsp.writeFile(temp, JSON.stringify(all), 'utf8')
    await fsp.rename(temp, this.file)
    this.cache.set(name, value)
  }
}
