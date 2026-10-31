import { createCipheriv, createDecipheriv, randomBytes } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import type { Cipher } from './secrets.js'

/**
 * Encryption for the files that hold conversations (chats/<id>.json, chats.json).
 *
 * Each file is sealed with AES-256-GCM under one data key, with a fresh 12-byte
 * nonce per write and the file's place in the store (workspace and chat) as
 * additional data, so a sealed chat moved into another chat's or another
 * workspace's file does not open. The data key is random, made on first use and
 * kept in a file of its own, itself encrypted with the operating system's store
 * (DPAPI on Windows, through Electron's safeStorage): only this user account on
 * this machine can open it.
 *
 * Plain files from before encryption still open, and are sealed on the next
 * write (and at startup, WorkspaceStore.sealPlainFiles).
 *
 * If the key file is there but cannot be opened (another account, a reinstalled
 * system), nothing is read and nothing is written: making a new key would leave
 * the existing chats unreadable for good, and writing them in plain text would
 * defeat the point. Without a system store at all, files stay plain, as before.
 *
 * On Windows, safeStorage is not DPAPI on each call: it encrypts under a key of
 * Chromium's own, which DPAPI protects and which is kept in the profile's
 * "Local State" file. That file is written lazily, and an app killed in the
 * first seconds of its first run was seen to leave none. A data key sealed under
 * a system key that never reached the disk is lost with the next start, and every
 * chat with it. So the data key is only made once the system key is known to be
 * saved (`systemKeySaved`); until then files are written as before, and sealed
 * on a later write.
 */

export const SEALED_FORMAT = 'local-ai-studio.sealed'

interface Envelope {
  format: typeof SEALED_FORMAT
  version: 1
  /** base64, 12 bytes */
  iv: string
  /** base64, 16 bytes */
  tag: string
  /** base64 ciphertext */
  data: string
}

function isEnvelope(value: unknown): value is Envelope {
  const v = value as Partial<Envelope> | null
  return (
    !!v &&
    typeof v === 'object' &&
    v.format === SEALED_FORMAT &&
    v.version === 1 &&
    typeof v.iv === 'string' &&
    typeof v.tag === 'string' &&
    typeof v.data === 'string'
  )
}

export class FileSealer {
  private key: Buffer | null = null

  private systemKeyKnownSaved = false

  constructor(
    private readonly keyFile: string,
    private readonly cipher: Cipher,
    /** Whether the system store's own key is on disk, so a key sealed under it survives a restart. */
    private readonly systemKeySaved: () => boolean = () => true
  ) {}

  /** Whether files written now are sealed: a key exists, or one can be made safely. */
  get sealing(): boolean {
    if (!this.enabled) return false
    if (this.key || fs.existsSync(this.keyFile)) return true
    if (!this.systemKeyKnownSaved) this.systemKeyKnownSaved = this.systemKeySaved()
    return this.systemKeyKnownSaved
  }

  /** Whether files are encrypted on this system at all. */
  get enabled(): boolean {
    return this.cipher.isEncryptionAvailable()
  }

  /** The data key; made and saved on first use. Throws when an existing key cannot be opened. */
  private dataKey(): Buffer {
    if (this.key) return this.key
    let saved: string | null = null
    try {
      saved = fs.readFileSync(this.keyFile, 'utf8')
    } catch {
      saved = null
    }
    if (saved !== null) {
      try {
        const sealed = (JSON.parse(saved) as { key?: unknown }).key
        if (typeof sealed !== 'string') throw new Error('no key')
        const key = Buffer.from(this.cipher.decryptString(Buffer.from(sealed, 'base64')), 'base64')
        if (key.length !== 32) throw new Error('bad key')
        this.key = key
        return key
      } catch {
        throw new Error(
          'The key for saved chats could not be opened on this user account, so chats are neither read nor saved. The key file is left as it is.'
        )
      }
    }
    const key = randomBytes(32)
    fs.mkdirSync(path.dirname(this.keyFile), { recursive: true })
    const temp = `${this.keyFile}.${process.pid}.tmp`
    fs.writeFileSync(temp, JSON.stringify({ version: 1, key: this.cipher.encryptString(key.toString('base64')).toString('base64') }), 'utf8')
    // Never over a key that appeared meanwhile: linking fails if the name is taken,
    // and then that key is the one to use.
    try {
      fs.linkSync(temp, this.keyFile)
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'EEXIST') return this.dataKey()
      throw err
    } finally {
      fs.rmSync(temp, { force: true })
    }
    this.key = key
    return key
  }

  /** The text to write for `json`: sealed when the system can, as it is otherwise. */
  seal(json: string, place: string): string {
    if (!this.sealing) return json
    const key = this.dataKey()
    const iv = randomBytes(12)
    const c = createCipheriv('aes-256-gcm', key, iv)
    c.setAAD(Buffer.from(place, 'utf8'))
    const data = Buffer.concat([c.update(json, 'utf8'), c.final()])
    const envelope: Envelope = {
      format: SEALED_FORMAT,
      version: 1,
      iv: iv.toString('base64'),
      tag: c.getAuthTag().toString('base64'),
      data: data.toString('base64')
    }
    return JSON.stringify(envelope)
  }

  /** The value in a file's text, and whether it was sealed. Throws on a sealed file that does not open. */
  open(text: string, place: string): { value: unknown; sealed: boolean } {
    const parsed: unknown = JSON.parse(text)
    if (!isEnvelope(parsed)) return { value: parsed, sealed: false }
    if (!this.enabled) throw new Error('This chat is encrypted, and this system offers no secure storage to open it.')
    const d = createDecipheriv('aes-256-gcm', this.dataKey(), Buffer.from(parsed.iv, 'base64'))
    d.setAAD(Buffer.from(place, 'utf8'))
    d.setAuthTag(Buffer.from(parsed.tag, 'base64'))
    const json = Buffer.concat([d.update(Buffer.from(parsed.data, 'base64')), d.final()]).toString('utf8')
    return { value: JSON.parse(json), sealed: true }
  }
}
