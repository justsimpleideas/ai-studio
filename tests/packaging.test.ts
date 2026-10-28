import fs from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * The installed app ships the image engine's sources through an allowlist in
 * electron-builder.yml. A script the main process runs but the list leaves out works in
 * development and fails only in the installed app, so the two are checked against each other.
 */

const root = path.join(import.meta.dirname, '..')
const read = (file: string): string => fs.readFileSync(path.join(root, file), 'utf8')

describe('what the installer ships', () => {
  it('includes every engine script the main process runs, and only scripts that exist', () => {
    const main = read('src/main/index.ts')
    const used = [...main.matchAll(/path\.join\(imageEngineDir, 'scripts', '([\w-]+\.py)'\)/g)].map((m) => m[1]!)
    expect(used.length).toBeGreaterThanOrEqual(4)
    const shipped = [...read('electron-builder.yml').matchAll(/^\s+- scripts\/([\w-]+\.py)\s*$/gm)].map((m) => m[1]!)
    for (const script of used) expect(shipped, script).toContain(script)
    for (const script of shipped) expect(fs.existsSync(path.join(root, 'image-engine', 'scripts', script)), script).toBe(true)
  })

  // Without its own copy the packaged tray icon comes out blank: the app archive holds out/ only.
  it('ships the icon files the tray reads, next to the app', () => {
    const config = read('electron-builder.yml')
    for (const name of ['icon.ico', 'icon.png']) {
      expect(config, name).toMatch(new RegExp(`- from: resources/${name.replace('.', '\\.')}\\s+to: ${name.replace('.', '\\.')}`))
      expect(fs.existsSync(path.join(root, 'resources', name)), name).toBe(true)
    }
    expect(read('src/main/index.ts')).toMatch(/app\.isPackaged \? path\.join\(process\.resourcesPath, name\)/)
  })
})
