import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { DataLocationStore, LOCATION_FILE, defaultLocations, locationProblem } from '../src/main/locations.js'

/**
 * Moving the studio's own folders.
 *
 * The promise the setting makes is narrow and worth testing as such: a new
 * folder is where new data goes, and the folder left behind is not touched at
 * all. These tests use real folders, because "nothing was moved" is a claim
 * about the disk.
 */

let appData: string

beforeEach(() => {
  appData = fs.realpathSync.native(fs.mkdtempSync(path.join(os.tmpdir(), 'las-loc-')))
})

afterEach(() => {
  fs.rmSync(appData, { recursive: true, force: true })
})

const pointerFile = (): string => path.join(appData, LOCATION_FILE)

describe('with nothing chosen', () => {
  it('keeps both folders under the application data folder', () => {
    const store = new DataLocationStore(appData)
    expect(store.get('userData')).toBe(path.join(appData, 'userData'))
    expect(store.get('outputs')).toBe(path.join(appData, 'image', 'outputs'))
    expect(store.choice('userData')).toBeNull()
    const view = store.view()
    expect(view.userData.isDefault).toBe(true)
    expect(view.outputs.isDefault).toBe(true)
    expect(view.restartNeeded).toBe(false)
  })

  it('writes no file until something is chosen', () => {
    new DataLocationStore(appData).view()
    expect(fs.existsSync(pointerFile())).toBe(false)
  })
})

describe('choosing a folder', () => {
  it('leaves everything in the folder being left behind', () => {
    const store = new DataLocationStore(appData)
    const old = store.get('userData')
    fs.mkdirSync(path.join(old, 'workspaces'), { recursive: true })
    fs.writeFileSync(path.join(old, 'config.json'), '{"theme":"dark"}')
    const chosen = path.join(appData, 'elsewhere', 'chats')

    const view = store.set('userData', chosen)

    expect(view.userData.path).toBe(chosen)
    expect(view.userData.isDefault).toBe(false)
    expect(fs.readFileSync(path.join(old, 'config.json'), 'utf8')).toBe('{"theme":"dark"}')
    expect(fs.existsSync(path.join(old, 'workspaces'))).toBe(true)
    // Made ready to write to, and empty: a new folder starts with nothing in it.
    expect(fs.readdirSync(chosen)).toEqual([])
  })

  it('says a restart is needed for the chats folder, and the next run does not', () => {
    const store = new DataLocationStore(appData)
    expect(store.set('userData', path.join(appData, 'moved')).restartNeeded).toBe(true)
    // A new process reads the file and opens that folder, so there is nothing left to switch.
    const next = new DataLocationStore(appData)
    expect(next.get('userData')).toBe(path.join(appData, 'moved'))
    expect(next.view().restartNeeded).toBe(false)
  })

  it('needs no restart for the gallery, which the engine reopens', () => {
    const store = new DataLocationStore(appData)
    expect(store.set('outputs', path.join(appData, 'gallery')).restartNeeded).toBe(false)
  })

  it('keeps only the folders that were chosen in the file', () => {
    const store = new DataLocationStore(appData)
    store.set('outputs', path.join(appData, 'gallery'))
    expect(JSON.parse(fs.readFileSync(pointerFile(), 'utf8'))).toEqual({ outputs: path.join(appData, 'gallery') })
  })

  it('counts choosing the default folder as choosing nothing', () => {
    const store = new DataLocationStore(appData)
    const view = store.set('outputs', defaultLocations(appData).outputs)
    expect(view.outputs.isDefault).toBe(true)
    expect(JSON.parse(fs.readFileSync(pointerFile(), 'utf8'))).toEqual({})
  })
})

describe('going back to the default', () => {
  it('brings the old folder, and what is in it, back into use', () => {
    const store = new DataLocationStore(appData)
    const home = store.get('outputs')
    fs.mkdirSync(home, { recursive: true })
    fs.writeFileSync(path.join(home, 'gallery.db'), 'pretend')
    store.set('outputs', path.join(appData, 'gallery'))

    const view = store.set('outputs', null)

    expect(view.outputs.path).toBe(home)
    expect(view.outputs.isDefault).toBe(true)
    expect(fs.readFileSync(path.join(home, 'gallery.db'), 'utf8')).toBe('pretend')
  })
})

describe('folders that cannot be used', () => {
  it('refuses one inside the other, either way round', () => {
    const store = new DataLocationStore(appData)
    const chats = path.join(appData, 'chats')
    store.set('userData', chats)
    expect(() => store.set('outputs', path.join(chats, 'gallery'))).toThrow(/cannot be the same, or one inside the other/)
    expect(() => store.set('outputs', chats)).toThrow(/cannot be the same, or one inside the other/)
    expect(() => store.set('userData', path.join(appData, 'gallery'))).not.toThrow()
  })

  it('refuses a relative path', () => {
    const store = new DataLocationStore(appData)
    expect(() => store.set('userData', 'gallery')).toThrow(/folder on a drive/)
  })

  it('refuses the program folder, which an update empties', () => {
    const program = path.join(appData, 'program')
    const store = new DataLocationStore(appData, [program])
    expect(() => store.set('outputs', path.join(program, 'resources', 'gallery'))).toThrow(/inside the program itself/)
  })

  it('reports the problem without changing anything', () => {
    const store = new DataLocationStore(appData)
    const before = store.view()
    expect(() => store.set('userData', 'nowhere')).toThrow()
    expect(store.view()).toEqual(before)
    expect(fs.existsSync(pointerFile())).toBe(false)
  })
})

describe('the file itself', () => {
  it('falls back to the defaults when it cannot be read or makes no sense', () => {
    for (const body of ['not json at all', '{"userData":42}', '{"userData":"relative/path"}', '[]']) {
      fs.writeFileSync(pointerFile(), body)
      const store = new DataLocationStore(appData)
      expect(store.get('userData')).toBe(path.join(appData, 'userData'))
      expect(store.view().restartNeeded).toBe(false)
    }
  })

  it('is read as a plain path, so a studio told where to look obeys it', () => {
    fs.writeFileSync(pointerFile(), JSON.stringify({ userData: path.join(appData, 'from-file') }))
    expect(new DataLocationStore(appData).get('userData')).toBe(path.join(appData, 'from-file'))
  })
})

describe('locationProblem', () => {
  it('names the folder the user is choosing', () => {
    const gallery = path.join(appData, 'gallery')
    const others = { other: gallery, programFolders: [] }
    expect(locationProblem('userData', gallery, others)).toMatch(/chats folder and the gallery folder/)
    expect(locationProblem('outputs', path.join(appData, 'chats'), { other: path.join(appData, 'chats'), programFolders: [] }))
      .toMatch(/gallery folder and the chats folder/)
    expect(locationProblem('userData', path.join(appData, 'chats'), others)).toBeNull()
  })

  it('does not take a folder for its neighbour on a name prefix', () => {
    const others = { other: path.join(appData, 'work'), programFolders: [] }
    expect(locationProblem('userData', path.join(appData, 'work-notes'), others)).toBeNull()
  })
})
