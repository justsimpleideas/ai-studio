import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { WorkspaceStore } from '../src/main/workspaces.js'
import { composeSystemPrompt, type WorkingContext } from '../src/shared/config.js'
import { toolsFor } from '../src/shared/network.js'

/**
 * Strict mode: conversation only. Three things make it structural rather than
 * a request — no tool is given to the model, no tool is described to it, and
 * (in the main process) a call that arrives anyway is refused — and the first
 * two are checked here.
 */

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'strict-test-'))
afterAll(() => fs.rmSync(dir, { recursive: true, force: true }))

const context: WorkingContext = {
  workingFolder: 'C:\\work',
  extraFolders: ['D:\\more'],
  projectRules: [{ id: 'r', title: 'Style', body: 'Use tabs.', enabled: true }],
  readOnly: ['.git'],
  internet: 'download'
}

describe('strict mode', () => {
  it('gives the model no tools at all, whatever the internet level', () => {
    expect(toolsFor('offline', true)).toEqual([])
    expect(toolsFor('download', true)).toEqual([])
    expect(toolsFor('download', false).length).toBeGreaterThan(5)
  })

  it('tells the model it has nothing, and describes no folder, internet or protected path', () => {
    const prompt = composeSystemPrompt([], { ...context, strict: true })
    expect(prompt).toContain('in conversation only')
    expect(prompt).toContain('strict mode')
    for (const absent of ['C:\\work', 'D:\\more', 'web_search', 'download_file', '.git', 'working folder']) {
      expect(prompt).not.toContain(absent)
    }
    // The rules still apply: strict is about tools, not about instructions.
    expect(prompt).toContain('Use tabs.')
  })

  it('leaves the ordinary prompt alone when off', () => {
    const prompt = composeSystemPrompt([], { ...context, strict: false })
    expect(prompt).toContain('C:\\work')
    expect(prompt).toContain('web_search')
    expect(prompt).not.toContain('strict mode')
  })

  it('is off for a new workspace and for one saved before the setting existed', async () => {
    const data = path.join(dir, 'data')
    const folder = path.join(dir, 'project')
    fs.mkdirSync(folder, { recursive: true })
    const store = new WorkspaceStore(data)
    const ws = await store.open(folder)
    expect(ws.strict).toBe(false)
    const file = path.join(data, ws.id, 'workspace.json')
    const saved = JSON.parse(fs.readFileSync(file, 'utf8')) as Record<string, unknown>
    delete saved['strict']
    saved['internet'] = 'search'
    fs.writeFileSync(file, JSON.stringify(saved))
    const reopened = await new WorkspaceStore(data).open(folder)
    expect(reopened.strict).toBe(false)
    expect(reopened.internet).toBe('search')
  })
})
