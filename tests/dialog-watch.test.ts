import { describe, expect, it, vi } from 'vitest'

// A stand-in for Electron's dialog object: each dialog stays open until the test closes it.
const { closers, fake } = vi.hoisted(() => {
  const closers: Array<() => void> = []
  const fake = {
    showOpenDialog: vi.fn(() => new Promise((done) => closers.push(() => done({ canceled: true, filePaths: [] })))),
    showSaveDialog: vi.fn(() => new Promise((done) => closers.push(() => done({ canceled: true })))),
    showMessageBox: vi.fn(() => new Promise((_, fail) => closers.push(() => fail(new Error('gone')))))
  }
  return { closers, fake }
})
vi.mock('electron', () => ({ dialog: fake }))

const { dialogOpen, watchDialogs } = await import('../src/main/dialog-watch.js')
const { dialog } = await import('electron')

describe('the studio’s own dialogs, for the lock on losing focus', () => {
  it('count as open from the moment one is asked for until it closes, however it ends', async () => {
    const original = fake.showOpenDialog // watching replaces it with a counting wrapper
    watchDialogs()
    expect(dialogOpen()).toBe(false)
    const picker = dialog.showOpenDialog({})
    const box = dialog.showMessageBox({ message: 'x' })
    expect(dialogOpen()).toBe(true)
    closers.shift()!()
    await picker
    expect(dialogOpen()).toBe(true) // the message box is still up
    closers.shift()!()
    await expect(box).rejects.toThrow('gone')
    expect(dialogOpen()).toBe(false)
    const save = dialog.showSaveDialog({})
    expect(dialogOpen()).toBe(true)
    closers.shift()!()
    await save
    expect(dialogOpen()).toBe(false)
    expect(original).toHaveBeenCalledTimes(1)
  })
})
