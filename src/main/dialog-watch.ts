import { dialog } from 'electron'

/**
 * Whether one of the studio's own native dialogs (a file picker, a save dialog, a message
 * box) is open. Opening one takes focus from the window like switching to another program
 * does, and locking gallery folders on that (config.folderPrivacy.lockOnBlur) would lock
 * them halfway through "Save as…" from an open folder.
 *
 * Every dialog goes through Electron's one `dialog` object, so its three async methods are
 * counted here once, rather than at each of the places that open one.
 */

let open = 0

export function dialogOpen(): boolean {
  return open > 0
}

export function watchDialogs(): void {
  for (const name of ['showOpenDialog', 'showSaveDialog', 'showMessageBox'] as const) {
    const original = dialog[name].bind(dialog) as (...args: unknown[]) => Promise<unknown>
    ;(dialog as unknown as Record<string, unknown>)[name] = async (...args: unknown[]): Promise<unknown> => {
      open++
      try {
        return await original(...args)
      } finally {
        open--
      }
    }
  }
}
