/**
 * Where the studio keeps what it makes.
 *
 * Two folders can be moved: the one holding the chats and settings, and the
 * one holding the gallery (images and videos). Moving one only changes where
 * new data is written — nothing is copied or deleted, and the folder left
 * behind keeps everything in it, ready to be used again by choosing it back
 * (docs/decisions.md S18).
 */

export type DataLocationKey = 'userData' | 'outputs'

/** A folder the user can move: where it is now, and where it sits when nothing is chosen. */
export interface DataLocation {
  /** The folder that is being used, or will be after a restart (see `restartNeeded`). */
  path: string
  /** The place it goes when nothing has been chosen. */
  fallback: string
  /** Whether `path` is just `fallback`: nothing chosen. */
  isDefault: boolean
}

export interface DataLocationsView {
  /** Chats, settings, approvals and the chat key. */
  userData: DataLocation
  /** The gallery: images, videos, thumbnails and gallery.db. */
  outputs: DataLocation
  /**
   * The chats folder was changed and the studio is still running on the old
   * one: the stores are opened once, at startup, so only a restart switches.
   */
  restartNeeded: boolean
}
