/**
 * The change history the window shows: every file the agent wrote, edited or
 * deleted, grouped by the reply that did it, so a whole reply can be put back.
 */

export type ChangeKind = 'write' | 'edit' | 'delete' | 'restore'

export interface ChangeFile {
  path: string
  kind: ChangeKind
  /** False when the change created the file. */
  existedBefore: boolean
}

export interface ChangeGroup {
  /** The prompt id of the reply that made the changes, or a revert's own id. */
  id: string
  /** What the user asked for, shortened; or what was reverted. */
  label: string
  chatId: string | null
  at: string
  /** One per file, in the order first touched. */
  files: ChangeFile[]
  /** True once every change in the group has been put back. */
  reverted: boolean
  /** Set on a revert: the group it put back. Reverting a revert redoes the original. */
  revertOf: string | null
}

export interface RevertResult {
  /** False when nothing was done because of conflicts and `force` was not given. */
  done: boolean
  restored: string[]
  /**
   * Files that changed after the agent touched them (edited by you, or by a
   * later reply). Put back only with `force`; their current state is backed up
   * first either way, so the revert itself can be undone.
   */
  conflicts: string[]
}
