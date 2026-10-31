import type { GalleryFolder } from "./types";

/**
 * The gallery's pure helpers: choosing several items, the keyboard folder list, and
 * which pages to read so that a refresh never throws away what "Load more" brought in.
 */

/** What a click with Ctrl (toggle) or Shift (a run from the anchor) makes of a selection. */
export function selectWith(
  selected: ReadonlySet<string>,
  ids: readonly string[],
  id: string,
  how: { toggle: boolean; range: boolean },
  anchor: string | null
): Set<string> {
  const next = new Set(selected);
  if (how.range && anchor !== null) {
    const from = ids.indexOf(anchor);
    const to = ids.indexOf(id);
    if (from >= 0 && to >= 0) {
      for (const one of ids.slice(Math.min(from, to), Math.max(from, to) + 1)) next.add(one);
      return next;
    }
  }
  if (next.has(id)) next.delete(id);
  else next.add(id);
  return next;
}

export interface NumberedFolder {
  /** The number key that picks it, or null after the ninth folder. */
  key: number | null;
  /** null: no folder. */
  id: string | null;
  name: string;
}

/** "No folder" as 0, then the folders in list order, the first nine numbered 1 to 9. */
export function numberedFolders(folders: readonly GalleryFolder[]): NumberedFolder[] {
  return [
    { key: 0, id: null, name: "No folder" },
    ...folders.map((f, i) => ({
      key: i < 9 ? i + 1 : null,
      id: f.id,
      // A locked folder takes items without its PIN; they then leave view until it is unlocked.
      name: f.locked && !f.unlocked ? `🔒 ${f.name}` : f.name,
    })),
  ];
}

/** The digit a key press means, top row or number pad, whatever the keyboard layout. */
export function digitOf(e: { key: string; code?: string }): number | null {
  const code = /^(?:Digit|Numpad)([0-9])$/.exec(e.code ?? "");
  if (code) return Number(code[1]);
  return /^[0-9]$/.test(e.key) ? Number(e.key) : null;
}

/** The most records one gallery request may ask for (the main process caps it there). */
export const READ_LIMIT = 200;

export interface Loaded {
  /** The view, search and filter the items were read for. */
  key: string;
  /** The gallery version they were read at. */
  version: string;
  count: number;
}

/**
 * Which items to read next. More of the same list (Load more) reads only what is missing;
 * anything else — another view, a new version after a change — reads the list again from
 * the top, as far down as was wanted, so a refresh keeps every page already brought in.
 */
export function planRead(loaded: Loaded | null, want: { key: string; version: string; count: number }) {
  if (loaded && loaded.key === want.key && loaded.version === want.version && want.count > loaded.count) {
    return { offset: loaded.count, count: want.count - loaded.count, replace: false };
  }
  return { offset: 0, count: want.count, replace: true };
}

/** A read split into requests the main process accepts. */
export function readChunks(offset: number, count: number, limit = READ_LIMIT): { offset: number; limit: number }[] {
  const out = [];
  for (let at = offset; at < offset + count; at += limit) out.push({ offset: at, limit: Math.min(limit, offset + count - at) });
  return out;
}
