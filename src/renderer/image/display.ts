// What the Create and Video tabs show once something is made: the finished result on the
// stage, and the strip of this session's thumbnails under it. Each can be turned off on its
// own, for images and videos separately. Kept in the window's storage, as the forms are.

export interface ResultDisplay {
  /** A finished image or video takes over the stage as soon as it is done. */
  showResult: boolean;
  /** The strip of this session's thumbnails under the stage. */
  showStrip: boolean;
}

/**
 * Where the gallery opens: All, Favorites, one chosen folder ("folder:" and its id) each
 * time, or the view it was left on.
 */
export type GalleryStart = "all" | "favorites" | "last" | `folder:${string}`;

export function isGalleryStart(value: unknown): value is GalleryStart {
  return (
    value === "all" ||
    value === "favorites" ||
    value === "last" ||
    (typeof value === "string" && /^folder:[\w-]{1,64}$/.test(value))
  );
}

export interface DisplaySettings {
  image: ResultDisplay;
  video: ResultDisplay;
  galleryStart: GalleryStart;
}

export const DEFAULT_DISPLAY: DisplaySettings = {
  image: { showResult: true, showStrip: true },
  video: { showResult: true, showStrip: true },
  galleryStart: "all",
};

const STORAGE_KEY = "lis.display.v1";

function readOne(value: unknown): ResultDisplay {
  const part = (value ?? {}) as Partial<ResultDisplay>;
  return {
    showResult: typeof part.showResult === "boolean" ? part.showResult : true,
    showStrip: typeof part.showStrip === "boolean" ? part.showStrip : true,
  };
}

export function loadDisplay(storage: Pick<Storage, "getItem"> = localStorage): DisplaySettings {
  try {
    const parsed = JSON.parse(storage.getItem(STORAGE_KEY) ?? "null") as Partial<DisplaySettings> | null;
    if (!parsed || typeof parsed !== "object") return DEFAULT_DISPLAY;
    return {
      image: readOne(parsed.image),
      video: readOne(parsed.video),
      galleryStart: isGalleryStart(parsed.galleryStart) ? parsed.galleryStart : "all",
    };
  } catch {
    return DEFAULT_DISPLAY;
  }
}

/** The gallery view last open: All, Favorites or a folder (by id). */
export type GalleryViewKind = { kind: "all" } | { kind: "favorites" } | { kind: "folder"; id: string };

const VIEW_KEY = "lis.gallery.view.v1";

export function loadGalleryView(storage: Pick<Storage, "getItem"> = localStorage): GalleryViewKind {
  try {
    const parsed = JSON.parse(storage.getItem(VIEW_KEY) ?? "null") as { kind?: unknown; id?: unknown } | null;
    if (parsed?.kind === "favorites") return { kind: "favorites" };
    if (parsed?.kind === "folder" && typeof parsed.id === "string" && parsed.id) return { kind: "folder", id: parsed.id };
  } catch {
    // unreadable: All
  }
  return { kind: "all" };
}

/** The view the gallery opens on. A folder that is gone, locked or hidden then is left to the gallery, which shows All. */
export function startView(start: GalleryStart, last: () => GalleryViewKind = loadGalleryView): GalleryViewKind {
  if (start === "favorites") return { kind: "favorites" };
  if (start === "last") return last();
  if (start.startsWith("folder:")) return { kind: "folder", id: start.slice("folder:".length) };
  return { kind: "all" };
}

/**
 * How big the gallery's thumbnails are, as a file manager's view sizes: "large" is
 * the size the gallery has always had and stays the default, with two smaller ones
 * for seeing more at once. Only the window's own setting; nothing is re-rendered or
 * re-saved, so switching is instant and costs nothing.
 */
export type TileSize = "large" | "medium" | "small";

export const DEFAULT_TILE_SIZE: TileSize = "large";

/** The sizes in the order the buttons show them, biggest first. */
export const TILE_SIZES: readonly TileSize[] = ["large", "medium", "small"];

const TILE_KEY = "lis.gallery.tileSize.v1";

export function isTileSize(value: unknown): value is TileSize {
  return value === "large" || value === "medium" || value === "small";
}

export function loadTileSize(storage: Pick<Storage, "getItem"> = localStorage): TileSize {
  try {
    const stored = storage.getItem(TILE_KEY);
    return isTileSize(stored) ? stored : DEFAULT_TILE_SIZE;
  } catch {
    return DEFAULT_TILE_SIZE;
  }
}

/**
 * Covering the gallery's thumbnails with a mosaic: off, or two strengths.
 *
 * This is a cover for the eyes in the room, not protection: it hides what the
 * grid shows at a glance, while opening an item still shows it, and the files on
 * disk are untouched. For keeping things out of view altogether there are hidden
 * and locked folders (gallery.py).
 */
export type MosaicLevel = "off" | "weak" | "strong";

export const DEFAULT_MOSAIC: MosaicLevel = "off";

/** The choices in the order the buttons show them. */
export const MOSAIC_LEVELS: readonly MosaicLevel[] = ["off", "weak", "strong"];

const MOSAIC_KEY = "lis.gallery.mosaic.v1";

export function isMosaicLevel(value: unknown): value is MosaicLevel {
  return value === "off" || value === "weak" || value === "strong";
}

export function loadMosaic(storage: Pick<Storage, "getItem"> = localStorage): MosaicLevel {
  try {
    const stored = storage.getItem(MOSAIC_KEY);
    return isMosaicLevel(stored) ? stored : DEFAULT_MOSAIC;
  } catch {
    return DEFAULT_MOSAIC;
  }
}

export function saveMosaic(level: MosaicLevel, storage: Pick<Storage, "setItem"> = localStorage): void {
  try {
    storage.setItem(MOSAIC_KEY, level);
  } catch {
    // storage full or disabled: the choice lasts until the window closes
  }
}

/**
 * The order of the folder list after moving `id` in front of `before` (to the end
 * when that is null). The folder is taken out first, so `before` names a folder in
 * the list as it will be without it — the same rule the engine follows (gallery.py
 * move_folder), which keeps what the drag shows and what is saved in step.
 */
export function reorderedIds(ids: readonly string[], id: string, before: string | null): string[] {
  // A folder that is not in the list (deleted, hidden while the drag was on) is not
  // put into it: the list the drag started from is what comes back.
  if (!ids.includes(id)) return [...ids];
  const rest = ids.filter((i) => i !== id);
  const at = before === null ? rest.length : rest.indexOf(before);
  if (at < 0) return [...ids];
  return [...rest.slice(0, at), id, ...rest.slice(at)];
}

/**
 * Moving a folder one place up or down: the folder to put it before, or null for
 * the end of the list. Null as the whole answer means it cannot go that way.
 *
 * Down is the awkward one: dropping a folder in front of its neighbour's neighbour
 * is what puts it after its neighbour, because the folder is taken out of the list
 * before being put back.
 */
export function nudgeTarget(
  ids: readonly string[],
  id: string,
  direction: -1 | 1,
): { before: string | null } | null {
  const at = ids.indexOf(id);
  if (at < 0) return null;
  if (direction === -1) return at === 0 ? null : { before: ids[at - 1] ?? null };
  if (at === ids.length - 1) return null;
  return { before: ids[at + 2] ?? null };
}

export function saveTileSize(size: TileSize, storage: Pick<Storage, "setItem"> = localStorage): void {
  try {
    storage.setItem(TILE_KEY, size);
  } catch {
    // storage full or disabled: the choice lasts until the window closes
  }
}

export function saveGalleryView(view: GalleryViewKind, storage: Pick<Storage, "setItem"> = localStorage): void {
  try {
    storage.setItem(VIEW_KEY, JSON.stringify(view));
  } catch {
    // storage full or disabled
  }
}

export function saveDisplay(display: DisplaySettings, storage: Pick<Storage, "setItem"> = localStorage): void {
  try {
    storage.setItem(STORAGE_KEY, JSON.stringify(display));
  } catch {
    // storage full or disabled: the choice lasts until the window closes
  }
}
