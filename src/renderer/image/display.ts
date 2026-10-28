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
