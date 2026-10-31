import { useCallback, useEffect, useRef, useState } from "react";
import { FOLDER_NAME_MAX, FOLDER_PIN } from "@shared/image.js";
import { matchesShortcut } from "@shared/shortcuts.js";
import { useShortcuts, useShortcutsRef } from "../../useShortcuts";
import {
  loadMosaic,
  loadTileSize,
  MOSAIC_LEVELS,
  nudgeTarget,
  reorderedIds,
  saveGalleryView,
  saveMosaic,
  saveTileSize,
  startView,
  TILE_SIZES,
  type GalleryStart,
  type GalleryViewKind,
  type MosaicLevel,
  type TileSize,
} from "../display";
import { api } from "../api";
import { formatSeconds } from "../format";
import { planRead, readChunks, selectWith, type Loaded } from "../selection";
import type { GalleryFolder, ImageRecord, MediaKind, SaveManyOptions } from "../types";
import type { Folders } from "../useFolders";
import { FolderPicker } from "./FolderPicker";

const PAGE = 60;
/**
 * How far the pointer must travel, in pixels, before a press on a folder becomes a
 * drag that reorders the list. Generous on purpose: a folder is clicked far more
 * often than it is moved, and a list that reshuffles on a twitch is worse than one
 * that needs a deliberate pull.
 */
const REORDER_SLOP = 8;
const FOLDERS_KEY = "lis.gallery.folders.v1";
/**
 * What a tile carries while it is dragged onto a folder: its id, or the ids of every chosen
 * item, comma-separated. Not a file, so the file drop zones ignore it.
 */
export const ITEM_DRAG_TYPE = "application/x-studio-item";

export type { GalleryViewKind } from "../display";

interface Props {
  version: number | string;
  folders: Folders;
  onOpen: (image: ImageRecord, list: ImageRecord[]) => void;
  /** Lets the parent push in-place updates (favorite, delete) without a refetch. */
  registerUpdater: (fn: (image: ImageRecord | { id: string; deleted: true }) => void) => void;
  onError: (message: string) => void;
  /** File an item in a folder (null: take it out of its folder). */
  onMove: (image: ImageRecord, folderId: string | null) => Promise<void>;
  /** File several at once (a multi-select, or a drag of one); resolves to how many moved. */
  onMoveMany: (ids: string[], folderId: string | null) => Promise<number>;
  /** Delete an item for good (already confirmed). */
  onDelete: (image: ImageRecord) => Promise<void>;
  /** Delete several for good (already confirmed). */
  onDeleteMany: (images: ImageRecord[]) => Promise<void>;
  /** A note that something went as asked ("Saved 3 items to ..."). */
  onNotice: (message: string) => void;
  /** Something is open over the gallery (an item's details, the queue): its keys are not the gallery's. */
  covered?: boolean;
  /** Changes when every open PIN folder was just locked at once (a shortcut, Ctrl+Shift+L by default). */
  lockedAll?: number;
  /** Open on All, Favorites or a chosen folder each time, or on the view last open (Settings). */
  start?: GalleryStart;
}

export function GalleryView({
  version, folders, onOpen, registerUpdater, onError, onMove, onMoveMany, onDelete, onDeleteMany, onNotice,
  covered = false, lockedAll = 0, start = "all",
}: Props) {
  const [items, setItems] = useState<ImageRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [query, setQuery] = useState("");
  const [view, setView] = useState<GalleryViewKind>(() => startView(start));
  // Kept either way, so switching to "where it was left" picks up from now.
  useEffect(() => saveGalleryView(view), [view]);
  const [media, setMedia] = useState<MediaKind | null>(null);
  // How big the thumbnails are, as in a file manager. Remembered for next time.
  const [tileSize, setTileSize] = useState<TileSize>(loadTileSize);
  useEffect(() => saveTileSize(tileSize), [tileSize]);
  // A mosaic over the thumbnails, for when the screen is not private. Remembered too.
  const [mosaic, setMosaic] = useState<MosaicLevel>(loadMosaic);
  useEffect(() => saveMosaic(mosaic), [mosaic]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [menu, setMenu] = useState<TileMenuAt | null>(null);
  const closeMenu = useCallback(() => setMenu(null), []);
  const request = useRef(0);
  const sectionRef = useRef<HTMLElement>(null);

  // The folder list (All, Favorites, folders) can be put away; the choice is remembered.
  const [showFolders, setShowFolders] = useState(() => {
    try {
      return localStorage.getItem(FOLDERS_KEY) !== "hidden";
    } catch {
      return true;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(FOLDERS_KEY, showFolders ? "shown" : "hidden");
    } catch {
      // storage full or disabled: the choice lasts until the window closes
    }
  }, [showFolders]);
  // The sidebar shortcut (Ctrl+B by default), as on the chat side, while the gallery is in front
  // (the image side stays mounted behind the chat).
  const shortcuts = useShortcutsRef();
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!matchesShortcut(e, shortcuts.current.toggleSidebar)) return;
      if (!sectionRef.current || sectionRef.current.offsetParent === null) return;
      e.preventDefault();
      setShowFolders((shown) => !shown);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const folder = view.kind === "folder" ? folders.list.find((f) => f.id === view.id) : undefined;

  // A folder that was just hidden or locked (or deleted) is no longer a view to stay in. Not
  // decided before the folder list first arrives, so a remembered folder is not dropped.
  const viewGone = view.kind === "folder" && folders.loaded && (!folder || (folder.locked && !folder.unlocked));
  useEffect(() => {
    if (viewGone) setView({ kind: "all" });
  }, [viewGone]);

  // Locking everything at once while in a PIN folder goes to Favorites (on the same render,
  // before the locked list arrives, so the step back to All above does not apply).
  const folderRef = useRef(folder);
  folderRef.current = folder;
  useEffect(() => {
    if (lockedAll && folderRef.current?.locked) {
      setView({ kind: "favorites" });
      setMenu(null);
    }
  }, [lockedAll]);

  // What is on screen is read as "the first `want` items of this list". Load more asks for more
  // of the same list and reads only what is missing; a change elsewhere (a new image, a move,
  // a deletion) reads the list again as far down as it was, so pages already brought in stay.
  // Only the newest read counts: a slow answer for a view that was left is dropped.
  const listKey = JSON.stringify([view, media]);
  const readKey = JSON.stringify([view, media, query.trim()]);
  const [want, setWant] = useState(PAGE);
  const [refreshTick, setRefreshTick] = useState(0);
  const readVersion = `${version}:${refreshTick}`;
  const loaded = useRef<Loaded | null>(null);

  // Another view or filter: what the last one showed goes at once, rather than standing in
  // for the new one until its answer arrives.
  const shownKey = useRef(listKey);
  if (shownKey.current !== listKey) {
    shownKey.current = listKey;
    loaded.current = null;
    setItems([]);
    setTotal(0);
    setWant(PAGE);
  }

  useEffect(() => {
    if (viewGone) return;
    const plan = planRead(loaded.current, { key: readKey, version: readVersion, count: want });
    const id = ++request.current;
    const base = {
      q: query.trim(),
      favorites: view.kind === "favorites",
      ...(view.kind === "folder" ? { folder: view.id } : {}),
      ...(media ? { media } : {}),
    };
    // Typing in the search box waits for a pause; nothing else does.
    const typing = plan.replace && query !== "" && loaded.current !== null && loaded.current.key !== readKey;
    const t = window.setTimeout(async () => {
      setLoading(true);
      try {
        const got: ImageRecord[] = [];
        let all = 0;
        for (const chunk of readChunks(plan.offset, plan.count)) {
          const page = await api.gallery({ ...base, ...chunk });
          if (id !== request.current) return;
          got.push(...page.items);
          all = page.total;
          if (page.items.length < chunk.limit) break;
        }
        setTotal(all);
        setItems((prev) => {
          if (plan.replace) return got;
          const have = new Set(prev.map((i) => i.id));
          return [...prev, ...got.filter((i) => !have.has(i.id))];
        });
        loaded.current = { key: readKey, version: readVersion, count: (plan.replace ? 0 : loaded.current?.count ?? 0) + got.length };
        setError(null);
      } catch (e) {
        if (id === request.current) setError((e as Error).message);
      } finally {
        if (id === request.current) setLoading(false);
      }
    }, typing ? 200 : 0);
    return () => window.clearTimeout(t);
  }, [readKey, readVersion, want, viewGone]);

  // ---------------------------------------------------------------- choosing several

  const [selecting, setSelecting] = useState(false);
  const [selected, setSelected] = useState<ReadonlySet<string>>(() => new Set());
  const [bulk, setBulk] = useState<"move" | "save" | "delete" | null>(null);
  const [bulkBusy, setBulkBusy] = useState(false);
  const anchor = useRef<string | null>(null);
  const endSelecting = useCallback(() => {
    setSelecting(false);
    setSelected(new Set());
    setBulk(null);
    anchor.current = null;
  }, []);
  // A selection belongs to the list it was made in.
  useEffect(() => endSelecting(), [listKey, endSelecting]);
  // Items that left the list (moved out, deleted, out of view) leave the selection too.
  useEffect(() => {
    setSelected((s) => {
      if (s.size === 0) return s;
      const present = new Set(items.map((i) => i.id));
      const kept = [...s].filter((id) => present.has(id));
      return kept.length === s.size ? s : new Set(kept);
    });
  }, [items]);
  const chosen = items.filter((i) => selected.has(i.id));
  const selectAll = (): void => {
    setSelecting(true);
    setSelected(new Set(items.map((i) => i.id)));
  };

  const moveChosen = async (folderId: string | null): Promise<void> => {
    const ids = chosen.map((i) => i.id);
    if (ids.length === 0) return;
    setBulk(null);
    setBulkBusy(true);
    try {
      const moved = await onMoveMany(ids, folderId);
      const where = folderId === null
        ? "out of their folders"
        : `to \u201c${folders.list.find((f) => f.id === folderId)?.name ?? "the folder"}\u201d`;
      onNotice(`Moved ${moved} item${moved === 1 ? "" : "s"} ${where}.`);
      endSelecting();
    } catch (e) {
      onError((e as Error).message);
    } finally {
      setBulkBusy(false);
    }
  };

  const deleteChosen = async (): Promise<void> => {
    setBulkBusy(true);
    try {
      await onDeleteMany(chosen);
      endSelecting();
    } catch (e) {
      onError((e as Error).message);
    } finally {
      setBulkBusy(false);
      setRefreshTick((n) => n + 1);
    }
  };

  // Ctrl+A chooses every item shown, Escape lets go, and the "Move to a folder" shortcut opens
  // the numbered folder list for what is chosen, while the gallery is in front and no text box
  // has the keys.
  const keys = useRef({ selecting, chosen: chosen.length, covered, menu: menu !== null, selectAll, endSelecting });
  keys.current = { selecting, chosen: chosen.length, covered, menu: menu !== null, selectAll, endSelecting };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const k = keys.current;
      if (k.covered || k.menu || !sectionRef.current || sectionRef.current.offsetParent === null) return;
      const t = e.target;
      if (t instanceof HTMLInputElement || t instanceof HTMLTextAreaElement || t instanceof HTMLSelectElement) return;
      if (matchesShortcut(e, "Ctrl+A")) {
        e.preventDefault();
        k.selectAll();
      } else if (e.key === "Escape" && k.selecting) {
        k.endSelecting();
      } else if (matchesShortcut(e, shortcuts.current.pickFolder) && k.chosen > 0) {
        e.preventDefault();
        setBulk((b) => (b === "move" ? null : "move"));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  const pickKeys = useShortcuts().pickFolder;

  useEffect(() => {
    registerUpdater((update) => {
      if ("deleted" in update) {
        setItems((prev) => prev.filter((i) => i.id !== update.id));
        setTotal((t) => Math.max(0, t - 1));
      } else {
        setItems((prev) => prev.map((i) => (i.id === update.id ? update : i)));
      }
    });
  }, [registerUpdater]);
  const moveDropped = (ids: string[], folderId: string | null): void => {
    onMoveMany(ids, folderId).catch((e: Error) => onError(e.message));
  };

  const heading = view.kind === "all" ? "All" : view.kind === "favorites" ? "Favorites" : folder?.name ?? "";
  const noun = media ?? "item";

  return (
    <section className={`gallery-view ${showFolders ? "" : "folders-hidden"}`} ref={sectionRef}>
      {showFolders && <FolderList view={view} onView={setView} folders={folders} onError={onError} onMoveItems={moveDropped} />}
      <div className="gallery-main">
        <div className="gallery-toolbar">
          <button
            className="icon-button folders-toggle"
            aria-label="Folders"
            aria-pressed={showFolders}
            title={`${showFolders ? "Hide" : "Show"} folders (${shortcuts.current.toggleSidebar})`}
            onClick={() => setShowFolders((shown) => !shown)}
          >
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden>
              <rect x="3" y="4.5" width="18" height="15" />
              <path d="M9 4.5v15" />
              {showFolders ? <rect x="3" y="4.5" width="6" height="15" fill="currentColor" opacity="0.45" stroke="none" /> : null}
            </svg>
          </button>
          <h2 className="gallery-heading" title={heading}>{heading}</h2>
          <input
            className="input search"
            type="search"
            placeholder="Search prompts"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <div className="segmented media-filter" role="radiogroup" aria-label="Show">
            {([[null, "All"], ["image", "Images"], ["video", "Videos"]] as const).map(([value, label]) => (
              <button
                type="button"
                key={label}
                role="radio"
                aria-checked={media === value}
                className={media === value ? "active" : ""}
                onClick={() => setMedia(value)}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="segmented tile-size" role="radiogroup" aria-label="Thumbnail size">
            {TILE_SIZES.map((size) => (
              <button
                type="button"
                key={size}
                role="radio"
                aria-checked={tileSize === size}
                className={tileSize === size ? "active" : ""}
                title={`${size === "large" ? "Large" : size === "medium" ? "Medium" : "Small"} thumbnails`}
                onClick={() => setTileSize(size)}
              >
                {size === "large" ? "Large" : size === "medium" ? "Medium" : "Small"}
              </button>
            ))}
          </div>
          <div className="segmented mosaic-level" role="radiogroup" aria-label="Mosaic">
            {MOSAIC_LEVELS.map((level) => (
              <button
                type="button"
                key={level}
                role="radio"
                aria-checked={mosaic === level}
                className={mosaic === level ? "active" : ""}
                title={
                  level === "off"
                    ? "Thumbnails as they are"
                    : `A ${level} mosaic over the thumbnails. Opening one still shows it.`
                }
                onClick={() => setMosaic(level)}
              >
                {level === "off" ? "No mosaic" : level === "weak" ? "Weak" : "Strong"}
              </button>
            ))}
          </div>
          <span className="muted small">{total} {noun}{total === 1 ? "" : "s"}</span>
          <button
            type="button"
            className={`small select-toggle ${selecting ? "active" : ""}`}
            aria-pressed={selecting}
            title="Choose several items to move, save or delete together (or Ctrl+click and Shift+click a tile)"
            onClick={() => (selecting ? endSelecting() : setSelecting(true))}
          >
            {selecting ? "Done" : "Select"}
          </button>
        </div>
        {selecting && (
          <div className="selection-bar" role="toolbar" aria-label="Chosen items">
            <span className="selection-count">{chosen.length === 0 ? "Click items to choose them" : `${chosen.length} chosen`}</span>
            <button type="button" className="ghost small" onClick={selectAll} disabled={items.length === 0}>
              Select all{items.length < total ? ` ${items.length} shown` : ""}
            </button>
            {chosen.length > 0 && (
              <button type="button" className="ghost small" onClick={() => setSelected(new Set())}>Clear</button>
            )}
            <span className="selection-actions">
              <span className="selection-anchor">
                <button type="button" className="small" disabled={chosen.length === 0 || bulkBusy} aria-expanded={bulk === "move"} onClick={() => setBulk(bulk === "move" ? null : "move")}>
                  Move to folder… <span className="folder-shortcut">{pickKeys.replaceAll("+", " + ")}</span>
                </button>
                {bulk === "move" && (
                  <FolderPicker
                    folders={folders.list}
                    subject={`${chosen.length} item${chosen.length === 1 ? "" : "s"}`}
                    onPick={(folderId) => void moveChosen(folderId)}
                    onClose={() => setBulk(null)}
                  />
                )}
              </span>
              <span className="selection-anchor">
                <button type="button" className="small" disabled={chosen.length === 0 || bulkBusy} aria-expanded={bulk === "save"} onClick={() => setBulk(bulk === "save" ? null : "save")}>
                  Save copies…
                </button>
                {bulk === "save" && (
                  <SaveManyPanel
                    items={chosen}
                    onClose={() => setBulk(null)}
                    onSaved={(text) => {
                      setBulk(null);
                      onNotice(text);
                    }}
                    onError={onError}
                  />
                )}
              </span>
              <span className="selection-anchor">
                <button type="button" className="small ghost danger-text" disabled={chosen.length === 0 || bulkBusy} onClick={() => setBulk(bulk === "delete" ? null : "delete")}>
                  Delete…
                </button>
                {bulk === "delete" && (
                  <div className="bulk-pop" role="alertdialog" aria-label="Delete the chosen items">
                    <p>Delete {chosen.length} item{chosen.length === 1 ? "" : "s"} for good? This cannot be undone.</p>
                    <div className="row-gap">
                      <button type="button" className="danger small" disabled={bulkBusy} onClick={() => void deleteChosen()}>Delete for good</button>
                      <button type="button" className="ghost small" onClick={() => setBulk(null)}>Cancel</button>
                    </div>
                  </div>
                )}
              </span>
            </span>
          </div>
        )}
        {error && <div className="error-text">{error}</div>}
        {loading && items.length === 0 && <div className="empty-gallery muted">Loading…</div>}
        {!loading && items.length === 0 && !error && (
          <div className="empty-gallery muted">
            {query || media
              ? "Nothing matches."
              : view.kind === "favorites"
                ? "No favorites yet. Star an image or video in its details, wherever it is filed."
                : view.kind === "folder"
                  ? "This folder is empty. Drag images and videos onto it, or choose it in their details."
                  : "Nothing here yet. Generated images and videos are saved automatically."}
          </div>
        )}
        <div className="gallery-grid" data-size={tileSize} data-mosaic={mosaic}>
          {items.map((img) => (
            <button
              key={img.id}
              className={`tile ${selected.has(img.id) ? "selected" : ""}`}
              data-id={img.id}
              aria-pressed={selecting ? selected.has(img.id) : undefined}
              style={{ "--ar": img.width > 0 && img.height > 0 ? img.width / img.height : 1.6 } as React.CSSProperties}
              draggable
              onDragStart={(e) => {
                // A chosen tile takes every chosen item along.
                const ids = selected.has(img.id) ? chosen.map((i) => i.id) : [img.id];
                e.dataTransfer.setData(ITEM_DRAG_TYPE, ids.join(","));
                e.dataTransfer.effectAllowed = "move";
              }}
              onClick={(e) => {
                const toggle = e.ctrlKey || e.metaKey;
                const range = e.shiftKey;
                if (selecting || toggle || range) {
                  setSelecting(true);
                  setSelected((s) => selectWith(s, items.map((i) => i.id), img.id, { toggle, range }, anchor.current));
                  if (!range || anchor.current === null) anchor.current = img.id;
                  return;
                }
                onOpen(img, items);
              }}
              onContextMenu={(e) => {
                e.preventDefault();
                setMenu({ image: img, x: e.clientX, y: e.clientY });
              }}
              onKeyDown={(e) => {
                // The keyboard's menu key, or Shift+F10, opens the same menu at the tile.
                if (e.key === "ContextMenu" || (e.key === "F10" && e.shiftKey)) {
                  e.preventDefault();
                  const r = e.currentTarget.getBoundingClientRect();
                  setMenu({ image: img, x: r.left + 12, y: r.top + 12 });
                }
              }}
              title={img.prompt}
            >
              {mosaic === "off" ? (
                <img src={img.thumb_url} alt={img.prompt} loading="lazy" style={{ aspectRatio: `${img.width} / ${img.height}` }} />
              ) : (
                <MosaicThumb src={img.thumb_url} ratio={img.width / img.height} level={mosaic} />
              )}
              {img.favorite && <span className="badge" aria-label="Favorite">★</span>}
              {selecting && <span className="tile-check" aria-hidden>{selected.has(img.id) ? "✓" : ""}</span>}
              {img.media === "video" && (
                <span className="video-badge" aria-label="Video">▶ {formatSeconds(img.frames ?? 0, img.fps ?? 0)}</span>
              )}
            </button>
          ))}
        </div>
        {items.length < total && (
          <div className="center">
            <button onClick={() => setWant(items.length + PAGE)} disabled={loading}>{loading ? "Loading…" : "Load more"}</button>
          </div>
        )}
      </div>
      {menu && (
        <TileMenu
          at={menu}
          folders={folders.list}
          onClose={closeMenu}
          onOpen={() => onOpen(menu.image, items)}
          onMove={(folderId) => onMove(menu.image, folderId)}
          onDelete={() => onDelete(menu.image)}
          onError={onError}
        />
      )}
    </section>
  );
}

// ------------------------------------------------------------ saving copies

/**
 * Copies of the chosen items, saved into a folder picked next. The options: whether images keep
 * the prompt and settings inside them (the "Save as…" setting to start with), and whether
 * each item gets a .txt beside it with its prompt and settings. Nothing is overwritten.
 */
function SaveManyPanel({ items, onClose, onSaved, onError }: {
  items: ImageRecord[];
  onClose: () => void;
  onSaved: (message: string) => void;
  onError: (message: string) => void;
}) {
  const [options, setOptions] = useState<SaveManyOptions>({ includeSettings: false, promptFiles: false });
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    api.includeSettingsInExports().then((on) => setOptions((o) => ({ ...o, includeSettings: on })), () => undefined);
  }, []);
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", key);
    return () => document.removeEventListener("keydown", key);
  }, [onClose]);
  const videos = items.filter((i) => i.media === "video").length;

  const save = async (): Promise<void> => {
    setBusy(true);
    try {
      const result = await api.saveManyTo(items.map((i) => i.id), options);
      if (result) {
        const failed = result.failed ? ` ${result.failed} could not be saved.` : "";
        onSaved(`Saved ${result.saved} item${result.saved === 1 ? "" : "s"} to ${result.folder}.${failed}`);
      }
    } catch (e) {
      onError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="bulk-pop save-many" role="dialog" aria-label="Save copies">
      <p className="folder-picker-title">Save {items.length} item{items.length === 1 ? "" : "s"}</p>
      <label className="toggle wrap">
        <input type="checkbox" checked={options.includeSettings} onChange={(e) => setOptions({ ...options, includeSettings: e.target.checked })} />
        <span>Keep the prompt and settings inside each image</span>
      </label>
      <label className="toggle wrap">
        <input type="checkbox" checked={options.promptFiles} onChange={(e) => setOptions({ ...options, promptFiles: e.target.checked })} />
        <span>Also save each prompt and its settings as a .txt file beside it</span>
      </label>
      <p className="hint">
        {videos > 0 ? "Videos never carry the prompt inside the file. " : ""}
        Files already in the folder are kept; a copy with the same name gets a number.
      </p>
      <div className="row-gap">
        <button type="button" className="primary small" disabled={busy} onClick={() => void save()}>
          {busy ? "Saving…" : "Choose a folder and save"}
        </button>
        <button type="button" className="ghost small" onClick={onClose}>Cancel</button>
      </div>
    </div>
  );
}

// ------------------------------------------------------------ tile menu

interface TileMenuAt {
  image: ImageRecord;
  x: number;
  y: number;
}

const MENU_WIDTH = 240;
const MENU_MAX_HEIGHT = 360;

/**
 * Right-click on a tile: open it, file it in a folder (or take it out of its folder), or
 * delete it, which asks once more. Folders that are locked are left out until unlocked.
 * Closes on a click elsewhere, Escape, scrolling or resizing.
 */
function TileMenu({ at, folders, onClose, onOpen, onMove, onDelete, onError }: {
  at: TileMenuAt;
  folders: GalleryFolder[];
  onClose: () => void;
  onOpen: () => void;
  onMove: (folderId: string | null) => Promise<void>;
  onDelete: () => Promise<void>;
  onError: (message: string) => void;
}) {
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const image = at.image;
  const targets = folders.filter((f) => f.id !== image.folder_id && !(f.locked && !f.unlocked));
  const current = folders.find((f) => f.id === image.folder_id);

  useEffect(() => {
    const away = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) onClose();
    };
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("mousedown", away);
    document.addEventListener("keydown", key);
    window.addEventListener("resize", onClose);
    window.addEventListener("scroll", onClose, true);
    return () => {
      document.removeEventListener("mousedown", away);
      document.removeEventListener("keydown", key);
      window.removeEventListener("resize", onClose);
      window.removeEventListener("scroll", onClose, true);
    };
  }, [onClose]);

  // Keyboard users land on the first item, once per menu and again when the delete question shows.
  useEffect(() => {
    ref.current?.querySelector<HTMLButtonElement>("button")?.focus();
  }, [at, confirm]);

  const run = (action: () => Promise<void>) => () => {
    setBusy(true);
    void action().then(onClose, (e: Error) => {
      setBusy(false);
      onError(e.message);
    });
  };

  // Kept inside the window: opened near the right or bottom edge, it opens to the left or upward.
  const left = Math.max(8, Math.min(at.x, window.innerWidth - MENU_WIDTH - 8));
  const top = Math.max(8, Math.min(at.y, window.innerHeight - MENU_MAX_HEIGHT - 8));
  const noun = image.media === "video" ? "video" : "image";

  return (
    <div
      ref={ref}
      className="tile-menu"
      role="menu"
      aria-label={`Options for this ${noun}`}
      style={{ left, top, width: MENU_WIDTH, maxHeight: MENU_MAX_HEIGHT }}
      onContextMenu={(e) => e.preventDefault()}
    >
      {confirm ? (
        <div className="tile-menu-confirm" role="alertdialog" aria-label={`Delete this ${noun}`}>
          <p>Delete this {noun} for good? This cannot be undone.</p>
          <div className="row-gap">
            <button className="danger small" disabled={busy} onClick={run(onDelete)}>Delete for good</button>
            <button className="ghost small" onClick={() => setConfirm(false)}>Cancel</button>
          </div>
        </div>
      ) : (
        <>
          <button role="menuitem" className="ghost small" onClick={() => { onOpen(); onClose(); }}>Open</button>
          <p className="tile-menu-label">{current ? `In “${current.name}”. Move to` : "Move to folder"}</p>
          {targets.length === 0 && !current && <p className="tile-menu-note">No folders yet. Make one in the folder list.</p>}
          {targets.map((f) => (
            <button key={f.id} role="menuitem" className="ghost small" disabled={busy} title={f.name} onClick={run(() => onMove(f.id))}>
              <span className="folder-name">{f.name}</span>
            </button>
          ))}
          {current && (
            <button role="menuitem" className="ghost small" disabled={busy} onClick={run(() => onMove(null))}>
              Take out of “{current.name}”
            </button>
          )}
          <hr />
          <button role="menuitem" className="ghost small danger-text" onClick={() => setConfirm(true)}>Delete…</button>
        </>
      )}
    </div>
  );
}

// ------------------------------------------------------------ folders

function hasItem(e: React.DragEvent): boolean {
  return Array.from(e.dataTransfer.types).includes(ITEM_DRAG_TYPE);
}

/** A place dragged tiles can be dropped on: a folder, or "All" to take them out of their folders. */
function useDropTarget(onItems: (ids: string[]) => void) {
  const [over, setOver] = useState(false);
  return {
    over,
    handlers: {
      onDragOver: (e: React.DragEvent) => {
        if (!hasItem(e)) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = "move";
        setOver(true);
      },
      onDragLeave: () => setOver(false),
      onDrop: (e: React.DragEvent) => {
        if (!hasItem(e)) return;
        e.preventDefault();
        setOver(false);
        const ids = e.dataTransfer.getData(ITEM_DRAG_TYPE).split(",").filter(Boolean);
        if (ids.length > 0) onItems(ids);
      },
    },
  };
}

function FolderList({ view, onView, folders, onError, onMoveItems }: {
  view: GalleryViewKind;
  onView: (view: GalleryViewKind) => void;
  folders: Folders;
  onError: (message: string) => void;
  onMoveItems: (ids: string[], folderId: string | null) => void;
}) {
  const [creating, setCreating] = useState(false);
  // Dragging a folder into a new place. A press is not a drag: nothing moves until the
  // pointer has travelled REORDER_SLOP, so clicking a folder — or a slip of the hand on
  // the way to a click — never reorders the list.
  const [drag, setDrag] = useState<{ id: string; before: string | null } | null>(null);
  const dragged = useRef(false);
  const listRef = useRef<HTMLElement>(null);
  const attempt = async (run: () => Promise<unknown>): Promise<boolean> => {
    try {
      await run();
      return true;
    } catch (e) {
      onError((e as Error).message);
      return false;
    }
  };
  const all = useDropTarget((ids) => onMoveItems(ids, null));

  /** The folder a dragged one would land in front of, or null for the end of the list. */
  const landsBefore = (y: number, id: string): string | null => {
    const rows = [...(listRef.current?.querySelectorAll<HTMLElement>("[data-folder-id]") ?? [])];
    for (const row of rows) {
      const other = row.dataset.folderId;
      if (!other || other === id) continue;
      const box = row.getBoundingClientRect();
      if (y < box.top + box.height / 2) return other;
    }
    return null;
  };

  /** Whether moving `id` in front of `before` would change the order at all. */
  const wouldMove = (id: string, before: string | null): boolean => {
    const ids = folders.list.map((f) => f.id);
    return reorderedIds(ids, id, before).join() !== ids.join();
  };

  const startReorder = (e: React.PointerEvent, id: string): void => {
    dragged.current = false;
    // Left button only, one pointer, and never from the row's own buttons.
    if (e.button !== 0 || !e.isPrimary || (e.target as HTMLElement).closest(".folder-actions")) return;
    const from = { x: e.clientX, y: e.clientY };
    let moving = false;
    let landing: string | null = null;
    const stop = (): void => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onCancel);
      window.removeEventListener("keydown", onKey);
      setDrag(null);
    };
    const onMove = (ev: PointerEvent): void => {
      if (!moving) {
        if (Math.hypot(ev.clientX - from.x, ev.clientY - from.y) < REORDER_SLOP) return;
        moving = true;
        // From here the press is a drag, so the click it would end in is dropped.
        dragged.current = true;
      }
      landing = landsBefore(ev.clientY, id);
      setDrag({ id, before: landing });
    };
    const onUp = (): void => {
      const move = moving && wouldMove(id, landing);
      stop();
      if (move) void attempt(() => folders.reorder(id, landing));
    };
    const onCancel = (): void => stop();
    const onKey = (ev: KeyboardEvent): void => {
      if (ev.key === "Escape") {
        moving = false;
        stop();
      }
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onCancel);
    window.addEventListener("keydown", onKey);
  };

  /** Alt+Up/Down moves the focused folder one place, for anyone not using a mouse. */
  const nudge = (id: string, direction: -1 | 1): void => {
    const target = nudgeTarget(folders.list.map((f) => f.id), id, direction);
    if (!target) return;
    void attempt(() => folders.reorder(id, target.before));
  };

  const rest = folders.list.filter((f) => f.id !== drag?.id);

  return (
    <aside className={`folder-list ${drag ? "reordering" : ""}`} aria-label="Folders" ref={listRef}>
      <button
        className={`folder-item ${view.kind === "all" ? "active" : ""} ${all.over ? "drop" : ""}`}
        onClick={() => onView({ kind: "all" })}
        title="Everything. Drop an item here to take it out of its folder."
        {...all.handlers}
      >
        <span className="folder-name">All</span>
      </button>
      <button className={`folder-item ${view.kind === "favorites" ? "active" : ""}`} onClick={() => onView({ kind: "favorites" })}>
        <span className="folder-name">★ Favorites</span>
      </button>

      <div className="folder-head">
        <span title="Drag a folder to change the order, or press Alt and the up or down arrow.">Folders</span>
        <button className="icon-button small" title="New folder" aria-label="New folder" onClick={() => setCreating(true)}>+</button>
      </div>
      {creating && (
        <NameInput
          initial=""
          placeholder="Folder name"
          onSubmit={async (name) => {
            if (await attempt(async () => onView({ kind: "folder", id: (await folders.create(name)).id }))) setCreating(false);
          }}
          onCancel={() => setCreating(false)}
        />
      )}
      {folders.list.length === 0 && !creating && <p className="folder-empty">No folders yet.</p>}
      {folders.list.map((f) => (
        <FolderRow
          key={f.id}
          folder={f}
          folders={folders}
          active={view.kind === "folder" && view.id === f.id}
          onSelect={() => onView({ kind: "folder", id: f.id })}
          onDropItems={(ids) => onMoveItems(ids, f.id)}
          reorder={{
            dragging: drag?.id === f.id,
            dropBefore: drag !== null && drag.before === f.id,
            dropAfter: drag !== null && drag.before === null && rest.at(-1)?.id === f.id,
            clicked: () => {
              // The pointer-up that ends a drag also lands as a click; not a selection.
              if (!dragged.current) return true;
              dragged.current = false;
              return false;
            },
            onPointerDown: (e) => startReorder(e, f.id),
            onNudge: (direction) => nudge(f.id, direction),
          }}
          onRename={(name) => attempt(() => folders.rename(f.id, name))}
          onDelete={() =>
            attempt(async () => {
              await folders.remove(f.id);
              // Its items are still in All; that is where the view goes too.
              if (view.kind === "folder" && view.id === f.id) onView({ kind: "all" });
            })
          }
          onError={onError}
        />
      ))}
    </aside>
  );
}

type RowMode = "idle" | "rename" | "confirm" | "menu" | "unlock" | "set-pin" | "change-pin" | "remove-pin";

/** How many blocks across a thumbnail is drawn with, per strength. */
const MOSAIC_BLOCKS: Record<Exclude<MosaicLevel, "off">, number> = { weak: 24, strong: 8 };

/**
 * A thumbnail as a mosaic.
 *
 * The thumbnail is drawn into a canvas a few blocks wide and shown at the tile's
 * full size, so the browser blows up those blocks with no smoothing (CSS
 * `image-rendering: pixelated`). Scaling the picture itself down and back up
 * with a transform was tried first and only softened it: the compositor smooths
 * what it scales, whatever the image is asked to render like.
 *
 * It is a cover for the grid, not protection: opening the item shows it as it
 * is, the prompt is still in its tooltip and in search, and nothing on disk
 * changes. Hidden and locked folders are what keep things out of view.
 */
function MosaicThumb({ src, ratio, level }: {
  src: string;
  ratio: number;
  level: Exclude<MosaicLevel, "off">;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const across = MOSAIC_BLOCKS[level];
  const down = Math.max(2, Math.round(across / (ratio > 0 ? ratio : 1.6)));

  useEffect(() => {
    const target = canvas.current;
    if (!target) return;
    let gone = false;
    const image = new Image();
    image.onload = () => {
      if (gone) return;
      const ctx = target.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, target.width, target.height);
      ctx.drawImage(image, 0, 0, target.width, target.height);
      target.dataset.drawn = "yes";
    };
    image.src = src;
    return () => {
      gone = true;
    };
  }, [src, across, down]);

  // Not described: a mosaic is there so that what the picture shows is not read out either.
  // The prompt is still the tile's tooltip, as it is for every other item.
  return <canvas className="mosaic" ref={canvas} width={across} height={down} role="img" aria-label="Behind a mosaic" />;
}

/** What a row needs to take part in reordering; see FolderList's startReorder. */
interface Reorder {
  /** This row is the one being dragged. */
  dragging: boolean;
  /** The dragged folder would land in front of this row. */
  dropBefore: boolean;
  /** It would land at the end, after this (the last) row. */
  dropAfter: boolean;
  /** False when the click only ends a drag and must not select the folder. */
  clicked: () => boolean;
  onPointerDown: (e: React.PointerEvent) => void;
  onNudge: (direction: -1 | 1) => void;
}

function FolderRow({ folder, folders, active, onSelect, onDropItems, onRename, onDelete, onError, reorder }: {
  folder: GalleryFolder;
  folders: Folders;
  active: boolean;
  onSelect: () => void;
  onDropItems: (ids: string[]) => void;
  onRename: (name: string) => Promise<boolean>;
  onDelete: () => Promise<boolean>;
  onError: (message: string) => void;
  reorder: Reorder;
}) {
  const [mode, setMode] = useState<RowMode>("idle");
  const lockAllKeys = useShortcuts().lockFolders;
  const drop = useDropTarget(onDropItems);
  const closed = folder.locked && !folder.unlocked;
  const menuRef = useRef<HTMLDivElement>(null);

  // The options menu closes on a click anywhere else, or Escape.
  useEffect(() => {
    if (mode !== "menu") return;
    const away = (e: PointerEvent) => {
      if (!menuRef.current?.contains(e.target as Node)) setMode("idle");
    };
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMode("idle");
    };
    document.addEventListener("pointerdown", away);
    document.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("pointerdown", away);
      document.removeEventListener("keydown", key);
    };
  }, [mode]);
  const run = (action: () => Promise<unknown>) => () =>
    void action().then(() => setMode("idle"), (e: Error) => onError(e.message));

  if (mode === "unlock") {
    return (
      <PinForm
        title={`Unlock “${folder.name}”`}
        fields={["PIN"]}
        submitLabel="Unlock"
        onSubmit={async ([pin = ""]) => {
          await folders.unlock(folder.id, pin);
          setMode("idle");
          onSelect();
        }}
        onCancel={() => setMode("idle")}
      />
    );
  }
  if (mode === "set-pin" || mode === "change-pin") {
    const change = mode === "change-pin";
    return (
      <PinForm
        title={change ? `New PIN for “${folder.name}”` : `Lock “${folder.name}” with a PIN`}
        fields={change ? ["Current PIN", "New PIN", "New PIN again"] : ["PIN", "PIN again"]}
        submitLabel={change ? "Change PIN" : "Lock"}
        note="Six digits. There is no way to recover a forgotten PIN."
        onSubmit={async (values) => {
          const [current = null, pin = "", again = ""] = change ? values : [null, ...values];
          if (pin !== again) throw new Error("The two PINs are not the same.");
          await folders.setPin(folder.id, pin, current);
          setMode("idle");
        }}
        onCancel={() => setMode("idle")}
      />
    );
  }
  if (mode === "remove-pin") {
    return (
      <PinForm
        title={`Remove the PIN of “${folder.name}”`}
        fields={["Current PIN"]}
        submitLabel="Remove PIN"
        onSubmit={async ([current = ""]) => {
          await folders.setPin(folder.id, null, current);
          setMode("idle");
        }}
        onCancel={() => setMode("idle")}
      />
    );
  }
  if (mode === "menu") {
    return (
      <div className="folder-menu" role="menu" aria-label={`Options for ${folder.name}`} ref={menuRef}>
        <p className="folder-menu-title" title={folder.name}>{folder.name}</p>
        {!closed && (
          <button role="menuitem" className="ghost small" onClick={() => setMode("rename")}>Rename…</button>
        )}
        <button role="menuitem" className="ghost small" onClick={run(() => folders.setHidden(folder.id, !folder.hidden))}>
          {folder.hidden ? "Show in the folder list" : "Hide"}
        </button>
        {!folder.locked && (
          <button role="menuitem" className="ghost small" onClick={() => setMode("set-pin")}>Lock with a PIN…</button>
        )}
        {folder.unlocked && (
          <>
            <button role="menuitem" className="ghost small" onClick={run(() => folders.lock(folder.id))}>Lock now</button>
            <button role="menuitem" className="ghost small" onClick={() => setMode("change-pin")}>Change PIN…</button>
            <button role="menuitem" className="ghost small" onClick={() => setMode("remove-pin")}>Remove PIN…</button>
          </>
        )}
        {!closed && (
          <button role="menuitem" className="ghost small danger-text" onClick={() => setMode("confirm")}>Delete folder…</button>
        )}
        {closed && <p className="folder-menu-note">Unlock it to rename, delete or change its PIN.</p>}
        <button className="ghost small" onClick={() => setMode("idle")}>Close</button>
      </div>
    );
  }
  if (mode === "rename") {
    return (
      <NameInput
        initial={folder.name}
        placeholder="Folder name"
        onSubmit={async (name) => {
          if (name === folder.name || (await onRename(name))) setMode("idle");
        }}
        onCancel={() => setMode("idle")}
      />
    );
  }
  if (mode === "confirm") {
    return (
      <div className="folder-confirm" role="alertdialog" aria-label={`Delete folder ${folder.name}`}>
        <p>
          Delete “{folder.name}”?{" "}
          {folder.count ? `Its ${folder.count} item${folder.count === 1 ? "" : "s"} stay in All.` : "It is empty."}
        </p>
        <div className="row-gap">
          <button className="danger small" onClick={() => void onDelete().then((ok) => ok || setMode("idle"))}>Delete folder</button>
          <button className="ghost small" onClick={() => setMode("idle")}>Cancel</button>
        </div>
      </div>
    );
  }
  const marks = [
    active ? "active" : "",
    drop.over ? "drop" : "",
    folder.hidden ? "hidden-folder" : "",
    reorder.dragging ? "dragging" : "",
    reorder.dropBefore ? "drop-before" : "",
    reorder.dropAfter ? "drop-after" : "",
  ].join(" ");
  return (
    <div className={`folder-row ${marks}`} data-folder-id={folder.id} onPointerDown={reorder.onPointerDown} {...drop.handlers}>
      <button
        className="folder-item"
        onClick={() => {
          if (!reorder.clicked()) return;
          if (closed) setMode("unlock");
          else onSelect();
        }}
        onDoubleClick={closed ? undefined : () => setMode("rename")}
        onKeyDown={(e) => {
          if (!e.altKey || (e.key !== "ArrowUp" && e.key !== "ArrowDown")) return;
          e.preventDefault();
          reorder.onNudge(e.key === "ArrowUp" ? -1 : 1);
        }}
        title={closed ? `${folder.name} (locked)` : folder.name}
      >
        <span className="folder-name">{folder.name}</span>
        {folder.hidden && <span className="folder-tag">hidden</span>}
        {closed ? (
          <span className="folder-lock" aria-label="Locked">🔒</span>
        ) : (
          <span className="folder-count">{folder.count}</span>
        )}
      </button>
      <span className="folder-actions">
        {/* A folder with a PIN that is open for now locks again in one click. */}
        {folder.unlocked && (
          <button className="icon-button small" title={`Lock now (${lockAllKeys} locks every open folder)`} aria-label={`Lock ${folder.name} now`} onClick={run(() => folders.lock(folder.id))}>
            🔒
          </button>
        )}
        <button className="icon-button small" title="More" aria-label={`Options for ${folder.name}`} onClick={() => setMode("menu")}>⋯</button>
      </span>
    </div>
  );
}

/** Six-digit PIN fields, shown in place of a folder row. Errors (a wrong PIN) stay in the form. */
function PinForm({ title, fields, submitLabel, note, onSubmit, onCancel }: {
  title: string;
  fields: string[];
  submitLabel: string;
  note?: string;
  onSubmit: (values: string[]) => Promise<void>;
  onCancel: () => void;
}) {
  const [values, setValues] = useState<string[]>(() => fields.map(() => ""));
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const complete = values.every((v) => FOLDER_PIN.test(v));
  return (
    <form
      className="folder-pin"
      aria-label={title}
      onSubmit={(e) => {
        e.preventDefault();
        if (!complete || busy) return;
        setBusy(true);
        setError(null);
        void onSubmit(values)
          .catch((err: Error) => {
            setError(err.message);
            setValues(fields.map(() => ""));
          })
          .finally(() => setBusy(false));
      }}
      onKeyDown={(e) => {
        if (e.key === "Escape") onCancel();
      }}
    >
      <p className="folder-pin-title" title={title}>{title}</p>
      {fields.map((label, i) => (
        <input
          key={label}
          className="input pin-input"
          type="password"
          inputMode="numeric"
          autoComplete="off"
          autoFocus={i === 0}
          maxLength={6}
          placeholder={label}
          aria-label={label}
          value={values[i]}
          onChange={(e) => {
            const digits = e.target.value.replace(/[^0-9]/g, "").slice(0, 6);
            setValues((prev) => prev.map((v, j) => (j === i ? digits : v)));
          }}
        />
      ))}
      {note && <p className="folder-pin-note">{note}</p>}
      {error && <p className="error-text small" role="alert">{error}</p>}
      <div className="row-gap">
        <button type="submit" className="small" disabled={!complete || busy}>{busy ? "…" : submitLabel}</button>
        <button type="button" className="ghost small" onClick={onCancel}>Cancel</button>
      </div>
    </form>
  );
}

function NameInput({ initial, placeholder, onSubmit, onCancel }: {
  initial: string;
  placeholder: string;
  onSubmit: (name: string) => void | Promise<void>;
  onCancel: () => void;
}) {
  const [value, setValue] = useState(initial);
  // Enter submits, and the input then losing focus must not submit a second time.
  const done = useRef(false);
  const submit = (): void => {
    const name = value.trim();
    if (done.current) return;
    if (!name || name === initial) {
      done.current = true;
      onCancel();
      return;
    }
    done.current = true;
    void Promise.resolve(onSubmit(name)).finally(() => (done.current = false));
  };
  return (
    <form
      className="folder-name-input"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <input
        className="input"
        autoFocus
        value={value}
        maxLength={FOLDER_NAME_MAX}
        placeholder={placeholder}
        aria-label={placeholder}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Escape") {
            done.current = true;
            onCancel();
          }
        }}
        onBlur={submit}
      />
    </form>
  );
}
