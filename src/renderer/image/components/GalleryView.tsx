import { useCallback, useEffect, useRef, useState } from "react";
import { FOLDER_NAME_MAX, FOLDER_PIN } from "@shared/image.js";
import { matchesShortcut } from "@shared/shortcuts.js";
import { useShortcuts, useShortcutsRef } from "../../useShortcuts";
import { saveGalleryView, startView, type GalleryStart, type GalleryViewKind } from "../display";
import { api } from "../api";
import { formatSeconds } from "../format";
import type { GalleryFolder, ImageRecord, MediaKind } from "../types";
import type { Folders } from "../useFolders";

const PAGE = 60;
const FOLDERS_KEY = "lis.gallery.folders.v1";
/** What a tile carries while it is dragged onto a folder. Not a file, so the file drop zones ignore it. */
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
  /** Delete an item for good (already confirmed). */
  onDelete: (image: ImageRecord) => Promise<void>;
  /** Changes when every open PIN folder was just locked at once (a shortcut, Ctrl+Shift+L by default). */
  lockedAll?: number;
  /** Open on All, Favorites or a chosen folder each time, or on the view last open (Settings). */
  start?: GalleryStart;
}

export function GalleryView({ version, folders, onOpen, registerUpdater, onError, onMove, onDelete, lockedAll = 0, start = "all" }: Props) {
  const [items, setItems] = useState<ImageRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [query, setQuery] = useState("");
  const [view, setView] = useState<GalleryViewKind>(() => startView(start));
  // Kept either way, so switching to "where it was left" picks up from now.
  useEffect(() => saveGalleryView(view), [view]);
  const [media, setMedia] = useState<MediaKind | null>(null);
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

  const load = useCallback(async (offset: number) => {
    if (viewGone) return;
    const id = ++request.current;
    setLoading(true);
    try {
      const page = await api.gallery({
        offset,
        limit: PAGE,
        q: query.trim(),
        favorites: view.kind === "favorites",
        ...(view.kind === "folder" ? { folder: view.id } : {}),
        ...(media ? { media } : {}),
      });
      if (id !== request.current) return;
      setTotal(page.total);
      setItems((prev) => (offset === 0 ? page.items : [...prev, ...page.items]));
      setError(null);
    } catch (e) {
      if (id === request.current) setError((e as Error).message);
    } finally {
      if (id === request.current) setLoading(false);
    }
  }, [query, view, media, viewGone]);

  useEffect(() => {
    const t = window.setTimeout(() => load(0), query ? 200 : 0);
    return () => window.clearTimeout(t);
  }, [load, version, query]);

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

  const heading = view.kind === "all" ? "All" : view.kind === "favorites" ? "Favorites" : folder?.name ?? "";
  const noun = media ?? "item";

  return (
    <section className={`gallery-view ${showFolders ? "" : "folders-hidden"}`} ref={sectionRef}>
      {showFolders && <FolderList view={view} onView={setView} folders={folders} onError={onError} />}
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
          <span className="muted small">{total} {noun}{total === 1 ? "" : "s"}</span>
        </div>
        {error && <div className="error-text">{error}</div>}
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
        <div className="gallery-grid">
          {items.map((img) => (
            <button
              key={img.id}
              className="tile"
              data-id={img.id}
              style={{ "--ar": img.width > 0 && img.height > 0 ? img.width / img.height : 1.6 } as React.CSSProperties}
              draggable
              onDragStart={(e) => {
                e.dataTransfer.setData(ITEM_DRAG_TYPE, img.id);
                e.dataTransfer.effectAllowed = "move";
              }}
              onClick={() => onOpen(img, items)}
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
              <img src={img.thumb_url} alt={img.prompt} loading="lazy" style={{ aspectRatio: `${img.width} / ${img.height}` }} />
              {img.favorite && <span className="badge" aria-label="Favorite">★</span>}
              {img.media === "video" && (
                <span className="video-badge" aria-label="Video">▶ {formatSeconds(img.frames ?? 0, img.fps ?? 0)}</span>
              )}
            </button>
          ))}
        </div>
        {items.length < total && (
          <div className="center">
            <button onClick={() => load(items.length)} disabled={loading}>{loading ? "Loading…" : "Load more"}</button>
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

/** A place a dragged tile can be dropped on: a folder, or "All" to take it out of its folder. */
function useDropTarget(onItem: (id: string) => void) {
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
        const id = e.dataTransfer.getData(ITEM_DRAG_TYPE);
        if (id) onItem(id);
      },
    },
  };
}

function FolderList({ view, onView, folders, onError }: {
  view: GalleryViewKind;
  onView: (view: GalleryViewKind) => void;
  folders: Folders;
  onError: (message: string) => void;
}) {
  const [creating, setCreating] = useState(false);
  const attempt = async (run: () => Promise<unknown>): Promise<boolean> => {
    try {
      await run();
      return true;
    } catch (e) {
      onError((e as Error).message);
      return false;
    }
  };
  const all = useDropTarget((id) => void attempt(() => folders.move(id, null)));

  return (
    <aside className="folder-list" aria-label="Folders">
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
        <span>Folders</span>
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
          onDropItem={(id) => void attempt(() => folders.move(id, f.id))}
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

function FolderRow({ folder, folders, active, onSelect, onDropItem, onRename, onDelete, onError }: {
  folder: GalleryFolder;
  folders: Folders;
  active: boolean;
  onSelect: () => void;
  onDropItem: (id: string) => void;
  onRename: (name: string) => Promise<boolean>;
  onDelete: () => Promise<boolean>;
  onError: (message: string) => void;
}) {
  const [mode, setMode] = useState<RowMode>("idle");
  const lockAllKeys = useShortcuts().lockFolders;
  const drop = useDropTarget(onDropItem);
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
  return (
    <div className={`folder-row ${active ? "active" : ""} ${drop.over ? "drop" : ""} ${folder.hidden ? "hidden-folder" : ""}`} {...drop.handlers}>
      <button
        className="folder-item"
        onClick={closed ? () => setMode("unlock") : onSelect}
        onDoubleClick={closed ? undefined : () => setMode("rename")}
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
