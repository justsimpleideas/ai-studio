import { useCallback, useEffect, useRef, useState } from "react";
import { FOLDER_NAME_MAX } from "@shared/image.js";
import { api } from "../api";
import { formatSeconds } from "../format";
import type { GalleryFolder, ImageRecord, MediaKind } from "../types";
import type { Folders } from "../useFolders";

const PAGE = 60;
/** What a tile carries while it is dragged onto a folder. Not a file, so the file drop zones ignore it. */
export const ITEM_DRAG_TYPE = "application/x-studio-item";

export type GalleryViewKind = { kind: "all" } | { kind: "favorites" } | { kind: "folder"; id: string };

interface Props {
  version: number | string;
  folders: Folders;
  onOpen: (image: ImageRecord, list: ImageRecord[]) => void;
  /** Lets the parent push in-place updates (favorite, delete) without a refetch. */
  registerUpdater: (fn: (image: ImageRecord | { id: string; deleted: true }) => void) => void;
  onError: (message: string) => void;
}

export function GalleryView({ version, folders, onOpen, registerUpdater, onError }: Props) {
  const [items, setItems] = useState<ImageRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [query, setQuery] = useState("");
  const [view, setView] = useState<GalleryViewKind>({ kind: "all" });
  const [media, setMedia] = useState<MediaKind | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const request = useRef(0);

  const folder = view.kind === "folder" ? folders.list.find((f) => f.id === view.id) : undefined;

  const load = useCallback(async (offset: number) => {
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
  }, [query, view, media]);

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
    <section className="gallery-view">
      <FolderList view={view} onView={setView} folders={folders} onError={onError} />
      <div className="gallery-main">
        <div className="gallery-toolbar">
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
    </section>
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
        />
      ))}
    </aside>
  );
}

function FolderRow({ folder, active, onSelect, onDropItem, onRename, onDelete }: {
  folder: GalleryFolder;
  active: boolean;
  onSelect: () => void;
  onDropItem: (id: string) => void;
  onRename: (name: string) => Promise<boolean>;
  onDelete: () => Promise<boolean>;
}) {
  const [mode, setMode] = useState<"idle" | "rename" | "confirm">("idle");
  const drop = useDropTarget(onDropItem);

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
          {folder.count > 0 ? `Its ${folder.count} item${folder.count === 1 ? "" : "s"} stay in All.` : "It is empty."}
        </p>
        <div className="row-gap">
          <button className="danger small" onClick={() => void onDelete().then((ok) => ok || setMode("idle"))}>Delete folder</button>
          <button className="ghost small" onClick={() => setMode("idle")}>Cancel</button>
        </div>
      </div>
    );
  }
  return (
    <div className={`folder-row ${active ? "active" : ""} ${drop.over ? "drop" : ""}`} {...drop.handlers}>
      <button className="folder-item" onClick={onSelect} onDoubleClick={() => setMode("rename")} title={folder.name}>
        <span className="folder-name">{folder.name}</span>
        <span className="folder-count">{folder.count}</span>
      </button>
      <span className="folder-actions">
        <button className="icon-button small" title="Rename" aria-label={`Rename ${folder.name}`} onClick={() => setMode("rename")}>✎</button>
        <button className="icon-button small" title="Delete folder" aria-label={`Delete ${folder.name}`} onClick={() => setMode("confirm")}>✕</button>
      </span>
    </div>
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
