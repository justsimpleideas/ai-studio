import { useEffect, useRef } from "react";
import { digitOf, numberedFolders } from "../selection";
import type { GalleryFolder } from "../types";

interface Props {
  folders: GalleryFolder[];
  /** The folder the item is in now (one item), or undefined when moving several. */
  current?: string | null;
  /** What is being moved, for the heading: "this image", "3 items". */
  subject: string;
  onPick: (folderId: string | null) => void;
  onClose: () => void;
}

/**
 * A list of folders numbered for the keyboard: 0 is "No folder", 1 to 9 the first nine
 * folders in the order of the folder list. A number key picks, Escape closes; folders
 * after the ninth are there to click. Opened by the "Move to a folder" shortcut.
 */
export function FolderPicker({ folders, current, subject, onPick, onClose }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const entries = numberedFolders(folders);

  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === "Escape") {
        e.preventDefault();
        e.stopImmediatePropagation();
        onClose();
        return;
      }
      if (e.ctrlKey || e.altKey || e.metaKey) return;
      const digit = digitOf(e);
      if (digit === null) return;
      const entry = entries.find((x) => x.key === digit);
      if (!entry) return;
      e.preventDefault();
      e.stopImmediatePropagation();
      onPick(entry.id);
    };
    // Capture, so a number never reaches a text box or the window's own keys while this is open.
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [entries, onPick, onClose]);

  useEffect(() => {
    const away = (e: MouseEvent): void => {
      if (!ref.current?.contains(e.target as Node)) onClose();
    };
    document.addEventListener("mousedown", away);
    return () => document.removeEventListener("mousedown", away);
  }, [onClose]);

  return (
    <div className="folder-picker" role="menu" aria-label={`Move ${subject} to a folder`} ref={ref}>
      <p className="folder-picker-title">Move {subject} to</p>
      {entries.map((entry) => (
        <button
          key={entry.id ?? "none"}
          type="button"
          role="menuitem"
          className={`ghost small ${entry.id === current && current !== undefined ? "current" : ""}`}
          aria-current={current !== undefined && entry.id === current ? "true" : undefined}
          onClick={() => onPick(entry.id)}
          title={entry.name}
        >
          <kbd className={entry.key === null ? "blank" : ""}>{entry.key ?? ""}</kbd>
          <span className="folder-name">{entry.name}</span>
        </button>
      ))}
      <p className="hint">Press a number, or Esc to close.</p>
    </div>
  );
}
