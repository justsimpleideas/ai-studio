import { useEffect, useState } from "react";
import { studioMediaUrl } from "@shared/image.js";
import { api } from "../api";
import { formatBytes, formatDate, formatDuration, formatSeconds } from "../format";
import type { GalleryFolder, GifOptions, ImageRecord } from "../types";
import { GifExport } from "./GifExport";
import { VideoPlayer } from "./VideoPreview";

/** "org/model@f332072aa78b…" shortened to "org/model@f332072". */
function shortRevision(revision: string): string {
  const at = revision.lastIndexOf("@");
  return at < 0 ? revision : `${revision.slice(0, at)}@${revision.slice(at + 1, at + 8)}`;
}

interface Props {
  image: ImageRecord;
  list: ImageRecord[];
  onNavigate: (image: ImageRecord) => void;
  onClose: () => void;
  onReuse: (image: ImageRecord) => void;
  onFavorite: (image: ImageRecord) => void;
  onDelete: (image: ImageRecord) => void;
  onSaveAs: (image: ImageRecord) => Promise<string | null>;
  /** Save a video as a GIF; the saved path, or null if cancelled. */
  onSaveAsGif: (video: ImageRecord, options: GifOptions) => Promise<string | null>;
  /** Make a video that starts from this image. */
  onAnimate: (image: ImageRecord) => void;
  /** Make a video that starts from this video's last frame. */
  onContinue: (video: ImageRecord) => void;
  folders: GalleryFolder[];
  /** Put it in a folder, or (null) in none. */
  onMove: (image: ImageRecord, folderId: string | null) => void;
}

export function ImageDetail({ image, list, onNavigate, onClose, onReuse, onFavorite, onDelete, onSaveAs, onSaveAsGif, onAnimate, onContinue, folders, onMove }: Props) {
  const [copied, setCopied] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [saved, setSaved] = useState<string | null>(null);
  const [gifOpen, setGifOpen] = useState(false);
  const [exportsKeepSettings, setExportsKeepSettings] = useState<boolean | null>(null);
  const index = list.findIndex((i) => i.id === image.id);
  const video = image.media === "video";
  const provenance = image.provenance ?? {};
  const madeWith = [
    provenance.app,
    ...Object.entries(provenance.libraries ?? {}).map(([name, version]) => `${name} ${version}`),
  ].filter(Boolean);

  useEffect(() => {
    api.includeSettingsInExports().then(setExportsKeepSettings, () => setExportsKeepSettings(null));
  }, []);
  const prev = index > 0 ? list[index - 1] : undefined;
  const next = index >= 0 ? list[index + 1] : undefined;

  useEffect(() => {
    setConfirmDelete(false);
    setCopied(false);
    setSaved(null);
    setGifOpen(false);
  }, [image.id]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement) return;
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft" && prev) onNavigate(prev);
      if (e.key === "ArrowRight" && next) onNavigate(next);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [prev, next, onClose, onNavigate]);

  const copyPrompt = async () => {
    try {
      await navigator.clipboard.writeText(image.prompt);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal detail" role="dialog" aria-modal="true" aria-label={video ? "Video details" : "Image details"}>
        <div className="detail-image">
          {video ? <VideoPlayer video={image} /> : <img src={image.image_url} alt={image.prompt} />}
          {prev && (
            <button className="nav prev" onClick={() => onNavigate(prev)} aria-label="Previous">‹</button>
          )}
          {next && (
            <button className="nav next" onClick={() => onNavigate(next)} aria-label="Next">›</button>
          )}
        </div>
        <aside className="detail-side">
          <div className="detail-top">
            <button className="icon-button" onClick={onClose} aria-label="Close">✕</button>
          </div>
          <p className="detail-prompt">{image.prompt}</p>
          <label className="detail-folder">
            <span className="field-label">Folder</span>
            <select
              className="input"
              value={image.folder_id ?? ""}
              onChange={(e) => onMove(image, e.target.value || null)}
            >
              <option value="">No folder</option>
              {folders.map((f) => (
                // A locked folder takes items without its PIN; they then leave view until it is unlocked.
                <option key={f.id} value={f.id}>{f.locked && !f.unlocked ? `🔒 ${f.name}` : f.name}</option>
              ))}
            </select>
          </label>
          <dl className="meta">
            <dt>Size</dt><dd>{image.width} × {image.height}</dd>
            {video && (<><dt>Length</dt><dd>{formatSeconds(image.frames ?? 0, image.fps ?? 0)} · {image.frames} frames at {image.fps} fps</dd></>)}
            {video && image.start_image && (
              <>
                <dt>Starts from</dt>
                <dd><img className="start-thumb" src={studioMediaUrl(image.start_image, "input")} alt="The picture this video starts from" /></dd>
              </>
            )}
            <dt>Seed</dt><dd className="tabular">{image.seed}</dd>
            <dt>Steps</dt><dd>{image.steps}</dd>
            <dt>Guidance</dt><dd>{image.guidance}</dd>
            {image.negative_prompt && (<><dt>Negative</dt><dd>{image.negative_prompt}</dd></>)}
            <dt>Time</dt><dd>{formatDuration(image.duration_ms)}</dd>
            {image.stats.peak_vram_bytes ? (<><dt>Peak VRAM</dt><dd>{formatBytes(image.stats.peak_vram_bytes)}</dd></>) : null}
            {image.stats.memory_plan && (<><dt>Memory</dt><dd>{image.stats.memory_plan}</dd></>)}
            {image.stats.encoder && (<><dt>Format</dt><dd>{image.stats.encoder}</dd></>)}
            <dt>Model</dt>
            <dd>
              {image.model}
              {provenance.model_revision && (
                <span className="muted small block mono" title={provenance.model_revision}>
                  {shortRevision(provenance.model_revision)}
                </span>
              )}
            </dd>
            {madeWith.length > 0 && (<><dt>Made with</dt><dd className="small">{madeWith.join(", ")}</dd></>)}
            <dt>Created</dt><dd>{formatDate(image.created)}</dd>
            <dt>File</dt><dd className="mono small">{image.file_name}</dd>
          </dl>
          <div className="detail-actions">
            <button className="primary" onClick={() => onReuse(image)}>Use these settings</button>
            {video ? (
              <button onClick={() => onContinue(image)} title="A new clip that starts where this one ends">Continue from last frame</button>
            ) : (
              <button onClick={() => onAnimate(image)} title="A video that starts from this image">Animate</button>
            )}
            <button onClick={copyPrompt}>{copied ? "Copied" : "Copy prompt"}</button>
            <button onClick={() => onFavorite(image)} aria-pressed={image.favorite}>
              {image.favorite ? "★ Favorite" : "☆ Favorite"}
            </button>
            <button onClick={() => void onSaveAs(image).then((p) => p && setSaved(p)).catch(() => setSaved(null))}>
              {saved ? "Saved" : "Save as…"}
            </button>
            {video && (
              <button onClick={() => setGifOpen((open) => !open)} aria-expanded={gifOpen}>Save as GIF…</button>
            )}
            {video && gifOpen && (
              <GifExport key={image.id} video={image} onSave={onSaveAsGif} onClose={() => setGifOpen(false)} />
            )}
            {confirmDelete ? (
              <button className="danger" onClick={() => onDelete(image)}>Delete for good</button>
            ) : (
              <button className="ghost danger-text" onClick={() => setConfirmDelete(true)}>Delete</button>
            )}
          </div>
          {video ? (
            <p className="hint export-note">Videos are saved without the prompt or settings inside the file.</p>
          ) : exportsKeepSettings !== null && (
            <p className="hint export-note">
              {exportsKeepSettings
                ? "“Save as…” keeps the prompt and settings inside the file."
                : "“Save as…” saves the picture only, without the prompt and settings."}
            </p>
          )}
        </aside>
      </div>
    </div>
  );
}
