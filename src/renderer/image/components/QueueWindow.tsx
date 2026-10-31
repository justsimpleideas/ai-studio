import { useEffect, useState } from "react";
import { stageLabel } from "../format";
import { describeJob, draftFrom, isEditable, queueOrder, requestFrom, type QueueDraft } from "../queue";
import { ASPECT_RATIOS, RESOLUTIONS, closestRatio, sizeFor } from "../sizes";
import type { GenerationRequest, ImageProfile, Job, VideoInfo, VideoRequest } from "../types";
import { QUALITIES, SHAPES, geometryFor, qualityOf, shapeOf, sizeFor as videoSizeFor, type Shape } from "../videoForm";

interface Props {
  jobs: Record<string, Job>;
  imageProfile: ImageProfile | null;
  video: VideoInfo | undefined;
  openOnGenerate: boolean;
  onOpenOnGenerate: (open: boolean) => void;
  onCancel: (job: Job) => Promise<void>;
  onUpdate: (job: Job, request: GenerationRequest | VideoRequest) => Promise<void>;
  onClose: () => void;
}

const MAX_SEED = 2 ** 32 - 1;

/** 2 → "2nd": the place in line after the next one. */
function ordinal(n: number): string {
  const teen = n % 100 >= 11 && n % 100 <= 13;
  return `${n}${teen ? "th" : ["th", "st", "nd", "rd"][n % 10] ?? "th"}`;
}
const SIX_B = { steps: 8, guidance: 0, distilled: true };

/**
 * Everything waiting for the GPU, in the order it runs. Each image or video in line can
 * be changed or cancelled; the one being made can be cancelled, or changed, which stops
 * it and starts it again first in line.
 */
export function QueueWindow({ jobs, imageProfile, video, openOnGenerate, onOpenOnGenerate, onCancel, onUpdate, onClose }: Props) {
  const list = queueOrder(jobs);
  const [editing, setEditing] = useState<string | null>(null);
  const editingJob = list.find((j) => j.id === editing && isEditable(j)) ?? null;

  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      if (e.key !== "Escape") return;
      if (editing) setEditing(null);
      else onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [editing, onClose]);

  let place = 0;
  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal queue-window" role="dialog" aria-modal="true" aria-label="Queue">
        <header className="queue-top">
          <h2>Queue</h2>
          <span className="muted small">{list.length === 0 ? "Nothing waiting" : `${list.length} not done yet`}</span>
          <button type="button" className="ghost small queue-close" onClick={onClose} aria-label="Close">
            Close
          </button>
        </header>
        {list.length === 0 ? (
          <p className="queue-empty muted">Everything asked for is done. New images and videos line up here.</p>
        ) : (
          <ol className="queue-list">
            {list.map((job) => {
              const running = job.status === "running";
              if (!running) place += 1;
              return (
                <QueueRow
                  key={job.id}
                  job={job}
                  label={running ? (job.cancelling ? "Stopping" : "Making now") : place === 1 ? "Next" : `${ordinal(place)} in line`}
                  editing={editingJob?.id === job.id}
                  onEdit={() => setEditing(job.id)}
                  onCancel={() => onCancel(job)}
                  editor={
                    editingJob?.id === job.id ? (
                      <QueueEditor
                        job={job}
                        imageProfile={imageProfile}
                        video={video}
                        onDiscard={() => setEditing(null)}
                        onSave={async (request) => {
                          await onUpdate(job, request);
                          setEditing(null);
                        }}
                      />
                    ) : null
                  }
                />
              );
            })}
          </ol>
        )}
        <footer className="queue-foot">
          <label className="toggle">
            <input type="checkbox" checked={openOnGenerate} onChange={(e) => onOpenOnGenerate(e.target.checked)} />
            <span>Open this window after every Generate</span>
          </label>
        </footer>
      </div>
    </div>
  );
}

function QueueRow({ job, label, editing, onEdit, onCancel, editor }: {
  job: Job;
  label: string;
  editing: boolean;
  onEdit: () => void;
  onCancel: () => Promise<void>;
  editor: React.ReactNode;
}) {
  const [error, setError] = useState<string | null>(null);
  const running = job.status === "running";
  const task = job.kind === "task";
  const kind = job.kind === "video" ? "video" : "image";
  const prompt = "prompt" in job.request ? job.request.prompt : "label" in job.request ? job.request.label : "";
  const fraction = job.steps > 0 ? Math.min(1, (job.index + job.step / job.steps) / Math.max(1, job.count)) : 0;

  return (
    <li className={`queue-row ${running ? "running" : ""}`} data-job={job.id}>
      <div className="queue-row-head">
        <span className="queue-place">{label}</span>
        <span className="tag">{task ? "Training" : kind === "video" ? "Video" : "Image"}</span>
        {running && !task && (
          <span className="muted small tabular">{stageLabel(job.stage, job.step, job.steps, job.index, job.count, kind)}</span>
        )}
      </div>
      <p className="queue-prompt" title={prompt}>{prompt}</p>
      {!task && <span className="muted small queue-summary">{describeJob(job)}</span>}
      {running && !task && (
        <div className={`bar ${job.stage === "sampling" || job.stage === "decoding" ? "" : "indeterminate"}`}>
          <div style={{ width: job.stage === "sampling" || job.stage === "decoding" ? `${fraction * 100}%` : undefined }} />
        </div>
      )}
      {task ? (
        <span className="hint">Training is stopped from the Train tab.</span>
      ) : (
        isEditable(job) && !editing && (
          <div className="queue-actions">
            <button type="button" className="small" onClick={onEdit}>Edit</button>
            <button
              type="button"
              className="ghost small"
              onClick={() => {
                setError(null);
                onCancel().catch((e: Error) => setError(e.message));
              }}
            >
              {running ? "Cancel this one" : "Remove from queue"}
            </button>
          </div>
        )
      )}
      {error && <div className="error-text small" role="alert">{error}</div>}
      {editor}
    </li>
  );
}

function QueueEditor({ job, imageProfile, video, onSave, onDiscard }: {
  job: Job;
  imageProfile: ImageProfile | null;
  video: VideoInfo | undefined;
  onSave: (request: GenerationRequest | VideoRequest) => Promise<void>;
  onDiscard: () => void;
}) {
  const [draft, setDraft] = useState<QueueDraft>(() => draftFrom(job));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const set = <K extends keyof QueueDraft>(key: K, value: QueueDraft[K]) => setDraft((d) => ({ ...d, [key]: value }));
  const isVideo = draft.kind === "video";
  const running = job.status === "running";

  const model = imageProfile ?? SIX_B;
  const videoProfile = video?.profile ?? null;
  const fixed = isVideo && videoProfile?.fixed === true;
  const geometry = geometryFor(videoProfile);
  const negativeOn = isVideo ? draft.guidance > 1 : draft.guidance > 0;

  const save = async (): Promise<void> => {
    if (!draft.prompt.trim()) return;
    setSaving(true);
    setError(null);
    try {
      await onSave(requestFrom(draft, job));
    } catch (e) {
      setError((e as Error).message);
      setSaving(false);
    }
  };

  return (
    <form
      className="queue-editor"
      onSubmit={(e) => {
        e.preventDefault();
        void save();
      }}
    >
      <label className="field">
        <span className="field-label">Prompt</span>
        <textarea className="prompt" rows={4} value={draft.prompt} autoFocus onChange={(e) => set("prompt", e.target.value)} />
      </label>

      {isVideo ? (
        <VideoSize draft={draft} geometry={geometry} onChange={(size) => setDraft((d) => ({ ...d, ...size }))} />
      ) : (
        <ImageSize draft={draft} onChange={(size) => setDraft((d) => ({ ...d, ...size }))} />
      )}

      {isVideo && (
        <div className="field">
          <span className="field-label">Length</span>
          <div className="segmented" role="radiogroup" aria-label="Length">
            {geometry.lengths.map((l) => (
              <button
                type="button"
                key={l.frames}
                role="radio"
                aria-checked={draft.frames === l.frames}
                className={draft.frames === l.frames ? "active" : ""}
                onClick={() => set("frames", l.frames)}
              >
                {l.seconds} s
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="field">
        <span className="field-label">
          {isVideo ? "Videos" : "Images"} <span className="value">{draft.count}</span>
        </span>
        <input
          type="range"
          aria-label={isVideo ? "Videos" : "Images"}
          min={1}
          max={isVideo ? video?.limits.max_count ?? 4 : 8}
          value={draft.count}
          onChange={(e) => set("count", Number(e.target.value))}
        />
      </div>

      {!fixed && (
        <div className="queue-pair">
          <div className="field">
            <span className="field-label">
              Steps <span className="value">{draft.steps}</span>
            </span>
            <input
              type="range"
              aria-label="Steps"
              min={1}
              max={isVideo ? video?.limits.max_steps ?? 100 : model.distilled ? 30 : 50}
              value={draft.steps}
              onChange={(e) => set("steps", Number(e.target.value))}
            />
          </div>
          <div className="field">
            <span className="field-label">
              Guidance <span className="value">{draft.guidance.toFixed(1)}</span>
            </span>
            <input
              type="range"
              aria-label="Guidance"
              min={0}
              max={10}
              step={0.5}
              value={draft.guidance}
              onChange={(e) => set("guidance", Number(e.target.value))}
            />
          </div>
        </div>
      )}

      {!fixed && (
        <label className="field">
          <span className="field-label">Negative prompt</span>
          <textarea
            className="input"
            rows={2}
            value={draft.negative_prompt}
            disabled={!negativeOn}
            placeholder={negativeOn ? "" : `Only used when guidance is above ${isVideo ? 1 : 0}`}
            onChange={(e) => set("negative_prompt", e.target.value)}
          />
        </label>
      )}

      <div className="field">
        <span className="field-label">Seed</span>
        <div className="seed-row">
          <label className="toggle">
            <input type="checkbox" checked={draft.randomSeed} onChange={(e) => set("randomSeed", e.target.checked)} />
            <span>New random</span>
          </label>
          <input
            type="number"
            className="input seed-input"
            aria-label="Seed"
            min={0}
            max={MAX_SEED}
            value={draft.seed}
            disabled={draft.randomSeed}
            onChange={(e) => {
              const v = Number(e.target.value);
              if (Number.isInteger(v) && v >= 0 && v <= MAX_SEED) set("seed", v);
            }}
          />
        </div>
      </div>

      {running && (
        <span className="hint warn-text queue-restart-note">
          Saving stops this one and starts it again, first in line, with the changes.
          {job.images.length > 0 ? " What it already finished stays in the gallery." : ""}
        </span>
      )}
      {error && <div className="error-text small" role="alert">{error}</div>}
      <div className="queue-actions">
        <button type="submit" className="primary small" disabled={saving || !draft.prompt.trim()}>
          {saving ? "Saving…" : running ? "Save and restart" : "Save changes"}
        </button>
        <button type="button" className="ghost small" onClick={onDiscard}>Discard</button>
      </div>
    </form>
  );
}

function ImageSize({ draft, onChange }: { draft: QueueDraft; onChange: (size: { width: number; height: number }) => void }) {
  const ratio = closestRatio(draft.width, draft.height);
  const mp = (draft.width * draft.height) / 2 ** 20;
  const megapixels = RESOLUTIONS.reduce((a, b) => (Math.abs(b.megapixels - mp) < Math.abs(a.megapixels - mp) ? b : a)).megapixels;
  return (
    <>
      <div className="field">
        <span className="field-label">
          Aspect ratio <span className="value">{draft.width} × {draft.height}</span>
        </span>
        <div className="chips" role="radiogroup" aria-label="Aspect ratio">
          {ASPECT_RATIOS.map((r) => (
            <button
              type="button"
              key={r.label}
              role="radio"
              aria-checked={r.label === ratio.label}
              className={`chip ${r.label === ratio.label ? "active" : ""}`}
              onClick={() => onChange(sizeFor(r, megapixels))}
            >
              <span className="ratio-glyph" style={{ aspectRatio: `${r.w} / ${r.h}` }} aria-hidden />
              {r.label}
            </button>
          ))}
        </div>
      </div>
      <div className="field">
        <span className="field-label">Resolution</span>
        <div className="segmented" role="radiogroup" aria-label="Resolution">
          {RESOLUTIONS.map((r) => (
            <button
              type="button"
              key={r.label}
              role="radio"
              aria-checked={megapixels === r.megapixels}
              className={megapixels === r.megapixels ? "active" : ""}
              onClick={() => onChange(sizeFor(ratio, r.megapixels))}
            >
              {r.label}
            </button>
          ))}
        </div>
      </div>
    </>
  );
}

function VideoSize({ draft, geometry, onChange }: {
  draft: QueueDraft;
  geometry: ReturnType<typeof geometryFor>;
  onChange: (size: { width: number; height: number }) => void;
}) {
  const shape = shapeOf(draft.width, draft.height);
  const quality = qualityOf(draft.width, draft.height, geometry);
  return (
    <>
      <div className="field">
        <span className="field-label">
          Shape <span className="value">{draft.width} × {draft.height}</span>
        </span>
        <div className="chips three" role="radiogroup" aria-label="Shape">
          {SHAPES.map((s) => (
            <button
              type="button"
              key={s.value}
              role="radio"
              aria-checked={shape === s.value}
              className={`chip ${shape === s.value ? "active" : ""}`}
              onClick={() => onChange(videoSizeFor(s.value as Shape, quality, geometry))}
            >
              <span className="ratio-glyph" style={{ aspectRatio: `${s.w} / ${s.h}` }} aria-hidden />
              {s.label}
            </button>
          ))}
        </div>
      </div>
      <div className="field">
        <span className="field-label">Size</span>
        <div className="segmented" role="radiogroup" aria-label="Size">
          {QUALITIES.map((q, i) => (
            <button
              type="button"
              key={q.label}
              role="radio"
              aria-checked={quality === i}
              className={quality === i ? "active" : ""}
              onClick={() => onChange(videoSizeFor(shape, i, geometry))}
            >
              {q.label}
            </button>
          ))}
        </div>
      </div>
    </>
  );
}
