import { useEffect, useRef, useState } from "react";
import { formatDuration, formatRemaining, stageLabel } from "../format";
import type { ResultDisplay } from "../display";
import type { ImageRecord, Job } from "../types";

interface Props {
  job: Job | null;
  images: ImageRecord[];
  onOpen: (image: ImageRecord, list: ImageRecord[]) => void;
  onCancel: (job: Job) => void;
  modelMessage: string;
  starting: boolean;
  display: ResultDisplay;
}

export function Preview({ job, images, onOpen, onCancel, modelMessage, starting, display }: Props) {
  const latest = images[0] ?? null;
  const [selectedId, setSelectedId] = useState<string | null>(null);
  // With results off, the stage shows only what is picked from the strip.
  const selected = images.find((i) => i.id === selectedId) ?? (display.showResult ? latest : null);

  // A fresh result takes over the stage, or clears it when results are off.
  useEffect(() => setSelectedId(null), [latest?.id]);

  return (
    <section className="preview">
      <div className="stage">
        {selected ? (
          <button className="stage-image" onClick={() => onOpen(selected, images)} title="Open details">
            <img src={selected.image_url} alt={selected.prompt} width={selected.width} height={selected.height} />
          </button>
        ) : (
          <EmptyStage busy={!!job} starting={starting} done={!!latest} />
        )}
        {job && <Progress job={job} modelMessage={modelMessage} onCancel={() => onCancel(job)} />}
      </div>
      {display.showStrip && images.length > 0 && (
        <div className="filmstrip" aria-label="This session">
          {images.map((img) => (
            <button
              key={img.id}
              className={`film ${img.id === selected?.id ? "active" : ""}`}
              onClick={() => setSelectedId(img.id)}
              onDoubleClick={() => onOpen(img, images)}
              title={img.prompt}
            >
              <img src={img.thumb_url} alt="" loading="lazy" />
            </button>
          ))}
        </div>
      )}
    </section>
  );
}

function EmptyStage({ busy, starting, done }: { busy: boolean; starting: boolean; done: boolean }) {
  return (
    <div className="empty-stage">
      <svg viewBox="0 0 64 64" width="56" height="56" aria-hidden>
        <rect x="8" y="8" width="48" height="48" rx="6" fill="none" stroke="currentColor" strokeWidth="2.5" />
        <circle cx="24" cy="24" r="5" fill="currentColor" />
        <path d="M12 50l14-15 9 9 7-7 10 13z" fill="currentColor" />
      </svg>
      <p>
        {starting
          ? "Starting the image engine…"
          : busy
            ? "Your image is on its way."
            : done
              ? "Done. It is in the gallery."
              : "Describe an image and press Generate."}
      </p>
      <p className="muted small">
        {done && !busy ? "Showing results here is off in Settings." : "Everything runs on this computer."}
      </p>
    </div>
  );
}

/** Shown when time left would be at least this long; shorter jobs are over before it helps. */
const SHOW_REMAINING_MS = 20_000;

export function Progress({ job, modelMessage, onCancel }: { job: Job; modelMessage: string; onCancel: () => void }) {
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now() / 1000), 250);
    return () => window.clearInterval(t);
  }, []);

  // Time left comes from the steps seen so far for this image or video, so loading
  // the model and reading the prompt do not skew it.
  const key = `${job.id}:${job.index}`;
  const firstStep = useRef<{ key: string; at: number; step: number } | null>(null);
  useEffect(() => {
    if (job.stage === "sampling" && job.step > 0 && firstStep.current?.key !== key) {
      firstStep.current = { key, at: Date.now() / 1000, step: job.step };
    }
  }, [key, job.stage, job.step]);

  const kind = job.kind === "video" ? "video" : "image"; // this card shows image and video jobs only
  const perImage = job.steps > 0 ? job.step / job.steps : 0;
  const stageWeight = job.stage === "decoding" || job.stage === "encoding video" ? 1 : job.stage === "sampling" ? perImage : 0;
  const fraction = Math.min(1, (job.index + stageWeight) / Math.max(1, job.count));
  const indeterminate = job.stage === "loading model" || job.stage === "encoding prompt" || job.status === "queued";
  const label = job.stage === "loading model" && modelMessage ? modelMessage : stageLabel(job.stage, job.step, job.steps, job.index, job.count, kind);
  const elapsed = job.started ? (now - job.started) * 1000 : 0;

  let remaining = NaN;
  const seen = firstStep.current;
  if (seen && seen.key === key && job.stage === "sampling" && job.step > seen.step) {
    const perStep = (now - seen.at) / (job.step - seen.step);
    remaining = ((job.steps - job.step) + (job.count - job.index - 1) * job.steps) * perStep * 1000;
  }
  const left = remaining >= SHOW_REMAINING_MS ? ` · ${formatRemaining(remaining)}` : "";

  return (
    <div className="progress-card" role="status" aria-live="polite">
      <div className="progress-head">
        <span>{label}</span>
        <span className="muted tabular">{job.started ? `${formatDuration(elapsed)}${left}` : ""}</span>
      </div>
      <div className={`bar ${indeterminate ? "indeterminate" : ""}`}>
        <div style={{ width: indeterminate ? undefined : `${fraction * 100}%` }} />
      </div>
      <div className="progress-foot">
        <span className="muted small truncate">{"prompt" in job.request ? job.request.prompt : ""}</span>
        <button className="ghost small" onClick={onCancel}>Cancel</button>
      </div>
    </div>
  );
}
