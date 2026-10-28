import { useEffect, useState } from "react";
import { formatSeconds } from "../format";
import type { ResultDisplay } from "../display";
import type { ImageRecord, Job } from "../types";
import { Progress } from "./Preview";

interface Props {
  job: Job | null;
  videos: ImageRecord[];
  onOpen: (video: ImageRecord, list: ImageRecord[]) => void;
  onCancel: (job: Job) => void;
  modelMessage: string;
  starting: boolean;
  display: ResultDisplay;
}

/** A gallery video: MP4 in a player, or (saved without ffmpeg) an animated WebP that plays as a picture. */
export function VideoPlayer({ video, controls = true }: { video: ImageRecord; controls?: boolean }) {
  if (video.mime === "image/webp") {
    return <img className="video-frame" src={video.video_url} alt={video.prompt} width={video.width} height={video.height} />;
  }
  return (
    <video
      key={video.id}
      className="video-frame"
      src={video.video_url}
      poster={video.thumb_url}
      width={video.width}
      height={video.height}
      controls={controls}
      autoPlay
      loop
      muted
      playsInline
    />
  );
}

export function VideoPreview({ job, videos, onOpen, onCancel, modelMessage, starting, display }: Props) {
  const latest = videos[0] ?? null;
  const [selectedId, setSelectedId] = useState<string | null>(null);
  // With results off, the stage shows only what is picked from the strip.
  const selected = videos.find((v) => v.id === selectedId) ?? (display.showResult ? latest : null);
  const done = !!latest && !job;

  // A fresh result takes over the stage, or clears it when results are off.
  useEffect(() => setSelectedId(null), [latest?.id]);

  return (
    <section className="preview">
      <div className="stage">
        {selected ? (
          <div className="stage-video">
            <VideoPlayer video={selected} />
            <button className="ghost small stage-details" onClick={() => onOpen(selected, videos)}>
              Details
            </button>
          </div>
        ) : (
          <div className="empty-stage">
            <svg viewBox="0 0 64 64" width="56" height="56" aria-hidden>
              <rect x="6" y="14" width="40" height="36" rx="5" fill="none" stroke="currentColor" strokeWidth="2.5" />
              <path d="M46 26l12-7v26l-12-7z" fill="currentColor" />
              <path d="M21 25l12 7-12 7z" fill="currentColor" />
            </svg>
            <p>
              {starting
                ? "Starting the image engine…"
                : job
                  ? "Your video is on its way."
                  : done
                    ? "Done. It is in the gallery."
                    : "Describe a scene and how it moves, then press Generate video."}
            </p>
            <p className="muted small">
              {done
                ? "Showing results here is off in Settings."
                : "Everything runs on this computer. The chat model waits while a video is made."}
            </p>
          </div>
        )}
        {job && <Progress job={job} modelMessage={modelMessage} onCancel={() => onCancel(job)} />}
      </div>
      {display.showStrip && videos.length > 0 && (
        <div className="filmstrip" aria-label="Videos this session">
          {videos.map((v) => (
            <button
              key={v.id}
              className={`film ${v.id === selected?.id ? "active" : ""}`}
              onClick={() => setSelectedId(v.id)}
              onDoubleClick={() => onOpen(v, videos)}
              title={v.prompt}
            >
              <img src={v.thumb_url} alt="" loading="lazy" />
              <span className="film-length">{formatSeconds(v.frames ?? 0, v.fps ?? 0)}</span>
            </button>
          ))}
        </div>
      )}
    </section>
  );
}
