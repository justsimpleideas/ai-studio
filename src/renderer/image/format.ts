export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 GB";
  const gb = bytes / 2 ** 30;
  return gb >= 10 ? `${gb.toFixed(0)} GB` : `${gb.toFixed(1)} GB`;
}

export function formatDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return "–";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60);
  return `${m}m ${Math.round(s - m * 60)}s`;
}

export function formatDate(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString(undefined, {
    year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

/** Human-readable label for the engine's job stage. */
export function stageLabel(stage: string, step: number, steps: number, index: number, count: number,
                           kind: "image" | "video" = "image"): string {
  const batch = count > 1 ? ` · ${kind} ${index + 1} of ${count}` : "";
  switch (stage) {
    case "queued": return "Waiting in queue";
    case "loading model": return "Loading model";
    case "encoding prompt": return "Reading prompt";
    case "sampling": return `Step ${step} of ${steps}${batch}`;
    case "decoding": return kind === "video" ? `Decoding frames${batch}` : `Finishing image${batch}`;
    case "encoding video": return `Saving video${batch}`;
    case "done": return "Done";
    case "cancelled": return "Cancelled";
    case "failed": return "Failed";
    default: return stage;
  }
}

/** A rough "time left" for long jobs: whole minutes, or "under a minute". */
export function formatRemaining(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return "";
  if (ms < 60_000) return "under a minute left";
  return `about ${Math.round(ms / 60_000)} min left`;
}

/** "5.0 s" for a clip's length. */
export function formatSeconds(frames: number, fps: number): string {
  return fps > 0 ? `${(frames / fps).toFixed(1)} s` : "–";
}
