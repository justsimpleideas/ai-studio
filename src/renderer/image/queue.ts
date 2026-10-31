import type { GenerationRequest, Job, VideoRequest } from "./types";

/**
 * The queue window's logic: which jobs are in line and in what order, and how a
 * job's request becomes an editable draft and back. The engine runs one job at a
 * time; a queued job can be changed in place, and changing the running one
 * cancels it and puts its changed copy first in line (image_engine/jobs.py).
 */

const STORAGE_KEY = "lis.queue.openOnGenerate.v1";

/** Whether Generate opens the queue window; on unless switched off in that window. */
export function loadOpenOnGenerate(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) !== "false";
  } catch {
    return true;
  }
}

export function saveOpenOnGenerate(open: boolean): void {
  try {
    localStorage.setItem(STORAGE_KEY, String(open));
  } catch {
    // Not remembered; the window still works.
  }
}

/**
 * Jobs not done yet, in the order the GPU takes them: the running one(s) first, then
 * the queued ones by when they were asked for. A restarted job keeps its original's
 * `created`, so it sorts ahead of everything queued after that, as it runs.
 */
export function queueOrder(jobs: Record<string, Job>): Job[] {
  const live = Object.values(jobs).filter((j) => j.status === "running" || j.status === "queued");
  const rank = (j: Job): number => (j.status === "running" ? 0 : 1);
  return live.sort((a, b) => rank(a) - rank(b) || a.created - b.created || a.id.localeCompare(b.id));
}

/** Jobs the queue window can change: images and videos that are not already stopping. */
export function isEditable(job: Job): boolean {
  return job.kind !== "task" && !job.cancelling && (job.status === "queued" || job.status === "running");
}

export interface QueueDraft {
  kind: "image" | "video";
  prompt: string;
  negative_prompt: string;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  /** The job's seed. The engine picks a random one at submit time, so a queued job always has one. */
  seed: number;
  /** Ask for a new random seed instead of `seed`. */
  randomSeed: boolean;
  count: number;
  /** Videos only. */
  frames: number;
}

export function draftFrom(job: Job): QueueDraft {
  const r = job.request as GenerationRequest & Partial<VideoRequest>;
  return {
    kind: job.kind === "video" ? "video" : "image",
    prompt: r.prompt,
    negative_prompt: r.negative_prompt ?? "",
    width: r.width,
    height: r.height,
    steps: r.steps,
    guidance: r.guidance,
    seed: r.seed,
    randomSeed: false,
    count: r.count,
    frames: r.frames ?? 0,
  };
}

/**
 * The request a draft asks for. What the window does not show stays as the job had
 * it: an image's LoRAs, a video's start picture.
 */
export function requestFrom(draft: QueueDraft, job: Job): GenerationRequest | VideoRequest {
  const seed = draft.randomSeed ? -1 : draft.seed;
  if (draft.kind === "video") {
    const r = job.request as VideoRequest;
    return {
      kind: "video",
      prompt: draft.prompt,
      // The video pipeline uses the negative prompt only when guidance is above 1 (videoForm.ts).
      negative_prompt: draft.guidance > 1 ? draft.negative_prompt : "",
      width: draft.width,
      height: draft.height,
      frames: draft.frames,
      steps: draft.steps,
      guidance: draft.guidance,
      seed,
      count: draft.count,
      start_image: r.start_image ?? null,
    };
  }
  const r = job.request as GenerationRequest;
  return {
    prompt: draft.prompt,
    // Any guidance above 0 turns the negative prompt on (form.ts).
    negative_prompt: draft.guidance > 0 ? draft.negative_prompt : "",
    width: draft.width,
    height: draft.height,
    steps: draft.steps,
    guidance: draft.guidance,
    seed,
    count: draft.count,
    ...(r.loras?.length ? { loras: r.loras } : {}),
  };
}

/** One line for a queued job: "1024 × 1024 · 2 images · 8 steps · seed 42". */
export function describeJob(job: Job): string {
  if (job.kind === "task") return "Training";
  const r = job.request as GenerationRequest & Partial<VideoRequest> & { fps?: number };
  const fps = r.fps;
  const video = job.kind === "video";
  const parts = [`${r.width} × ${r.height}`];
  if (video && r.frames) parts.push(fps ? `${Math.round(r.frames / fps)} s` : `${r.frames} frames`);
  parts.push(`${r.count} ${video ? "video" : "image"}${r.count === 1 ? "" : "s"}`);
  parts.push(`${r.steps} steps`);
  parts.push(`seed ${r.seed}`);
  if (!video && r.loras?.length) parts.push(`${r.loras.length} LoRA${r.loras.length === 1 ? "" : "s"}`);
  if (video && r.start_image) parts.push("from a picture");
  return parts.join(" · ");
}
