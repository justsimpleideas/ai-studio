import type { ImageRecord, StartImage, VideoRequest } from "./types";

/**
 * The video form: what it holds, the sizes and lengths it offers, and how it
 * becomes a request. Every preset here is a size and frame count the engine
 * accepts (sides in multiples of 32, at most 1280×704 pixels; 4k + 1 frames).
 */

const STORAGE_KEY = "lis.videoForm.v1";
export const VIDEO_FPS = 24;

/** The model's own default negative prompt, in English (its text encoder reads many languages). */
export const DEFAULT_NEGATIVE_PROMPT =
  "oversaturated colors, overexposed, static, blurry details, subtitles, text, artwork, painting, still frame, " +
  "overall gray, worst quality, low quality, JPEG artifacts, ugly, deformed, extra fingers, poorly drawn hands, " +
  "poorly drawn face, malformed limbs, fused fingers, motionless scene, cluttered background, three legs, " +
  "crowded background, walking backwards";

export interface VideoFormState {
  prompt: string;
  negative_prompt: string;
  width: number;
  height: number;
  frames: number;
  steps: number;
  guidance: number;
  seed: number;
  randomSeed: boolean;
  count: number;
  /** The picture the video opens on, or null for text alone. */
  start: StartImage | null;
}

export const DEFAULT_VIDEO_FORM: VideoFormState = {
  prompt: "",
  negative_prompt: DEFAULT_NEGATIVE_PROMPT,
  width: 1280,
  height: 704,
  frames: 121,
  steps: 50,
  guidance: 5,
  seed: 42,
  randomSeed: true,
  count: 1,
  start: null,
};

export type Shape = "landscape" | "portrait" | "square";

export const SHAPES: { value: Shape; label: string; w: number; h: number }[] = [
  { value: "landscape", label: "16:9", w: 16, h: 9 },
  { value: "portrait", label: "9:16", w: 9, h: 16 },
  { value: "square", label: "1:1", w: 1, h: 1 },
];

/** Three sizes per shape, largest first. The model was trained at the first. */
export const QUALITIES = [
  { label: "720p", hint: "best" },
  { label: "540p", hint: "" },
  { label: "480p", hint: "fastest" },
];

const SIZES: Record<Shape, { width: number; height: number }[]> = {
  landscape: [{ width: 1280, height: 704 }, { width: 960, height: 544 }, { width: 832, height: 480 }],
  portrait: [{ width: 704, height: 1280 }, { width: 544, height: 960 }, { width: 480, height: 832 }],
  square: [{ width: 928, height: 928 }, { width: 736, height: 736 }, { width: 640, height: 640 }],
};

export const LENGTHS = [
  { seconds: 2, frames: 49 },
  { seconds: 3, frames: 73 },
  { seconds: 4, frames: 97 },
  { seconds: 5, frames: 121 },
];

export function sizeFor(shape: Shape, quality: number): { width: number; height: number } {
  const sizes = SIZES[shape];
  return sizes[Math.min(Math.max(0, quality), sizes.length - 1)]!;
}

export function shapeOf(width: number, height: number): Shape {
  const ratio = width / height;
  if (ratio > 1.15) return "landscape";
  if (ratio < 0.87) return "portrait";
  return "square";
}

/** The preset size closest in pixel count, so the UI can highlight it. */
export function qualityOf(width: number, height: number): number {
  const sizes = SIZES[shapeOf(width, height)];
  let best = 0;
  sizes.forEach((s, i) => {
    if (Math.abs(s.width * s.height - width * height) < Math.abs(sizes[best]!.width * sizes[best]!.height - width * height)) best = i;
  });
  return best;
}

/** The form with a start picture, reshaped to the picture's orientation at the same quality. */
export function withStart(form: VideoFormState, start: StartImage | null): VideoFormState {
  if (!start) return { ...form, start: null };
  return { ...form, start, ...sizeFor(shapeOf(start.width, start.height), qualityOf(form.width, form.height)) };
}

export function toVideoRequest(form: VideoFormState): VideoRequest {
  return {
    kind: "video",
    prompt: form.prompt,
    // The video pipeline uses the negative prompt only when guidance is above 1.
    negative_prompt: form.guidance > 1 ? form.negative_prompt : "",
    width: form.width,
    height: form.height,
    frames: form.frames,
    steps: form.steps,
    guidance: form.guidance,
    seed: form.randomSeed ? -1 : form.seed,
    count: form.count,
    start_image: form.start?.id ?? null,
  };
}

/** Settings that reproduce a gallery video exactly (seed locked, same start picture). */
export function fromVideo(video: ImageRecord, current: VideoFormState): VideoFormState {
  return {
    ...current,
    prompt: video.prompt,
    negative_prompt: video.negative_prompt,
    width: video.width,
    height: video.height,
    frames: video.frames ?? current.frames,
    steps: video.steps,
    guidance: video.guidance,
    seed: video.seed,
    randomSeed: false,
    count: 1,
    start: video.start_image ? { id: video.start_image, width: video.width, height: video.height } : null,
  };
}

function isStart(value: unknown): value is StartImage {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return typeof v.id === "string" && /^[0-9a-f]{16}$/.test(v.id) && typeof v.width === "number" && typeof v.height === "number";
}

export function loadVideoForm(storage: Pick<Storage, "getItem"> = localStorage): VideoFormState {
  try {
    const raw = storage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_VIDEO_FORM;
    const parsed = JSON.parse(raw) as Partial<Record<keyof VideoFormState, unknown>>;
    const form: VideoFormState = { ...DEFAULT_VIDEO_FORM };
    for (const key of Object.keys(DEFAULT_VIDEO_FORM) as (keyof VideoFormState)[]) {
      if (key === "start") continue;
      if (typeof parsed[key] === typeof DEFAULT_VIDEO_FORM[key]) (form as unknown as Record<string, unknown>)[key] = parsed[key];
    }
    form.start = isStart(parsed.start) ? parsed.start : null;
    return form;
  } catch {
    return DEFAULT_VIDEO_FORM;
  }
}

export function saveVideoForm(form: VideoFormState, storage: Pick<Storage, "setItem"> = localStorage): void {
  try {
    storage.setItem(STORAGE_KEY, JSON.stringify(form));
  } catch {
    // storage full or disabled: the form simply won't persist
  }
}
