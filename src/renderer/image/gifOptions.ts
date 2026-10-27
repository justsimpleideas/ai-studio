import { DEFAULT_GIF_OPTIONS, type GifOptions } from "@shared/image.js";
import type { ImageRecord } from "./types";

/** The choices offered when saving a video as a GIF (tested in tests/renderer/gif-options.test.ts). */

const WIDTHS = [320, 480, 640];
const FRAME_RATES = [10, 12, 15, 24];

/** The widths offered for a video: the smaller presets, then its own width. */
export function gifWidths(videoWidth: number): { value: number; label: string }[] {
  return [
    ...WIDTHS.filter((w) => w < videoWidth).map((w) => ({ value: w, label: `${w}` })),
    { value: videoWidth, label: `Full (${videoWidth})` },
  ];
}

/** The frame rates offered: never more than the video's own. */
export function gifFrameRates(videoFps: number): number[] {
  const rates = FRAME_RATES.filter((f) => f <= videoFps);
  return rates.length > 0 ? rates : [Math.max(1, videoFps)];
}

/** The defaults, fitted to what this video offers. */
export function initialGifOptions(video: Pick<ImageRecord, "width" | "fps">): GifOptions {
  const widths = gifWidths(video.width).map((w) => w.value);
  const rates = gifFrameRates(video.fps || 24);
  return {
    width: widths.includes(DEFAULT_GIF_OPTIONS.width) ? DEFAULT_GIF_OPTIONS.width : widths[widths.length - 1]!,
    fps: rates.includes(DEFAULT_GIF_OPTIONS.fps) ? DEFAULT_GIF_OPTIONS.fps : rates[rates.length - 1]!,
  };
}

/** How many frames the GIF will have. */
export function gifFrameCount(video: Pick<ImageRecord, "frames" | "fps">, fps: number): number {
  return Math.max(1, Math.round(((video.frames ?? 1) * fps) / (video.fps || fps)));
}
