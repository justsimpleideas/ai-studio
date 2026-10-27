/** Image size helpers. The engine requires both sides to be multiples of 16. */

export const SIDE_MULTIPLE = 16;
export const MIN_SIDE = 256;
export const MAX_SIDE = 2048;

export interface AspectRatio {
  label: string;
  w: number;
  h: number;
}

export const SQUARE: AspectRatio = { label: "1:1", w: 1, h: 1 };

export const ASPECT_RATIOS: AspectRatio[] = [
  SQUARE,
  { label: "4:3", w: 4, h: 3 },
  { label: "3:4", w: 3, h: 4 },
  { label: "3:2", w: 3, h: 2 },
  { label: "2:3", w: 2, h: 3 },
  { label: "16:9", w: 16, h: 9 },
  { label: "9:16", w: 9, h: 16 },
  { label: "21:9", w: 21, h: 9 },
];

/** Megapixel presets. The image model is trained around 1 MP and handles up to ~4 MP. */
export const RESOLUTIONS = [
  { label: "Draft", megapixels: 0.5 },
  { label: "Standard", megapixels: 1.0 },
  { label: "Large", megapixels: 2.0 },
];

export function snap(value: number): number {
  const snapped = Math.round(value / SIDE_MULTIPLE) * SIDE_MULTIPLE;
  return Math.min(MAX_SIDE, Math.max(MIN_SIDE, snapped));
}

/** Width and height for an aspect ratio at roughly the given pixel count. */
export function sizeFor(ratio: AspectRatio, megapixels: number): { width: number; height: number } {
  const pixels = megapixels * 1024 * 1024;
  const height = Math.sqrt((pixels * ratio.h) / ratio.w);
  const width = (height * ratio.w) / ratio.h;
  return { width: snap(width), height: snap(height) };
}

/** The preset ratio closest to an arbitrary size, so the UI can highlight it. */
export function closestRatio(width: number, height: number): AspectRatio {
  const target = Math.log(width / height);
  let best = SQUARE;
  for (const r of ASPECT_RATIOS) {
    if (Math.abs(Math.log(r.w / r.h) - target) < Math.abs(Math.log(best.w / best.h) - target)) best = r;
  }
  return best;
}

export function isValidSide(value: number): boolean {
  return Number.isInteger(value) && value >= MIN_SIDE && value <= MAX_SIDE && value % SIDE_MULTIPLE === 0;
}
