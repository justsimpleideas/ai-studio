import type { FormState } from "./form";

/**
 * Reading generation settings back out of a PNG the studio made.
 *
 * The file comes from wherever the user dragged it from, so it is treated as
 * untrusted: the size is capped before reading, chunk lengths are checked
 * against the file, compressed text is inflated only up to a limit, and every
 * value is type- and range-checked against the engine's own limits (the engine
 * validates the request again when it is submitted). Nothing here touches the
 * disk or the network.
 */

/** Far above any image the studio makes; bigger files are refused before reading. */
export const MAX_PNG_BYTES = 64 * 2 ** 20;
const MAX_TEXT_BYTES = 2 ** 20;
/** The current key, then the one used before the image studio was merged in. */
const KEYS = ["local-ai-studio", "local-image-studio"];
const SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

// The engine's limits (image-engine/image_engine/params.py).
const MAX_PROMPT_CHARS = 4000;
const MIN_SIDE = 256;
const MAX_SIDE = 2048;
const SIDE_MULTIPLE = 16;
const MAX_PIXELS = 2048 * 2048;
const MAX_STEPS = 50;
const MAX_GUIDANCE = 10;
const MAX_SEED = 2 ** 32 - 1;

export type PngSettings = { ok: true; settings: Partial<FormState> } | { ok: false; reason: string };

export async function settingsFromPng(bytes: Uint8Array): Promise<PngSettings> {
  if (bytes.length > MAX_PNG_BYTES) return { ok: false, reason: "That file is too large to be an image from this studio." };
  let text: Map<string, string>;
  try {
    text = await readPngText(bytes);
  } catch {
    return { ok: false, reason: "That file is not a PNG image." };
  }
  const raw = KEYS.map((key) => text.get(key)).find((value) => value !== undefined);
  if (raw === undefined) {
    return {
      ok: false,
      reason:
        text.size > 0
          ? "This image has settings from another app, in a form this studio cannot read."
          : "This image has no settings saved in it. Copies saved from the studio leave them out unless “Include the prompt and settings” is on in the image side's Settings.",
    };
  }

  let meta: unknown;
  try {
    meta = JSON.parse(raw);
  } catch {
    meta = null;
  }
  if (!meta || typeof meta !== "object" || Array.isArray(meta)) {
    return { ok: false, reason: "The settings in this image could not be read." };
  }
  const m = meta as Record<string, unknown>;
  if (typeof m.prompt !== "string" || !m.prompt.trim() || m.prompt.length > MAX_PROMPT_CHARS) {
    return { ok: false, reason: "The settings in this image could not be read." };
  }

  // Each value is taken only when it is something the engine would accept; the rest keep the form's current values.
  const settings: Partial<FormState> = { prompt: m.prompt, randomSeed: false, count: 1 };
  if (typeof m.negative_prompt === "string" && m.negative_prompt.length <= MAX_PROMPT_CHARS) {
    settings.negative_prompt = m.negative_prompt;
  }
  const width = side(m.width);
  const height = side(m.height);
  if (width !== null && height !== null && width * height <= MAX_PIXELS) {
    settings.width = width;
    settings.height = height;
  }
  const steps = integer(m.steps, 1, MAX_STEPS);
  if (steps !== null) settings.steps = steps;
  if (typeof m.guidance === "number" && Number.isFinite(m.guidance) && m.guidance >= 0 && m.guidance <= MAX_GUIDANCE) {
    settings.guidance = m.guidance;
  }
  const seed = integer(m.seed, 0, MAX_SEED);
  if (seed !== null) settings.seed = seed;
  else settings.randomSeed = true; // no usable seed: the closest thing is a fresh one
  return { ok: true, settings };
}

function integer(value: unknown, min: number, max: number): number | null {
  return typeof value === "number" && Number.isInteger(value) && value >= min && value <= max ? value : null;
}

function side(value: unknown): number | null {
  const n = integer(value, MIN_SIDE, MAX_SIDE);
  return n !== null && n % SIDE_MULTIPLE === 0 ? n : null;
}

/** The text chunks of a PNG, keyword to text. Compressed ones are inflated, within a budget. */
export async function readPngText(bytes: Uint8Array): Promise<Map<string, string>> {
  if (bytes.length < SIGNATURE.length || SIGNATURE.some((b, i) => bytes[i] !== b)) throw new Error("not a PNG");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Map<string, string>();
  let budget = MAX_TEXT_BYTES;
  let at = SIGNATURE.length;
  while (at + 12 <= bytes.length && budget > 0) {
    const length = view.getUint32(at);
    const type = latin1(bytes.subarray(at + 4, at + 8));
    const start = at + 8;
    const end = start + length;
    if (end + 4 > bytes.length || type === "IEND") break;
    if (type === "tEXt" || type === "zTXt" || type === "iTXt") {
      const entry = await textChunk(type, bytes.subarray(start, end), budget);
      if (entry) {
        budget -= entry[1].length;
        if (!out.has(entry[0])) out.set(entry[0], entry[1]);
      }
    }
    at = end + 4;
  }
  return out;
}

async function textChunk(type: string, data: Uint8Array, budget: number): Promise<[string, string] | null> {
  const nul = data.indexOf(0);
  if (nul < 1 || nul > 79) return null;
  const keyword = latin1(data.subarray(0, nul));
  if (type === "tEXt") return [keyword, latin1(data.subarray(nul + 1))];
  if (type === "zTXt") {
    if (data[nul + 1] !== 0) return null;
    const text = await inflate(data.subarray(nul + 2), budget);
    return text ? [keyword, latin1(text)] : null;
  }
  // iTXt: keyword, compression flag, method, language tag, translated keyword, then UTF-8 text.
  const compressed = data[nul + 1] === 1;
  const langEnd = data.indexOf(0, nul + 3);
  const translatedEnd = langEnd < 0 ? -1 : data.indexOf(0, langEnd + 1);
  if (translatedEnd < 0) return null;
  const body = data.subarray(translatedEnd + 1);
  const text = compressed ? await inflate(body, budget) : body;
  return text ? [keyword, new TextDecoder("utf-8").decode(text)] : null;
}

function latin1(bytes: Uint8Array): string {
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return s;
}

/** zlib-inflate, giving up (null) past `limit` bytes of output or on corrupt input. */
async function inflate(data: Uint8Array, limit: number): Promise<Uint8Array | null> {
  const reader = new Blob([data.slice()]).stream().pipeThrough(new DecompressionStream("deflate")).getReader();
  const parts: Uint8Array[] = [];
  let total = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.length;
      if (total > limit) {
        await reader.cancel();
        return null;
      }
      parts.push(value);
    }
  } catch {
    return null;
  }
  const out = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}
