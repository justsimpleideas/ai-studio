import { MAX_LORAS, MAX_LORA_STRENGTH, type LoraUse } from "@shared/image.js";
import type { GenerationRequest, ImageRecord } from "./types";

const STORAGE_KEY = "lis.form.v1";

export interface FormState {
  prompt: string;
  negative_prompt: string;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  seed: number;
  randomSeed: boolean;
  count: number;
  /** Library LoRAs to apply. Ids no longer in the library are dropped when the request is made. */
  loras: LoraUse[];
}

export const DEFAULT_FORM: FormState = {
  prompt: "",
  negative_prompt: "",
  width: 1024,
  height: 1024,
  steps: 8,
  guidance: 0,
  seed: 42,
  randomSeed: true,
  count: 1,
  loras: [],
};

/** Keep only well-formed LoRA uses: a library id, a strength in range, each id once, at most MAX_LORAS. */
export function cleanLoras(value: unknown, known?: Set<string>): LoraUse[] {
  if (!Array.isArray(value)) return [];
  const out: LoraUse[] = [];
  for (const entry of value) {
    if (!entry || typeof entry !== "object") continue;
    const { id, strength } = entry as Record<string, unknown>;
    if (typeof id !== "string" || !/^[0-9a-f]{16}$/.test(id) || (known && !known.has(id))) continue;
    if (typeof strength !== "number" || !Number.isFinite(strength) || strength < 0 || strength > MAX_LORA_STRENGTH) continue;
    if (out.some((u) => u.id === id)) continue;
    out.push({ id, strength });
  }
  return out.slice(0, MAX_LORAS);
}

export function toRequest(form: FormState, library?: Set<string>): GenerationRequest {
  const loras = cleanLoras(form.loras, library);
  return {
    prompt: form.prompt,
    negative_prompt: form.guidance > 0 ? form.negative_prompt : "",
    width: form.width,
    height: form.height,
    steps: form.steps,
    guidance: form.guidance,
    seed: form.randomSeed ? -1 : form.seed,
    count: form.count,
    ...(loras.length ? { loras } : {}),
  };
}

/** Steps and guidance a model starts from. */
export interface ModelDefaults {
  steps: number;
  guidance: number;
}

/**
 * The form for a newly chosen image model: if steps and guidance are still the previous
 * model's defaults, they become the new model's; values the user chose stay. With no
 * previous model on record, the form's own defaults (the 6B model's) count as its.
 */
export function fitToModel(form: FormState, previous: ModelDefaults | null, next: ModelDefaults): FormState {
  const was = previous ?? { steps: DEFAULT_FORM.steps, guidance: DEFAULT_FORM.guidance };
  if (form.steps !== was.steps || form.guidance !== was.guidance) return form;
  if (form.steps === next.steps && form.guidance === next.guidance) return form;
  return { ...form, steps: next.steps, guidance: next.guidance };
}

/** The image model the form was last fitted to, so a restart does not fit it again. */
export interface SeenModel extends ModelDefaults {
  name: string;
}

const MODEL_KEY = "lis.form.model.v1";

export function loadSeenModel(storage: Pick<Storage, "getItem"> = localStorage): SeenModel | null {
  try {
    const parsed = JSON.parse(storage.getItem(MODEL_KEY) ?? "null") as Partial<SeenModel> | null;
    if (parsed && typeof parsed.name === "string" && typeof parsed.steps === "number" && typeof parsed.guidance === "number") {
      return { name: parsed.name, steps: parsed.steps, guidance: parsed.guidance };
    }
  } catch {
    // unreadable: as if no model was seen
  }
  return null;
}

export function saveSeenModel(model: SeenModel, storage: Pick<Storage, "setItem"> = localStorage): void {
  try {
    storage.setItem(MODEL_KEY, JSON.stringify(model));
  } catch {
    // storage full or disabled
  }
}

/** Settings that reproduce a gallery image exactly (seed locked). */
export function fromImage(image: ImageRecord, current: FormState): FormState {
  return {
    ...current,
    prompt: image.prompt,
    negative_prompt: image.negative_prompt,
    width: image.width,
    height: image.height,
    steps: image.steps,
    guidance: image.guidance,
    seed: image.seed,
    randomSeed: false,
    count: 1,
    loras: cleanLoras((image.provenance?.loras ?? []).map((l) => ({ id: l.id, strength: l.strength }))),
  };
}

export function loadForm(storage: Pick<Storage, "getItem"> = localStorage): FormState {
  try {
    const raw = storage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_FORM;
    const parsed = JSON.parse(raw) as Partial<FormState>;
    const form = { ...DEFAULT_FORM };
    for (const key of Object.keys(DEFAULT_FORM) as (keyof FormState)[]) {
      if (typeof parsed[key] === typeof DEFAULT_FORM[key]) (form as Record<string, unknown>)[key] = parsed[key];
    }
    form.loras = cleanLoras(parsed.loras);
    return form;
  } catch {
    return DEFAULT_FORM;
  }
}

export function saveForm(form: FormState, storage: Pick<Storage, "setItem"> = localStorage): void {
  try {
    storage.setItem(STORAGE_KEY, JSON.stringify(form));
  } catch {
    // storage full or disabled: the form simply won't persist
  }
}
