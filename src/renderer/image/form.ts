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
};

export function toRequest(form: FormState): GenerationRequest {
  return {
    prompt: form.prompt,
    negative_prompt: form.guidance > 0 ? form.negative_prompt : "",
    width: form.width,
    height: form.height,
    steps: form.steps,
    guidance: form.guidance,
    seed: form.randomSeed ? -1 : form.seed,
    count: form.count,
  };
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
