import { describe, expect, it } from "vitest";
import { DEFAULT_FORM, fitToModel, fromImage, loadForm, loadSeenModel, saveSeenModel, toRequest } from '../../src/renderer/image/form.js';
import { formatBytes, formatDuration, stageLabel } from '../../src/renderer/image/format.js';
import { ASPECT_RATIOS, SQUARE, closestRatio, isValidSide, sizeFor, snap } from '../../src/renderer/image/sizes.js';
import type { ImageRecord, Job, Status } from '../../src/renderer/image/types.js';
import { activeJob, engineReducer, initialEngineState, queuedCount } from '../../src/renderer/image/useEngine.js';

describe("sizes", () => {
  it("every preset produces sizes the engine accepts", () => {
    for (const ratio of ASPECT_RATIOS) {
      for (const mp of [0.5, 1, 2]) {
        const { width, height } = sizeFor(ratio, mp);
        expect(isValidSide(width), `${ratio.label} ${mp}MP width ${width}`).toBe(true);
        expect(isValidSide(height), `${ratio.label} ${mp}MP height ${height}`).toBe(true);
      }
    }
  });

  it("1:1 at 1 MP is 1024 × 1024", () => {
    expect(sizeFor(SQUARE, 1)).toEqual({ width: 1024, height: 1024 });
  });

  it("keeps the requested pixel count within a few percent", () => {
    const { width, height } = sizeFor({ label: "16:9", w: 16, h: 9 }, 1);
    expect(Math.abs((width * height) / 2 ** 20 - 1)).toBeLessThan(0.05);
  });

  it("snap clamps and rounds to multiples of 16", () => {
    expect(snap(1000)).toBe(1008);
    expect(snap(10)).toBe(256);
    expect(snap(99999)).toBe(2048);
  });

  it("closestRatio recognises presets and near misses", () => {
    expect(closestRatio(1344, 768).label).toBe("16:9");
    expect(closestRatio(768, 1024).label).toBe("3:4");
    expect(closestRatio(1000, 1010).label).toBe("1:1");
  });
});

describe("form", () => {
  it("random seed is sent as -1 and negative prompt only with real guidance", () => {
    const req = toRequest({ ...DEFAULT_FORM, prompt: "x", negative_prompt: "blurry", randomSeed: true });
    expect(req.seed).toBe(-1);
    expect(req.negative_prompt).toBe("");
    const cfg = toRequest({ ...DEFAULT_FORM, prompt: "x", negative_prompt: "blurry", guidance: 4, randomSeed: false, seed: 9 });
    expect(cfg.seed).toBe(9);
    expect(cfg.negative_prompt).toBe("blurry");
  });

  it("any guidance above 0 sends the negative prompt, matching the diffusers pipeline's CFG rule", () => {
    const req = toRequest({ ...DEFAULT_FORM, prompt: "x", negative_prompt: "blurry", guidance: 0.5 });
    expect(req.negative_prompt).toBe("blurry");
  });

  it("defaults follow the Turbo recipe: 8 steps, no guidance", () => {
    expect(DEFAULT_FORM.steps).toBe(8);
    expect(DEFAULT_FORM.guidance).toBe(0);
  });

  it("fromImage locks the seed so the image can be reproduced", () => {
    const image = { prompt: "p", negative_prompt: "", width: 768, height: 1344, steps: 9, guidance: 0, seed: 77 } as ImageRecord;
    const form = fromImage(image, { ...DEFAULT_FORM, count: 4 });
    expect(form).toMatchObject({ prompt: "p", width: 768, height: 1344, seed: 77, randomSeed: false, count: 1 });
  });

  it("a newly chosen model brings its own steps and guidance unless the user changed them", () => {
    const large = { steps: 28, guidance: 4.5 };
    const sixB = { steps: 8, guidance: 0 };
    // First sight of the 12B model, coming from the form's defaults (the 6B model's).
    expect(fitToModel(DEFAULT_FORM, null, large)).toMatchObject(large);
    // Back to the 6B model.
    expect(fitToModel({ ...DEFAULT_FORM, ...large }, large, sixB)).toMatchObject(sixB);
    // Values the user chose stay.
    const chosen = { ...DEFAULT_FORM, steps: 12, guidance: 0 };
    expect(fitToModel(chosen, sixB, large)).toBe(chosen);
    // Nothing to change: the same object comes back.
    expect(fitToModel(DEFAULT_FORM, null, sixB)).toBe(DEFAULT_FORM);
  });

  it("the model the form was fitted to survives a restart, and bad storage reads as none", () => {
    let stored = "";
    saveSeenModel({ name: "large", steps: 28, guidance: 4.5 }, { setItem: (_k, v) => (stored = v) });
    expect(loadSeenModel({ getItem: () => stored })).toEqual({ name: "large", steps: 28, guidance: 4.5 });
    expect(loadSeenModel({ getItem: () => "{not json" })).toBeNull();
    expect(loadSeenModel({ getItem: () => JSON.stringify({ name: "large", steps: "28" }) })).toBeNull();
    expect(loadSeenModel({ getItem: () => null })).toBeNull();
  });

  it("loadForm ignores corrupt or mistyped storage", () => {
    expect(loadForm({ getItem: () => "{not json" })).toEqual(DEFAULT_FORM);
    const loaded = loadForm({ getItem: () => JSON.stringify({ prompt: "kept", width: "huge", evil: 1 }) });
    expect(loaded.prompt).toBe("kept");
    expect(loaded.width).toBe(DEFAULT_FORM.width);
    expect("evil" in loaded).toBe(false);
  });
});

describe("format", () => {
  it("bytes and durations", () => {
    expect(formatBytes(12 * 2 ** 30)).toBe("12 GB");
    expect(formatBytes(6.25 * 2 ** 30)).toBe("6.3 GB");
    expect(formatDuration(850)).toBe("850 ms");
    expect(formatDuration(6400)).toBe("6.4 s");
    expect(formatDuration(125000)).toBe("2m 5s");
  });

  it("stage labels mention the batch position only for batches", () => {
    expect(stageLabel("sampling", 3, 9, 0, 1)).toBe("Step 3 of 9");
    expect(stageLabel("sampling", 3, 9, 1, 4)).toBe("Step 3 of 9 · image 2 of 4");
  });
});

function job(id: string, status: Job["status"], created: number): Job {
  return {
    id, status, created, stage: status, index: 0, count: 1, step: 0, steps: 9, images: [], error: null,
    started: null, finished: null,
    request: { prompt: id, negative_prompt: "", width: 1024, height: 1024, steps: 9, guidance: 0, seed: 1, count: 1 },
  };
}

describe("engine state", () => {
  it("prefers the running job, then the oldest queued one", () => {
    const jobs = { a: job("a", "queued", 2), b: job("b", "queued", 1), c: job("c", "done", 0) };
    expect(activeJob(jobs)?.id).toBe("b");
    expect(activeJob({ ...jobs, d: job("d", "running", 3) })?.id).toBe("d");
    expect(queuedCount(jobs)).toBe(2);
    expect(activeJob({ c: jobs.c })).toBeNull();
  });

  it("new images go to the front of the session and bump the gallery", () => {
    const img = (id: string) => ({ id } as ImageRecord);
    let s = engineReducer(initialEngineState, { type: "event", event: { type: "image", job_id: "j", image: img("1") } });
    s = engineReducer(s, { type: "event", event: { type: "image", job_id: "j", image: img("2") } });
    expect(s.session.map((i) => i.id)).toEqual(["2", "1"]);
    expect(s.galleryVersion).toBe(2);
    s = engineReducer(s, { type: "forget-image", id: "2" });
    expect(s.session.map((i) => i.id)).toEqual(["1"]);
  });

  it("a folder going out of view takes its items out of the session strip", () => {
    const img = (id: string, folder_id: string | null) => ({ id, folder_id } as ImageRecord);
    let s = initialEngineState;
    for (const image of [img("loose", null), img("kept", "open"), img("secret", "vault")]) {
      s = engineReducer(s, { type: "event", event: { type: "image", job_id: "j", image } });
    }
    s = engineReducer(s, { type: "event", event: { type: "privacy", show_hidden: false, blocked: ["vault"] } });
    expect(s.session.map((i) => i.id)).toEqual(["kept", "loose"]);
    expect(s.blockedFolders).toEqual(["vault"]);
    expect([s.privacyVersion, s.galleryVersion]).toEqual([1, 4]);
  });

  it("images from a training task go to the gallery only, not the session strip", () => {
    const task = { ...job("t", "running", 1), kind: "task" as const };
    let s = engineReducer(initialEngineState, { type: "event", event: { type: "job", job: task } });
    s = engineReducer(s, { type: "event", event: { type: "image", job_id: "t", image: { id: "c" } as ImageRecord } });
    expect(s.session).toEqual([]);
    expect(s.galleryVersion).toBe(1);
  });

  it("a fresh engine makes the window read the gallery again", () => {
    const hello = { type: "hello" as const, status: {} as Status, jobs: [] };
    let s = engineReducer(initialEngineState, { type: "event", event: { type: "image", job_id: "j", image: { id: "1" } as ImageRecord } });
    s = engineReducer(s, { type: "event", event: hello });
    expect(s.galleryVersion).toBe(2);
    // The strip is kept: the same engine restarted on the same folder still has them.
    expect(s.session.map((i) => i.id)).toEqual(["1"]);
  });

  it("a moved gallery folder leaves nothing of the old one on screen", () => {
    let s = engineReducer(initialEngineState, { type: "event", event: { type: "image", job_id: "j", image: { id: "1" } as ImageRecord } });
    s = engineReducer(s, { type: "gallery-moved" });
    expect(s.session).toEqual([]);
    expect(s.galleryVersion).toBe(2);
  });

  it("job events upsert by id", () => {
    let s = engineReducer(initialEngineState, { type: "event", event: { type: "job", job: job("a", "queued", 1) } });
    s = engineReducer(s, { type: "event", event: { type: "job", job: job("a", "running", 1) } });
    expect(Object.keys(s.jobs)).toEqual(["a"]);
    expect(s.jobs.a?.status).toBe("running");
  });
});
