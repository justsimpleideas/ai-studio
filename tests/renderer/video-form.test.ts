import { describe, expect, it } from "vitest";
import { stageLabel } from "../../src/renderer/image/format.js";
import type { ImageRecord, Job, Status } from "../../src/renderer/image/types.js";
import { activeJob, engineReducer, initialEngineState } from "../../src/renderer/image/useEngine.js";
import {
  DEFAULT_VIDEO_FORM,
  LENGTHS,
  QUALITIES,
  SHAPES,
  fromVideo,
  loadVideoForm,
  qualityOf,
  shapeOf,
  sizeFor,
  toVideoRequest,
  withStart,
} from "../../src/renderer/image/videoForm.js";

// The engine's rules (image_engine/params.py): sides multiples of 32 within 256..1280,
// at most 1280x704 pixels, frames 4k + 1 up to 121.
const engineAccepts = (w: number, h: number) =>
  w % 32 === 0 && h % 32 === 0 && w >= 256 && h >= 256 && w <= 1280 && h <= 1280 && w * h <= 1280 * 704;

describe("video sizes and lengths", () => {
  it("every preset is a size the engine accepts", () => {
    for (const shape of SHAPES) {
      QUALITIES.forEach((_q, i) => {
        const { width, height } = sizeFor(shape.value, i);
        expect(engineAccepts(width, height), `${shape.value} ${i}: ${width}x${height}`).toBe(true);
        expect(shapeOf(width, height)).toBe(shape.value);
        expect(qualityOf(width, height)).toBe(i);
      });
    }
  });

  it("every length is 4k + 1 frames at most 121", () => {
    for (const { frames, seconds } of LENGTHS) {
      expect((frames - 1) % 4).toBe(0);
      expect(frames).toBeLessThanOrEqual(121);
      expect(Math.abs(frames / 24 - seconds)).toBeLessThan(0.1);
    }
  });

  it("the defaults are the model's own: 720p, 5 seconds, 50 steps, guidance 5", () => {
    const req = toVideoRequest(DEFAULT_VIDEO_FORM);
    expect([req.width, req.height, req.frames, req.steps, req.guidance]).toEqual([1280, 704, 121, 50, 5]);
    expect(req.kind).toBe("video");
    expect(req.seed).toBe(-1);
    expect(req.start_image).toBeNull();
  });
});

describe("video requests", () => {
  it("the negative prompt goes along only when guidance is above 1", () => {
    expect(toVideoRequest({ ...DEFAULT_VIDEO_FORM, guidance: 5 }).negative_prompt).toBe(DEFAULT_VIDEO_FORM.negative_prompt);
    expect(toVideoRequest({ ...DEFAULT_VIDEO_FORM, guidance: 1 }).negative_prompt).toBe("");
  });

  it("a start picture reshapes the video to its orientation at the same quality", () => {
    const draft = { ...DEFAULT_VIDEO_FORM, ...sizeFor("landscape", 2) };
    const portrait = withStart(draft, { id: "0123456789abcdef", width: 1080, height: 1920 });
    expect([portrait.width, portrait.height]).toEqual([480, 832]);
    expect(toVideoRequest({ ...portrait, randomSeed: false, seed: 7 })).toMatchObject({ start_image: "0123456789abcdef", seed: 7 });
    expect(withStart(portrait, null).start).toBeNull();
  });

  it("reusing a video locks its seed and keeps its start picture", () => {
    const video = {
      id: "f".repeat(16), media: "video", prompt: "waves", negative_prompt: "blurry", width: 960, height: 544,
      frames: 73, fps: 24, steps: 30, guidance: 4, seed: 99, start_image: "0123456789abcdef",
    } as ImageRecord;
    const form = fromVideo(video, { ...DEFAULT_VIDEO_FORM, count: 3 });
    expect(form).toMatchObject({ prompt: "waves", width: 960, height: 544, frames: 73, steps: 30, guidance: 4,
                                 seed: 99, randomSeed: false, count: 1 });
    expect(form.start?.id).toBe("0123456789abcdef");
  });
});

describe("video form storage", () => {
  const store = (value: unknown) => ({ getItem: () => JSON.stringify(value) });

  it("keeps a valid start picture and drops a forged one", () => {
    const good = { ...DEFAULT_VIDEO_FORM, start: { id: "0123456789abcdef", width: 10, height: 20 } };
    expect(loadVideoForm(store(good)).start?.id).toBe("0123456789abcdef");
    expect(loadVideoForm(store({ ...good, start: { id: "../../x", width: 1, height: 1 } })).start).toBeNull();
    expect(loadVideoForm(store({ ...good, start: "0123456789abcdef" })).start).toBeNull();
  });

  it("ignores fields of the wrong type and survives junk", () => {
    expect(loadVideoForm(store({ frames: "121", prompt: 4 })).frames).toBe(121);
    expect(loadVideoForm({ getItem: () => "{nope" })).toEqual(DEFAULT_VIDEO_FORM);
  });
});

describe("video jobs in the engine state", () => {
  const job = (id: string, kind: "image" | "video" | undefined, status: Job["status"]) =>
    ({ id, kind, status, created: 1, stage: status, index: 0, count: 1, step: 0, steps: 50, images: [],
       error: null, started: null, finished: null, request: {} } as unknown as Job);

  it("each tab follows only its own jobs; jobs without a kind are images", () => {
    const jobs = { a: job("a", "video", "running"), b: job("b", undefined, "queued") };
    expect(activeJob(jobs, "video")?.id).toBe("a");
    expect(activeJob(jobs, "image")?.id).toBe("b");
    expect(activeJob(jobs)?.id).toBe("a");
  });

  it("model events update the model they name", () => {
    const status = {
      model: { id: "img", state: "ready", message: "" },
      models: { image: { id: "img", state: "ready", message: "" }, video: { id: "vid", state: "unloaded", message: "" } },
    } as unknown as Status;
    let s = engineReducer({ ...initialEngineState, status }, { type: "status", status });
    s = engineReducer(s, { type: "event", event: { type: "model", kind: "image", state: "unloaded", message: "" } });
    s = engineReducer(s, { type: "event", event: { type: "model", kind: "video", state: "loading", message: "Reading the VAE" } });
    expect(s.status?.model.state).toBe("unloaded");
    expect(s.status?.models?.video).toEqual({ id: "vid", state: "loading", message: "Reading the VAE" });
    // An engine from before video sends no kind: that is the image model.
    s = engineReducer(s, { type: "event", event: { type: "model", state: "ready", message: "" } });
    expect(s.status?.model.state).toBe("ready");
    expect(s.status?.models?.video?.state).toBe("loading");
  });

  it("stage labels speak of videos", () => {
    expect(stageLabel("sampling", 3, 50, 1, 2, "video")).toBe("Step 3 of 50 · video 2 of 2");
    expect(stageLabel("decoding", 0, 0, 0, 1, "video")).toBe("Decoding frames");
    expect(stageLabel("encoding video", 0, 0, 0, 1, "video")).toBe("Saving video");
    expect(stageLabel("decoding", 0, 0, 0, 1)).toBe("Finishing image");
  });
});
