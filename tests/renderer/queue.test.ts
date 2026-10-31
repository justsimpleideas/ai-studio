import { describe, expect, it } from "vitest";
import { describeJob, draftFrom, isEditable, queueOrder, requestFrom } from "../../src/renderer/image/queue.js";
import type { Job } from "../../src/renderer/image/types.js";

function job(id: string, fields: Partial<Job> = {}): Job {
  return {
    id,
    kind: "image",
    request: { prompt: `prompt ${id}`, negative_prompt: "", width: 1024, height: 1024, steps: 8, guidance: 0, seed: 7, count: 1 },
    created: 0,
    status: "queued",
    stage: "queued",
    index: 0,
    count: 1,
    step: 0,
    steps: 0,
    images: [],
    error: null,
    started: null,
    finished: null,
    ...fields,
  };
}

const video = (id: string, fields: Partial<Job> = {}): Job =>
  job(id, {
    kind: "video",
    request: {
      kind: "video", prompt: "waves", negative_prompt: "blurry", width: 1280, height: 704, frames: 121,
      steps: 50, guidance: 5, seed: 3, count: 1, start_image: "0123456789abcdef", fps: 24,
    },
    ...fields,
  });

describe("queue order", () => {
  it("puts the running job first, then queued ones by when they were asked for, and leaves out finished ones", () => {
    const jobs = Object.fromEntries(
      [
        job("c", { created: 3 }),
        job("a", { created: 1, status: "done" }),
        job("b", { created: 2 }),
        job("r", { created: 5, status: "running" }),
        job("x", { created: 0, status: "cancelled" }),
      ].map((j) => [j.id, j])
    );
    expect(queueOrder(jobs).map((j) => j.id)).toEqual(["r", "b", "c"]);
  });

  it("a restarted job, which keeps its original's time, goes ahead of what was queued after it", () => {
    const jobs = Object.fromEntries(
      [
        job("old", { created: 1, status: "running", cancelling: true }),
        job("later", { created: 2 }),
        job("again", { created: 1 }),
      ].map((j) => [j.id, j])
    );
    expect(queueOrder(jobs).map((j) => j.id)).toEqual(["old", "again", "later"]);
  });
});

describe("what can be changed", () => {
  it("images and videos in line or being made, but not tasks, finished jobs, or a job already stopping", () => {
    expect(isEditable(job("q"))).toBe(true);
    expect(isEditable(job("r", { status: "running" }))).toBe(true);
    expect(isEditable(job("s", { status: "running", cancelling: true }))).toBe(false);
    expect(isEditable(job("d", { status: "done" }))).toBe(false);
    expect(isEditable(job("t", { kind: "task", request: { kind: "task", task: "training.train", project: "p", params: {}, label: "Training" } }))).toBe(false);
  });
});

describe("drafts", () => {
  it("an image round-trips, keeping its LoRAs, and sends the negative prompt only with guidance", () => {
    const j = job("i", { request: { prompt: "a cat", negative_prompt: "blurry", width: 1024, height: 1024, steps: 8, guidance: 0, seed: 42, count: 2, loras: [{ id: "l1", strength: 1 }] } });
    const draft = draftFrom(j);
    expect(draft).toMatchObject({ kind: "image", prompt: "a cat", seed: 42, randomSeed: false, count: 2 });
    expect(requestFrom(draft, j)).toEqual({ prompt: "a cat", negative_prompt: "", width: 1024, height: 1024, steps: 8, guidance: 0, seed: 42, count: 2, loras: [{ id: "l1", strength: 1 }] });
    const changed = requestFrom({ ...draft, prompt: "a dog", width: 1344, height: 768, guidance: 3, randomSeed: true }, j);
    expect(changed).toMatchObject({ prompt: "a dog", width: 1344, height: 768, negative_prompt: "blurry", seed: -1 });
  });

  it("a video keeps its kind and start picture, and leaves its frame rate to the engine", () => {
    const j = video("v");
    const request = requestFrom({ ...draftFrom(j), frames: 49, guidance: 1 }, j);
    expect(request).toEqual({
      kind: "video", prompt: "waves", negative_prompt: "", width: 1280, height: 704, frames: 49,
      steps: 50, guidance: 1, seed: 3, count: 1, start_image: "0123456789abcdef",
    });
  });

  it("describes a job in one line", () => {
    expect(describeJob(job("i", { request: { prompt: "p", negative_prompt: "", width: 1024, height: 768, steps: 8, guidance: 0, seed: 9, count: 2 } }))).toBe(
      "1024 × 768 · 2 images · 8 steps · seed 9"
    );
    expect(describeJob(video("v"))).toBe("1280 × 704 · 5 s · 1 video · 50 steps · seed 3 · from a picture");
  });
});
