import { describe, expect, it } from "vitest";
import { gifFrameCount, gifFrameRates, gifWidths, initialGifOptions } from "../../src/renderer/image/gifOptions.js";

describe("GIF choices for a video", () => {
  it("offers the smaller widths and then the video's own", () => {
    expect(gifWidths(1280).map((w) => w.value)).toEqual([320, 480, 640, 1280]);
    expect(gifWidths(832).at(-1)).toEqual({ value: 832, label: "Full (832)" });
    expect(gifWidths(480).map((w) => w.value)).toEqual([320, 480]); // no duplicate 480
    expect(gifWidths(256).map((w) => w.value)).toEqual([256]);
  });

  it("never offers a frame rate above the video's", () => {
    expect(gifFrameRates(24)).toEqual([10, 12, 15, 24]);
    expect(gifFrameRates(12)).toEqual([10, 12]);
    expect(gifFrameRates(8)).toEqual([8]);
  });

  it("starts at 480 wide and 12 fps when the video allows it", () => {
    expect(initialGifOptions({ width: 1280, fps: 24 })).toEqual({ width: 480, fps: 12 });
    expect(initialGifOptions({ width: 256, fps: 8 })).toEqual({ width: 256, fps: 8 });
    expect(initialGifOptions({ width: 1280, fps: 0 })).toEqual({ width: 480, fps: 12 }); // fps unknown: as 24
  });

  it("counts the frames the GIF will have", () => {
    expect(gifFrameCount({ frames: 121, fps: 24 }, 12)).toBe(61);
    expect(gifFrameCount({ frames: 49, fps: 24 }, 24)).toBe(49);
    expect(gifFrameCount({ frames: 1, fps: 24 }, 10)).toBe(1);
  });
});
