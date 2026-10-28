import { describe, expect, it } from "vitest";
import {
  DEFAULT_DISPLAY,
  loadDisplay,
  loadGalleryView,
  saveDisplay,
  saveGalleryView,
  startView,
  type DisplaySettings,
} from "../../src/renderer/image/display.js";

const store = (value: unknown) => ({ getItem: () => JSON.stringify(value) });

describe("what the Create and Video tabs show once something is made", () => {
  it("shows the result and the strip by default", () => {
    expect(loadDisplay({ getItem: () => null })).toEqual(DEFAULT_DISPLAY);
    expect(DEFAULT_DISPLAY.image).toEqual({ showResult: true, showStrip: true });
    expect(DEFAULT_DISPLAY.video).toEqual({ showResult: true, showStrip: true });
  });

  it("keeps each choice on its own, for images and videos apart", () => {
    const saved: DisplaySettings = {
      image: { showResult: false, showStrip: true },
      video: { showResult: true, showStrip: false },
      galleryStart: "last",
    };
    let raw = "";
    saveDisplay(saved, { setItem: (_key, value) => (raw = value) });
    expect(loadDisplay({ getItem: () => raw })).toEqual(saved);
  });

  it("fills in what is missing or of the wrong type with the default", () => {
    expect(loadDisplay(store({ image: { showResult: false } }))).toEqual({
      image: { showResult: false, showStrip: true },
      video: { showResult: true, showStrip: true },
      galleryStart: "all",
    });
    expect(loadDisplay(store({ galleryStart: "somewhere" })).galleryStart).toBe("all");
    expect(loadDisplay(store({ galleryStart: "folder:../x" })).galleryStart).toBe("all");
    expect(loadDisplay(store({ galleryStart: "folder:" })).galleryStart).toBe("all");
    expect(loadDisplay(store({ galleryStart: "favorites" })).galleryStart).toBe("favorites");
    expect(loadDisplay(store({ galleryStart: "folder:0123abcd" })).galleryStart).toBe("folder:0123abcd");
    expect(loadDisplay(store({ video: { showStrip: "no" } })).video.showStrip).toBe(true);
  });

  it("opens the gallery on All by default, and keeps the view last open", () => {
    expect(DEFAULT_DISPLAY.galleryStart).toBe("all");
    let raw = "";
    const memory = { setItem: (_key: string, value: string) => (raw = value), getItem: () => raw };
    expect(loadGalleryView(memory)).toEqual({ kind: "all" });
    saveGalleryView({ kind: "folder", id: "f1" }, memory);
    expect(loadGalleryView(memory)).toEqual({ kind: "folder", id: "f1" });
    saveGalleryView({ kind: "favorites" }, memory);
    expect(loadGalleryView(memory)).toEqual({ kind: "favorites" });
    expect(loadGalleryView(store({ kind: "folder", id: 3 }))).toEqual({ kind: "all" });
    expect(loadGalleryView({ getItem: () => "{nope" })).toEqual({ kind: "all" });
  });

  it("opens on All, Favorites, a chosen folder, or where it was left", () => {
    const last = () => ({ kind: "folder", id: "kept" }) as const;
    expect(startView("all", last)).toEqual({ kind: "all" });
    expect(startView("favorites", last)).toEqual({ kind: "favorites" });
    expect(startView("folder:f9", last)).toEqual({ kind: "folder", id: "f9" });
    expect(startView("last", last)).toEqual({ kind: "folder", id: "kept" });
  });

  it("survives junk and storage that fails", () => {
    expect(loadDisplay({ getItem: () => "{nope" })).toEqual(DEFAULT_DISPLAY);
    expect(loadDisplay(store(7))).toEqual(DEFAULT_DISPLAY);
    expect(loadDisplay({ getItem: () => { throw new Error("disabled"); } })).toEqual(DEFAULT_DISPLAY);
    expect(() => saveDisplay(DEFAULT_DISPLAY, { setItem: () => { throw new Error("full"); } })).not.toThrow();
  });
});
