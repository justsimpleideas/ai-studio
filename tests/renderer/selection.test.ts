import { describe, expect, it } from "vitest";
import { digitOf, numberedFolders, planRead, readChunks, selectWith } from "../../src/renderer/image/selection.js";
import type { GalleryFolder } from "../../src/renderer/image/types.js";

const ids = ["a", "b", "c", "d", "e"];

describe("choosing several", () => {
  it("Ctrl toggles one item in or out", () => {
    const one = selectWith(new Set(), ids, "b", { toggle: true, range: false }, null);
    expect([...one]).toEqual(["b"]);
    expect([...selectWith(one, ids, "b", { toggle: true, range: false }, "b")]).toEqual([]);
  });

  it("Shift adds the run from the anchor, either way, keeping what was chosen", () => {
    expect([...selectWith(new Set(["a"]), ids, "d", { toggle: false, range: true }, "b")].sort()).toEqual(["a", "b", "c", "d"]);
    expect([...selectWith(new Set(), ids, "a", { toggle: false, range: true }, "c")].sort()).toEqual(["a", "b", "c"]);
  });

  it("Shift without an anchor, or with one no longer listed, chooses just the item", () => {
    expect([...selectWith(new Set(), ids, "c", { toggle: false, range: true }, null)]).toEqual(["c"]);
    expect([...selectWith(new Set(), ids, "c", { toggle: false, range: true }, "gone")]).toEqual(["c"]);
  });
});

describe("the keyboard folder list", () => {
  const folder = (i: number, extra: Partial<GalleryFolder> = {}): GalleryFolder =>
    ({ id: `f${i}`, name: `Folder ${i}`, count: 0, hidden: false, locked: false, unlocked: false, ...extra }) as GalleryFolder;

  it("numbers No folder 0 and the first nine folders 1 to 9, in list order; later ones are unnumbered", () => {
    const list = numberedFolders(Array.from({ length: 11 }, (_, i) => folder(i + 1)));
    expect(list[0]).toEqual({ key: 0, id: null, name: "No folder" });
    expect(list.slice(1, 10).map((f) => f.key)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(list[9]).toMatchObject({ key: 9, id: "f9" });
    expect(list.slice(10).map((f) => f.key)).toEqual([null, null]);
  });

  it("marks a locked folder", () => {
    expect(numberedFolders([folder(1, { locked: true })])[1]!.name).toBe("🔒 Folder 1");
    expect(numberedFolders([folder(1, { locked: true, unlocked: true })])[1]!.name).toBe("Folder 1");
  });

  it("reads digits from the top row and the number pad, whatever the layout says", () => {
    expect(digitOf({ key: "&", code: "Digit1" })).toBe(1);
    expect(digitOf({ key: "7", code: "Numpad7" })).toBe(7);
    expect(digitOf({ key: "0" })).toBe(0);
    expect(digitOf({ key: "a", code: "KeyA" })).toBeNull();
  });
});

describe("reading the gallery", () => {
  it("Load more on the same list reads only the next part", () => {
    expect(planRead({ key: "all", version: "1", count: 60 }, { key: "all", version: "1", count: 120 })).toEqual({ offset: 60, count: 60, replace: false });
  });

  it("a refresh after a change reads again from the top, as far as was loaded, so pages already shown stay", () => {
    expect(planRead({ key: "all", version: "1", count: 180 }, { key: "all", version: "2", count: 180 })).toEqual({ offset: 0, count: 180, replace: true });
  });

  it("a refresh that lands while Load more is waiting still brings in the extra page", () => {
    // Load more raised `want` to 120; the gallery changed before that read finished.
    expect(planRead({ key: "all", version: "1", count: 60 }, { key: "all", version: "2", count: 120 })).toEqual({ offset: 0, count: 120, replace: true });
  });

  it("another view starts from the top", () => {
    expect(planRead({ key: "all", version: "1", count: 120 }, { key: "folder", version: "1", count: 60 })).toEqual({ offset: 0, count: 60, replace: true });
    expect(planRead(null, { key: "all", version: "1", count: 60 })).toEqual({ offset: 0, count: 60, replace: true });
  });

  it("splits reads into requests the main process accepts", () => {
    expect(readChunks(0, 60)).toEqual([{ offset: 0, limit: 60 }]);
    expect(readChunks(0, 450)).toEqual([{ offset: 0, limit: 200 }, { offset: 200, limit: 200 }, { offset: 400, limit: 50 }]);
    expect(readChunks(120, 60)).toEqual([{ offset: 120, limit: 60 }]);
  });
});
