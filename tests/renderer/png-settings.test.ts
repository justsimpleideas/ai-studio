import { describe, expect, it } from "vitest";
import { readPngText, settingsFromPng } from "../../src/renderer/image/png-settings.js";

/**
 * PNG files built chunk by chunk with web APIs only, so each test says exactly
 * what is inside. The reader never decodes pixels or checks checksums, so a
 * token IDAT and zeroed CRCs are enough here.
 */
const SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
const latin1 = (s: string) => Uint8Array.from(s, (c) => c.charCodeAt(0));
const utf8 = (s: string) => new TextEncoder().encode(s);

function concat(...parts: Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

function chunk(type: string, data: Uint8Array): Uint8Array {
  const head = new Uint8Array(8);
  new DataView(head.buffer).setUint32(0, data.length);
  head.set(latin1(type), 4);
  return concat(head, data, new Uint8Array(4));
}

async function deflate(data: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([data.slice()]).stream().pipeThrough(new CompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

const tEXt = (key: string, text: string) => chunk("tEXt", latin1(`${key}\0${text}`));
const zTXt = async (key: string, text: string) => chunk("zTXt", concat(latin1(`${key}\0\0`), await deflate(latin1(text))));
const iTXt = async (key: string, text: string, compress = false) =>
  chunk("iTXt", concat(latin1(`${key}\0`), Uint8Array.of(compress ? 1 : 0, 0), latin1("\0\0"), compress ? await deflate(utf8(text)) : utf8(text)));

const png = (...chunks: Uint8Array[]) =>
  concat(Uint8Array.from(SIGNATURE), chunk("IHDR", new Uint8Array(13)), ...chunks, chunk("IDAT", Uint8Array.of(0)), chunk("IEND", new Uint8Array(0)));

const SETTINGS = {
  format: 1,
  prompt: "a lighthouse at dusk, “warm” light",
  negative_prompt: "blurry",
  width: 1024,
  height: 768,
  steps: 8,
  guidance: 0,
  seed: 1234,
  count: 3,
  model: "turbo-6b",
  app: "Local AI Studio 0.1.0",
};

describe("reading settings back from a PNG", () => {
  it("reads plain, international and compressed text chunks", async () => {
    const text = await readPngText(png(tEXt("a", "one"), await iTXt("b", "twö"), await iTXt("c", "three", true), await zTXt("d", "four")));
    expect(Object.fromEntries(text)).toEqual({ a: "one", b: "twö", c: "three", d: "four" });
  });

  it("restores the studio's settings with the seed locked, as one image", async () => {
    const result = await settingsFromPng(png(await iTXt("local-ai-studio", JSON.stringify(SETTINGS))));
    expect(result).toEqual({
      ok: true,
      settings: {
        prompt: SETTINGS.prompt,
        negative_prompt: "blurry",
        width: 1024,
        height: 768,
        steps: 8,
        guidance: 0,
        seed: 1234,
        randomSeed: false,
        count: 1,
      },
    });
  });

  it("still reads images saved under the name used before the merge", async () => {
    const result = await settingsFromPng(png(tEXt("local-image-studio", JSON.stringify({ prompt: "old", seed: 5 }))));
    expect(result.ok && result.settings.prompt).toBe("old");
  });

  it("keeps out values the engine would refuse, and picks a new seed when none is usable", async () => {
    const bad = { prompt: "x", width: 1000, height: 4096, steps: 500, guidance: -1, seed: -7, negative_prompt: 5 };
    const result = await settingsFromPng(png(tEXt("local-ai-studio", JSON.stringify(bad))));
    expect(result).toEqual({ ok: true, settings: { prompt: "x", randomSeed: true, count: 1 } });
  });

  it("says why when there is nothing it can use", async () => {
    const none = await settingsFromPng(png());
    expect(!none.ok && none.reason).toMatch(/no settings saved/);
    const other = await settingsFromPng(png(tEXt("parameters", "a cat, Steps: 20")));
    expect(!other.ok && other.reason).toMatch(/another app/);
    const broken = await settingsFromPng(png(tEXt("local-ai-studio", '{"prompt":')));
    expect(!broken.ok && broken.reason).toMatch(/could not be read/);
    const notPng = await settingsFromPng(utf8("not an image at all"));
    expect(!notPng.ok && notPng.reason).toMatch(/not a PNG/);
  });

  it("does not inflate a compressed chunk past its limit", async () => {
    const bomb = await iTXt("local-ai-studio", "a".repeat(8 * 2 ** 20), true);
    const text = await readPngText(png(bomb));
    expect(text.has("local-ai-studio")).toBe(false);
  });
});
