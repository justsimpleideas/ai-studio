import { useEffect, useRef, useState } from "react";
import { gifSize, type GifOptions } from "@shared/image.js";
import { gifFrameCount, gifFrameRates, gifWidths, initialGifOptions } from "../gifOptions";
import type { ImageRecord } from "../types";

interface Props {
  video: ImageRecord;
  onSave: (video: ImageRecord, options: GifOptions) => Promise<string | null>;
  onClose: () => void;
}

/** Size and frame rate for a GIF copy of a video, then a save dialog. */
export function GifExport({ video, onSave, onClose }: Props) {
  const [options, setOptions] = useState<GifOptions>(() => initialGifOptions(video));
  const [state, setState] = useState<{ kind: "idle" | "busy" | "saved" } | { kind: "error"; message: string }>({ kind: "idle" });
  const panel = useRef<HTMLDivElement>(null);
  // The details column can be long (a negative prompt, provenance): bring the panel into view when it opens.
  // Only that column scrolls; scrollIntoView would also shift the app's overflow-hidden ancestors.
  useEffect(() => {
    const el = panel.current;
    const column = el?.closest(".detail-side");
    if (!el || !column) return;
    const below = el.getBoundingClientRect().bottom - column.getBoundingClientRect().bottom;
    if (below > 0) column.scrollTop += below + 12;
  }, []);
  const widths = gifWidths(video.width);
  const rates = gifFrameRates(video.fps || 24);
  const size = gifSize(video.width, video.height, options.width);
  const frames = gifFrameCount(video, options.fps);

  const pick = (changes: Partial<GifOptions>) => {
    setOptions({ ...options, ...changes });
    if (state.kind !== "busy") setState({ kind: "idle" });
  };

  const save = async () => {
    setState({ kind: "busy" });
    try {
      const path = await onSave(video, options);
      setState(path ? { kind: "saved" } : { kind: "idle" });
    } catch (err) {
      setState({ kind: "error", message: err instanceof Error ? err.message : String(err) });
    }
  };

  return (
    <div className="gif-export" role="group" aria-label="Save as GIF" ref={panel}>
      <span className="field-label">GIF width</span>
      <div className="segmented" role="radiogroup" aria-label="GIF width" style={{ gridTemplateColumns: `repeat(${widths.length}, 1fr)` }}>
        {widths.map((w) => (
          <button type="button" key={w.value} role="radio" aria-checked={options.width === w.value}
            className={options.width === w.value ? "active" : ""} onClick={() => pick({ width: w.value })}>
            {w.label}
          </button>
        ))}
      </div>
      <span className="field-label">Frame rate</span>
      <div className="segmented" role="radiogroup" aria-label="GIF frame rate" style={{ gridTemplateColumns: `repeat(${rates.length}, 1fr)` }}>
        {rates.map((f) => (
          <button type="button" key={f} role="radio" aria-checked={options.fps === f}
            className={options.fps === f ? "active" : ""} onClick={() => pick({ fps: f })}>
            {f} fps
          </button>
        ))}
      </div>
      <p className="hint gif-summary">
        {size.width} × {size.height} · {frames} frames. A GIF has 256 colors per frame and grows large quickly; a
        smaller size and frame rate keep it light.
      </p>
      <div className="row-gap">
        <button className="primary grow" onClick={() => void save()} disabled={state.kind === "busy"}>
          {state.kind === "busy" ? "Making the GIF…" : state.kind === "saved" ? "Saved" : "Save GIF…"}
        </button>
        <button className="ghost" onClick={onClose} disabled={state.kind === "busy"}>Close</button>
      </div>
      {state.kind === "error" && <p className="hint error-text" role="alert">{state.message}</p>}
    </div>
  );
}
