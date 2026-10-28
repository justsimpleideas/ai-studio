import { useState } from "react";
import { studioMediaUrl } from "@shared/image.js";
import type { VideoInfo } from "../types";
import {
  DEFAULT_VIDEO_FORM,
  QUALITIES,
  SHAPES,
  geometryFor,
  qualityOf,
  shapeOf,
  sizeFor,
  type Shape,
  type VideoFormState,
} from "../videoForm";

interface Props {
  form: VideoFormState;
  onChange: (form: VideoFormState) => void;
  onGenerate: () => void;
  canGenerate: boolean;
  pending: number;
  error: string | null;
  /** The engine's video limits and encoder; undefined when the engine has no video model. */
  video: VideoInfo | undefined;
  onPickStart: () => void;
}

const MAX_SEED = 2 ** 32 - 1;

export function VideoPanel({ form, onChange, onGenerate, canGenerate, pending, error, video, onPickStart }: Props) {
  const [advanced, setAdvanced] = useState(false);
  const set = <K extends keyof VideoFormState>(key: K, value: VideoFormState[K]) => onChange({ ...form, [key]: value });
  const profile = video?.profile ?? null;
  const geometry = geometryFor(profile);
  const fps = video?.limits.fps ?? geometry.fps;
  const shape = shapeOf(form.width, form.height);
  const quality = qualityOf(form.width, form.height, geometry);
  const available = video !== undefined;
  // A distilled model fixes steps and guidance itself; the form keeps its own values for the original.
  const fast = profile?.fixed ? profile : null;
  const cfg = !fast && form.guidance > 1;
  const large = profile?.experts === 2;
  // The 14B models: one starts from text only, the other from a picture only.
  const start = profile?.start ?? "any";
  const needsStart = start === "image" && !form.start;
  const defaults = video?.defaults ?? DEFAULT_VIDEO_FORM;

  const submit = () => {
    if (canGenerate && available && form.prompt.trim() && !needsStart) onGenerate();
  };

  return (
    <form
      className="panel generate-panel video-panel"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <label className="field">
        <span className="field-label">Prompt</span>
        <textarea
          className="prompt"
          value={form.prompt}
          placeholder="A paper boat drifts down a rain-filled gutter, water rippling around it. The camera follows slowly at street level."
          rows={6}
          onChange={(e) => set("prompt", e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
              e.preventDefault();
              submit();
            }
          }}
        />
        <span className="hint">Say what moves and how the camera moves. One or two actions fit in a few seconds.</span>
      </label>

      {start === "text" ? (
        <div className="field">
          <span className="field-label">Start picture</span>
          <span className="hint">
            This model makes video from the prompt alone. To start from a picture, choose an image-to-video model in
            Settings.
          </span>
        </div>
      ) : (
      <div className="field">
        <span className="field-label">
          Start picture{" "}
          <span className="value">
            {form.start ? `${form.start.width} × ${form.start.height}` : start === "image" ? "required" : "optional"}
          </span>
        </span>
        {form.start ? (
          <div className="start-picture">
            <img src={studioMediaUrl(form.start.id, "input")} alt="The picture the video starts from" />
            <div className="start-actions">
              <button type="button" className="small" onClick={onPickStart}>Change…</button>
              {start !== "image" && <button type="button" className="small ghost" onClick={() => set("start", null)}>Remove</button>}
            </div>
          </div>
        ) : (
          <div className="start-empty">
            <button type="button" onClick={onPickStart}>Choose a picture…</button>
            <span className="hint">
              Or drop one on this side, or use “Animate” on an image in the gallery.{" "}
              {start === "image" ? "This model makes video from a picture: it needs one." : "Without one, the video comes from the prompt alone."}
            </span>
          </div>
        )}
      </div>
      )}

      <div className="field">
        <span className="field-label">Shape</span>
        <div className="chips three" role="radiogroup" aria-label="Shape">
          {SHAPES.map((s) => (
            <button
              type="button"
              key={s.value}
              role="radio"
              aria-checked={shape === s.value}
              className={`chip ${shape === s.value ? "active" : ""}`}
              onClick={() => onChange({ ...form, ...sizeFor(s.value as Shape, quality, geometry) })}
            >
              <span className="ratio-glyph" style={{ aspectRatio: `${s.w} / ${s.h}` }} aria-hidden />
              {s.label}
            </button>
          ))}
        </div>
        {form.start && shapeOf(form.start.width, form.start.height) !== shape && (
          <span className="hint">The start picture is cropped to this shape.</span>
        )}
      </div>

      <div className="field">
        <span className="field-label">
          Size <span className="value">{form.width} × {form.height}</span>
        </span>
        <div className="segmented" role="radiogroup" aria-label="Size">
          {QUALITIES.map((q, i) => (
            <button
              type="button"
              key={q.label}
              role="radio"
              aria-checked={quality === i}
              className={quality === i ? "active" : ""}
              title={q.hint ? `${q.label}, ${q.hint}` : q.label}
              onClick={() => onChange({ ...form, ...sizeFor(shape, i, geometry) })}
            >
              {q.label}
            </button>
          ))}
        </div>
        {large ? (
          <span className="hint">
            The model makes 480p and 720p. On a 12 GB card 480p is the practical size: 720p takes more than twice as long.
          </span>
        ) : (
          quality > 0 && <span className="hint">The model is trained at 720p. Smaller is much faster but can look softer.</span>
        )}
      </div>

      <div className="field">
        <span className="field-label">
          Length <span className="value">{form.frames} frames · {fps} fps</span>
        </span>
        <div className="segmented" role="radiogroup" aria-label="Length">
          {geometry.lengths.map((l) => (
            <button
              type="button"
              key={l.frames}
              role="radio"
              aria-checked={form.frames === l.frames}
              className={form.frames === l.frames ? "active" : ""}
              onClick={() => set("frames", l.frames)}
            >
              {l.seconds} s
            </button>
          ))}
        </div>
        <span className="hint">5 seconds is the longest one take. Longer videos: continue a clip from its last frame.</span>
        {fast && form.frames !== fast.trained_frames && (
          <span className="hint warn-text fast-length-note">
            The {large ? "distilled" : "fast"} model is trained on {Math.round(fast.trained_frames / fps)}-second clips;
            shorter ones can lose quality toward the end.
          </span>
        )}
      </div>

      <div className="field">
        <span className="field-label">
          Videos <span className="value">{form.count}</span>
        </span>
        <input type="range" min={1} max={video?.limits.max_count ?? 4} value={form.count}
               onChange={(e) => set("count", Number(e.target.value))} />
      </div>

      <button type="button" className="disclosure" aria-expanded={advanced} onClick={() => setAdvanced(!advanced)}>
        {advanced ? "▾" : "▸"} Advanced
      </button>
      {advanced && (
        <div className="advanced">
          {fast ? (
            <p className="hint fixed-note">
              {fast.label}: this model makes a clip in {fast.steps} steps without guidance, so steps, guidance
              and the negative prompt are set by the model.
            </p>
          ) : (
          <>
          <div className="field">
            <span className="field-label">
              Steps <span className="value">{form.steps}</span>
            </span>
            <input type="range" min={10} max={50} value={form.steps} onChange={(e) => set("steps", Number(e.target.value))} />
            <span className="hint">
              {defaults.steps} is the model's default. Around {Math.round(defaults.steps * 0.6)} makes a quicker draft.
            </span>
          </div>
          <div className="field">
            <span className="field-label">
              Guidance <span className="value">{form.guidance.toFixed(1)}</span>
            </span>
            <input type="range" min={1} max={10} step={0.5} value={form.guidance}
                   onChange={(e) => set("guidance", Number(e.target.value))} />
            <span className="hint">
              {defaults.guidance} is the model's default
              {large && profile?.guidance_2 && profile.guidance_2 !== profile.guidance
                ? ` (the second expert runs at ${profile.guidance_2}, in the same proportion)`
                : ""}
              . At 1 the negative prompt is off and each step takes half the time, usually at a cost in quality.
            </span>
          </div>
          <label className="field">
            <span className="field-label">Negative prompt</span>
            <textarea
              className="input"
              rows={3}
              value={form.negative_prompt}
              disabled={!cfg}
              placeholder={cfg ? "" : "Only used when guidance is above 1"}
              onChange={(e) => set("negative_prompt", e.target.value)}
            />
          </label>
          </>
          )}
          <div className="field">
            <span className="field-label">Seed</span>
            <div className="seed-row">
              <label className="toggle">
                <input type="checkbox" checked={form.randomSeed} onChange={(e) => set("randomSeed", e.target.checked)} />
                <span>Random</span>
              </label>
              <input
                type="number"
                className="input seed-input"
                min={0}
                max={MAX_SEED}
                value={form.seed}
                disabled={form.randomSeed}
                aria-label="Seed"
                onChange={(e) => {
                  const v = Number(e.target.value);
                  if (Number.isInteger(v) && v >= 0 && v <= MAX_SEED) set("seed", v);
                }}
              />
            </div>
          </div>
          {!fast && (
            <button
              type="button"
              className="link-button"
              onClick={() =>
                onChange({ ...form, steps: defaults.steps, guidance: defaults.guidance, negative_prompt: DEFAULT_VIDEO_FORM.negative_prompt })
              }
            >
              Reset to the model's defaults
            </button>
          )}
        </div>
      )}

      <div className="generate-footer">
        {!available && (
          <div className="error-text" role="alert">This engine has no video model. Update the studio's image engine.</div>
        )}
        {available && video.encoder === "webp" && (
          <div className="hint warn-text">
            ffmpeg was not found, so videos are saved as animated WebP. Choose ffmpeg in Settings for MP4.
          </div>
        )}
        {error && <div className="error-text" role="alert">{error}</div>}
        {needsStart && <div className="hint warn-text">Choose a start picture: this model makes video from a picture.</div>}
        <button type="submit" className="primary generate-button"
                disabled={!canGenerate || !available || !form.prompt.trim() || needsStart}>
          {pending > 0 ? `Add to queue (${pending} waiting)` : form.count > 1 ? `Generate ${form.count} videos` : "Generate video"}
        </button>
        <span className="hint center">
          {large
            ? `Ctrl + Enter · ${fast ? `${fast.steps} steps` : "a clip"} with a 14B model takes many minutes on a 12 GB card`
            : fast ? `Ctrl + Enter · ${fast.steps} steps per clip with the fast model` : "Ctrl + Enter · a 720p clip takes minutes, not seconds"}
        </span>
      </div>
    </form>
  );
}
