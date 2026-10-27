import { useRef, useState } from "react";
import type { FormState } from "../form";
import { ASPECT_RATIOS, RESOLUTIONS, closestRatio, isValidSide, sizeFor, snap, type AspectRatio } from "../sizes";

interface Props {
  form: FormState;
  onChange: (form: FormState) => void;
  onGenerate: () => void;
  canGenerate: boolean;
  pending: number;
  error: string | null;
  /** A PNG chosen to take its generation settings from. */
  onImport: (file: File) => void;
}

const MAX_SEED = 2 ** 32 - 1;

export function GeneratePanel({ form, onChange, onGenerate, canGenerate, pending, error, onImport }: Props) {
  const [advanced, setAdvanced] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const [megapixels, setMegapixels] = useState(() => {
    const mp = (form.width * form.height) / 2 ** 20;
    return RESOLUTIONS.reduce((a, b) => (Math.abs(b.megapixels - mp) < Math.abs(a.megapixels - mp) ? b : a)).megapixels;
  });
  const ratio = closestRatio(form.width, form.height);
  const set = <K extends keyof FormState>(key: K, value: FormState[K]) => onChange({ ...form, [key]: value });

  const pickRatio = (r: AspectRatio) => onChange({ ...form, ...sizeFor(r, megapixels) });
  const pickResolution = (mp: number) => {
    setMegapixels(mp);
    onChange({ ...form, ...sizeFor(ratio, mp) });
  };

  const submit = () => {
    if (canGenerate && form.prompt.trim()) onGenerate();
  };

  return (
    <form
      className="panel generate-panel"
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
          placeholder="A lighthouse on a rocky coast at dusk, warm window light, long exposure, 35mm photo"
          rows={7}
          autoFocus
          onChange={(e) => set("prompt", e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
              e.preventDefault();
              submit();
            }
          }}
        />
      </label>

      <div className="field">
        <span className="field-label">Aspect ratio</span>
        <div className="chips" role="radiogroup" aria-label="Aspect ratio">
          {ASPECT_RATIOS.map((r) => (
            <button
              type="button"
              key={r.label}
              role="radio"
              aria-checked={r.label === ratio.label}
              className={`chip ${r.label === ratio.label ? "active" : ""}`}
              onClick={() => pickRatio(r)}
            >
              <span className="ratio-glyph" style={{ aspectRatio: `${r.w} / ${r.h}` }} aria-hidden />
              {r.label}
            </button>
          ))}
        </div>
      </div>

      <div className="field">
        <span className="field-label">Resolution</span>
        <div className="segmented" role="radiogroup" aria-label="Resolution">
          {RESOLUTIONS.map((r) => (
            <button
              type="button"
              key={r.label}
              role="radio"
              aria-checked={megapixels === r.megapixels}
              className={megapixels === r.megapixels ? "active" : ""}
              onClick={() => pickResolution(r.megapixels)}
            >
              {r.label}
            </button>
          ))}
        </div>
        <div className="size-inputs">
          <NumberInput label="W" value={form.width} onCommit={(v) => set("width", snap(v))} valid={isValidSide} />
          <span className="times">×</span>
          <NumberInput label="H" value={form.height} onCommit={(v) => set("height", snap(v))} valid={isValidSide} />
          <span className="muted small">{((form.width * form.height) / 1e6).toFixed(2)} MP</span>
        </div>
      </div>

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
          <button
            type="button"
            className="icon-button"
            title="New random seed"
            onClick={() => onChange({ ...form, seed: Math.floor(Math.random() * MAX_SEED), randomSeed: false })}
          >
            ⚄
          </button>
        </div>
      </div>

      <div className="field">
        <span className="field-label">
          Images <span className="value">{form.count}</span>
        </span>
        <input type="range" min={1} max={8} value={form.count} onChange={(e) => set("count", Number(e.target.value))} />
      </div>

      <button type="button" className="disclosure" aria-expanded={advanced} onClick={() => setAdvanced(!advanced)}>
        {advanced ? "▾" : "▸"} Advanced
      </button>
      {advanced && (
        <div className="advanced">
          <div className="field">
            <span className="field-label">
              Steps <span className="value">{form.steps}</span>
            </span>
            <input type="range" min={1} max={30} value={form.steps} onChange={(e) => set("steps", Number(e.target.value))} />
            <span className="hint">Turbo is distilled for 8 steps. More rarely helps.</span>
          </div>
          <div className="field">
            <span className="field-label">
              Guidance <span className="value">{form.guidance.toFixed(1)}</span>
            </span>
            <input
              type="range" min={0} max={10} step={0.5} value={form.guidance}
              onChange={(e) => set("guidance", Number(e.target.value))}
            />
            <span className="hint">0 is right for Turbo. Any value above 0 turns on the negative prompt and doubles the work per step.</span>
          </div>
          <label className="field">
            <span className="field-label">Negative prompt</span>
            <textarea
              className="input"
              rows={2}
              value={form.negative_prompt}
              disabled={form.guidance === 0}
              placeholder={form.guidance === 0 ? "Only used when guidance is above 0" : "blurry, low quality"}
              onChange={(e) => set("negative_prompt", e.target.value)}
            />
          </label>
          <button type="button" className="link-button" onClick={() => onChange({ ...form, steps: 8, guidance: 0 })}>
            Reset to Turbo defaults
          </button>
        </div>
      )}

      <div className="generate-footer">
        {error && <div className="error-text" role="alert">{error}</div>}
        <button type="submit" className="primary generate-button" disabled={!canGenerate || !form.prompt.trim()}>
          {pending > 0 ? `Add to queue (${pending} waiting)` : form.count > 1 ? `Generate ${form.count} images` : "Generate"}
        </button>
        <span className="hint center">Ctrl + Enter</span>
        <button
          type="button"
          className="link-button from-png"
          title="Or drop a PNG anywhere on this side"
          onClick={() => fileRef.current?.click()}
        >
          Use the settings of a PNG…
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".png,image/png"
          hidden
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.target.value = "";
            if (file) onImport(file);
          }}
        />
      </div>
    </form>
  );
}

function NumberInput({ label, value, onCommit, valid }: {
  label: string; value: number; onCommit: (v: number) => void; valid: (v: number) => boolean;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const shown = draft ?? String(value);
  const commit = () => {
    if (draft !== null) {
      const v = Number(draft);
      if (Number.isFinite(v)) onCommit(v);
      setDraft(null);
    }
  };
  return (
    <label className="number-input">
      <span>{label}</span>
      <input
        className={`input ${draft !== null && !valid(Number(draft)) ? "invalid" : ""}`}
        inputMode="numeric"
        value={shown}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), commit())}
      />
    </label>
  );
}
