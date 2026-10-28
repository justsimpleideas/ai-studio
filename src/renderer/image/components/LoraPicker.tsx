import { MAX_LORAS, MAX_LORA_STRENGTH, type LoraUse } from "@shared/image.js";
import type { LoraItem } from "@shared/training.js";

/** Up to MAX_LORAS library LoRAs for the next images, each with its own strength. */
export function LoraPicker({ library, value, onChange }: {
  library: LoraItem[];
  value: LoraUse[];
  onChange: (value: LoraUse[]) => void;
}) {
  const byId = new Map(library.map((l) => [l.id, l]));
  const unused = library.filter((l) => !value.some((u) => u.id === l.id));
  const shown = value.filter((u) => byId.has(u.id));

  return (
    <div className="field lora-picker">
      <span className="field-label">LoRA</span>
      {library.length === 0 && <span className="hint">None yet. Train one in the Train tab, then choose “Use as LoRA”.</span>}
      {shown.map((u) => (
        <div key={u.id} className="lora-row">
          <span className="truncate" title={byId.get(u.id)!.name}>{byId.get(u.id)!.name}</span>
          <input
            type="range" min={0} max={MAX_LORA_STRENGTH} step={0.05} value={u.strength}
            aria-label={`Strength of ${byId.get(u.id)!.name}`}
            onChange={(e) => onChange(value.map((v) => (v.id === u.id ? { ...v, strength: Number(e.target.value) } : v)))}
          />
          <span className="value tabular">{u.strength.toFixed(2)}</span>
          <button type="button" className="icon-button" title="Remove" onClick={() => onChange(value.filter((v) => v.id !== u.id))}>
            ×
          </button>
        </div>
      ))}
      {unused.length > 0 && shown.length < MAX_LORAS && (
        <select
          className="input"
          value=""
          aria-label="Add a LoRA"
          onChange={(e) => e.target.value && onChange([...shown, { id: e.target.value, strength: 1 }])}
        >
          <option value="">Add a LoRA…</option>
          {unused.map((l) => <option key={l.id} value={l.id}>{l.name}</option>)}
        </select>
      )}
    </div>
  );
}
