import { useEffect, useState } from 'react'
import { SUBJECT_LIMITS, cleanSubject, previewCaption, type PresetName, type Subject } from '@shared/subject.js'

/** The editor's working copy: choices as the text the user types, checked as a whole on every change. */
interface Draft {
  preset: PresetName
  noun: string
  trigger: string
  levels: { label: string; required: boolean; in_caption: boolean }[]
  fields: { label: string; question: string; choices: string; important: boolean; consistent: boolean; phrase: string }[]
}

const PRESET_LABEL: Record<string, string> = { general: 'General', cars: 'Cars', custom: 'Custom' }

export function presetLabel(preset: string): string {
  return PRESET_LABEL[preset] ?? preset
}

function toDraft(s: Subject): Draft {
  return {
    preset: s.preset,
    noun: s.noun,
    trigger: s.trigger,
    levels: s.levels.map(({ label, required, in_caption }) => ({ label, required, in_caption })),
    fields: s.fields.map((f) => ({ label: f.label, question: f.question, choices: f.choices.join(', '),
      important: f.important, consistent: f.consistent, phrase: f.phrase }))
  }
}

export function fromDraft(d: Draft): { subject: Subject | null; error: string | null } {
  try {
    return {
      subject: cleanSubject({
        ...d,
        fields: d.fields.map((f) => ({ ...f, choices: f.choices.split(',').map((c) => c.trim()).filter(Boolean) }))
      }),
      error: null
    }
  } catch (e) {
    return { subject: null, error: (e as Error).message }
  }
}

/**
 * Define what a training project is about (docs/training.md, "Subject"): the noun
 * every caption carries, an optional trigger word, the folder levels, and the
 * fields each photo is described by. Starts from a preset; everything is editable.
 * Reports a checked definition, or null with the reason, on every change.
 */
export function SubjectEditor({ initial, presets, disabled, onChange }: {
  initial: Subject
  presets: Subject[]
  disabled?: boolean
  onChange: (subject: Subject | null, error: string | null) => void
}): React.JSX.Element {
  const [draft, setDraft] = useState<Draft>(() => toDraft(initial))
  const checked = fromDraft(draft)
  // Report each new draft once (not on every render of the parent).
  useEffect(() => onChange(checked.subject, checked.error), [draft])

  const level = (i: number, change: Partial<Draft['levels'][number]>) =>
    setDraft((d) => ({ ...d, levels: d.levels.map((l, j) => (j === i ? { ...l, ...change } : l)) }))
  const field = (i: number, change: Partial<Draft['fields'][number]>) =>
    setDraft((d) => ({ ...d, fields: d.fields.map((f, j) => (j === i ? { ...f, ...change } : f)) }))

  return (
    <fieldset className="subject-editor" disabled={disabled}>
      <div className="subject-basics">
        <label className="field">
          <span className="field-label">Start from</span>
          <select className="input" value={draft.preset} aria-label="Start from"
            onChange={(e) => {
              const preset = presets.find((p) => p.preset === e.target.value)
              if (preset) setDraft({ ...toDraft(preset), noun: draft.noun && preset.noun === '' ? draft.noun : preset.noun })
            }}>
            {presets.map((p) => <option key={p.preset} value={p.preset}>{presetLabel(p.preset)}</option>)}
            {!presets.some((p) => p.preset === draft.preset) && <option value={draft.preset}>{presetLabel(draft.preset)}</option>}
          </select>
        </label>
        <label className="field">
          <span className="field-label">What the photos show</span>
          <input className="input" maxLength={SUBJECT_LIMITS.noun} value={draft.noun} placeholder="e.g. wristwatch, dog, chair"
            aria-label="What the photos show" onChange={(e) => setDraft({ ...draft, noun: e.target.value })} />
        </label>
        <label className="field">
          <span className="field-label">Trigger word (optional)</span>
          <input className="input" maxLength={SUBJECT_LIMITS.trigger} value={draft.trigger} placeholder="e.g. mywatch"
            aria-label="Trigger word" onChange={(e) => setDraft({ ...draft, trigger: e.target.value })} />
        </label>
      </div>
      <p className="hint">
        Every caption carries the word for what the photos show, so the LoRA builds on something the model already
        knows. A trigger word, put first in every caption, lets a prompt call the LoRA up on purpose.
      </p>

      <span className="field-label">Folder levels</span>
      <p className="hint">
        Each folder below the photo folder is one level, in this order. A <span className="mono">labels.json</span> in
        any folder can set the exact names.
      </p>
      <div className="subject-levels">
        {draft.levels.map((l, i) => (
          <div key={i} className="subject-level">
            <span className="muted small tabular">{i + 1}</span>
            <input className="input" maxLength={SUBJECT_LIMITS.label} value={l.label} aria-label={`Level ${i + 1} name`}
              placeholder="e.g. Brand" onChange={(e) => level(i, { label: e.target.value })} />
            <label className="toggle small"><input type="checkbox" checked={l.required}
              onChange={(e) => level(i, { required: e.target.checked })} /><span>Required</span></label>
            <label className="toggle small" title="Part of the subject's name in every caption"><input type="checkbox"
              checked={l.in_caption} onChange={(e) => level(i, { in_caption: e.target.checked })} /><span>In caption</span></label>
            <button type="button" className="icon-button" title="Remove this level" disabled={draft.levels.length === 1}
              onClick={() => setDraft({ ...draft, levels: draft.levels.filter((_, j) => j !== i) })}>×</button>
          </div>
        ))}
      </div>
      <div className="caption-actions">
        <button type="button" disabled={draft.levels.length >= SUBJECT_LIMITS.levels}
          onClick={() => setDraft({ ...draft, levels: [...draft.levels, { label: '', required: false, in_caption: true }] })}>
          Add a level
        </button>
      </div>

      <details className="subject-fields">
        <summary>
          <span className="field-label">Description fields ({draft.fields.length})</span>
        </summary>
        <p className="hint">
          What each photo is described by. The captioning model answers each question; a field with choices is answered
          from the list and can be checked twice, free text cannot. Write <span className="mono">{'{noun}'}</span> in a
          question for what the photos show, and <span className="mono">{'{value}'}</span> in the caption phrase for the answer.
        </p>
        {draft.fields.map((f, i) => (
          <div key={i} className="subject-field">
            <input className="input" maxLength={SUBJECT_LIMITS.label} value={f.label} placeholder="Name, e.g. View"
              aria-label={`Field ${i + 1} name`} onChange={(e) => field(i, { label: e.target.value })} />
            <input className="input" maxLength={SUBJECT_LIMITS.question} value={f.question}
              placeholder="Question, e.g. which side of the {noun} faces the camera" aria-label={`Field ${i + 1} question`}
              onChange={(e) => field(i, { question: e.target.value })} />
            <input className="input" value={f.choices} placeholder="Choices, comma-separated (empty: free text)"
              aria-label={`Field ${i + 1} choices`} onChange={(e) => field(i, { choices: e.target.value })} />
            <input className="input" maxLength={SUBJECT_LIMITS.phrase} value={f.phrase} placeholder="{value}"
              aria-label={`Field ${i + 1} caption phrase`} onChange={(e) => field(i, { phrase: e.target.value })} />
            <div className="subject-field-flags">
              <label className="toggle small" title="An unknown answer puts the photo up for review"><input type="checkbox"
                checked={f.important} onChange={(e) => field(i, { important: e.target.checked })} /><span>Important</span></label>
              <label className="toggle small" title="The same for every photo of one subject; labels.json may set it">
                <input type="checkbox" checked={f.consistent} onChange={(e) => field(i, { consistent: e.target.checked })} />
                <span>Same for the whole subject</span>
              </label>
              <span className="grow" />
              <button type="button" className="icon-button" title="Remove this field"
                onClick={() => setDraft({ ...draft, fields: draft.fields.filter((_, j) => j !== i) })}>×</button>
            </div>
          </div>
        ))}
        <div className="caption-actions">
          <button type="button" disabled={draft.fields.length >= SUBJECT_LIMITS.fields}
            onClick={() => setDraft({ ...draft, fields: [...draft.fields, { label: '', question: '', choices: '', important: false, consistent: false, phrase: '{value}' }] })}>
            Add a field
          </button>
        </div>
      </details>

      {checked.subject ? (
        <p className="small subject-preview">Caption example: <span className="mono">{previewCaption(checked.subject)}</span></p>
      ) : (
        <p className="error-text" role="alert">{checked.error}</p>
      )}
    </fieldset>
  )
}
