/**
 * A training project's subject definition: what the photos show, how their
 * folders are named, and what each photo is described by. The user defines it;
 * "General" and "Cars" are only starting points (they come from the engine).
 *
 * Mirrors image_engine/training/subject.py, which has the authoritative checks;
 * these let the window explain a mistake before anything is sent.
 */

export const SUBJECT_LIMITS = {
  levels: 4,
  fields: 10,
  choices: 20,
  label: 30,
  noun: 40,
  trigger: 32,
  question: 200,
  choice: 40,
  phrase: 60
} as const

export const VALUE = '{value}'
export const NOUN = '{noun}'
export const PRESET_NAMES = ['general', 'cars', 'custom'] as const
export type PresetName = (typeof PRESET_NAMES)[number]

export interface SubjectLevel {
  /** Derived from the label; labels.json uses it. */
  key: string
  label: string
  /** Photos without it are not trained on. */
  required: boolean
  /** Part of the subject's name in captions. */
  in_caption: boolean
}

export interface SubjectField {
  key: string
  label: string
  /** What the captioning model is asked; "{noun}" stands for the subject's noun. */
  question: string
  /** Allowed answers; empty for free text. */
  choices: string[]
  /** An unknown answer is flagged for review. */
  important: boolean
  /** The same for every photo of one subject: labels.json may set it, and odd answers are flagged. */
  consistent: boolean
  /** How a known answer is written into the caption, e.g. "{value} view". */
  phrase: string
}

export interface Subject {
  preset: PresetName
  /** The class word, e.g. "wristwatch"; every caption carries it. */
  noun: string
  /** An optional made-up word put first in every caption. */
  trigger: string
  levels: SubjectLevel[]
  fields: SubjectField[]
}

/** The key for a label: lower case, runs of anything but letters and digits become "_". */
export function subjectKey(label: string): string {
  return label.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, '_').replace(/^_+|_+$/g, '')
}

function text(value: unknown, what: string, limit: number, required = false): string {
  if (value === undefined || value === null) value = ''
  if (typeof value !== 'string') throw new Error(`${what} must be text.`)
  const clean = value.split(/\s+/).filter(Boolean).join(' ')
  if (required && !clean) throw new Error(`${what} is required.`)
  if (clean.length > limit) throw new Error(`${what} must be at most ${limit} characters.`)
  return clean
}

function flag(value: unknown, what: string, fallback: boolean): boolean {
  if (value === undefined || value === null) return fallback
  if (typeof value !== 'boolean') throw new Error(`${what} must be true or false.`)
  return value
}

function object(value: unknown, what: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${what} must be an object.`)
  return value as Record<string, unknown>
}

function key(label: string, what: string, taken: Set<string>): string {
  const k = subjectKey(label)
  if (!k) throw new Error(`${what}: the name needs a letter or a digit.`)
  if (k === 'unknown') throw new Error(`${what}: "unknown" cannot be used as a name.`)
  if (taken.has(k)) throw new Error(`${what}: another level or field already has the name "${label}".`)
  taken.add(k)
  return k
}

/** Rebuild a definition field by field, deriving keys from labels; throws a readable message. */
export function cleanSubject(raw: unknown): Subject {
  const d = object(raw, 'The subject')
  const preset = (d.preset ?? 'custom') as PresetName
  if (!PRESET_NAMES.includes(preset)) throw new Error(`preset must be one of ${PRESET_NAMES.join(', ')}.`)
  const noun = text(d.noun, 'The noun', SUBJECT_LIMITS.noun)
  const trigger = text(d.trigger, 'The trigger word', SUBJECT_LIMITS.trigger)
  if (!/^[\p{L}\p{N}_-]*$/u.test(trigger)) throw new Error('The trigger word must be one word: letters, digits, "_" or "-".')

  if (!Array.isArray(d.levels) || d.levels.length < 1 || d.levels.length > SUBJECT_LIMITS.levels) {
    throw new Error(`Give 1 to ${SUBJECT_LIMITS.levels} folder levels.`)
  }
  const taken = new Set<string>()
  const levels = d.levels.map((raw, i): SubjectLevel => {
    const what = `Folder level ${i + 1}`
    const r = object(raw, what)
    const label = text(r.label, `${what}: the name`, SUBJECT_LIMITS.label, true)
    return { key: key(label, what, taken), label, required: flag(r.required, what, false), in_caption: flag(r.in_caption, what, true) }
  })
  if (!levels.some((l) => l.required)) throw new Error('At least one folder level must be required, or loose photos would be trained on.')

  const rawFields = d.fields ?? []
  if (!Array.isArray(rawFields) || rawFields.length > SUBJECT_LIMITS.fields) {
    throw new Error(`Give at most ${SUBJECT_LIMITS.fields} description fields.`)
  }
  const fields = rawFields.map((raw, i): SubjectField => {
    const what = `Description field ${i + 1}`
    const r = object(raw, what)
    const label = text(r.label, `${what}: the name`, SUBJECT_LIMITS.label, true)
    const k = key(label, `${what} (${label})`, taken)
    const question = text(r.question, `${label}: the question`, SUBJECT_LIMITS.question, true)
    const rawChoices = r.choices ?? []
    if (!Array.isArray(rawChoices) || rawChoices.length > SUBJECT_LIMITS.choices) {
      throw new Error(`${label}: give at most ${SUBJECT_LIMITS.choices} choices.`)
    }
    const choices: string[] = []
    for (const c of rawChoices) {
      const choice = text(c, `${label}: a choice`, SUBJECT_LIMITS.choice, true).toLowerCase()
      if (choice === 'unknown') throw new Error(`${label}: "unknown" is always allowed; do not list it.`)
      if (choices.includes(choice)) throw new Error(`${label}: the choice "${choice}" is listed twice.`)
      choices.push(choice)
    }
    if (choices.length === 1) throw new Error(`${label}: give at least two choices, or none for free text.`)
    const phrase = text(r.phrase, `${label}: the caption phrase`, SUBJECT_LIMITS.phrase) || VALUE
    if (phrase.split(VALUE).length !== 2) throw new Error(`${label}: the caption phrase must contain ${VALUE} once, e.g. "${VALUE} view".`)
    const consistent = flag(r.consistent, label, false)
    if (consistent && choices.length === 0) throw new Error(`${label}: only a field with choices can be the same for the whole subject.`)
    return { key: k, label, question, choices, important: flag(r.important, label, false), consistent, phrase }
  })
  return { preset, noun, trigger, levels, fields }
}

/** The subject's name from a photo's facts: its in-caption levels, in order. */
export function subjectName(subject: Subject, facts: Record<string, string>): string {
  return subject.levels.filter((l) => l.in_caption && facts[l.key]).map((l) => facts[l.key]).join(' ')
}

/** A caption as the engine builds it (subject.py, Subject.caption). */
export function buildCaption(subject: Subject, facts: Record<string, string>, description: Record<string, string>): string {
  const parts = [subject.trigger, subjectName(subject, facts), subject.noun]
  for (const f of subject.fields) {
    const value = description[f.key]
    if (value && value !== 'unknown') parts.push(f.phrase.replace(VALUE, value))
  }
  return parts.filter(Boolean).join(', ')
}

/** An example caption for the editor: level names stand in for the facts, first choices for answers. */
export function previewCaption(subject: Subject): string {
  const facts = Object.fromEntries(subject.levels.map((l) => [l.key, `<${l.label}>`]))
  const description = Object.fromEntries(subject.fields.map((f) => [f.key, f.choices[0] ?? `<${f.label.toLowerCase()}>`]))
  return buildCaption(subject, facts, description)
}

/** The folder layout in words, e.g. "Brand / Model / Reference". */
export function folderLayout(subject: Subject): string {
  return subject.levels.map((l) => l.label).join(' / ')
}
