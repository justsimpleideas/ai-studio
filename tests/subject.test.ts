import { describe, expect, it } from 'vitest'
import { buildCaption, cleanSubject, folderLayout, previewCaption, subjectKey } from '@shared/subject.js'

const watches = {
  noun: 'wristwatch',
  trigger: 'mywatch',
  levels: [{ label: 'Brand', required: true }, { label: 'Model', required: true }, { label: 'Reference', in_caption: false }],
  fields: [
    { label: 'View', question: 'which side of the {noun} faces the camera', choices: ['Front', 'back'], phrase: '{value} view' },
    { label: 'Dial colour', question: 'the dial colour', phrase: '{value} dial' }
  ]
}

describe('subject definitions', () => {
  it('derives keys from labels like the engine does', () => {
    expect(subjectKey('Body style')).toBe('body_style')
    expect(subjectKey('  Dial / Colour! ')).toBe('dial_colour')
    expect(subjectKey('μάρκα όνομα')).toBe('μάρκα_όνομα')
    const s = cleanSubject(watches)
    expect(s.levels.map((l) => [l.key, l.required, l.in_caption])).toEqual([['brand', true, true], ['model', true, true], ['reference', false, false]])
    expect(s.fields.map((f) => [f.key, f.choices, f.phrase])).toEqual([['view', ['front', 'back'], '{value} view'], ['dial_colour', [], '{value} dial']])
    expect(s.preset).toBe('custom')
  })

  it('builds captions the same way as the engine (subject.py)', () => {
    const s = cleanSubject(watches)
    // The same caption is asserted in image-engine/tests/test_training_captions.py.
    expect(buildCaption(s, { brand: 'Seiko', model: 'SKX007', reference: 'SKX007K2' }, { view: 'front', dial_colour: 'black' }))
      .toBe('mywatch, Seiko SKX007, wristwatch, front view, black dial')
    expect(buildCaption(s, { brand: 'Seiko', model: 'SKX007' }, { view: 'unknown' })).toBe('mywatch, Seiko SKX007, wristwatch')
    expect(previewCaption(s)).toBe('mywatch, <Brand> <Model>, wristwatch, front view, <dial colour> dial')
    expect(folderLayout(s)).toBe('Brand / Model / Reference')
  })

  it('refuses what the engine refuses, with a reason', () => {
    const bad: [string, unknown][] = [
      ['not an object', []],
      ['no levels', { ...watches, levels: [] }],
      ['no required level', { ...watches, levels: [{ label: 'Brand' }] }],
      ['a label of symbols', { ...watches, levels: [{ label: '!!', required: true }] }],
      ['the same name twice', { ...watches, levels: [{ label: 'Brand', required: true }, { label: 'brand' }] }],
      ['a field named like a level', { ...watches, fields: [{ label: 'Model', question: 'q' }] }],
      ['a trigger with a space', { ...watches, trigger: 'my watch' }],
      ['a single choice', { ...watches, fields: [{ label: 'V', question: 'q', choices: ['a'] }] }],
      ['a choice twice', { ...watches, fields: [{ label: 'V', question: 'q', choices: ['A', 'a'] }] }],
      ['unknown as a choice', { ...watches, fields: [{ label: 'V', question: 'q', choices: ['a', 'unknown'] }] }],
      ['a phrase without {value}', { ...watches, fields: [{ label: 'V', question: 'q', phrase: 'view' }] }],
      ['consistent free text', { ...watches, fields: [{ label: 'V', question: 'q', consistent: true }] }],
      ['an unknown preset', { ...watches, preset: 'boats' }]
    ]
    for (const [why, raw] of bad) expect(() => cleanSubject(raw), why).toThrow()
  })
})
