import { describe, expect, it } from 'vitest'
import { cleanLoras, DEFAULT_FORM, fromImage, loadForm, toRequest } from '../../src/renderer/image/form.js'
import type { Subject } from '@shared/subject.js'
import { movingAverage, nearestIndex, niceTicks } from '../../src/renderer/image/training/LossChart.js'
import { fromDraft } from '../../src/renderer/image/training/SubjectEditor.js'

const P = '0123456789abcdef'
const L = 'fedcba9876543210'

describe('LoRAs in the image form', () => {
  it('adds loras to the request only when there are some, and only known ones', () => {
    expect(toRequest({ ...DEFAULT_FORM, prompt: 'x' })).not.toHaveProperty('loras')
    const form = { ...DEFAULT_FORM, prompt: 'x', loras: [{ id: L, strength: 0.8 }, { id: P, strength: 1 }] }
    expect(toRequest(form).loras).toEqual(form.loras)
    expect(toRequest(form, new Set([L])).loras).toEqual([{ id: L, strength: 0.8 }])
    expect(toRequest(form, new Set())).not.toHaveProperty('loras')
  })

  it('cleans stored or reused values', () => {
    expect(cleanLoras('x')).toEqual([])
    expect(cleanLoras([{ id: L, strength: 3 }, { id: 'bad', strength: 1 }, { id: P, strength: 1 }, { id: P, strength: 0.5 }]))
      .toEqual([{ id: P, strength: 1 }])
    const many = ['a', 'b', 'c', 'd'].map((c) => ({ id: c.repeat(16), strength: 1 }))
    expect(cleanLoras(many)).toHaveLength(3)
  })

  it('reuse takes the loras from provenance; storage keeps them', () => {
    const record = { prompt: 'p', negative_prompt: '', width: 512, height: 512, steps: 8, guidance: 0, seed: 3,
      provenance: { loras: [{ id: L, name: 'n', sha256: 'h', strength: 0.6 }] } }
    expect(fromImage(record as never, DEFAULT_FORM).loras).toEqual([{ id: L, strength: 0.6 }])
    const stored = new Map([['lis.form.v1', JSON.stringify({ ...DEFAULT_FORM, loras: [{ id: L, strength: 1 }, 'junk'] })]])
    expect(loadForm({ getItem: (k: string) => stored.get(k) ?? null }).loras).toEqual([{ id: L, strength: 1 }])
  })
})

describe('loss chart maths', () => {
  it('moving average over a window, warming up at the start', () => {
    expect(movingAverage([2, 4, 6, 8], 2)).toEqual([2, 3, 5, 7])
    expect(movingAverage([], 5)).toEqual([])
  })

  it('round ticks that reach the maximum', () => {
    expect(niceTicks(0.9)).toEqual([0, 0.5, 1])
    expect(niceTicks(0.12)).toEqual([0, 0.05, 0.1, 0.15])
    expect(niceTicks(0)).toEqual([0, 1])
    for (const max of [0.033, 1.7, 42]) {
      const t = niceTicks(max)
      expect(t[t.length - 1]).toBeGreaterThanOrEqual(max)
      expect(t.length).toBeLessThanOrEqual(5)
    }
  })

  it('snaps to the nearest step', () => {
    const pts = [1, 5, 10].map((step) => ({ step, loss: 0, sigma: 0, sample: 0, dropped: false }))
    expect(nearestIndex(pts, 0)).toBe(0)
    expect(nearestIndex(pts, 6)).toBe(1)
    expect(nearestIndex(pts, 99)).toBe(2)
  })
})

describe('the subject editor', () => {
  it('reports a checked definition, or the reason it is not one', () => {
    const draft = {
      preset: 'custom' as const, noun: 'dog', trigger: '',
      levels: [{ label: 'Breed', required: true, in_caption: true }],
      fields: [{ label: 'Pose', question: 'how the {noun} stands', choices: 'sitting, standing , ,lying', important: false, consistent: false, phrase: '{value}' }]
    }
    const ok = fromDraft(draft)
    expect(ok.error).toBeNull()
    expect((ok.subject as Subject).fields[0]!.choices).toEqual(['sitting', 'standing', 'lying'])
    const bad = fromDraft({ ...draft, levels: [{ label: '', required: true, in_caption: true }] })
    expect(bad.subject).toBeNull()
    expect(bad.error).toContain('Folder level 1')
  })
})
