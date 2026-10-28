import { describe, expect, it } from 'vitest'
import { engineImagePath, type ImageJob } from '@shared/image.js'
import {
  DEFAULT_TRAIN_SETTINGS,
  cleanCompareRequest,
  cleanEdits,
  cleanName,
  cleanTrainSettings,
  describeFlag,
  isProjectId,
  isRunId,
  studioTrainingPhotoUrl,
  taskOf
} from '@shared/training.js'

const P = '0123456789abcdef'
const R = '0123456789ab'
const L = 'fedcba9876543210'

describe('training ids and photo URLs', () => {
  it('accepts only engine-shaped ids', () => {
    expect(isProjectId(P)).toBe(true)
    expect(isRunId(R)).toBe(true)
    for (const bad of [R, P.toUpperCase(), '../etc', '', 5, null]) expect(isProjectId(bad)).toBe(false)
    for (const bad of [P, 'zzzzzzzzzzzz', '']) expect(isRunId(bad)).toBe(false)
  })

  it('maps a training photo URL to its engine route and nothing else', () => {
    expect(engineImagePath(studioTrainingPhotoUrl(P, L))).toBe(`/training/${P}/${L}.webp`)
    for (const bad of [
      `studio-image://training/${P}/../${L}.webp`,
      `studio-image://training/${P}/${L}.png`,
      `studio-image://training/${P}/${L}.webp?x=1`,
      `studio-image://training/${P}`
    ]) {
      expect(engineImagePath(bad), bad).toBeNull()
    }
  })
})

describe('training settings', () => {
  it('passes good settings through, field by field', () => {
    expect(cleanTrainSettings({ ...DEFAULT_TRAIN_SETTINGS, extra: 'dropped' })).toEqual(DEFAULT_TRAIN_SETTINGS)
    expect(cleanTrainSettings({ ...DEFAULT_TRAIN_SETTINGS, run: R }).run).toBe(R)
  })

  it('refuses anything out of range or mistyped', () => {
    const bad: Record<string, unknown>[] = [
      { rank: 0 }, { rank: 1.5 }, { learning_rate: 0.5 }, { steps: '100' }, { steps: 20001 },
      { checkpoint_every: 0 }, { caption_dropout: 0.9 }, { seed: -1 }, { target_side: 640 }, { run: '../x' }
    ]
    for (const change of bad) {
      expect(() => cleanTrainSettings({ ...DEFAULT_TRAIN_SETTINGS, ...change }), JSON.stringify(change)).toThrow()
    }
    expect(() => cleanTrainSettings(null)).toThrow()
  })

  it('checks comparison requests', () => {
    const good = { run: R, steps: [250, 500], prompts: [' a coupe '], size: 768, seed: 1 }
    expect(cleanCompareRequest(good)).toEqual({ ...good, prompts: ['a coupe'] })
    for (const change of [{ run: P }, { steps: [] }, { steps: [1, 2, 3, 4, 5] }, { steps: [0] }, { prompts: ['', 'x'] },
      { prompts: ['a', 'b', 'c', 'd', 'e'] }, { size: 640 }, { seed: -1 }]) {
      expect(() => cleanCompareRequest({ ...good, ...change }), JSON.stringify(change)).toThrow()
    }
  })

  it('checks caption edits and names', () => {
    expect(cleanEdits(undefined)).toBeUndefined()
    expect(cleanEdits({ view: 'side', dial_colour: 'red', χρώμα: 'red' })).toEqual({ view: 'side', dial_colour: 'red', χρώμα: 'red' })
    const many = Object.fromEntries(Array.from({ length: 11 }, (_, i) => [`f${i}`, 'x']))
    for (const bad of [null, [], { 'a b': 'x' }, { '../x': 'y' }, { view: 3 }, { colour: 'x'.repeat(61) }, many]) {
      expect(() => cleanEdits(bad), JSON.stringify(bad)).toThrow()
    }
    expect(cleanName('  Coupe   v1 ', 'Name')).toBe('Coupe v1')
    for (const bad of ['', '   ', 'x'.repeat(61), 3]) expect(() => cleanName(bad, 'Name')).toThrow()
  })
})

describe('task jobs and flags', () => {
  const job = (request: unknown, kind?: string): ImageJob => ({ kind, request } as unknown as ImageJob)

  it('recognises training tasks only', () => {
    expect(taskOf(job({ kind: 'task', task: 'training.train', project: P }, 'task'))).toEqual({ task: 'training.train', project: P })
    expect(taskOf(job({ prompt: 'x' }, 'image'))).toBeNull()
    expect(taskOf(job({ prompt: 'x' }))).toBeNull()
  })

  it('describes flags in words', () => {
    expect(describeFlag('disagree:view')).toBe('two answers differ on view')
    expect(describeFlag('outlier:body_style')).toBe('body style differs from most photos of this subject')
    expect(describeFlag('single_pass')).toBe('only one answer to check')
    expect(describeFlag('off_subject:subject_shown')).toBe('not the real subject')
    expect(describeFlag('off_subject:subject_size')).toBe('subject too small in the photo')
    expect(describeFlag('partial:subject_shown')).toBe('only part of the subject, or its inside')
    expect(describeFlag('something_new')).toBe('something_new')
  })
})
