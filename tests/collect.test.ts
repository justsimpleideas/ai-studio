import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { describe, expect, it } from 'vitest'
import { cleanSubject } from '@shared/subject.js'
import {
  COLLECT_PROJECT_URL,
  collectUserAgent,
  IDLE_COLLECT,
  cleanCollectPlan,
  feedCollect,
  parseCheckEvent,
  parseCollectEvent,
  type CollectPlan,
  type CollectState
} from '@shared/training.js'
import { PhotoCollector } from '../src/main/image/collect.js'
import type { Runner } from '../src/main/image/setup.js'

const GB = 1024 ** 3
const subject = cleanSubject({ noun: 'wristwatch', levels: [{ label: 'Brand', required: true }, { label: 'Model', required: true }, { label: 'Reference' }] })
const target = { values: ['Example Watch Co', 'Diver', ''], query: '', category: '', count: 200 }
const plan: CollectPlan = {
  project: 'Five watches', subject, targets: [target], max_total_bytes: 2 * GB, sources: ['commons', 'openverse'],
  share_alike: false, contact: ''
}

describe('collection plans', () => {
  it('passes a good plan through, cleaned', () => {
    expect(cleanCollectPlan({ ...plan, targets: [{ ...target, values: ['  Example   Watch Co ', 'Diver', ''] }], extra: 1 }))
      .toEqual({ ...plan, targets: [target] })
    expect(cleanCollectPlan({ ...plan, targets: [{ ...target, query: ' diver  watch ' }] }).targets[0]!.query).toBe('diver watch')
    expect(cleanCollectPlan({ ...plan, sources: ['commons', 'commons'] }).sources).toEqual(['commons'])
    expect(cleanCollectPlan({ ...plan, share_alike: true, contact: ' you@example.com ' }))
      .toMatchObject({ share_alike: true, contact: 'you@example.com' })
  })

  it('refuses anything else', () => {
    const bad: Record<string, unknown>[] = [
      { targets: [] }, { targets: Array(11).fill(target) }, { targets: [{ ...target, values: ['', 'Diver', ''] }] },
      { targets: [{ ...target, values: ['A', 'B'] }] }, { targets: [{ ...target, values: 'A B' }] },
      { targets: [{ ...target, count: 0 }] }, { targets: [{ ...target, count: 1001 }] }, { targets: [target, target] },
      { targets: [{ ...target, values: ['A', 'x'.repeat(61), ''] }] }, { targets: [{ ...target, query: 'x'.repeat(61) }] },
      { max_total_bytes: 3 * GB }, { max_total_bytes: '2' }, { sources: [] }, { sources: ['google'] }, { project: '' },
      { subject: undefined }, { subject: { ...subject, levels: [] } },
      { share_alike: 'yes' }, { share_alike: undefined }, { contact: 'a\r\nX-Evil: 1' }, { contact: 'a;b' },
      { contact: 'x'.repeat(101) }, { contact: 5 }
    ]
    for (const change of bad) expect(() => cleanCollectPlan({ ...plan, ...change }), JSON.stringify(change)).toThrow()
  })
})

describe('collection progress', () => {
  it('parses only JSON event lines', () => {
    expect(parseCollectEvent('{"event": "done", "photos": 3}')).toEqual({ event: 'done', photos: 3 })
    for (const line of ['Coupe: 0 of 10 already here', '{"no": "event"}', '{broken', '[1]']) expect(parseCollectEvent(line)).toBeNull()
  })

  it('folds events into what the window shows', () => {
    let s: CollectState = IDLE_COLLECT
    s = feedCollect(s, { event: 'start', targets: ['A', 'B'], bytes: 0, max_total_bytes: GB })
    s = feedCollect(s, { event: 'walking', index: 0, label: 'A', categories: 40 })
    expect(s.walking).toEqual({ label: 'A', categories: 40 })
    s = feedCollect(s, { event: 'target', index: 0, label: 'A', have: 0, want: 2 })
    expect(s.current).toBe(0)
    expect(s.walking).toBeNull()
    s = feedCollect(s, { event: 'saved', index: 0, have: 1, want: 2, bytes: 500 })
    s = feedCollect(s, { event: 'saved', index: 0, have: 2, want: 2, bytes: 900 })
    s = feedCollect(s, { event: 'target_done', index: 0, have: 2, want: 2 })
    expect(s).toMatchObject({ phase: 'running', photos: 2, bytes: 900, maxBytes: GB, current: null })
    expect(s.targets).toEqual([{ label: 'A', have: 2, want: 2 }, { label: 'B', have: 0, want: 0 }])
    s = feedCollect(s, { event: 'done', photos: 2, bytes: 900, skipped: { licence: 3, bad: 'x' }, size_limit_reached: true })
    expect(s).toMatchObject({ phase: 'done', skipped: { licence: 3 }, sizeLimitReached: true })
    expect(feedCollect(s, { event: 'error', message: 'Invalid plan' })).toMatchObject({ phase: 'failed', message: 'Invalid plan' })
    expect(feedCollect(s, { event: 'mystery' })).toBe(s)
  })
})

describe('checking what is available', () => {
  it('reads check events, keeping only known fields', () => {
    const event = {
      event: 'check', index: 0, label: 'Example Watch Co Diver', want: 200,
      commons: { mode: 'category', found: 8, categories: 3, skipped: ['Category:X logos', 4], truncated: false,
        sampled: 8, kept: 5, estimate: 5, extra: 'dropped' },
      openverse: { error: 'openverse did not answer (timed out)' }
    }
    expect(parseCheckEvent(event)).toEqual({
      label: 'Example Watch Co Diver', want: 200,
      commons: { mode: 'category', found: 8, categories: 3, skipped: ['Category:X logos'], sampled: 8, kept: 5, estimate: 5 },
      openverse: { found: 0, sampled: 0, kept: 0, estimate: 0, error: 'openverse did not answer (timed out)' }
    })
    expect(parseCheckEvent({ event: 'saved' })).toBeNull()
  })

  it('names the app to the services exactly as the script does', () => {
    const script = fs.readFileSync(path.join(__dirname, '..', 'image-engine', 'scripts', 'collect_photos.py'), 'utf8')
    expect(script).toContain(`PROJECT_URL = "${COLLECT_PROJECT_URL}"`)
    expect(script).toContain('return f"LocalAIStudio-collect/1.0 ({contact or PROJECT_URL}; dataset tool for personal training)"')
    expect(collectUserAgent()).toBe(`LocalAIStudio-collect/1.0 (${COLLECT_PROJECT_URL}; dataset tool for personal training)`)
    expect(collectUserAgent(' you@example.com ')).toBe('LocalAIStudio-collect/1.0 (you@example.com; dataset tool for personal training)')
  })
})

describe('the collector', () => {
  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'collect-test-'))

  function collector(run: Runner, free: number | null = 100 * GB) {
    const states: CollectState[] = []
    const made: string[] = []
    const c = new PhotoCollector({
      python: () => 'python', script: 'collect_photos.py', workDir, run,
      freeBytes: async () => free,
      apis: { commons: 'http://127.0.0.1:1/commons' },
      makeProject: async (name, folder, s) => (made.push(`${name}@${folder}:${s.noun}`), 'abcdef0123456789'),
      onState: (s) => states.push(s)
    })
    return { c, states, made }
  }

  it('runs the script with the plan, reads its progress, then makes the project', async () => {
    let seen: string[] = []
    const run: Runner = async (_cmd, args, options) => {
      seen = args
      const written = JSON.parse(fs.readFileSync(args[args.indexOf('--plan') + 1]!, 'utf8')) as { levels: unknown[]; targets: unknown[]; share_alike: boolean; contact: string }
      expect(written.levels).toEqual([{ key: 'brand', required: true }, { key: 'model', required: true }, { key: 'reference', required: false }])
      expect(written.targets).toEqual([target])
      expect(written).toMatchObject({ share_alike: false, contact: '' })
      options?.onOutput?.('{"event": "start", "targets": ["Example Motors Coupe GT Mk2"], "bytes": 0, "max_total_bytes": 2}\n{"event": "saved", "ind', 'stdout')
      options?.onOutput?.('ex": 0, "have": 1, "want": 200, "bytes": 10}\nnot json\n', 'stdout')
      options?.onOutput?.('{"event": "done", "photos": 1, "bytes": 10, "skipped": {}}\n', 'stdout')
      return { code: 0, stdout: '' }
    }
    const { c, states, made } = collector(run)
    await c.collect(plan, 'C:/photos')
    expect(seen).toEqual(expect.arrayContaining(['--dest', 'C:/photos', '--json', '--commons-api', 'http://127.0.0.1:1/commons']))
    expect(seen).not.toContain('--openverse-api')
    expect(made).toEqual(['Five watches@C:/photos:wristwatch'])
    expect(states[0]!.targets).toEqual([{ label: 'Example Watch Co Diver', have: 0, want: 200 }])
    expect(c.state).toMatchObject({ phase: 'done', photos: 1, projectId: 'abcdef0123456789' })
    expect(c.state.message).toContain('1 photos collected')
    expect(states.some((s) => s.targets[0]?.have === 1)).toBe(true)
    expect(c.running).toBe(false)
  })

  it('checks without a folder and without downloading, returning what the script found', async () => {
    let seen: string[] = []
    const { c, made, states } = collector(async (_cmd, args, options) => {
      seen = args
      options?.onOutput?.('{"event": "walking", "index": 0, "label": "Example Watch Co Diver", "categories": 12}\n', 'stdout')
      options?.onOutput?.('{"event": "check", "index": 0, "label": "Example Watch Co Diver", "want": 200, ' +
        '"commons": {"mode": "search", "found": 40, "sampled": 40, "kept": 30, "estimate": 30}}\n{"event": "checked", "targets": 1}\n', 'stdout')
      return { code: 0, stdout: '' }
    })
    const results = await c.check(plan)
    expect(seen).toContain('--check')
    expect(seen).not.toContain('--dest')
    expect(seen[seen.indexOf('--cache') + 1]).toBe(path.join(workDir, 'trees'))
    expect(states.some((s) => s.walking?.categories === 12)).toBe(true)
    expect(c.state.walking).toBeNull()
    expect(results).toEqual([{ label: 'Example Watch Co Diver', want: 200, commons: { mode: 'search', found: 40, sampled: 40, kept: 30, estimate: 30 } }])
    expect(made).toEqual([])
    expect(c.running).toBe(false)
    const failing = collector(async (_c, _a, o) => {
      o?.onOutput?.('{"event": "error", "message": "Invalid plan: give 1 to 10 targets"}\n', 'stdout')
      return { code: 2, stdout: '' }
    })
    await expect(failing.c.check(plan)).rejects.toThrow('Invalid plan')
  })

  it('refuses when the drive cannot hold the size limit', async () => {
    const { c, made } = collector(async () => ({ code: 0, stdout: '' }), GB)
    await expect(c.collect(plan, 'C:/photos')).rejects.toThrow('Not enough free space')
    expect(made).toEqual([])
    expect(c.state.phase).toBe('failed')
  })

  it('reports the script failing, and does not make a project', async () => {
    const { c, made } = collector(async (_c, _a, o) => {
      o?.onOutput?.('{"event": "error", "message": "Invalid plan: give 1 to 10 targets"}\n', 'stdout')
      return { code: 2, stdout: '' }
    })
    await expect(c.collect(plan, 'C:/photos')).rejects.toThrow('Invalid plan')
    expect(made).toEqual([])
  })

  it('stopping keeps it resumable and makes no project', async () => {
    let collectorRef: PhotoCollector | null = null
    const { c, made } = collector(async (_c, _a, o) => {
      collectorRef!.cancel()
      expect(o?.signal?.aborted).toBe(true)
      return { code: null, stdout: '' }
    })
    collectorRef = c
    await c.collect(plan, 'C:/photos')
    expect(c.state.phase).toBe('cancelled')
    expect(c.state.message).toContain('resumes')
    expect(made).toEqual([])
  })

  it('one collection at a time', async () => {
    let release: () => void = () => undefined
    const { c } = collector(() => new Promise((done) => (release = () => done({ code: 0, stdout: '' }))))
    const first = c.collect(plan, 'C:/a')
    await new Promise((r) => setTimeout(r, 10))
    await expect(c.collect(plan, 'C:/b')).rejects.toThrow('already')
    release()
    await first.catch(() => undefined)
  })
})
