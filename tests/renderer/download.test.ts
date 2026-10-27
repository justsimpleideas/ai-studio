import { describe, expect, it } from 'vitest'
import type { ModelDownloadState } from '../../src/shared/setup.js'
import { downloadSummary } from '../../src/renderer/image/download.js'

const base: ModelDownloadState = {
  phase: 'downloading',
  id: 'image',
  dest: 'D',
  doneBytes: 12.34e9,
  totalBytes: 32.85e9,
  file: 'transformer/x.safetensors',
  activity: 'downloading',
  bytesPerSecond: 20e6,
  message: ''
}

describe('the line under a download', () => {
  it('says how far, how fast and how long', () => {
    expect(downloadSummary(base)).toBe('12.3 of 32.9 GB · 20.0 MB/s · about 17 min left')
    expect(downloadSummary({ ...base, bytesPerSecond: 2e6 })).toBe('12.3 of 32.9 GB · 2.0 MB/s · about 2.8 h left')
    expect(downloadSummary({ ...base, bytesPerSecond: null })).toBe('12.3 of 32.9 GB')
  })

  it('names the file being checked or linked instead of a speed', () => {
    expect(downloadSummary({ ...base, activity: 'verifying', bytesPerSecond: null })).toBe('12.3 of 32.9 GB · checking x.safetensors')
    expect(downloadSummary({ ...base, activity: 'linking' })).toBe('12.3 of 32.9 GB · linking x.safetensors')
  })

  it('shows how it ended', () => {
    expect(downloadSummary({ ...base, phase: 'failed', message: 'Verification failed: x' })).toBe('Verification failed: x')
    expect(downloadSummary({ ...base, phase: 'idle' })).toBe('')
  })
})
