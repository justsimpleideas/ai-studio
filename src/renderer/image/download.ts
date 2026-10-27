import type { ModelDownloadState } from '@shared/setup.js'

const gb = (bytes: number): string => (bytes / 1e9).toFixed(1)

/** One line under a download: how far, how fast, what it is doing, or how it ended. */
export function downloadSummary(state: ModelDownloadState): string {
  switch (state.phase) {
    case 'idle':
      return ''
    case 'starting':
      return state.message || 'Starting…'
    case 'done':
    case 'failed':
    case 'cancelled':
      return state.message
  }
  const parts = [`${gb(state.doneBytes)} of ${gb(state.totalBytes)} GB`]
  if (state.activity === 'verifying') parts.push(`checking ${fileName(state.file)}`)
  else if (state.activity === 'linking') parts.push(`linking ${fileName(state.file)}`)
  else if (state.bytesPerSecond && state.bytesPerSecond > 0) {
    parts.push(`${(state.bytesPerSecond / 1e6).toFixed(1)} MB/s`)
    const left = (state.totalBytes - state.doneBytes) / state.bytesPerSecond
    parts.push(left < 60 ? 'under a minute left' : left < 5400 ? `about ${Math.round(left / 60)} min left` : `about ${(left / 3600).toFixed(1)} h left`)
  }
  if (state.message) parts.push(state.message)
  return parts.join(' · ')
}

function fileName(file: string | null): string {
  return file ? (file.split('/').pop() ?? file) : 'files'
}
