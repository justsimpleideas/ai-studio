import type { ImageEngineState } from '@shared/image.js'
import { formatBytes } from '../format.js'
import type { Status } from '../types.js'

interface Props {
  engine: ImageEngineState
  connected: boolean
  status: Status | null
}

export function StatusChip({ engine, connected, status }: Props): React.JSX.Element {
  if (engine.state === 'starting' || (engine.state === 'running' && !status)) {
    return <div className="status-chip loading"><span className="dot" />Starting image engine</div>
  }
  if (!connected || !status) {
    return <div className="status-chip offline"><span className="dot" />Image engine not running</div>
  }
  const { device } = status
  // The engine holds one model at a time; show the video model whenever it is the one in play.
  const video = status.models?.video
  const showVideo = video !== undefined && video.state !== 'unloaded'
  const model = showVideo ? video : status.model
  const noun = showVideo ? 'Video' : 'Image'
  const label = {
    unloaded: `${noun} model not loaded`,
    loading: `Loading ${noun.toLowerCase()} model`,
    ready: `${noun} model ready`,
    error: `${noun} model error`
  }[model.state]
  const used = device.total_bytes ? device.used_bytes / device.total_bytes : 0
  return (
    <div className={`status-chip ${model.state}`} title={model.message || label}>
      <span className="dot" />
      <span>{label}</span>
      {device.cuda && (
        <span className="vram" title={`${formatBytes(device.used_bytes)} of ${formatBytes(device.total_bytes)} VRAM in use`}>
          <span className="vram-bar"><span style={{ width: `${Math.min(100, used * 100)}%` }} /></span>
          <span className="muted small tabular">{formatBytes(device.used_bytes)} / {formatBytes(device.total_bytes)}</span>
        </span>
      )}
    </div>
  )
}
