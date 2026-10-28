import { useMemo, useState } from 'react'
import type { LossPoint } from '@shared/training.js'

/**
 * Training loss over steps. Per-step loss is very noisy (each step draws its own
 * noise level), so the story is the moving average: that is the emphasised line,
 * and the raw values stay behind it as faint context. One hue, two weights.
 * Hovering snaps a crosshair to the nearest step; the table holds the values.
 */

const W = 560
const H = 180
const PAD = { left: 44, right: 12, top: 10, bottom: 24 }

export function movingAverage(values: number[], span: number): number[] {
  const out: number[] = []
  let sum = 0
  values.forEach((v, i) => {
    sum += v
    if (i >= span) sum -= values[i - span] ?? 0
    out.push(sum / Math.min(i + 1, span))
  })
  return out
}

/** Round tick values from 0 to at least `max`: about three clean steps. */
export function niceTicks(max: number): number[] {
  if (!(max > 0)) return [0, 1]
  const rough = max / 3
  const mag = 10 ** Math.floor(Math.log10(rough))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= rough) ?? rough
  const ticks = [0]
  while (ticks[ticks.length - 1]! < max - step * 1e-9) ticks.push(Number((ticks.length * step).toPrecision(6)))
  return ticks
}

/** Index of the point whose step is nearest `step`. */
export function nearestIndex(points: LossPoint[], step: number): number {
  let best = 0
  points.forEach((p, i) => {
    if (Math.abs(p.step - step) < Math.abs(points[best]!.step - step)) best = i
  })
  return best
}

export function LossChart({ points }: { points: LossPoint[] }): React.JSX.Element {
  const [hover, setHover] = useState<number | null>(null)
  const span = Math.max(5, Math.min(50, Math.round(points.length / 20)))
  const avg = useMemo(() => movingAverage(points.map((p) => p.loss), span), [points, span])

  const first = points[0]
  const last = points[points.length - 1]
  if (!first || !last || points.length < 2) return <p className="muted small">The loss chart appears after a few steps.</p>

  const minStep = first.step
  const maxStep = last.step
  // Scale to the average, not the spikes: a few outliers would flatten the line that matters.
  const yTicks = niceTicks(Math.max(...avg) * 1.6)
  const yMax = yTicks[yTicks.length - 1] ?? 1
  const x = (step: number) => PAD.left + ((step - minStep) / Math.max(1, maxStep - minStep)) * (W - PAD.left - PAD.right)
  const y = (v: number) => PAD.top + (1 - Math.min(v, yMax) / yMax) * (H - PAD.top - PAD.bottom)
  const path = (values: number[]) =>
    values.map((v, i) => `${i ? 'L' : 'M'}${x(points[i]?.step ?? 0).toFixed(1)},${y(v).toFixed(1)}`).join('')

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const box = e.currentTarget.getBoundingClientRect()
    const px = ((e.clientX - box.left) / box.width) * W
    setHover(nearestIndex(points, minStep + ((px - PAD.left) / (W - PAD.left - PAD.right)) * (maxStep - minStep)))
  }
  const h = hover !== null ? points[hover] : undefined
  const hAvg = hover !== null ? avg[hover] : undefined
  const every = Math.max(1, Math.round(points.length / 40))

  return (
    <figure className="loss-chart">
      <figcaption className="loss-legend">
        <span><i className="key key-avg" aria-hidden /> Average of {span} steps</span>
        <span><i className="key key-raw" aria-hidden /> Loss per step</span>
      </figcaption>
      <div className="loss-plot">
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Training loss over ${maxStep} steps`}
          onPointerMove={onMove} onPointerLeave={() => setHover(null)}>
          {yTicks.map((t) => (
            <g key={t}>
              <line className="grid" x1={PAD.left} x2={W - PAD.right} y1={y(t)} y2={y(t)} />
              <text className="tick" x={PAD.left - 6} y={y(t) + 4} textAnchor="end">{t}</text>
            </g>
          ))}
          <text className="tick" x={PAD.left} y={H - 6}>{minStep}</text>
          <text className="tick" x={W - PAD.right} y={H - 6} textAnchor="end">step {maxStep}</text>
          <path className="raw" d={path(points.map((p) => p.loss))} />
          <path className="avg" d={path(avg)} />
          {h && hAvg !== undefined && (
            <g>
              <line className="crosshair" x1={x(h.step)} x2={x(h.step)} y1={PAD.top} y2={H - PAD.bottom} />
              <circle className="dot" cx={x(h.step)} cy={y(hAvg)} r={4} />
            </g>
          )}
        </svg>
        {h && hAvg !== undefined && (
          <div className="loss-tip" style={{ left: `${(x(h.step) / W) * 100}%` }} role="status">
            <strong>Step {h.step}</strong>
            <span>Average {hAvg.toFixed(4)}</span>
            <span>This step {h.loss.toFixed(4)}</span>
          </div>
        )}
      </div>
      <details className="loss-table">
        <summary>Values</summary>
        <table>
          <thead><tr><th>Step</th><th>Loss</th><th>Average</th></tr></thead>
          <tbody>
            {points.map((p, i) => (i % every === 0 || i === points.length - 1) && (
              <tr key={p.step}><td>{p.step}</td><td>{p.loss.toFixed(4)}</td><td>{(avg[i] ?? 0).toFixed(4)}</td></tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  )
}
