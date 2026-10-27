import { useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import type { ModelInfo } from '@shared/engine-protocol.js'
import {
  analyzeUsage,
  CATEGORY_EXPLANATION,
  CATEGORY_LABEL,
  formatShare,
  formatTokens,
  USAGE_CATEGORIES,
  type Insight,
  type TurnBreakdown,
  type UsageCategory,
  type UsageReport,
  type UsageSnapshot
} from '@shared/usage.js'

/**
 * Where the context window went, and why.
 *
 * The findings come first and the charts second: most people opening this want
 * to know why the model has started forgetting things, and a sentence answers
 * that faster than a stacked bar does. The charts are there to check the
 * sentence against.
 */

function Swatch({ category }: { category: UsageCategory }): React.JSX.Element {
  return <span className={`swatch cat-${category}`} aria-hidden />
}

/** A hover tooltip that follows the pointer inside a positioned container. */
function useTooltip(): {
  tip: ReactNode
  show: (event: React.MouseEvent, content: ReactNode) => void
  hide: () => void
  anchor: React.RefObject<HTMLDivElement | null>
} {
  const anchor = useRef<HTMLDivElement>(null)
  const [state, setState] = useState<{ x: number; y: number; content: ReactNode } | null>(null)
  const show = (event: React.MouseEvent, content: ReactNode): void => {
    const box = anchor.current?.getBoundingClientRect()
    if (!box) return
    setState({ x: event.clientX - box.left, y: event.clientY - box.top, content })
  }
  const hide = (): void => setState(null)
  const tip = state ? (
    <div
      className="usage-tip"
      role="tooltip"
      style={{
        left: Math.min(state.x + 14, (anchor.current?.clientWidth ?? 0) - 240),
        top: state.y + 14
      }}
    >
      {state.content}
    </div>
  ) : null
  return { tip, show, hide, anchor }
}

function useWidth(ref: React.RefObject<HTMLElement | null>): number {
  const [width, setWidth] = useState(0)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    setWidth(el.clientWidth)
    const observer = new ResizeObserver(() => setWidth(el.clientWidth))
    observer.observe(el)
    return () => observer.disconnect()
  }, [ref])
  return width
}

function Tile({ label, value, note }: { label: string; value: string; note?: string }): React.JSX.Element {
  return (
    <div className="usage-tile">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {note ? <div className="note">{note}</div> : null}
    </div>
  )
}

/** The whole window as one bar: what is in it, in the fixed category order, and what is left. */
function WindowBar({ report }: { report: UsageReport }): React.JSX.Element {
  const { tip, show, hide, anchor } = useTooltip()
  // When the conversation has outgrown the window the bar is scaled to the
  // conversation instead, with a marker where the window ends.
  const scale = Math.max(report.used, report.contextSize)
  const parts = report.categories.filter((c) => c.tokens > 0)
  const measured = parts.reduce((sum, c) => sum + c.tokens, 0)
  const factor = measured > 0 ? report.used / measured : 0

  return (
    <div className="usage-window" ref={anchor}>
      <div className="usage-bar" onMouseLeave={hide}>
        {parts.map((c) => (
          <div
            key={c.category}
            className={`part cat-${c.category}`}
            style={{ width: `${((c.tokens * factor) / scale) * 100}%` }}
            onMouseMove={(e) =>
              show(
                e,
                <>
                  <div className="tip-head">
                    <Swatch category={c.category} /> {CATEGORY_LABEL[c.category]}
                  </div>
                  <div>
                    {c.tokens.toLocaleString('en-US')} tokens · {formatShare(c.tokens / report.contextSize)} of
                    the window
                  </div>
                </>
              )
            }
          />
        ))}
        {report.overflowing ? (
          <div
            className="usage-limit"
            style={{ left: `${(report.contextSize / scale) * 100}%` }}
            title="Where the context window ends"
          />
        ) : null}
      </div>
      <div className="usage-axis">
        <span>0</span>
        <span>
          {report.overflowing
            ? `window ends at ${report.contextSize.toLocaleString('en-US')}`
            : `${report.contextSize.toLocaleString('en-US')} tokens`}
        </span>
      </div>
      <div className="usage-legend">
        {USAGE_CATEGORIES.map((category) => {
          const total = report.categories.find((c) => c.category === category)
          if (!total || total.tokens === 0) return null
          return (
            <span key={category}>
              <Swatch category={category} />
              {CATEGORY_LABEL[category]} <b>{formatTokens(total.tokens)}</b>
            </span>
          )
        })}
        <span>
          <span className="swatch free" aria-hidden />
          Free <b>{formatTokens(report.free)}</b>
        </span>
      </div>
      {tip}
    </div>
  )
}

const TONE_ICON: Record<Insight['tone'], string> = { info: 'i', warn: '!', critical: '!!' }
const TONE_LABEL: Record<Insight['tone'], string> = {
  info: 'Note',
  warn: 'Worth a look',
  critical: 'Needs action'
}

function Insights({ insights }: { insights: Insight[] }): React.JSX.Element {
  return (
    <div className="usage-insights">
      {insights.map((insight, index) => (
        <div className={`insight ${insight.tone}`} key={index}>
          <span className="icon" aria-label={TONE_LABEL[insight.tone]}>
            {TONE_ICON[insight.tone]}
          </span>
          <div>
            <div className="title">{insight.title}</div>
            <div className="desc">{insight.body}</div>
          </div>
        </div>
      ))}
    </div>
  )
}

/** Categories ranked largest first; each row opens to say what the tokens are. */
function ByKind({ report }: { report: UsageReport }): React.JSX.Element {
  const max = report.ranked[0]?.tokens ?? 1
  return (
    <div className="usage-kinds">
      {report.ranked.map((c) => (
        <details key={c.category}>
          <summary>
            <span className="name">{CATEGORY_LABEL[c.category]}</span>
            <span className="track">
              <span className={`fill cat-${c.category}`} style={{ width: `${(c.tokens / max) * 100}%` }} />
            </span>
            <span className="num">{formatTokens(c.tokens)}</span>
            <span className="num share">{formatShare(c.share)}</span>
          </summary>
          <p>{CATEGORY_EXPLANATION[c.category]}</p>
        </details>
      ))}
    </div>
  )
}

function TopItems({ report }: { report: UsageReport }): React.JSX.Element {
  return (
    <table className="usage-table">
      <thead>
        <tr>
          <th>Item</th>
          <th>Turn</th>
          <th className="num">Tokens</th>
          <th className="num">Share</th>
        </tr>
      </thead>
      <tbody>
        {report.topItems.map((item, index) => (
          <tr key={index}>
            <td>
              <div className="item">
                <Swatch category={item.category} />
                <span>
                  <span className="kind">{CATEGORY_LABEL[item.category]}</span>{' '}
                  {item.category === 'toolCall' || item.category === 'toolResult' ? (
                    <code>{item.label}</code>
                  ) : item.category === 'system' ? (
                    item.label
                  ) : null}
                  {item.detail ? <span className="detail">{item.detail}</span> : null}
                </span>
              </div>
            </td>
            <td>{item.turn === 0 ? 'setup' : (item.turn ?? '—')}</td>
            <td className="num">{item.tokens.toLocaleString('en-US')}</td>
            <td className="num">{formatShare(item.share)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function niceStep(max: number, ticks: number): number {
  const raw = max / ticks
  const magnitude = 10 ** Math.floor(Math.log10(raw))
  const residual = raw / magnitude
  const nice = residual > 5 ? 10 : residual > 2 ? 5 : residual > 1 ? 2 : 1
  return nice * magnitude
}

const MAX_COLUMNS = 40
const PLOT_HEIGHT = 180
const AXIS_LEFT = 48
const AXIS_BOTTOM = 22
const GAP = 2

function turnTooltip(turn: TurnBreakdown): ReactNode {
  return (
    <>
      <div className="tip-head">
        Turn {turn.turn} · {turn.total.toLocaleString('en-US')} tokens added
      </div>
      {USAGE_CATEGORIES.map((category) => {
        const value = turn.byCategory[category]
        if (!value) return null
        return (
          <div className="tip-row" key={category}>
            <Swatch category={category} />
            <span>{CATEGORY_LABEL[category]}</span>
            <b>{value.toLocaleString('en-US')}</b>
          </div>
        )
      })}
      {turn.inputTokens !== null ? (
        <div className="tip-foot">
          Model read {turn.inputTokens.toLocaleString('en-US')} · wrote{' '}
          {(turn.outputTokens ?? 0).toLocaleString('en-US')}
        </div>
      ) : null}
    </>
  )
}

/** How much each turn added, stacked by kind, so the turn that blew the budget stands out. */
function PerTurn({ report }: { report: UsageReport }): React.JSX.Element {
  const { tip, show, hide, anchor } = useTooltip()
  const box = useRef<HTMLDivElement>(null)
  const width = useWidth(box)
  const turns = report.turns.slice(-MAX_COLUMNS)
  const hidden = report.turns.length - turns.length

  const maxTotal = Math.max(1, ...turns.map((t) => t.total))
  const step = niceStep(maxTotal, 4)
  const top = Math.ceil(maxTotal / step) * step
  const ticks = Array.from({ length: Math.round(top / step) + 1 }, (_, i) => i * step)

  const plotWidth = Math.max(0, width - AXIS_LEFT - 8)
  const band = turns.length > 0 ? plotWidth / turns.length : 0
  const barWidth = Math.max(4, Math.min(24, band * 0.6))
  const y = (value: number): number => PLOT_HEIGHT - (value / top) * PLOT_HEIGHT
  const peak = turns.reduce<TurnBreakdown | null>((best, t) => (!best || t.total > best.total ? t : best), null)

  return (
    <div className="usage-columns" ref={anchor}>
      <div ref={box} onMouseLeave={hide}>
        {width > 0 ? (
          <svg width={width} height={PLOT_HEIGHT + AXIS_BOTTOM + 8} role="img" aria-label="Tokens added per turn">
            <g transform="translate(0,8)">
              {ticks.map((tick) => (
                <g key={tick}>
                  <line className="grid" x1={AXIS_LEFT} x2={width - 8} y1={y(tick)} y2={y(tick)} />
                  <text className="tick" x={AXIS_LEFT - 8} y={y(tick)} dy="0.32em" textAnchor="end">
                    {formatTokens(tick)}
                  </text>
                </g>
              ))}
              {turns.map((turn, index) => {
                const cx = AXIS_LEFT + band * index + band / 2
                const x = cx - barWidth / 2
                const present = USAGE_CATEGORIES.filter((c) => (turn.byCategory[c] ?? 0) > 0)
                let base = 0
                const shapes = present.map((category, i) => {
                  const value = turn.byCategory[category] ?? 0
                  const y0 = y(base)
                  base += value
                  const y1 = y(base)
                  const isTop = i === present.length - 1
                  // Two pixels of surface between stacked segments; the data end is rounded.
                  const height = Math.max(0, y0 - y1 - (isTop ? 0 : GAP))
                  const topY = isTop ? y1 : y1 + GAP
                  if (height <= 0) return null
                  if (!isTop) {
                    return <rect key={category} className={`cat-${category}`} x={x} y={topY} width={barWidth} height={height} />
                  }
                  const r = Math.min(4, height, barWidth / 2)
                  const d = `M${x},${topY + height}V${topY + r}Q${x},${topY} ${x + r},${topY}H${x + barWidth - r}Q${x + barWidth},${topY} ${x + barWidth},${topY + r}V${topY + height}Z`
                  return <path key={category} className={`cat-${category}`} d={d} />
                })
                const showLabel = turns.length <= 16 || index % Math.ceil(turns.length / 16) === 0
                return (
                  <g key={turn.turn}>
                    {shapes}
                    {peak === turn && turn.total > 0 ? (
                      <text className="peak" x={cx} y={y(turn.total) - 6} textAnchor="middle">
                        {formatTokens(turn.total)}
                      </text>
                    ) : null}
                    {showLabel ? (
                      <text className="tick" x={cx} y={PLOT_HEIGHT + 15} textAnchor="middle">
                        {turn.turn}
                      </text>
                    ) : null}
                    <rect
                      className="hit"
                      x={cx - band / 2}
                      y={0}
                      width={band}
                      height={PLOT_HEIGHT}
                      onMouseMove={(e) => show(e, turnTooltip(turn))}
                    />
                  </g>
                )
              })}
              <line className="baseline" x1={AXIS_LEFT} x2={width - 8} y1={PLOT_HEIGHT} y2={PLOT_HEIGHT} />
            </g>
          </svg>
        ) : null}
      </div>
      <div className="usage-axis">
        <span>{hidden > 0 ? `Last ${turns.length} turns shown; ${hidden} earlier in the table` : 'Turn'}</span>
        <span>Template markup is spread over every turn and not included here</span>
      </div>
      {tip}
      <details className="usage-tableview">
        <summary>Table view</summary>
        <table className="usage-table">
          <thead>
            <tr>
              <th>Turn</th>
              {USAGE_CATEGORIES.filter((c) => c !== 'format' && c !== 'system' && c !== 'tools').map((c) => (
                <th className="num" key={c}>
                  {CATEGORY_LABEL[c]}
                </th>
              ))}
              <th className="num">Added</th>
              <th className="num">Model read</th>
              <th className="num">Model wrote</th>
            </tr>
          </thead>
          <tbody>
            {report.turns.map((t) => (
              <tr key={t.turn}>
                <td>{t.turn}</td>
                {USAGE_CATEGORIES.filter((c) => c !== 'format' && c !== 'system' && c !== 'tools').map((c) => (
                  <td className="num" key={c}>
                    {(t.byCategory[c] ?? 0).toLocaleString('en-US')}
                  </td>
                ))}
                <td className="num">{t.total.toLocaleString('en-US')}</td>
                <td className="num">{t.inputTokens?.toLocaleString('en-US') ?? '—'}</td>
                <td className="num">{t.outputTokens?.toLocaleString('en-US') ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  )
}

export function UsagePanel({
  snapshot,
  model,
  onRefresh,
  onNewChat
}: {
  snapshot: UsageSnapshot | null
  model: ModelInfo | null
  onRefresh: () => void
  onNewChat: () => void
}): React.JSX.Element {
  const report = useMemo(() => (snapshot ? analyzeUsage(snapshot) : null), [snapshot])

  if (!model || !report) {
    return (
      <div className="panel">
        <div className="empty" style={{ marginTop: 80 }}>
          {model
            ? 'Measuring…'
            : 'Load a model to see where its context window goes. Every figure is measured with the model’s own tokenizer.'}
        </div>
      </div>
    )
  }

  const hasTurns = report.turns.length > 0

  return (
    <div className="panel usage">
      <section>
        <div className="row usage-head">
          <div>
            <h3>Token usage</h3>
            <p className="help">
              What the current conversation takes up in the model’s context window, measured with its own
              tokenizer after every reply. When the window fills, the oldest part stops being seen.
            </p>
          </div>
          <div className="grow" />
          <button onClick={onRefresh}>Measure again</button>
          <button onClick={onNewChat} disabled={!hasTurns}>
            Start a new chat
          </button>
        </div>

        <div className="usage-tiles">
          <Tile
            label="Context used"
            value={formatShare(report.fill)}
            note={`${report.used.toLocaleString('en-US')} of ${report.contextSize.toLocaleString('en-US')} tokens`}
          />
          <Tile label="Free" value={formatTokens(report.free)} note="tokens left in the window" />
          <Tile
            label="Turns left"
            value={report.turnsLeft === null ? '—' : report.overflowing ? '0' : `~${report.turnsLeft}`}
            note={
              report.growthPerTurn === null
                ? 'after the first reply'
                : `at ~${formatTokens(report.growthPerTurn)} tokens a turn`
            }
          />
          <Tile
            label="Model wrote"
            value={formatTokens(report.generatedTokens)}
            note={`and read ${formatTokens(report.processedTokens)} to do it`}
          />
        </div>

        <WindowBar report={report} />
      </section>

      <section>
        <h3>What is taking up the room</h3>
        <Insights insights={report.insights} />
      </section>

      <section className="usage-split">
        <div>
          <h3>By kind</h3>
          <p className="help">Open a row for what those tokens are and what can be done about them.</p>
          <ByKind report={report} />
        </div>
        <div>
          <h3>Largest single items</h3>
          <p className="help">Everything here stays in the conversation for every turn after it.</p>
          <TopItems report={report} />
        </div>
      </section>

      {hasTurns ? (
        <section>
          <h3>What each turn added</h3>
          <p className="help">Hover a column for the breakdown and how much the model read and wrote.</p>
          <PerTurn report={report} />
        </section>
      ) : null}

      {report.tools.length > 0 ? (
        <section>
          <h3>By tool</h3>
          <table className="usage-table narrow">
            <thead>
              <tr>
                <th>Tool</th>
                <th className="num">Calls</th>
                <th className="num">Tokens</th>
                <th className="num">Per call</th>
              </tr>
            </thead>
            <tbody>
              {report.tools.map((t) => (
                <tr key={t.tool}>
                  <td>
                    <code>{t.tool}</code>
                  </td>
                  <td className="num">{t.calls}</td>
                  <td className="num">{t.tokens.toLocaleString('en-US')}</td>
                  <td className="num">{t.calls > 0 ? formatTokens(t.tokens / t.calls) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ) : null}
    </div>
  )
}

/** The small always-visible meter in the top bar. */
export function ContextMeter({
  snapshot,
  onOpen
}: {
  snapshot: UsageSnapshot | null
  onOpen: () => void
}): React.JSX.Element | null {
  const report = useMemo(() => (snapshot ? analyzeUsage(snapshot) : null), [snapshot])
  if (!report) return null
  const tone = report.fill >= 0.9 ? 'critical' : report.fill >= 0.7 ? 'warn' : ''
  return (
    <button
      className={`context-meter ${tone}`}
      onClick={onOpen}
      title={`${formatTokens(report.used)} of ${formatTokens(report.contextSize)} tokens used. Open the token usage breakdown.`}
    >
      <span className="mini">
        <span style={{ width: `${Math.min(1, report.fill) * 100}%` }} />
      </span>
      <span className="cm-count">
        {formatTokens(report.used)} / {formatTokens(report.contextSize)} ·
      </span>
      {formatShare(report.fill)}
    </button>
  )
}
