import { useEffect, useReducer } from 'react'
import type { ImageEngineState } from '@shared/image.js'
import { api } from './api.js'
import type { EngineEvent, ImageRecord, Job, MediaKind, Status } from './types.js'

export interface EngineState {
  engine: ImageEngineState
  /** True once the engine runs and its status is known. */
  connected: boolean
  status: Status | null
  jobs: Record<string, Job>
  /** Images produced since the window was opened, newest first. */
  session: ImageRecord[]
  /** Increments whenever the gallery changed on the engine side. */
  galleryVersion: number
}

export type EngineAction =
  | { type: 'event'; event: EngineEvent }
  | { type: 'engine'; engine: ImageEngineState }
  | { type: 'status'; status: Status }
  | { type: 'jobs'; jobs: Job[] }
  | { type: 'forget-image'; id: string }
  | { type: 'replace-image'; image: ImageRecord }

export const initialEngineState: EngineState = {
  engine: { state: 'stopped' },
  connected: false,
  status: null,
  jobs: {},
  session: [],
  galleryVersion: 0
}

const SESSION_LIMIT = 48

export function engineReducer(state: EngineState, action: EngineAction): EngineState {
  switch (action.type) {
    case 'engine': {
      const running = action.engine.state === 'running'
      return { ...state, engine: action.engine, connected: running && state.status !== null, ...(running ? {} : { jobs: {} }) }
    }
    case 'status':
      return { ...state, status: action.status, connected: state.engine.state === 'running' }
    case 'jobs':
      return { ...state, jobs: Object.fromEntries(action.jobs.map((j) => [j.id, j])) }
    case 'forget-image':
      return { ...state, session: state.session.filter((i) => i.id !== action.id), galleryVersion: state.galleryVersion + 1 }
    case 'replace-image':
      return { ...state, session: state.session.map((i) => (i.id === action.image.id ? action.image : i)) }
    case 'event': {
      const e = action.event
      switch (e.type) {
        case 'hello':
          return { ...state, status: e.status, jobs: Object.fromEntries(e.jobs.map((j) => [j.id, j])) }
        case 'job':
          return { ...state, jobs: { ...state.jobs, [e.job.id]: e.job } }
        case 'image':
          return {
            ...state,
            session: [e.image, ...state.session.filter((i) => i.id !== e.image.id)].slice(0, SESSION_LIMIT),
            galleryVersion: state.galleryVersion + 1
          }
        case 'model': {
          if (!state.status) return state
          const kind = e.kind ?? 'image'
          const before = state.status.models?.[kind] ?? (kind === 'image' ? state.status.model : { id: kind, state: 'unloaded', message: '' })
          const info = { ...before, state: e.state, message: e.message }
          const models = { ...state.status.models, [kind]: info }
          return { ...state, status: { ...state.status, models, ...(kind === 'image' ? { model: info } : {}) } }
        }
      }
    }
  }
  return state
}

export function jobKind(job: Job): MediaKind {
  return job.kind ?? 'image'
}

/**
 * The job the UI should show progress for: the running one, else the oldest queued
 * one. With a kind, only jobs of that kind count (each tab shows its own).
 */
export function activeJob(jobs: Record<string, Job>, kind?: MediaKind): Job | null {
  const list = Object.values(jobs).filter((j) => !kind || jobKind(j) === kind)
  const running = list.find((j) => j.status === 'running')
  if (running) return running
  const queued = list.filter((j) => j.status === 'queued').sort((a, b) => a.created - b.created)
  return queued[0] ?? null
}

export function queuedCount(jobs: Record<string, Job>): number {
  return Object.values(jobs).filter((j) => j.status === 'queued' || j.status === 'running').length
}

/**
 * Engine state for the image UI. Starts the engine the first time the image side
 * is opened; Python and PyTorch take a few seconds to come up, so it is not
 * started with the app.
 */
export function useEngine(active: boolean): [EngineState, React.Dispatch<EngineAction>] {
  const [state, dispatch] = useReducer(engineReducer, initialEngineState)

  useEffect(() => {
    const offEvent = window.api.image.onEvent((event) => dispatch({ type: 'event', event }))
    const offState = window.api.image.onEngineState((engine) => dispatch({ type: 'engine', engine }))
    void window.api.image.engineState().then((engine) => dispatch({ type: 'engine', engine }))
    return () => {
      offEvent()
      offState()
    }
  }, [])

  useEffect(() => {
    if (active && state.engine.state === 'stopped') void api.start().catch(() => undefined)
  }, [active, state.engine.state])

  // Catch up whenever the engine (re)starts: events emitted before this window
  // subscribed are not replayed.
  useEffect(() => {
    if (state.engine.state !== 'running') return
    void api.status().then((status) => dispatch({ type: 'status', status })).catch(() => undefined)
    void api.jobs().then((jobs) => dispatch({ type: 'jobs', jobs })).catch(() => undefined)
  }, [state.engine.state])

  // Memory use changes while generating; refresh device numbers now and then.
  useEffect(() => {
    if (!active || state.engine.state !== 'running') return
    const timer = window.setInterval(() => {
      void api.status().then((status) => dispatch({ type: 'status', status })).catch(() => undefined)
    }, 4000)
    return () => window.clearInterval(timer)
  }, [active, state.engine.state])

  return [state, dispatch]
}
