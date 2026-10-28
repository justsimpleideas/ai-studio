import type { BrowserWindow, Dialog, IpcMain, OpenDialogOptions } from 'electron'
import { isEngineId, type ImageJob } from '@shared/image.js'
import { cleanSubject, type Subject } from '@shared/subject.js'
import {
  cleanCollectPlan,
  cleanCompareRequest,
  cleanEdits,
  cleanName,
  cleanTrainSettings,
  isProjectId,
  isRunId,
  type CaptionItem,
  type CaptionPage,
  type CaptionView,
  type LoraItem,
  type LossPoint,
  type TrainingProject
} from '@shared/training.js'
import type { GpuArbiter } from '../gpu.js'
import type { PhotoCollector } from './collect.js'
import type { ImageEngine } from './engine.js'

/**
 * The Train tab's operations, each a named handler (docs/training.md).
 *
 * Same rules as the image handlers: ids must look like engine ids, settings are
 * rebuilt field by field, and the one path involved — a project's photo folder —
 * only ever comes from a native dialog the user answered. Training and comparing
 * load the image model, so they claim the GPU from the chat model first.
 */

interface Deps {
  ipcMain: IpcMain
  dialog: Dialog
  imageEngine: ImageEngine
  gpu: GpuArbiter
  getWindow: () => BrowserWindow | null
  collector: PhotoCollector
}

const VIEWS: CaptionView[] = ['all', 'review', 'excluded']
const ACTIONS = ['accept', 'edit', 'exclude'] as const

function projectId(id: unknown): string {
  if (!isProjectId(id)) throw new Error('Invalid project.')
  return id
}

function runId(id: unknown): string {
  if (!isRunId(id)) throw new Error('Invalid run.')
  return id
}

function count(value: unknown, lo: number, hi: number): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < lo || value > hi) throw new Error('Invalid number.')
  return value
}

export function registerTrainingHandlers({ ipcMain, dialog, imageEngine, gpu, getWindow, collector }: Deps): void {
  const api = async (method: string, route: string, body?: unknown): Promise<unknown> => {
    await imageEngine.start()
    return imageEngine.request(method, route, body)
  }
  const project = (id: unknown) => `/api/training/projects/${projectId(id)}`

  ipcMain.handle('training:projects', async () =>
    ((await api('GET', '/api/training/projects')) as { items: TrainingProject[] }).items)

  ipcMain.handle('training:subjectPresets', async () =>
    ((await api('GET', '/api/training/subjects')) as { items: Subject[] }).items)

  ipcMain.handle('training:createProject', async (_e, name: unknown, subject: unknown) => {
    const clean = cleanName(name, 'A project name')
    const definition = cleanSubject(subject)
    // The end-to-end test names the folder up front instead of answering a dialog.
    let folder = process.env['STUDIO_PICK_TRAINING_FOLDER'] ?? null
    if (!folder) {
      const options: OpenDialogOptions = {
        title: 'Choose the photo folder',
        buttonLabel: 'Use this folder',
        properties: ['openDirectory']
      }
      const win = getWindow()
      const picked = win ? await dialog.showOpenDialog(win, options) : await dialog.showOpenDialog(options)
      folder = picked.canceled ? null : picked.filePaths[0] ?? null
    }
    if (!folder) return null
    return api('POST', '/api/training/projects', { name: clean, photo_root: folder, subject: definition })
  })

  ipcMain.handle('training:project', (_e, id: unknown) => api('GET', project(id)))

  ipcMain.handle('training:setSubject', (_e, id: unknown, subject: unknown) =>
    api('PUT', `${project(id)}/subject`, cleanSubject(subject)) as Promise<TrainingProject>)

  ipcMain.handle('training:deleteProject', async (_e, id: unknown) => {
    const win = getWindow()
    const options = {
      type: 'warning' as const,
      buttons: ['Delete', 'Cancel'],
      defaultId: 1,
      cancelId: 1,
      message: 'Delete this training project?',
      detail: 'Its captions, caches, runs and checkpoints are deleted. Your photos are not touched, ' +
        'and LoRAs already added to the library stay there.'
    }
    const answer = win ? await dialog.showMessageBox(win, options) : await dialog.showMessageBox(options)
    if (answer.response !== 0) return
    await api('DELETE', project(id))
  })

  ipcMain.handle('training:scan', (_e, id: unknown) => api('POST', `${project(id)}/scan`, {}) as Promise<ImageJob>)

  ipcMain.handle('training:caption', async (_e, id: unknown) => {
    const route = `${project(id)}/caption`
    await imageEngine.start()
    await gpu.claim('image') // the captioning model runs in the image engine and needs the GPU to itself
    return imageEngine.request('POST', route, {}) as Promise<ImageJob>
  })

  ipcMain.handle('training:captions', async (_e, id: unknown, view: unknown, offset: unknown, limit: unknown) => {
    if (!VIEWS.includes(view as CaptionView)) throw new Error('Invalid view.')
    const query = new URLSearchParams({
      view: view as string,
      offset: String(count(offset, 0, 1_000_000)),
      limit: String(count(limit, 1, 200))
    })
    return (await api('GET', `${project(id)}/captions?${query}`)) as CaptionPage
  })

  ipcMain.handle('training:review', (_e, id: unknown, item: unknown, action: unknown, edits: unknown) => {
    if (!isEngineId(item)) throw new Error('Invalid photo.')
    if (!ACTIONS.includes(action as (typeof ACTIONS)[number])) throw new Error('Invalid action.')
    const clean = cleanEdits(edits)
    return api('POST', `${project(id)}/captions/${item}`, { action, ...(clean ? { edits: clean } : {}) }) as Promise<CaptionItem>
  })

  ipcMain.handle('training:train', async (_e, id: unknown, settings: unknown) => {
    const route = `${project(id)}/train`
    const clean = cleanTrainSettings(settings)
    await imageEngine.start()
    await gpu.claim('image')
    return imageEngine.request('POST', route, clean) as Promise<ImageJob>
  })

  ipcMain.handle('training:compare', async (_e, id: unknown, request: unknown) => {
    const route = `${project(id)}/compare`
    const clean = cleanCompareRequest(request)
    await imageEngine.start()
    await gpu.claim('image')
    return imageEngine.request('POST', route, clean) as Promise<ImageJob>
  })

  ipcMain.handle('training:log', async (_e, id: unknown, run: unknown) =>
    ((await api('GET', `${project(id)}/runs/${runId(run)}/log`)) as { items: LossPoint[] }).items)

  ipcMain.handle('training:useCheckpoint', (_e, id: unknown, run: unknown, step: unknown, name: unknown) =>
    api('POST', `${project(id)}/runs/${runId(run)}/use`, {
      step: count(step, 1, 20000),
      name: cleanName(name, 'A LoRA name')
    }) as Promise<LoraItem>)

  ipcMain.handle('training:loras', async () => ((await api('GET', '/api/loras')) as { items: LoraItem[] }).items)

  ipcMain.handle('training:renameLora', (_e, id: unknown, name: unknown) => {
    if (!isEngineId(id)) throw new Error('Invalid LoRA.')
    return api('PATCH', `/api/loras/${id}`, { name: cleanName(name, 'A LoRA name') }) as Promise<LoraItem>
  })

  ipcMain.handle('training:deleteLora', async (_e, id: unknown) => {
    if (!isEngineId(id)) throw new Error('Invalid LoRA.')
    await api('DELETE', `/api/loras/${id}`)
  })

  ipcMain.handle('training:collectState', () => collector.state)

  ipcMain.handle('training:collect', async (_e, plan: unknown) => {
    const clean = cleanCollectPlan(plan)
    if (collector.running) throw new Error('Photos are already being collected.')
    // The end-to-end test names the folder up front instead of answering a dialog.
    let folder = process.env['STUDIO_PICK_COLLECT_FOLDER'] ?? null
    if (!folder) {
      const options: OpenDialogOptions = {
        title: 'Choose where the photos go (an empty folder, or one collected into before)',
        buttonLabel: 'Collect here',
        properties: ['openDirectory', 'createDirectory']
      }
      const win = getWindow()
      const picked = win ? await dialog.showOpenDialog(win, options) : await dialog.showOpenDialog(options)
      folder = picked.canceled ? null : picked.filePaths[0] ?? null
    }
    if (!folder) return
    await imageEngine.start()
    await collector.collect(clean, folder)
  })

  ipcMain.handle('training:checkCollect', async (_e, plan: unknown) => {
    const clean = cleanCollectPlan(plan)
    await imageEngine.start() // its Python runs the script
    return collector.check(clean)
  })

  ipcMain.handle('training:cancelCollect', () => collector.cancel())
}
