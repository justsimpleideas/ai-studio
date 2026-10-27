import { contextBridge, ipcRenderer, type IpcRendererEvent } from 'electron'
import type { RendererApi } from '@shared/ipc.js'

/**
 * The context bridge.
 *
 * Every entry is a named operation, never a general "call this channel"
 * escape hatch. The difference matters: a generic bridge would make the
 * renderer's authority equal to the main process's, and the whole arrangement
 * where the window cannot reach the disk would be decoration.
 */

function subscribe<T>(channel: string, handler: (payload: T) => void): () => void {
  const listener = (_event: IpcRendererEvent, payload: T) => handler(payload)
  ipcRenderer.on(channel, listener)
  return () => ipcRenderer.removeListener(channel, listener)
}

const api: RendererApi = {
  getConfig: () => ipcRenderer.invoke('config:get'),

  setClassMode: (risk, mode) => ipcRenderer.invoke('config:setClassMode', risk, mode),
  setConfirmEverything: (value) => ipcRenderer.invoke('config:setConfirmEverything', value),
  setKvCache: (kv) => ipcRenderer.invoke('config:setKvCache', kv),
  setBackground: (prefs) => ipcRenderer.invoke('config:setBackground', prefs),

  clearSessionGrants: () => ipcRenderer.invoke('session:clearGrants'),

  saveRules: (rules) => ipcRenderer.invoke('rules:save', rules),

  pickModel: () => ipcRenderer.invoke('model:pick'),
  loadModel: (modelPath) => ipcRenderer.invoke('model:load', modelPath),
  unloadModel: () => ipcRenderer.invoke('model:unload'),

  sendMessage: (text) => ipcRenderer.invoke('chat:send', text),
  abort: (promptId) => ipcRenderer.invoke('chat:abort', promptId),
  refreshUsage: () => ipcRenderer.invoke('usage:refresh'),

  respondToApproval: (id, response) => ipcRenderer.invoke('approval:respond', id, response),
  listChanges: () => ipcRenderer.invoke('journal:list'),
  revertChanges: (groupId, force) => ipcRenderer.invoke('journal:revert', groupId, force),
  onChangesUpdated: (handler) => subscribe('journal:changed', () => handler()),

  onEngineEvent: (handler) => subscribe('engine:event', handler),
  onApproval: (handler) => subscribe('approval:request', handler),
  onAudit: (handler) => subscribe('audit:entry', handler),
  onToolOutcome: (handler) => subscribe('tool:outcome', handler),
  onConfigChanged: (handler) => subscribe('config:changed', handler),
  onNotice: (handler) => subscribe('studio:notice', handler),

  workspace: {
    get: () => ipcRenderer.invoke('workspace:get'),
    openFolder: () => ipcRenderer.invoke('workspace:openFolder'),
    switchTo: (id) => ipcRenderer.invoke('workspace:switch', id),
    forget: (id) => ipcRenderer.invoke('workspace:forget', id),
    saveRules: (rules) => ipcRenderer.invoke('workspace:saveRules', rules),
    addFolder: () => ipcRenderer.invoke('workspace:addFolder'),
    setFolderEnabled: (id, enabled) => ipcRenderer.invoke('workspace:setFolderEnabled', id, enabled),
    removeFolder: (id) => ipcRenderer.invoke('workspace:removeFolder', id),
    removeGrant: (id) => ipcRenderer.invoke('workspace:removeGrant', id),
    setProtected: (list) => ipcRenderer.invoke('workspace:setProtected', list),
    setInternet: (level) => ipcRenderer.invoke('workspace:setInternet', level),
    newChat: () => ipcRenderer.invoke('chat:new'),
    openChat: (id) => ipcRenderer.invoke('chat:open', id),
    deleteChat: (id) => ipcRenderer.invoke('chat:delete', id),
    onChanged: (handler) => subscribe('workspace:changed', handler)
  },

  internet: {
    keyStatus: () => ipcRenderer.invoke('internet:keyStatus'),
    setKey: (key) => ipcRenderer.invoke('internet:setKey', key),
    log: () => ipcRenderer.invoke('network:log'),
    onLogChanged: (handler) => subscribe('network:changed', () => handler())
  },

  image: {
    engineState: () => ipcRenderer.invoke('image:engineState'),
    start: () => ipcRenderer.invoke('image:start'),
    status: () => ipcRenderer.invoke('image:status'),
    settings: () => ipcRenderer.invoke('image:settings'),
    saveSettings: (changes) => ipcRenderer.invoke('image:saveSettings', changes),
    loadModel: (kind) => ipcRenderer.invoke('image:loadModel', kind),
    unloadModel: (kind) => ipcRenderer.invoke('image:unloadModel', kind),
    submit: (request) => ipcRenderer.invoke('image:submit', request),
    submitVideo: (request) => ipcRenderer.invoke('image:submitVideo', request),
    pickStartImage: () => ipcRenderer.invoke('image:pickStartImage'),
    uploadStartImage: (data) => ipcRenderer.invoke('image:uploadStartImage', data),
    startImageFrom: (id, frame) => ipcRenderer.invoke('image:startImageFrom', id, frame),
    cancel: (jobId) => ipcRenderer.invoke('image:cancel', jobId),
    jobs: () => ipcRenderer.invoke('image:jobs'),
    gallery: (query) => ipcRenderer.invoke('image:gallery', query),
    setFavorite: (id, favorite) => ipcRenderer.invoke('image:setFavorite', id, favorite),
    deleteImage: (id) => ipcRenderer.invoke('image:delete', id),
    folders: () => ipcRenderer.invoke('image:folders'),
    createFolder: (name) => ipcRenderer.invoke('image:createFolder', name),
    renameFolder: (id, name) => ipcRenderer.invoke('image:renameFolder', id, name),
    deleteFolder: (id) => ipcRenderer.invoke('image:deleteFolder', id),
    moveToFolder: (id, folderId) => ipcRenderer.invoke('image:moveToFolder', id, folderId),
    saveImageAs: (id) => ipcRenderer.invoke('image:saveAs', id),
    saveAsGif: (id, options) => ipcRenderer.invoke('image:saveAsGif', id, options),
    includeSettingsInExports: () => ipcRenderer.invoke('image:includeSettingsInExports'),
    setIncludeSettingsInExports: (value) => ipcRenderer.invoke('image:setIncludeSettingsInExports', value),
    pickModelFolder: (kind) => ipcRenderer.invoke('image:pickModelFolder', kind),
    pickFfmpeg: () => ipcRenderer.invoke('image:pickFfmpeg'),
    pickPython: () => ipcRenderer.invoke('image:pickPython'),
    onEvent: (handler) => subscribe('image:event', handler),
    onEngineState: (handler) => subscribe('image:engineState', handler),
    setupState: () => ipcRenderer.invoke('image:setupState'),
    installEngine: () => ipcRenderer.invoke('image:installEngine'),
    cancelInstall: () => ipcRenderer.invoke('image:cancelInstall'),
    onSetupState: (handler) => subscribe('image:setupState', handler),
    downloadState: () => ipcRenderer.invoke('image:downloadState'),
    modelsFolder: () => ipcRenderer.invoke('image:modelsFolder'),
    chooseModelsFolder: () => ipcRenderer.invoke('image:chooseModelsFolder'),
    downloadModel: (id) => ipcRenderer.invoke('image:downloadModel', id),
    cancelDownload: () => ipcRenderer.invoke('image:cancelDownload'),
    openModelLink: (id, which) => ipcRenderer.invoke('image:openModelLink', id, which),
    onDownloadState: (handler) => subscribe('image:downloadState', handler)
  }
}

contextBridge.exposeInMainWorld('api', api)
