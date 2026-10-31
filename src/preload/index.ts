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
  setEngine: (engine, server) => ipcRenderer.invoke('config:setEngine', engine, server),
  pickServer: () => ipcRenderer.invoke('model:pickServer'),
  setContextSize: (size) => ipcRenderer.invoke('config:setContextSize', size),
  setGeneration: (prefs) => ipcRenderer.invoke('config:setGeneration', prefs),
  setBackground: (prefs) => ipcRenderer.invoke('config:setBackground', prefs),
  setFolderPrivacy: (prefs) => ipcRenderer.invoke('config:setFolderPrivacy', prefs),
  setNotifications: (prefs) => ipcRenderer.invoke('config:setNotifications', prefs),
  setShortcuts: (shortcuts) => ipcRenderer.invoke('config:setShortcuts', shortcuts),
  setTheme: (theme) => ipcRenderer.invoke('config:setTheme', theme),

  locations: {
    get: () => ipcRenderer.invoke('locations:get'),
    choose: (key) => ipcRenderer.invoke('locations:choose', key),
    reset: (key) => ipcRenderer.invoke('locations:reset', key),
    restart: () => ipcRenderer.invoke('locations:restart')
  },

  clearSessionGrants: () => ipcRenderer.invoke('session:clearGrants'),

  saveRules: (rules) => ipcRenderer.invoke('rules:save', rules),

  pickModel: () => ipcRenderer.invoke('model:pick'),
  loadModel: (modelPath) => ipcRenderer.invoke('model:load', modelPath),
  unloadModel: () => ipcRenderer.invoke('model:unload'),
  inspectModel: () => ipcRenderer.invoke('model:inspect'),

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
    setStrict: (strict) => ipcRenderer.invoke('workspace:setStrict', strict),
    newChat: () => ipcRenderer.invoke('chat:new'),
    openChat: (id) => ipcRenderer.invoke('chat:open', id),
    deleteChat: (id) => ipcRenderer.invoke('chat:delete', id),
    onChanged: (handler) => subscribe('workspace:changed', handler)
  },

  internet: {
    keyStatus: () => ipcRenderer.invoke('internet:keyStatus'),
    setKey: (key) => ipcRenderer.invoke('internet:setKey', key),
    setSearch: (prefs) => ipcRenderer.invoke('internet:setSearch', prefs),
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
    updateJob: (jobId, request) => ipcRenderer.invoke('image:updateJob', jobId, request),
    jobs: () => ipcRenderer.invoke('image:jobs'),
    gallery: (query) => ipcRenderer.invoke('image:gallery', query),
    setFavorite: (id, favorite) => ipcRenderer.invoke('image:setFavorite', id, favorite),
    deleteImage: (id) => ipcRenderer.invoke('image:delete', id),
    folders: () => ipcRenderer.invoke('image:folders'),
    createFolder: (name) => ipcRenderer.invoke('image:createFolder', name),
    renameFolder: (id, name) => ipcRenderer.invoke('image:renameFolder', id, name),
    deleteFolder: (id) => ipcRenderer.invoke('image:deleteFolder', id),
    moveToFolder: (id, folderId) => ipcRenderer.invoke('image:moveToFolder', id, folderId),
    moveManyToFolder: (ids, folderId) => ipcRenderer.invoke('image:moveManyToFolder', ids, folderId),
    reorderFolder: (id, before) => ipcRenderer.invoke('image:reorderFolder', id, before),
    setFolderHidden: (id, hidden) => ipcRenderer.invoke('image:setFolderHidden', id, hidden),
    setFolderPin: (id, pin, current) => ipcRenderer.invoke('image:setFolderPin', id, pin, current),
    unlockFolder: (id, pin) => ipcRenderer.invoke('image:unlockFolder', id, pin),
    lockFolder: (id) => ipcRenderer.invoke('image:lockFolder', id),
    pickMergeSource: () => ipcRenderer.invoke('image:pickMergeSource'),
    inspectMerge: (tokens, renames) => ipcRenderer.invoke('image:inspectMerge', tokens, renames),
    startMerge: (tokens, renames) => ipcRenderer.invoke('image:startMerge', tokens, renames),
    mergeStatus: () => ipcRenderer.invoke('image:mergeStatus'),
    cancelMerge: () => ipcRenderer.invoke('image:cancelMerge'),
    showHiddenFolders: () => ipcRenderer.invoke('image:showHiddenFolders'),
    setShowHiddenFolders: (show) => ipcRenderer.invoke('image:setShowHiddenFolders', show),
    saveImageAs: (id) => ipcRenderer.invoke('image:saveAs', id),
    saveManyTo: (ids, options) => ipcRenderer.invoke('image:saveManyTo', ids, options),
    saveAsGif: (id, options) => ipcRenderer.invoke('image:saveAsGif', id, options),
    includeSettingsInExports: () => ipcRenderer.invoke('image:includeSettingsInExports'),
    setIncludeSettingsInExports: (value) => ipcRenderer.invoke('image:setIncludeSettingsInExports', value),
    pickModelFolder: (kind) => ipcRenderer.invoke('image:pickModelFolder', kind),
    pickFfmpeg: () => ipcRenderer.invoke('image:pickFfmpeg'),
    pickPython: () => ipcRenderer.invoke('image:pickPython'),
    onEvent: (handler) => subscribe('image:event', handler),
    onEngineState: (handler) => subscribe('image:engineState', handler),
    onGalleryMoved: (handler) => subscribe('image:galleryMoved', () => handler()),
    setupState: () => ipcRenderer.invoke('image:setupState'),
    installEngine: () => ipcRenderer.invoke('image:installEngine'),
    cancelInstall: () => ipcRenderer.invoke('image:cancelInstall'),
    onSetupState: (handler) => subscribe('image:setupState', handler),
    downloadState: () => ipcRenderer.invoke('image:downloadState'),
    hardware: () => ipcRenderer.invoke('image:hardware'),
    modelsFolder: () => ipcRenderer.invoke('image:modelsFolder'),
    chooseModelsFolder: () => ipcRenderer.invoke('image:chooseModelsFolder'),
    downloadModel: (id) => ipcRenderer.invoke('image:downloadModel', id),
    cancelDownload: () => ipcRenderer.invoke('image:cancelDownload'),
    openModelLink: (id, which) => ipcRenderer.invoke('image:openModelLink', id, which),
    onDownloadState: (handler) => subscribe('image:downloadState', handler),
    inspectState: () => ipcRenderer.invoke('image:inspectState'),
    inspectModelFile: () => ipcRenderer.invoke('image:inspectModelFile'),
    cancelInspect: () => ipcRenderer.invoke('image:cancelInspect'),
    onInspectState: (handler) => subscribe('image:inspectState', handler),
    importState: () => ipcRenderer.invoke('image:importState'),
    pickImportFile: (target, slot) => ipcRenderer.invoke('image:pickImportFile', target, slot ?? 'first'),
    swapImportExperts: () => ipcRenderer.invoke('image:swapImportExperts'),
    startImport: (options) => ipcRenderer.invoke('image:startImport', options),
    cancelImport: () => ipcRenderer.invoke('image:cancelImport'),
    clearImport: () => ipcRenderer.invoke('image:clearImport'),
    onImportState: (handler) => subscribe('image:importState', handler)
  },

  training: {
    projects: () => ipcRenderer.invoke('training:projects'),
    subjectPresets: () => ipcRenderer.invoke('training:subjectPresets'),
    createProject: (name, subject) => ipcRenderer.invoke('training:createProject', name, subject),
    project: (id) => ipcRenderer.invoke('training:project', id),
    setSubject: (id, subject) => ipcRenderer.invoke('training:setSubject', id, subject),
    deleteProject: (id) => ipcRenderer.invoke('training:deleteProject', id),
    scan: (id) => ipcRenderer.invoke('training:scan', id),
    caption: (id) => ipcRenderer.invoke('training:caption', id),
    captions: (id, view, offset, limit) => ipcRenderer.invoke('training:captions', id, view, offset, limit),
    review: (id, item, action, edits) => ipcRenderer.invoke('training:review', id, item, action, edits),
    train: (id, settings) => ipcRenderer.invoke('training:train', id, settings),
    compare: (id, request) => ipcRenderer.invoke('training:compare', id, request),
    log: (id, run) => ipcRenderer.invoke('training:log', id, run),
    useCheckpoint: (id, run, step, name) => ipcRenderer.invoke('training:useCheckpoint', id, run, step, name),
    loras: () => ipcRenderer.invoke('training:loras'),
    renameLora: (id, name) => ipcRenderer.invoke('training:renameLora', id, name),
    deleteLora: (id) => ipcRenderer.invoke('training:deleteLora', id),
    collectState: () => ipcRenderer.invoke('training:collectState'),
    collect: (plan) => ipcRenderer.invoke('training:collect', plan),
    checkCollect: (plan) => ipcRenderer.invoke('training:checkCollect', plan),
    cancelCollect: () => ipcRenderer.invoke('training:cancelCollect'),
    onCollectState: (handler) => subscribe('training:collectState', handler)
  }
}

contextBridge.exposeInMainWorld('api', api)
