import type { AppConfig, BackgroundPreferences, GlobalRule, KvCacheType } from './config.js'
import type { ClassMode, ProtectedPath } from './permissions.js'
import type { ChangeGroup, RevertResult } from './changes.js'
import type { RiskClass } from './tools.js'
import type { EngineToMain } from './engine-protocol.js'
import type { ApprovalRequest, ApprovalResponse, AuditEntry, ToolOutcome } from './broker-types.js'
import type { ImageApi } from './image.js'
import type { TrainingApi } from './training.js'
import type { WorkspaceSnapshot, WorkspaceState } from './workspace.js'
import type { InternetLevel, NetworkLogEntry } from './network.js'

/**
 * Workspaces and their chats. Opening a folder and sharing another one go
 * through a native dialog in the main process, like everything else that names
 * a path; ids are the only thing the window passes back.
 */
export interface WorkspaceApi {
  get(): Promise<WorkspaceSnapshot>
  openFolder(): Promise<WorkspaceSnapshot>
  switchTo(id: string): Promise<WorkspaceSnapshot>
  /** Asks for confirmation in a native dialog first. */
  forget(id: string): Promise<WorkspaceSnapshot>
  saveRules(rules: GlobalRule[]): Promise<WorkspaceState>
  addFolder(): Promise<WorkspaceState>
  setFolderEnabled(id: string, enabled: boolean): Promise<WorkspaceState>
  removeFolder(id: string): Promise<WorkspaceState>
  removeGrant(id: string): Promise<WorkspaceState>
  /** Replace the workspace's protected paths. Built-in ones are kept, switched off at most. */
  setProtected(list: ProtectedPath[]): Promise<WorkspaceState>
  setInternet(level: InternetLevel): Promise<WorkspaceState>
  newChat(): Promise<WorkspaceSnapshot>
  openChat(id: string): Promise<WorkspaceSnapshot>
  deleteChat(id: string): Promise<WorkspaceSnapshot>
  onChanged(handler: (state: WorkspaceState) => void): () => void
}

/**
 * The surface the window is allowed to reach.
 *
 * Nothing here takes a filesystem path from the renderer for the purpose of
 * reading or writing it. Adding a folder and choosing a model both go through
 * a native dialog in the main process, so the window can ask for a picker but
 * can never name a file it was not handed. It keeps the trust boundary in one
 * place: the renderer is a view, and the only component with authority over the
 * disk is the broker.
 */
export interface RendererApi {
  getConfig(): Promise<AppConfig>

  setClassMode(risk: RiskClass, mode: ClassMode): Promise<AppConfig>
  setConfirmEverything(value: boolean): Promise<AppConfig>
  /** Context memory precision for the next model load. */
  setKvCache(kv: KvCacheType): Promise<AppConfig>
  /** What closing the window does, and which models stay loaded in the background. */
  setBackground(prefs: BackgroundPreferences): Promise<AppConfig>

  clearSessionGrants(): Promise<void>

  saveRules(rules: GlobalRule[]): Promise<AppConfig>

  pickModel(): Promise<string | null>
  loadModel(modelPath: string): Promise<void>
  unloadModel(): Promise<void>

  sendMessage(text: string): Promise<string>
  abort(promptId: string): Promise<void>
  /** Ask for a fresh token-usage snapshot; it arrives as an engine event. */
  refreshUsage(): Promise<void>

  respondToApproval(id: string, response: ApprovalResponse): Promise<void>
  /** What the agent changed in the open workspace, grouped by reply, newest first. */
  listChanges(): Promise<ChangeGroup[]>
  /** Put back one reply's changes. Without `force`, conflicts are returned and nothing is done. */
  revertChanges(groupId: string, force: boolean): Promise<RevertResult>
  onChangesUpdated(handler: () => void): () => void

  onEngineEvent(handler: (event: EngineToMain) => void): () => void
  onApproval(handler: (request: ApprovalRequest) => void): () => void
  onAudit(handler: (entry: AuditEntry) => void): () => void
  onToolOutcome(handler: (outcome: ToolOutcome) => void): () => void
  onConfigChanged(handler: (config: AppConfig) => void): () => void
  /** Short messages from the studio itself, e.g. that a model was unloaded to free the GPU. */
  onNotice(handler: (message: string) => void): () => void

  workspace: WorkspaceApi

  /** The web search key (never returned, only whether one is set) and the open workspace's internet log. */
  internet: {
    keyStatus(): Promise<{ hasKey: boolean }>
    setKey(key: string | null): Promise<{ hasKey: boolean }>
    log(): Promise<NetworkLogEntry[]>
    onLogChanged(handler: () => void): () => void
  }

  /** The image side. Every operation goes through the main process (decisions S3). */
  image: ImageApi

  /** Training LoRAs on the image engine, and the LoRA library (docs/training.md). */
  training: TrainingApi
}

declare global {
  interface Window {
    api: RendererApi
  }
}
