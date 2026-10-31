import type {
  AppConfig,
  BackgroundPreferences,
  ChatEngine,
  FolderPrivacyPreferences,
  NotificationPreferences,
  GlobalRule,
  KvCacheType,
  ServerPreferences,
  Theme
} from './config.js'
import type { GenerationPreferences } from './generation.js'
import type { SearchPreferences } from './search.js'
import type { ShortcutMap } from './shortcuts.js'
import type { ClassMode, ProtectedPath } from './permissions.js'
import type { ChangeGroup, RevertResult } from './changes.js'
import type { RiskClass } from './tools.js'
import type { EngineToMain } from './engine-protocol.js'
import type { ApprovalRequest, ApprovalResponse, AuditEntry, ToolOutcome } from './broker-types.js'
import type { ImageApi } from './image.js'
import type { TrainingApi } from './training.js'
import type { WorkspaceSnapshot, WorkspaceState } from './workspace.js'
import type { InternetLevel, NetworkLogEntry } from './network.js'
import type { DataLocationKey, DataLocationsView } from './locations.js'

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
  /** Strict mode: conversation only, no tools of any kind. */
  setStrict(strict: boolean): Promise<WorkspaceState>
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
  /** Which engine runs the chat model, and how the server is started; for the next load. */
  setEngine(engine: ChatEngine, server: ServerPreferences): Promise<AppConfig>
  /** Choose the llama-server program through a native dialog; null when cancelled. */
  pickServer(): Promise<AppConfig | null>
  /** Context size for the next model load; null for the default (trained size, capped). */
  setContextSize(size: number | null): Promise<AppConfig>
  /** Sampling and thinking; applied from the next message on. */
  setGeneration(prefs: GenerationPreferences): Promise<AppConfig>
  /** What closing the window does, and which models stay loaded in the background. */
  setBackground(prefs: BackgroundPreferences): Promise<AppConfig>
  /** When open gallery folders lock again: on minimize, on closing to the background. */
  setFolderPrivacy(prefs: FolderPrivacyPreferences): Promise<AppConfig>
  /** Whether the tray tells you, in the background, that a reply or an image or video is done. */
  setNotifications(prefs: NotificationPreferences): Promise<AppConfig>
  /** Every changeable shortcut at once; refused if any binding is unusable or taken twice. */
  setShortcuts(shortcuts: ShortcutMap): Promise<AppConfig>
  /** Dark or light; the window and its title bar change at once. */
  setTheme(theme: Theme): Promise<AppConfig>

  /**
   * Where the studio keeps its own folders (shared/locations.ts). Choosing one
   * opens a native dialog in the main process: the window asks, and is told
   * what came back, but never names a path itself.
   */
  locations: {
    get(): Promise<DataLocationsView>
    /** Open the picker for one folder. Cancelling returns the locations unchanged. */
    choose(key: DataLocationKey): Promise<DataLocationsView>
    /** Put one folder back where it started. */
    reset(key: DataLocationKey): Promise<DataLocationsView>
    /** Quit and start again, which is what moves the chats folder over. */
    restart(): Promise<void>
  }

  clearSessionGrants(): Promise<void>

  saveRules(rules: GlobalRule[]): Promise<AppConfig>

  pickModel(): Promise<string | null>
  loadModel(modelPath: string): Promise<void>
  unloadModel(): Promise<void>
  /** Inspect the chat model's own file (shared/inspect.ts); the report arrives as image inspect state. False when no model file is known. */
  inspectModel(): Promise<boolean>

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

  /** The web search service and its key (never returned, only whether one is set), and the open workspace's internet log. */
  internet: {
    /** Whether a key is set for the current search service. */
    keyStatus(): Promise<{ hasKey: boolean }>
    /** Save or remove the key for the current search service. */
    setKey(key: string | null): Promise<{ hasKey: boolean }>
    /** Which search service web_search uses; the SearXNG address is checked before it is kept. */
    setSearch(prefs: SearchPreferences): Promise<AppConfig>
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
