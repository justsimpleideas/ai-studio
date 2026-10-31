import { TOOL_NAMES, type ToolName, type WebToolName } from './tools.js'

/**
 * How much of the internet the agent may use in a workspace.
 *
 * 'offline' is the default and means the web tools do not exist for the model:
 * they are left out of what it is told it can call, so there is nothing to
 * misuse. The other levels add tools; every request they make goes through the
 * main process, is shown to the user before it is sent (unless they chose
 * otherwise), and is written to the workspace's internet log.
 */
export type InternetLevel = 'offline' | 'search' | 'download'

export const INTERNET_LEVELS: Array<{ level: InternetLevel; title: string; help: string }> = [
  {
    level: 'offline',
    title: 'Offline',
    help: 'No internet at all. The agent works only with the files in this workspace.'
  },
  {
    level: 'search',
    title: 'Search only',
    help: 'The agent can search the web and read public pages. Nothing is saved to disk.'
  },
  {
    level: 'download',
    title: 'Search and download',
    help: 'As above, and the agent can download files from the web into this folder as new files.'
  }
]

const WEB_TOOLS_FOR: Record<InternetLevel, WebToolName[]> = {
  offline: [],
  search: ['web_search', 'fetch_page'],
  download: ['web_search', 'fetch_page', 'download_file']
}

/**
 * The tools a model is given in a workspace at this level. In strict mode
 * there are none: the model is not told about any tool, and the grammar that
 * shapes its output has no function calls in it to make.
 */
export function toolsFor(level: InternetLevel, strict = false): ToolName[] {
  if (strict) return []
  return [...TOOL_NAMES, ...WEB_TOOLS_FOR[level]]
}

export function webToolAllowed(level: InternetLevel, tool: WebToolName): boolean {
  return WEB_TOOLS_FOR[level].includes(tool)
}

export function isInternetLevel(value: unknown): value is InternetLevel {
  return value === 'offline' || value === 'search' || value === 'download'
}

/**
 * One line of a workspace's internet log: every request the agent asked for,
 * including the ones that were refused or declined and never sent.
 */
export interface NetworkLogEntry {
  id: string
  at: string
  tool: WebToolName
  /** Exactly what would be sent: the search query, or the full address. */
  sent: string
  /** Where it would go. */
  host: string | null
  outcome: 'sent' | 'refused' | 'declined' | 'failed'
  /** Why it was refused or failed, or a summary of what came back. */
  detail: string
  /** Bytes received. */
  bytes: number | null
  /** For downloads, where the file was saved. */
  savedTo: string | null
  chatId: string | null
  promptId: string | null
}
