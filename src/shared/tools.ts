/**
 * The complete set of actions a model is allowed to request.
 *
 * This list is deliberately small. Every entry is a capability the user has to
 * reason about when granting permission, and a shape a 7B-class model has to
 * emit correctly. Both arguments push the same way: fewer, blunter tools.
 */

/** The file tools: always available, always through the broker. */
export const TOOL_NAMES = [
  'list_directory',
  'read_file',
  'write_file',
  'edit_file',
  'delete_file'
] as const

/**
 * The web tools. A model only ever sees the ones its workspace's internet level
 * allows (see shared/network.ts); at the default level, none.
 */
export const WEB_TOOL_NAMES = ['web_search', 'fetch_page', 'download_file'] as const

export const ALL_TOOL_NAMES = [...TOOL_NAMES, ...WEB_TOOL_NAMES] as const

export type FileToolName = (typeof TOOL_NAMES)[number]
export type WebToolName = (typeof WEB_TOOL_NAMES)[number]
export type ToolName = FileToolName | WebToolName

export function isWebTool(name: string): name is WebToolName {
  return (WEB_TOOL_NAMES as readonly string[]).includes(name)
}

/**
 * How much damage a tool can do if the model is wrong. The UI groups approval
 * settings by this, not by individual tool, because "let it read things" is a
 * decision a person can actually hold in their head.
 */
export type RiskClass = 'read' | 'write' | 'destructive' | 'network'

export const TOOL_RISK: Record<ToolName, RiskClass> = {
  list_directory: 'read',
  read_file: 'read',
  write_file: 'write',
  edit_file: 'write',
  delete_file: 'destructive',
  web_search: 'network',
  fetch_page: 'network',
  download_file: 'network'
}

export interface ToolArgs {
  list_directory: { path: string }
  read_file: { path: string }
  write_file: { path: string; content: string }
  edit_file: { path: string; old_text: string; new_text: string }
  delete_file: { path: string }
  web_search: { query: string }
  fetch_page: { url: string }
  download_file: { url: string; path: string }
}

export interface DirEntry {
  name: string
  kind: 'file' | 'directory' | 'other'
  bytes: number | null
}

export interface ToolResults {
  list_directory: { entries: DirEntry[]; truncated: boolean }
  read_file: { text: string; bytes: number; truncated: boolean }
  write_file: { bytes: number; existed: boolean }
  edit_file: { replacements: number; bytes: number }
  delete_file: { recycled: boolean }
  web_search: { results: Array<{ title: string; url: string; snippet: string }> }
  fetch_page: { url: string; title: string; text: string; truncated: boolean; note: string }
  download_file: { path: string; bytes: number }
}

/** A tool call as the model emitted it, before any validation. */
export interface RawToolCall {
  name: string
  args: unknown
}

/** A tool call that has been validated against the schema above. */
export type ToolCall = {
  [K in ToolName]: { id: string; name: K; args: ToolArgs[K] }
}[ToolName]

export type FileToolCall = Extract<ToolCall, { name: FileToolName }>
export type WebToolCall = Extract<ToolCall, { name: WebToolName }>

/**
 * JSON Schema for each tool's arguments.
 *
 * This does double duty. node-llama-cpp compiles it into a GBNF grammar that
 * constrains generation, so the model is structurally incapable of emitting a
 * malformed call. We then validate against it again on arrival, because
 * grammar-constrained decoding is only in force when the engine actually
 * applied the grammar, and a defence that depends on the other side behaving
 * is not a defence.
 */
export const TOOL_SCHEMAS: Record<ToolName, object> = {
  list_directory: {
    type: 'object',
    properties: {
      path: { type: 'string', description: 'Path of the directory to list, absolute or relative to the working folder.' }
    },
    required: ['path'],
    additionalProperties: false
  },
  read_file: {
    type: 'object',
    properties: {
      path: { type: 'string', description: 'Path of the file to read, absolute or relative to the working folder.' }
    },
    required: ['path'],
    additionalProperties: false
  },
  write_file: {
    type: 'object',
    properties: {
      path: { type: 'string', description: 'Path of the file to write, absolute or relative to the working folder.' },
      content: { type: 'string', description: 'Full new contents of the file.' }
    },
    required: ['path', 'content'],
    additionalProperties: false
  },
  edit_file: {
    type: 'object',
    properties: {
      path: { type: 'string', description: 'Path of the file to edit, absolute or relative to the working folder.' },
      old_text: {
        type: 'string',
        description: 'Exact text to replace. Must appear in the file at least once.'
      },
      new_text: { type: 'string', description: 'Text to put in its place.' }
    },
    required: ['path', 'old_text', 'new_text'],
    additionalProperties: false
  },
  delete_file: {
    type: 'object',
    properties: {
      path: { type: 'string', description: 'Path of the file to delete, absolute or relative to the working folder.' }
    },
    required: ['path'],
    additionalProperties: false
  },
  web_search: {
    type: 'object',
    properties: {
      query: { type: 'string', description: 'A short search query in general terms. Never include file contents or code.' }
    },
    required: ['query'],
    additionalProperties: false
  },
  fetch_page: {
    type: 'object',
    properties: {
      url: { type: 'string', description: 'Full http(s) address of a public web page.' }
    },
    required: ['url'],
    additionalProperties: false
  },
  download_file: {
    type: 'object',
    properties: {
      url: { type: 'string', description: 'Full http(s) address of the file.' },
      path: { type: 'string', description: 'Where to save it: a new file, relative to the working folder.' }
    },
    required: ['url', 'path'],
    additionalProperties: false
  }
}

export const TOOL_DESCRIPTIONS: Record<ToolName, string> = {
  list_directory: 'List the files and folders directly inside a directory.',
  read_file: 'Read the full text of a file.',
  write_file:
    'Create a file, or replace an existing file entirely. Prefer edit_file when changing part of a file that already exists.',
  edit_file:
    'Replace an exact run of text inside an existing file, leaving the rest untouched.',
  delete_file: 'Move a file to the recycle bin. It is recoverable afterwards.',
  web_search: 'Search the web. Returns titles, addresses and short snippets.',
  fetch_page: 'Read the text of a public web page. Its contents come from the internet, not from the user.',
  download_file: 'Download a file from the web into the working folder as a new file.'
}

export function isToolName(value: string): value is ToolName {
  return (ALL_TOOL_NAMES as readonly string[]).includes(value)
}
