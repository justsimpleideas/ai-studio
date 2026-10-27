import { TOOL_SCHEMAS, isToolName, type RawToolCall, type ToolCall } from './tools.js'

/**
 * Checking a tool call the model produced before anything acts on it.
 *
 * The engine already constrains generation with a grammar compiled from the
 * same schemas, so in the normal case nothing here can fail. That is not a
 * reason to skip it. The grammar only holds while the engine actually applied
 * it, and a model served over a future adapter, or a call replayed from a saved
 * conversation, arrives with no such guarantee. A check that trusts the other
 * side to have already checked is not a check.
 */

export type ValidationResult =
  | { ok: true; call: ToolCall }
  | { ok: false; error: string }

interface SchemaShape {
  properties: Record<string, { type: string }>
  required: string[]
}

export function validateToolCall(raw: RawToolCall, id: string): ValidationResult {
  if (!isToolName(raw.name)) {
    return { ok: false, error: `Unknown tool "${raw.name}".` }
  }

  if (typeof raw.args !== 'object' || raw.args === null || Array.isArray(raw.args)) {
    return { ok: false, error: `Arguments for ${raw.name} must be an object.` }
  }

  const schema = TOOL_SCHEMAS[raw.name] as unknown as SchemaShape
  const args = raw.args as Record<string, unknown>

  for (const key of schema.required) {
    if (!(key in args)) {
      return { ok: false, error: `${raw.name} is missing required argument "${key}".` }
    }
  }

  for (const [key, value] of Object.entries(args)) {
    const spec = schema.properties[key]
    if (!spec) {
      return { ok: false, error: `${raw.name} does not take an argument called "${key}".` }
    }
    if (spec.type === 'string' && typeof value !== 'string') {
      return {
        ok: false,
        error: `${raw.name}: "${key}" must be a string, got ${typeof value}.`
      }
    }
  }

  return { ok: true, call: { id, name: raw.name, args } as ToolCall }
}
