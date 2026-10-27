import { describe, expect, it } from 'vitest'
import { validateToolCall } from '../src/shared/validate.js'

/**
 * The second line of defence.
 *
 * In normal operation the grammar makes every one of these impossible, which is
 * exactly why they are worth testing: nobody would notice this check rotting.
 */
describe('validateToolCall', () => {
  const id = 'call-1'

  it('accepts a well-formed call', () => {
    const result = validateToolCall(
      { name: 'read_file', args: { path: 'C:/work/notes.txt' } },
      id
    )
    expect(result.ok).toBe(true)
    if (result.ok) {
      expect(result.call.name).toBe('read_file')
      expect(result.call.id).toBe(id)
    }
  })

  it('rejects a tool that does not exist', () => {
    const result = validateToolCall({ name: 'run_shell', args: {} }, id)
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error).toContain('run_shell')
  })

  it('rejects arguments that are not an object', () => {
    expect(validateToolCall({ name: 'read_file', args: 'C:/x' }, id).ok).toBe(false)
    expect(validateToolCall({ name: 'read_file', args: null }, id).ok).toBe(false)
    expect(validateToolCall({ name: 'read_file', args: ['C:/x'] }, id).ok).toBe(false)
  })

  it('rejects a missing required argument', () => {
    const result = validateToolCall({ name: 'edit_file', args: { path: 'C:/x' } }, id)
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error).toContain('old_text')
  })

  it('rejects an argument the tool does not take', () => {
    const result = validateToolCall(
      { name: 'read_file', args: { path: 'C:/x', encoding: 'utf8' } },
      id
    )
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error).toContain('encoding')
  })

  it('rejects an argument of the wrong type', () => {
    const result = validateToolCall({ name: 'read_file', args: { path: 42 } }, id)
    expect(result.ok).toBe(false)
    if (!result.ok) expect(result.error).toContain('must be a string')
  })

  it('accepts every tool in its well-formed shape', () => {
    const calls = [
      { name: 'list_directory', args: { path: 'C:/w' } },
      { name: 'read_file', args: { path: 'C:/w/a.txt' } },
      { name: 'write_file', args: { path: 'C:/w/a.txt', content: 'x' } },
      { name: 'edit_file', args: { path: 'C:/w/a.txt', old_text: 'a', new_text: 'b' } },
      { name: 'delete_file', args: { path: 'C:/w/a.txt' } }
    ]
    for (const call of calls) {
      expect(validateToolCall(call, id).ok, call.name).toBe(true)
    }
  })
})
