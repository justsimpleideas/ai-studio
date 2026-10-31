import { describe, expect, it } from 'vitest'
import { modelPickerStart } from '../src/main/picker.js'

describe('where the model dialog opens', () => {
  const last = 'C:/Users/You/models/chat/model.gguf'

  it('on the last model loaded', () => {
    expect(modelPickerStart(last, () => true)).toBe(last)
  })

  it('in its folder when the file has gone', () => {
    expect(modelPickerStart(last, (p) => p === 'C:/Users/You/models/chat')).toBe('C:/Users/You/models/chat')
  })

  it('where the system likes when neither is there, or nothing was loaded yet', () => {
    expect(modelPickerStart(last, () => false)).toBeUndefined()
    expect(modelPickerStart(null, () => true)).toBeUndefined()
  })
})
