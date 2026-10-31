/**
 * A server-sent events reader, for the streamed reply of a chat completion.
 *
 * Only what the server emits is handled: `data:` lines, events separated by a
 * blank line, and the final `[DONE]`. Bytes arrive in arbitrary pieces, so the
 * parser keeps whatever has not ended in a newline for the next piece.
 */
export class SseParser {
  private rest = ''

  /** Feed a piece of the stream; returns the data of each complete event in it. `[DONE]` is returned as is. */
  push(piece: string): string[] {
    // Lines may end in CRLF; a CR left at the end of a piece waits for its LF.
    this.rest = (this.rest + piece).replace(/\r\n/g, '\n')
    const events: string[] = []
    let at: number
    while ((at = this.rest.indexOf('\n\n')) !== -1) {
      const block = this.rest.slice(0, at)
      this.rest = this.rest.slice(at + 2)
      const data = block
        .split('\n')
        .map((line) => line.replace(/\r$/, ''))
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).replace(/^ /, ''))
      if (data.length > 0) events.push(data.join('\n'))
    }
    return events
  }

  /** Whatever a stream that ended without a final blank line still held. */
  flush(): string[] {
    const tail = this.rest
    this.rest = ''
    return tail.trim() ? this.push(`${tail}\n\n`) : []
  }
}
