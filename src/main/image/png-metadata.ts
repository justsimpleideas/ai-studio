/**
 * Removing what a PNG says about itself.
 *
 * The gallery's own copy of an image keeps its generation settings in a text
 * chunk, which is how "Use these settings" and dropping an image back in work.
 * A copy saved out of the studio is a different matter: a prompt can be
 * private, and a shared image carries it wherever it goes. So exports go
 * through here unless the user asked to include the settings.
 *
 * Chunks are copied byte for byte, so the picture is untouched and every
 * remaining chunk keeps its checksum. Anything that does not parse as a PNG is
 * refused rather than written out with its metadata possibly still inside.
 */

const SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])

/** Chunks that can describe the image: text (plain, compressed, international), edit time, camera data. */
export const METADATA_CHUNKS: ReadonlySet<string> = new Set(['tEXt', 'zTXt', 'iTXt', 'tIME', 'eXIf'])

export function withoutPngMetadata(png: Buffer): Buffer {
  if (png.length < SIGNATURE.length || !png.subarray(0, SIGNATURE.length).equals(SIGNATURE)) {
    throw new Error('That file is not a PNG image.')
  }
  const kept: Buffer[] = [png.subarray(0, SIGNATURE.length)]
  let at = SIGNATURE.length
  while (at + 12 <= png.length) {
    const length = png.readUInt32BE(at)
    const type = png.toString('latin1', at + 4, at + 8)
    const end = at + 12 + length
    if (!/^[A-Za-z]{4}$/.test(type) || end > png.length) break
    if (!METADATA_CHUNKS.has(type)) kept.push(png.subarray(at, end))
    // Whatever follows the end marker is not part of the image and is dropped too.
    if (type === 'IEND') return Buffer.concat(kept)
    at = end
  }
  throw new Error('The PNG image is damaged, so it was not saved.')
}
