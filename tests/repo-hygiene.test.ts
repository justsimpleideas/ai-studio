import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * The repository is code only. Model weights, Python environments, per-user
 * settings and generated media are different on every machine and live in the
 * user's data folder, so none of them may ever be committed. Checks every file
 * git would include (tracked, or new and not ignored).
 */

const ROOT = path.join(import.meta.dirname, '..')

function repoFiles(): string[] | null {
  try {
    const out = execFileSync('git', ['ls-files', '--cached', '--others', '--exclude-standard', '-z'], { cwd: ROOT, encoding: 'utf8' })
    return out.split('\0').filter((f) => f && fs.existsSync(path.join(ROOT, f)))
  } catch {
    return null
  }
}

const files = repoFiles()

// Formats that hold weights or packages, never source.
const FORBIDDEN = /\.(gguf|safetensors|ckpt|pt|pth|bin|onnx|pkl|pickle|whl|db|exe|dll|zip|7z|mp4|webm|webp)$/i
// The only binary files the repository carries: the app icon, and the README's
// screenshots (WebP, in one folder, each checked for size like everything else).
const SCREENSHOT = /^docs\/screenshots\/[a-z0-9-]+\.webp$/
const isAllowedBinary = (f: string): boolean => f === 'resources/icon.png' || f === 'resources/icon.ico' || SCREENSHOT.test(f)
const MAX_BYTES = 1024 * 1024

describe.skipIf(files === null)('what the repository contains', () => {
  it('has no weights, environments, databases or media', () => {
    expect(files!.filter((f) => FORBIDDEN.test(f) && !SCREENSHOT.test(f))).toEqual([])
    expect(files!.filter((f) => /(^|\/)(models|image-python|\.venv|venv|wheelhouse|userData|outputs|release|out)\//.test(f))).toEqual([])
  })

  it('has no large files', () => {
    const large = files!.filter((f) => fs.statSync(path.join(ROOT, f)).size > MAX_BYTES)
    expect(large).toEqual([])
  })

  it('has no binary files but the icon and the screenshots', () => {
    const binary = files!.filter((f) => {
      if (isAllowedBinary(f)) return false
      const head = fs.readFileSync(path.join(ROOT, f)).subarray(0, 8000)
      return head.includes(0)
    })
    expect(binary).toEqual([])
  })

  it('names no path from a particular machine', () => {
    // A drive path into a real account's home folder (the README's placeholder "You" is fine).
    const userPath = /[A-Za-z]:[\\/]+Users[\\/]+(?!You[\\/]|<)[A-Za-z0-9._-]+[\\/]/
    const hits = files!
      .filter((f) => !isAllowedBinary(f) && f !== 'package-lock.json')
      .filter((f) => userPath.test(fs.readFileSync(path.join(ROOT, f), 'utf8')))
    expect(hits).toEqual([])
  })
})
