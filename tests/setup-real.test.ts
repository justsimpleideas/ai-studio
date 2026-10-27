import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { describe, expect, it } from 'vitest'
import { EngineSetup } from '../src/main/image/setup.js'

/**
 * The real install, with real Python and pip, from a folder of wheels so nothing is
 * downloaded. Skipped unless STUDIO_SETUP_WHEELHOUSE names that folder; it takes a
 * few minutes and needs an NVIDIA GPU for the last step.
 *
 *   STUDIO_SETUP_WHEELHOUSE=/path/to/wheelhouse npx vitest run tests/setup-real.test.ts
 */

const wheelhouse = process.env['STUDIO_SETUP_WHEELHOUSE']
const engine = path.join(import.meta.dirname, '..', 'image-engine')

describe.skipIf(!wheelhouse)('installing the engine for real (from a wheelhouse)', () => {
  it('creates the environment and ends with PyTorch seeing the GPU', { timeout: 30 * 60_000 }, async () => {
    const venvDir = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'studio-setup-')), 'image-python')
    const phases: string[] = []
    const setup = new EngineSetup({
      venvDir,
      requirements: path.join(engine, 'requirements.txt'),
      constraints: path.join(engine, 'constraints.txt'),
      source: { wheelhouse: wheelhouse ?? null },
      onState: (s) => {
        if (phases.at(-1) !== s.phase) phases.push(s.phase)
      }
    })
    try {
      const python = await setup.install()
      expect(fs.existsSync(python)).toBe(true)
      expect(phases).toEqual(['checking', 'creating', 'torch', 'packages', 'verifying', 'done'])
      console.log(setup.state.gpu, '\n', setup.state.log.slice(-3).join('\n'))
      // Running it again repairs in place: no new environment, nothing to install.
      phases.length = 0
      await setup.install()
      expect(phases).toEqual(['checking', 'creating', 'torch', 'packages', 'verifying', 'done'])
      expect(setup.state.log.some((l) => l.includes('-m venv'))).toBe(false)
    } finally {
      console.log(`environment left at ${venvDir} (delete when done)`)
    }
  })
})
