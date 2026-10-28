import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import { DEFAULT_CONFIG, isKvCacheType, parseBackground, type AppConfig } from '@shared/config.js'
import { DEFAULT_PERMISSION_SETTINGS } from '@shared/permissions.js'

/**
 * Configuration on disk.
 *
 * Written atomically, and read defensively. A config file that has been
 * hand-edited into nonsense, or truncated by a power cut, must not come back as
 * a permission set with an empty roots array that silently means something
 * different from what the user last saw. Anything unreadable falls back to the
 * defaults, which are the restrictive ones.
 */
export class ConfigStore {
  private cached: AppConfig

  constructor(private readonly file: string) {
    this.cached = this.readFromDisk()
  }

  get(): AppConfig {
    return this.cached
  }

  private readFromDisk(): AppConfig {
    try {
      const parsed = JSON.parse(fs.readFileSync(this.file, 'utf8')) as Partial<AppConfig>
      return {
        permissions: {
          ...DEFAULT_PERMISSION_SETTINGS,
          ...(parsed.permissions ?? {}),
          roots: parsed.permissions?.roots ?? [],
          rules: parsed.permissions?.rules ?? [],
          modes: { ...DEFAULT_PERMISSION_SETTINGS.modes, ...(parsed.permissions?.modes ?? {}) }
        },
        rules: parsed.rules ?? DEFAULT_CONFIG.rules,
        model: {
          ...DEFAULT_CONFIG.model,
          ...(parsed.model ?? {}),
          // An unknown value would reach the engine as a cache type it cannot build.
          kvCache: isKvCacheType(parsed.model?.kvCache) ? parsed.model.kvCache : DEFAULT_CONFIG.model.kvCache
        },
        image: {
          ...DEFAULT_CONFIG.image,
          ...(parsed.image ?? {}),
          modelsFolder: typeof parsed.image?.modelsFolder === 'string' ? parsed.image.modelsFolder : null,
          partFolders: Array.isArray(parsed.image?.partFolders)
            ? parsed.image.partFolders.filter((f): f is string => typeof f === 'string')
            : [],
          // Anything but an explicit yes keeps prompts out of exported files.
          includeSettingsInExports: parsed.image?.includeSettingsInExports === true
        },
        // Anything unrecognised means the default: closing the window quits.
        background: parseBackground(parsed.background),
        lastWorkspaceId: typeof parsed.lastWorkspaceId === 'string' ? parsed.lastWorkspaceId : null
      }
    } catch {
      return structuredClone(DEFAULT_CONFIG)
    }
  }

  async update(mutate: (config: AppConfig) => void): Promise<AppConfig> {
    const next = structuredClone(this.cached)
    mutate(next)
    this.cached = next
    await this.flush()
    return next
  }

  private async flush(): Promise<void> {
    await fsp.mkdir(path.dirname(this.file), { recursive: true })
    const temp = `${this.file}.${process.pid}.tmp`
    await fsp.writeFile(temp, JSON.stringify(this.cached, null, 2), 'utf8')
    await fsp.rename(temp, this.file)
  }
}
