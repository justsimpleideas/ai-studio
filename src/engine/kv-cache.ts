import type { KvCacheType } from '@shared/config.js'

/**
 * Context options for a KV cache type.
 *
 * 16-bit is the library default, so it adds nothing. 8-bit sets both halves of
 * the cache and turns flash attention on, which llama.cpp requires for a
 * quantized value cache; without it the context fails to build, and the
 * application falls back to 16-bit.
 */
export function kvCacheOptions(kv: KvCacheType): {
  flashAttention?: true
  experimentalKvCacheKeyType?: 'Q8_0'
  experimentalKvCacheValueType?: 'Q8_0'
} {
  if (kv === 'q8_0') {
    return { flashAttention: true, experimentalKvCacheKeyType: 'Q8_0', experimentalKvCacheValueType: 'Q8_0' }
  }
  return {}
}
