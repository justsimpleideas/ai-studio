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

/**
 * The same choice, given to the model load.
 *
 * How many layers go on the GPU is decided when the weights are loaded, by
 * fitting the context next to them, and that fit sizes the context with the
 * model's defaults. Given only to the context, an 8-bit cache would still be
 * planned for as 16-bit: half its room on the GPU left unused instead of
 * holding more layers.
 */
export function kvModelOptions(kv: KvCacheType): {
  defaultContextFlashAttention?: true
  experimentalDefaultContextKvCacheKeyType?: 'Q8_0'
  experimentalDefaultContextKvCacheValueType?: 'Q8_0'
} {
  if (kv === 'q8_0') {
    return {
      defaultContextFlashAttention: true,
      experimentalDefaultContextKvCacheKeyType: 'Q8_0',
      experimentalDefaultContextKvCacheValueType: 'Q8_0'
    }
  }
  return {}
}
