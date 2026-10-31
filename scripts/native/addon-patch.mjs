/**
 * The studio's change to node-llama-cpp: one load option, `tensorBufferOverrides`.
 *
 * llama.cpp can be told to keep some of a model's tensors in system memory while
 * the rest of each layer goes to the GPU (llama_model_params.tensor_buft_overrides,
 * what llama-server's --cpu-moe uses). For a mixture-of-experts model bigger than
 * the GPU that is the difference between a usable speed and a crawl: the experts,
 * most of the file and each one rarely used, stay on the CPU, and the attention
 * and shared weights every token goes through are all on the GPU. node-llama-cpp
 * does not pass that field through, so the studio rebuilds its addon with it.
 *
 * Only the addon (llama-addon.node) is rebuilt. llama.cpp itself, the CUDA
 * backend and everything else stay the prebuilt files node-llama-cpp ships, so
 * the build needs a C++ compiler and nothing from NVIDIA.
 *
 * Every edit is an exact text replacement at a known anchor, and any anchor not
 * found exactly once is an error: a new node-llama-cpp release that moved the
 * code fails the build loudly rather than producing an addon that silently
 * ignores the option.
 */

/** The node-llama-cpp release the patch is written against. */
export const NODE_LLAMA_CPP_VERSION = '3.21.1'

/** Reported by the patched addon, so the engine can tell it from the stock one. */
export const ADDON_FEATURE = 'tensorBufferOverrides'

/** Marks the patched JavaScript file, so patching twice is a no-op. */
export const JS_MARKER = '/* local-ai-studio: tensorBufferOverrides */'

function replaceOnce(text, anchor, replacement, where) {
  const first = text.indexOf(anchor)
  if (first === -1) throw new Error(`${where}: anchor not found: ${JSON.stringify(anchor.slice(0, 80))}`)
  if (text.indexOf(anchor, first + 1) !== -1) throw new Error(`${where}: anchor found more than once: ${JSON.stringify(anchor.slice(0, 80))}`)
  return text.slice(0, first) + replacement + text.slice(first + anchor.length)
}

/** AddonModel.h: storage for the patterns, which must outlive the load. */
export function patchAddonModelHeader(text) {
  const anchor = '        std::vector<llama_model_kv_override> kv_overrides;\n'
  return replaceOnce(
    text,
    anchor,
    anchor +
      '        // local-ai-studio: tensor buffer overrides. The strings are kept here because\n' +
      '        // llama.cpp reads the patterns while the model loads, after the constructor returns.\n' +
      '        std::vector<std::string> tensor_buft_override_patterns;\n' +
      '        std::vector<llama_model_tensor_buft_override> tensor_buft_overrides;\n',
    'AddonModel.h'
  )
}

/**
 * AddonModel.cpp: read `tensorBufferOverrides: { pattern: string, target: "cpu" }[]`.
 *
 * "cpu" is the only target. The buffer type is the one llama.cpp compares
 * against to recognise a CPU override (and then pick the CPU's repacked
 * layouts), so it has to be ggml_backend_cpu_buffer_type() itself.
 */
export function patchAddonModelSource(text) {
  const anchor = '        if (options.Has("onLoadProgress")) {\n'
  const block =
    '        // local-ai-studio: tensor buffer overrides (scripts/native/addon-patch.mjs).\n' +
    '        if (options.Has("tensorBufferOverrides")) {\n' +
    '            Napi::Array overrides = options.Get("tensorBufferOverrides").As<Napi::Array>();\n' +
    '            if (overrides.Length() >= llama_max_tensor_buft_overrides()) {\n' +
    '                throw Napi::RangeError::New(info.Env(), "Too many tensor buffer overrides");\n' +
    '            }\n' +
    '            tensor_buft_override_patterns.reserve(overrides.Length());\n' +
    '            for (uint32_t i = 0; i < overrides.Length(); i++) {\n' +
    '                Napi::Object item = overrides.Get(i).As<Napi::Object>();\n' +
    '                auto pattern = item.Get("pattern").As<Napi::String>().Utf8Value();\n' +
    '                auto target = item.Get("target").As<Napi::String>().Utf8Value();\n' +
    '                if (pattern.empty() || pattern.length() > 256) {\n' +
    '                    throw Napi::RangeError::New(info.Env(), "A tensor buffer override pattern must be 1 to 256 characters");\n' +
    '                }\n' +
    '                if (target != "cpu") {\n' +
    '                    throw Napi::TypeError::New(info.Env(), "Unsupported tensor buffer override target: " + target);\n' +
    '                }\n' +
    '                tensor_buft_override_patterns.push_back(pattern);\n' +
    '            }\n' +
    '            for (const auto & pattern : tensor_buft_override_patterns) {\n' +
    '                tensor_buft_overrides.push_back({ pattern.c_str(), ggml_backend_cpu_buffer_type() });\n' +
    '            }\n' +
    '            if (!tensor_buft_overrides.empty()) {\n' +
    '                tensor_buft_overrides.push_back({ nullptr, nullptr });\n' +
    '                model_params.tensor_buft_overrides = tensor_buft_overrides.data();\n' +
    '            }\n' +
    '        }\n' +
    '\n'
  return replaceOnce(text, anchor, block + anchor, 'AddonModel.cpp')
}

/** addon.cpp: a `getStudioFeatures()` export naming what this build adds. */
export function patchAddonEntry(text) {
  const fnAnchor = 'Napi::Object registerCallback(Napi::Env env, Napi::Object exports) {\n'
  const withFn = replaceOnce(
    text,
    fnAnchor,
    '// local-ai-studio: lets the engine tell this build from the stock addon.\n' +
      'Napi::Value addonGetStudioFeatures(const Napi::CallbackInfo& info) {\n' +
      '    Napi::Array features = Napi::Array::New(info.Env(), 1);\n' +
      `    features.Set((uint32_t)0, Napi::String::New(info.Env(), "${ADDON_FEATURE}"));\n` +
      '    return features;\n' +
      '}\n' +
      '\n' +
      fnAnchor,
    'addon.cpp'
  )
  const exportAnchor = '        Napi::PropertyDescriptor::Function("markLoaded", markLoaded),\n'
  return replaceOnce(
    withFn,
    exportAnchor,
    exportAnchor + '        Napi::PropertyDescriptor::Function("getStudioFeatures", addonGetStudioFeatures),\n',
    'addon.cpp'
  )
}

/**
 * dist/evaluator/LlamaModel/LlamaModel.js: pass the option from loadModel() to
 * the addon. The library builds the addon's options field by field, so an
 * extra field would otherwise be dropped.
 */
export function patchLlamaModelJs(text) {
  if (text.includes(JS_MARKER)) return text
  const ctorAnchor = 'constructor({ modelPath, gpuLayers, vocabOnly = false, useMmap, useDirectIo, useMlock = false, checkTensors, lazyMode, onLoadProgress, loadSignal, metadataOverrides }, '
  let out = replaceOnce(
    text,
    ctorAnchor,
    'constructor({ modelPath, gpuLayers, vocabOnly = false, useMmap, useDirectIo, useMlock = false, checkTensors, lazyMode, onLoadProgress, loadSignal, metadataOverrides, tensorBufferOverrides }, ',
    'LlamaModel.js'
  )
  const optionAnchor = '            hasLoadAbortSignal: loadSignal != null,\n'
  out = replaceOnce(
    out,
    optionAnchor,
    optionAnchor + `            ${JS_MARKER}\n            tensorBufferOverrides,\n`,
    'LlamaModel.js'
  )
  return out
}
