/**
 * Print what a GGUF file actually is, without loading it.
 *
 * Useful because a model file's name tells you nothing reliable — a file called
 * "model.gguf", or a blob named only after its hash, still carries its real
 * architecture, size and chat template in the header. Reading that first is how
 * you find out whether a model supports tool calling before waiting ten minutes
 * for it to load and then discovering it does not.
 *
 *   node scripts/gguf-probe.mjs <path-to.gguf> [...]
 */
import { readGgufFileInfo } from 'node-llama-cpp'

const files = process.argv.slice(2)
if (files.length === 0) {
  console.error('usage: node scripts/gguf-probe.mjs <file.gguf> [...]')
  process.exit(1)
}

const GB = 1024 ** 3

for (const file of files) {
  let info
  try {
    info = await readGgufFileInfo(file)
  } catch (err) {
    console.log(`${file}\n  not readable as GGUF: ${err.message}\n`)
    continue
  }

  const general = info.metadata.general ?? {}
  const architecture = general.architecture
  const arch = info.metadata[architecture] ?? {}
  const template = info.metadata.tokenizer?.chat_template

  console.log(file)
  console.log(`  name          : ${general.name ?? '(unnamed)'}`)
  console.log(`  architecture  : ${architecture ?? '(unknown)'}`)
  if (general.size_label) console.log(`  size label    : ${general.size_label}`)
  console.log(`  file type     : ${general.file_type ?? '(unknown)'}`)
  console.log(`  train context : ${arch.context_length ?? '(unknown)'}`)
  console.log(`  layers        : ${arch.block_count ?? '(unknown)'}`)
  console.log(`  embedding     : ${arch.embedding_length ?? '(unknown)'}`)
  console.log(`  tensors       : ${info.tensorInfo?.length ?? '(unknown)'}`)
  if (info.splicedParts > 1) console.log(`  parts         : ${info.splicedParts}`)
  console.log(`  size on disk  : ${(Number(info.totalFileSize ?? 0) / GB).toFixed(2)} GB`)

  // Tool use is a property of the chat template, not of the weights. A template
  // with no mention of tools means the model was not trained to emit calls in a
  // form its wrapper can read, whatever the model card claims.
  if (!template) {
    console.log('  chat template : none embedded')
  } else {
    const mentionsTools = /tool|function/i.test(template)
    console.log(
      `  chat template : ${template.length} chars, ${
        mentionsTools ? 'mentions tools/functions' : 'no mention of tools'
      }`
    )
  }
  console.log()
}
