#!/usr/bin/env node
/**
 * Builds node-llama-cpp's CUDA addon with the studio's one change (addon-patch.mjs)
 * and puts it in place.
 *
 *   node scripts/native/addon.mjs build     compile build/native/win-x64-cuda/llama-addon.node
 *   node scripts/native/addon.mjs install   put it in node_modules, keeping the stock file beside it
 *   node scripts/native/addon.mjs restore   put the stock file back
 *   node scripts/native/addon.mjs status    say which one is in place
 *
 * Windows x64 only, and only the CUDA variant: that is where a mixture-of-experts
 * model bigger than the GPU is worth splitting by tensor. It needs the MSVC
 * compiler (Visual Studio Build Tools, "Desktop development with C++" or just
 * the x64 compiler and a Windows SDK) and git. Nothing is downloaded: the
 * llama.cpp headers come from the source bundle node-llama-cpp ships, and the
 * addon links against the prebuilt llama.cpp DLLs already in node_modules, which
 * are left exactly as they are.
 *
 * Options: --work <dir> for the scratch folder (default: the system temp folder),
 * --jobs <n> for parallel compiles (default 4).
 */
import { execFileSync, spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  ADDON_FEATURE,
  JS_MARKER,
  NODE_LLAMA_CPP_VERSION,
  patchAddonEntry,
  patchAddonModelHeader,
  patchAddonModelSource,
  patchLlamaModelJs
} from './addon-patch.mjs'
import { moduleDefinition, readExports, readImports } from './pe.mjs'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..')
const NLC = path.join(ROOT, 'node_modules', 'node-llama-cpp')
const VARIANT = 'win-x64-cuda'
const BINS = path.join(ROOT, 'node_modules', '@node-llama-cpp', VARIANT, 'bins', VARIANT)
const ADDON = path.join(BINS, 'llama-addon.node')
const STOCK = path.join(BINS, 'llama-addon.stock.node')
const LLAMA_MODEL_JS = path.join(NLC, 'dist', 'evaluator', 'LlamaModel', 'LlamaModel.js')
const OUT_DIR = path.join(ROOT, 'build', 'native', VARIANT)
const OUT = path.join(OUT_DIR, 'llama-addon.node')

function fail(message) {
  console.error(`addon: ${message}`)
  process.exit(1)
}

function option(name, fallback) {
  const i = process.argv.indexOf(`--${name}`)
  return i === -1 ? fallback : process.argv[i + 1]
}

const sha256 = (file) => createHash('sha256').update(fs.readFileSync(file)).digest('hex')

function checkPlatform() {
  if (process.platform !== 'win32' || process.arch !== 'x64') fail('only Windows x64 is supported')
  const version = JSON.parse(fs.readFileSync(path.join(NLC, 'package.json'), 'utf8')).version
  if (version !== NODE_LLAMA_CPP_VERSION) {
    fail(`node-llama-cpp is ${version}, the patch is written for ${NODE_LLAMA_CPP_VERSION}; check addon-patch.mjs against the new release first`)
  }
  if (!fs.existsSync(ADDON)) fail(`no prebuilt addon at ${path.relative(ROOT, ADDON)} (run npm ci)`)
}

/** The environment vcvars64.bat sets up, read back from a child cmd. */
function msvcEnv() {
  const vswhere = path.join(process.env['ProgramFiles(x86)'] ?? 'C:\\Program Files (x86)', 'Microsoft Visual Studio', 'Installer', 'vswhere.exe')
  if (!fs.existsSync(vswhere)) fail('Visual Studio Build Tools not found (no vswhere.exe)')
  const install = execFileSync(
    vswhere,
    ['-latest', '-products', '*', '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'],
    { encoding: 'utf8' }
  ).trim()
  if (!install) fail('no Visual Studio installation with the x64 C++ compiler')
  const vcvars = path.join(install, 'VC', 'Auxiliary', 'Build', 'vcvars64.bat')
  // vcvars64.bat looks for vswhere.exe on the PATH as well.
  const base = { ...process.env }
  const pathKey = Object.keys(base).find((k) => k.toUpperCase() === 'PATH') ?? 'PATH'
  base[pathKey] = `${path.dirname(vswhere)};${base[pathKey] ?? ''}`
  const out = execFileSync('cmd.exe', ['/d', '/s', '/c', `"call "${vcvars}" >nul && set"`], {
    encoding: 'utf8',
    windowsVerbatimArguments: true,
    env: base
  })
  const env = {}
  for (const line of out.split(/\r?\n/)) {
    const eq = line.indexOf('=')
    if (eq > 0) env[line.slice(0, eq)] = line.slice(eq + 1)
  }
  if (!env.VCToolsVersion) fail('vcvars64.bat did not set up the compiler')
  return env
}

function run(exe, args, env, cwd) {
  const r = spawnSync(exe, args, { env, cwd, stdio: 'inherit', windowsHide: true })
  if (r.error) fail(`${exe}: ${r.error.message}`)
  if (r.status !== 0) fail(`${exe} exited with ${r.status}`)
}

/** llama.cpp's source at the commit the prebuilt DLLs were built from. */
function llamaSource(work) {
  const dir = path.join(work, 'llama.cpp')
  const bundle = path.join(NLC, 'llama', 'gitRelease.bundle')
  if (!fs.existsSync(bundle)) fail('node-llama-cpp has no llama.cpp source bundle')
  const want = execFileSync('git', ['bundle', 'list-heads', bundle], { encoding: 'utf8' }).trim().split(/\s+/)[0]
  if (fs.existsSync(dir)) {
    const have = execFileSync('git', ['-C', dir, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim()
    if (have === want) return { dir, commit: want }
    fs.rmSync(dir, { recursive: true, force: true })
  }
  execFileSync('git', ['-c', 'advice.detachedHead=false', 'clone', '--quiet', bundle, dir], { stdio: 'inherit' })
  return { dir, commit: want }
}

/** The addon's sources, copied out of node_modules and patched. */
function addonSource(work) {
  const from = path.join(NLC, 'llama', 'addon')
  const dir = path.join(work, 'addon')
  fs.rmSync(dir, { recursive: true, force: true })
  fs.cpSync(from, dir, { recursive: true })
  const edit = (name, patch) => {
    const file = path.join(dir, name)
    fs.writeFileSync(file, patch(fs.readFileSync(file, 'utf8')))
  }
  edit('AddonModel.h', patchAddonModelHeader)
  edit('AddonModel.cpp', patchAddonModelSource)
  edit('addon.cpp', patchAddonEntry)
  const sources = []
  const walk = (d) => {
    for (const entry of fs.readdirSync(d, { withFileTypes: true })) {
      const p = path.join(d, entry.name)
      if (entry.isDirectory()) walk(p)
      else if (entry.name.endsWith('.cpp')) sources.push(p)
    }
  }
  walk(dir)
  return { dir, sources }
}

/** Import libraries for the prebuilt DLLs the stock addon links to, and for node.exe's N-API. */
function importLibraries(work, env) {
  const dir = path.join(work, 'lib')
  fs.mkdirSync(dir, { recursive: true })
  const { normal, delayed } = readImports(fs.readFileSync(ADDON))
  const local = normal.filter((name) => fs.existsSync(path.join(BINS, name)))
  if (!local.some((n) => /^llama\./i.test(n)) || !local.includes('ggml-base.dll')) {
    fail(`the stock addon's imports are not what this script expects: ${local.join(', ')}`)
  }
  if (delayed.length !== 1 || delayed[0].toUpperCase() !== 'NODE.EXE') fail(`unexpected delay-loaded imports: ${delayed.join(', ')}`)
  const libs = []
  for (const dll of local) {
    const def = path.join(dir, dll.replace(/\.dll$/i, '.def'))
    const lib = path.join(dir, dll.replace(/\.dll$/i, '.lib'))
    fs.writeFileSync(def, moduleDefinition(dll, readExports(fs.readFileSync(path.join(BINS, dll)))))
    run('lib.exe', ['/nologo', '/machine:x64', `/def:${def}`, `/out:${lib}`], env, dir)
    libs.push(lib)
  }
  // N-API comes from the process that loads the addon (node.exe or the Electron
  // binary); the delay-load hook below redirects NODE.EXE to whichever it is.
  const headers = path.join(ROOT, 'node_modules', 'node-api-headers', 'def')
  const napi = ['js_native_api.def', 'node_api.def']
    .flatMap((f) => fs.readFileSync(path.join(headers, f), 'utf8').split(/\r?\n/).slice(1))
    .filter((l) => l.trim() && l.trim() !== 'EXPORTS')
  const nodeDef = path.join(dir, 'node.def')
  fs.writeFileSync(nodeDef, ['NAME NODE.EXE', 'EXPORTS', ...napi].join('\r\n') + '\r\n')
  const nodeLib = path.join(dir, 'node.lib')
  run('lib.exe', ['/nologo', '/machine:x64', `/def:${nodeDef}`, `/out:${nodeLib}`], env, dir)
  libs.push(nodeLib)
  return { libs, dlls: local }
}

function build() {
  checkPlatform()
  const work = path.resolve(option('work', path.join(os.tmpdir(), 'local-ai-studio-addon')))
  const jobs = Math.max(1, Math.min(32, Number(option('jobs', '4')) || 4))
  fs.mkdirSync(work, { recursive: true })
  const env = msvcEnv()
  console.log(`addon: MSVC ${env.VCToolsVersion}, work folder ${work}`)
  const llama = llamaSource(work)
  const addon = addonSource(work)
  const { libs, dlls } = importLibraries(work, env)
  const hook = path.join(ROOT, 'node_modules', 'cmake-js', 'lib', 'cpp', 'win_delay_load_hook.cc')
  if (!fs.existsSync(hook)) fail('cmake-js delay-load hook not found')

  const obj = path.join(work, 'obj')
  fs.rmSync(obj, { recursive: true, force: true })
  fs.mkdirSync(obj, { recursive: true })
  const include = [
    path.join(ROOT, 'node_modules', 'node-addon-api'),
    path.join(ROOT, 'node_modules', 'node-api-headers', 'include'),
    llama.dir,
    path.join(llama.dir, 'include'),
    path.join(llama.dir, 'ggml', 'include'),
    path.join(llama.dir, 'common'),
    path.join(llama.dir, 'src'),
    addon.dir
  ]
  // What the prebuilt build's targets export to their users (llama.cpp's CMake:
  // LLAMA_SHARED, GGML_SHARED, GGML_BACKEND_SHARED, GGML_BACKEND_DL), and the
  // release settings CMake uses for MSVC.
  const defines = ['NAPI_VERSION=7', 'BUILDING_NODE_EXTENSION', 'LLAMA_SHARED', 'GGML_SHARED', 'GGML_BACKEND_SHARED', 'GGML_BACKEND_DL', 'NDEBUG', '_CRT_SECURE_NO_WARNINGS']
  run(
    'cl.exe',
    [
      '/nologo', '/c', '/std:c++17', '/EHsc', '/MD', '/O2', '/Ob2', '/utf-8', '/bigobj', '/W1', `/MP${jobs}`,
      ...defines.map((d) => `/D${d}`),
      ...include.map((i) => `/I${i}`),
      `/Fo${obj}\\`,
      ...addon.sources,
      hook
    ],
    env,
    work
  )
  fs.mkdirSync(OUT_DIR, { recursive: true })
  const objects = fs.readdirSync(obj).filter((f) => f.endsWith('.obj')).map((f) => path.join(obj, f))
  run(
    'link.exe',
    ['/nologo', '/DLL', '/MACHINE:X64', '/OPT:REF', '/OPT:ICF', '/INCREMENTAL:NO', `/OUT:${OUT}`, ...objects, ...libs, 'delayimp.lib', '/DELAYLOAD:NODE.EXE'],
    env,
    work
  )
  for (const leftover of ['llama-addon.lib', 'llama-addon.exp']) fs.rmSync(path.join(OUT_DIR, leftover), { force: true })

  // The new file must link to exactly what the stock one does, or it would load
  // some other llama.cpp than the one beside it.
  const built = readImports(fs.readFileSync(OUT))
  const missing = dlls.filter((d) => !built.normal.includes(d))
  if (missing.length > 0) fail(`the built addon does not import ${missing.join(', ')}`)
  const record = {
    variant: VARIANT,
    nodeLlamaCpp: NODE_LLAMA_CPP_VERSION,
    llamaCppCommit: llama.commit,
    msvc: env.VCToolsVersion,
    stockSha256: sha256(fs.existsSync(STOCK) ? STOCK : ADDON),
    sha256: sha256(OUT),
    feature: ADDON_FEATURE
  }
  fs.writeFileSync(path.join(OUT_DIR, 'build.json'), JSON.stringify(record, null, 2) + '\n')
  console.log(`addon: built ${path.relative(ROOT, OUT)} (${record.sha256})`)
}

/** Which addon file is in place: the stock one, the studio's build, or something else. */
function state() {
  const record = fs.existsSync(path.join(OUT_DIR, 'build.json')) ? JSON.parse(fs.readFileSync(path.join(OUT_DIR, 'build.json'), 'utf8')) : null
  const current = fs.existsSync(ADDON) ? sha256(ADDON) : null
  const js = fs.readFileSync(LLAMA_MODEL_JS, 'utf8').includes(JS_MARKER)
  if (current !== null && record && current === record.sha256) return { addon: 'patched', js, record }
  if (current !== null && record && current === record.stockSha256) return { addon: 'stock', js, record }
  if (!fs.existsSync(STOCK)) return { addon: 'stock', js, record }
  return { addon: 'unknown', js, record }
}

function install() {
  checkPlatform()
  if (!fs.existsSync(OUT)) fail('nothing built yet (run: node scripts/native/addon.mjs build)')
  const { addon, record } = state()
  if (addon === 'unknown') fail('the addon in node_modules is neither the stock file nor this build; run npm ci, then build again')
  if (addon === 'stock') {
    if (sha256(ADDON) !== record.stockSha256) fail('the stock addon changed since the build; build again')
    fs.copyFileSync(ADDON, STOCK)
    fs.copyFileSync(OUT, ADDON)
  }
  const js = fs.readFileSync(LLAMA_MODEL_JS, 'utf8')
  const patched = patchLlamaModelJs(js)
  if (patched !== js) fs.writeFileSync(LLAMA_MODEL_JS, patched)
  console.log(`addon: installed (${sha256(ADDON)})`)
}

function restore() {
  if (fs.existsSync(STOCK)) {
    fs.copyFileSync(STOCK, ADDON)
    fs.rmSync(STOCK)
  }
  console.log('addon: stock addon in place. The JavaScript patch only adds an option the stock addon ignores; npm ci removes it.')
}

const command = process.argv[2]
if (command === 'build') build()
else if (command === 'install') install()
else if (command === 'restore') restore()
else if (command === 'status') {
  const { addon, js } = state()
  console.log(`addon: ${addon}, LlamaModel.js ${js ? 'patched' : 'stock'}`)
} else fail('usage: addon.mjs build | install | restore | status [--work <dir>] [--jobs <n>]')
