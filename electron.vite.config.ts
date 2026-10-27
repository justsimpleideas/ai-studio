import { resolve } from 'node:path'
import { defineConfig, externalizeDepsPlugin } from 'electron-vite'
import react from '@vitejs/plugin-react'

// The engine host is built as a separate entry: it runs in its own utilityProcess
// so that a native crash inside llama.cpp cannot take the whole app down, and so
// that it never shares a process with the code that is allowed to touch the disk.
export default defineConfig({
  main: {
    plugins: [externalizeDepsPlugin()],
    build: {
      rollupOptions: {
        input: {
          index: resolve('src/main/index.ts'),
          engineHost: resolve('src/engine/host.ts')
        }
      }
    },
    resolve: {
      alias: { '@shared': resolve('src/shared') }
    }
  },
  preload: {
    plugins: [externalizeDepsPlugin()],
    build: {
      rollupOptions: {
        input: { index: resolve('src/preload/index.ts') },
        // Emitted as CommonJS on purpose. A sandboxed preload cannot load an ES
        // module, and the sandbox is worth more here than the module syntax is:
        // this is the one script that sits on both sides of the context bridge.
        output: { format: 'cjs', entryFileNames: '[name].cjs' }
      }
    },
    resolve: {
      alias: { '@shared': resolve('src/shared') }
    }
  },
  renderer: {
    root: resolve('src/renderer'),
    plugins: [react()],
    build: {
      rollupOptions: {
        input: { index: resolve('src/renderer/index.html') }
      }
    },
    resolve: {
      alias: {
        '@shared': resolve('src/shared'),
        '@': resolve('src/renderer')
      }
    },
    server: {
      // A fixed port, strict so a clash with another dev server fails loudly
      // instead of silently moving to a different one.
      port: 5197,
      strictPort: true
    }
  }
})
