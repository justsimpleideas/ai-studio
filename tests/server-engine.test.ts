import type { ChatHistoryItem } from 'node-llama-cpp'
import { describe, expect, it } from 'vitest'
import { emptyServerLog, parseDeviceList, readServerLine, serverArgs, serverEnv } from '../src/engine/server/args.js'
import { measureRemote, type MeasureClient } from '../src/engine/server/measure.js'
import {
  completionBody,
  parseArguments,
  readChunk,
  ReplyAccumulator,
  toServerMessages,
  toolDefinitions
} from '../src/engine/server/messages.js'
import { SseParser } from '../src/engine/server/sse.js'
import { DEFAULT_CONFIG, parseServer } from '../src/shared/config.js'

describe('starting llama-server', () => {
  const base = {
    modelPath: 'C:/models/a.gguf',
    port: 41234,
    contextSize: 8192,
    kvCache: 'f16' as const,
    gpuLayers: null,
    threads: null,
    mixtureOfExperts: false,
    expertsOnCpu: null
  }

  it('listens on the loopback address only, with its web page off and the file template on', () => {
    const args = serverArgs(base)
    expect(args).toContain('--host')
    expect(args[args.indexOf('--host') + 1]).toBe('127.0.0.1')
    expect(args[args.indexOf('--port') + 1]).toBe('41234')
    expect(args).toContain('--no-webui')
    expect(args).toContain('--offline')
    expect(args).toContain('--no-cors-credentials')
    expect(args).toContain('--no-slots')
    expect(args).toContain('--jinja')
    expect(args[args.indexOf('--reasoning-format') + 1]).toBe('deepseek')
    expect(args[args.indexOf('--ctx-size') + 1]).toBe('8192')
    expect(args[args.indexOf('--n-gpu-layers') + 1]).toBe('all')
    expect(args.join(' ')).not.toMatch(/api-key/)
  })

  it('keeps every expert on the CPU for a mixture of experts unless told how many layers', () => {
    expect(serverArgs({ ...base, mixtureOfExperts: true })).toContain('--cpu-moe')
    const some = serverArgs({ ...base, mixtureOfExperts: true, expertsOnCpu: 16 })
    expect(some).not.toContain('--cpu-moe')
    expect(some[some.indexOf('--n-cpu-moe') + 1]).toBe('16')
    const none = serverArgs({ ...base, mixtureOfExperts: true, expertsOnCpu: 0 })
    expect(none).not.toContain('--cpu-moe')
    expect(none).not.toContain('--n-cpu-moe')
    // A dense model has no experts to place.
    expect(serverArgs({ ...base, expertsOnCpu: 8 }).join(' ')).not.toMatch(/moe/)
  })

  it('passes 8-bit memory, threads and a layer count through', () => {
    const args = serverArgs({ ...base, kvCache: 'q8_0', threads: 8, gpuLayers: 20 })
    expect(args[args.indexOf('--cache-type-k') + 1]).toBe('q8_0')
    expect(args[args.indexOf('--cache-type-v') + 1]).toBe('q8_0')
    expect(args[args.indexOf('--threads') + 1]).toBe('8')
    expect(args[args.indexOf('--n-gpu-layers') + 1]).toBe('20')
  })

  it('gives the server none of the variables it would read as settings, and the one key it needs', () => {
    const env = serverEnv(
      { PATH: 'C:/bin', LLAMA_ARG_HOST: '0.0.0.0', LLAMA_ARG_HF_REPO: 'someone/model', HF_TOKEN: 'secret', llama_arg_port: '80' },
      'k3y'
    )
    expect(env.PATH).toBe('C:/bin')
    expect(env.LLAMA_API_KEY).toBe('k3y')
    expect(env.HF_HUB_OFFLINE).toBe('1')
    for (const key of ['LLAMA_ARG_HOST', 'LLAMA_ARG_HF_REPO', 'HF_TOKEN', 'llama_arg_port']) expect(env).not.toHaveProperty(key)
  })

  it('reads the devices llama-server lists without loading anything', () => {
    expect(
      parseDeviceList('Available devices:\r\n  CUDA0: NVIDIA GeForce RTX 4070 (12282 MiB, 11200 MiB free)\r\n')
    ).toEqual({ backend: 'cuda', devices: ['NVIDIA GeForce RTX 4070'] })
    expect(parseDeviceList('Available devices:\n  Vulkan0: AMD Radeon RX 7800 XT (16368 MiB, 16000 MiB free)\n  Vulkan1: Intel(R) Graphics (2048 MiB, 2000 MiB free)')).toEqual({
      backend: 'vulkan',
      devices: ['AMD Radeon RX 7800 XT', 'Intel(R) Graphics']
    })
    expect(parseDeviceList('Available devices:\n')).toEqual({ backend: null, devices: [] })
  })

  it('reads the backend, devices and layer placement from the server log', () => {
    const log = emptyServerLog()
    for (const line of [
      'ggml_cuda_init: found 1 CUDA devices:',
      '  Device 0: NVIDIA GeForce RTX 4070, compute capability 8.9, VMM: yes',
      'load_tensors: offloaded 49/49 layers to GPU',
      'print_info: file size = 16.51 GiB'
    ]) {
      readServerLine(log, line)
    }
    expect(log).toEqual({ backend: 'cuda', devices: ['NVIDIA GeForce RTX 4070'], gpuLayers: 49, totalLayers: 49 })
  })

  it('keeps only sane server preferences from the config file', () => {
    expect(parseServer(undefined)).toEqual(DEFAULT_CONFIG.model.server)
    expect(parseServer({ path: 'C:/llama/llama-server.exe', expertsOnCpu: 16, threads: 8 })).toEqual({
      path: 'C:/llama/llama-server.exe',
      expertsOnCpu: 16,
      threads: 8
    })
    expect(parseServer({ path: '  ', expertsOnCpu: -1, threads: 0 })).toEqual({ path: null, expertsOnCpu: null, threads: null })
    expect(parseServer({ expertsOnCpu: 2.5, threads: 'many' })).toEqual({ path: null, expertsOnCpu: null, threads: null })
  })
})

describe('the server-sent event stream', () => {
  it('yields each event once, however the bytes were cut', () => {
    const parser = new SseParser()
    const got = [
      ...parser.push('data: {"a":1}\n\nda'),
      ...parser.push('ta: {"b":2}\r\n\r\ndata: [DONE]\n\n'),
      ...parser.flush()
    ]
    expect(got).toEqual(['{"a":1}', '{"b":2}', '[DONE]'])
  })

  it('hands back a final event the stream ended without terminating', () => {
    const parser = new SseParser()
    expect(parser.push('data: {"x":1}')).toEqual([])
    expect(parser.flush()).toEqual(['{"x":1}'])
  })

  it('reads text, thought, tool calls, the finish and the timings out of a chunk', () => {
    expect(readChunk('{"choices":[{"delta":{"content":"Hi"}}]}')).toMatchObject({ content: 'Hi', reasoning: '', finish: null })
    expect(readChunk('{"choices":[{"delta":{"reasoning_content":"hmm"}}]}')).toMatchObject({ reasoning: 'hmm' })
    expect(
      readChunk('{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"read_file","arguments":"{\\"pa"}}]},"finish_reason":null}]}')
    ).toMatchObject({ toolCalls: [{ index: 0, id: 'c1', function: { name: 'read_file', arguments: '{"pa' } }] })
    expect(readChunk('{"choices":[{"delta":{},"finish_reason":"tool_calls"}],"timings":{"prompt_n":120,"predicted_n":30}}')).toMatchObject({
      finish: 'tool_calls',
      usage: { input: 120, output: 30 }
    })
    expect(readChunk('{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":7}}').usage).toEqual({ input: 5, output: 7 })
    expect(readChunk('not json')).toMatchObject({ content: '', toolCalls: [] })
  })
})

describe('the conversation in the two shapes', () => {
  it('turns the saved form into the server form, splitting a reply around its tool call', () => {
    const history: ChatHistoryItem[] = [
      { type: 'system', text: 'Be careful.' },
      { type: 'user', text: 'List the folder.' },
      {
        type: 'model',
        response: [
          { type: 'segment', segmentType: 'thought', text: 'I should list it.', ended: true },
          { type: 'functionCall', name: 'list_directory', params: { path: '.' }, result: { entries: [], truncated: false }, rawCall: { id: 'call_1' } as never },
          'It is empty.'
        ]
      }
    ]
    expect(toServerMessages(history)).toEqual([
      { role: 'system', content: 'Be careful.' },
      { role: 'user', content: 'List the folder.' },
      {
        role: 'assistant',
        content: null,
        reasoning_content: 'I should list it.',
        tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'list_directory', arguments: '{"path":"."}' } }]
      },
      { role: 'tool', tool_call_id: 'call_1', content: '{"entries":[],"truncated":false}' },
      { role: 'assistant', content: 'It is empty.' }
    ])
  })

  it('makes up an id for a call recorded by the built-in engine without one', () => {
    const [assistant, tool] = toServerMessages([
      { type: 'model', response: [{ type: 'functionCall', name: 'read_file', params: { path: 'a' }, result: 'x' }] }
    ]) as unknown as [{ tool_calls: { id: string }[] }, { tool_call_id: string }]
    expect(assistant.tool_calls[0]?.id).toMatch(/^call_/)
    expect(tool.tool_call_id).toBe(assistant.tool_calls[0]?.id)
  })

  it('gathers a streamed reply back into the saved form', () => {
    const reply = new ReplyAccumulator()
    reply.thought('Let me ')
    reply.thought('think.')
    reply.toolCall({ index: 0, id: 'c9', function: { name: 'read_' } })
    reply.toolCall({ index: 0, function: { name: 'file', arguments: '{"path":' } })
    reply.toolCall({ index: 0, function: { arguments: '"a.txt"}' } })
    const [call] = reply.takeCalls()
    expect(call).toEqual({ id: 'c9', type: 'function', function: { name: 'read_file', arguments: '{"path":"a.txt"}' } })
    expect(parseArguments(call as never)).toEqual({ path: 'a.txt' })
    reply.result(call as never, { path: 'a.txt' }, { text: 'hello', bytes: 5, truncated: false })
    reply.text('It says ')
    reply.text('hello.')
    expect(reply.takeCalls()).toEqual([])
    expect(reply.parts).toEqual([
      { type: 'segment', segmentType: 'thought', text: 'Let me think.', ended: true },
      { type: 'functionCall', name: 'read_file', params: { path: 'a.txt' }, result: { text: 'hello', bytes: 5, truncated: false }, rawCall: { id: 'c9' } },
      'It says hello.'
    ])
    expect(reply.replyText).toBe('It says hello.')
    expect(parseArguments({ id: 'x', type: 'function', function: { name: 'f', arguments: '{broken' } })).toEqual({ _raw: '{broken' })
  })

  it('asks for a streamed reply with the sampling, the tools and the thinking switch', () => {
    const sampling = { temperature: 0.6, topP: 0.95, topK: 20, minP: 0 }
    const on = completionBody({ messages: [], tools: ['read_file'], sampling, thinking: 'on', thinkingBudget: undefined })
    expect(on).toMatchObject({ stream: true, temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0, tool_choice: 'auto', parallel_tool_calls: false })
    expect((on.tools as object[]).length).toBe(1)
    expect(on).not.toHaveProperty('chat_template_kwargs')
    expect(on).not.toHaveProperty('reasoning_budget_tokens')
    const capped = completionBody({ messages: [], tools: [], sampling, thinking: 'on', thinkingBudget: 2048 })
    expect(capped).toMatchObject({ reasoning_budget_tokens: 2048 })
    expect(capped).not.toHaveProperty('tools')
    const off = completionBody({ messages: [], tools: [], sampling, thinking: 'off', thinkingBudget: 0 })
    expect(off).toMatchObject({ chat_template_kwargs: { enable_thinking: false } })
    expect(off).not.toHaveProperty('reasoning_budget_tokens')
  })

  it('describes the tools from the same schemas the broker checks', () => {
    const [def] = toolDefinitions(['edit_file']) as [{ type: string; function: { name: string; parameters: { required: string[] } } }]
    expect(def.type).toBe('function')
    expect(def.function.name).toBe('edit_file')
    expect(def.function.parameters.required).toEqual(['path', 'old_text', 'new_text'])
  })
})

describe('measuring through the server', () => {
  /** Answers like a tokenizer that charges one token per word, and a template that adds two tokens per message. */
  const client: MeasureClient = {
    tokenCount: async (text) => text.split(/\s+/).filter(Boolean).length,
    render: async (messages, tools) =>
      messages.map((m) => `<m> ${'content' in m ? m.content ?? '' : ''}`).join(' ') + (tools ? ` tools ${tools.length} ${tools.length}` : '')
  }

  it('charges each piece to its category and the rest to the template', async () => {
    const history: ChatHistoryItem[] = [
      { type: 'system', text: 'Be careful.' },
      { type: 'user', text: 'hello there' },
      { type: 'model', response: ['hi back'] }
    ]
    const snapshot = await measureRemote(client, history, [{ label: 'Base', text: 'Be careful.' }], [], 4096, ['read_file'])
    const by = (category: string): number => snapshot.pieces.filter((p) => p.category === category).reduce((s, p) => s + p.tokens, 0)
    expect(by('system')).toBe(2)
    expect(by('user')).toBe(2)
    expect(by('reply')).toBe(2)
    // The fake template adds "tools 1 1" (three words) for one tool; the tool's own share is what removing it saves.
    expect(by('tools')).toBe(3)
    expect(snapshot.renderedTokens).toBe(12)
    expect(by('format')).toBe(12 - 2 - 2 - 2 - 3)
    expect(snapshot.contextSize).toBe(4096)
  })
})
