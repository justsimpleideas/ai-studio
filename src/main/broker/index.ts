import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import { randomUUID } from 'node:crypto'
import {
  humanizeReason,
  type PermissionSettings,
  type PolicyRule
} from '@shared/permissions.js'
import { TOOL_RISK, type FileToolCall } from '@shared/tools.js'
import type {
  ApprovalRequest,
  ApprovalResponse,
  AuditEntry,
  Grant,
  Preview,
  ToolOutcome
} from '@shared/broker-types.js'
import {
  deleteFile,
  editFile,
  listDirectory,
  readFile,
  writeFile,
  Journal,
  ToolError
} from '../tools/fs-tools.js'
import { anchorToFolder, canonicalize, comparablePath } from './paths.js'
import { evaluate, protectionFor } from './policy.js'
import { LeakDetector } from '../net/guard.js'
import type { NewFileCheck } from '../net/gate.js'

/**
 * The broker: the one road between a model that wants something and a disk that
 * can give it.
 *
 * Every tool call is resolved, ruled on, possibly shown to the user, resolved a
 * second time, and only then carried out. The second resolution is not
 * redundant. Between the moment a person reads an approval prompt and the moment
 * they click it, the path they were shown can stop meaning what it meant, and
 * the whole value of asking them rests on those two being the same file.
 */

export interface BrokerDeps {
  getSettings: () => PermissionSettings
  saveRule: (rule: PolicyRule) => Promise<void>
  prompt: (request: ApprovalRequest) => Promise<ApprovalResponse>
  trash: (target: string) => Promise<void>
  audit: (entry: AuditEntry) => void
  journal: Journal
  /** The workspace's working folder, which relative paths are anchored to. Null: absolute paths only. */
  workingFolder?: string | null
}

const PREVIEW_LIMIT = 20_000

/**
 * How many changes one reply may make before every further one is put to the
 * user, whatever the modes say. A model that has started deleting files in a
 * loop is stopped at the fourth, not the four-hundredth.
 */
export const REPLY_LIMITS = { deletes: 3, writes: 20 }

/** Which reply a call belongs to; changes are grouped and counted by it. */
export interface CallContext {
  promptId?: string | undefined
}

interface Seen {
  mtimeMs: number
  size: number
}

export class Broker {
  /** Grants that live and die with this session. Never written to disk. */
  private sessionRules: PolicyRule[] = []
  /** Files the agent has read in this conversation, as they were when it read them. */
  private seen = new Map<string, Seen>()
  /** Changes made per reply, for REPLY_LIMITS. */
  private counts = new Map<string, { deletes: number; writes: number }>()
  /**
   * Everything the agent read from or wrote into files in this conversation,
   * so the internet gate can refuse a request that would carry any of it out.
   */
  readonly contents = new LeakDetector()

  constructor(private readonly deps: BrokerDeps) {}

  /**
   * Where a download may be saved: a path the agent could write to, that does
   * not exist yet. `ask` says whether writing there would be asked about.
   */
  async checkNewFile(rawPath: unknown): Promise<NewFileCheck> {
    const anchored = anchorToFolder(this.deps.workingFolder ?? null, rawPath)
    const verdict = evaluate(this.effectiveSettings(), 'write_file', anchored)
    if (verdict.decision === 'deny') return { ok: false, error: humanizeReason(verdict.reason) }
    const resolved = verdict.resolved as string
    if (fs.existsSync(resolved)) {
      return { ok: false, error: `${path.basename(resolved)} already exists. Downloads are only saved as new files.` }
    }
    return { ok: true, resolved, ask: verdict.decision === 'ask' }
  }

  /**
   * Save downloaded bytes as a new file at a path checkNewFile approved. The
   * path is resolved again first, and the file must still not exist; the write
   * is journalled like any other, so a download can be reverted.
   */
  async saveNewFile(resolved: string, bytes: Buffer, ctx: CallContext): Promise<ToolOutcome> {
    const call = { id: randomUUID(), name: 'write_file', args: { path: resolved, content: '' } } as FileToolCall
    const recheck = canonicalize(resolved)
    if (!recheck.ok || comparablePath(recheck.path) !== comparablePath(resolved) || recheck.existed) {
      return this.refuse(call, resolved, 'The destination changed while the download was running, so nothing was saved.', true)
    }
    const verdict = evaluate(this.effectiveSettings(), 'write_file', resolved)
    if (verdict.decision === 'deny') return this.refuse(call, resolved, humanizeReason(verdict.reason), true)
    const entry = await this.deps.journal.record('write', resolved, ctx.promptId)
    const temp = path.join(path.dirname(resolved), `.${path.basename(resolved)}.${randomUUID()}.tmp`)
    try {
      await fsp.mkdir(path.dirname(resolved), { recursive: true })
      await fsp.writeFile(temp, bytes)
      await fsp.rename(temp, resolved)
    } catch (err) {
      await fsp.rm(temp, { force: true })
      return { ok: false, tool: 'download_file', resolved, error: (err as Error).message, refused: false }
    } finally {
      await this.deps.journal.settle(entry)
    }
    this.countChange(call, ctx)
    this.deps.audit({
      at: new Date().toISOString(),
      tool: 'download_file',
      path: resolved,
      decision: 'allowed',
      asked: true,
      reason: `Saved a download (${bytes.length} bytes).`
    })
    return { ok: true, tool: 'download_file', resolved, result: { path: resolved, bytes: bytes.length }, asked: true }
  }

  clearSessionGrants(): void {
    this.sessionRules = []
  }

  listSessionGrants(): readonly PolicyRule[] {
    return this.sessionRules
  }

  /**
   * Forget what the agent has read. Called when the conversation changes: a
   * file read in another chat is not one this chat's model has seen.
   */
  resetConversation(): void {
    this.seen.clear()
    this.counts.clear()
    this.contents.clear()
  }

  private async remember(target: string): Promise<void> {
    const stat = await fsp.stat(target).catch(() => null)
    if (stat?.isFile()) this.seen.set(comparablePath(target), { mtimeMs: stat.mtimeMs, size: stat.size })
  }

  /**
   * An existing file may only be replaced or edited once the agent has read it
   * in this conversation, and only while it is unchanged since. This is what
   * stops a model from overwriting a file it guessed the contents of, or one
   * that changed after it looked.
   */
  private async unseenChange(call: FileToolCall, resolved: string): Promise<string | null> {
    if (call.name !== 'write_file' && call.name !== 'edit_file') return null
    const stat = await fsp.stat(resolved).catch(() => null)
    if (!stat?.isFile()) return null
    const seen = this.seen.get(comparablePath(resolved))
    if (!seen) {
      return `${path.basename(resolved)} already exists and has not been read in this conversation. Read it with read_file first, then change it.`
    }
    if (seen.mtimeMs !== stat.mtimeMs || seen.size !== stat.size) {
      return `${path.basename(resolved)} has changed since it was read. Read it again before changing it.`
    }
    return null
  }

  private overLimit(call: FileToolCall, ctx: CallContext): string | null {
    if (!ctx.promptId) return null
    const count = this.counts.get(ctx.promptId) ?? { deletes: 0, writes: 0 }
    if (call.name === 'delete_file' && count.deletes >= REPLY_LIMITS.deletes) {
      return `This reply has already deleted ${count.deletes} files, so every further delete is asked about.`
    }
    if ((call.name === 'write_file' || call.name === 'edit_file') && count.writes >= REPLY_LIMITS.writes) {
      return `This reply has already changed ${count.writes} files, so every further change is asked about.`
    }
    return null
  }

  private countChange(call: FileToolCall, ctx: CallContext): void {
    if (!ctx.promptId) return
    const count = this.counts.get(ctx.promptId) ?? { deletes: 0, writes: 0 }
    if (call.name === 'delete_file') count.deletes += 1
    if (call.name === 'write_file' || call.name === 'edit_file') count.writes += 1
    this.counts.set(ctx.promptId, count)
  }

  private effectiveSettings(): PermissionSettings {
    const base = this.deps.getSettings()
    return { ...base, rules: [...base.rules, ...this.sessionRules] }
  }

  async execute(requested: FileToolCall, ctx: CallContext = {}): Promise<ToolOutcome> {
    // Anchored once, up front, so the approval prompt, the policy check, the
    // recheck and the action all see the same absolute path.
    const call = {
      ...requested,
      args: { ...requested.args, path: anchorToFolder(this.deps.workingFolder ?? null, requested.args.path) }
    } as FileToolCall
    const risk = TOOL_RISK[call.name]
    const verdict = evaluate(this.effectiveSettings(), call.name, call.args.path)

    if (verdict.decision === 'deny') {
      return this.refuse(call, verdict.resolved, humanizeReason(verdict.reason), false)
    }

    // Refused before anyone is asked: there is nothing for the user to approve
    // until the agent has looked at the file.
    const unseen = await this.unseenChange(call, verdict.resolved as string)
    if (unseen) return this.refuse(call, verdict.resolved, unseen, false)

    const limit = this.overLimit(call, ctx)
    let asked = false
    if (verdict.decision === 'ask' || limit) {
      asked = true
      const resolved = verdict.resolved as string
      const request: ApprovalRequest = {
        id: randomUUID(),
        call,
        risk,
        resolved,
        because: limit ?? humanizeReason(verdict.reason),
        preview: await this.buildPreview(call, resolved)
      }

      const response = await this.deps.prompt(request)
      if (!response.allow) {
        return this.refuse(call, resolved, 'You declined this action.', true)
      }
      if (response.grant) {
        await this.applyGrant(response.grant)
      }
    }

    // Resolve again, now, and insist it is still the same file. A junction
    // swapped underneath us between the check and here would otherwise turn an
    // approval for one path into an action on another.
    const recheck = canonicalize(call.args.path)
    if (!recheck.ok || comparablePath(recheck.path) !== comparablePath(verdict.resolved as string)) {
      return this.refuse(
        call,
        verdict.resolved,
        'That path changed while it was waiting for a decision, so nothing was done.',
        asked
      )
    }

    return this.run(call, recheck.path, asked, ctx)
  }

  private async applyGrant(grant: Grant): Promise<void> {
    const rule: PolicyRule = {
      id: randomUUID(),
      pattern: grant.pattern,
      applies: grant.applies,
      decision: 'allow',
      note: grant.scope === 'session' ? 'granted for this session' : 'granted from an approval',
      createdAt: new Date().toISOString()
    }
    if (grant.scope === 'session') {
      this.sessionRules.push(rule)
    } else {
      await this.deps.saveRule(rule)
    }
  }

  private refuse(
    call: FileToolCall,
    resolved: string | null,
    message: string,
    asked: boolean
  ): ToolOutcome {
    this.deps.audit({
      at: new Date().toISOString(),
      tool: call.name,
      path: resolved,
      decision: 'refused',
      asked,
      reason: message
    })
    return { ok: false, tool: call.name, resolved, error: message, refused: true }
  }

  /**
   * Run the approved call against the resolved path.
   *
   * The canonical path is substituted into the arguments rather than the string
   * the model wrote, so what runs is exactly what was checked.
   */
  private async run(call: FileToolCall, resolved: string, asked: boolean, ctx: CallContext): Promise<ToolOutcome> {
    const group = ctx.promptId
    try {
      let result: unknown
      switch (call.name) {
        case 'list_directory': {
          const listing = await listDirectory({ path: resolved })
          // Hidden paths are left out of listings: the agent is not told they exist.
          const settings = this.effectiveSettings()
          result = {
            ...listing,
            entries: listing.entries.filter(
              (e) => protectionFor(settings, path.join(resolved, e.name))?.level !== 'hidden'
            )
          }
          break
        }
        case 'read_file':
          result = await readFile({ path: resolved })
          this.contents.add((result as { text: string }).text)
          await this.remember(resolved)
          break
        case 'write_file':
          result = await writeFile({ path: resolved, content: call.args.content }, this.deps.journal, group)
          this.contents.add(call.args.content)
          await this.remember(resolved)
          break
        case 'edit_file':
          result = await editFile(
            { path: resolved, old_text: call.args.old_text, new_text: call.args.new_text },
            this.deps.journal,
            group
          )
          this.contents.add(call.args.new_text)
          await this.remember(resolved)
          break
        case 'delete_file':
          result = await deleteFile({ path: resolved }, this.deps.journal, this.deps.trash, group)
          this.seen.delete(comparablePath(resolved))
          break
      }
      this.countChange(call, ctx)
      this.deps.audit({
        at: new Date().toISOString(),
        tool: call.name,
        path: resolved,
        decision: 'allowed',
        asked,
        reason: asked ? 'You approved it.' : 'Allowed without asking.'
      })
      return { ok: true, tool: call.name, resolved, result, asked }
    } catch (err) {
      const message = err instanceof ToolError ? err.message : (err as Error).message
      const hint = err instanceof ToolError ? err.hint : undefined
      this.deps.audit({
        at: new Date().toISOString(),
        tool: call.name,
        path: resolved,
        decision: 'failed',
        asked,
        reason: message
      })
      return {
        ok: false,
        tool: call.name,
        resolved,
        error: message,
        ...(hint ? { hint } : {}),
        refused: false
      }
    }
  }

  /**
   * Work out what the user is about to agree to.
   *
   * Approving a write without seeing the write is not consent, it is a habit.
   * The preview is built from the same arguments the call will use, so what the
   * dialog shows is what lands.
   */
  private async buildPreview(call: FileToolCall, resolved: string): Promise<Preview> {
    try {
      switch (call.name) {
        case 'write_file': {
          const before = await this.readIfPresent(resolved)
          return before === null
            ? { kind: 'create', after: clip(call.args.content) }
            : { kind: 'overwrite', before: clip(before), after: clip(call.args.content) }
        }
        case 'edit_file': {
          const before = await this.readIfPresent(resolved)
          if (before === null) return { kind: 'none' }
          const at = before.indexOf(call.args.old_text)
          if (at < 0) return { kind: 'none' }
          const after = before.replace(call.args.old_text, call.args.new_text)
          return { kind: 'edit', before: clip(before), after: clip(after), at }
        }
        case 'delete_file': {
          const stat = await fsp.stat(resolved).catch(() => null)
          return stat ? { kind: 'delete', bytes: stat.size } : { kind: 'none' }
        }
        default:
          return { kind: 'none' }
      }
    } catch {
      // A preview that cannot be built is not a reason to block the prompt; the
      // user simply decides with less in front of them.
      return { kind: 'none' }
    }
  }

  private async readIfPresent(target: string): Promise<string | null> {
    try {
      return await fsp.readFile(target, 'utf8')
    } catch {
      return null
    }
  }
}

function clip(text: string): string {
  return text.length > PREVIEW_LIMIT ? text.slice(0, PREVIEW_LIMIT) + '\n…' : text
}
