import { useEffect, useMemo, useState } from 'react'
import { composeSystemPrompt, estimateTokens, type GlobalRule, type WorkingContext } from '@shared/config.js'
import type { ModelInfo } from '@shared/engine-protocol.js'

/**
 * Standing instructions: one editor for the global rules, one for the open
 * workspace's project rules, and the prompt the two add up to.
 *
 * The token meter is the part that earns its place. Against a hosted model a
 * long preamble is invisible; here it is subtracted directly from the room the
 * model has to think in, and a user who writes six paragraphs of rules against
 * a 4096-token context deserves to be told what they just did rather than to
 * wonder why the assistant got worse.
 */

const WARN_FRACTION = 0.15

function newRule(): GlobalRule {
  return {
    id: `rule-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    title: 'New rule',
    body: '',
    enabled: true
  }
}

export function RulesPanel({
  heading,
  help,
  rules,
  model,
  onSave
}: {
  heading: string
  help: string
  rules: GlobalRule[]
  model: ModelInfo | null
  onSave: (rules: GlobalRule[]) => void
}): React.JSX.Element {
  const [draft, setDraft] = useState<GlobalRule[]>(rules)
  const [dirty, setDirty] = useState(false)

  // Rules changed elsewhere (or loaded at startup) should show up here, but not
  // over the top of edits in progress.
  useEffect(() => {
    if (!dirty) setDraft(rules)
  }, [rules, dirty])

  // The meter counts these rules alone; the preview below shows the whole prompt.
  const tokens = useMemo(
    () => draft.filter((r) => r.enabled).reduce((sum, r) => sum + estimateTokens(r.body), 0),
    [draft]
  )
  const contextSize = model?.contextSize ?? null
  const share = contextSize ? tokens / contextSize : null
  const hot = share !== null && share > WARN_FRACTION

  const mutate = (next: GlobalRule[]): void => {
    setDraft(next)
    setDirty(true)
  }

  return (
      <section>
        <h3>{heading}</h3>
        <p className="help">{help}</p>

        {draft.map((rule, index) => (
          <div className="rule-editor" key={rule.id}>
            <div className="head">
              <input
                type="checkbox"
                checked={rule.enabled}
                aria-label={`Enable ${rule.title}`}
                onChange={(e) =>
                  mutate(
                    draft.map((r, i) => (i === index ? { ...r, enabled: e.target.checked } : r))
                  )
                }
              />
              <input
                type="text"
                value={rule.title}
                aria-label="Rule title"
                onChange={(e) =>
                  mutate(draft.map((r, i) => (i === index ? { ...r, title: e.target.value } : r)))
                }
              />
              <button className="danger" onClick={() => mutate(draft.filter((_, i) => i !== index))}>
                Remove
              </button>
            </div>
            <textarea
              rows={3}
              value={rule.body}
              aria-label="Rule text"
              placeholder="Write the instruction as you would say it to a person."
              onChange={(e) =>
                mutate(draft.map((r, i) => (i === index ? { ...r, body: e.target.value } : r)))
              }
            />
            <div className="token-meter">~{estimateTokens(rule.body)} tokens</div>
          </div>
        ))}

        <div className="row">
          <button onClick={() => mutate([...draft, newRule()])}>Add a rule</button>
          <div className="grow" />
          <div className={`token-meter ${hot ? 'hot' : ''}`}>
            ~{tokens} tokens
            {contextSize ? ` · ${Math.round((share as number) * 100)}% of the ${contextSize} context` : ''}
          </div>
        </div>
        {hot ? (
          <p className="help" style={{ color: 'var(--warn)' }}>
            These rules take up a sizeable share of the context window before the conversation even
            starts. Shorter rules usually work better on a local model than thorough ones.
          </p>
        ) : null}

        <div className="row">
          <button
            className="primary"
            disabled={!dirty}
            onClick={() => {
              onSave(draft)
              setDirty(false)
            }}
          >
            Save
          </button>
          <button
            disabled={!dirty}
            onClick={() => {
              setDraft(rules)
              setDirty(false)
            }}
          >
            Discard changes
          </button>
        </div>
      </section>
  )
}

/** The exact system prompt, from the saved rules, for the open workspace. */
export function PromptPreview({
  rules,
  context
}: {
  rules: GlobalRule[]
  context: WorkingContext | undefined
}): React.JSX.Element {
  const prompt = useMemo(() => composeSystemPrompt(rules, context), [rules, context])
  return (
    <section>
      <h3>What the model will actually see</h3>
      <p className="help">
        The exact text placed before every conversation{context ? ' in this folder' : ''}, from the
        saved rules. Saved changes apply from the next message, in the chat that is open as well.
      </p>
      <pre className="diff" style={{ padding: 12, border: '1px solid var(--line)' }}>
        {prompt}
      </pre>
    </section>
  )
}
