import { useCallback, useEffect, useState } from 'react'
import {
  api,
  ApiError,
  type ExtractedSession,
  type Item,
  type Segment,
} from './api/client'

const FIELDS: { field: Item['field']; title: string }[] = [
  { field: 'concept', title: 'Key concepts to review later' },
  { field: 'gap', title: 'Knowledge gaps I had while asking' },
  { field: 'explore', title: 'Things worth exploring on my own' },
  { field: 'source', title: 'External sources mentioned in the chat' },
]

const STATUS: Record<Segment['status'], string> = {
  pending: 'Waiting',
  running: 'Processing…',
  done: 'Extracted',
  held: 'French: held for later',
  skipped: 'Skipped',
  needs_attention: 'Needs attention',
}

const LABEL: Record<NonNullable<Segment['label']>, string> = {
  ai: 'AI',
  french: 'French',
  messy: 'Messy',
}

/** Extracted items by session: each segment's label, summary, items and reasons, and failures. */
export default function ExtractedPage({
  onOpenSession,
}: {
  onOpenSession: (id: number) => void
}) {
  const [sessions, setSessions] = useState<ExtractedSession[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  const load = useCallback(async () => setSessions(await api.extracted()), [])

  const run = async (action: () => Promise<void>) => {
    setError(null)
    try {
      await action()
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
    }
  }

  useEffect(() => {
    api
      .extracted()
      .then(setSessions)
      .catch((e) => setError(String(e)))
  }, [])

  // Extraction runs on the server; poll while anything is still waiting or running.
  const busy = sessions?.some((v) =>
    v.segments.some((s) => s.status === 'pending' || s.status === 'running'),
  )
  useEffect(() => {
    if (!busy) return
    const timer = setInterval(() => void load().catch(() => {}), 2000)
    return () => clearInterval(timer)
  }, [busy, load])

  const processNow = () =>
    run(async () => {
      const { queued } = await api.processNow()
      setNotice(
        queued
          ? `Queued ${queued} session${queued > 1 ? 's' : ''}.`
          : 'Nothing new to process.',
      )
      await load()
    })

  const retry = (segment: Segment) =>
    run(async () => {
      await api.retrySegment(segment.id)
      await load()
    })

  return (
    <div className="extracted">
      <div className="session-header">
        <h1>Extracted</h1>
        <button onClick={processNow}>Process now</button>
      </div>
      {error && (
        <div className="banner error" role="alert">
          {error}
        </div>
      )}
      {notice && <div className="banner">{notice}</div>}
      {sessions?.length === 0 && (
        <div className="empty">
          Nothing extracted yet. Chat, then start a new session or press Process
          now.
        </div>
      )}
      {sessions?.map((view) => (
        <section key={view.session.id} className="extracted-session">
          <h2>
            <button
              className="link"
              onClick={() => onOpenSession(view.session.id)}
            >
              {view.session.name || 'New session'}
            </button>
          </h2>
          {view.segments.map((s) => (
            <SegmentView key={s.id} segment={s} onRetry={() => retry(s)} />
          ))}
        </section>
      ))}
    </div>
  )
}

function SegmentView({
  segment,
  onRetry,
}: {
  segment: Segment
  onRetry: () => void
}) {
  return (
    <div className={`segment ${segment.status}`}>
      <div className="segment-meta">
        {segment.label && (
          <span className={`chip ${segment.label}`}>
            {LABEL[segment.label]}
          </span>
        )}
        <span>{STATUS[segment.status]}</span>
        <span className="muted">
          messages {segment.first_message_id}–{segment.last_message_id}
        </span>
      </div>
      {segment.subject && <h3>{segment.subject}</h3>}
      {segment.summary && <p>{segment.summary}</p>}
      {segment.status !== 'done' && segment.label_reason && (
        <p className="muted">{segment.label_reason}</p>
      )}
      {segment.error && <p className="label failed">{segment.error}</p>}
      {segment.status === 'needs_attention' && (
        <button onClick={onRetry}>Retry</button>
      )}
      {FIELDS.map(({ field, title }) => {
        const items = segment.items.filter((i) => i.field === field)
        if (!items.length) return null
        return (
          <div key={field} className="field">
            <h4>{title}</h4>
            <ul>
              {items.map((i) => (
                <li key={i.id}>
                  {i.text}
                  <div className="muted">Why: {i.reason}</div>
                </li>
              ))}
            </ul>
          </div>
        )
      })}
    </div>
  )
}
