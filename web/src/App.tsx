import { useCallback, useEffect, useRef, useState } from 'react'
import {
  api,
  ApiError,
  followReply,
  type ContentBlock,
  type Message,
  type Session,
  type SessionView,
} from './api/client'

function appendDelta(content: ContentBlock[], kind: 'text' | 'thinking', text: string): ContentBlock[] {
  const last = content[content.length - 1]
  if (last && last.type === kind) {
    return [...content.slice(0, -1), { ...last, [kind]: String(last[kind] ?? '') + text }]
  }
  return [...content, { type: kind, [kind]: text }]
}

function sessionLabel(session: Session): string {
  return session.name || 'New session'
}

export default function App() {
  const [sessions, setSessions] = useState<Session[]>([])
  const [currentId, setCurrentId] = useState<number | null>(null)
  const [view, setView] = useState<SessionView | null>(null)
  const [error, setError] = useState<string | null>(null)
  const stopFollowing = useRef<(() => void) | null>(null)

  const refreshSessions = useCallback(async () => {
    setSessions(await api.listSessions())
  }, [])

  const loadSession = useCallback(async (id: number) => {
    const loaded = await api.getSession(id)
    setView(loaded)
    return loaded
  }, [])

  // Follow a streaming reply, patching it into the view as text arrives.
  const follow = useCallback(
    (sessionId: number, messageId: number) => {
      stopFollowing.current?.()
      const patch = (update: (m: Message) => Message) =>
        setView((v) =>
          v && v.session.id === sessionId
            ? { ...v, messages: v.messages.map((m) => (m.id === messageId ? update(m) : m)) }
            : v,
        )
      stopFollowing.current = followReply(messageId, {
        onSnapshot: (content) => patch((m) => ({ ...m, content })),
        onDelta: (kind, text) => patch((m) => ({ ...m, content: appendDelta(m.content, kind, text) })),
        onDone: () => {
          stopFollowing.current = null
          void loadSession(sessionId).then(refreshSessions)
        },
      })
    },
    [loadSession, refreshSessions],
  )

  useEffect(() => {
    api
      .listSessions()
      .then(setSessions)
      .catch((e) => setError(String(e)))
    return () => stopFollowing.current?.()
  }, [])

  // Open a session; if its last reply is still being written, follow it.
  const open = async (id: number) => {
    stopFollowing.current?.()
    stopFollowing.current = null
    setCurrentId(id)
    const loaded = await loadSession(id)
    const last = loaded.messages[loaded.messages.length - 1]
    if (last?.status === 'streaming') follow(id, last.id)
  }

  const run = async (action: () => Promise<void>) => {
    setError(null)
    try {
      await action()
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e))
    }
  }

  const newSession = () =>
    run(async () => {
      const session = await api.createSession()
      await refreshSessions()
      await open(session.id)
    })

  const send = (text: string) =>
    run(async () => {
      if (currentId === null) return
      const sent = await api.sendMessage(currentId, text)
      setView((v) =>
        v ? { ...v, can_retry: false, messages: [...v.messages, sent.user_message, sent.assistant_message] } : v,
      )
      follow(currentId, sent.assistant_message.id)
      void refreshSessions()
    })

  const retry = () =>
    run(async () => {
      if (currentId === null) return
      const reply = await api.retry(currentId)
      setView((v) => (v ? { ...v, can_retry: false, messages: [...v.messages, reply] } : v))
      follow(currentId, reply.id)
    })

  const rename = (name: string) =>
    run(async () => {
      if (currentId === null) return
      await api.renameSession(currentId, name)
      await Promise.all([refreshSessions(), loadSession(currentId)])
    })

  const replying = view?.messages.some((m) => m.status === 'streaming') ?? false

  return (
    <div className="app">
      <aside className="sidebar">
        <button className="new-session" onClick={newSession}>
          + New session
        </button>
        <ul>
          {sessions.map((s) => (
            <li key={s.id}>
              <button className={s.id === currentId ? 'active' : ''} onClick={() => run(() => open(s.id))}>
                {sessionLabel(s)}
              </button>
            </li>
          ))}
        </ul>
      </aside>
      <main className="chat">
        {error && (
          <div className="banner error" role="alert">
            {error}
          </div>
        )}
        {view ? (
          <>
            <SessionHeader session={view.session} onRename={rename} />
            {view.context_warning && (
              <div className="banner warning">
                This conversation is about {Math.round(view.context_tokens / 1000)}K tokens long. Start a new
                session to keep replies fast and cheap.
              </div>
            )}
            <MessageList messages={view.messages} />
            {view.can_retry && !replying && (
              <div className="retry">
                <button onClick={retry}>Retry the last reply</button>
              </div>
            )}
            <Composer disabled={replying} onSend={send} />
          </>
        ) : (
          <div className="empty">Pick a session, or start a new one.</div>
        )}
      </main>
    </div>
  )
}

function SessionHeader({ session, onRename }: { session: Session; onRename: (name: string) => void }) {
  const [editing, setEditing] = useState(false)
  const [name, setName] = useState('')

  if (editing) {
    return (
      <form
        className="session-header"
        onSubmit={(e) => {
          e.preventDefault()
          if (name.trim()) onRename(name.trim())
          setEditing(false)
        }}
      >
        <input autoFocus value={name} onChange={(e) => setName(e.target.value)} aria-label="Session name" />
        <button type="submit">Save</button>
        <button type="button" onClick={() => setEditing(false)}>
          Cancel
        </button>
      </form>
    )
  }
  return (
    <div className="session-header">
      <h1>{sessionLabel(session)}</h1>
      <button
        onClick={() => {
          setName(session.name ?? '')
          setEditing(true)
        }}
      >
        Rename
      </button>
    </div>
  )
}

function MessageList({ messages }: { messages: Message[] }) {
  const end = useRef<HTMLDivElement>(null)
  useEffect(() => {
    end.current?.scrollIntoView({ block: 'end' })
  }, [messages])
  return (
    <div className="messages">
      {messages.map((m) => (
        <MessageView key={m.id} message={m} />
      ))}
      <div ref={end} />
    </div>
  )
}

function statusLabel(m: Message): string | null {
  if (m.status === 'failed') return `Failed${m.error ? `: ${m.error}` : ''}`
  if (m.status === 'interrupted') return 'Interrupted: the server stopped before this reply finished'
  if (m.stop_reason === 'max_tokens') return 'Cut off: the reply reached its length limit'
  if (m.stop_reason === 'refusal') return 'Refused: Claude declined to answer this'
  return null
}

function MessageView({ message }: { message: Message }) {
  const thinking = message.content
    .filter((b) => b.type === 'thinking')
    .map((b) => String(b.thinking ?? ''))
    .join('\n\n')
  const text = message.content
    .filter((b) => b.type === 'text')
    .map((b) => String(b.text ?? ''))
    .join('')
  const label = statusLabel(message)
  return (
    <div className={`message ${message.role}`}>
      {thinking && (
        <details className="thinking">
          <summary>Thinking</summary>
          <p>{thinking}</p>
        </details>
      )}
      {text && <div className="text">{text}</div>}
      {message.status === 'streaming' && <div className="typing">Writing…</div>}
      {label && <div className={`label ${message.status}`}>{label}</div>}
      <time dateTime={message.created_at}>{new Date(message.created_at).toLocaleString()}</time>
    </div>
  )
}

function Composer({ disabled, onSend }: { disabled: boolean; onSend: (text: string) => void }) {
  const [text, setText] = useState('')
  const submit = () => {
    if (!text.trim() || disabled) return
    onSend(text)
    setText('')
  }
  return (
    <form
      className="composer"
      onSubmit={(e) => {
        e.preventDefault()
        submit()
      }}
    >
      <textarea
        value={text}
        placeholder="Message Claude (Enter to send, Shift+Enter for a new line)"
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault()
            submit()
          }
        }}
      />
      <button type="submit" disabled={disabled || !text.trim()}>
        Send
      </button>
    </form>
  )
}
