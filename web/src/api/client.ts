// Calls to the server. Types come from src/api/schema.d.ts (npm run gen:api).
import type { components } from './schema'

export type Session = components['schemas']['SessionOut']
export type Message = components['schemas']['MessageOut']
export type SessionView = components['schemas']['SessionView']
export type SendResult = components['schemas']['SendResult']
export type ContentBlock = Message['content'][number]
export type ExtractedSession = components['schemas']['ExtractedSession']
export type Segment = components['schemas']['SegmentOut']
export type Item = components['schemas']['ItemOut']
export type TriggerResult = components['schemas']['TriggerResult']

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const response = await fetch(path, {
    method,
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!response.ok) {
    let detail = response.statusText
    try {
      const data = await response.json()
      if (typeof data.detail === 'string') detail = data.detail
    } catch {
      // not JSON
    }
    throw new ApiError(response.status, detail)
  }
  return response.json() as Promise<T>
}

export const api = {
  listSessions: () => request<Session[]>('GET', '/api/sessions'),
  createSession: () => request<Session>('POST', '/api/sessions', {}),
  getSession: (id: number) => request<SessionView>('GET', `/api/sessions/${id}`),
  renameSession: (id: number, name: string) =>
    request<Session>('PATCH', `/api/sessions/${id}`, { name }),
  sendMessage: (id: number, text: string) =>
    request<SendResult>('POST', `/api/sessions/${id}/messages`, { text }),
  retry: (id: number) => request<Message>('POST', `/api/sessions/${id}/retry`),
  extracted: () => request<ExtractedSession[]>('GET', '/api/extracted'),
  processNow: () => request<TriggerResult>('POST', '/api/extraction/run'),
  retrySegment: (id: number) => request<Segment>('POST', `/api/segments/${id}/retry`),
}

export type ReplyEvents = {
  onSnapshot: (content: ContentBlock[]) => void
  onDelta: (kind: 'text' | 'thinking', text: string) => void
  onDone: (message: Message | null) => void
}

/** Follow one reply as it streams. Returns a function that stops listening. */
export function followReply(messageId: number, events: ReplyEvents): () => void {
  const source = new EventSource(`/api/messages/${messageId}/events`)
  source.addEventListener('snapshot', (e) => events.onSnapshot(JSON.parse(e.data)))
  source.addEventListener('delta', (e) => {
    const data = JSON.parse(e.data)
    events.onDelta(data.kind, data.text)
  })
  source.addEventListener('done', (e) => {
    source.close()
    events.onDone(JSON.parse(e.data))
  })
  source.addEventListener('error', () => {
    // The reply keeps running on the server; reload what it has saved.
    source.close()
    events.onDone(null)
  })
  return () => source.close()
}
