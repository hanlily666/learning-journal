"""The chat service: sessions, append-only history, and replies run as server tasks.

Each reply runs as its own asyncio task, not inside the HTTP request, so it keeps
streaming and saving if the browser closes. The browser follows a reply through the
broker, and reloads the stored message when it reconnects.
"""

import asyncio
import logging
import time
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.claude import ClaudeClient, Delta, PartialReply, blocks_for_replay, text_of
from app.config import Settings
from app.models import ApiRequest, ChatSession, Message
from app.schemas import MessageOut, SendResult, SessionOut, SessionView

log = logging.getLogger(__name__)

NAME_LENGTH = 60
INTERRUPTED_ERROR = "The server stopped before this reply finished."


class SessionNotFound(Exception):
    pass


class ReplyInProgress(Exception):
    """A session runs one reply at a time."""


class NothingToRetry(Exception):
    """Only a failed or interrupted latest reply can be tried again."""


def name_from(text: str) -> str:
    """A session name from the first ~60 characters of the first message."""
    text = " ".join(text.split())
    if len(text) <= NAME_LENGTH:
        return text
    cut = text[:NAME_LENGTH]
    space = cut.rfind(" ")
    return (cut[:space] if space > NAME_LENGTH // 2 else cut) + "…"


def replayable(message: Message) -> bool:
    """Only complete messages go to the API. A refused reply is left out too: its
    partial output must be discarded, not treated as an answer."""
    if message.status != "complete" or not message.content:
        return False
    return not (message.role == "assistant" and message.stop_reason == "refusal")


def can_retry(last: Message | None) -> bool:
    """Only a failed or interrupted latest reply can be tried again."""
    return last is not None and last.role == "assistant" and last.status in ("failed", "interrupted")


class Broker:
    """In-memory fan-out of live reply events to any browser that is watching."""

    def __init__(self) -> None:
        self._partials: dict[int, PartialReply] = {}
        self._queues: dict[int, list[asyncio.Queue]] = {}

    def start(self, message_id: int) -> None:
        self._partials[message_id] = PartialReply()
        self._queues.setdefault(message_id, [])

    def publish(self, message_id: int, partial: PartialReply, delta: Delta) -> None:
        self._partials[message_id] = partial
        for q in self._queues.get(message_id, []):
            q.put_nowait({"type": "delta", "kind": delta.kind, "text": delta.text})

    def finish(self, message_id: int, message: MessageOut | None) -> None:
        self._partials.pop(message_id, None)
        for q in self._queues.pop(message_id, []):
            q.put_nowait({"type": "done", "message": message.model_dump(mode="json") if message else None})

    def subscribe(self, message_id: int) -> tuple[list[dict[str, Any]], asyncio.Queue] | None:
        """The reply's content so far plus a queue of what follows, or None if it isn't running."""
        if message_id not in self._partials:
            return None
        q: asyncio.Queue = asyncio.Queue()
        self._queues[message_id].append(q)
        return [dict(b) for b in self._partials[message_id].blocks], q

    def unsubscribe(self, message_id: int, q: asyncio.Queue) -> None:
        queues = self._queues.get(message_id)
        if queues and q in queues:
            queues.remove(q)


class ChatService:
    def __init__(self, sessionmaker: async_sessionmaker, claude: ClaudeClient, settings: Settings,
                 broker: Broker | None = None):
        self.sessionmaker = sessionmaker
        self.claude = claude
        self.settings = settings
        self.broker = broker or Broker()
        self._tasks: set[asyncio.Task] = set()

    # Sessions

    async def create_session(self, name: str | None = None) -> SessionOut:
        async with self.sessionmaker() as db:
            session = ChatSession(name=name or None, source="app")
            db.add(session)
            await db.commit()
            return SessionOut.model_validate(session)

    async def list_sessions(self) -> list[SessionOut]:
        async with self.sessionmaker() as db:
            rows = await db.scalars(select(ChatSession).order_by(ChatSession.updated_at.desc(),
                                                                 ChatSession.id.desc()))
            return [SessionOut.model_validate(s) for s in rows]

    async def rename_session(self, session_id: int, name: str) -> SessionOut:
        async with self.sessionmaker() as db:
            session = await db.get(ChatSession, session_id)
            if session is None:
                raise SessionNotFound(session_id)
            session.name = name.strip()
            await db.commit()
            return SessionOut.model_validate(session)

    async def get_session(self, session_id: int) -> SessionView:
        async with self.sessionmaker() as db:
            session = await db.get(ChatSession, session_id)
            if session is None:
                raise SessionNotFound(session_id)
            messages = list(await db.scalars(
                select(Message).where(Message.session_id == session_id).order_by(Message.id)))
            latest_call = await db.scalar(
                select(ApiRequest)
                .where(ApiRequest.session_id == session_id, ApiRequest.step == "chat",
                       ApiRequest.error.is_(None))
                .order_by(ApiRequest.id.desc()).limit(1))
        context_tokens = 0
        if latest_call is not None:
            context_tokens = sum(t or 0 for t in (
                latest_call.input_tokens, latest_call.cache_read_tokens,
                latest_call.cache_write_tokens, latest_call.output_tokens))
        last = messages[-1] if messages else None
        return SessionView(
            session=SessionOut.model_validate(session),
            messages=[MessageOut.model_validate(m) for m in messages],
            context_tokens=context_tokens,
            context_warning=context_tokens >= self.settings.context_warning_tokens,
            can_retry=can_retry(last),
        )

    async def get_message(self, message_id: int) -> MessageOut | None:
        async with self.sessionmaker() as db:
            message = await db.get(Message, message_id)
            return MessageOut.model_validate(message) if message else None

    # Messages and replies

    async def send_message(self, session_id: int, text: str) -> SendResult:
        """Save my message and an empty reply in one transaction, then start the reply."""
        async with self.sessionmaker() as db, db.begin():
            session = await self._lock_idle_session(db, session_id)
            if session.name is None:
                session.name = name_from(text)
            session.updated_at = func.now()
            user = Message(session_id=session_id, role="user", content=[{"type": "text", "text": text}],
                           text=text, status="complete", source=session.source)
            db.add(user)
            await db.flush()
            assistant = self._new_reply(session)
            db.add(assistant)
        self._start(session_id, assistant.id)
        return SendResult(user_message=MessageOut.model_validate(user),
                          assistant_message=MessageOut.model_validate(assistant))

    async def retry(self, session_id: int) -> MessageOut:
        """Try a failed or interrupted reply again. The failed one stays in the history."""
        async with self.sessionmaker() as db, db.begin():
            session = await self._lock_idle_session(db, session_id)
            last = await db.scalar(select(Message).where(Message.session_id == session_id)
                                   .order_by(Message.id.desc()).limit(1))
            if not can_retry(last):
                raise NothingToRetry(session_id)
            session.updated_at = func.now()
            assistant = self._new_reply(session)
            db.add(assistant)
        self._start(session_id, assistant.id)
        return MessageOut.model_validate(assistant)

    async def recover_interrupted(self) -> int:
        """On startup: replies left 'streaming' by a stopped server are marked interrupted."""
        async with self.sessionmaker() as db, db.begin():
            result = await db.execute(update(Message).where(Message.status == "streaming")
                                      .values(status="interrupted", error=INTERRUPTED_ERROR))
            return result.rowcount

    async def wait_idle(self) -> None:
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def abandon(self) -> None:
        """Stop running replies without saving anything, as a crash would."""
        for task in list(self._tasks):
            task.cancel()
        await self.wait_idle()

    # Internals

    async def _lock_idle_session(self, db, session_id: int) -> ChatSession:
        session = await db.get(ChatSession, session_id, with_for_update=True)
        if session is None:
            raise SessionNotFound(session_id)
        running = await db.scalar(select(Message.id).where(
            Message.session_id == session_id, Message.status == "streaming"))
        if running is not None:
            raise ReplyInProgress(session_id)
        return session

    @staticmethod
    def _new_reply(session: ChatSession) -> Message:
        return Message(session_id=session.id, role="assistant", content=[], text="",
                       status="streaming", source=session.source)

    def _start(self, session_id: int, message_id: int) -> None:
        self.broker.start(message_id)
        task = asyncio.create_task(self._run_reply(session_id, message_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _history(self, session_id: int) -> list[dict[str, Any]]:
        async with self.sessionmaker() as db:
            messages = await db.scalars(select(Message).where(Message.session_id == session_id)
                                        .order_by(Message.id))
            return [
                {"role": m.role,
                 "content": blocks_for_replay(m.content) if m.role == "assistant" else m.content}
                for m in messages if replayable(m)
            ]

    async def _save_partial(self, message_id: int, blocks: list[dict[str, Any]]) -> None:
        async with self.sessionmaker() as db, db.begin():
            await db.execute(update(Message)
                             .where(Message.id == message_id, Message.status == "streaming")
                             .values(content=blocks, text=text_of(blocks)))

    async def _finish(self, message_id: int, **values: Any) -> MessageOut | None:
        async with self.sessionmaker() as db, db.begin():
            await db.execute(update(Message)
                             .where(Message.id == message_id, Message.status == "streaming")
                             .values(**values))
            message = await db.get(Message, message_id)
            return MessageOut.model_validate(message) if message else None

    async def _run_reply(self, session_id: int, message_id: int) -> None:
        finished: MessageOut | None = None
        try:
            history = await self._history(session_id)
            last_save = time.monotonic()

            async def on_delta(delta: Delta, partial: PartialReply) -> None:
                nonlocal last_save
                self.broker.publish(message_id, partial, delta)
                if time.monotonic() - last_save >= self.settings.partial_save_interval:
                    last_save = time.monotonic()
                    await self._save_partial(message_id, partial.blocks)

            result = await self.claude.stream_reply(step="chat", session_id=session_id,
                                                    message_id=message_id, messages=history,
                                                    on_delta=on_delta)
            finished = await self._finish(
                message_id,
                content=result.content,
                text=text_of(result.content),
                status="complete" if result.ok else "failed",
                stop_reason=result.stop_reason,
                error=result.error,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # a bug or a database problem: the reply must not stay "streaming"
            log.exception("reply %s failed", message_id)
            try:
                finished = await self._finish(message_id, status="failed", error=f"Server error: {e}")
            except Exception:
                log.exception("could not mark reply %s as failed", message_id)
        finally:
            self.broker.finish(message_id, finished)
