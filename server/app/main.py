"""HTTP API. Run with: uv run uvicorn app.main:app --reload"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette import EventSourceResponse

from app.chat import ChatService, NothingToRetry, ReplyInProgress, SessionNotFound
from app.claude import AnthropicTransport, ClaudeClient, Transport
from app.config import Settings, get_settings
from app.db import make_engine, make_sessionmaker
from app.extraction import ExtractionService, SegmentNotFound, SegmentNotRetryable
from app.schemas import (ExtractedSession, MessageCreate, MessageOut, SegmentOut, SendResult,
                         SessionCreate, SessionOut, SessionRename, SessionView, TriggerResult)

log = logging.getLogger(__name__)

WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"


def create_app(settings: Settings | None = None, transport: Transport | None = None,
               client: anthropic.AsyncAnthropic | None = None) -> FastAPI:
    """`transport` replaces Claude for chat and `client` for extraction (tests pass fakes)."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = make_engine(settings.database_url)
        sessionmaker = make_sessionmaker(engine)
        api_client = client or anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        claude = ClaudeClient(transport or AnthropicTransport(api_client), sessionmaker, settings)
        app.state.chat = ChatService(sessionmaker, claude, settings)
        app.state.extraction = ExtractionService(sessionmaker, api_client, settings)
        recovered = await app.state.chat.recover_interrupted()
        if recovered:
            log.warning("marked %d unfinished replies as interrupted", recovered)
        requeued = await app.state.extraction.recover()
        if requeued:
            log.warning("queued %d unfinished extraction segments again", requeued)
            app.state.extraction.kick()
        if not settings.anthropic_api_key and transport is None:
            log.warning("ANTHROPIC_API_KEY is not set; replies will fail")
        yield
        await app.state.chat.abandon()
        await app.state.extraction.abandon()
        await engine.dispose()

    app = FastAPI(title="Learning Journal", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=[settings.web_origin],
                       allow_methods=["*"], allow_headers=["*"])

    def chat(request: Request) -> ChatService:
        return request.app.state.chat

    def extraction(request: Request) -> ExtractionService:
        return request.app.state.extraction

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/sessions", response_model=list[SessionOut])
    async def list_sessions(request: Request):
        return await chat(request).list_sessions()

    @app.post("/api/sessions", response_model=SessionOut, status_code=201)
    async def create_session(body: SessionCreate, request: Request):
        """Starting a session also extracts every other session's unprocessed messages."""
        session = await chat(request).create_session(body.name)
        await extraction(request).trigger()
        return session

    @app.get("/api/sessions/{session_id}", response_model=SessionView)
    async def get_session(session_id: int, request: Request):
        try:
            return await chat(request).get_session(session_id)
        except SessionNotFound:
            raise HTTPException(404, "Session not found")

    @app.patch("/api/sessions/{session_id}", response_model=SessionOut)
    async def rename_session(session_id: int, body: SessionRename, request: Request):
        try:
            return await chat(request).rename_session(session_id, body.name)
        except SessionNotFound:
            raise HTTPException(404, "Session not found")

    @app.post("/api/sessions/{session_id}/messages", response_model=SendResult, status_code=202)
    async def send_message(session_id: int, body: MessageCreate, request: Request):
        try:
            return await chat(request).send_message(session_id, body.text)
        except SessionNotFound:
            raise HTTPException(404, "Session not found")
        except ReplyInProgress:
            raise HTTPException(409, "A reply is still being written in this session")

    @app.post("/api/sessions/{session_id}/retry", response_model=MessageOut, status_code=202)
    async def retry(session_id: int, request: Request):
        try:
            return await chat(request).retry(session_id)
        except SessionNotFound:
            raise HTTPException(404, "Session not found")
        except ReplyInProgress:
            raise HTTPException(409, "A reply is still being written in this session")
        except NothingToRetry:
            raise HTTPException(409, "Only a failed or interrupted latest reply can be retried")

    @app.get("/api/messages/{message_id}/events")
    async def message_events(message_id: int, request: Request):
        """Server-sent events for one reply: `snapshot` (content so far), then `delta`s,
        then `done` with the stored message. A finished reply sends only `done`."""
        service = chat(request)
        subscription = service.broker.subscribe(message_id)

        async def events():
            if subscription is None:
                message = await service.get_message(message_id)
                if message is None:
                    yield {"event": "error", "data": json.dumps({"detail": "Message not found"})}
                else:
                    yield {"event": "done", "data": message.model_dump_json()}
                return
            snapshot, queue = subscription
            try:
                yield {"event": "snapshot", "data": json.dumps(snapshot)}
                while True:
                    event = await queue.get()
                    if event["type"] == "delta":
                        yield {"event": "delta", "data": json.dumps(event)}
                    else:
                        yield {"event": "done", "data": json.dumps(event["message"])}
                        return
            finally:
                service.broker.unsubscribe(message_id, queue)

        return EventSourceResponse(events())

    @app.get("/api/extracted", response_model=list[ExtractedSession])
    async def extracted(request: Request):
        return await extraction(request).extracted()

    @app.post("/api/extraction/run", response_model=TriggerResult, status_code=202)
    async def process_now(request: Request):
        """The Process now button: the same trigger as starting a new session."""
        return TriggerResult(queued=await extraction(request).trigger())

    @app.post("/api/segments/{segment_id}/retry", response_model=SegmentOut, status_code=202)
    async def retry_segment(segment_id: int, request: Request):
        try:
            return await extraction(request).retry(segment_id)
        except SegmentNotFound:
            raise HTTPException(404, "Segment not found")
        except SegmentNotRetryable:
            raise HTTPException(409, "Only a segment that needs attention can be retried")

    if WEB_DIST.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")

    return app


app = create_app()
