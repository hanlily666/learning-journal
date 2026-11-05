"""HTTP API. Run with: uv run uvicorn app.main:app --reload"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette import EventSourceResponse

from app.chat import ChatService, NothingToRetry, ReplyInProgress, SessionNotFound
from app.claude import AnthropicTransport, ClaudeClient, Transport
from app.config import Settings, get_settings
from app.db import make_engine, make_sessionmaker
from app.schemas import (MessageCreate, MessageOut, SendResult, SessionCreate, SessionOut,
                         SessionRename, SessionView)

log = logging.getLogger(__name__)

WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"


def create_app(settings: Settings | None = None, transport: Transport | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = make_engine(settings.database_url)
        sessionmaker = make_sessionmaker(engine)
        claude = ClaudeClient(transport or AnthropicTransport(api_key=settings.anthropic_api_key),
                              sessionmaker, settings)
        app.state.chat = ChatService(sessionmaker, claude, settings)
        recovered = await app.state.chat.recover_interrupted()
        if recovered:
            log.warning("marked %d unfinished replies as interrupted", recovered)
        if not settings.anthropic_api_key and transport is None:
            log.warning("ANTHROPIC_API_KEY is not set; replies will fail")
        yield
        await app.state.chat.abandon()
        await engine.dispose()

    app = FastAPI(title="Learning Journal", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=[settings.web_origin],
                       allow_methods=["*"], allow_headers=["*"])

    def chat(request: Request) -> ChatService:
        return request.app.state.chat

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/sessions", response_model=list[SessionOut])
    async def list_sessions(request: Request):
        return await chat(request).list_sessions()

    @app.post("/api/sessions", response_model=SessionOut, status_code=201)
    async def create_session(body: SessionCreate, request: Request):
        return await chat(request).create_session(body.name)

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

    if WEB_DIST.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")

    return app


app = create_app()
