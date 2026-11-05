"""The one wrapper every Claude call goes through.

`ClaudeClient` builds the request, streams it through a transport, and logs every
attempt (failed ones included) in the `api_request` table. The real transport talks
to the Anthropic API; tests swap in a scripted model with the same `stream()` shape.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal, Protocol

import anthropic
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.config import Settings
from app.models import ApiRequest
from app.pricing import cost_usd

SYSTEM_PROMPT = (
    "You are a patient tutor in the user's learning journal app. The user studies two "
    "subjects here: AI and the French language. Answer clearly and accurately, and say "
    "so when you are unsure."
)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class Delta:
    kind: Literal["text", "thinking"]
    text: str


@dataclass
class Final:
    content: list[dict[str, Any]]
    stop_reason: str | None
    model: str | None
    usage: dict[str, Any]
    request_id: str | None = None


class TransportError(Exception):
    def __init__(self, message: str, *, status: int | None = None, request_id: str | None = None,
                 model: str | None = None):
        super().__init__(message)
        self.status = status
        self.request_id = request_id
        self.model = model


class Transport(Protocol):
    def stream(self, request: dict[str, Any]) -> AsyncIterator[Delta | Final]: ...


@dataclass
class ReplyResult:
    ok: bool
    content: list[dict[str, Any]]
    stop_reason: str | None = None
    model: str | None = None
    error: str | None = None


@dataclass
class PartialReply:
    """Content built up from stream deltas, so a broken stream still keeps what arrived."""

    blocks: list[dict[str, Any]] = field(default_factory=list)

    def add(self, delta: Delta) -> None:
        if not self.blocks or self.blocks[-1]["type"] != delta.kind:
            self.blocks.append({"type": delta.kind, delta.kind: ""})
        self.blocks[-1][delta.kind] += delta.text


def text_of(content: list[dict[str, Any]]) -> str:
    return "".join(b.get("text", "") for b in content if b.get("type") == "text")


def blocks_for_replay(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Content to send back to the API for a stored assistant message.

    Stored blocks are replayed unchanged, except after a mid-reply refusal fallback:
    the blocks the declining model produced before the last `fallback` marker (its
    thinking and tool calls) are left out, as the API requires.
    """
    last_fallback = max((i for i, b in enumerate(content) if b.get("type") == "fallback"), default=-1)
    if last_fallback < 0:
        return content
    dropped = {"thinking", "redacted_thinking", "tool_use", "server_tool_use"}
    return [b for i, b in enumerate(content) if i > last_fallback or b.get("type") not in dropped]


class ClaudeClient:
    def __init__(self, transport: Transport, sessionmaker: async_sessionmaker, settings: Settings):
        self.transport = transport
        self.sessionmaker = sessionmaker
        self.settings = settings

    def chat_request(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "model": self.settings.claude_model,
            "max_tokens": self.settings.reply_max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": messages,
            # Opus 5.5 always thinks; ask for a readable summary of it.
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": self.settings.claude_effort},
            # Automatic caching: the breakpoint moves forward as the conversation grows.
            "cache_control": {"type": "ephemeral"},
            # On a safety decline, the API re-runs the request on its recommended fallback model.
            "fallbacks": "default",
            "betas": [FALLBACK_BETA],
        }

    async def stream_reply(
        self,
        *,
        step: str,
        session_id: int | None,
        message_id: int | None,
        messages: list[dict[str, Any]],
        on_delta: Callable[[Delta, PartialReply], Awaitable[None]],
    ) -> ReplyResult:
        request = self.chat_request(messages)
        partial = PartialReply()
        log = ApiRequest(step=step, session_id=session_id, message_id=message_id,
                         model_requested=request["model"])
        result: ReplyResult
        try:
            final: Final | None = None
            async for event in self.transport.stream(request):
                if isinstance(event, Final):
                    final = event
                else:
                    partial.add(event)
                    await on_delta(event, partial)
            if final is None:
                raise TransportError("the stream ended before the reply was complete")
            self._record_usage(log, final)
            result = ReplyResult(ok=True, content=final.content, stop_reason=final.stop_reason,
                                 model=final.model)
        except TransportError as e:
            log.error = str(e)
            log.request_id = e.request_id
            log.model_served = e.model
            result = ReplyResult(ok=False, content=partial.blocks, error=str(e))
        except Exception as e:  # e.g. no API key: the SDK fails before any request is sent
            log.error = f"{type(e).__name__}: {e}"
            result = ReplyResult(ok=False, content=partial.blocks, error=log.error)
        await self._save_log(log)
        return result

    def _record_usage(self, log: ApiRequest, final: Final) -> None:
        u = final.usage
        log.model_served = final.model
        log.stop_reason = final.stop_reason
        log.request_id = final.request_id
        log.input_tokens = u.get("input_tokens")
        log.output_tokens = u.get("output_tokens")
        log.cache_read_tokens = u.get("cache_read_input_tokens")
        log.cache_write_tokens = u.get("cache_creation_input_tokens")
        log.usage = u
        cost = cost_usd(u, final.model)
        log.cost_usd = Decimal(str(round(cost, 6))) if cost is not None else None

    async def _save_log(self, log: ApiRequest) -> None:
        async with self.sessionmaker() as db:
            db.add(log)
            await db.commit()


class AnthropicTransport:
    """The real transport: a streaming beta Messages call through the Anthropic SDK."""

    def __init__(self, client: anthropic.AsyncAnthropic | None = None, api_key: str | None = None):
        # The key may come from .env, which the SDK doesn't read itself.
        self.client = client or anthropic.AsyncAnthropic(api_key=api_key)

    async def stream(self, request: dict[str, Any]) -> AsyncIterator[Delta | Final]:
        stream = None
        try:
            async with self.client.beta.messages.stream(**request) as stream:
                async for event in stream:
                    if event.type != "content_block_delta":
                        continue
                    if event.delta.type == "text_delta":
                        yield Delta("text", event.delta.text)
                    elif event.delta.type == "thinking_delta":
                        yield Delta("thinking", event.delta.thinking)
                message = await stream.get_final_message()
                yield Final(
                    content=[block.to_dict() for block in message.content],
                    stop_reason=message.stop_reason,
                    model=message.model,
                    usage=message.usage.to_dict(),
                    request_id=stream.request_id,
                )
        except anthropic.APIStatusError as e:
            raise TransportError(e.message, status=e.status_code, request_id=e.request_id) from e
        except anthropic.APIError as e:
            snapshot = stream._current_snapshot_or_none() if stream is not None else None
            raise TransportError(
                str(e) or type(e).__name__,
                request_id=stream.request_id if stream is not None else None,
                model=snapshot.model if snapshot is not None else None,
            ) from e
