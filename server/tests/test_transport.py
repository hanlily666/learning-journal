"""The real Anthropic transport, fed canned HTTP responses (no network, no API key)."""

import json

import anthropic
import httpx2
import pytest

from app.claude import AnthropicTransport, ClaudeClient, Delta, Final, TransportError
from app.config import Settings


def sse(*events: dict) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


STREAM = sse(
    {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 1, "cache_read_input_tokens": 0,
                  "cache_creation_input_tokens": 0}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Hmm."}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig1"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Bon"}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "jour"}},
    {"type": "content_block_stop", "index": 1},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
     "usage": {"output_tokens": 7}},
    {"type": "message_stop"},
)


def transport_with(handler) -> AnthropicTransport:
    client = anthropic.AsyncAnthropic(
        api_key="test-key", max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)))
    return AnthropicTransport(client)


def chat_request() -> dict:
    client = ClaudeClient(transport_with(lambda r: httpx2.Response(500)), None, Settings())  # type: ignore[arg-type]
    return client.chat_request([{"role": "user", "content": [{"type": "text", "text": "Salut"}]}])


async def test_streams_deltas_then_the_final_message_with_exact_blocks():
    seen = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["body"] = json.loads(request.content)
        seen["beta"] = request.headers.get("anthropic-beta")
        return httpx2.Response(200, content=STREAM, headers={
            "content-type": "text/event-stream", "request-id": "req_abc"})

    events = [e async for e in transport_with(handler).stream(chat_request())]

    assert events[:3] == [Delta("thinking", "Hmm."), Delta("text", "Bon"), Delta("text", "jour")]
    final = events[-1]
    assert isinstance(final, Final)
    assert final.content == [{"type": "thinking", "thinking": "Hmm.", "signature": "sig1"},
                             {"type": "text", "text": "Bonjour"}]
    assert (final.stop_reason, final.model, final.request_id) == ("end_turn", "claude-opus-5-5", "req_abc")
    assert final.usage["output_tokens"] == 7
    assert seen["beta"] == "server-side-fallback-2026-07-01"
    body = seen["body"]
    assert body["fallbacks"] == "default" and body["stream"] is True
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "medium"}
    assert body["cache_control"] == {"type": "ephemeral"}
    assert "betas" not in body


async def test_api_errors_become_transport_errors_with_the_request_id():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(529, headers={"request-id": "req_busy"}, json={
            "type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})

    with pytest.raises(TransportError) as raised:
        [e async for e in transport_with(handler).stream(chat_request())]
    assert (raised.value.status, raised.value.request_id) == (529, "req_busy")
