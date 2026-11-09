"""A fake Anthropic API: a real `AsyncAnthropic` client whose HTTP calls get canned replies.

Used for extraction (classification and the Tool Runner agent). Each request takes the
next reply in the queue; every request body is kept so tests can check what was sent.
"""

import itertools
import json
from typing import Any

import anthropic
import httpx2

MODEL = "claude-opus-5-5"
_ids = itertools.count(1)


class FakeAPI:
    def __init__(self, *replies: dict[str, Any] | httpx2.Response):
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.client = anthropic.AsyncAnthropic(
            api_key="test-key", max_retries=0,
            http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(self._handle)))

    def add(self, *replies: dict[str, Any] | httpx2.Response) -> None:
        self.replies.extend(replies)

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        if not self.replies:
            raise AssertionError("the fake API has no reply left")
        reply = self.replies.pop(0)
        if isinstance(reply, httpx2.Response):
            return reply
        return httpx2.Response(200, json=reply, headers={"request-id": f"req_{next(_ids)}"})


def usage(input_tokens=1000, output_tokens=100, cache_read=0, cache_write=0) -> dict[str, Any]:
    return {"input_tokens": input_tokens, "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read, "cache_creation_input_tokens": cache_write}


def message(content: list[dict[str, Any]], stop_reason="end_turn", usage_=None) -> dict[str, Any]:
    return {"id": f"msg_{next(_ids)}", "type": "message", "role": "assistant", "model": MODEL,
            "content": content, "stop_reason": stop_reason, "stop_sequence": None,
            "usage": usage_ or usage()}


def label(name: str, reason: str = "Because.") -> dict[str, Any]:
    """A classification reply."""
    return message([{"type": "text", "text": json.dumps({"label": name, "reason": reason})}])


def tool_call(name: str, input_: dict[str, Any], *more: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    """An agent turn that calls one or more tools."""
    calls = [(name, input_), *more]
    return message([{"type": "tool_use", "id": f"toolu_{next(_ids)}", "name": n, "input": i}
                    for n, i in calls], stop_reason="tool_use")


def submit(subject: str, summary: str, *items: dict[str, Any]) -> dict[str, Any]:
    return tool_call("submit_items", {"subject": subject, "summary": summary, "items": list(items)})


def item(field: str, text: str, sources: list[int], reason: str = "Core to the goal.") -> dict[str, Any]:
    return {"field": field, "text": text, "source_message_ids": sources, "reason": reason}


def text(answer: str) -> dict[str, Any]:
    return message([{"type": "text", "text": answer}])


def overloaded() -> httpx2.Response:
    return httpx2.Response(529, headers={"request-id": "req_busy"}, json={
        "type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})


def tool_results(request: dict[str, Any]) -> list[dict[str, Any]]:
    """The tool results the runner sent back in a request's last message."""
    return [b for b in request["messages"][-1]["content"] if b.get("type") == "tool_result"]
