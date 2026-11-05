"""A scripted model: a fake Claude transport that replays prepared replies.

Each call to `stream()` consumes the next script. A script is a list of steps:
a `Delta` or `Final` is yielded, an exception is raised, and a `Gate` blocks the
stream until the test opens it.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any

from app.claude import Delta, Final, TransportError

MODEL = "claude-opus-5-5"


@dataclass
class Gate:
    reached: asyncio.Event = field(default_factory=asyncio.Event)
    opened: asyncio.Event = field(default_factory=asyncio.Event)


class ScriptedModel:
    def __init__(self, *scripts: list[Any]):
        self.scripts = list(scripts)
        self.requests: list[dict[str, Any]] = []

    def add(self, *scripts: list[Any]) -> None:
        self.scripts.extend(scripts)

    async def stream(self, request: dict[str, Any]):
        self.requests.append(request)
        if not self.scripts:
            raise AssertionError("the scripted model has no reply left")
        for step in self.scripts.pop(0):
            if isinstance(step, Gate):
                step.reached.set()
                await step.opened.wait()
            elif isinstance(step, BaseException):
                raise step
            else:
                yield step


def usage(input_tokens=100, output_tokens=20, cache_read=0, cache_write=0) -> dict[str, Any]:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_write,
    }


def final(text: str, *, thinking: str | None = "Thinking it over.", stop_reason="end_turn",
          model=MODEL, request_id="req_test", usage_=None) -> Final:
    content: list[dict[str, Any]] = []
    if thinking is not None:
        content.append({"type": "thinking", "thinking": thinking, "signature": "sig-" + thinking[:8]})
    if text:
        content.append({"type": "text", "text": text})
    return Final(content=content, stop_reason=stop_reason, model=model,
                 usage=usage_ or usage(), request_id=request_id)


def reply(text: str, **kwargs) -> list[Any]:
    """A normal streamed reply: thinking, then the text in two chunks, then the final message."""
    half = len(text) // 2
    steps: list[Any] = []
    thinking = kwargs.get("thinking", "Thinking it over.")
    if thinking:
        steps.append(Delta("thinking", thinking))
    steps += [Delta("text", text[:half]), Delta("text", text[half:]), final(text, **kwargs)]
    return steps


def api_error(status: int | None = 529, message="Overloaded", request_id="req_err") -> TransportError:
    return TransportError(message, status=status, request_id=request_id)
