"""Parse a Claude data export (conversations.json) into plain-text conversations.

Keeps my messages and Claude's text replies only: thinking, tool calls, tool results,
and attachments are dropped, and so are messages left with no text.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROLES = {"human": "user", "assistant": "assistant"}


@dataclass
class ExportMessage:
    id: str
    role: str
    text: str
    created_at: datetime


@dataclass
class Conversation:
    id: str
    name: str
    created_at: datetime
    messages: list[ExportMessage] = field(default_factory=list)


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _text(message: dict[str, Any]) -> str:
    blocks = message.get("content")
    if not blocks:
        return message.get("text") or ""
    return "".join(b.get("text") or "" for b in blocks if b.get("type") == "text")


def parse(path: Path) -> list[Conversation]:
    conversations = []
    for raw in json.loads(Path(path).read_text()):
        messages = [
            ExportMessage(id=m["uuid"], role=ROLES[m["sender"]], text=text.strip(),
                          created_at=_time(m["created_at"]))
            for m in raw.get("chat_messages") or []
            if m.get("sender") in ROLES and (text := _text(m)).strip()
        ]
        if messages:
            conversations.append(Conversation(id=raw["uuid"], name=raw.get("name") or "",
                                              created_at=_time(raw["created_at"]),
                                              messages=messages))
    return conversations
