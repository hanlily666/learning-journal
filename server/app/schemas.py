"""Shapes the API returns. The browser's TypeScript types are generated from these."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MessageStatus = Literal["streaming", "complete", "interrupted", "failed"]


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str | None
    source: str
    subject: str | None
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: int
    role: Literal["user", "assistant"]
    content: list[dict[str, Any]]
    text: str
    created_at: datetime
    status: MessageStatus
    stop_reason: str | None
    error: str | None
    injected_context: bool


class SessionView(BaseModel):
    session: SessionOut
    messages: list[MessageOut]
    # Size of the conversation as of the latest reply (input + cached + output tokens).
    context_tokens: int
    # True once the conversation is big enough that a new session is the better choice.
    context_warning: bool
    # True when the latest reply failed or was interrupted and can be tried again.
    can_retry: bool


class SessionCreate(BaseModel):
    name: str | None = Field(default=None, max_length=200)


class SessionRename(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class MessageCreate(BaseModel):
    text: str = Field(min_length=1, max_length=200_000)


class SendResult(BaseModel):
    user_message: MessageOut
    assistant_message: MessageOut
