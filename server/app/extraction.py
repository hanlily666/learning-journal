"""Extraction: after I chat, pick the key points out of new messages.

Every complete message after a session's checkpoint forms a segment. One background
worker processes pending segments one at a time, oldest first:

1. Classify the segment as AI, French, or messy with one structured request.
2. AI: the agent (Tool Runner, tools `read_session` and `submit_items`) extracts items.
   French: held for Phase 6. Messy: skipped, with the classifier's reason kept.
3. The items, the segment's status, and the session's checkpoint are saved in one
   transaction, so a segment is never half-applied.

A failed attempt is retried once; then the segment needs my attention (a Retry button),
and later messages of that session wait behind it.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
from anthropic import beta_async_tool
from anthropic.lib.tools import ToolError
from pydantic import BaseModel, Field
from sqlalchemy import exists, func, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.claude import FALLBACK_BETA, record_usage, save_log
from app.config import Settings
from app.models import ApiRequest, ChatSession, Item, Message, Segment
from app.schemas import ExtractedSession, ItemField, ItemOut, SegmentOut, SessionOut

log = logging.getLogger(__name__)

SKILL_PATH = Path(__file__).resolve().parent / "skills" / "ai_selection.md"
MAX_TOOL_CALLS = 8  # per extraction run, counted here rather than trusted to the prompt
MAX_ATTEMPTS = 2  # the first try and one retry
CLASSIFY_CHARS = 40_000  # the classifier sees at most this much of a segment
READ_LIMIT = 20  # earlier messages per read_session call

CLASSIFY_SYSTEM = """You sort segments of a learning journal's chat history by subject.

Labels:
- ai: learning about artificial intelligence (machine learning, LLMs, agents, prompting, AI tools and research).
- french: learning the French language (vocabulary, grammar, conjugation, translation, practice).
- messy: anything else, including a segment that mixes AI and French, or mixes either with unrelated topics.

Give the label and a one-sentence reason."""

CLASSIFY_FORMAT = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": ["ai", "french", "messy"]},
            "reason": {"type": "string"},
        },
        "required": ["label", "reason"],
        "additionalProperties": False,
    },
}

AGENT_SYSTEM = """You extract learning notes from one segment of my chat with an AI tutor, for my learning journal. The segment is about learning AI.

Fill the AI learning schema:
- subject: the subject of the chat, as a short title
- summary: what I asked and learned, in a few sentences
- items, each in one field:
  - explore: things worth exploring on my own
  - source: external sources mentioned in the chat (papers, tools, sites, people)
  - gap: knowledge gaps I had while asking
  - concept: key concepts to review later

Extract only from the messages inside <new_messages>. Earlier parts of the session were processed already: their summaries are given for context, and read_session returns earlier messages if you need them, but don't extract anything that appears only there. Each item cites the IDs of the new messages it comes from and says why it was picked.

Finish by calling submit_items exactly once. If it returns an error, fix the input and call it again.

The selection skill below decides what is worth picking.

<selection_skill>
{skill}
</selection_skill>"""


class ExtractionFailed(Exception):
    pass


class SegmentNotFound(Exception):
    pass


class SegmentNotRetryable(Exception):
    """Only a segment that needs attention can be retried."""


class SubmittedItem(BaseModel):
    field: ItemField
    text: str = Field(min_length=1)
    source_message_ids: list[int] = Field(min_length=1)
    reason: str = Field(min_length=1)


@dataclass
class Submission:
    subject: str
    summary: str
    items: list[SubmittedItem]


def render(messages: list[Message]) -> str:
    return "\n".join(f'<message id="{m.id}" role="{m.role}">\n{m.text}\n</message>' for m in messages)


def usable(messages: list[Message]) -> list[Message]:
    """Plain text only: no injected context, nothing unfinished, nothing empty."""
    return [m for m in messages
            if m.status == "complete" and not m.injected_context and m.text.strip()]


def parse_label(message: Any) -> tuple[str, str]:
    """The (label, reason) of a classification reply."""
    try:
        answer = json.loads(next(b.text for b in message.content if b.type == "text"))
        return answer["label"], answer["reason"]
    except (StopIteration, ValueError, KeyError, TypeError) as e:
        raise ExtractionFailed(f"Classification returned no label (stop reason "
                               f"{message.stop_reason}).") from e


async def segment_messages(db: Any, segment: Segment) -> list[Message]:
    """A segment's usable messages, in order."""
    return usable(list(await db.scalars(select(Message).where(
        Message.session_id == segment.session_id,
        Message.id.between(segment.first_message_id, segment.last_message_id))
        .order_by(Message.id))))


class ExtractionService:
    def __init__(self, sessionmaker: async_sessionmaker, client: anthropic.AsyncAnthropic,
                 settings: Settings, skill: str | None = None):
        self.sessionmaker = sessionmaker
        self.client = client
        self.settings = settings
        self.skill = skill if skill is not None else SKILL_PATH.read_text()
        self._worker: asyncio.Task | None = None
        self._again = False

    # Triggers

    async def trigger(self) -> int:
        """Queue a segment for every session with unprocessed messages, then work through them."""
        queued = await self.queue_segments()
        self.kick()
        return queued

    async def queue_segments(self) -> int:
        unprocessed = (select(Message.id).where(
            Message.session_id == ChatSession.id, Message.status == "complete",
            Message.id > func.coalesce(ChatSession.checkpoint, 0)).exists())
        queued = 0
        async with self.sessionmaker() as db, db.begin():
            candidates = list(await db.scalars(
                select(ChatSession.id).where(unprocessed).order_by(ChatSession.id)))
            for session_id in candidates:
                # The same lock a new message takes, so a segment never cuts a reply in half.
                session = await db.get(ChatSession, session_id, with_for_update=True)
                assert session is not None
                busy = await db.scalar(select(
                    exists().where(Message.session_id == session_id, Message.status == "streaming")
                    | exists().where(Segment.session_id == session_id,
                                     Segment.status.in_(("pending", "running", "needs_attention")))))
                if busy:
                    continue
                first, last = (await db.execute(
                    select(func.min(Message.id), func.max(Message.id)).where(
                        Message.session_id == session_id, Message.status == "complete",
                        Message.id > (session.checkpoint or 0)))).one()
                if first is None:
                    continue
                db.add(Segment(session_id=session_id, first_message_id=first, last_message_id=last))
                queued += 1
        return queued

    def kick(self) -> None:
        """Start the worker, or tell a running one to look for new segments again."""
        self._again = True
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._work())

    async def recover(self) -> int:
        """On startup: segments a stopped server left running go back in the queue."""
        async with self.sessionmaker() as db, db.begin():
            result = await db.execute(update(Segment).where(Segment.status == "running")
                                      .values(status="pending", updated_at=func.now()))
            return result.rowcount

    async def retry(self, segment_id: int) -> SegmentOut:
        async with self.sessionmaker() as db, db.begin():
            segment = await db.get(Segment, segment_id, with_for_update=True)
            if segment is None:
                raise SegmentNotFound(segment_id)
            if segment.status != "needs_attention":
                raise SegmentNotRetryable(segment_id)
            segment.status = "pending"
            segment.attempts = 0
            segment.updated_at = func.now()
        self.kick()
        return await self._segment_out(segment_id)

    async def wait_idle(self) -> None:
        while self._worker is not None and not self._worker.done():
            await asyncio.gather(self._worker, return_exceptions=True)

    async def abandon(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        await self.wait_idle()

    # The Extracted page

    async def extracted(self) -> list[ExtractedSession]:
        """Sessions with segments, most recently processed first; segments oldest first."""
        async with self.sessionmaker() as db:
            segments = list(await db.scalars(select(Segment).order_by(Segment.id)))
            items = list(await db.scalars(select(Item).order_by(Item.id)))
            sessions = {s.id: s for s in await db.scalars(
                select(ChatSession).where(ChatSession.id.in_({s.session_id for s in segments})))}
        by_segment: dict[int, list[ItemOut]] = {}
        for item in items:
            by_segment.setdefault(item.segment_id, []).append(ItemOut.model_validate(item))
        views: dict[int, ExtractedSession] = {}
        for segment in segments:
            out = SegmentOut.model_validate(segment).model_copy(
                update={"items": by_segment.get(segment.id, [])})
            view = views.setdefault(segment.session_id, ExtractedSession(
                session=SessionOut.model_validate(sessions[segment.session_id]), segments=[]))
            view.segments.append(out)
        return sorted(views.values(), key=lambda v: v.segments[-1].id, reverse=True)

    # The worker

    async def _work(self) -> None:
        try:
            while self._again:
                self._again = False
                while (segment_id := await self._claim()) is not None:
                    await self._process(segment_id)
                    self._again = False  # this pass sees every segment queued so far
        except Exception:  # e.g. the database is down: the next trigger starts a new worker
            log.exception("the extraction worker stopped")

    async def _claim(self) -> int | None:
        async with self.sessionmaker() as db, db.begin():
            segment = await db.scalar(select(Segment).where(Segment.status == "pending")
                                      .order_by(Segment.id).limit(1)
                                      .with_for_update(skip_locked=True))
            if segment is None:
                return None
            segment.status = "running"
            segment.attempts += 1
            segment.updated_at = func.now()
            return segment.id

    async def _process(self, segment_id: int) -> None:
        try:
            async with self.sessionmaker() as db:
                segment = await db.get(Segment, segment_id)
                assert segment is not None
                messages = await segment_messages(db, segment)
            if not messages:
                await self._finish(segment, "skipped", label_reason="No plain-text messages.")
                return
            label = segment.label
            if label is None:
                label, reason = await self._classify(segment, messages)
                await self.save_label(segment, label, reason)
            if label == "ai":
                submission = await self._extract(segment, messages)
                await self._finish(segment, "done", submission=submission)
            elif label == "french":
                await self._finish(segment, "held")
            else:
                await self._finish(segment, "skipped")
        except asyncio.CancelledError:
            raise
        except ExtractionFailed as e:
            await self._fail(segment_id, str(e))
        except Exception as e:  # a bug or a database problem
            log.exception("segment %s failed", segment_id)
            await self._fail(segment_id, f"Server error: {e}")

    async def save_label(self, segment: Segment, label: str, reason: str) -> None:
        """Store a segment's label, and show it as the session's latest subject."""
        async with self.sessionmaker() as db, db.begin():
            await db.execute(update(Segment).where(Segment.id == segment.id)
                             .values(label=label, label_reason=reason, updated_at=func.now()))
            await db.execute(update(ChatSession).where(ChatSession.id == segment.session_id)
                             .values(subject=label))
        segment.label, segment.label_reason = label, reason

    async def _finish(self, segment: Segment, status: str, *, submission: Submission | None = None,
                      label_reason: str | None = None) -> None:
        """Items, status, and checkpoint together, or none of them."""
        async with self.sessionmaker() as db, db.begin():
            values: dict[str, Any] = {"status": status, "error": None, "updated_at": func.now()}
            if label_reason is not None:
                values["label_reason"] = label_reason
            if submission is not None:
                values.update(subject=submission.subject, summary=submission.summary)
                db.add_all(Item(segment_id=segment.id, field=i.field, text=i.text,
                                source_message_ids=i.source_message_ids, reason=i.reason)
                           for i in submission.items)
            await db.execute(update(Segment).where(Segment.id == segment.id).values(**values))
            await db.execute(update(ChatSession).where(ChatSession.id == segment.session_id).values(
                checkpoint=func.greatest(func.coalesce(ChatSession.checkpoint, 0),
                                         segment.last_message_id)))

    async def _fail(self, segment_id: int, error: str) -> None:
        async with self.sessionmaker() as db, db.begin():
            segment = await db.get(Segment, segment_id, with_for_update=True)
            assert segment is not None
            segment.status = "pending" if segment.attempts < MAX_ATTEMPTS else "needs_attention"
            segment.error = error
            segment.updated_at = func.now()

    async def _segment_out(self, segment_id: int) -> SegmentOut:
        async with self.sessionmaker() as db:
            segment = await db.get(Segment, segment_id)
            items = await db.scalars(select(Item).where(Item.segment_id == segment_id).order_by(Item.id))
            return SegmentOut.model_validate(segment).model_copy(
                update={"items": [ItemOut.model_validate(i) for i in items]})

    # Claude calls

    def _params(self, *, system: str, messages: list[dict[str, Any]], effort: str,
                max_tokens: int, output_format: dict[str, Any] | None = None) -> dict[str, Any]:
        output_config: dict[str, Any] = {"effort": effort}
        if output_format is not None:
            output_config["format"] = output_format
        return {
            "model": self.settings.claude_model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
            "thinking": {"type": "adaptive"},
            "output_config": output_config,
            "cache_control": {"type": "ephemeral"},
            "fallbacks": "default",
            "betas": [FALLBACK_BETA],
        }

    async def _log(self, step: str, session_id: int, message: Any = None,
                   error: anthropic.APIError | None = None) -> None:
        row = ApiRequest(step=step, session_id=session_id, model_requested=self.settings.claude_model)
        if message is not None:
            record_usage(row, model=message.model, stop_reason=message.stop_reason,
                         request_id=getattr(message, "_request_id", None),
                         usage=message.usage.to_dict())
        if error is not None:
            row.error = str(error) or type(error).__name__
            row.request_id = getattr(error, "request_id", None)
        await save_log(self.sessionmaker, row)

    def classification_request(self, messages: list[Message]) -> dict[str, Any]:
        """The classification request for a segment's (usable) messages."""
        return self._params(system=CLASSIFY_SYSTEM, effort="low", max_tokens=4000,
                            output_format=CLASSIFY_FORMAT,
                            messages=[{"role": "user", "content": render(messages)[:CLASSIFY_CHARS]}])

    async def _classify(self, segment: Segment, messages: list[Message]) -> tuple[str, str]:
        try:
            message = await self.client.beta.messages.create(**self.classification_request(messages))
        except anthropic.APIError as e:
            await self._log("classification", segment.session_id, error=e)
            raise ExtractionFailed(f"Classification failed: {e}") from e
        await self._log("classification", segment.session_id, message)
        return parse_label(message)

    async def _earlier_summaries(self, segment: Segment) -> list[str]:
        async with self.sessionmaker() as db:
            rows = await db.execute(select(Segment.subject, Segment.summary).where(
                Segment.session_id == segment.session_id, Segment.id < segment.id,
                Segment.status == "done").order_by(Segment.id))
            return [f"- {subject}: {summary}" for subject, summary in rows]

    async def _extract(self, segment: Segment, messages: list[Message]) -> Submission:
        new_ids = {m.id for m in messages}
        submitted: list[Submission] = []

        @beta_async_tool
        async def read_session(before_message_id: int | None = None) -> str:
            """Read earlier messages of this session, from before the new messages.

            Returns up to 20 messages, the latest first-to-last. Call again with the
            smallest ID you got to read further back.

            Args:
                before_message_id: Only messages with a smaller ID. Leave out to start just
                    before the new messages.
            """
            limit = segment.first_message_id if before_message_id is None else min(
                before_message_id, segment.first_message_id)
            async with self.sessionmaker() as db:
                rows = list(await db.scalars(select(Message).where(
                    Message.session_id == segment.session_id, Message.id < limit)
                    .order_by(Message.id.desc()).limit(READ_LIMIT)))
            earlier = usable(rows[::-1])
            return render(earlier) if earlier else "No earlier messages."

        @beta_async_tool
        async def submit_items(subject: str, summary: str, items: list[SubmittedItem]) -> str:
            """Save this segment's subject, summary, and items. Call it exactly once, at the end.

            Args:
                subject: The subject of the chat, as a short title.
                summary: What I asked and learned, in a few sentences.
                items: The picked items. Each has a field (explore, source, gap, or concept),
                    its text, the IDs of the new messages it comes from, and why it was picked.
            """
            if submitted:
                raise ToolError("Items were already submitted for this segment.")
            outside = sorted({i for item in items for i in item.source_message_ids} - new_ids)
            if outside:
                raise ToolError(f"Not new messages of this segment: {outside}. "
                                "Cite only IDs of messages inside <new_messages>.")
            submitted.append(Submission(subject, summary, items))
            return "Saved."

        earlier = await self._earlier_summaries(segment)
        prompt = ""
        if earlier:
            prompt += ("<earlier_summaries>\n" + "\n".join(earlier) + "\n</earlier_summaries>\n\n")
        prompt += (f"<new_messages>\n{render(messages)}\n</new_messages>\n\n"
                   "Extract items from the new messages only, then call submit_items.")
        params = self._params(system=AGENT_SYSTEM.format(skill=self.skill.strip()),
                              effort=self.settings.claude_effort, max_tokens=16_000,
                              messages=[{"role": "user", "content": prompt}])
        runner = self.client.beta.messages.tool_runner(
            **params, tools=[read_session, submit_items],
            # Send the whole history each turn, as plain Messages requests.
            thread=False, max_iterations=MAX_TOOL_CALLS + 1)
        calls = 0
        try:
            async for message in runner:
                await self._log("extraction", segment.session_id, message)
                if message.stop_reason in ("max_tokens", "refusal"):
                    raise ExtractionFailed(f"The agent stopped early ({message.stop_reason}).")
                calls += sum(1 for b in message.content if b.type == "tool_use")
                if calls > MAX_TOOL_CALLS:
                    raise ExtractionFailed(f"The agent hit the limit of {MAX_TOOL_CALLS} tool calls.")
                if message.stop_reason == "tool_use":
                    # Run the tools now (the runner reuses these results), so the run can
                    # end as soon as the items are in, without another turn.
                    await runner.generate_tool_call_response()
                    if submitted:
                        break
        except anthropic.APIError as e:
            await self._log("extraction", segment.session_id, error=e)
            raise ExtractionFailed(f"Extraction failed: {e}") from e
        finally:
            await runner.close()
        if not submitted:
            raise ExtractionFailed("The agent ended without submitting items.")
        return submitted[0]
