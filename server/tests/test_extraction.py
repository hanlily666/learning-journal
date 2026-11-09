"""Extraction: real Postgres, chat through the scripted model, and the extraction calls
(classification and the Tool Runner agent) through a fake Anthropic API."""

import json

import pytest
from sqlalchemy import select, text

from app.chat import ChatService
from app.claude import ClaudeClient
from app.config import Settings
from app.extraction import MAX_TOOL_CALLS, ExtractionService, SegmentNotRetryable
from app.models import ApiRequest
from tests import fake_api as api
from tests.scripted import Gate, ScriptedModel, reply

SETTINGS = Settings(partial_save_interval=0, anthropic_api_key=None)


class World:
    """Chat and extraction over one database, with the fake API's replies queued per test."""

    def __init__(self, db, *replies):
        self.db = db
        self.model = ScriptedModel()
        self.chat = ChatService(db, ClaudeClient(self.model, db, SETTINGS), SETTINGS)
        self.api = api.FakeAPI(*replies)
        self.extraction = ExtractionService(db, self.api.client, SETTINGS, skill="Pick what matters.")

    async def converse(self, session_id: int, *turns: tuple[str, str]) -> list[int]:
        """Send each (question, answer) turn; returns the new message IDs in order."""
        ids = []
        for question, answer in turns:
            self.model.add(reply(answer))
            sent = await self.chat.send_message(session_id, question)
            await self.chat.wait_idle()
            ids += [sent.user_message.id, sent.assistant_message.id]
        return ids

    async def new_session(self) -> int:
        """Start a session the way the app does: it triggers extraction of the others."""
        session = await self.chat.create_session()
        await self.extraction.trigger()
        await self.extraction.wait_idle()
        return session.id


async def test_starting_a_new_session_extracts_items_from_an_ai_session(db):
    world = World(db)
    first = await world.new_session()
    q, a = await world.converse(first, ("What is RAG?", "Retrieval-augmented generation is ..."))
    world.api.add(
        api.label("ai", "About retrieval-augmented generation."),
        api.submit("RAG basics", "I asked what RAG is.",
                   api.item("concept", "Retrieval-augmented generation", [q, a]),
                   api.item("explore", "Compare chunking strategies", [a])),
    )

    await world.new_session()

    [view] = await world.extraction.extracted()
    assert view.session.id == first
    assert view.session.subject == "ai"
    [segment] = view.segments
    assert (segment.first_message_id, segment.last_message_id) == (q, a)
    assert (segment.label, segment.status) == ("ai", "done")
    assert (segment.subject, segment.summary) == ("RAG basics", "I asked what RAG is.")
    assert [(i.field, i.text, i.source_message_ids) for i in segment.items] == [
        ("concept", "Retrieval-augmented generation", [q, a]),
        ("explore", "Compare chunking strategies", [a]),
    ]
    assert segment.items[0].reason == "Core to the goal."
    assert segment.items[0].review_state == "unreviewed"
    # The submit call ends the run: no extra turn after it.
    assert len(world.api.requests) == 2
    async with db() as s:
        steps = [r.step for r in await s.scalars(select(ApiRequest).order_by(ApiRequest.id))]
    assert steps == ["chat", "classification", "extraction"]


async def test_continuing_a_session_extracts_only_its_new_messages(db):
    world = World(db)
    first = await world.new_session()
    q1, a1 = await world.converse(first, ("What is RAG?", "Retrieval-augmented generation ..."))
    world.api.add(api.label("ai"), api.submit("RAG basics", "What RAG is.",
                                              api.item("concept", "RAG", [q1])))
    await world.new_session()

    q2, a2 = await world.converse(first, ("How do I evaluate it?", "Measure retrieval recall ..."))
    world.api.add(api.label("ai"), api.submit("Evaluating RAG", "How to measure it.",
                                              api.item("gap", "How to evaluate retrieval", [q2])))
    await world.new_session()

    [view] = await world.extraction.extracted()
    older, newer = view.segments
    assert (newer.first_message_id, newer.last_message_id) == (q2, a2)
    assert [i.text for i in older.items] == ["RAG"]
    assert [i.text for i in newer.items] == ["How to evaluate retrieval"]
    prompt = world.api.requests[-1]["messages"][0]["content"]
    assert "- RAG basics: What RAG is." in prompt
    assert "How do I evaluate it?" in prompt and "What is RAG?" not in prompt
    # Nothing is left to process: another new session sends no request.
    sent = len(world.api.requests)
    await world.new_session()
    assert len(world.api.requests) == sent


async def test_french_is_held_and_messy_is_skipped_with_the_reason_kept(db):
    world = World(db)
    french = await world.new_session()
    await world.converse(french, ("Conjugate aller", "Je vais, tu vas ..."))
    messy = await world.chat.create_session()
    await world.converse(messy.id, ("Best pizza in Paris?", "Try ..."))
    world.api.add(api.label("french", "Conjugation practice."), api.label("messy", "Food, not learning."))

    await world.new_session()

    segments = {v.session.id: (v.session.subject, v.segments[0]) for v in await world.extraction.extracted()}
    assert segments[french][0] == "french"
    assert (segments[french][1].status, segments[french][1].label_reason) == ("held", "Conjugation practice.")
    assert segments[messy.id][0] == "messy"
    assert (segments[messy.id][1].status, segments[messy.id][1].label_reason) == ("skipped", "Food, not learning.")
    assert len(world.api.requests) == 2  # two classifications, no agent run
    # Both checkpoints moved on: nothing is queued again.
    assert await world.extraction.trigger() == 0


async def test_the_agent_and_classifier_see_plain_text_only(db):
    world = World(db)
    first = await world.new_session()
    await world.converse(first, ("What is RLHF?", "Reinforcement learning from human feedback ..."))
    world.api.add(api.label("ai"), api.submit("RLHF", "What RLHF is."))
    await world.new_session()

    for request in world.api.requests:
        sent = json.dumps(request["messages"])
        assert "What is RLHF?" in sent
        assert "Thinking it over." not in sent  # the chat reply's thinking summary


async def test_a_session_is_not_cut_while_a_reply_is_streaming(db):
    world = World(db)
    first = await world.new_session()
    await world.converse(first, ("What is RAG?", "Retrieval ..."))
    gate = Gate()
    world.model.add([gate, *reply("A later answer.")])
    await world.chat.send_message(first, "And fine-tuning?")
    await gate.reached.wait()

    assert await world.extraction.trigger() == 0

    gate.opened.set()
    await world.chat.wait_idle()
    world.api.add(api.label("ai"), api.submit("RAG and fine-tuning", "Both."))
    assert await world.extraction.trigger() == 1
    await world.extraction.wait_idle()
    [view] = await world.extraction.extracted()
    assert view.segments[0].status == "done"


async def test_a_bad_citation_comes_back_as_an_error_the_agent_can_fix(db):
    world = World(db)
    first = await world.new_session()
    q, a = await world.converse(first, ("What is RAG?", "Retrieval ..."))
    world.api.add(
        api.label("ai"),
        api.submit("RAG", "What RAG is.", api.item("concept", "RAG", [q, 9999])),
        api.submit("RAG", "What RAG is.", api.item("concept", "RAG", [q])),
    )

    await world.new_session()

    [result] = api.tool_results(world.api.requests[-1])
    assert result["is_error"] is True and "9999" in str(result["content"])
    [view] = await world.extraction.extracted()
    assert [i.source_message_ids for i in view.segments[0].items] == [[q]]


async def test_the_agent_can_read_earlier_messages_of_the_session(db):
    world = World(db)
    first = await world.new_session()
    q1, _ = await world.converse(first, ("What is RAG?", "Retrieval ..."))
    world.api.add(api.label("ai"), api.submit("RAG", "What RAG is."))
    await world.new_session()
    q2, _ = await world.converse(first, ("And its limits?", "Stale indexes ..."))
    world.api.add(api.label("ai"), api.tool_call("read_session", {}),
                  api.submit("RAG limits", "Its limits.", api.item("gap", "RAG limits", [q2])))

    await world.new_session()

    [result] = api.tool_results(world.api.requests[-1])
    assert f'<message id="{q1}" role="user">\nWhat is RAG?' in str(result["content"])
    assert "And its limits?" not in str(result["content"])


async def test_a_run_over_the_tool_call_limit_fails_then_waits_for_retry(db):
    world = World(db)
    first = await world.new_session()
    q, _ = await world.converse(first, ("What is RAG?", "Retrieval ..."))
    reads = [api.tool_call("read_session", {}) for _ in range(MAX_TOOL_CALLS + 1)]
    world.api.add(api.label("ai"), *reads, *reads)  # the first try and the one retry

    await world.new_session()

    [view] = await world.extraction.extracted()
    [segment] = view.segments
    assert segment.status == "needs_attention"
    assert segment.error == f"The agent hit the limit of {MAX_TOOL_CALLS} tool calls."
    assert segment.items == []
    # Classified once; each run stops at the 9th call without running it.
    assert len(world.api.requests) == 1 + 2 * (MAX_TOOL_CALLS + 1)
    assert len(api.tool_results(world.api.requests[-1])) == 1

    # Later messages of the session wait behind it.
    await world.converse(first, ("More?", "Yes ..."))
    assert await world.extraction.trigger() == 0

    world.api.add(api.submit("RAG", "What RAG is.", api.item("concept", "RAG", [q])))
    retried = await world.extraction.retry(segment.id)
    assert retried.status == "pending"
    await world.extraction.wait_idle()
    [view] = await world.extraction.extracted()
    assert (view.segments[0].status, view.segments[0].error) == ("done", None)
    # The label was kept, so the retry made no new classification request.
    assert "submit_items" in json.dumps(world.api.requests[-1]["tools"])
    # Now the later messages are picked up.
    assert await world.extraction.trigger() == 1


async def test_an_api_error_is_retried_once_and_logged(db):
    world = World(db)
    first = await world.new_session()
    q, _ = await world.converse(first, ("What is RAG?", "Retrieval ..."))
    world.api.add(api.label("ai"), api.overloaded(),
                  api.submit("RAG", "What RAG is.", api.item("concept", "RAG", [q])))

    await world.new_session()

    [view] = await world.extraction.extracted()
    assert (view.segments[0].status, view.segments[0].attempts) == ("done", 2)
    async with db() as s:
        failed = list(await s.scalars(select(ApiRequest).where(ApiRequest.error.is_not(None))))
    assert [(r.step, r.request_id) for r in failed] == [("extraction", "req_busy")]


async def test_a_run_that_never_submits_needs_attention(db):
    world = World(db)
    first = await world.new_session()
    await world.converse(first, ("What is RAG?", "Retrieval ..."))
    world.api.add(api.label("ai"), api.text("All done!"), api.text("All done!"))

    await world.new_session()

    [view] = await world.extraction.extracted()
    assert view.segments[0].status == "needs_attention"
    assert view.segments[0].error == "The agent ended without submitting items."
    world.api.add(api.submit("RAG", "What RAG is."))
    await world.extraction.retry(view.segments[0].id)
    with pytest.raises(SegmentNotRetryable):  # it is pending again, not waiting for attention
        await world.extraction.retry(view.segments[0].id)
    await world.extraction.wait_idle()


async def test_segments_left_running_by_a_stopped_server_are_queued_again(db):
    world = World(db)
    first = await world.new_session()
    await world.converse(first, ("What is RAG?", "Retrieval ..."))
    gate_reply = api.label("ai")
    world.api.add(gate_reply)
    await world.extraction.queue_segments()
    async with db() as s, s.begin():
        await s.execute(text("UPDATE segment SET status = 'running'"))

    assert await world.extraction.recover() == 1

    world.api.add(api.submit("RAG", "What RAG is."))
    world.extraction.kick()
    await world.extraction.wait_idle()
    [view] = await world.extraction.extracted()
    assert view.segments[0].status == "done"
