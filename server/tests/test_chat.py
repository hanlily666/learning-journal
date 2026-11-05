"""Chat behaviour, through the chat service with a scripted model in place of Claude."""

from datetime import timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select

from app.chat import ChatService, NothingToRetry, ReplyInProgress
from app.claude import ClaudeClient
from app.config import Settings
from app.models import ApiRequest
from app.claude import Delta
from tests.scripted import Gate, ScriptedModel, api_error, final, reply, usage

SETTINGS = Settings(partial_save_interval=0, anthropic_api_key=None)


def make_service(db, model: ScriptedModel) -> ChatService:
    return ChatService(db, ClaudeClient(model, db, SETTINGS), SETTINGS)


async def test_normal_reply_is_stored_with_its_content_blocks(db):
    model = ScriptedModel(reply("Bonjour! How can I help?"))
    chat = make_service(db, model)

    session = await chat.create_session()
    await chat.send_message(session.id, "Hello there")
    await chat.wait_idle()

    view = await chat.get_session(session.id)
    user, assistant = view.messages
    assert (user.role, user.text, user.status) == ("user", "Hello there", "complete")
    assert user.content == [{"type": "text", "text": "Hello there"}]
    assert assistant.role == "assistant"
    assert assistant.status == "complete"
    assert assistant.stop_reason == "end_turn"
    assert assistant.text == "Bonjour! How can I help?"
    assert assistant.content == [
        {"type": "thinking", "thinking": "Thinking it over.", "signature": "sig-Thinking"},
        {"type": "text", "text": "Bonjour! How can I help?"},
    ]
    assert user.id < assistant.id
    assert user.created_at.utcoffset() == timedelta(0)


async def test_next_turn_replays_history_unchanged_with_opus_settings(db):
    model = ScriptedModel(reply("First answer."), reply("Second answer.", thinking="More thought."))
    chat = make_service(db, model)
    session = await chat.create_session()

    await chat.send_message(session.id, "Question one")
    await chat.wait_idle()
    await chat.send_message(session.id, "Question two")
    await chat.wait_idle()

    first, second = model.requests
    assert second["messages"][: len(first["messages"])] == first["messages"]
    assert second["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Question one"}]},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "Thinking it over.", "signature": "sig-Thinking"},
            {"type": "text", "text": "First answer."},
        ]},
        {"role": "user", "content": [{"type": "text", "text": "Question two"}]},
    ]
    assert second["model"] == "claude-opus-5-5"
    assert second["max_tokens"] == 64_000
    assert second["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert second["output_config"] == {"effort": "medium"}
    assert second["cache_control"] == {"type": "ephemeral"}
    assert second["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in second["betas"]
    assert "context_management" not in second  # no compaction
    assert second["system"] == first["system"]
    assert second["messages"][-1]["role"] == "user"  # never a prefill


async def test_stream_that_stops_halfway_keeps_partial_text_and_is_not_replayed(db):
    model = ScriptedModel([Delta("text", "Le passé comp"), api_error(None, "Connection reset")],
                          reply("Answer to the next question."))
    chat = make_service(db, model)
    session = await chat.create_session()

    await chat.send_message(session.id, "Explain the passé composé")
    await chat.wait_idle()
    view = await chat.get_session(session.id)
    broken = view.messages[-1]
    assert broken.status == "failed"
    assert broken.text == "Le passé comp"
    assert "Connection reset" in (broken.error or "")
    assert view.can_retry

    await chat.send_message(session.id, "Something else")
    await chat.wait_idle()
    sent = model.requests[-1]["messages"]
    assert [m["role"] for m in sent] == ["user", "user"]


async def test_error_then_retry_appends_a_new_reply_and_keeps_the_failed_one(db):
    model = ScriptedModel([api_error(529, "Overloaded")], reply("Here you go."))
    chat = make_service(db, model)
    session = await chat.create_session()

    await chat.send_message(session.id, "Tell me about transformers")
    await chat.wait_idle()
    await chat.retry(session.id)
    await chat.wait_idle()

    view = await chat.get_session(session.id)
    assert [(m.role, m.status) for m in view.messages] == [
        ("user", "complete"), ("assistant", "failed"), ("assistant", "complete")]
    assert view.messages[-1].text == "Here you go."
    assert model.requests[1]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Tell me about transformers"}]}]
    assert not view.can_retry
    with pytest.raises(NothingToRetry):
        await chat.retry(session.id)  # completed replies are never regenerated


async def test_refusal_is_labelled_and_left_out_of_later_requests(db):
    model = ScriptedModel(
        [Delta("text", "I can"), final("I can", thinking=None, stop_reason="refusal")],
        reply("Sure."))
    chat = make_service(db, model)
    session = await chat.create_session()

    await chat.send_message(session.id, "Risky question")
    await chat.wait_idle()
    refused = (await chat.get_session(session.id)).messages[-1]
    assert (refused.status, refused.stop_reason) == ("complete", "refusal")
    with pytest.raises(NothingToRetry):
        await chat.retry(session.id)

    await chat.send_message(session.id, "A different question")
    await chat.wait_idle()
    assert [m["role"] for m in model.requests[-1]["messages"]] == ["user", "user"]


async def test_length_cut_off_is_labelled_and_still_replayed(db):
    model = ScriptedModel(reply("A very long answer that", stop_reason="max_tokens"), reply("More."))
    chat = make_service(db, model)
    session = await chat.create_session()

    await chat.send_message(session.id, "Write a lot")
    await chat.wait_idle()
    cut = (await chat.get_session(session.id)).messages[-1]
    assert (cut.status, cut.stop_reason) == ("complete", "max_tokens")

    await chat.send_message(session.id, "Continue")
    await chat.wait_idle()
    assert [m["role"] for m in model.requests[-1]["messages"]] == ["user", "assistant", "user"]


async def test_restart_mid_reply_marks_it_interrupted_and_it_can_be_retried(db):
    gate = Gate()
    model = ScriptedModel([Delta("text", "Half an ans"), gate, final("never sent")])
    chat = make_service(db, model)
    session = await chat.create_session()
    await chat.send_message(session.id, "Long question")
    await gate.reached.wait()
    await chat.abandon()  # the server process dies mid-reply

    restarted = make_service(db, ScriptedModel(reply("Full answer.")))
    assert await restarted.recover_interrupted() == 1
    view = await restarted.get_session(session.id)
    interrupted = view.messages[-1]
    assert interrupted.status == "interrupted"
    assert interrupted.text == "Half an ans"
    assert view.can_retry

    await restarted.retry(session.id)
    await restarted.wait_idle()
    view = await restarted.get_session(session.id)
    assert [m.status for m in view.messages] == ["complete", "interrupted", "complete"]


async def test_one_reply_at_a_time_per_session(db):
    gate = Gate()
    model = ScriptedModel([gate, *reply("Done.")])
    chat = make_service(db, model)
    session = await chat.create_session()
    await chat.send_message(session.id, "First")
    await gate.reached.wait()

    with pytest.raises(ReplyInProgress):
        await chat.send_message(session.id, "Second, too soon")

    other = await chat.create_session()
    model.add(reply("Other session is fine."))
    await chat.send_message(other.id, "Hi")
    gate.opened.set()
    await chat.wait_idle()
    assert len((await chat.get_session(session.id)).messages) == 2


async def api_requests(db) -> list[ApiRequest]:
    async with db() as s:
        return list(await s.scalars(select(ApiRequest).order_by(ApiRequest.id)))


async def test_every_call_is_logged_with_tokens_and_cost_failed_ones_included(db):
    model = ScriptedModel(
        [api_error(529, "Overloaded", request_id="req_fail")],
        reply("Answer.", request_id="req_ok",
              usage_=usage(input_tokens=1_000, output_tokens=500, cache_read=10_000, cache_write=2_000)))
    chat = make_service(db, model)
    session = await chat.create_session()
    await chat.send_message(session.id, "Hi")
    await chat.wait_idle()
    await chat.retry(session.id)
    await chat.wait_idle()

    failed, ok = await api_requests(db)
    view = await chat.get_session(session.id)
    assert (failed.step, failed.session_id, failed.request_id) == ("chat", session.id, "req_fail")
    assert failed.message_id == view.messages[1].id
    assert "Overloaded" in (failed.error or "") and failed.cost_usd is None
    assert (ok.model_requested, ok.model_served, ok.stop_reason, ok.request_id) == (
        "claude-opus-5-5", "claude-opus-5-5", "end_turn", "req_ok")
    assert (ok.input_tokens, ok.output_tokens, ok.cache_read_tokens, ok.cache_write_tokens) == (
        1_000, 500, 10_000, 2_000)
    # 1000*$4 + 500*$20 + 10000*$0.20 + 2000*$5, per million tokens
    assert ok.cost_usd == Decimal("0.026000")
    assert ok.error is None


async def test_refusal_fallback_logs_the_served_model_and_prices_each_attempt(db):
    fallback_usage = {
        **usage(input_tokens=1_000, output_tokens=100),
        "iterations": [
            {"type": "message", "model": "claude-opus-5-5", "input_tokens": 1_000, "output_tokens": 0},
            {"type": "fallback_message", "model": "claude-opus-4-8", "input_tokens": 1_000,
             "output_tokens": 100},
        ],
    }
    model = ScriptedModel(reply("Answered by the fallback.", model="claude-opus-4-8",
                                usage_=fallback_usage))
    chat = make_service(db, model)
    session = await chat.create_session()
    await chat.send_message(session.id, "Hi")
    await chat.wait_idle()

    (logged,) = await api_requests(db)
    assert (logged.model_requested, logged.model_served) == ("claude-opus-5-5", "claude-opus-4-8")
    # $0.004 for the declined attempt + (1000*$5 + 100*$25)/1M for the fallback
    assert logged.cost_usd == Decimal("0.011500")


async def test_warns_to_start_a_new_session_at_150k_tokens(db):
    model = ScriptedModel(
        reply("Small.", usage_=usage(input_tokens=50_000, output_tokens=1_000)),
        reply("Big.", usage_=usage(input_tokens=500, output_tokens=1_000, cache_read=148_600)))
    chat = make_service(db, model)
    session = await chat.create_session()

    await chat.send_message(session.id, "One")
    await chat.wait_idle()
    view = await chat.get_session(session.id)
    assert (view.context_tokens, view.context_warning) == (51_000, False)

    await chat.send_message(session.id, "Two")
    await chat.wait_idle()
    view = await chat.get_session(session.id)
    assert (view.context_tokens, view.context_warning) == (150_100, True)


async def test_session_is_named_from_my_first_message_and_can_be_renamed(db):
    model = ScriptedModel(reply("A."), reply("B."))
    chat = make_service(db, model)
    session = await chat.create_session()
    first = ("How does attention work in transformers, and why does it scale "
             "quadratically with sequence length?")
    await chat.send_message(session.id, first)
    await chat.wait_idle()
    await chat.send_message(session.id, "Second message does not rename it")
    await chat.wait_idle()

    named = (await chat.get_session(session.id)).session
    assert named.name == "How does attention work in transformers, and why does it…"
    assert named.source == "app" and named.subject is None

    await chat.rename_session(session.id, "Attention")
    assert (await chat.get_session(session.id)).session.name == "Attention"


async def test_reply_rescued_mid_stream_by_fallback_replays_without_the_declined_thinking(db):
    rescued = final("", thinking=None, model="claude-opus-4-8")
    rescued.content = [
        {"type": "thinking", "thinking": "Declined model's thought.", "signature": "sig-a"},
        {"type": "text", "text": "Partial "},
        {"type": "fallback", "from": {"model": "claude-opus-5-5"}, "to": {"model": "claude-opus-4-8"}},
        {"type": "text", "text": "rest of the answer."},
    ]
    model = ScriptedModel([Delta("text", "Partial "), rescued], reply("Next."))
    chat = make_service(db, model)
    session = await chat.create_session()
    await chat.send_message(session.id, "Q1")
    await chat.wait_idle()
    stored = (await chat.get_session(session.id)).messages[-1]
    assert stored.content == rescued.content  # stored exactly as returned
    assert stored.text == "Partial rest of the answer."

    await chat.send_message(session.id, "Q2")
    await chat.wait_idle()
    assert model.requests[-1]["messages"][1]["content"] == rescued.content[1:]


async def test_unexpected_client_errors_fail_the_reply_and_are_still_logged(db):
    model = ScriptedModel([TypeError("Could not resolve authentication method")])
    chat = make_service(db, model)
    session = await chat.create_session()
    await chat.send_message(session.id, "Hi")
    await chat.wait_idle()

    view = await chat.get_session(session.id)
    assert view.messages[-1].status == "failed"
    assert view.can_retry
    (logged,) = await api_requests(db)
    assert "authentication" in (logged.error or "")
    assert logged.message_id == view.messages[-1].id
