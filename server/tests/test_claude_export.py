"""The Claude export parser, on the made-up example export."""

from pathlib import Path

from evaluation.claude_export import parse

EXAMPLE = Path(__file__).resolve().parents[2] / "eval" / "example_claude_export.json"


def test_keeps_only_my_messages_and_claudes_text_replies():
    [conversation] = parse(EXAMPLE)  # the empty conversation is left out
    assert (conversation.id, conversation.name) == ("conv-0001", "What is RAG?")
    assert [(m.id, m.role, m.text) for m in conversation.messages] == [
        ("msg-0001", "user", "What is retrieval-augmented generation?"),
        ("msg-0002", "assistant", "It retrieves relevant documents first, then answers using them."),
        ("msg-0004", "user", "Thanks! And how do I evaluate it?"),
    ]
    assert conversation.messages[0].created_at.isoformat() == "2026-10-01T09:00:00+00:00"
