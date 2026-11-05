"""The HTTP API end to end, with the scripted model in place of Claude."""

import json

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.scripted import ScriptedModel, reply


def read_events(response) -> list[tuple[str | None, dict]]:
    events, name = [], None
    for line in response.iter_lines():
        if line.startswith("event:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            events.append((name, json.loads(line.split(":", 1)[1].strip())))
    return events


def test_chat_over_http_streams_the_reply_and_stores_it(db, database_url):
    settings = Settings(database_url=database_url, partial_save_interval=0)
    with TestClient(create_app(settings, ScriptedModel(reply("Salut! Ça va?")))) as client:
        session = client.post("/api/sessions", json={}).json()
        sent = client.post(f"/api/sessions/{session['id']}/messages", json={"text": "Bonjour"})
        assert sent.status_code == 202
        reply_id = sent.json()["assistant_message"]["id"]
        assert sent.json()["assistant_message"]["status"] == "streaming"

        with client.stream("GET", f"/api/messages/{reply_id}/events") as response:
            events = read_events(response)
        name, done = events[-1]
        assert name == "done"
        assert (done["status"], done["text"]) == ("complete", "Salut! Ça va?")

        view = client.get(f"/api/sessions/{session['id']}").json()
        assert [m["text"] for m in view["messages"]] == ["Bonjour", "Salut! Ça va?"]
        assert view["session"]["name"] == "Bonjour"
        assert client.post(f"/api/sessions/{session['id']}/retry").status_code == 409
        assert client.get("/api/sessions/999").status_code == 404
        assert [s["id"] for s in client.get("/api/sessions").json()] == [session["id"]]
