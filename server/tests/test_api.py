"""The HTTP API end to end, with the scripted model in place of Claude."""

import json
import time

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests import fake_api as api
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


def wait_for(client, check, seconds=5.0):
    """Poll the Extracted page until `check` passes (extraction runs in the background)."""
    deadline = time.monotonic() + seconds
    while True:
        extracted = client.get("/api/extracted").json()
        if check(extracted) or time.monotonic() > deadline:
            return extracted
        time.sleep(0.02)


def test_new_session_and_process_now_extract_over_http(db, database_url):
    settings = Settings(database_url=database_url, partial_save_interval=0)
    model = ScriptedModel(reply("RAG retrieves documents first."))
    fake = api.FakeAPI()
    with TestClient(create_app(settings, model, fake.client)) as client:
        ai = client.post("/api/sessions", json={}).json()
        sent = client.post(f"/api/sessions/{ai['id']}/messages", json={"text": "What is RAG?"}).json()
        with client.stream("GET", f"/api/messages/{sent['assistant_message']['id']}/events") as r:
            read_events(r)
        q = sent["user_message"]["id"]

        # The agent fails twice, so the segment needs attention.
        fake.add(api.label("ai"), api.text("Done."), api.text("Done."))
        assert client.post("/api/sessions", json={}).status_code == 201
        [view] = wait_for(client, lambda e: e and e[0]["segments"][0]["status"] == "needs_attention")
        segment = view["segments"][0]
        assert (view["session"]["id"], view["session"]["subject"]) == (ai["id"], "ai")

        fake.add(api.submit("RAG", "What RAG is.", api.item("concept", "RAG", [q])))
        assert client.post(f"/api/segments/{segment['id']}/retry").status_code == 202
        [view] = wait_for(client, lambda e: e[0]["segments"][0]["status"] == "done")
        assert [i["text"] for i in view["segments"][0]["items"]] == ["RAG"]
        assert client.post(f"/api/segments/{segment['id']}/retry").status_code == 409
        assert client.post("/api/segments/999/retry").status_code == 404

        assert client.post("/api/extraction/run").json() == {"queued": 0}
