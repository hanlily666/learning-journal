"""The extraction evaluation, step by step. Run from server/:

    uv run python -m evaluation list                    # conversations in my Claude export
    uv run python -m evaluation sample ID... --tune ID... --split ID...
    #   then fill in eval/data/labels.json (label and items for each session)
    uv run python -m evaluation run [--set tune]        # the real pipeline, in journal_eval
    uv run python -m evaluation baseline [--set tune]   # one structured request per session
    uv run python -m evaluation judge eval/data/runs/<run>

Everything it reads and writes is under the repo's eval/ folder, which is never committed.
Each paid step prints an estimate and asks first; results that didn't change are reused.
"""

import argparse
import asyncio
import hashlib
import json
import re
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import anthropic
import psycopg
from sqlalchemy import func, select
from sqlalchemy.engine import make_url

from app.config import Settings, get_settings
from app.db import make_engine, make_sessionmaker
from app.extraction import (AGENT_SYSTEM, CLASSIFY_SYSTEM, ExtractionService, parse_label,
                            segment_messages)
from app.models import ApiRequest, ChatSession, Message, Segment
from app.pricing import PRICES, cost_usd
from evaluation.claude_export import parse
from evaluation.scoring import Judged, score

EVAL = Path(__file__).resolve().parents[2] / "eval"
EXPORT = EVAL / "exports" / "claude" / "conversations.json"
DATA = EVAL / "data"
SESSIONS = DATA / "sessions.json"
LABELS = DATA / "labels.json"
READABLE = DATA / "read"  # the sampled sessions as Markdown, to read while labelling
# Conversations used as examples in the selection skill: never sampled for the evaluation.
SKILL_EXAMPLES = EVAL / "skill_examples.txt"
EVAL_DB = "journal_eval"
JUDGE_MODEL = "claude-haiku-5-5"
BATCH_DISCOUNT = 0.5

# Rough output sizes for the estimate (tokens). Input is counted as characters / 3.
EST_OUT = {"classification": 300, "extraction": 3000, "baseline": 3000, "judge": 600}
EXTRACTION_SOURCE = Path(__file__).resolve().parents[1] / "app" / "extraction.py"
SKILL = Path(__file__).resolve().parents[1] / "app" / "skills" / "ai_selection.md"

BASELINE_SYSTEM = AGENT_SYSTEM.replace(
    "Finish by calling submit_items exactly once. If it returns an error, fix the input and call it again.",
    "Answer with the subject, summary, and items as JSON.").replace(
    " Earlier parts of the session were processed already: their summaries are given for context, "
    "and read_session returns earlier messages if you need them, but don't extract anything that "
    "appears only there.", "")

assert "submit_items" not in BASELINE_SYSTEM and "read_session" not in BASELINE_SYSTEM, \
    "AGENT_SYSTEM was reworded: update how BASELINE_SYSTEM is derived from it"

BASELINE_FORMAT = {"type": "json_schema", "schema": {
    "type": "object", "additionalProperties": False, "required": ["subject", "summary", "items"],
    "properties": {
        "subject": {"type": "string"}, "summary": {"type": "string"},
        "items": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["field", "text", "reason"],
            "properties": {"field": {"type": "string", "enum": ["explore", "source", "gap", "concept"]},
                           "text": {"type": "string"}, "reason": {"type": "string"}}}}}}}

JUDGE_SYSTEM = """You grade extracted learning notes against a person's own labels.

Match each extracted item to the labelled item that says essentially the same thing, in the same field. Each extracted item and each label can be matched at most once; leave anything without a real counterpart unmatched.

Also list pairs of extracted items that say the same thing as each other (duplicates)."""

JUDGE_FORMAT = {"type": "json_schema", "schema": {
    "type": "object", "additionalProperties": False, "required": ["matches", "duplicates"],
    "properties": {
        "matches": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["extracted", "label"],
            "properties": {"extracted": {"type": "integer"}, "label": {"type": "integer"}}}},
        "duplicates": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["a", "b"],
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}}}}}


# Files

def read(path: Path) -> Any:
    return json.loads(path.read_text())


def write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")


def chosen(set_name: str) -> list[dict[str, Any]]:
    sessions = read(SESSIONS)
    return sessions if set_name == "all" else [s for s in sessions if s["set"] == set_name]


def cache_key(kind: str, session: dict[str, Any], *parts: Any) -> str:
    blob = json.dumps([kind, session["messages"], session.get("split_at"), *parts], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def new_run_dir(kind: str) -> Path:
    base = DATA / "runs" / f"{kind}-{datetime.now():%Y%m%d-%H%M%S}"
    path, n = base, 1
    while path.exists():
        n += 1
        path = base.with_name(f"{base.name}-{n}")
    return path


# Cost

def tokens(text: str) -> int:
    return len(text) // 3 + 1


def dollars(model: str, input_tokens: int, output_tokens: int, batch: bool) -> float:
    p_in, p_out = PRICES[model][:2]
    cost = (input_tokens * p_in + output_tokens * p_out) / 1_000_000
    return cost * (BATCH_DISCOUNT if batch else 1)


def confirm(estimate: float, what: str, yes: bool) -> bool:
    print(f"Estimated cost of {what}: about ${estimate:.2f} (a rough guess from text length).")
    if yes:
        return True
    return input("Go ahead? [y/N] ").strip().lower() == "y"


def batch_cost(message: Any) -> float:
    return (cost_usd(message.usage.to_dict(), message.model) or 0.0) * BATCH_DISCOUNT


def spend(messages: list[Any]) -> dict[str, Any]:
    """What a batch cost, and how much of its input was read from the prompt cache."""
    return {"usd": round(sum(batch_cost(m) for m in messages), 6),
            "cache_read_tokens": sum(m.usage.cache_read_input_tokens or 0 for m in messages)}


# Message Batches

async def run_batch(client: anthropic.AsyncAnthropic, requests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Send requests as one batch (half price), wait for it, and return messages by ID."""
    if not requests:
        return {}
    batch = await client.messages.batches.create(requests=cast(Any, [
        {"custom_id": key, "params": {k: v for k, v in params.items() if k not in ("betas", "fallbacks")}}
        for key, params in requests.items()]))
    print(f"Batch {batch.id}: {len(requests)} requests sent; checking every 30 seconds.")
    while batch.processing_status != "ended":
        await asyncio.sleep(30)
        batch = await client.messages.batches.retrieve(batch.id)
        print(f"  {batch.request_counts.processing} still processing")
    results: dict[str, Any] = {}
    async for result in await client.messages.batches.results(batch.id):
        if result.result.type == "succeeded":
            results[result.custom_id] = result.result.message
        else:
            print(f"  {result.custom_id}: {result.result.type}")
    return results


# list and sample

def cmd_list(_: argparse.Namespace) -> None:
    for c in parse(EXPORT):
        print(f"{c.id}  {c.created_at:%Y-%m-%d}  {len(c.messages):3d} messages  {c.name}")


def skill_example_ids() -> set[str]:
    """One conversation ID per line; text after # is a note."""
    if not SKILL_EXAMPLES.exists():
        return set()
    return {line.split("#")[0].strip() for line in SKILL_EXAMPLES.read_text().splitlines()} - {""}


def cmd_sample(args: argparse.Namespace) -> None:
    by_id = {c.id: c for c in parse(EXPORT)}
    missing = [i for i in args.ids if i not in by_id]
    if missing:
        raise SystemExit(f"Not in the export: {missing}")
    used = [i for i in args.ids if i in skill_example_ids()]
    if used:
        raise SystemExit(f"Used as examples in the skill ({SKILL_EXAMPLES}): {used}")
    sessions = []
    for i in args.ids:
        c = by_id[i]
        split_at = None
        if i in args.split:
            # The second part starts with my message, nearest the middle.
            starts = [n for n, m in enumerate(c.messages) if m.role == "user" and n > 0]
            if not starts:
                raise SystemExit(f"{i} has no later message of mine to split at")
            split_at = min(starts, key=lambda n: abs(n - len(c.messages) // 2))
        sessions.append({"id": c.id, "name": c.name, "set": "tune" if i in args.tune else "test",
                         "split_at": split_at,
                         "messages": [asdict(m) for m in c.messages]})
    write(SESSIONS, sessions)
    old = {l["id"]: l for l in read(LABELS)} if LABELS.exists() else {}
    write(LABELS, [old.get(s["id"]) or {"id": s["id"], "name": s["name"],
                                        "label": "ai | french | messy",
                                        "items": [{"field": "concept | gap | explore | source",
                                                   "text": ""}]}
                   for s in sessions])
    write_readable(sessions)
    print(f"Wrote {len(sessions)} sessions to {SESSIONS}; label them in {LABELS}.")
    print(f"Read them in {READABLE}.")


def write_readable(sessions: list[dict[str, Any]]) -> None:
    """One Markdown file per session, named by set and numbered in labels.json order."""
    shutil.rmtree(READABLE, ignore_errors=True)
    READABLE.mkdir(parents=True)
    for n, s in enumerate(sessions, 1):
        slug = re.sub(r"[^\w]+", "-", s["name"]).strip("-")[:50] or "untitled"
        lines = [f"# {s['name']}", "", f"- id: `{s['id']}`", f"- set: {s['set']}"]
        if s["split_at"] is not None:
            lines.append(f"- split: processed in two parts; part 2 starts at message {s['split_at'] + 1}")
        for i, m in enumerate(s["messages"]):
            if i == s["split_at"]:
                lines += ["", "---", "", "**[ part 2 starts here ]**"]
            lines += ["", f"## {i + 1}. {m['role']} ({m['created_at'][:16]})", "", m["text"]]
        (READABLE / f"{s['set']}-{n:02d}-{slug}.md").write_text("\n".join(lines) + "\n")


def cmd_read(_: argparse.Namespace) -> None:
    write_readable(read(SESSIONS))
    print(f"Wrote {len(read(SESSIONS))} sessions to {READABLE}.")


# run: the real pipeline in journal_eval

def eval_url(settings: Settings) -> str:
    url = make_url(settings.database_url).set(database=EVAL_DB)
    return url.render_as_string(hide_password=False)


def reset_eval_db(settings: Settings) -> str:
    admin = make_url(settings.database_url).set(drivername="postgresql", database="postgres")
    with psycopg.connect(admin.render_as_string(hide_password=False), autocommit=True) as conn:
        conn.execute(f"DROP DATABASE IF EXISTS {EVAL_DB} WITH (FORCE)")
        conn.execute(f"CREATE DATABASE {EVAL_DB}")
    from alembic import command
    from alembic.config import Config
    server = Path(__file__).resolve().parents[1]
    cfg = Config(str(server / "alembic.ini"))
    cfg.set_main_option("script_location", str(server / "migrations"))
    url = eval_url(settings)
    cfg.attributes["database_url"] = url
    cfg.attributes["configure_logging"] = False
    command.upgrade(cfg, "head")
    return url


async def add_messages(sessionmaker, session_ids: dict[str, int], session: dict[str, Any],
                       messages: list[dict[str, Any]]) -> None:
    async with sessionmaker() as db, db.begin():
        if session["id"] not in session_ids:
            row = ChatSession(name=session["name"], source="claude")
            db.add(row)
            await db.flush()
            session_ids[session["id"]] = row.id
        for m in messages:
            db.add(Message(session_id=session_ids[session["id"]], role=m["role"], text=m["text"],
                           content=[{"type": "text", "text": m["text"]}], status="complete",
                           source="claude", external_id=m["id"]))


async def classify_pending(service: ExtractionService, sessionmaker) -> list[Any]:
    """Label every pending segment through one batch; returns the batch's replies."""
    async with sessionmaker() as db:
        segments = list(await db.scalars(select(Segment).where(Segment.status == "pending",
                                                               Segment.label.is_(None))))
        requests = {f"seg{s.id}": service.classification_request(await segment_messages(db, s))
                    for s in segments}
    results = await run_batch(service.client, requests)
    for s in segments:
        message = results.get(f"seg{s.id}")
        if message is not None:  # a failed one is classified by the worker itself
            label, reason = parse_label(message)
            await service.save_label(s, label, reason)
    return list(results.values())


async def cmd_run(args: argparse.Namespace) -> None:
    settings = get_settings()
    skill = SKILL.read_text()
    sessions = chosen(args.set)
    # The whole extraction module is hashed: prompts, tool descriptions, and limits.
    keys = {s["id"]: cache_key("agent", s, skill, EXTRACTION_SOURCE.read_text(),
                               settings.claude_model, settings.claude_effort) for s in sessions}
    cache = DATA / "cache" / "agent"
    todo = [s for s in sessions if not (cache / f"{keys[s['id']]}.json").exists()]
    print(f"{len(sessions) - len(todo)} of {len(sessions)} sessions reused from earlier runs.")
    estimate = sum(
        dollars(settings.claude_model, tokens(json.dumps(s["messages"])) * (1 if s["split_at"] is None else 2),
                EST_OUT["classification"], batch=True)
        + dollars(settings.claude_model, 2 * (tokens(json.dumps(s["messages"])) + tokens(skill) + 1000),
                  EST_OUT["extraction"], batch=False)
        for s in todo)
    if todo and not confirm(estimate, f"{len(todo)} sessions", args.yes):
        return

    classified: list[Any] = []
    agent: dict[str, Any] = {"usd": 0.0, "cache_read_tokens": 0}
    results: dict[str, Any] = {}
    if todo:
        url = reset_eval_db(settings)
        engine = make_engine(url)
        sessionmaker = make_sessionmaker(engine)
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        service = ExtractionService(sessionmaker, client, settings, skill=skill)
        ids: dict[str, int] = {}
        # First parts (whole sessions unless split), then the rest of the split sessions:
        # the second pass is a session continued later.
        for part in (0, 1):
            for s in todo:
                cut = s["split_at"] or len(s["messages"])
                messages = s["messages"][:cut] if part == 0 else s["messages"][cut:]
                if messages:
                    await add_messages(sessionmaker, ids, s, messages)
            await service.queue_segments()
            classified += await classify_pending(service, sessionmaker)
            service.kick()
            await service.wait_idle()

        external = {}
        async with sessionmaker() as db:
            for m in await db.scalars(select(Message)):
                external[m.id] = m.external_id
            spent = (await db.execute(select(func.sum(ApiRequest.cost_usd),
                                             func.sum(ApiRequest.cache_read_tokens)))).one()
        views = {v.session.id: v for v in await service.extracted()}
        for s in todo:
            view = views.get(ids[s["id"]])
            segments = [] if view is None else [{
                "label": seg.label, "status": seg.status, "error": seg.error,
                "subject": seg.subject, "summary": seg.summary,
                "items": [{"field": i.field, "text": i.text, "reason": i.reason,
                           "sources": [external[m] for m in i.source_message_ids]}
                          for i in seg.items]} for seg in view.segments]
            results[s["id"]] = {"segments": segments}
            write(cache / f"{keys[s['id']]}.json", results[s["id"]])
        await engine.dispose()
        agent = {"usd": float(spent[0] or 0), "cache_read_tokens": int(spent[1] or 0)}
    record = {"new_sessions": len(todo), "estimate_usd": round(estimate, 4),
              "agent": agent, "classification_batch": spend(classified)}
    if todo:
        print(f"Spent: agent ${agent['usd']:.4f} ({agent['cache_read_tokens']} cache-read tokens), "
              f"batched classification ${record['classification_batch']['usd']:.4f}.")
    for s in sessions:
        results.setdefault(s["id"], read(cache / f"{keys[s['id']]}.json"))
    run_dir = new_run_dir("agent")
    write(run_dir / "results.json", results)
    write(run_dir / "cost.json", record)
    print(f"Results: {run_dir / 'results.json'}")


# baseline: one structured request per session, with the same skill

async def cmd_baseline(args: argparse.Namespace) -> None:
    settings = get_settings()
    skill = SKILL.read_text()
    labels = {l["id"]: l["label"] for l in read(LABELS)}
    sessions = [s for s in chosen(args.set) if labels.get(s["id"]) == "ai"]
    system = BASELINE_SYSTEM.format(skill=skill.strip())
    keys = {s["id"]: cache_key("baseline", s, system, settings.claude_model) for s in sessions}
    cache = DATA / "cache" / "baseline"
    todo = [s for s in sessions if not (cache / f"{keys[s['id']]}.json").exists()]
    print(f"{len(sessions) - len(todo)} of {len(sessions)} AI sessions reused from earlier runs.")
    estimate = sum(dollars(settings.claude_model, tokens(json.dumps(s["messages"])) + tokens(system),
                           EST_OUT["baseline"], batch=True) for s in todo)
    if todo and not confirm(estimate, f"{len(todo)} baseline requests", args.yes):
        return
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    requests = {}
    for n, s in enumerate(todo):
        transcript = "\n".join(f'<message id="{m["id"]}" role="{m["role"]}">\n{m["text"]}\n</message>'
                               for m in s["messages"])
        requests[f"s{n}"] = {
            "model": settings.claude_model, "max_tokens": 16_000, "system": system,
            "thinking": {"type": "adaptive"}, "cache_control": {"type": "ephemeral"},
            "output_config": {"effort": settings.claude_effort, "format": BASELINE_FORMAT},
            "messages": [{"role": "user", "content": f"<new_messages>\n{transcript}\n</new_messages>"}]}
    messages = await run_batch(client, requests)
    results: dict[str, Any] = {}
    for n, s in enumerate(todo):
        message = messages.get(f"s{n}")
        if message is None:
            continue
        answer = json.loads(next(b.text for b in message.content if b.type == "text"))
        results[s["id"]] = {"segments": [{"label": "ai", "status": "done", **answer}]}
        write(cache / f"{keys[s['id']]}.json", results[s["id"]])
    for s in sessions:
        path = cache / f"{keys[s['id']]}.json"
        if path.exists():
            results.setdefault(s["id"], read(path))
    run_dir = new_run_dir("baseline")
    write(run_dir / "results.json", results)
    cost = {"new_sessions": len(todo), "estimate_usd": round(estimate, 4),
            "baseline_batch": spend(list(messages.values()))}
    write(run_dir / "cost.json", cost)
    print(f"Spent ${cost['baseline_batch']['usd']:.4f} "
          f"({cost['baseline_batch']['cache_read_tokens']} cache-read tokens). "
          f"Results: {run_dir / 'results.json'}")


# judge: match items to my labels, then score

def numbered(items: list[dict[str, Any]]) -> str:
    return "\n".join(f'{n}. [{i["field"]}] {i["text"]}' for n, i in enumerate(items)) or "(none)"


async def cmd_judge(args: argparse.Namespace) -> None:
    settings = get_settings()
    run_dir = Path(args.run)
    results = read(run_dir / "results.json")
    labels = {l["id"]: l for l in read(LABELS)}
    unlabelled = [i for i in results if labels.get(i, {}).get("label") not in ("ai", "french", "messy")]
    if unlabelled:
        raise SystemExit(f"Label these sessions in {LABELS} first: {unlabelled}")

    predicted = {i: [{**item, "segment": n} for n, seg in enumerate(r["segments"])
                     for item in seg.get("items", [])] for i, r in results.items()}
    cache = DATA / "cache" / "judge"
    answers: dict[str, Any] = {}
    requests, keys, order = {}, {}, list(results)
    for n, i in enumerate(order):
        mine = [x for x in labels[i]["items"] if x.get("text")]
        if not predicted[i] or not (mine or len(predicted[i]) > 1):
            continue
        content = (f"<labels>\n{numbered(mine)}\n</labels>\n\n"
                   f"<extracted>\n{numbered(predicted[i])}\n</extracted>")
        keys[i] = hashlib.sha256(json.dumps([JUDGE_SYSTEM, JUDGE_MODEL, content]).encode()).hexdigest()[:24]
        if (cache / f"{keys[i]}.json").exists():  # same items, same labels: reuse the verdict
            answers[i] = read(cache / f"{keys[i]}.json")
            continue
        requests[f"s{n}"] = {
            "model": JUDGE_MODEL, "max_tokens": 4000, "system": JUDGE_SYSTEM,
            "cache_control": {"type": "ephemeral"},
            "output_config": {"effort": "low", "format": JUDGE_FORMAT},
            "messages": [{"role": "user", "content": content}]}
    estimate = sum(dollars(JUDGE_MODEL, tokens(json.dumps(r["messages"])), EST_OUT["judge"], batch=True)
                   for r in requests.values())
    if requests and not confirm(estimate, f"{len(requests)} judge requests", args.yes):
        return
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    messages = await run_batch(client, requests)

    judged, spot_check = [], []
    for n, i in enumerate(order):
        mine = [x for x in labels[i]["items"] if x.get("text")]
        answer = answers.get(i, {"matches": [], "duplicates": []})
        if (message := messages.get(f"s{n}")) is not None:
            answer = json.loads(next(b.text for b in message.content if b.type == "text"))
            write(cache / f"{keys[i]}.json", answer)
        segments = results[i]["segments"]
        judged.append(Judged(
            session_id=i, expected_label=labels[i]["label"],
            # The baseline doesn't classify: its segments carry no label of their own.
            predicted_labels=[] if run_dir.name.startswith("baseline") else
            [seg["label"] for seg in segments if seg["label"]],
            labels=mine, predicted=predicted[i],
            matches=[(m["extracted"], m["label"]) for m in answer["matches"]],
            duplicates=[(d["a"], d["b"]) for d in answer["duplicates"]]))
        spot_check.append({"session": i, "matches": [
            {"extracted": predicted[i][m["extracted"]]["text"], "label": mine[m["label"]]["text"]}
            for m in answer["matches"]
            if 0 <= m["extracted"] < len(predicted[i]) and 0 <= m["label"] < len(mine)]})
    report = score(judged)
    report["judge"] = {**spend(list(messages.values())), "reused": len(answers)}
    cost = run_dir / "cost.json"
    if cost.exists():
        report["run_cost"] = read(cost)
    write(run_dir / "report.json", report)
    write(run_dir / "judge_matches.json", spot_check)
    print(json.dumps(report, indent=2))
    print(f"Spot-check the judge's matches in {run_dir / 'judge_matches.json'}.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m evaluation", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)
    sub.add_parser("list", help="list the conversations in the export").set_defaults(func=cmd_list)
    p = sub.add_parser("sample", help="choose the sessions to label")
    p.add_argument("ids", nargs="+")
    p.add_argument("--tune", nargs="*", default=[], help="the small set to tune on")
    p.add_argument("--split", nargs="*", default=[], help="sessions to process in two parts")
    p.set_defaults(func=cmd_sample)
    sub.add_parser("read", help="write the sampled sessions as Markdown to read").set_defaults(func=cmd_read)
    for name, func_, help_ in (("run", cmd_run, "run the pipeline (classification + agent)"),
                               ("baseline", cmd_baseline, "run the single-request baseline")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--set", choices=["tune", "test", "all"], default="tune")
        p.add_argument("--yes", action="store_true", help="don't ask before spending")
        p.set_defaults(func=func_)
    p = sub.add_parser("judge", help="score a run against my labels")
    p.add_argument("run", help="a folder under eval/data/runs")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_judge)
    args = parser.parse_args()
    result = args.func(args)
    if asyncio.iscoroutine(result):
        asyncio.run(result)


if __name__ == "__main__":
    main()
