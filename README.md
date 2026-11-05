# Learning Journal

Chat with Claude in one place, and (in later phases) turn those chats into a journal, a French
dictionary, and a daily recap with to-dos.

This is **Phase 1: chat and storage**. It's a plain chat app that never loses a message. Every
message is stored with its ID and UTC timestamp, and you can close the app and resume a session
days later.

- `server/` is the Python API (FastAPI, SQLAlchemy 2, Alembic, Pydantic) with Postgres.
- `web/` is the browser app (Vite, React, TypeScript), with types generated from the API's OpenAPI schema.

## Run it locally

You need [Docker](https://docs.docker.com/get-docker/), [uv](https://docs.astral.sh/uv/),
Node.js 20 or later, and your own [Anthropic API key](https://console.anthropic.com/).

```bash
# 1. Settings: copy the example and add your API key
cp .env.example .env

# 2. Database: Postgres 17 in Docker
docker compose up -d

# 3. Server: install, create the tables, start the API on :8000
cd server
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload

# 4. Browser app, in a second terminal: open http://localhost:5173
cd web
npm install
npm run dev
```

To serve everything from one process instead, run `npm run build` in `web/` and open
http://localhost:8000. The server serves `web/dist` when it exists.

Stop the database with `docker compose down`. Your data stays in the `pgdata` volume until you
run `docker compose down -v`.

### Using Supabase instead of Docker

Create a project on the free plan, then set `DATABASE_URL` in `.env` to its **direct connection**
string (Project Settings → Database). Run `uv run alembic upgrade head` once. Every table has
Row Level Security turned on with no policies, so Supabase's public API can't read anything. Only
this server, which connects as the table owner, can. Free projects pause after a week without use,
and they have no backups, which is why the backup script below exists.

## How chat works

- **Model:** Claude Opus 5.5 at medium effort, with a collapsible thinking summary, a reply limit
  of 64K tokens, and automatic prompt caching. Refusals use the API's server-side fallback
  (`fallbacks: "default"`). Replies that were cut off at the length limit or refused are labelled.
- **Append-only history:** messages are never edited, regenerated, or deleted. A database trigger
  enforces this. A failed or interrupted reply can be retried, and the retry is added as a new
  message. Each message's content blocks, thinking included, are stored exactly as the API
  returned them and sent back unchanged. Only complete messages are sent.
- **Replies run on the server:** each reply runs as its own server task, so it keeps saving if you
  close the browser. A session runs one reply at a time. Your message is saved before the API call.
  On startup, any reply that was still streaming when the server stopped is marked interrupted.
- **Long sessions:** at about 150K tokens the app suggests starting a new session. Nothing is
  compacted.
- **Cost log:** every API call, failed ones included, is logged in the `api_request` table, with
  the step, session, model requested and served, token counts, stop reason, request ID, and cost.

Run the server as a single process (one uvicorn worker), because live reply streaming is kept
in memory.

## Tests

```bash
cd server
uv run pytest          # starts an embedded Postgres; or set TEST_DATABASE_URL to an empty database
uv run pyright         # type check

cd ../web
npm run typecheck && npm run lint
```

The tests never call the real API. Every Claude call goes through one wrapper
(`server/app/claude.py`), and the tests replace its transport with a **scripted model**
(`server/tests/scripted.py`) that replays prepared replies. The scripted scenarios are:

- a normal reply
- a stream that stops halfway
- an error, then a retry
- a refusal
- a length-limit cut-off
- a restart mid-reply

The real transport is tested against canned HTTP responses.

After changing the API, regenerate the browser's types:

```bash
cd server && uv run python -m scripts.export_openapi
cd ../web && npm run gen:api
```

## Backups

`scripts/backup.sh` runs `pg_dump` against `DATABASE_URL` and writes a compressed dump to a
private folder. Set `BACKUP_DIR` in `.env`; the default is `~/learning-journal-backups`. Every
dump is kept unless you set `BACKUP_KEEP_DAYS`. To run it daily, add a cron entry (`crontab -e`):

```
0 21 * * * cd /path/to/repo && scripts/backup.sh >> "$HOME/learning-journal-backups/backup.log" 2>&1
```

You need a `pg_dump` from Postgres 17 or later (`brew install postgresql@17`). Or borrow
Docker's: for the local database use `PG_DUMP="docker compose exec -T db pg_dump"`, and for
Supabase use `PG_DUMP="docker run --rm -i postgres:17 pg_dump"`. Restore with
`pg_restore --no-owner --dbname "$DATABASE_URL" <file>.dump`.

## Not in Phase 1

The following are not built yet: assistant tools, extraction, the journal page, deployment, and
login. Hosting on Railway and Supabase Auth come after Phase 1, and nothing is deployed before
login exists.
