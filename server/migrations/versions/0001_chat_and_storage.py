"""Phase 1: sessions, messages, and the API request log.

Revision ID: 0001
Revises:
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE session (
            id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            name        text,
            source      text NOT NULL DEFAULT 'app',
            subject     text CHECK (subject IN ('ai', 'french', 'mixed')),
            checkpoint  bigint,
            created_at  timestamptz NOT NULL DEFAULT now(),
            updated_at  timestamptz NOT NULL DEFAULT now()
        );

        CREATE TABLE message (
            id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            session_id        bigint NOT NULL REFERENCES session(id),
            role              text NOT NULL CHECK (role IN ('user', 'assistant')),
            content           jsonb NOT NULL DEFAULT '[]'::jsonb,
            text              text NOT NULL DEFAULT '',
            -- clock_timestamp(): the moment each message is saved, even within one transaction.
            created_at        timestamptz NOT NULL DEFAULT clock_timestamp(),
            status            text NOT NULL
                              CHECK (status IN ('streaming', 'complete', 'interrupted', 'failed')),
            stop_reason       text,
            error             text,
            injected_context  boolean NOT NULL DEFAULT false,
            source            text NOT NULL DEFAULT 'app',
            external_id       text,
            CONSTRAINT message_external_id_unique UNIQUE (source, external_id)
        );
        CREATE INDEX message_session_idx ON message (session_id, id);
        -- One reply at a time per session.
        CREATE UNIQUE INDEX message_one_streaming_reply
            ON message (session_id) WHERE status = 'streaming';

        CREATE TABLE api_request (
            id                  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            created_at          timestamptz NOT NULL DEFAULT now(),
            step                text NOT NULL,
            session_id          bigint REFERENCES session(id),
            message_id          bigint REFERENCES message(id),
            model_requested     text NOT NULL,
            model_served        text,
            input_tokens        bigint,
            output_tokens       bigint,
            cache_read_tokens   bigint,
            cache_write_tokens  bigint,
            stop_reason         text,
            request_id          text,
            cost_usd            numeric(12, 6),
            error               text,
            usage               jsonb
        );
        CREATE INDEX api_request_session_idx ON api_request (session_id, id);

        -- History is append-only. A message can be written while it streams; once it
        -- leaves 'streaming' its content is frozen, and no message is ever deleted.
        CREATE FUNCTION message_append_only() RETURNS trigger AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION 'messages are append-only and cannot be deleted';
            END IF;
            IF NEW.session_id <> OLD.session_id OR NEW.role <> OLD.role
               OR NEW.created_at <> OLD.created_at THEN
                RAISE EXCEPTION 'message % cannot be moved or re-dated', OLD.id;
            END IF;
            IF OLD.status <> 'streaming' AND NEW IS DISTINCT FROM OLD THEN
                RAISE EXCEPTION 'message % is finished and cannot be edited', OLD.id;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        CREATE TRIGGER message_append_only
            BEFORE UPDATE OR DELETE ON message
            FOR EACH ROW EXECUTE FUNCTION message_append_only();

        -- Row Level Security with no policies: only the table owner (the server's
        -- direct connection) can read or write. Supabase's public API roles see nothing.
        ALTER TABLE session ENABLE ROW LEVEL SECURITY;
        ALTER TABLE message ENABLE ROW LEVEL SECURITY;
        ALTER TABLE api_request ENABLE ROW LEVEL SECURITY;
        ALTER TABLE alembic_version ENABLE ROW LEVEL SECURITY;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE alembic_version DISABLE ROW LEVEL SECURITY;
        DROP TABLE api_request;
        DROP TABLE message;
        DROP FUNCTION message_append_only();
        DROP TABLE session;
        """
    )
