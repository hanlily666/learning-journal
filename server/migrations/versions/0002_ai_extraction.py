"""Phase 2: segments and extracted items; the 'mixed' subject becomes 'messy'.

Revision ID: 0002
Revises: 0001
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE session DROP CONSTRAINT session_subject_check;
        UPDATE session SET subject = 'messy' WHERE subject = 'mixed';
        ALTER TABLE session ADD CONSTRAINT session_subject_check
            CHECK (subject IN ('ai', 'french', 'messy'));

        -- A run of new messages processed together: every complete message of the
        -- session from first_message_id to last_message_id (inclusive).
        CREATE TABLE segment (
            id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            session_id        bigint NOT NULL REFERENCES session(id),
            first_message_id  bigint NOT NULL REFERENCES message(id),
            last_message_id   bigint NOT NULL REFERENCES message(id),
            label             text CHECK (label IN ('ai', 'french', 'messy')),
            label_reason      text,
            -- pending -> running -> done (AI, extracted) | held (French, for Phase 6)
            --                     | skipped (messy) | needs_attention (failed twice)
            status            text NOT NULL DEFAULT 'pending' CHECK (status IN
                              ('pending', 'running', 'done', 'held', 'skipped', 'needs_attention')),
            subject           text,
            summary           text,
            error             text,
            attempts          integer NOT NULL DEFAULT 0,
            created_at        timestamptz NOT NULL DEFAULT now(),
            updated_at        timestamptz NOT NULL DEFAULT now(),
            CHECK (first_message_id <= last_message_id)
        );
        CREATE INDEX segment_session_idx ON segment (session_id, id);
        -- Later messages of a session wait behind its unfinished segment.
        CREATE UNIQUE INDEX segment_one_unfinished_per_session ON segment (session_id)
            WHERE status IN ('pending', 'running', 'needs_attention');

        CREATE TABLE item (
            id                  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            segment_id          bigint NOT NULL REFERENCES segment(id),
            field               text NOT NULL CHECK (field IN ('explore', 'source', 'gap', 'concept')),
            text                text NOT NULL,
            source_message_ids  bigint[] NOT NULL,
            reason              text NOT NULL,
            review_state        text NOT NULL DEFAULT 'unreviewed',
            agent_state         text NOT NULL DEFAULT 'queued',
            created_at          timestamptz NOT NULL DEFAULT now()
        );
        CREATE INDEX item_segment_idx ON item (segment_id, id);

        ALTER TABLE segment ENABLE ROW LEVEL SECURITY;
        ALTER TABLE item ENABLE ROW LEVEL SECURITY;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE item;
        DROP TABLE segment;
        ALTER TABLE session DROP CONSTRAINT session_subject_check;
        UPDATE session SET subject = 'mixed' WHERE subject = 'messy';
        ALTER TABLE session ADD CONSTRAINT session_subject_check
            CHECK (subject IN ('ai', 'french', 'mixed'));
        """
    )
