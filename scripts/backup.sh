#!/usr/bin/env bash
# Daily database backup with pg_dump, from this laptop to a private folder.
#
#   scripts/backup.sh                 # uses DATABASE_URL and BACKUP_DIR from .env
#
# Schedule it once a day with cron (crontab -e), for example at 21:00:
#   0 21 * * * cd /path/to/repo && scripts/backup.sh >> "$HOME/learning-journal-backups/backup.log" 2>&1
#
# pg_dump's major version must be at least the server's (Supabase and the Docker image run 17).
# Without a local pg_dump, set PG_DUMP="docker run --rm -i postgres:17 pg_dump".
set -euo pipefail

cd "$(dirname "$0")/.."
# Read KEY=value lines from .env; variables already set in the environment win.
if [[ -f .env ]]; then
  while IFS='=' read -r key value; do
    if [[ $key =~ ^[A-Z_][A-Z0-9_]*$ && -z "${!key:-}" ]]; then
      export "$key=$value"
    fi
  done < .env
fi

: "${DATABASE_URL:?DATABASE_URL is not set}"
BACKUP_DIR="${BACKUP_DIR:-$HOME/learning-journal-backups}"
# Optional: delete dumps older than this many days. Unset keeps every dump.
BACKUP_KEEP_DAYS="${BACKUP_KEEP_DAYS:-}"
PG_DUMP="${PG_DUMP:-pg_dump}"

mkdir -p "$BACKUP_DIR"
chmod 700 "$BACKUP_DIR"
umask 077

file="$BACKUP_DIR/learning-journal-$(date -u +%Y%m%dT%H%M%SZ).dump"
# Custom format: compressed, and restorable table by table with pg_restore.
$PG_DUMP --format=custom --no-owner --no-privileges --dbname="$DATABASE_URL" > "$file.partial"
mv "$file.partial" "$file"
echo "backed up to $file ($(du -h "$file" | cut -f1))"

if [[ -n "$BACKUP_KEEP_DAYS" ]]; then
  find "$BACKUP_DIR" -name 'learning-journal-*.dump' -mtime +"$BACKUP_KEEP_DAYS" -delete
fi
