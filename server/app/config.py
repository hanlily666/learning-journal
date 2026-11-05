"""Settings read from the environment (and an optional .env file)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    # postgresql://user:password@host:5432/dbname (a direct connection, not the REST API)
    database_url: str = "postgresql://postgres:postgres@localhost:5432/learning_journal"
    # Read by the Anthropic SDK itself; listed here so a missing key is visible at startup.
    anthropic_api_key: str | None = None

    claude_model: str = "claude-opus-5-5"
    claude_effort: str = "medium"
    reply_max_tokens: int = 64_000
    context_warning_tokens: int = 150_000
    # How often (seconds) a streaming reply's partial text is written to the database.
    partial_save_interval: float = 1.0
    # Origin of the Vite dev server, allowed by CORS.
    web_origin: str = "http://localhost:5173"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def sqlalchemy_url(url: str) -> str:
    """Point a plain postgres URL at the psycopg 3 driver."""
    for prefix in ("postgresql+psycopg://", "postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url
