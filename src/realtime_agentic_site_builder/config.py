"""Process configuration, read once from ``RASB_``-prefixed environment variables."""

from functools import lru_cache
from typing import Annotated, Final, Literal

from pydantic import Field, RedisDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from realtime_agentic_site_builder.domain.swarm import DEFAULT_LOCK_TIMEOUT_MS, MAX_LOCK_TIMEOUT_MS

ENV_PREFIX: Final = "RASB_"
MAX_BROADCAST_FPS: Final = 240

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
# "<endpoint>:<model>", so each tier can use a different vendor.
ModelRoute = Annotated[str, Field(pattern=r"^(anthropic|openrouter):\S+$")]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX, frozen=True, extra="forbid", env_ignore_empty=True
    )

    # Hidden from repr because the URL can embed a password. None selects the in-memory bus.
    redis_url: RedisDsn | None = Field(default=None, repr=False)
    session_ttl_seconds: int = Field(default=3600, ge=1)
    lock_lease_ms: int = Field(default=DEFAULT_LOCK_TIMEOUT_MS, ge=1, le=MAX_LOCK_TIMEOUT_MS)
    max_patch_size_bytes: int = Field(default=65536, ge=1)
    broadcast_fps: int = Field(default=30, ge=1, le=MAX_BROADCAST_FPS)
    log_level: LogLevel = "INFO"

    anthropic_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    architect_model: ModelRoute = "anthropic:claude-sonnet-5-5"
    coder_model: ModelRoute = "openrouter:deepseek/deepseek-v4.1-flash"
    stylist_model: ModelRoute = "anthropic:claude-haiku-5-5"
    critic_model: ModelRoute = "openrouter:openai/gpt-4o-mini"
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    agent_max_iterations: int = Field(default=3, ge=1, le=20)
    agent_lock_wait_ms: int = Field(default=120_000, ge=1)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
