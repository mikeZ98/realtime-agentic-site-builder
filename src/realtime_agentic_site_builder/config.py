"""Process configuration, read once from ``RASB_``-prefixed environment variables."""

from functools import lru_cache
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, RedisDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from realtime_agentic_site_builder.domain.swarm import DEFAULT_LOCK_TIMEOUT_MS, MAX_LOCK_TIMEOUT_MS

ENV_PREFIX: Final = "RASB_"
MAX_BROADCAST_FPS: Final = 240

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
# "<endpoint>:<model>", so each tier can use a different vendor.
ModelRoute = Annotated[str, Field(pattern=r"^(anthropic|openrouter):\S+$")]


class ModelPrice(BaseModel):
    """What a model charges, in US dollars per million tokens."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_usd_per_mtok: float = Field(ge=0, allow_inf_nan=False)
    output_usd_per_mtok: float = Field(ge=0, allow_inf_nan=False)

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_usd_per_mtok + output_tokens * self.output_usd_per_mtok
        ) / 1_000_000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX, frozen=True, extra="forbid", env_ignore_empty=True
    )

    # Hidden from repr because the URL can embed a password. None selects the in-memory bus.
    redis_url: RedisDsn | None = Field(default=None, repr=False)
    session_ttl_seconds: int = Field(default=3600, ge=1)
    lock_lease_ms: int = Field(default=DEFAULT_LOCK_TIMEOUT_MS, ge=1, le=MAX_LOCK_TIMEOUT_MS)
    max_patch_size_bytes: int = Field(default=65536, ge=1)
    # Every delta re-serializes the whole bundle on the event loop, so its size bounds that cost.
    max_bundle_bytes: int = Field(default=2 * 1024 * 1024, ge=1)
    # Each frame of the replay log holds a full bundle. Oldest frames go once this is exceeded.
    replay_history_bytes: int = Field(default=64 * 1024 * 1024, ge=1)
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
    # Keyed by the model id a provider reports, such as "claude-sonnet-5-5". Prices change, so
    # none are built in: a model without one reports tokens, and no cost.
    model_prices: dict[str, ModelPrice] = Field(default_factory=dict)

    # Signs client tokens. The gateway refuses to start without it: `prompt` spends real money.
    gateway_secret: SecretStr | None = Field(default=None, min_length=32)
    gateway_sweep_interval_seconds: float = Field(default=1.0, gt=0)
    gateway_drain_timeout_seconds: float = Field(default=5.0, gt=0)
    gateway_max_sessions: int = Field(default=1000, ge=1)
    gateway_max_concurrent_runs: int = Field(default=8, ge=1)
    # Per-person limits, so that no one tenant can exhaust what the gateway's limits share out.
    gateway_max_sessions_per_user: int = Field(default=10, ge=1)
    gateway_max_concurrent_exports: int = Field(default=4, ge=1)
    ws_max_connections_per_user: int = Field(default=8, ge=1)
    ws_max_deltas_per_second: int = Field(default=10, ge=1)
    ws_max_locks_per_user: int = Field(default=32, ge=1)
    max_request_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    thought_backlog_size: int = Field(default=1000, ge=1)
    ws_catchup_deadline_ms: int = Field(default=5000, ge=1)
    ws_heartbeat_timeout_seconds: float = Field(default=60.0, gt=0)
    ws_max_message_bytes: int = Field(default=128 * 1024, ge=1)
    ws_max_messages_per_second: int = Field(default=120, ge=1)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
