"""Runtime configuration. Everything has a working default so the platform
runs with no API key at all — the planner falls back to a deterministic
rule-based planner, which is also what the eval suite pins against."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    # --- model ---
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
    llm_timeout_s: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_S", 30.0))

    # --- persistence ---
    checkpoint_db: str = field(
        default_factory=lambda: os.getenv("CHECKPOINT_DB", str(ROOT / "data" / "checkpoints.sqlite"))
    )
    trace_dir: str = field(default_factory=lambda: os.getenv("TRACE_DIR", str(ROOT / "data" / "traces")))

    # --- guardrails ---
    max_steps: int = field(default_factory=lambda: _env_int("MAX_STEPS", 25))
    max_tool_calls: int = field(default_factory=lambda: _env_int("MAX_TOOL_CALLS", 15))
    max_usd: float = field(default_factory=lambda: _env_float("MAX_USD", 0.50))

    # If true, reversible write actions (drafting a document) run without a
    # human gate. Irreversible ones (granting access, sending mail) never do.
    auto_approve_reversible: bool = field(
        default_factory=lambda: _env_bool("AUTO_APPROVE_REVERSIBLE", True)
    )

    # --- retries ---
    default_retries: int = field(default_factory=lambda: _env_int("DEFAULT_RETRIES", 2))
    default_timeout_s: float = field(default_factory=lambda: _env_float("DEFAULT_TIMEOUT_S", 10.0))
    backoff_base_s: float = field(default_factory=lambda: _env_float("BACKOFF_BASE_S", 0.25))

    # --- fault injection, for demoing the retry path ---
    flaky_tools: bool = field(default_factory=lambda: _env_bool("FLAKY_TOOLS", False))

    def ensure_dirs(self) -> None:
        Path(self.checkpoint_db).parent.mkdir(parents=True, exist_ok=True)
        Path(self.trace_dir).mkdir(parents=True, exist_ok=True)

    @property
    def planner_backend(self) -> str:
        return "gemini" if self.gemini_api_key else "deterministic"


settings = Settings()
