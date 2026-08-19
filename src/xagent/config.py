"""Providers, models, and credential resolution.

Two providers, chosen explicitly rather than inferred from the model name:

  anthropic  the Messages API. Prompt caching is real here, which is what makes
             the cache-vs-compaction tradeoff measurable.
  codiv      an SGLang server at api.codiv.ai speaking Anthropic-compatible
             /v1/messages, currently serving qwen. Needs no credentials and always
             emits thinking blocks -- signature-less ones, so they must not be
             echoed back.

Both cache prefixes, by different mechanisms: Anthropic on explicit cache_control
breakpoints, SGLang automatically via its radix cache. So the compaction policy --
stable prefix, compact rarely and in batches, never touch the recent tail -- earns
its keep on both. Only Anthropic reports hit rates, so only there is it measurable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

CLAUDE_CODE_SYSTEM = "You are Claude Code, Anthropic's official CLI for Claude."

# Extended-thinking budgets in tokens, by level.
THINKING_BUDGETS = {"low": 2048, "medium": 8192, "high": 24576}

# Sampling presets recommended for Qwen 3.8. `extra` holds the parameters the
# Anthropic wire format has no field for; they ride in extra_body, which the
# SGLang backend accepts and the real Anthropic API rejects.
SAMPLING = {
    "thinking": {
        "params": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "extra": {"min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0},
    },
    "instruct": {
        "params": {"temperature": 0.7, "top_p": 0.80, "top_k": 20},
        "extra": {"min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0},
    },
}
DEFAULT_SAMPLING = "thinking"


@dataclass(frozen=True)
class Backend:
    name: str
    base_url: str | None
    driver_model: str
    worker_model: str      # subagents and compaction summaries
    context_budget: int      # compaction target, deliberately under the model max
    max_output: int          # per-turn output ceiling
    # Set when the server rejects requests whose input + max_tokens exceeds the
    # model window, so max_tokens has to be clamped per request rather than fixed.
    total_window: int | None
    sampling: bool           # send temperature/top_p/top_k (and extra_body)?
    cache_breakpoints: bool  # send explicit cache_control markers?
    reports_cache: bool      # does usage carry cache_read/cache_write?
    replay_thinking: bool    # must prior thinking blocks be echoed back?
    env_keys: tuple[str, ...]
    needs_credentials: bool = True
    system_prefix: bool = False  # OAuth tokens require the Claude Code identity


BACKENDS: dict[str, Backend] = {
    "anthropic": Backend(
        name="anthropic",
        base_url=None,
        driver_model="claude-opus-5",
        worker_model="claude-sonnet-5",
        context_budget=160_000,
        # 128k is exactly the API ceiling for these models; above it the request is
        # rejected outright. Input is budgeted separately, so no total-window clamp.
        max_output=128_000,
        total_window=None,
        # This API rejects `temperature` as deprecated for these models.
        sampling=False,
        cache_breakpoints=True,
        reports_cache=True,
        replay_thinking=True,
        env_keys=("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"),
        system_prefix=True,
    ),
    "codiv": Backend(
        name="codiv",
        base_url="https://api.codiv.ai",
        driver_model="qwen-3.8-27b",
        worker_model="qwen-3.8-27b",
        context_budget=180_000,
        # qwen reasons at length before emitting a tool call, and a budget that is
        # too tight is spent entirely on thinking -- the turn then returns no action.
        max_output=128_000,
        # SGLang counts input + max_tokens against one 262,144-token window and
        # rejects the request when they overflow it, so max_tokens must shrink as
        # the context grows.
        total_window=262_144,
        sampling=True,
        # Harmless here (verified accepted); SGLang's radix cache keys off the
        # prefix itself, so the same stable-prefix discipline applies.
        cache_breakpoints=True,
        reports_cache=False,
        replay_thinking=False,
        env_keys=("CODIV_API_KEY",),
        needs_credentials=False,
    ),
}

DEFAULT_PROVIDER = "anthropic"

# Compaction policy (provider-independent).
WATERMARK = 0.80
# Cells newer than this are never compacted, so the actively appended tail of the
# prompt never shifts and stays cache-warm.
KEEP_RECENT = 10

MAX_AGENT_DEPTH = 2
MAX_TOTAL_AGENTS = 64


def get_backend(name: str | None) -> Backend:
    key = (name or os.environ.get("XAGENT_PROVIDER") or DEFAULT_PROVIDER).lower()
    if key not in BACKENDS:
        raise SystemExit(f"unknown provider {key!r}; choose from {', '.join(BACKENDS)}")
    return BACKENDS[key]


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


# Only these may be imported from a .env. xagent is routinely pointed at other
# people's repositories, and any .env in the working directory is their file, not
# the operator's -- importing it wholesale lets it set ANTHROPIC_BASE_URL and
# redirect the operator's credentials to another host.
DOTENV_KEYS = (
    "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY", "CODIV_API_KEY",
    "XAGENT_PROVIDER", "XAGENT_MODEL",
)


def load_dotenv() -> None:
    """Populate os.environ from .env files without clobbering real env vars.

    Reads every candidate rather than stopping at the first: a foreign .env in the
    working directory used to shadow the operator's own.
    """
    for candidate in (project_root() / ".env", Path.cwd() / ".env"):
        if not candidate.is_file():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            if key in DOTENV_KEYS:
                os.environ.setdefault(key, val.strip().strip("'\""))


def client_kwargs(backend: Backend) -> dict:
    """Constructor kwargs for the Anthropic SDK pointed at `backend`."""
    load_dotenv()
    kwargs: dict = {"max_retries": 4, "timeout": 600.0}
    if backend.base_url:
        kwargs["base_url"] = backend.base_url

    token = next((os.environ[k] for k in backend.env_keys if os.environ.get(k)), None)

    if token is None:
        if backend.needs_credentials:
            raise SystemExit(
                f"provider {backend.name!r} needs one of: {', '.join(backend.env_keys)}.\n"
                "Set it in the environment or in a .env file."
            )
        kwargs["api_key"] = "not-required"
        return kwargs

    # A Claude Code OAuth token authenticates as `Authorization: Bearer`, which the
    # SDK exposes as auth_token. A plain API key takes the normal x-api-key path.
    if token.startswith("sk-ant-oat"):
        kwargs["auth_token"] = token
        kwargs["default_headers"] = {"anthropic-beta": "oauth-2025-04-20"}
    else:
        kwargs["api_key"] = token
    return kwargs
