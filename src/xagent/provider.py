"""Anthropic Messages API adapter, for both the real API and Anthropic-compatible
servers (see xagent.config.BACKENDS).

The model acts through exactly one tool -- `python(code)` -- rather than fenced
code in free text. That buys well-delimited code, no fence-parsing failure modes,
and the API's own stop handling.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import anthropic

from xagent import config
from xagent.config import Backend

PYTHON_TOOL = {
    "name": "python",
    "description": (
        "Run Python in your persistent IPython kernel. State carries across calls: "
        "variables, imports, and definitions all survive. The value of the final "
        "expression is displayed (capped), and stdout is captured. This is how you "
        "both act and think -- one call per step."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python source to execute."}},
        "required": ["code"],
    },
}


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0

    def add(self, u) -> None:
        self.input += getattr(u, "input_tokens", 0) or 0
        self.output += getattr(u, "output_tokens", 0) or 0
        self.cache_read += getattr(u, "cache_read_input_tokens", 0) or 0
        self.cache_write += getattr(u, "cache_creation_input_tokens", 0) or 0

    @property
    def cache_hit_rate(self) -> float:
        seen = self.cache_read + self.cache_write + self.input
        return self.cache_read / seen if seen else 0.0

    def __repr__(self) -> str:
        return (f"<Usage in={self.input:,} out={self.output:,} "
                f"cache_read={self.cache_read:,} cache_write={self.cache_write:,} "
                f"hit={self.cache_hit_rate:.0%}>")


@dataclass
class Turn:
    text: str = ""
    thinking: str = ""
    thinking_blocks: list[dict] = field(default_factory=list)
    code: str | None = None
    tool_use_id: str | None = None
    tool_name: str | None = None
    stop_reason: str = ""
    input_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0


class Provider:
    # Leave room for the request framing when clamping against the model window.
    WINDOW_MARGIN = 4_000
    MIN_OUTPUT = 4_096

    def __init__(self, backend: Backend | str | None = None, model: str | None = None,
                 thinking: str | None = None, max_tokens: int | None = None,
                 sampling: str | None = None):
        self.backend = backend if isinstance(backend, Backend) else config.get_backend(backend)
        self.model = model or self.backend.driver_model
        self.max_tokens = max_tokens or self.backend.max_output
        self.thinking = thinking if thinking in config.THINKING_BUDGETS else None
        self.sampling = sampling or config.DEFAULT_SAMPLING
        self.client = anthropic.Anthropic(**config.client_kwargs(self.backend))
        self.usage = Usage()
        self.calls = 0
        self.truncated_turns = 0
        self.last_retry: str | None = None

    # ----------------------------------------------------------------- sampling

    def _sampling_kw(self) -> tuple[dict, dict | None]:
        """Sampling parameters, split into wire fields and extra_body fields.

        Only backends that accept them get them: the Anthropic API rejects
        `temperature` as deprecated for these models, and rejects the extra_body
        fields as unknown inputs.
        """
        if not self.backend.sampling:
            return {}, None
        preset = config.SAMPLING.get(self.sampling) or config.SAMPLING[config.DEFAULT_SAMPLING]
        return dict(preset["params"]), dict(preset["extra"])

    def _clamp(self, budget: int, system: str, messages: list[dict]) -> int:
        """Shrink max_tokens so input + output stays inside the model window.

        Only some servers enforce this, but where they do, a fixed output budget
        starts failing exactly when a session grows long enough to need compaction.
        """
        budget = min(budget, self.backend.max_output)
        window = self.backend.total_window
        if not window:
            return budget
        approx_input = (len(system) + len(json.dumps(messages))) // 4
        room = window - approx_input - self.WINDOW_MARGIN
        if room < self.MIN_OUTPUT:
            # Flooring at MIN_OUTPUT here would send a request guaranteed to be
            # rejected for overflowing the window. Say so instead.
            raise RuntimeError(
                f"context is too large for {self.model}: ~{approx_input:,} input tokens "
                f"leaves {room:,} of a {window:,}-token window for output. "
                f"Lower --budget so compaction fires sooner."
            )
        return max(self.MIN_OUTPUT, min(budget, room))

    # ------------------------------------------------------------------ helpers

    def _thinking_kw(self, max_tokens: int) -> dict:
        if not self.thinking:
            return {}
        budget = config.THINKING_BUDGETS[self.thinking]
        # max_tokens must leave room for a visible answer on top of the budget, and
        # budget_tokens must stay above the API minimum of 1024.
        return {"thinking": {"type": "enabled",
                             "budget_tokens": max(1024, min(budget, max_tokens - 512))}}

    def _budget(self) -> int:
        if not self.thinking:
            return self.max_tokens
        return max(self.max_tokens, config.THINKING_BUDGETS[self.thinking] + 2048)

    def system_blocks(self, system: str) -> list[dict]:
        blocks: list[dict] = []
        if self.backend.system_prefix:
            blocks.append({"type": "text", "text": config.CLAUDE_CODE_SYSTEM})
        blocks.append({"type": "text", "text": system})
        if self.backend.cache_breakpoints:
            blocks[-1]["cache_control"] = {"type": "ephemeral"}
        return blocks

    RETRY_DEADLINE = 900.0  # seconds for all attempts at one request, combined

    def _create(self, **kw):
        """One retry ring above the SDK's, for sustained 429/529 weather.

        Deadline-bounded: APITimeoutError subclasses APIConnectionError and is
        retryable at both layers, so attempt-count alone would allow hours of
        stacked timeouts on a single request.
        """
        delay = 2.0
        deadline = time.monotonic() + self.RETRY_DEADLINE
        for attempt in range(5):
            try:
                return self.client.messages.create(**kw)
            except (anthropic.RateLimitError, anthropic.InternalServerError,
                    anthropic.APIConnectionError) as e:
                self.last_retry = f"{type(e).__name__}: {str(e)[:120]}"
                if attempt == 4 or time.monotonic() + delay > deadline:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 30)
        raise RuntimeError("unreachable")

    # ------------------------------------------------------------------- sample

    def sample(self, system: str, messages: list[dict]) -> Turn:
        """Sample one turn, retrying a turn that reasoned itself out of room.

        A reasoning model can spend its whole output budget thinking and stop before
        emitting the tool call. That is not the model declining to act, so widen the
        budget and ask again rather than treating it as the end of the run.
        """
        budget = self._clamp(self._budget(), system, messages)
        for _ in range(3):
            turn = self._sample_once(system, messages, budget)
            if turn.tool_use_id or turn.stop_reason != "max_tokens":
                return turn
            wider = self._clamp(budget * 2, system, messages)
            if wider <= budget:
                # Already at the ceiling. Re-asking would reissue the identical
                # request and truncate identically, at full price each time.
                self.truncated_turns += 1
                return turn
            budget = wider
            self.truncated_turns += 1
        return turn

    def _sample_once(self, system: str, messages: list[dict], budget: int) -> Turn:
        params, extra = self._sampling_kw()
        thinking = self._thinking_kw(budget)
        if thinking:
            # Extended thinking fixes the sampling distribution; sending top_p/top_k
            # alongside it is rejected.
            params = {}
        resp = self._create(
            model=self.model,
            max_tokens=budget,
            system=self.system_blocks(system),
            messages=messages,
            tools=[PYTHON_TOOL],
            extra_body=extra,
            **params,
            **thinking,
        )
        self.calls += 1
        self.usage.add(resp.usage)

        turn = Turn(
            stop_reason=resp.stop_reason or "",
            input_tokens=resp.usage.input_tokens or 0,
            cache_read=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
            cache_write=getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
        )
        for block in resp.content:
            if block.type == "text":
                turn.text += block.text
            elif block.type == "thinking":
                turn.thinking += getattr(block, "thinking", "") or ""
                # Kept raw: Anthropic requires prior thinking blocks (with their
                # signatures) to be echoed back on the next tool-use turn.
                turn.thinking_blocks.append(block.model_dump(exclude_none=True))
            elif block.type == "redacted_thinking":
                turn.thinking_blocks.append(block.model_dump(exclude_none=True))
            elif block.type == "tool_use":
                turn.tool_name = block.name
                turn.tool_use_id = block.id
                if isinstance(block.input, dict):
                    turn.code = block.input.get("code") or ""
                else:
                    turn.code = ""
        return turn

    def complete(self, prompt: str, model: str | None = None, max_tokens: int = 4096) -> str:
        """Plain single-shot completion, no tools. Used for compaction summaries.

        Reasoning models spend output tokens on thinking before they emit any text,
        so a budget that looks generous for a 300-word summary can be consumed
        entirely by the reasoning block. Retry wider once, then fall back to the
        reasoning itself rather than returning nothing.
        """
        system: list[dict] = []
        if self.backend.system_prefix:
            system.append({"type": "text", "text": config.CLAUDE_CODE_SYSTEM})

        params, extra = self._sampling_kw()
        for budget in (max_tokens, max_tokens * 3):
            resp = self._create(
                model=model or self.backend.worker_model,
                max_tokens=min(budget, self.backend.max_output),
                system=system or anthropic.NOT_GIVEN,
                messages=[{"role": "user", "content": prompt}],
                extra_body=extra,
                **params,
            )
            self.calls += 1
            self.usage.add(resp.usage)
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            if text:
                return text
            if resp.stop_reason != "max_tokens":
                break
        # Deliberately NOT falling back to the reasoning block. It is the model
        # arguing with itself about word counts, and passing it off as a summary
        # puts that into the agent's authoritative record of its own history.
        return ""
