"""Anthropic Messages API adapter, for both the real API and Anthropic-compatible
servers (see xagent.config.BACKENDS).

The model acts through exactly one tool -- `python(code)` -- rather than fenced
code in free text. That buys well-delimited code, no fence-parsing failure modes,
and the API's own stop handling.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
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
    # Names of any further tool_use blocks in the same turn that were not acted
    # on, so the caller can correct the model without discarding what it did.
    ignored_tools: list[str] = field(default_factory=list)
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
        self.thinking = config.resolve_thinking(thinking)
        self.sampling = sampling or config.DEFAULT_SAMPLING
        self.client = anthropic.Anthropic(**config.client_kwargs(self.backend))
        self.usage = Usage()
        self.calls = 0
        self.truncated_turns = 0
        self.last_retry: str | None = None
        # Set by the caller to watch a turn as it is generated: on_delta(kind, text)
        # with kind in {thinking, text, code, retry}. None means nobody is looking.
        self.on_delta = None
        self._code_seen = ""

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

    @contextmanager
    def muted(self):
        """Suppress delta relay for requests the caller is not watching."""
        saved, self.on_delta = self.on_delta, None
        try:
            yield
        finally:
            self.on_delta = saved

    def _relay(self, event) -> None:
        """Forward one streamed event to whoever is watching this turn."""
        if self.on_delta is None:
            return
        if event.type == "content_block_start":
            block = getattr(event, "content_block", None)
            if getattr(block, "type", None) == "tool_use":
                # Per block, not per request: a second tool_use starts its own
                # arguments, and diffing them against the first block's snapshot
                # would emit nothing or garbage.
                self._code_seen = ""
                self.on_delta("tool", getattr(block, "name", "") or "")
        elif event.type == "text":
            self.on_delta("text", event.text)
        elif event.type == "thinking":
            self.on_delta("thinking", event.thinking)
        elif event.type == "input_json":
            # `snapshot` is the tool input parsed so far, so the JSON string
            # escaping is already undone -- diffing it against what has been
            # relayed yields printable source, where partial_json would emit raw
            # `\n`-laden fragments. Partial parses can rewind mid-escape, so only
            # extend when the new snapshot still starts with what was sent.
            snapshot = event.snapshot if isinstance(event.snapshot, dict) else {}
            code = snapshot.get("code") or ""
            if isinstance(code, str) and code.startswith(self._code_seen):
                if len(code) > len(self._code_seen):
                    self.on_delta("code", code[len(self._code_seen):])
                self._code_seen = code

    def _create(self, **kw):
        """One retry ring above the SDK's, for sustained 429/529 weather.

        Always streamed, and not for the display alone. A non-streamed request
        sends no bytes until the whole turn is finished, and a reasoning model
        writing a long file stays silent for minutes. Anything in between with an
        idle timeout -- a CDN, a tunnel, a load balancer -- then drops the
        connection; both retry rings read that as transient weather and reissue
        the identical request, so the run regenerates the same turn forever, at
        full GPU cost, with nothing on screen. Streaming keeps bytes moving, so
        the timeout never fires and the turn is visible while it happens.

        Deadline-bounded: APITimeoutError subclasses APIConnectionError and is
        retryable at both layers, so attempt-count alone would allow hours of
        stacked timeouts on a single request.
        """
        delay = 2.0
        deadline = time.monotonic() + self.RETRY_DEADLINE
        for attempt in range(5):
            try:
                with self.client.messages.stream(**kw) as stream:
                    for event in stream:
                        self._relay(event)
                    return stream.get_final_message()
            except (anthropic.RateLimitError, anthropic.InternalServerError,
                    anthropic.APIConnectionError) as e:
                self.last_retry = f"{type(e).__name__}: {str(e)[:120]}"
                if attempt == 4 or time.monotonic() + delay > deadline:
                    raise
                # Say so. A silent retry discards a turn the operator watched
                # being generated, and looks identical to a hang.
                if self.on_delta:
                    self.on_delta("retry", f"{self.last_retry} — attempt "
                                           f"{attempt + 2}/5 in {delay:.0f}s")
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
        self._code_seen = ""
        params, extra = self._sampling_kw()
        thinking = self._thinking_kw(budget)
        # No sampling override here: _sampling_kw already returns nothing for the
        # Anthropic API, which is the only backend that rejects sampling alongside
        # extended thinking. Blanking it unconditionally would strip qwen's
        # recommended preset from every run now that thinking is on by default --
        # and SGLang accepts both together (verified: 200 on thinking + top_p/top_k).
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

        tool_blocks = []
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
                tool_blocks.append(block)

        # A turn can carry more than one tool_use block -- qwen is trained with
        # shell tools and reaches for one it was never given. Last-write-wins let
        # that stray block overwrite a perfectly good `python` call, so the code
        # was silently dropped and the turn was spent reporting the stray name.
        # Act on the python call wherever it sits, and merely report the rest.
        acted = next((b for b in tool_blocks if b.name == "python"), None)
        if acted is None and tool_blocks:
            acted = tool_blocks[0]
        if acted is not None:
            turn.tool_name = acted.name
            turn.tool_use_id = acted.id
            turn.code = (acted.input.get("code") or "") if isinstance(acted.input, dict) else ""
            turn.ignored_tools = [b.name for b in tool_blocks if b is not acted]
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
            with self.muted():
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
