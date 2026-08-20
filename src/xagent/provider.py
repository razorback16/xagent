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
import httpx

from xagent import config
from xagent.config import Backend
from xagent.kernel import CELL_TIMEOUT, MAX_CELL_TIMEOUT

# The middle of the python tool's description is the same for both roles; only the
# first and last paragraphs -- which name the tools that exist and how the run ends
# -- differ. Shared here so the two renderings cannot drift apart.
_PYTHON_BODY = (
    "State carries across calls: variables, imports and definitions all "
    "survive, so one call is one cell of one long session, and a cell may "
    "hold as many statements as the step needs. stdout is captured and the "
    "value of the final expression is displayed, both capped -- a large value "
    "shows as a one-line handle while the object itself stays whole in the "
    "kernel. An exception prints a traceback and destroys nothing.\n\n"
    f"A cell runs for {CELL_TIMEOUT:g} seconds by default and is then interrupted, "
    f"which ends that cell but keeps the namespace. When you already know a step "
    f"will take longer -- a build, an install, a large download, a long batch -- "
    f"pass `timeout` with the seconds it needs (up to {MAX_CELL_TIMEOUT:g}) in the "
    f"same call, rather than letting it be cut off and retried.\n\n"
)

_PYTHON_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "Python source to execute."},
        "timeout": {
            "type": "number",
            "description": (
                f"Optional. Seconds this cell may run before the kernel is "
                f"interrupted. Defaults to {CELL_TIMEOUT:g}; raise it for a step "
                f"you already know is slow. Capped at {MAX_CELL_TIMEOUT:g}."
            ),
        },
    },
    "required": ["code"],
}

# This string is the highest-attention place to state the contract: the Qwen chat
# template renders it inside the `<tools>` block, above the system prompt. Prose,
# not tables -- the template puts it through `tojson`, so every newline reaches the
# model as a literal \n and any column alignment is destroyed.
PYTHON_TOOL = {
    "name": "python",
    "description": (
        "Run Python in a persistent IPython kernel. This and `done` are the only "
        "tools that exist -- there is no shell tool and no file tool. Everything "
        "you do, you do by writing Python here.\n\n"
        + _PYTHON_BODY +
        "The namespace already holds the helpers you reach for most -- files, "
        "search, shell, subagents, context control -- plus the whole Python "
        "standard library. Those are ordinary functions, not tools: write "
        "`sh(\"git status\")` as code inside `code`. A tool call by any name "
        "other than `python` or `done` runs nothing and wastes the turn."
    ),
    "input_schema": _PYTHON_SCHEMA,
}

# A subagent finishes by value, from inside the kernel, so for it `done` really is
# a function and really is not a tool -- the contract this description has always
# stated. It keeps that text; only the top level's changed.
PYTHON_TOOL_SUBAGENT = {
    "name": "python",
    "description": (
        "Run Python in a persistent IPython kernel. This is the only tool that "
        "exists -- there is no shell tool, no file tool, no `done` tool. "
        "Everything you do, you do by writing Python here.\n\n"
        + _PYTHON_BODY +
        "The namespace already holds the helpers you reach for most -- files, "
        "search, shell, subagents, context control, `done` -- plus the whole "
        "Python standard library. Those are ordinary functions, not tools: write "
        "`sh(\"git status\")` or `done(value)` as code inside `code`. A tool call "
        "by any name other than `python` runs nothing and wastes the turn -- "
        "including `done`, which returns your result only when it arrives as one "
        "more line of Python in this tool, never as a tool call of its own."
    ),
    "input_schema": _PYTHON_SCHEMA,
}

# Offered to the top-level agent only. It takes no input at all, which is the whole
# reason it can be a tool: the answer is prose the model has to write out anyway,
# so there is no value for an argument to carry -- unlike a subagent's result,
# which lives in its kernel and would have to be transcribed to fit in one.
DONE_TOOL = {
    "name": "done",
    "description": (
        "End the run. Call it in the same turn as your final answer, which you "
        "write as ordinary text beside this call: the text is the answer, and "
        "this call is only the full stop after it. It takes no input -- there is "
        "nothing to pass, and anything you would put here is an answer you should "
        "have written as text. If you send a `python` call in the same turn, that "
        "code runs first and the run ends after it. Call this only when the work "
        "is genuinely finished; if a step failed, investigate instead."
    ),
    "input_schema": {"type": "object", "properties": {}},
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


def _cell_timeout(raw) -> float | None:
    """Validate the model's requested cell timeout.

    Anything unusable -- a string, a negative, a NaN -- is dropped rather than
    raised on: the cell it came with is real work, and running it under the
    default beats failing the turn over an argument the model got wrong. A
    request above the cap is clamped, not refused, for the same reason.
    """
    if raw is None:
        return None
    try:
        secs = float(raw)
    except (TypeError, ValueError):
        return None
    if not secs > 0 or secs != secs:  # non-positive or NaN
        return None
    return min(secs, MAX_CELL_TIMEOUT)


@dataclass
class Turn:
    text: str = ""
    thinking: str = ""
    thinking_blocks: list[dict] = field(default_factory=list)
    code: str | None = None
    # The cell timeout the model asked for, already validated and clamped. None
    # means it did not ask, and the kernel's default stands.
    timeout: float | None = None
    tool_use_id: str | None = None
    tool_name: str | None = None
    # The top-level finish. Set when a `done` tool block arrived; it is extracted
    # rather than treated as a call to act on, so it never competes with `python`
    # for the one acted-on slot and never lands in `ignored_tools`.
    done: bool = False
    done_id: str | None = None
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
        # Set by Runner to the role: the top level is offered `done` as a tool, a
        # subagent is not (it finishes by value, from inside its kernel).
        self.offer_done = True
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

    def _thinking_kw(self) -> dict:
        """The thinking configuration. One shape, because both backends take it.

        Adaptive is not a rename of the budget: the model decides how much to think
        per turn and `output_config.effort` sets the depth it aims for. The old
        `budget_tokens` shape failed silently at both ends -- ignored rather than
        refused -- which is exactly how a `medium` run and a `high` run came out
        indistinguishable. See config.THINKING_LEVELS for the measurements.

        `display: "summarized"` is explicit because the default is `omitted` on the
        current Claude models: thinking blocks would arrive with empty text, and the
        CLI that streams them would show a long silence instead of the reasoning.
        SGLang accepts the field, warns that it cannot suppress reasoning, and
        emits it anyway -- which is what we want from it regardless.
        """
        if not self.thinking:
            # Off has to be said. Both backends think by default -- the Claude
            # models adaptively, qwen because its template defaults to xhigh -- so
            # omitting the field asks for the most thinking, not none.
            return {"thinking": {"type": "disabled"}}
        return {"thinking": {"type": "adaptive", "display": "summarized"},
                "output_config": {"effort": self.thinking}}

    def _budget(self) -> int:
        # Nothing to leave room for any more: effort has no token budget of its own,
        # max_tokens bounds the whole turn, and the model paces itself inside it.
        return self.max_tokens

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

        `httpx.TransportError` is caught alongside the SDK's own errors because
        the SDK does not wrap it here. It converts what happens while *sending*
        the request, but a stream that dies while being *read* -- a server that
        closes mid-body, the `RemoteProtocolError: incomplete chunked read` a
        local inference server produces when it drops a decode -- raises straight
        out of the iterator. Uncaught, that ended runs sixty turns deep over one
        dropped connection.
        """
        delay = 2.0
        deadline = time.monotonic() + self.RETRY_DEADLINE
        for attempt in range(5):
            # Per attempt, not per request: the previous attempt relayed part of a
            # tool call before it died, and diffing this attempt's code against
            # that leftover snapshot would relay none of it.
            self._code_seen = ""
            try:
                with self.client.messages.stream(**kw) as stream:
                    for event in stream:
                        self._relay(event)
                    return stream.get_final_message()
            except (anthropic.RateLimitError, anthropic.InternalServerError,
                    anthropic.APIConnectionError, httpx.TransportError) as e:
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

    def sample(self, system: str, messages: list[dict], *, tools: bool = True) -> Turn:
        """Sample one turn, retrying a turn that reasoned itself out of room.

        A reasoning model can spend its whole output budget thinking and stop before
        emitting the tool call. That is not the model declining to act, so widen the
        budget and ask again rather than treating it as the end of the run.

        `tools=False` withholds the tool for the final answer turn, where the
        deliverable is prose. The same retry applies, against text rather than a
        tool call: an answer that never arrived is the same failure.
        """
        budget = self._clamp(self._budget(), system, messages)
        for _ in range(3):
            turn = self._sample_once(system, messages, budget, tools=tools)
            # A lone `done` call is a produced turn: the model acted, it simply
            # acted by finishing. Reading it as truncation would re-ask, at full
            # price, for a turn that already said everything it meant to.
            produced = (turn.tool_use_id or turn.done) if tools else turn.text.strip()
            if produced or turn.stop_reason != "max_tokens":
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

    def _tools(self) -> list[dict]:
        if self.offer_done:
            return [PYTHON_TOOL, DONE_TOOL]
        return [PYTHON_TOOL_SUBAGENT]

    def _sample_once(self, system: str, messages: list[dict], budget: int,
                     *, tools: bool = True) -> Turn:
        self._code_seen = ""
        params, extra = self._sampling_kw()
        thinking = self._thinking_kw()
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
            tools=self._tools() if tools else anthropic.NOT_GIVEN,
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

        # The finish is extracted before anything competes for the acted-on slot.
        # It is not a call to run: it is the turn saying it is the last one, and a
        # `python` block beside it still deserves to run. Only the top level is
        # offered it; for a subagent a `done` block is a stray like any other, and
        # falls through to the correction below -- guessing at the value it meant
        # to return would fabricate the subagent's whole result.
        if self.offer_done:
            finish = next((b for b in tool_blocks if b.name == "done"), None)
            if finish is not None:
                turn.done = True
                turn.done_id = finish.id
                tool_blocks = [b for b in tool_blocks if b is not finish]

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
            if isinstance(acted.input, dict):
                turn.timeout = _cell_timeout(acted.input.get("timeout"))
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
        # A 300-word summary is not what deep reasoning is for, and on a model that
        # thinks by default the whole budget can go to it -- which is the failure
        # this loop's widen-and-retry was written for. Ask for the shallow end.
        effort = {"output_config": {"effort": "low"}}
        for budget in (max_tokens, max_tokens * 3):
            with self.muted():
                resp = self._create(
                    model=model or self.backend.worker_model,
                    max_tokens=min(budget, self.backend.max_output),
                    system=system or anthropic.NOT_GIVEN,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body=extra,
                    **effort,
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
