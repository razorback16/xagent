"""The agent loop: render context -> sample code -> execute -> append cell.

A run is a conversation rather than a single errand. `ask()` produces one response
to one prompt and leaves the kernel and the context standing; the caller can ask
again, and the namespace the last answer was built from is still there. `run()` is
the one-prompt case, unchanged for every caller that only ever wanted that.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from xagent import config, pool
from xagent.compress import Compressor, var_table_failed
from xagent.context import ContextStore
from xagent.kernel import Kernel, is_secret_key
from xagent.prompts import SUBAGENT_CODA, SYSTEM, SYSTEM_SUBAGENT
from xagent.provider import Provider
from xagent.audio import AudioAttachment, normalize_audio
from xagent.vision import ImageAttachment, normalize_images

MAX_EMPTY_CALLS = 3

# How many turns a run gets before the harness intervenes. Not a wall: reaching it
# triggers a compaction and grants another block of the same size, because running
# out of turns is almost never the model being lost -- it is a long job whose
# context has grown, which is what compaction is for. A subagent keeps a tighter
# block of its own (see spawn.MAX_TURNS), since its job is scoped and its cost
# multiplies by the fan-out.
MAX_TURNS = 256

# The backstop the soft limit needs. A model looping unproductively would otherwise
# extend forever, compacting cheerfully at each boundary, because there is always
# something left to reclaim once new cells have accumulated. Four blocks is far
# past any real run and still terminates.
MAX_TURN_BLOCKS = 4


def fmt_duration(seconds: float) -> str:
    """Compact and readable at every scale a cell can take: 0.4s, 12s, 3m07s."""
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


DONE_EXPR = "__import__('xagent.runtime', fromlist=['_STATE'])._STATE['done'][0]"



# A trailing `done()` the model typed into its prose: punctuation it narrated rather
# than part of what it was telling anyone. Nothing has to be unpicked to make the
# turn work -- a response ends on a turn that calls nothing -- but models trained on
# the older spelling still narrate it, and a person reading the answer should not be
# shown a function call where the last sentence belongs.
TRAILING_DONE = re.compile(r"(?:\n+\s*(?:```\w*\s*)?`?done\(\s*\)`?\s*(?:```)?\s*)+\Z")


def strip_trailing_done(text: str) -> str:
    return TRAILING_DONE.sub("", text).rstrip()


@dataclass
class RunResult:
    value: object = None      # what a parent agent (or an embedding program) receives
    answer: str = ""          # the prose a person reads; top-level runs only
    brief: str = ""
    degraded: bool = False   # value is a rendering, not the object done() was given
    turns: int = 0
    # The last prose the run wrote, on every exit path. `answer` is for the person
    # a top-level run reports to and stays empty for a subagent; this is for the
    # program, so that a child that crashed, ran out of turns or was killed can
    # still tell its parent how far it got.
    last_text: str = ""
    error: str | None = None
    usage: object = None
    compactions: list = field(default_factory=list)
    seconds: float = 0.0
    finish: str = ""  # answered | done | max_turns | interrupted | killed | error

    @property
    def ok(self) -> bool:
        return self.error is None


class Runner:
    def __init__(
        self,
        task: str,
        *,
        provider: Provider | None = None,
        backend: str | None = None,
        model: str | None = None,
        thinking: str | None = None,
        sampling: str | None = None,
        cwd: str | Path | None = None,
        max_turns: int = MAX_TURNS,
        depth: int = 0,
        seed: dict | None = None,
        images=None,
        audio=None,
        is_subagent: bool = False,
        budget: int | None = None,
        on_event=None,
        channel=None,
    ):
        self.task = task
        self.provider = provider or Provider(backend=backend, model=model,
                                             thinking=thinking, sampling=sampling)
        self.max_turns = max_turns
        self.depth = depth
        self.is_subagent = is_subagent
        self.seed = seed
        self.images: list[ImageAttachment] = normalize_images(images)
        self.audio: list[AudioAttachment] = normalize_audio(audio)
        self.cwd = str(cwd or Path.cwd())
        # Whoever can reach this run while it works: the person at the terminal, or
        # the parent that spawned it. One protocol for both -- messages to deliver,
        # a switch that ends the current response, and somewhere to publish the
        # kernel so a stop can interrupt the cell it is in the middle of. None means
        # nobody is listening and nothing can arrive. See spawn._Link and
        # session.UserChannel.
        self.channel = channel
        # What a message from the channel is wrapped in. The two callers are not
        # alike and the model is owed the difference: one is the person the work is
        # for, the other is the agent that scoped it.
        self._prompt_tag = "message-from-parent" if is_subagent else "user-message"
        self.on_event = on_event or (lambda *_a, **_k: None)
        if budget is not None and budget <= 0:
            raise ValueError(f"budget must be positive, got {budget}")
        self.budget = budget or self.provider.backend.context_budget

        # The two roles finish differently, so they get different prompts rather
        # than one prompt describing both -- each would otherwise be reading how
        # the other one works.
        system = (SYSTEM_SUBAGENT + SUBAGENT_CODA) if is_subagent else SYSTEM
        self.store = ContextStore(
            task=task,
            system=system,
            images=self.images,
            audio=self.audio,
            replay_thinking=self.provider.backend.replay_thinking and bool(self.provider.thinking),
        )
        self.compressor = Compressor(self.provider, budget=self.budget)
        # Which rendering of the python tool description goes out. Only a subagent
        # is told about in-kernel `done` -- see runtime.done().
        self.provider.is_subagent = is_subagent
        self.kernel: Kernel | None = None
        self._seed_file: str | None = None
        # Wall-clock origin and the last context reading, both for the status line
        # every cell output carries.
        self._run_started = time.monotonic()
        self._last_used = 0
        # Session-wide, so the limits mean the same thing however many prompts the
        # conversation takes: turns counted from the first, and the soft ceiling
        # _extend_turns() raises a block at a time.
        self._turn_no = 0
        self._limit = max_turns
        self._empty = 0
        self._warned_stalled = False
        # How many prompts this run has answered. It starts at 1 for a session opened
        # with no task, because the first thing the person says is then a message like
        # any other -- there is no `<task>` block already holding it.
        self._asked = 0 if task.strip() else 1
        # What the loop has produced so far, reachable from run() so a crash
        # reports the work instead of an empty result.
        self._result: RunResult | None = None
        self._last_text = ""
        # Only a watched run relays deltas. Subagents build their own Provider and
        # leave this unset, so parallel fan-out cannot interleave into one stream.
        if on_event is not None:
            self.provider.on_delta = lambda part, text: self._emit(
                "delta", part=part, text=text)

    # ------------------------------------------------------------------- setup

    def _kernel_env(self) -> dict:
        # Kernel() strips credential-shaped variables itself; this mirror keeps the
        # dict we build here honest about what the child will actually see.
        env = {k: v for k, v in os.environ.items() if not is_secret_key(k)}
        env["XAGENT_DEPTH"] = str(self.depth)
        # Who is waiting, which is what decides how this kernel finishes. Set from
        # the role rather than inferred from depth: the two coincide in production
        # but answer different questions, and a kernel that guessed wrong would
        # never be able to end its own run. Always set, "top" included.
        env["XAGENT_ROLE"] = "subagent" if self.is_subagent else "top"
        env["XAGENT_PROVIDER"] = self.provider.backend.name
        env["XAGENT_MODEL"] = self.provider.model
        # Always set, `off` included: an unset variable reads as "unspecified" to
        # the child, which would then re-apply the default the parent disabled.
        env["XAGENT_THINKING"] = self.provider.thinking or config.THINKING_OFF
        env["XAGENT_SAMPLING"] = self.provider.sampling
        # Always clear it first: the environment is inherited from this kernel,
        # which may itself have been seeded, and a stale XAGENT_SEED would hand a
        # grandchild the parent's data it was never given.
        env.pop("XAGENT_SEED", None)
        if self.seed:
            import cloudpickle

            fd, path = tempfile.mkstemp(prefix="xagent-seed-", suffix=".pkl")
            with os.fdopen(fd, "wb") as fh:
                cloudpickle.dump(self.seed, fh)
            self._seed_file = path
            env["XAGENT_SEED"] = path
        return env

    def _emit(self, kind: str, **data) -> None:
        try:
            self.on_event(kind, data)
        except Exception:
            pass

    # --------------------------------------------------------------------- run

    @property
    def turns(self) -> int:
        """Turns sampled across the whole conversation, not just the last response."""
        return self._turn_no

    def start(self) -> None:
        """Acquire the kernel. Idempotent, so `ask()` can call it blindly."""
        if self.kernel is not None:
            return
        # The seed file is written by _kernel_env() and read at bootstrap, which
        # acquire() runs inside adopt() -- so the ordering the cold path had is
        # exactly preserved.
        self.kernel = pool.acquire(cwd=self.cwd, env=self._kernel_env())
        if self.channel is not None:
            self.channel.attach(self.kernel)
        self._emit("kernel_ready", depth=self.depth)

    def run(self) -> RunResult:
        """One prompt, one response, and the kernel torn down after it."""
        try:
            return self.ask(self.task)
        finally:
            self.close()

    def ask(self, text: str, *, images=None, audio=None) -> RunResult:
        """Produce one response to one prompt.

        The kernel and the context survive it, so the caller can ask again and the
        namespace the last answer was built from is still standing. The first prompt
        is already the `<task>` block at the head of the conversation; adding a mark
        for it as well would say the same thing twice and cost the cached opening.
        """
        started = time.monotonic()
        result = RunResult()
        try:
            self.start()
            if self._asked:
                if extra := normalize_images(images):
                    self.images.extend(extra)
                if clips := normalize_audio(audio):
                    self.audio.extend(clips)
                self._deliver(text)
            self._asked += 1
            result = self._loop()
        except KeyboardInterrupt:
            # Ctrl-C used to leave a traceback and nothing else: the turns, the
            # cells and the paragraph the person had just watched being written
            # all went with it. An interrupted response is a partial result, not a
            # crash, and it is reported as one.
            result = self._result or result
            result.finish = "interrupted"
            result.turns = len(self.store.cells)
            result.error = f"interrupted at turn {result.turns}"
            self._salvage(result)
        except Exception as e:
            # Everything the loop had accumulated used to die with the exception:
            # `result` was still the empty one built above, so a run that fell over
            # sixty turns in reported "0 turns" and discarded every cell and every
            # paragraph it had written. Keep the loop's own result and record the
            # failure on it.
            result = self._result or result
            result.error = f"{type(e).__name__}: {e}"
            result.finish = "error"
            result.turns = len(self.store.cells)
            self._salvage(result)
        finally:
            result.seconds = time.monotonic() - started
            result.usage = self.provider.usage
            result.compactions = self.compressor.events
        return result

    def _salvage(self, result: RunResult) -> None:
        """Keep the prose a failed or interrupted response had already written.

        A crash is no reason to pretend the paragraph the person watched being
        written never arrived, and the next prompt needs it in the context or the
        conversation reads as a question with no reply.
        """
        result.last_text = strip_trailing_done(self._last_text)
        if not self.is_subagent and not result.answer:
            result.answer = result.last_text
        if result.answer:
            self.store.add_answer(result.answer)

    def close(self) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        if self.kernel:
            # A kernel that spawned subagents has a warm pool of its own inside
            # it. Reaping it from here is the first of three defences, and the
            # only one that runs while the process is still healthy.
            if self.kernel.is_alive():
                try:
                    self.kernel.probe("import xagent.pool as _p; _p.shutdown_pool()",
                                      timeout=10)
                except Exception:
                    pass
            self.kernel.shutdown()
        if self._seed_file:
            try:
                os.unlink(self._seed_file)
            except OSError:
                pass

    def _loop(self) -> RunResult:
        result = self._result = RunResult()
        assert self.kernel is not None
        # Prose from a turn the harness refused to run: an answer the model has
        # already written, one rejection away from the turn that ends the response.
        stranded = ""

        # The most recent prose of this response, kept for the turn-limit exit: a
        # response cut off there still wrote paragraphs worth salvaging.
        last_text = ""

        ceiling = self.max_turns * MAX_TURN_BLOCKS
        while self._turn_no < ceiling:
            self._turn_no += 1
            turn_no = self._turn_no
            # The top of a turn is the end of the one before it: the last turn's
            # cells have run and nothing has been sampled yet. Checked here rather
            # than at the bottom of the body because the body has half a dozen
            # `continue`s in it, each of them the end of a turn too.
            if self.channel is not None and (stopped := self._boundary(result)):
                return stopped
            if turn_no > self._limit:
                self._limit = self._extend_turns(self._limit)
            # Announced before sampling, not after: on a long turn this header is
            # the only thing on screen until the first token arrives.
            self._emit("turn_start", n=turn_no, tokens=self.store.real_tokens)
            turn = self.provider.sample(self.store.system, self.store.messages())
            self.store.observe(turn.input_tokens + turn.cache_read + turn.cache_write)
            self.store.cache_read += turn.cache_read
            self.store.cache_write += turn.cache_write
            self._emit("turn", n=turn_no, text=turn.text, thinking=turn.thinking,
                       code=turn.code, codes=[c.code for c in turn.calls],
                       tokens=self.store.real_tokens)
            last_text = turn.text.strip() or last_text
            self._last_text = last_text

            if self.channel is not None and self.channel.stopping:
                # The stop landed while this turn was being generated. Its cells
                # have not run yet, and running them is exactly what was asked to
                # stop -- so the prose above is the last thing this response says.
                return self._killed(result)

            # Every `python` call the turn made, in order. More than one is a
            # model batching independent steps, which the kernel takes as it
            # takes any run of cells; the empty ones are dropped here so that a
            # single blank call still reaches the recovery path below.
            runnable = [c for c in turn.calls
                        if c.name == "python" and (c.code or "").strip()]

            if not turn.calls:
                # No tool call at all, which is how a response ends: the prose is
                # the answer, and there is no punctuation after it. The cost of the
                # reading is a model that stalls mid-task being recorded as having
                # answered -- but it is also what every model trained to end a turn
                # on a final answer does, and the alternative is nudging one that
                # really has finished, in a loop it cannot escape.
                result.value = turn.text.strip() or stranded
                result.brief = result.value[:400]
                finish = "answered"
                if not result.value:
                    result.error = f"model returned neither code nor text (stop={turn.stop_reason})"
                    finish = "error"
                return self._finish(result, finish=finish, text=turn.text,
                                    carried=stranded)

            if not any(c.name == "python" for c in turn.calls):
                # Reached only when the turn had no `python` call at all; a stray
                # tool alongside a real one is handled after execution instead.
                stranded = turn.text.strip() or stranded
                self.store.add(
                    code="", output=f"[unknown tool {turn.tool_name!r}. The only tool is `python`; "
                                    f"everything else is a function in the kernel, so send "
                                    f"`{turn.tool_name}(...)` as Python in the `code` "
                                    f"argument instead.]",
                    tool_use_id=turn.tool_use_id, thought=turn.text,
                    thinking_blocks=turn.thinking_blocks, turn=turn_no,
                )
                self._emit("warning", message=f"unknown tool {turn.tool_name!r}; corrected")
                continue

            if not runnable:
                # A tool call arrived with empty arguments. Some Anthropic-compatible
                # servers do this intermittently. It is recoverable: say so in the
                # tool_result and let the model try again.
                self._empty += 1
                stranded = turn.text.strip() or stranded
                if self._empty > MAX_EMPTY_CALLS:
                    result.error = (f"{self._empty} consecutive empty tool calls from "
                                    f"{self.provider.model}; giving up")
                    return self._finish(result, finish="error", text=turn.text,
                                        carried=stranded)
                self.store.add(
                    code=turn.code or "",
                    output="[no code received — the tool call arrived with empty "
                           "arguments. Resend the Python you intended to run.]",
                    tool_use_id=turn.tool_use_id,
                    thought=turn.text,
                    thinking_blocks=turn.thinking_blocks,
                    turn=turn_no,
                )
                self._emit("warning",
                           message=f"empty tool call (attempt {self._empty}); retrying")
                continue

            self._empty = 0
            # Stranded prose expires here, at the next cell that runs -- unless the
            # response ends, which is the one turn it was being kept for.
            carried, stranded = stranded, ""
            if len(runnable) > 1:
                self._emit("note", message=(f"{len(runnable)} calls in this turn; "
                                            f"running in order"))
            cell = None
            seen_ids: set[str] = set()
            blank = sum(1 for c in turn.calls
                        if c.name == "python" and not (c.code or "").strip())
            for i, call in enumerate(runnable):
                last = i == len(runnable) - 1
                # The kernel is primed to report on this cell, so what used to be
                # two blocking probes per turn rides back on the cell itself. The
                # variable table walks and renders every user variable, so a batch
                # asks for it once, on the cell whose answer is still current when
                # the turn ends.
                want_vars = bool(self.store.state_report) and last
                nonce = self._push_turn_context(want_vars)
                # A cell that named its own timeout runs under it; everything else
                # runs under the kernel's default.
                cell_limit = {"timeout": call.timeout} if call.timeout else {}
                if call.timeout:
                    # Not a warning -- asking for longer is the supported thing to
                    # do -- but worth saying, so a cell that sits silent for ten
                    # minutes reads as intended rather than hung.
                    self._emit("note", message=f"cell timeout {call.timeout:g}s")
                started = time.time()
                output = self.kernel.execute(call.code, control_nonce=nonce, **cell_limit)
                elapsed = time.time() - started
                rendered = output.render()
                if last and (dropped := self._strays(turn)):
                    # The python calls still ran; say what was dropped so the model
                    # stops reaching for a tool it does not have, without costing it
                    # the cells it just wrote.
                    rendered += (f"\n\n[also called {dropped} in this turn — ignored. "
                                 f"The only tool is `python`.]")
                    self._emit("warning", message=f"ignored extra tool call(s): {dropped}")
                if last and blank:
                    # A call with empty arguments beside calls that carried code.
                    # Nothing was lost -- there was nothing in it to run -- but the
                    # model is owed the arithmetic, or it reads one output short and
                    # cannot tell which of its calls the missing one was.
                    rendered += (f"\n\n[{blank} further python call(s) in this turn "
                                 f"arrived with no code, and ran nothing.]")
                    self._emit("warning", message=(f"{blank} empty call(s) alongside "
                                                   f"{len(runnable)} that ran"))
                # A batch shares one assistant message, so its ids have to be
                # distinct within it: a server that repeats one would otherwise
                # send two tool_use blocks the results cannot be told apart by.
                tool_use_id = call.tool_use_id or f"tu_{turn_no}_{i}"
                if tool_use_id in seen_ids:
                    tool_use_id = f"{tool_use_id}_{i}"
                seen_ids.add(tool_use_id)
                cell = self.store.add(
                    code=call.code,
                    output=rendered,
                    tool_use_id=tool_use_id,
                    # The prose and the thinking belong to the turn rather than to
                    # any one of its calls, and the batch renders as a single
                    # assistant message -- so they ride the first cell, and are
                    # neither repeated nor attributed to a cell that came later.
                    thought=turn.text if i == 0 else "",
                    thinking_blocks=turn.thinking_blocks if i == 0 else [],
                    images=output.images or [],
                    turn=turn_no,
                )
                # The status line is written after the cell exists so the accounting
                # it quotes includes the cell it is attached to -- a footer that
                # described the context as it was before this output would be wrong
                # by exactly the amount the model is trying to track.
                cell.output = rendered + "\n\n" + self._status_line(elapsed)
                result.turns = len(self.store.cells)
                self._emit("cell", n=cell.n, output=cell.output, ok=output.ok)

                payload = output.control
                if payload is None or output.control_error:
                    # No payload, or one that arrived unusable. Rare enough to pay a
                    # probe for, and never worth proceeding on a half-read one.
                    payload = self._fallback_payload(want_vars, output.control_error)
                signals = self._apply_payload(payload, want_vars)

                if request := signals.get("compress"):
                    events = self.compressor.apply(
                        self.store, self.kernel,
                        mode=request.get("mode", "evict"), before=request.get("before"),
                    )
                    for event in events:
                        self._emit("compaction", detail=str(event), requested=True)

                if signals.get("done"):
                    # The in-kernel finish, which only a subagent has: its result is
                    # an object in that kernel, so it crosses by value here. The run
                    # is over at the cell that called it, so whatever the turn
                    # batched behind that cell is not run -- said out loud, because
                    # code the model wrote and never saw execute is exactly the kind
                    # of thing a reader assumes ran.
                    if not last:
                        self._emit("note", message=(
                            f"done() ended the run; {len(runnable) - i - 1} later "
                            f"call(s) in this turn were not run"))
                    result.value = self._pull_done(signals, result)
                    result.brief = signals.get("done_brief") or ""
                    return self._finish(result, finish="done", text=turn.text,
                                        carried=carried)

            if self.compressor.should_compact(self.store):
                for event in self.compressor.apply(self.store, self.kernel, mode="auto"):
                    self._emit("compaction", detail=str(event), requested=False)
            elif self.compressor.stalled and not self._warned_stalled:
                self._warned_stalled = True
                self._emit("warning", message=(
                    f"over the {self.budget:,}-token budget but compaction cannot "
                    f"reclaim more — the floor (system prompt, task, pinned notes, "
                    f"protected tail) is already at the watermark. Raise --budget."))

        # The response never ended on an answer and is reported as a failure, but
        # the prose written on the way is usually a real partial one, and throwing
        # it away left the person with an error where a paragraph they had just
        # watched being written would do. It is not reprinted -- it streamed live --
        # so the error says where to look instead.
        result.error = (f"hit the hard ceiling of {ceiling} turns "
                        f"({MAX_TURN_BLOCKS} × {self.max_turns}, compacted at each "
                        f"boundary) without answering; the text above is a partial "
                        f"answer from a response that never ended")
        return self._finish(result, finish="max_turns", text=last_text,
                            carried=stranded)

    def _finish(self, result: RunResult, *, finish: str, text: str = "",
                carried: str = "") -> RunResult:
        """Resolve the answer the same way on every path out of the loop.

        The exits each used to read the prose their own way, so the identical turn
        finished differently depending on which one it took -- a `done()` the model
        narrated survived into the answer on one path and not another, and prose
        stranded by a refused turn was reachable from one exit and not from the next.
        """
        result.finish = finish
        # Recorded on every path, both roles: `answer` is what a person reads and
        # `last_text` is what a program is told. They diverge for a subagent, whose
        # parent gets no paragraph but does need to know how far a run that
        # produced no value had got.
        result.last_text = strip_trailing_done(self._last_text)
        if self.is_subagent:
            # Its caller is a program, `value` is the whole of what crosses back,
            # and there is nobody at the other end to read a paragraph.
            result.answer = ""
            return result
        # The prose the turn wrote is the answer. It already streamed and it cost no
        # extra request; `carried` picks up prose the harness refused to act on, so a
        # paragraph one rejected turn back is not lost.
        result.answer = strip_trailing_done((text or "").strip() or carried)
        if result.answer:
            # Recorded in the context, so the next prompt has a reply to follow and
            # the conversation does not read as a list of unanswered questions.
            self.store.add_answer(result.answer)
        return result

    # -------------------------------------------------------------- the channel

    def _boundary(self, result: RunResult) -> RunResult | None:
        """Let whoever is watching get a word in, between one turn and the next.

        A run takes as long as its job takes, so there is no wall clock to intervene
        with -- the person, or the parent, talks instead. Everything sent since the
        last turn is delivered here, and a stop that was asked for ends the response
        here rather than at whatever happened to check next.
        """
        if self.channel.stopping:
            return self._killed(result)
        for text in self.channel.drain():
            self._deliver(text)
            if self.channel.stopping:
                # A stop queued behind the message it followed. Deliver nothing
                # further; the response is over.
                return self._killed(result)
        return None

    def _deliver(self, text: str) -> None:
        """Put a message into the context as the user message it is.

        It used to arrive as a fabricated `inbox()` cell -- a person's words dressed
        as a REPL block the model never wrote, which is exactly the confusion the
        transcript should not contain. A `Mark` renders as a real `user` message, and
        compaction leaves it alone: an instruction from whoever the work is for is
        the least droppable thing in the window.
        """
        text = str(text).strip()
        if not text:
            return
        self.store.add_prompt(text, tag=self._prompt_tag)
        self._emit("prompt", text=text, tag=self._prompt_tag)

    def _killed(self, result: RunResult) -> RunResult:
        """End where the stop asked, keeping everything the response produced.

        The same salvage as the crash path in ask(): a child killed at turn forty has
        forty turns of work and a paragraph behind it, and reporting the stop with
        none of it is how a parent loses the only account of what its subagent got to
        before it changed its mind.
        """
        reason = self.channel.stop_reason
        result.finish = self.channel.stop_finish
        result.error = f"{self.channel.stop_label}: " + (reason or "no reason given")
        result.turns = len(self.store.cells)
        result.last_text = strip_trailing_done(self._last_text)
        if not self.is_subagent:
            result.answer = result.answer or result.last_text
            if result.answer:
                self.store.add_answer(result.answer)
        self._emit("note", message=(f"{self.channel.stop_label}: "
                                    f"{reason or 'no reason given'}"))
        return result

    # ----------------------------------------------------------------- helpers

    def _strays(self, turn) -> str:
        """Tool calls in this turn that ran nothing, as a printable list.

        Every `python` block in the turn runs, so what is left is a tool that does
        not exist -- including `done`, which a model trained on the older contract
        still reaches for and which is no longer offered at any depth.
        """
        names = list(turn.ignored_tools)
        names += [c.name for c in turn.calls if c.name != "python"]
        return ", ".join(sorted(set(names)))

    def _push_turn_context(self, want_vars: bool) -> str | None:
        """Prime the kernel to report on the cell about to run; return the nonce.

        Fire and forget, so it costs no round-trip: the kernel runs shell requests
        in arrival order, and this one only writes into `_STATE`. It carries the
        accounting the model's own ctx() reads, which is why it goes in rather
        than being asked for -- that is something the harness knows and the kernel
        does not.
        """
        info = self.store.accounting(self.budget)
        if not self.provider.backend.reports_cache:
            info["cache_read"] = None
        nonce = secrets.token_hex(16)
        try:
            self.kernel.push(f"import xagent.runtime as _r; "
                             f"_r._arm_turn({info!r}, {want_vars!r}, {nonce!r})")
        except Exception:
            # Nothing armed means no payload, which the fallback covers. Never
            # fatal: losing a context refresh must not end a run that is fine.
            return None
        return nonce

    def _extend_turns(self, limit: int) -> int:
        """Reaching the turn limit compacts and buys another block, rather than
        ending the run.

        A run that has spent its turns is usually a long job carrying a long
        context, not a model that has lost the plot -- and ending it there threw
        away every cell and every namespace it had built, for a limit that was
        only ever a guess at how much work the task was. So the same policy the
        budget uses fires here too: fold what can be folded, escalate to eviction
        if the context is over the watermark, and carry on with the namespace
        intact. `MAX_TURN_BLOCKS` is what still stops a run that is genuinely
        going nowhere.
        """
        events = self.compressor.apply(self.store, self.kernel, mode="auto")
        for event in events:
            self._emit("compaction", detail=str(event), requested=False)
        new_limit = limit + self.max_turns
        reclaimed = (f"compacted ({', '.join(str(e) for e in events)})" if events
                     else "nothing to compact")
        self._emit("note", message=(f"turn limit {limit} reached — {reclaimed}; "
                                    f"continuing to {new_limit} of "
                                    f"{self.max_turns * MAX_TURN_BLOCKS}"))
        return new_limit

    def _status_line(self, elapsed: float) -> str:
        """The one line of harness truth appended to every cell output.

        Three things the model cannot see from inside the kernel and used to have
        to spend a turn asking for: what time it is, how long that cell took, and
        what its context now costs. Reporting them unprompted is cheaper than the
        `ctx()` call it replaces -- roughly thirty tokens against a cell of its
        own -- and it makes context growth a trend the model watches rather than a
        number it samples occasionally and usually too late.
        """
        info = self.store.accounting(self.budget)
        used, budget = info["used"], max(info["budget"], 1)
        pct = 100.0 * used / budget
        grew = used - self._last_used
        self._last_used = used
        delta = f" {grew:+,}" if grew else ""
        parts = [
            time.strftime("%H:%M:%S"),
            f"cell {fmt_duration(elapsed)}",
            f"run {fmt_duration(time.monotonic() - self._run_started)}",
            f"ctx {used:,}/{budget:,} ({pct:.0f}%{delta})",
        ]
        if info["folded"] or info["evicted"]:
            parts.append(f"{info['live']} live/{info['folded']} folded/"
                         f"{info['evicted']} evicted")
        line = "[" + " · ".join(parts) + "]"
        if pct >= 70:
            line += "\n[context is filling — compress() when the next step allows it]"
        return line

    def _apply_payload(self, payload: dict, want_vars: bool) -> dict:
        """Fold one turn payload into the store; return its control signals.

        The file ledger rides every turn: a written file is recoverable by
        re-reading it, but only if the model still knows it exists, and it costs
        one line per file. The variable table is asked for only once something has
        been evicted, since before that the cells still say what is live. A frozen
        table inside the compacted-history block would describe the namespace as
        it was at compaction and drift from the kernel with every rebinding -- the
        exact stale-value failure the design set out to avoid.

        Each half is written only when the payload actually carried it. A payload
        that failed must leave the previous truth standing rather than replace it
        with an empty table, which would read as "no variables" when what happened
        was "could not look".
        """
        if (files := payload.get("files")) is not None:
            # Re-indented rather than trusted: unindented, these rows read as
            # prose rather than as a ledger.
            self.store.live_files = "\n".join(
                "  " + ln.strip() for ln in files.splitlines() if ln.strip())
        if want_vars and payload.get("vars") is not None:
            self.store.live_vars = self.compressor._format_var_table(payload["vars"])
        signals = payload.get("signals") or {}
        if signals.get("notes"):
            self.store.notes.update(signals["notes"])
        return signals

    def _fallback_payload(self, want_vars: bool, why: str | None = None) -> dict:
        """Fetch the payload by probe, for the rare cell that did not carry one."""
        self._emit("warning", message=(f"turn payload unavailable "
                                       f"({why or 'none emitted'}); probing instead"))
        try:
            # The table walks and renders every user variable, so a turn that asks
            # for one keeps the long timeout it had when this was the only path.
            return json.loads(self.kernel.probe(
                f"import xagent.runtime as _r; _r._emit_turn_payload({want_vars!r})",
                timeout=45 if want_vars else 30) or "{}")
        except Exception as e:
            if want_vars:
                self.store.live_vars = var_table_failed(e)
            return {}

    def _pull_done(self, signals: dict, result: "RunResult"):
        """Bring the answer across the process boundary, by value when possible."""
        try:
            return self.kernel.probe_pickle(DONE_EXPR)
        except Exception as e:
            result.degraded = True
            message = (f"done() value could not be serialized ({type(e).__name__}); "
                       f"the caller receives its rendering, not the object")
            self._emit("warning", message=message)
            print(f"[xagent] warning: {message}", file=sys.stderr)
            return signals.get("done_brief") or ""
