"""The agent loop: render context -> sample code -> execute -> append cell."""

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



# A trailing `done()` the model typed into its prose as well as into the tool call:
# punctuation it narrated rather than part of what it was telling anyone. Finishing
# is a tool now, so nothing has to be unpicked to make the turn work -- but models
# trained on the older spelling still narrate it, and a person reading the answer
# should not be shown a function call where the last sentence belongs.
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
    error: str | None = None
    usage: object = None
    compactions: list = field(default_factory=list)
    seconds: float = 0.0
    finish: str = ""  # answered | done | max_turns | error

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
        is_subagent: bool = False,
        budget: int | None = None,
        on_event=None,
    ):
        self.task = task
        self.provider = provider or Provider(backend=backend, model=model,
                                             thinking=thinking, sampling=sampling)
        self.max_turns = max_turns
        self.depth = depth
        self.is_subagent = is_subagent
        self.seed = seed
        self.cwd = str(cwd or Path.cwd())
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
            replay_thinking=self.provider.backend.replay_thinking and bool(self.provider.thinking),
        )
        self.compressor = Compressor(self.provider, budget=self.budget)
        # Only the top level is offered `done` as a tool. A subagent finishes by
        # value from inside its kernel, and a tool argument could not carry what
        # it returns -- see runtime.done().
        self.provider.offer_done = not is_subagent
        self.kernel: Kernel | None = None
        self._seed_file: str | None = None
        # Wall-clock origin and the last context reading, both for the status line
        # every cell output carries.
        self._run_started = time.monotonic()
        self._last_used = 0
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

    def run(self) -> RunResult:
        started = time.monotonic()
        self._run_started = started
        result = RunResult()
        try:
            # The seed file is written by _kernel_env() and read at bootstrap,
            # which acquire() runs inside adopt() -- so the ordering the cold path
            # had is exactly preserved.
            self.kernel = pool.acquire(cwd=self.cwd, env=self._kernel_env())
            self._emit("kernel_ready", depth=self.depth)
            result = self._loop()
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
            if not result.answer and not self.is_subagent:
                # The prose from the last turn that wrote any is a partial answer
                # the person watched being written; a crash is no reason to
                # pretend it never arrived.
                result.answer = strip_trailing_done(self._last_text)
        finally:
            result.seconds = time.monotonic() - started
            result.usage = self.provider.usage
            result.compactions = self.compressor.events
            self._cleanup()
        return result

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
        empty = 0
        warned_stalled = False
        # Prose from a turn the harness refused to run: an answer the model has
        # already written, one rejection away from the call that finishes.
        stranded = ""

        # The most recent prose of the whole run, kept for the turn-limit exit:
        # a run that never called done() still wrote paragraphs worth salvaging.
        last_text = ""

        # The soft limit, raised a block at a time by _extend_turns().
        limit = self.max_turns
        for turn_no in range(1, self.max_turns * MAX_TURN_BLOCKS + 1):
            if turn_no > limit:
                limit = self._extend_turns(limit)
            # Announced before sampling, not after: on a long turn this header is
            # the only thing on screen until the first token arrives.
            self._emit("turn_start", n=turn_no, tokens=self.store.real_tokens)
            turn = self.provider.sample(self.store.system, self.store.messages())
            self.store.observe(turn.input_tokens + turn.cache_read + turn.cache_write)
            self.store.cache_read += turn.cache_read
            self.store.cache_write += turn.cache_write
            self._emit("turn", n=turn_no, text=turn.text, thinking=turn.thinking,
                       code=turn.code, tokens=self.store.real_tokens)
            last_text = turn.text.strip() or last_text
            self._last_text = last_text

            finishing = turn.done and not self.is_subagent
            has_cell = turn.tool_name == "python" and bool((turn.code or "").strip())

            if finishing and not has_cell:
                # The finish arrived with no cell to run, so the run ends here on
                # the prose beside it. Checked before the no-tool-call branch
                # below, because extracting the `done` block is exactly what
                # leaves this turn without a tool_use_id of its own.
                if strays := self._strays(turn):
                    # Reported, not acted on: nothing ran, and the finish is what
                    # the turn was for.
                    self._emit("warning", message=f"ignored extra tool call(s): {strays}")
                if (turn.text.strip() or stranded):
                    return self._finish(result, finish="done", text=turn.text,
                                        carried=stranded)
                # Finished having written nothing. Give _finish a cell to rewrite
                # so the recovery turn has somewhere to put the ask; no `cell`
                # event, because nothing ran for a reader to be shown.
                blank = self.store.add(
                    code="", output="", tool_use_id=turn.done_id or f"tu_{turn_no}",
                    thought=turn.text, thinking_blocks=turn.thinking_blocks,
                )
                return self._finish(result, finish="done", text=turn.text,
                                    carried=stranded, cell=blank)

            if turn.tool_use_id is None:
                # No tool call at all. Under the finishing contract the answer is
                # the prose, and the call after it is only punctuation, so a model
                # that writes the answer and stops has finished -- most of them do,
                # having been trained to end a turn on a final answer. Take it.
                # The cost of the reading is a model that stalls mid-task being
                # recorded as having answered; the alternative is nudging a model
                # that really has finished, in a loop it cannot escape.
                result.value = turn.text.strip()
                result.brief = result.value[:400]
                finish = "answered"
                if not result.value:
                    result.error = f"model returned neither code nor text (stop={turn.stop_reason})"
                    finish = "error"
                return self._finish(result, finish=finish, text=turn.text,
                                    carried=stranded)

            if turn.tool_name and turn.tool_name != "python":
                # Reached only when the turn had no `python` call at all; a stray
                # tool alongside a real one is handled after execution instead.
                stranded = turn.text.strip() or stranded
                self.store.add(
                    code="", output=f"[unknown tool {turn.tool_name!r}. {self._only_tools()}; "
                                    f"everything else is a function in the kernel, so send "
                                    f"`{turn.tool_name}(...)` as Python in the `code` "
                                    f"argument instead.]",
                    tool_use_id=turn.tool_use_id, thought=turn.text,
                    thinking_blocks=turn.thinking_blocks,
                )
                self._emit("warning", message=f"unknown tool {turn.tool_name!r}; corrected")
                continue

            if not (turn.code or "").strip():
                # A tool call arrived with empty arguments. Some Anthropic-compatible
                # servers do this intermittently. It is recoverable: say so in the
                # tool_result and let the model try again.
                empty += 1
                stranded = turn.text.strip() or stranded
                if empty > MAX_EMPTY_CALLS:
                    result.error = (f"{empty} consecutive empty tool calls from "
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
                )
                self._emit("warning", message=f"empty tool call (attempt {empty}); retrying")
                continue

            empty = 0
            # The kernel is primed to report on this cell, so what used to be two
            # blocking probes per turn rides back on the cell itself.
            want_vars = bool(self.store.state_report)
            nonce = self._push_turn_context(want_vars)
            # A cell that named its own timeout runs under it; everything else
            # runs under the kernel's default.
            cell_limit = {"timeout": turn.timeout} if turn.timeout else {}
            if turn.timeout:
                # Not a warning -- asking for longer is the supported thing to do
                # -- but worth saying, so a cell that sits silent for ten minutes
                # reads as intended rather than hung.
                self._emit("note", message=f"cell timeout {turn.timeout:g}s")
            started = time.time()
            output = self.kernel.execute(turn.code, control_nonce=nonce, **cell_limit)
            elapsed = time.time() - started
            # Stranded prose expires here, at the next cell that runs -- unless this
            # is the cell that finishes, which is the one turn it was being kept for.
            carried, stranded = stranded, ""
            rendered = output.render()
            if dropped := self._strays(turn):
                # The python call still ran; say what was dropped so the model
                # stops reaching for a tool it does not have, without costing it
                # the cell it just wrote.
                rendered += (f"\n\n[also called {dropped} in this turn — ignored. "
                             f"{self._only_tools()}.]")
                self._emit("warning", message=f"ignored extra tool call(s): {dropped}")
            cell = self.store.add(
                code=turn.code,
                output=rendered,
                tool_use_id=turn.tool_use_id or f"tu_{turn_no}",
                thought=turn.text,
                thinking_blocks=turn.thinking_blocks,
            )
            # The status line is written after the cell exists so the accounting it
            # quotes includes the cell it is attached to -- a footer that described
            # the context as it was before this output would be wrong by exactly
            # the amount the model is trying to track.
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
                # an object in that kernel, so it crosses by value here.
                result.value = self._pull_done(signals, result)
                result.brief = signals.get("done_brief") or ""
                return self._finish(result, finish="done", text=turn.text,
                                    carried=carried, cell=cell)

            if finishing:
                # A `done` block beside a cell means "run this, then finish", and
                # it means both halves -- so the cell above has already run. There
                # is nothing to pull out of the kernel: a top-level finish carries
                # no value, and `result.value` staying None is exactly right.
                return self._finish(result, finish="done", text=turn.text,
                                    carried=carried, cell=cell)

            if self.compressor.should_compact(self.store):
                for event in self.compressor.apply(self.store, self.kernel, mode="auto"):
                    self._emit("compaction", detail=str(event), requested=False)
            elif self.compressor.stalled and not warned_stalled:
                warned_stalled = True
                self._emit("warning", message=(
                    f"over the {self.budget:,}-token budget but compaction cannot "
                    f"reclaim more — the floor (system prompt, task, pinned notes, "
                    f"protected tail) is already at the watermark. Raise --budget."))

        # The run failed to finish and is reported as a failure, but the prose it
        # wrote on the way is usually a real partial answer, and throwing it away
        # left the person with an error where a paragraph they had just watched
        # being written would do. It is not reprinted -- it streamed live -- so
        # the error says where to look instead.
        result.error = (f"hit the hard ceiling of "
                        f"{self.max_turns * MAX_TURN_BLOCKS} turns "
                        f"({MAX_TURN_BLOCKS} × {self.max_turns}, compacted at each "
                        f"boundary) without finishing; the text above is a partial "
                        f"answer from a run that never finished")
        return self._finish(result, finish="max_turns", text=last_text,
                            carried=stranded)

    def _finish(self, result: RunResult, *, finish: str, text: str = "",
                carried: str = "", cell=None) -> RunResult:
        """Resolve the answer the same way on every path out of the loop.

        The exits each used to read the prose their own way, so the identical
        turn finished differently depending on which one it took -- a `done()`
        the model narrated as well as called survived into the answer when there
        was no tool call to go with it, and prose stranded by a refused turn was
        reachable from one finish and not from another. `cell` is the only thing
        that legitimately varies: the fallback turn asks by rewriting a cell
        output the model is about to read, so only a finish with a cell to spare
        can offer it.
        """
        result.finish = finish
        if self.is_subagent:
            # Its caller is a program, `value` is the whole of what crosses back,
            # and there is nobody at the other end to read a paragraph -- least of
            # all worth a request of its own.
            result.answer = ""
            return result
        # The prose the model wrote beside the call is the answer. It already
        # streamed, it cost no extra request, and asking for it again only
        # produces the same paragraph twice. The tool-free turn is the fallback
        # for a model that called `done` having said nothing -- a person is still
        # owed an answer.
        result.answer = strip_trailing_done(
            (text or "").strip() or carried
            or (self._answer(cell) if cell is not None else ""))
        return result

    def _answer(self, cell) -> str:
        """Ask for the answer a top-level run finished without writing.

        The contract is that the model writes its conclusion as ordinary text and
        calls the `done` tool in the same turn, so this is the recovery path: it
        sent the full stop with nothing in front of it. Sampling once more with
        the tools withheld leaves it nothing to do but write the answer. Subagents
        never reach here -- their caller is a program, and `value` is the answer.

        The ask replaces a cell output the model is about to read. When the finish
        arrived alone there was no cell, so the loop adds an empty one purely to
        carry this: it exists for the length of one request and is never shown.
        """
        cell.output = ("[done] the run ended with no answer written. Write it now, "
                       "as ordinary text: no tool, no code, just the answer for the "
                       "person who asked.")
        # The cell event went out with the real output, a turn's worth of screen
        # ago; the rewrite above lands after anyone reading along has been shown
        # something else. Carrying the ask on the header for the turn it provoked
        # is what stops the transcript from crediting the answer to a prompt the
        # model was never given.
        self._emit("answer_start", ask=cell.output)
        try:
            turn = self.provider.sample(
                self.store.system, self.store.messages(), tools=False)
        except Exception as e:
            # Never fatal. The work is done and the transcript holds it; losing the
            # closing paragraph is not worth discarding the run over.
            self._emit("warning", message=f"final answer turn failed ({type(e).__name__}: {e})")
            return ""
        self._emit("answer", text=turn.text)
        return turn.text.strip()

    # ----------------------------------------------------------------- helpers

    def _only_tools(self) -> str:
        """How to name the tool surface when correcting a model that missed it."""
        if self.is_subagent:
            return "The only tool is `python`"
        return "The only tools are `python` and `done`"

    def _strays(self, turn) -> str:
        """Tool calls in this turn that ran nothing, as a printable list.

        `done` never appears here: the provider extracts it before anything
        competes for the acted-on slot, because it is the turn saying it is the
        last one rather than a call that wanted running.
        """
        names = list(turn.ignored_tools)
        if turn.tool_name and turn.tool_name != "python":
            names.append(turn.tool_name)
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
