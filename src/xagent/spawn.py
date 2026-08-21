"""Subagent spawning.

This module executes inside a kernel process, because `agent()` is called from the
model's own code. That is what makes the design simple: a subagent is an ordinary
library call, so no request/response channel back to the parent harness is needed.
The kernel that spawns becomes the harness for its children.

Each child gets its own kernel process and its own context window, seeded only with
what the parent hands it. Values return by cloudpickle; anything that will not cross
a process boundary degrades to its rendering with a loud warning rather than
silently vanishing.

The Runner itself, though, runs in a thread of the parent's own process, and that
is what `_Link` is built on: the parent can watch a child's progress, post it a
message and stop it outright with nothing but shared memory in between. There is no
wall clock anywhere in here on purpose -- a subagent takes as long as its job takes,
and a parent that wants to intervene does it by talking rather than by timing out.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor

from xagent import config

MAX_CONCURRENCY = 8

# A subagent's turn limit. Half the top level's (runner.MAX_TURNS), because its
# job is scoped and the cost multiplies by the fan-out -- but high enough that a
# genuinely wide job is not cut off at the point where it would have returned.
MAX_TURNS = 128

_pool: ThreadPoolExecutor | None = None
_lock = threading.Lock()
_spawned = 0
# Every handle this session made, so `poll()` can be called with no arguments.
# Handles only, never runs: a finished subagent's transcript lives and dies in its
# own kernel, so what is kept here is a label and a few counters.
_handles: list["Handle"] = []


def _executor() -> ThreadPoolExecutor:
    global _pool
    with _lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(
                max_workers=MAX_CONCURRENCY, thread_name_prefix="xagent-sub"
            )
        return _pool


class AgentError:
    """A failed subagent. Returned in place of a value so one failure does not
    discard a whole batch. Falsy, so `[r for r in results if r]` filters it out.

    `still_running` separates the two things that used to read alike: a child that
    failed, and a child a gather deadline gave up on while it was still working.
    The second one is not a failure at all -- the handle is alive and can be
    gathered again -- and reporting it as one is how a batch gets thrown away.
    """

    def __init__(self, label: str, message: str, *, turns: int = 0,
                 last_text: str = "", still_running: bool = False,
                 is_timeout: bool = False):
        self.label = label
        self.message = message
        self.turns = turns
        # The last prose the child wrote before it stopped. A subagent hands its
        # parent a value and no paragraph, so on the paths where there is no value
        # -- a crash, the turn ceiling, a kill -- this is the only account of how
        # far it got, and it used to be discarded with the run.
        self.last_text = last_text
        self.still_running = still_running
        self.is_timeout = is_timeout

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"<AgentError {self.label}: {self.message}>"


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _dur(seconds: float) -> str:
    # Imported here rather than at module scope: runner imports kernel imports
    # runtime imports this module, so a top-level import would close the cycle.
    from xagent.runner import fmt_duration

    return fmt_duration(seconds)


class _Link:
    """The shared state between a running subagent and the Handle its parent holds.

    Separate from the Handle because the worker is submitted to the executor
    before the Handle exists, and both ends need the same object from the first
    instant -- a child that started running before the assignment landed would
    otherwise write its progress into nothing.

    Parent and child are threads in one process, so there is no channel here and
    nothing to serialize: a message is an append, a kill is an Event. Every field
    is written by the child's thread and read from the parent's cell, so all of it
    moves under one lock.

    This is also the Runner's message channel (see `Runner.channel`), which is what
    makes a parent's `send()` and a person typing at the terminal the same mechanism
    at the other end. The two constants below are the only place the two differ.
    """

    # How a stop is reported. A parent changing its mind is a kill; the person at
    # the terminal pressing Esc is not, and the words a model reads should not say
    # it was.
    stop_label = "killed by parent"
    stop_finish = "killed"

    def __init__(self, label: str):
        self.label = label
        self._lock = threading.Lock()
        self._messages: list[str] = []
        self._stop = threading.Event()
        self.stop_reason = ""
        self.state = "queued"
        self.turns = 0
        self.cells = 0
        self.tokens = 0
        self.last_text = ""
        self.sent = 0
        self.started = time.monotonic()
        self.ended: float | None = None
        # Published by the child's Runner once it has one, so a kill can interrupt
        # the cell the child is in the middle of instead of waiting it out.
        self.kernel = None

    # -------------------------------------------------------------- parent side

    def post(self, text: str) -> int:
        with self._lock:
            self._messages.append(text)
            self.sent += 1
            return len(self._messages)

    def stop(self, reason: str) -> None:
        with self._lock:
            self.stop_reason = reason
        # Set last: a child that wakes on the Event must find the reason already
        # written, or it reports a kill with no reason for want of a moment.
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def _elapsed(self) -> float:
        """Wall time so far. Caller holds the lock."""
        return (self.ended if self.ended is not None else time.monotonic()) - self.started

    @property
    def seconds(self) -> float:
        with self._lock:
            return self._elapsed()

    def snapshot(self) -> dict:
        """Every field a parent reads, taken under one lock rather than six.

        A repr or a status row built from separate property reads could show a
        turn count from before a boundary and a duration from after it.
        """
        with self._lock:
            return {"state": self.state, "turns": self.turns, "cells": self.cells,
                    "tokens": self.tokens, "last_text": self.last_text,
                    "sent": self.sent, "seconds": self._elapsed()}

    # --------------------------------------------------------------- child side

    def drain(self) -> list[str]:
        with self._lock:
            pending, self._messages = self._messages, []
        return pending

    def resume(self) -> None:
        """Clear a stop. Never used by a subagent -- a kill is final for a child --
        but part of the channel protocol, because the person at the terminal
        interrupts a response and then keeps the session."""
        self._stop.clear()
        with self._lock:
            self.stop_reason = ""

    def attach(self, kernel) -> None:
        with self._lock:
            self.kernel = kernel

    def begin(self) -> None:
        with self._lock:
            self.state = "running"
            self.started = time.monotonic()

    def record(self, *, turns: int = 0, tokens: int = 0, last_text: str = "") -> None:
        """The run's own final numbers, filling gaps the event stream left.

        `turns` never overwrites a count the events already produced. A RunResult
        counts *cells*, and the two are not the same number, so taking its figure at
        the end made poll()'s turn column jump the moment the child finished. It is
        still the fallback for a run that emitted no events at all, which is the
        only case where it is the better of the two.
        """
        with self._lock:
            if turns and not self.turns:
                self.turns = turns
            self.tokens = tokens or self.tokens
            self.last_text = last_text or self.last_text

    def finish(self, state: str) -> None:
        with self._lock:
            self.state = state
            self.ended = time.monotonic()
            # Dropped so a kill() arriving after the run ended cannot interrupt a
            # kernel that has since been recycled into somebody else's pool.
            self.kernel = None

    def on_event(self, kind: str, data: dict) -> None:
        """Live progress for a parent that is not waiting.

        The alternative was a parent with nothing to read until the future
        resolved, which is what made a long fan-out indistinguishable from a hung
        one. Cheap by construction: this runs on the child's thread, including
        once per streamed delta, so anything it does is paid per token.
        """
        if kind == "turn_start":
            with self._lock:
                self.turns = data.get("n", self.turns)
                self.tokens = data.get("tokens") or self.tokens
        elif kind == "cell":
            with self._lock:
                self.cells += 1
        elif kind == "turn":
            text = (data.get("text") or "").strip()
            if text:
                with self._lock:
                    self.last_text = _clip(text, 160)


class Handle:
    """A running subagent: progress to read, a mailbox to post to, a kill switch.

    `result()` blocks and `gather()` blocks over many, but nothing else here does
    -- `poll()`, `send()` and `kill()` all return at once, because a subagent runs
    for as long as its job takes and a parent that can only wait is a parent that
    finds out how it went and nothing else.
    """

    def __init__(self, future: Future, label: str, prompt: str,
                 link: "_Link | None" = None):
        self._future = future
        self._link = link if link is not None else _Link(label)
        self.label = label
        self.prompt = prompt

    # Read from the parent's cell while the child's thread writes them, so they
    # are properties over the link rather than attributes that could go stale.
    @property
    def state(self) -> str:
        state = self._link.state
        # The link is marked by the worker, and the future is the one thing that
        # cannot be out of date: a handle whose worker has not marked it yet -- or
        # was never the one doing the marking -- must not report "queued" at a
        # subagent that plainly started, or "running" at one that has stopped.
        if self._link.ended is None:
            if self._future.done():
                if self._future.cancelled():
                    return "killed"
                return "failed" if self._future.exception() is not None else "done"
            if state == "queued" and self._future.running():
                return "running"
        return state

    @property
    def running(self) -> bool:
        return not self._future.done()

    @property
    def done(self) -> bool:
        return self._future.done()

    @property
    def turns(self) -> int:
        return self._link.turns

    @property
    def cells(self) -> int:
        return self._link.cells

    @property
    def tokens(self) -> int:
        return self._link.tokens

    @property
    def seconds(self) -> float:
        return self._link.seconds

    @property
    def last_text(self) -> str:
        return self._link.last_text

    @property
    def messages_sent(self) -> int:
        return self._link.sent

    def send(self, text: str) -> str:
        """Queue a message; the subagent reads it at its next turn boundary."""
        text = str(text).strip()
        if not text:
            return "[send] nothing queued: the message is empty."
        # Both, because they do not settle together: `_work` marks the link done
        # before its future resolves, and a message posted in that window was
        # counted as queued and then never read by anyone.
        if self._future.done() or self._link.ended is not None:
            return f"[send] {self.label!r} has already finished; nothing was queued."
        waiting = self._link.post(text)
        return (f"[send] queued for {self.label!r} ({waiting} waiting) — it arrives "
                f"as a message from you at the end of its current turn.")

    def kill(self, reason: str = "") -> str:
        """Stop this subagent. Ends at its current cell, keeping what it produced."""
        try:
            reason = str(reason).strip()
            if self._future.done():
                return f"[kill] {self.label!r} has already finished; nothing to stop."
            if self._future.cancel():
                # Never started: it was still queued behind the concurrency limit,
                # so there is no run to end and no work to salvage.
                self._link.stop(reason)
                self._link.finish("killed")
                return f"[kill] {self.label!r} was cancelled before it started."
            self._link.stop(reason)
            kernel = self._link.kernel
            if kernel is not None:
                # The cell it is in the middle of dies now rather than at its own
                # timeout; the turn boundary right behind it is where the run ends.
                kernel.interrupt()
            return (f"[kill] {self.label!r} is stopping"
                    f"{f' ({reason})' if reason else ''} — result() hands you an "
                    f"AgentError naming the turn it reached.")
        except Exception as e:
            # Killing must never raise: it is the thing a parent reaches for when
            # something has already gone wrong.
            return f"[kill] could not stop {self.label!r}: {type(e).__name__}: {e}"

    def result(self, timeout: float | None = None):
        try:
            run = self._future.result(timeout=timeout)
        except CancelledError:
            # Cancelled means it never ran, so there is nothing to salvage and no
            # turn to name. Usually a kill that beat the executor to it; the other
            # way to get here is the pool shutting down under a queued job.
            return AgentError(self.label,
                              (self._kill_message() + " — cancelled before it started")
                              if self._link.stopping else "cancelled before it started",
                              turns=self._link.turns, last_text=self._link.last_text)
        except TimeoutError as e:
            if not self._future.done():
                # The deadline, not the child: it is still working. Said plainly,
                # because an AgentError here used to read as a failure and get the
                # whole batch thrown away.
                return AgentError(
                    self.label,
                    f"still running after {_dur(self._link.seconds)} (turn "
                    f"{self._link.turns}) — this is the gather deadline, not a "
                    f"failure; poll() to watch it or gather() it again",
                    turns=self._link.turns, last_text=self._link.last_text,
                    still_running=True, is_timeout=True,
                )
            return AgentError(self.label, f"TimeoutError: {e}")
        except Exception as e:
            # Exception, not BaseException: a KeyboardInterrupt here is the
            # parent's own cell being interrupted, and swallowing it left the
            # kernel running this wait while the harness reported the cell dead.
            #
            # The link is read for the salvage rather than the run, because there
            # is no run: this is the path where the worker itself died, and it was
            # the one exit that still reported a child's forty turns as nothing.
            return AgentError(self.label, f"{type(e).__name__}: {e}",
                              turns=self._link.turns,
                              last_text=self._link.last_text)
        usage = getattr(run, "usage", None)
        self._link.record(turns=run.turns,
                          tokens=(usage.input + usage.output) if usage else 0,
                          last_text=getattr(run, "last_text", ""))
        if not run.ok:
            return AgentError(self.label, self._failure_message(run),
                              turns=self._link.turns,
                              last_text=getattr(run, "last_text", ""))
        return run.value

    def _failure_message(self, run) -> str:
        """What went wrong, how far it got, and the last thing it said.

        A one-line error was all a parent used to get for a child that died a
        hundred turns in, so the work was reported as if it had never happened.
        """
        parts = [run.error or "failed"]
        if turns := self._link.turns:
            parts.append(f"stopped at turn {turns}")
        if last := getattr(run, "last_text", ""):
            parts.append(f'last wrote: "{_clip(last, 160)}"')
        return " — ".join(parts)

    def _kill_message(self) -> str:
        """Why it stopped, in the words the parent used.

        The "before it started" half belongs only to the cancelled path, and
        appending it unconditionally produced `killed by parent: not needed after
        all — cancelled before it started` for a child that had plainly been
        running for minutes. This string is read by a model.
        """
        return "killed by parent: " + (self._link.stop_reason or "no reason given")

    def __repr__(self) -> str:
        s = self._link.snapshot()
        parts = [self.state]
        if s["turns"]:
            parts.append(f"turn {s['turns']}")
        parts.append(_dur(s["seconds"]))
        if s["tokens"]:
            parts.append(f"{s['tokens']:,} tok")
        if s["last_text"]:
            parts.append(f'"{_clip(s["last_text"], 40)}"')
        # Both halves are clipped: a label is a prompt's first line and the last
        # thing said is a sentence, and a repr that wrapped would be read as two.
        return f"<agent {_clip(self.label, 32)!r} " + " · ".join(parts) + ">"


def registry() -> list[Handle]:
    """Every handle spawned in this kernel, newest last. What `poll()` defaults to."""
    with _lock:
        return list(_handles)


def status_table(handles=None) -> str:
    """A compact status line per subagent. Never waits for any of them.

    Deliberately small: this is printed into a context window, so it says what a
    parent needs to decide whether to wait, send or kill, and nothing else.
    """
    hs = list(handles) if handles is not None else registry()
    if not hs:
        return "[no subagents in this session]"
    running = sum(1 for h in hs if not h.done)
    rows = [f"[{len(hs)} subagent(s), {running} still running]"]
    for h in hs:
        s = h._link.snapshot()
        row = (f"  {_clip(h.label, 18):<18} {h.state:<7} "
               f"turn {s['turns']:<4} {_dur(s['seconds']):>7} "
               f"{s['tokens']:>8,} tok")
        if s["sent"]:
            row += f" {s['sent']}↑"
        if s["last_text"]:
            row += f'  "{_clip(s["last_text"], 52)}"'
        rows.append(row)
    return "\n".join(rows)


def _depth() -> int:
    try:
        return int(os.environ.get("XAGENT_DEPTH", "0"))
    except ValueError:
        return 0


def _work(link: _Link, prompt: str, seed, model, max_turns, thinking, sampling):
    from xagent.runner import Runner

    link.begin()
    try:
        run = Runner(
            prompt,
            backend=os.environ.get("XAGENT_PROVIDER"),
            model=model,
            thinking=thinking,
            sampling=sampling,
            max_turns=max_turns,
            depth=_depth() + 1,
            seed=seed,
            is_subagent=True,
            on_event=link.on_event,
            channel=link,
        ).run()
    except BaseException:
        link.finish("failed")
        raise
    usage = getattr(run, "usage", None)
    link.record(turns=run.turns,
                tokens=(usage.input + usage.output) if usage else 0,
                last_text=run.last_text)
    # Set before the future resolves, so a parent that sees `done` never finds the
    # state still saying "running".
    link.finish("killed" if run.finish == "killed" else "done" if run.ok else "failed")
    return run


def _register(handle: Handle) -> Handle:
    with _lock:
        _handles.append(handle)
    return handle


def spawn(prompt: str, *, seed: dict | None = None, model: str | None = None,
          max_turns: int = MAX_TURNS, label: str | None = None) -> Handle:
    global _spawned

    label = label or (prompt.strip().split("\n")[0][:48] or "subagent")
    future: Future = Future()
    link = _Link(label)

    if _depth() + 1 > config.MAX_AGENT_DEPTH:
        return _refuse(future, link, prompt,
                       f"depth limit {config.MAX_AGENT_DEPTH} reached")

    with _lock:
        _spawned += 1
        count = _spawned
    if count > config.MAX_TOTAL_AGENTS:
        return _refuse(future, link, prompt,
                       f"agent cap {config.MAX_TOTAL_AGENTS} reached")

    if seed is not None:
        _check_seed(seed)

    # Start warming here rather than at the first acquire: the checks above can
    # take a moment on a large seed, and a kernel built during them is one the
    # subagent below does not wait for.
    from xagent import pool

    pool.ensure_started()

    backend = config.get_backend(os.environ.get("XAGENT_PROVIDER"))
    return _register(Handle(
        _executor().submit(
            _work, link, prompt, seed, model or backend.worker_model, max_turns,
            os.environ.get("XAGENT_THINKING"), os.environ.get("XAGENT_SAMPLING"),
        ),
        label,
        prompt,
        link,
    ))


def _refuse(future: Future, link: _Link, prompt: str, reason: str) -> Handle:
    """A handle for a subagent that will never run.

    A refusal is still a Handle rather than an exception: a fan-out that hits the
    cap mid-list should come back with one dead slot and the rest of the batch,
    not with a traceback where the batch was.
    """
    future.set_result(_failed(reason))
    link.finish("failed")
    return _register(Handle(future, link.label, prompt, link))


def _check_seed(seed: dict) -> None:
    """Fail at spawn time, naming the offending key. Never drop it silently."""
    import cloudpickle

    from xagent import runtime

    reserved = {obj.__name__ for obj in runtime.PUBLIC} | {"Path", "re", "json"}
    if not isinstance(seed, dict):
        raise TypeError(f"seed must be a dict of names to values, got {type(seed).__name__}")
    for key, value in seed.items():
        if not isinstance(key, str) or not key.isidentifier():
            raise ValueError(f"seed key {key!r} is not a valid Python identifier")
        if key in reserved:
            raise ValueError(
                f"seed key {key!r} would shadow the built-in tool of the same name, "
                f"leaving the subagent unable to call it. Rename it, e.g. {key}_text."
            )
        try:
            cloudpickle.dumps(value)
        except Exception as e:
            raise TypeError(
                f"seed[{key!r}] ({type(value).__name__}) cannot cross a process "
                f"boundary: {type(e).__name__}: {e}. Pass a plain value instead — "
                f"e.g. read the text rather than an open file handle."
            ) from e


def _failed(message: str):
    from xagent.runner import RunResult

    return RunResult(error=message, finish="error")
