"""Subagent spawning.

This module executes inside a kernel process, because `agent()` is called from the
model's own code. That is what makes the design simple: a subagent is an ordinary
library call, so no request/response channel back to the parent harness is needed.
The kernel that spawns becomes the harness for its children.

Each child gets its own kernel process and its own context window, seeded only with
what the parent hands it. Values return by cloudpickle; anything that will not cross
a process boundary degrades to its rendering with a loud warning rather than
silently vanishing.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor

from xagent import config

MAX_CONCURRENCY = 8

_pool: ThreadPoolExecutor | None = None
_lock = threading.Lock()
_spawned = 0


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
    discard a whole batch. Falsy, so `[r for r in results if r]` filters it out."""

    def __init__(self, label: str, message: str):
        self.label = label
        self.message = message

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"<AgentError {self.label}: {self.message}>"


class Handle:
    """A running subagent. `result()` blocks; `gather()` blocks over many."""

    def __init__(self, future: Future, label: str, prompt: str):
        self._future = future
        self.label = label
        self.prompt = prompt
        self.turns = 0
        self.tokens = 0
        self.seconds = 0.0

    @property
    def running(self) -> bool:
        return not self._future.done()

    def result(self, timeout: float | None = None):
        try:
            run = self._future.result(timeout=timeout)
        except BaseException as e:
            # BaseException, not Exception: SystemExit and KeyboardInterrupt are
            # both reachable here and would otherwise discard every sibling result.
            return AgentError(self.label, f"{type(e).__name__}: {e}")
        self.turns = run.turns
        self.seconds = run.seconds
        usage = getattr(run, "usage", None)
        self.tokens = (usage.input + usage.output) if usage else 0
        if not run.ok:
            return AgentError(self.label, run.error or "failed")
        return run.value

    def __repr__(self) -> str:
        state = "running" if self.running else "done"
        extra = f" turns={self.turns} tokens={self.tokens:,}" if not self.running and self.turns else ""
        return f"<agent {self.label!r} {state}{extra}>"


def _depth() -> int:
    try:
        return int(os.environ.get("XAGENT_DEPTH", "0"))
    except ValueError:
        return 0


def _work(prompt: str, seed, model, max_turns, thinking, sampling):
    from xagent.runner import Runner

    return Runner(
        prompt,
        backend=os.environ.get("XAGENT_PROVIDER"),
        model=model,
        thinking=thinking,
        sampling=sampling,
        max_turns=max_turns,
        depth=_depth() + 1,
        seed=seed,
        is_subagent=True,
    ).run()


def spawn(prompt: str, *, seed: dict | None = None, model: str | None = None,
          max_turns: int = 30, label: str | None = None) -> Handle:
    global _spawned

    label = label or (prompt.strip().split("\n")[0][:48] or "subagent")
    future: Future = Future()

    depth = _depth()
    if depth + 1 > config.MAX_AGENT_DEPTH:
        future.set_result(_failed(f"depth limit {config.MAX_AGENT_DEPTH} reached"))
        return Handle(future, label, prompt)

    with _lock:
        _spawned += 1
        count = _spawned
    if count > config.MAX_TOTAL_AGENTS:
        future.set_result(_failed(f"agent cap {config.MAX_TOTAL_AGENTS} reached"))
        return Handle(future, label, prompt)

    if seed is not None:
        _check_seed(seed)

    backend = config.get_backend(os.environ.get("XAGENT_PROVIDER"))
    return Handle(
        _executor().submit(
            _work, prompt, seed, model or backend.worker_model, max_turns,
            os.environ.get("XAGENT_THINKING"), os.environ.get("XAGENT_SAMPLING"),
        ),
        label,
        prompt,
    )


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
