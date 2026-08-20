"""A pool of warm kernels, so spawning a subagent does not wait for one.

Starting a kernel is the slow half of `agent()`: a process spawn, channel setup,
and a wait for the interpreter to answer, together worth a second or two. A fan-out
of thirty paid that thirty times, serially, inside the parent's cell timeout --
before a single token of anyone's work.

So the expensive half is done ahead of time, on kernels that have no role yet, and
`acquire()` finishes one with `Kernel.adopt()`. What it cannot do ahead of time is
anything that depends on the run: the seed file, the depth, the provider. Those
land at adoption, in that order, because `runtime.install()` reads them at bootstrap.

This module is imported inside kernel processes as well as by the CLI, because
`agent()` is called from model code and the kernel that spawns becomes the harness
for its children. Every warm kernel is therefore a potential orphan, and three
things stop it becoming one: the pool's atexit, the graceful first half of
`Kernel.shutdown()` that lets that atexit run, and a PDEATHSIG on the kernel itself
for when the process is killed outright.
"""

from __future__ import annotations

import atexit
import os
import threading

from xagent import config
from xagent.kernel import Kernel

_warm: list[Kernel] = []
_lock = threading.Lock()
_wake = threading.Event()
_thread: threading.Thread | None = None
_closing = False


def target_size() -> int:
    """How many warm kernels to keep ready. 0 disables the pool entirely."""
    raw = os.environ.get("XAGENT_KERNEL_POOL")
    if raw is not None:
        if raw.strip().lower() in ("off", "no", "false"):
            return 0
        try:
            return max(0, int(raw))
        except ValueError:
            return config.KERNEL_POOL_SIZE
    return config.KERNEL_POOL_SIZE


def ensure_started() -> None:
    """Start warming, if the pool is enabled and not already running."""
    global _thread
    if target_size() <= 0:
        return
    with _lock:
        if _closing:
            return
        if _thread is None:
            _thread = threading.Thread(target=_replenish, name="xagent-pool",
                                       daemon=True)
            _thread.start()
            atexit.register(shutdown_pool)
    _wake.set()


def acquire(cwd, env: dict) -> Kernel:
    """A kernel ready to run, warm if one is going spare and cold otherwise.

    Never raises on account of the pool: a warm kernel that died in the queue, or
    refuses its role, is discarded and the next one tried, and an empty pool just
    means paying the startup cost the caller would have paid anyway.
    """
    if target_size() <= 0:
        return Kernel(cwd=cwd, env=env)

    ensure_started()
    while True:
        with _lock:
            kernel = _warm.pop() if _warm else None
        if kernel is None:
            return Kernel(cwd=cwd, env=env)
        _wake.set()  # one fewer in the pool; start building its replacement now
        if not kernel.is_alive():
            _discard(kernel)
            continue
        try:
            kernel.adopt(cwd, env)
        except Exception:
            _discard(kernel)
            continue
        return kernel


def _discard(kernel: Kernel) -> None:
    try:
        kernel.shutdown()
    except Exception:
        pass


def _replenish() -> None:
    while not _closing:
        _wake.wait()
        _wake.clear()
        while not _closing:
            with _lock:
                if len(_warm) >= target_size():
                    break
            try:
                kernel = Kernel(prewarm=True)
            except Exception:
                # Transient: out of file descriptors, a slow machine, a kernel
                # that would not answer. Sleeping keeps a hard failure from
                # spinning this thread against the same wall.
                _wake.wait(timeout=5.0)
                break
            keep = False
            with _lock:
                # `_closing` can have been set while that kernel was starting, and
                # a kernel added after the drain is one nobody will ever reap.
                if not _closing and len(_warm) < target_size():
                    _warm.append(kernel)
                    keep = True
            if not keep:
                _discard(kernel)
                break


def shutdown_pool() -> None:
    """Reap every warm kernel. Registered with atexit; safe to call twice."""
    global _closing
    _closing = True
    _wake.set()
    while True:
        with _lock:
            if not _warm:
                return
            kernel = _warm.pop()
        _discard(kernel)


def stats() -> dict:
    """What the pool is holding. For tests and for the CLI's diagnostics."""
    with _lock:
        return {"warm": len(_warm), "target": target_size(),
                "running": _thread is not None and _thread.is_alive()}
