"""Capping value rendering. This module runs INSIDE the kernel process.

The premise of xagent: the interpreter namespace holds the data, the context
window holds only a one-line handle to it. That only works if it is enforced
mechanically rather than requested in a prompt, which is what this module does.

Two enforcement points, because in-kernel capping is bypassable (the model can
always call `sys.__stdout__.write`) and the harness-side one is absolute:

  1. here    -- `displayhook` renders cell results through `brief()`, and stdout
                is wrapped in a head+tail budgeted writer.
  2. kernel  -- a hard character cap on whatever comes back over the wire.
"""

from __future__ import annotations

import collections
import sys
from functools import singledispatch

MAX_CHARS = 700       # per-value display budget
PEEK_LINES = 3        # leading lines shown for text-ish values
PEEK_ITEMS = 3        # leading items shown for containers
STDOUT_HEAD = 2500    # chars of stdout forwarded verbatim
STDOUT_TAIL = 1200    # chars of stdout retained from the end

_SCI_MODULES = {"numpy", "pandas", "polars", "torch", "scipy", "xarray"}

# True only while IPython's displayhook is formatting a cell result. Outside that
# window nothing is being cached under `_N`, so advertising a handle would name a
# variable that never gets assigned.
_IN_DISPLAY = False

# Above this, containers are described from len()/type alone. Building the full
# repr first would allocate hundreds of MB just to print a one-line handle.
BIG_CONTAINER = 500


def _human(n: int) -> str:
    for unit in ("b", "kb", "mb", "gb"):
        if n < 1024 or unit == "gb":
            return f"{n:.1f}{unit}".replace(".0", "") if unit != "b" else f"{n}b"
        n /= 1024.0
    return f"{n}b"


def _tname(obj) -> str:
    return type(obj).__name__


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "…"


def _safe_repr(obj, limit: int = 100_000) -> str:
    try:
        r = repr(obj)
    except Exception as e:  # a broken __repr__ must not break the turn
        return f"<{_tname(obj)} with failing __repr__: {type(e).__name__}>"
    return r[:limit]


def _slot() -> str:
    """The name the just-displayed value is reachable under (IPython's `_N`).

    Two things matter here, and both were wrong before. The cache key is
    `displayhook.prompt_count`, NOT `execution_count` -- the latter has already
    advanced by the time formatting runs, so it names a slot one ahead of the real
    one. And outside a live displayhook call nothing is being cached at all, so
    there is no slot to name.
    """
    if not _IN_DISPLAY:
        return ""
    try:
        from IPython import get_ipython

        ip = get_ipython()
        dh = getattr(ip, "displayhook", None)
        if ip is not None and getattr(dh, "do_full_cache", False):
            count = getattr(dh, "prompt_count", None)
            if count is not None:
                return f"_{count}"
    except Exception:
        pass
    return ""


def _hint(slot: str) -> str:
    if not slot:
        return "… bind it to a name to keep it reachable"
    return f"… full value live in `{slot}` — peek({slot}) or slice it"


def _tag(slot: str) -> str:
    """Trailing handle for a header line, empty when there is no live slot."""
    return f" {slot}" if slot else ""


def _sci(obj) -> str | None:
    """Shape-first summary for array/frame types, without importing them."""
    mod = type(obj).__module__.split(".")[0]
    if mod not in _SCI_MODULES:
        return None
    slot = _slot()
    shape = getattr(obj, "shape", None)
    bits = [_tname(obj)]
    if shape is not None:
        bits.append(f"shape={tuple(shape)}")
    dtypes = getattr(obj, "dtypes", None)
    if dtypes is not None and hasattr(dtypes, "__iter__"):
        # A DataFrame's .dtypes is a Series; a Series' .dtypes is a bare scalar
        # dtype, which is not iterable.
        try:
            bits.append(f"dtypes={_clip(str(list(dtypes)), 120)}")
        except Exception:
            bits.append(f"dtype={dtypes}")
    elif dtypes is not None:
        bits.append(f"dtype={dtypes}")
    elif getattr(obj, "dtype", None) is not None:
        bits.append(f"dtype={obj.dtype}")
    head = ""
    for meth in ("head", "__str__"):
        try:
            head = str(obj.head(3)) if meth == "head" else str(obj)
            break
        except Exception:
            continue
    return f"<{' '.join(bits)}>{_tag(slot)}\n{_clip(head, MAX_CHARS)}\n{_hint(slot)}"


@singledispatch
def brief(obj) -> str:
    sci = _sci(obj)
    if sci is not None:
        return sci
    r = _safe_repr(obj)
    if len(r) <= MAX_CHARS:
        return r
    slot = _slot()
    return f"<{_tname(obj)} repr {len(r):,} chars>{_tag(slot)}\n{_clip(r, MAX_CHARS)}\n{_hint(slot)}"


@brief.register
def _(obj: str) -> str:
    if len(obj) <= MAX_CHARS:
        return _safe_repr(obj)
    slot = _slot()
    lines = obj.count("\n") + 1
    head = "\n".join(obj.split("\n")[:PEEK_LINES])
    return (
        f"<str {_human(len(obj))}, {lines:,} lines>{_tag(slot)}\n"
        f"{_clip(head, MAX_CHARS)}\n{_hint(slot)}"
    )


@brief.register
def _(obj: bytes) -> str:
    if len(obj) <= 200:
        return _safe_repr(obj)
    slot = _slot()
    return f"<bytes {_human(len(obj))}>{_tag(slot)}\n{_safe_repr(obj[:120])}\n{_hint(slot)}"


def _container(obj) -> str:
    if len(obj) <= BIG_CONTAINER:
        r = _safe_repr(obj)
        if len(r) <= MAX_CHARS:
            return r
    slot = _slot()
    items = list(obj)[:PEEK_ITEMS]
    kinds = {_tname(x) for x in items}
    of = f" of {kinds.pop()}" if len(kinds) == 1 else ""
    body = "\n".join("  " + _clip(_safe_repr(x), 160) for x in items)
    more = len(obj) - len(items)
    tail = f"\n  … {more:,} more" if more > 0 else ""
    return f"<{_tname(obj)} len={len(obj):,}{of}>{_tag(slot)}\n{body}{tail}\n{_hint(slot)}"


for _t in (list, tuple, set, frozenset):
    brief.register(_t, _container)


@brief.register
def _(obj: dict) -> str:
    if len(obj) <= BIG_CONTAINER:
        r = _safe_repr(obj)
        if len(r) <= MAX_CHARS:
            return r
    slot = _slot()
    keys = list(obj)[:8]
    shown = ", ".join(_clip(_safe_repr(k), 40) for k in keys)
    more = len(obj) - len(keys)
    first = ""
    if keys:
        k = keys[0]
        first = f"\n  {_safe_repr(k)}: {_clip(_safe_repr(obj[k]), 200)}"
    return (
        f"<dict {len(obj):,} keys>{_tag(slot)}\n  keys: {shown}"
        f"{f' … +{more:,}' if more > 0 else ''}{first}\n{_hint(slot)}"
    )


def raw_write(text: str) -> None:
    """Write straight to the real stream, bypassing the per-cell budget.

    The budget exists to protect the model's context. Harness control payloads --
    signals, the variable table, pickled return values -- are never shown to the
    model, and truncating them silently corrupts JSON and pickles rather than
    merely shortening prose. They must not share the model-facing path.
    """
    stream = sys.stdout
    inner = getattr(stream, "_inner", stream)
    inner.write(text)
    try:
        inner.flush()
    except Exception:
        pass


def safe_brief(obj) -> str:
    try:
        return brief(obj)
    except Exception as e:
        return f"<{_tname(obj)} — brief() failed: {type(e).__name__}: {e}>"


class BudgetedStream:
    """Forward the head of a cell's output, ring-buffer the tail, elide the middle.

    Head+tail rather than head-only because the interesting part of a long test
    run or build log lives at both ends.
    """

    def __init__(self, inner, head: int = STDOUT_HEAD, tail: int = STDOUT_TAIL):
        self._inner = inner
        self._head = head
        self._tailcap = tail
        self.reset()

    def reset(self) -> None:
        self._written = 0
        self._tail: collections.deque[str] = collections.deque()
        self._tail_len = 0
        self._elided = 0

    def write(self, s) -> int:
        if not isinstance(s, str):
            s = str(s)
        total = len(s)
        room = self._head - self._written
        if room > 0:
            self._inner.write(s[:room])
            self._written += min(room, total)
            s = s[room:]
        if s:
            self._elided += len(s)
            # Clip on append. Otherwise one giant write is never evicted (the
            # deque never exceeds one element) and never clipped, so a stream
            # meant to bound memory retains the whole thing.
            if len(s) > self._tailcap:
                s = s[-self._tailcap :]
            self._tail.append(s)
            self._tail_len += len(s)
            while len(self._tail) > 1 and self._tail_len - len(self._tail[0]) >= self._tailcap:
                self._tail_len -= len(self._tail.popleft())
        return total

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush_elision(self) -> None:
        if not self._elided:
            self.reset()
            return
        tail = "".join(self._tail)[-self._tailcap :]
        self._inner.write(f"\n… {self._elided:,} chars of output elided …\n{tail}")
        try:
            self._inner.flush()
        except Exception:
            pass
        self.reset()

    def __getattr__(self, name):
        return getattr(self._inner, name)


def install() -> None:
    """Route cell results through brief() and cap stdout. Idempotent."""
    from IPython import get_ipython

    ip = get_ipython()
    if ip is None or getattr(ip, "_xa_brief_installed", False):
        return

    dh = ip.displayhook

    # Patching compute_format_data (rather than replacing sys.displayhook) keeps
    # IPython's own `_N` / `Out[N]` bookkeeping -- which is exactly the handle
    # name brief() advertises -- and keeps the proper iopub execute_result path.
    def _format(result):
        global _IN_DISPLAY
        _IN_DISPLAY = True
        try:
            return {"text/plain": safe_brief(result)}, {}
        finally:
            _IN_DISPLAY = False

    dh.compute_format_data = _format

    stream = BudgetedStream(sys.stdout)
    sys.stdout = stream
    ip.events.register("pre_run_cell", lambda info: stream.reset())
    ip.events.register("post_run_cell", lambda result: stream.flush_elision())
    ip._xa_brief_installed = True
