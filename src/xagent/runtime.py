"""The in-kernel runtime: the surface the model programs against.

Everything here executes inside the kernel process, in the same namespace as the
model's own code. Harness-facing state lives in `_STATE` and is drained once per
turn by a `probe()` that never appears in the model's transcript -- so control
signals cost zero context.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from xagent.audio import Sound
from xagent.brief import MAX_CHARS, raw_write, safe_brief
# Safe at module scope: spawn imports runtime only inside _check_seed().
from xagent.spawn import MAX_TURNS as SUBAGENT_MAX_TURNS, AgentError


def _is_subagent() -> bool:
    """Whether a program is waiting for this kernel, rather than a person.

    Deliberately not `_depth() > 0`. Depth answers a different question -- how far
    down the spawn chain this kernel sits -- and the two only happen to coincide.
    Keying the finishing contract on the coincidence made a Runner whose role and
    depth disagreed fail silently: its done() would never signal and the run would
    spin to the turn limit. The role is set explicitly by the harness, always.
    """
    return os.environ.get("XAGENT_ROLE") == "subagent"

# The markers that fence a control payload off from the model's own output on the
# way back over the wire. \x1e (RS) is a control character no ordinary program
# prints, and the nonce -- fresh for every cell, pushed in just before it runs --
# is what stops model code that happens to print a lookalike from being mistaken
# for the harness talking to itself.
CTL_BEGIN = "\x1e<<XACTL:{}>>"
CTL_END = "<<XACTL-END:{}>>\x1e"

_STATE: dict = {
    "done": None,        # one-tuple once done() is called, so None stays a legal answer
    "finish": None,      # which verb ended the run, for the harness to report
    "notes": {},
    "compress": None,
    "ctx": {},
    "spawned": 0,
    # Messages the parent agent has sent, waiting for the inbox() cell the harness
    # runs at the next turn boundary.
    "inbox": [],
    # Armed by the harness before each model cell, disarmed by the hook that reads
    # it. Absent means "this cell is not one the harness is listening to", which is
    # true of every probe -- and is what keeps a probe's stdout free of payloads.
    "turn": None,
    # Filesystem mutations, recorded as they happen. Which files a session touched
    # is a fact, and facts belong in a ledger rather than in summary prose -- the
    # same reasoning that makes the variable table authoritative over the summary.
    "files": {},
}

IGNORE_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "dist", "build", ".tox", ".idea", ".xagent",
}

_TOOL_NAMES: list[str] = []


# --------------------------------------------------------------------- values


@dataclass
class Result:
    """Outcome of a shell command."""

    cmd: str
    code: int
    out: str = ""
    err: str = ""

    @property
    def ok(self) -> bool:
        return self.code == 0

    def lines(self) -> list[str]:
        return self.out.splitlines()

    def __repr__(self) -> str:
        tag = "ok" if self.ok else f"exit={self.code}"
        body = (self.out or self.err).strip()
        first = body.split("\n")[0][:160] if body else ""
        size = f", {len(body):,} chars" if len(body) > MAX_CHARS else ""
        return f"<Result {tag}{size}> {first}"


@dataclass
class Hit:
    """One grep match."""

    path: str
    line: int
    text: str

    def __repr__(self) -> str:
        return f"{self.path}:{self.line}: {self.text.strip()[:120]}"


# ------------------------------------------------------------------ filesystem


# Reading any of these puts a live credential into the model's context, into the
# CLI's stdout, and into any subagent it is seeded to. Blocked by default.
SECRET_NAMES = {".env", ".netrc", ".npmrc", ".pypirc", "credentials", "id_rsa", "id_ed25519"}
SECRET_PARTS = {".ssh", ".aws", ".gnupg", ".config/gcloud"}


def _guard_secret(p: Path) -> None:
    parts = set(p.parts)
    if p.name in SECRET_NAMES or parts & {s.split("/")[0] for s in SECRET_PARTS}:
        raise PermissionError(
            f"{p} looks like a credential store and is blocked. If you genuinely "
            f"need it, the operator must read it for you."
        )


def _resolve(path) -> Path:
    p = Path(path).expanduser()
    _guard_secret(p)
    return p


def read(path, lines: tuple[int, int] | None = None) -> str:
    """Read a file. Returns the full text; display is capped, the value is not."""
    p = _resolve(path)
    text = p.read_text(errors="replace")
    if lines:
        start, end = lines
        if start < 1:
            raise ValueError(f"line numbers are 1-based; got start={start}")
        text = "\n".join(text.splitlines()[start - 1 : end])
    return text


def write(path, text: str) -> str:
    """Create or overwrite a file; returns a one-line confirmation."""
    p = _resolve(path)
    existed = p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    _record(p, "modified" if existed else "created", len(text))
    return f"wrote {p} ({len(text):,} chars, {text.count(chr(10)) + 1:,} lines)"


def edit(path, old: str, new: str, count: int = 1) -> str:
    """Replace `old` with `new`. Raises unless `old` occurs exactly `count` times."""
    p = _resolve(path)
    # newline="" preserves the file's existing line endings; the default would
    # rewrite every line of a CRLF file and turn a one-line edit into a full diff.
    with open(p, newline="") as fh:
        raw = fh.read()
    found = raw.count(old)
    if found != count:
        raise ValueError(
            f"{p}: expected {count} occurrence(s) of the target text, found {found}. "
            "Widen the snippet until it is unique."
        )
    updated = raw.replace(old, new)
    with open(p, "w", newline="") as fh:
        fh.write(updated)
    _record(p, "modified", len(updated))
    return f"edited {p} ({found} replacement(s))"


def _record(p: Path, action: str, size: int) -> None:
    entry = _STATE["files"].setdefault(str(p), {"action": action, "edits": 0, "bytes": 0})
    # A file created then edited is still "created" by this session.
    if entry["action"] != "created":
        entry["action"] = action
    entry["edits"] += 1
    entry["bytes"] = size


def files_touched() -> dict:
    """Every path this session wrote or edited, with how many times."""
    return dict(_STATE["files"])


def _file_ledger() -> str:
    entries = _STATE["files"]
    if not entries:
        return ""
    rows = []
    for path, meta in sorted(entries.items()):
        times = "" if meta["edits"] == 1 else f", {meta['edits']} writes"
        rows.append(f"  {meta['action']:<9} {path}  ({meta['bytes']:,} bytes{times})")
    return "\n".join(rows)


def _emit_file_ledger() -> None:
    raw_write(_file_ledger())


def ls(path=".", depth: int = 1) -> list[str]:
    """List a directory; dotfiles and noise dirs are omitted."""
    root = _resolve(path)
    out: list[str] = []
    for entry in sorted(root.iterdir()):
        if entry.name in IGNORE_DIRS or entry.name.startswith("."):
            continue
        if entry.is_dir():
            out.append(f"{entry.name}/")
            if depth > 1:
                out.extend(f"{entry.name}/{c}" for c in ls(entry, depth - 1))
        else:
            out.append(entry.name)
    return out


def _walk(root: Path, glob: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        for name in filenames:
            full = Path(dirpath) / name
            rel = full.relative_to(root)
            if fnmatch.fnmatch(str(rel), glob) or fnmatch.fnmatch(name, glob):
                yield full, rel


def grep(pattern: str, glob: str = "*", path=".", ignore_case: bool = False,
         fixed: bool = False, max_hits: int = 5000) -> list[Hit]:
    """Search file contents. Returns every hit; bind it and reduce in Python."""
    root = _resolve(path)
    flags = re.IGNORECASE if ignore_case else 0
    rx = re.compile(re.escape(pattern) if fixed else pattern, flags)
    hits: list[Hit] = []
    for full, rel in _walk(root, glob):
        if full.name in SECRET_NAMES:
            continue
        try:
            with full.open(errors="replace") as fh:
                for n, line in enumerate(fh, 1):
                    if rx.search(line):
                        hits.append(Hit(str(rel), n, line.rstrip("\n")))
                        if len(hits) >= max_hits:
                            return hits
        except (OSError, UnicodeDecodeError):
            continue
    return hits


def files(glob: str = "*", path=".") -> list[str]:
    """Paths matching a glob, ignoring the usual noise directories."""
    return [str(rel) for _, rel in _walk(_resolve(path), glob)]


def sh(cmd: str, timeout: float = 120, cwd=None) -> Result:
    """Run a shell command. Never raises on non-zero exit; inspect `.code`."""
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=str(cwd) if cwd else None,
        )
        return Result(cmd, proc.returncode, proc.stdout, proc.stderr)
    except subprocess.TimeoutExpired as e:
        return Result(cmd, 124, e.stdout or "", f"timed out after {timeout}s")


# ------------------------------------------------------------ context control


def peek(x, n: int = 40, width: int = 200) -> None:
    """Print a slice of a held value. Deliberately bypasses the display cap."""
    if isinstance(x, str):
        body = x.splitlines()
        shown, total, unit = body[:n], len(body), "lines"
    elif isinstance(x, dict):
        items = list(x.items())[:n]
        shown = [f"{k!r}: {v!r}" for k, v in items]
        total, unit = len(x), "keys"
    elif isinstance(x, (list, tuple, set, frozenset)):
        seq = list(x)
        shown, total, unit = [repr(i) for i in seq[:n]], len(seq), "items"
    elif hasattr(x, "__iter__") and not isinstance(x, bytes):
        # Never list() an arbitrary iterable: it consumes generators (destroying
        # the value being inspected) and never returns on an unbounded one.
        import itertools

        head = list(itertools.islice(iter(x), n))
        shown, total, unit = [repr(i) for i in head], len(head), "items (lazy — not consumed past this)"
    else:
        shown, total, unit = [repr(x)], 1, "value"
    # Write past the cell budget deliberately -- but report the true count. The
    # previous version printed through the budget while claiming full coverage,
    # so the model was told it had seen everything when it had seen the head.
    body, used, budget = [], 0, 8000
    for i, line in enumerate(shown):
        clipped = line[:width]
        if used + len(clipped) > budget:
            shown = shown[:i]
            break
        body.append(clipped)
        used += len(clipped) + 1
    header = f"[peek {len(body)}/{total} {unit}]"
    if len(body) < min(n, total):
        header += " — stopped at the peek budget; slice it directly for more"
    raw_write(header + "\n" + "\n".join(body) + "\n")
    if total > len(body):
        raw_write(f"… {total - len(body):,} more {unit}\n")


def listen(source="speaker", seconds: float = 15, show: bool = True,
           width: int = 960, transcribe: bool = False, lang: str = "auto") -> Sound:
    """Look at sound: the speakers, the microphone, or an audio file.

    `source` is "speaker" (what this machine is playing right now), "mic", a
    device name the sound server knows, or the path of an audio file -- in which
    case `seconds` is ignored and the whole file is read.

    With `show`, the analysis panels are displayed, which attaches them to this
    cell for you to read on the next turn: waveform, log-frequency spectrogram and
    level over time on one shared axis, plus how long the clip spends at each
    level. The samples stay in the kernel, so `zoom(a, b)` re-renders any span of
    it and `.samples` is there for arithmetic the picture cannot answer.
    """
    from xagent.audio import Clip

    p = Path(str(source)).expanduser()
    if p.exists():
        clip = Clip.from_path(_resolve(source))
        label = str(p)
    else:
        from xagent import capture

        clip = capture.record(source, seconds)
        label = f"{source} {clip.seconds:.1f}s"
    sound = Sound(clip=clip, source=label)
    if transcribe:
        from xagent.asr import transcribe_clip

        sound.text = transcribe_clip(clip, lang=lang, on_text=raw_write).text
    return sound.show(width=width) if show else sound


def note(key: str, text: str | None = None):
    """Pin a fact to the top of the context. Survives every compaction.

    Use it for anything you must not lose: the goal, a constraint you discovered,
    a decision you made. Call with no text to read a note back.
    """
    if text is None:
        return _STATE["notes"].get(key)
    _STATE["notes"][str(key)] = str(text)
    return f"noted {key!r} ({len(_STATE['notes'])} note(s) pinned)"


def ctx() -> None:
    """Print the live token budget: what your context is costing, and where."""
    info = _STATE.get("ctx") or {}
    if not info:
        print("[no accounting yet — available from the second turn onward]")
        return
    used, budget = info.get("used", 0), info.get("budget", 1)
    pct = 100.0 * used / max(budget, 1)
    bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
    print(f"context  {bar} {used:,} / {budget:,} tokens ({pct:.0f}%)")
    print(f"cells    {info.get('live', 0)} live, {info.get('folded', 0)} folded, "
          f"{info.get('evicted', 0)} evicted")
    print(f"notes    {len(_STATE['notes'])} pinned")
    if info.get("cache_read") is not None:
        print(f"cache    {info['cache_read']:,} read, {info.get('cache_write', 0):,} written")
    top = info.get("heaviest") or []
    if top:
        print("heaviest cells:")
        for cell_id, tokens in top:
            print(f"  cell {cell_id}: ~{tokens:,} tokens")
    if pct >= 70:
        print("→ consider compress() — your variables survive it intact")


def compress(mode: str = "evict", before: int | None = None) -> str:
    """Queue a compaction, applied after this cell. Variables stay live and exact.

    mode="fold"   drop the *outputs* of old cells, keep the code (cheap, reversible
                  in effect: you can re-run anything).
    mode="evict"  replace old cells with a summary of what the code did plus a
                  freshly introspected table of every live variable.

    `before` limits it to cells older than that id; the default protects the most
    recent cells so the cache-warm tail of the prompt is left alone.
    """
    if mode not in ("fold", "evict"):
        raise ValueError("mode must be 'fold' or 'evict'")
    _STATE["compress"] = {"mode": mode, "before": before}
    return f"[compression queued: {mode}] — applied after this cell completes"


def done(value=None):
    """Finish a subagent run, handing `value` back to the parent's Python.

    Who is waiting decides how a run ends, and the two callers are not alike. A
    subagent reports to a program, and its result is an object that lives in this
    kernel -- often one it has never seen whole, because the display caps it. So
    it finishes here, by reference: name the value and cloudpickle carries it
    across the process boundary intact.

    The top-level agent reports to a person, and its answer is prose it has to
    write out either way. There is nothing for a value channel to carry, so that
    run finishes through the `done` tool instead, and this function does nothing
    at depth 0 but say so.
    """
    if not _is_subagent():
        # Deliberately not raising and not signalling. A model reaching for the
        # old spelling out of habit needs redirecting, not a traceback, and it
        # must not be able to end the run from in here -- the run ends on the
        # tool call, which is the one place the harness can see it coming.
        extra = "" if value is None else (
            " The value would be read by nobody: write what matters into the answer."
        )
        return ("[done() does not finish this run — write the answer as ordinary text "
                "and call the `done` tool in the same turn.]" + extra)
    _STATE["done"] = (value,)
    _STATE["finish"] = "done"
    # A parent may well read this one: it says what crossed back.
    return f"[done] {safe_brief(value)}"


# ---------------------------------------------------------------- subagents


def agent(prompt: str, *, seed: dict | None = None, model: str | None = None,
          max_turns: int = SUBAGENT_MAX_TURNS, label: str | None = None):
    """Spawn a subagent with its own kernel and its own context window.

    It starts fresh: it sees `prompt` and whatever you hand it in `seed`, nothing
    else. Returns immediately with a Handle; call `.result()` or `gather()`.

        hs = [agent(f"audit {f} for SQL injection", seed={"src": read(f)}) for f in fs]
        for r in gather(hs): ...

    Your context grows by one line no matter how many you spawn -- that is the
    point. Whatever the subagent passes to `done()` comes back as a real Python
    object, so you can filter and re-dispatch without re-reading anything.

    It runs under no wall clock. While it works, `poll()` shows you where it has
    got to, `send()` puts a message in front of it at its next turn, and `kill()`
    stops it -- none of the three blocks.
    """
    from xagent.spawn import spawn

    return spawn(prompt, seed=seed, model=model, max_turns=max_turns, label=label)


def gather(handles, timeout: float | None = None) -> list:
    """Block until every handle resolves; results keep the input order.

    There is no wall clock on a subagent: with no `timeout` this waits for exactly
    as long as the work takes. `timeout` is a deadline for the whole batch rather
    than for each handle in turn, and `timeout=0` polls once without waiting. A
    handle the deadline caught is still running -- its slot holds an AgentError
    that says so, and gathering it again picks the wait back up.

    A failed subagent yields an AgentError in its slot rather than raising, so one
    bad apple does not discard the batch. An interrupt is the exception: it means
    this cell is being cut short, so it comes straight back out rather than being
    absorbed into a slot while the kernel goes on waiting for the rest.
    """
    handles = list(handles)
    if timeout is not None and timeout < 0:
        raise ValueError(f"timeout must not be negative, got {timeout}")
    # `is not None`, because `timeout=0` means "do not wait" and used to read as
    # "wait forever" -- the one reading nobody could have wanted.
    deadline = (time.monotonic() + timeout) if timeout is not None else None
    out = []
    for i, h in enumerate(handles):
        left = max(0.0, deadline - time.monotonic()) if deadline is not None else None
        try:
            out.append(h.result(timeout=left))
        except KeyboardInterrupt:
            # Said out loud, because the interrupt is usually the cell timeout
            # rather than a person, and a batch that vanished with it read as
            # work that had been lost.
            # Everything read defensively: this runs inside the interrupt handler,
            # and an AttributeError raised here would replace the interrupt the
            # branch exists to preserve.
            live = [getattr(x, "label", "?") for x in handles[i:]
                    if not getattr(x, "done", False)]
            print(f"[gather] interrupted with {len(live)} subagent(s) still running: "
                  f"{', '.join(live[:3])}{'…' if len(live) > 3 else ''}. They are "
                  f"untouched and still working — poll() to watch them, gather() "
                  f"again to collect them.")
            raise
    return out


def poll(handles=None) -> None:
    """Print how every subagent is doing. Never waits for any of them."""
    from xagent.spawn import status_table

    print(status_table(handles))


def send(handle, text: str) -> str:
    """Send a message to a running subagent; it reads it at its next turn."""
    return handle.send(text)


def kill(handle, reason: str = "") -> str:
    """Stop a subagent now, keeping the work it had already done."""
    return handle.kill(reason)


def inbox() -> None:
    """Read what the parent agent has sent you, and clear it.

    The harness runs this as a cell of its own at a turn boundary whenever a
    message has arrived, so you will usually be reading its output rather than
    calling it -- but calling it costs nothing and is how you check.

    What it prints is your parent talking to you mid-run: instructions that arrived
    after the task you were given, and that outrank it where they disagree.
    """
    if not _is_subagent():
        print("[inbox] nothing sends here — you are the top-level agent, and the "
              "person you work for talks to you through the task.")
        return
    pending, _STATE["inbox"] = _STATE["inbox"], []
    if not pending:
        print("[inbox] no messages from your parent.")
        return
    print(f"[inbox] {len(pending)} message(s) from the parent that spawned you:")
    for message in pending:
        print(f"  {message}")


# ------------------------------------------------------- harness-side hooks


def _user_vars() -> dict:
    ip = _ipython()
    if ip is None:
        return {}
    hidden = set(_TOOL_NAMES) | {"In", "Out", "get_ipython", "exit", "quit", "open"}
    return {
        k: v
        for k, v in ip.user_ns.items()
        if not k.startswith("_") and k not in hidden and not callable_module(v)
    }


def callable_module(v) -> bool:
    """True for modules and for the injected tool callables themselves.

    Deliberately not `v.__module__ == "xagent.runtime"`: an *instance* of a
    dataclass defined here answers that too (attribute lookup falls through to the
    class), which hid every Result and Hit the model held from the variable table.
    """
    if isinstance(v, type(sys)):
        return True
    return callable(v) and getattr(v, "__module__", None) == "xagent.runtime"


def _var_table() -> str:
    """A factual inventory of live state. Regenerated on demand, so it cannot lie."""
    rows = []
    for name, val in sorted(_user_vars().items()):
        try:
            desc = safe_brief(val).split("\n")[0][:150]
        except Exception:
            desc = f"<{type(val).__name__}>"
        size = ""
        try:
            size = f" len={len(val):,}" if hasattr(val, "__len__") else ""
        except Exception:
            pass
        rows.append(f"  {name:<22} {type(val).__name__:<12}{size}  {desc}")
    if not rows:
        return "  (no user variables)"
    return "\n".join(rows)


def _signals_dict() -> dict:
    """Drain pending control signals. Called by the harness, never by the model."""
    payload = {
        "done": _STATE["done"] is not None,
        "done_brief": safe_brief(_STATE["done"][0]) if _STATE["done"] else None,
        "finish": _STATE["finish"],
        "compress": _STATE["compress"],
        "notes": dict(_STATE["notes"]),
        "spawned": _spawn_count(),
        "vars": len(_user_vars()),
    }
    _STATE["compress"] = None
    return payload


def _signals() -> str:
    return json.dumps(_signals_dict())


def _emit_signals() -> None:
    """Hand the harness its control payload, off the budgeted path."""
    raw_write(_signals())


def _emit_var_table() -> None:
    raw_write(_var_table())


def _arm_turn(info: dict, want_vars: bool, nonce: str) -> None:
    """Prime this kernel to report on the next cell it runs.

    Pushed in ahead of the cell rather than probed for afterwards. The accounting
    is stored first, so a table that fails to render later (a `__len__` that
    raises, a monstrous namespace) does not also cost the model its ctx().
    """
    _set_ctx(info)
    _STATE["turn"] = {"want_vars": want_vars, "nonce": nonce}


def _turn_payload(want_vars: bool) -> dict:
    """Everything the harness needs after a cell, in one payload.

    Three probes stood here once -- signals, ledger, table -- each a serial
    round-trip into this process, paid every turn. The table is rendered only when
    asked for: until history has been evicted the cells themselves still say what
    is live, and walking the namespace to say it again is the expensive half.
    """
    return {
        "v": 1,
        "signals": _signals_dict(),
        "files": _file_ledger(),
        "vars": _var_table() if want_vars else None,
    }


def _post_cell(result=None) -> None:
    """Emit the turn payload after a model cell. Registered on `post_run_cell`.

    Disarms first, before anything that can fail: a leftover arm would attach a
    payload to the next cell that ran, and the next cell is very often a probe
    whose stdout is parsed as JSON or a pickle.
    """
    arm = _STATE["turn"]
    _STATE["turn"] = None
    if not arm:
        return
    nonce = arm["nonce"]
    try:
        body = json.dumps(_turn_payload(arm["want_vars"]))
    except Exception as e:
        # A complete payload that says "this failed" is recoverable; half a
        # payload is not. The harness falls back to a probe when it sees this.
        body = json.dumps({"v": 1, "error": f"{type(e).__name__}: {e}"})
    try:
        raw_write(CTL_BEGIN.format(nonce) + body + CTL_END.format(nonce))
    except Exception:
        # Nothing left to say it with. Absence reads as "no payload" and takes
        # the same fallback as an error one.
        pass


def _emit_turn_payload(want_vars: bool = False) -> None:
    """The fallback path: the same payload, fetched by probe rather than pushed."""
    raw_write(json.dumps(_turn_payload(want_vars)))


def _deliver(text: str) -> None:
    """Queue a parent's message for the inbox() cell the harness runs next.

    Pushed in by the harness on the child's behalf, ahead of that cell, for the
    same reason `_arm_turn` is: what the parent said is something the harness
    knows and this kernel does not.
    """
    _STATE["inbox"].append(str(text))


def _spawn_count() -> int:
    try:
        from xagent import spawn as _s

        return _s._spawned
    except Exception:
        return 0


def _set_ctx(info: dict) -> None:
    _STATE["ctx"] = info


def _ipython():
    try:
        from IPython import get_ipython

        return get_ipython()
    except Exception:
        return None


def inventory(include_done: bool = True) -> str:
    """The namespace listing, generated from the live objects.

    Generated rather than written down, so it cannot drift from what `install()`
    actually bound. Rendered into the system prompt at import time, and available
    in-kernel through `helpers()`.

    `include_done` is False for the top-level agent, where finishing is the `done`
    tool and this function is only a redirect. It stays bound either way -- what
    changes is whether the listing offers it as something to reach for.
    """
    import inspect
    import textwrap

    lines = []
    for obj in PUBLIC:
        if obj is done and not include_done:
            continue
        try:
            params = ", ".join(
                str(p.replace(annotation=inspect.Parameter.empty))
                for p in inspect.signature(obj).parameters.values()
            )
            rendered = f"{obj.__name__}({params})"
        except (TypeError, ValueError):
            rendered = obj.__name__
        doc = (inspect.getdoc(obj) or "").split("\n")[0]
        # Signature and summary on separate lines: `grep` and `agent` are wide
        # enough that a single aligned column would wrap, losing the alignment the
        # listing depends on to read as a REPL banner rather than a tool table.
        lines.append(f"{rendered}")
        lines.append(f"    {textwrap.shorten(doc, 66, placeholder='...')}")
    lines.append("")
    lines.append("Path, re, json are already imported; import anything else you need.")
    return "\n".join(lines)


def helpers() -> None:
    """Print the namespace listing -- the same one the system prompt carries."""
    # Matched to the role, so the listing printed here and the one in the system
    # prompt never disagree about whether done() is something to call.
    print(inventory(include_done=_is_subagent()))


PUBLIC = [
    read, write, edit, ls, files, files_touched, grep, sh,
    peek, listen, note, ctx, compress, done,
    agent, gather, poll, send, kill, inbox, helpers,
    Result, Hit, Sound, AgentError,
]


def install(ns: dict) -> None:
    """Seed the model-facing surface into the kernel's user namespace."""
    for obj in PUBLIC:
        ns[obj.__name__] = obj
        _TOOL_NAMES.append(obj.__name__)
    ns["Path"] = Path
    ns["re"] = re
    ns["json"] = json
    _TOOL_NAMES.extend(["Path", "re", "json"])

    # Runs after brief's own post_run_cell hook, which is registered first by the
    # bootstrap -- so the elision tail is flushed before the payload is appended.
    ip = _ipython()
    if ip is not None and not getattr(ip, "_xa_rt_hook", False):
        ip.events.register("post_run_cell", _post_cell)
        ip._xa_rt_hook = True

    seed_path = os.environ.get("XAGENT_SEED")
    if seed_path and Path(seed_path).is_file():
        import cloudpickle

        with open(seed_path, "rb") as fh:
            ns.update(cloudpickle.load(fh))
