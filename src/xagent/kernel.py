"""A persistent IPython kernel, one per agent.

Out-of-process via jupyter_client so a segfault or a runaway loop in model-written
code cannot take the harness down, and so `interrupt` actually works.
"""

from __future__ import annotations

import base64
import io
import json
import os
import pickle
import queue
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from jupyter_client.kernelspec import KernelSpec
from jupyter_client.manager import KernelManager

from xagent.runtime import CTL_BEGIN, CTL_END
from xagent.vision import SUPPORTED_MEDIA_TYPES

_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

HARD_OUTPUT_CAP = 6000    # absolute backstop, in characters
# How long a model-written cell may run before the kernel is interrupted. It is a
# default, not a ceiling: the model passes `timeout` to the python tool for the
# cell it knows will take longer, so a long build or a slow download is a stated
# argument rather than a wall the run discovers by hitting it.
CELL_TIMEOUT = 180
MAX_CELL_TIMEOUT = 3600   # an hour; past this a cell should be a background job
STREAM_BUDGET = 400_000   # per-stream harness-side memory bound, in characters
# How long a timed-out cell is given to notice the interrupt and go idle. It is a
# hard bound, not a hope: a cell that ignores KeyboardInterrupt -- a C extension, a
# thread it started, code that catches BaseException -- goes on emitting output, and
# a drain that only checked its own deadline when the message queue happened to run
# dry would keep reading that output forever. Measured once at 15 minutes inside
# zmq_poll on a cell the harness had already reported as over.
DRAIN_GRACE = 5.0
# The control payload has its own budget, well clear of the stdout one: it is not
# the model's output and must not be truncated by a cell that flooded the stream.
CONTROL_BUDGET = 4_000_000

BOOTSTRAP = """
import xagent.brief as _xa_brief, xagent.runtime as _xa_rt
_xa_brief.install()
_xa_rt.install(globals())
del _xa_brief, _xa_rt
"""


SECRET_HINTS = ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "APIKEY", "CREDENTIAL", "PASSWD")


def is_secret_key(name: str) -> bool:
    upper = name.upper()
    return any(hint in upper for hint in SECRET_HINTS)


def strip_ansi(s: str) -> str:
    return _ANSI.sub("", s)


# Unpickling runs whatever the payload's __reduce__ names. For a subagent that
# payload is shaped by a context that may contain untrusted file content, and the
# unpickling happens in the *parent's* process -- so an unrestricted loads() hands
# arbitrary code execution across the isolation boundary the kernel exists to
# provide. Allow only the builtin container/scalar types by default.
_SAFE_BUILTINS = {
    "list", "dict", "set", "frozenset", "tuple", "str", "bytes", "bytearray",
    "int", "float", "bool", "complex", "NoneType", "range", "slice",
}
_SAFE_MODULES = {"builtins", "collections", "datetime", "decimal", "pathlib", "xagent.runtime"}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "builtins" and name in _SAFE_BUILTINS:
            return super().find_class(module, name)
        if module in _SAFE_MODULES and not name.startswith("_"):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"refusing to unpickle {module}.{name}: a subagent return value may only "
            f"contain plain data. Have the subagent return a dict/list of primitives."
        )


def safe_loads(blob: bytes):
    return RestrictedUnpickler(io.BytesIO(blob)).load()


def _tail_overlap(text: str, marker: str) -> int:
    """Length of the longest suffix of `text` that could start `marker`.

    A payload marker can be split across two stream messages, so the tail that
    might be half of one is held back rather than forwarded as model output.
    """
    n = min(len(text), len(marker) - 1)
    while n > 0:
        if text.endswith(marker[:n]):
            return n
        n -= 1
    return 0


class _ControlSplitter:
    """Pull nonce-fenced control payloads out of a cell's stdout as it streams.

    Runs before the stdout budget, so a cell that floods the stream cannot push
    the payload past the cap; and before `strip_ansi`, so nothing rewrites the
    markers underneath it. The model never sees a byte of this: everything between
    the markers is diverted, and the markers themselves are consumed.
    """

    def __init__(self, nonce: str):
        self.begin = CTL_BEGIN.format(nonce)
        self.end = CTL_END.format(nonce)
        self.inside = False
        self.buf: list[str] = []
        self.size = 0
        self.carry = ""
        self.payloads: list[dict] = []
        self.error: str | None = None

    def _absorb(self, chunk: str) -> None:
        if self.size + len(chunk) > CONTROL_BUDGET:
            self.error = "control payload over cap"
            return
        self.buf.append(chunk)
        self.size += len(chunk)

    def _close(self) -> None:
        body, self.buf, self.size = "".join(self.buf), [], 0
        try:
            parsed = json.loads(body)
        except Exception as e:
            self.error = f"unparseable control payload: {type(e).__name__}"
            return
        if not isinstance(parsed, dict):
            self.error = f"control payload was {type(parsed).__name__}, not an object"
            return
        self.payloads.append(parsed)

    def feed(self, text: str) -> str:
        """Take one stream chunk; return the part of it the model may see."""
        out: list[str] = []
        data, self.carry = self.carry + text, ""
        while data:
            marker = self.end if self.inside else self.begin
            cut = data.find(marker)
            if cut >= 0:
                head, data = data[:cut], data[cut + len(marker):]
                (self._absorb if self.inside else out.append)(head)
                if self.inside:
                    self._close()
                self.inside = not self.inside
                continue
            # No complete marker in what is left. Hold back only what could be
            # the beginning of one; forward the rest.
            keep = _tail_overlap(data, marker)
            if keep:
                data, self.carry = data[:-keep], data[-keep:]
            (self._absorb if self.inside else out.append)(data)
            data = ""
        return "".join(out)

    def finish(self) -> str:
        """Close the stream; return any held-back text that was never a marker."""
        if self.inside:
            # Interrupted or timed out mid-payload. Refusing a truncated payload
            # is the same rule probe() has kept all along: half a JSON document
            # fails somewhere far from here.
            self.error = self.error or "truncated control payload"
            return ""
        tail, self.carry = self.carry, ""
        return tail

    def result(self) -> tuple[dict | None, str | None]:
        # Last complete payload wins: the hook runs after the cell body, so if
        # model output happened to contain a lookalike, the real one is later.
        for payload in reversed(self.payloads):
            if payload.get("error"):
                return None, str(payload["error"])
            return payload, None
        return None, self.error


@dataclass
class CellOutput:
    stdout: str = ""
    stderr: str = ""
    result: str | None = None
    error: str | None = None
    timed_out: bool = False
    # The limit this cell ran under, so the timeout notice can name it.
    timeout: float | None = None
    # Whether the kernel actually went idle. False means the interrupt did not
    # take: the code may still be running, so the promise that the namespace is
    # quiescent -- and that the next cell starts promptly -- does not hold.
    stopped: bool = True
    # The control payload the in-kernel hook appended to this cell, already parsed
    # and already stripped out of `stdout`. None means none arrived; `control_error`
    # set means one did but could not be used, which is the harness's cue to fall
    # back to a probe rather than to proceed on a half-read payload.
    control: dict | None = None
    control_error: str | None = None
    # Raster MIME payloads emitted by IPython display_data/execute_result messages.
    # They remain structured content for the provider; they are never rendered as
    # base64 in the textual transcript.
    images: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None and not self.timed_out

    def render(self, cap: int = HARD_OUTPUT_CAP) -> str:
        parts: list[str] = []
        if self.stdout.strip():
            parts.append(self.stdout.rstrip())
        if self.stderr.strip():
            parts.append("[stderr]\n" + self.stderr.rstrip())
        if self.result:
            parts.append(self.result.rstrip())
        if self.images:
            noun = "image" if len(self.images) == 1 else "images"
            parts.append(f"[{len(self.images)} {noun} displayed; attached for visual inspection]")
        if self.error:
            parts.append(self.error.rstrip())
        if self.timed_out:
            limit = f" after {self.timeout:g}s" if self.timeout else ""
            if self.stopped:
                parts.append(
                    f"[timed out{limit} — kernel interrupted; the namespace survived. "
                    f"Pass a larger `timeout` to this tool if the work genuinely needs "
                    f"longer, or run it in a background thread and poll it.]"
                )
            else:
                # Saying "interrupted" here would be a lie the model then acts on:
                # it reads the cell as over, writes the next one, and that cell
                # queues behind the one still running and appears to hang too.
                parts.append(
                    f"[timed out{limit} — and the interrupt did not stop it. The code "
                    f"is still running in the kernel, so the next cell will wait for "
                    f"it to finish. Do not re-run this work; let it land, or expect "
                    f"the wait.]"
                )
        text = "\n".join(parts).strip() or "[no output]"
        if len(text) > cap:  # harness-side backstop
            keep = cap // 2
            text = (
                text[:keep]
                + f"\n\n… [harness cap: {len(text) - cap:,} chars dropped] …\n\n"
                + text[-keep:]
            )
        return text


class Kernel:
    """Synchronous facade over one out-of-process IPython kernel."""

    def __init__(self, cwd: str | Path | None = None, env: dict | None = None,
                 *, prewarm: bool = False):
        """Start a kernel. `prewarm` stops short of the xagent bootstrap.

        A prewarmed kernel is a live interpreter with no role yet: the per-run
        environment it needs -- depth, provider, and above all the seed file that
        `runtime.install()` reads at bootstrap -- is not known when it starts. It
        is finished later by `adopt()`, which is what the pool exists to make
        cheap. Everything up to that point is the slow half: process spawn,
        channel setup, and waiting for the kernel to answer.
        """
        self.cwd = str(cwd or Path.cwd())
        # jupyter_client's local provisioner indexes env directly, so it must be a
        # real dict rather than None.
        env = dict(env) if env else dict(os.environ)
        # The kernel runs model-written code and arbitrary shell; it has no need
        # for the operator's credentials. Filtered here rather than only in Runner
        # so that no caller can construct an unfiltered kernel.
        env = {k: v for k, v in env.items() if not is_secret_key(k)}
        self.adopted = not prewarm
        self.km = KernelManager()
        # Pin the interpreter to *this* venv rather than resolving the ambient
        # "python3" kernelspec, which may point at an unrelated interpreter that
        # cannot import xagent.
        self.km._kernel_spec = KernelSpec(
            argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}",
                  # Silences the unencrypted-TCP notice; these kernels are local.
                  "--log-level=ERROR"],
            display_name="xagent",
            language="python",
        )
        self.km.start_kernel(cwd=self.cwd, env=env)
        # From here the process exists, so every failure path must reap it --
        # the caller has no handle to shut down until __init__ returns.
        try:
            self.kc = self.km.client()
            self.kc.start_channels()
            self.kc.wait_for_ready(timeout=90)
            self._orphan_guard()
            if not prewarm:
                self._bootstrap()
        except BaseException:
            self.shutdown()
            raise

    def _bootstrap(self) -> None:
        boot = self.execute(BOOTSTRAP, timeout=60, store_history=False)
        if not boot.ok:
            raise RuntimeError(f"kernel bootstrap failed:\n{boot.render()}")

    def _orphan_guard(self) -> None:
        """Ask the OS to kill this kernel if the process that started it dies.

        A pool lives inside whichever process spawns subagents, and for a subagent
        that is itself a kernel -- one its own parent may SIGKILL, which runs no
        atexit and leaves warm kernels with no one to reap them. Linux-only and
        best-effort: the other two defences (graceful shutdown, and the pool's own
        atexit) cover the ordinary paths.
        """
        self.execute(
            "try:\n"
            "    import ctypes as _c, signal as _s\n"
            "    _c.CDLL(None).prctl(1, _s.SIGKILL)\n"
            "except Exception:\n"
            "    pass\n"
            "del _c, _s\n",
            timeout=20, store_history=False,
        )

    def adopt(self, cwd: str | Path, env: dict) -> None:
        """Give a prewarmed kernel its role, then bootstrap it.

        Ordering is the whole point: the environment has to land before BOOTSTRAP,
        because `runtime.install()` reads XAGENT_SEED at import time and
        `spawn._depth()` reads XAGENT_DEPTH. Stale XAGENT_* from the process that
        started the pool is cleared first -- inheriting a parent's seed path would
        hand a subagent data it was never given.
        """
        if self.adopted:
            raise RuntimeError("kernel has already been adopted")
        wanted = {k: v for k, v in env.items()
                  if k.startswith("XAGENT_") and not is_secret_key(k)}
        setup = self.execute(
            "import os\n"
            "for _k in [k for k in os.environ if k.startswith('XAGENT_')]:\n"
            "    del os.environ[_k]\n"
            f"os.environ.update({wanted!r})\n"
            f"os.chdir({str(cwd)!r})\n"
            "del _k\n",
            timeout=30, store_history=False,
        )
        if not setup.ok:
            raise RuntimeError(f"kernel adoption failed:\n{setup.render()}")
        self._bootstrap()
        self.cwd = str(cwd)
        self.adopted = True

    # ---------------------------------------------------------------- execute

    def push(self, code: str) -> str:
        """Send a cell and do not wait for it.

        `silent=True` is what makes this safe to fire and forget: IPython skips
        both cell hooks for a silent execute, so this neither resets the output
        budget nor triggers a control payload of its own. The kernel runs shell
        requests in arrival order, so a push issued immediately before a cell is
        guaranteed to have finished before that cell begins. Its reply is
        deliberately never read; execute() filters iopub by parent msg_id, so the
        push's own traffic is dropped rather than mixed into the cell's output.
        """
        return self.kc.execute(code, silent=True, store_history=False)

    def execute(self, code: str, timeout: float = CELL_TIMEOUT, store_history: bool = True,
                control_nonce: str | None = None) -> CellOutput:
        msg_id = self.kc.execute(code, store_history=store_history)
        out = CellOutput(timeout=timeout)
        stdout: list[str] = []
        stderr: list[str] = []
        results: list[str] = []
        splitter = _ControlSplitter(control_nonce) if control_nonce else None
        deadline = time.monotonic() + timeout
        interrupted = False

        while True:
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                if not interrupted:
                    self.km.interrupt_kernel()
                    out.timed_out = True
                    interrupted = True
                    deadline = now + DRAIN_GRACE
                    continue
                # The drain window is over and the kernel never went idle. Checked
                # here rather than only where the queue runs dry, because a cell
                # that keeps printing never lets it run dry -- and this loop then
                # outlives by minutes the moment the harness decided the cell had
                # ended.
                out.stopped = False
                break
            try:
                msg = self.kc.get_iopub_msg(timeout=max(0.05, min(remaining, 0.5)))
            except queue.Empty:
                if not self.is_alive():
                    out.error = "[kernel died]"
                    break
                continue

            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            mtype, content = msg["msg_type"], msg["content"]

            if mtype == "stream":
                is_stdout = content.get("name") == "stdout"
                sink = stdout if is_stdout else stderr
                text = content.get("text", "")
                if is_stdout and splitter is not None:
                    # Divert the control payload before the budget below can
                    # truncate it, and before anything rewrites its markers.
                    text = splitter.feed(text)
                # stderr is not wrapped in-kernel, so bound it here or a chatty
                # library can hand the harness hundreds of MB to buffer.
                budget = STREAM_BUDGET - sum(len(x) for x in sink)
                if budget > 0 and text:
                    sink.append(text[:budget])
            elif mtype in ("execute_result", "display_data"):
                data = content.get("data", {})
                text = data.get("text/plain")
                if text:
                    # Accumulate: a single slot meant earlier display() calls were
                    # silently overwritten by the last one.
                    results.append(strip_ansi(text))
                for media_type in SUPPORTED_MEDIA_TYPES:
                    encoded = data.get(media_type)
                    if isinstance(encoded, str) and encoded:
                        out.images.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": encoded,
                            },
                        })
            elif mtype == "error":
                out.error = strip_ansi("\n".join(content.get("traceback", []))).strip()
            elif mtype == "status" and content.get("execution_state") == "idle":
                break

        if splitter is not None:
            if tail := splitter.finish():
                stdout.append(tail)
            out.control, out.control_error = splitter.result()
        out.stdout = strip_ansi("".join(stdout))
        out.stderr = strip_ansi("".join(stderr))
        if results:
            out.result = "\n".join(results)
        return out

    def probe(self, code: str, timeout: float = 60) -> str:
        """Run harness-internal code that stays out of the model's transcript."""
        out = self.execute(code, timeout=timeout, store_history=False)
        if out.error:
            raise RuntimeError(f"probe failed: {out.error}")
        if out.timed_out:
            # A truncated control payload is worse than none: half a JSON document
            # or half a pickle parses as failure somewhere far from here.
            raise RuntimeError("probe timed out; payload would be truncated")
        return out.stdout.strip()

    def probe_pickle(self, expr: str, timeout: float = 120):
        """Pull a live object out of the kernel by value."""
        blob = self.probe(
            "import base64 as _b64, cloudpickle as _cp\n"
            "from xagent.brief import raw_write as _rw\n"
            f"_rw(_b64.b64encode(_cp.dumps({expr})).decode())",
            timeout=timeout,
        )
        return safe_loads(base64.b64decode(blob))

    # ------------------------------------------------------------- lifecycle

    def is_alive(self) -> bool:
        try:
            return self.km.is_alive()
        except Exception:
            return False

    def interrupt(self) -> None:
        try:
            self.km.interrupt_kernel()
        except Exception:
            pass

    def shutdown(self) -> None:
        """Stop the kernel, asking politely before insisting.

        `now=True` skips the shutdown request and kills the process outright, so
        nothing registered with atexit in there ever runs -- and a kernel that
        spawned subagents has a warm pool of its own to reap. Try the graceful
        path first and keep the hard kill as the fallback it should have been.
        """
        try:
            self.kc.stop_channels()
        except Exception:
            pass
        try:
            self.km.shutdown_kernel(now=False)
        except Exception:
            pass
        if self.is_alive():
            try:
                self.km.shutdown_kernel(now=True)
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
