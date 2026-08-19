"""A persistent IPython kernel, one per agent.

Out-of-process via jupyter_client so a segfault or a runaway loop in model-written
code cannot take the harness down, and so `interrupt` actually works.
"""

from __future__ import annotations

import base64
import io
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

_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

HARD_OUTPUT_CAP = 6000    # absolute backstop, in characters
STREAM_BUDGET = 400_000   # per-stream harness-side memory bound, in characters

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


@dataclass
class CellOutput:
    stdout: str = ""
    stderr: str = ""
    result: str | None = None
    error: str | None = None
    timed_out: bool = False
    signals: list[str] = field(default_factory=list)

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
        if self.error:
            parts.append(self.error.rstrip())
        if self.timed_out:
            parts.append("[timed out — kernel interrupted; the namespace survived]")
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

    def __init__(self, cwd: str | Path | None = None, env: dict | None = None):
        self.cwd = str(cwd or Path.cwd())
        # jupyter_client's local provisioner indexes env directly, so it must be a
        # real dict rather than None.
        env = dict(env) if env else dict(os.environ)
        # The kernel runs model-written code and arbitrary shell; it has no need
        # for the operator's credentials. Filtered here rather than only in Runner
        # so that no caller can construct an unfiltered kernel.
        env = {k: v for k, v in env.items() if not is_secret_key(k)}
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
            boot = self.execute(BOOTSTRAP, timeout=60, store_history=False)
            if not boot.ok:
                raise RuntimeError(f"kernel bootstrap failed:\n{boot.render()}")
        except BaseException:
            self.shutdown()
            raise

    # ---------------------------------------------------------------- execute

    def execute(self, code: str, timeout: float = 180, store_history: bool = True) -> CellOutput:
        msg_id = self.kc.execute(code, store_history=store_history)
        out = CellOutput()
        stdout: list[str] = []
        stderr: list[str] = []
        results: list[str] = []
        deadline = time.monotonic() + timeout
        interrupted = False

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 and not interrupted:
                self.km.interrupt_kernel()
                out.timed_out = True
                interrupted = True
                deadline = time.monotonic() + 5  # drain the KeyboardInterrupt
                continue
            try:
                msg = self.kc.get_iopub_msg(timeout=max(0.05, min(remaining, 0.5)))
            except queue.Empty:
                if interrupted and time.monotonic() > deadline:
                    break
                if not self.is_alive():
                    out.error = "[kernel died]"
                    break
                continue

            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            mtype, content = msg["msg_type"], msg["content"]

            if mtype == "stream":
                sink = stdout if content.get("name") == "stdout" else stderr
                text = content.get("text", "")
                # stderr is not wrapped in-kernel, so bound it here or a chatty
                # library can hand the harness hundreds of MB to buffer.
                budget = STREAM_BUDGET - sum(len(x) for x in sink)
                if budget > 0:
                    sink.append(text[:budget])
            elif mtype in ("execute_result", "display_data"):
                text = content.get("data", {}).get("text/plain")
                if text:
                    # Accumulate: a single slot meant earlier display() calls were
                    # silently overwritten by the last one.
                    results.append(strip_ansi(text))
            elif mtype == "error":
                out.error = strip_ansi("\n".join(content.get("traceback", []))).strip()
            elif mtype == "status" and content.get("execution_state") == "idle":
                break

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
        for step in (
            lambda: self.kc.stop_channels(),
            lambda: self.km.shutdown_kernel(now=True),
        ):
            try:
                step()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
