"""The agent loop: render context -> sample code -> execute -> append cell."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from xagent import config
from xagent.compress import Compressor
from xagent.context import ContextStore
from xagent.kernel import Kernel, is_secret_key
from xagent.prompts import SUBAGENT_CODA, SYSTEM
from xagent.provider import Provider

MAX_EMPTY_CALLS = 3

DONE_EXPR = "__import__('xagent.runtime', fromlist=['_STATE'])._STATE['done'][0]"
SIGNALS = "import xagent.runtime as _r; _r._emit_signals()"


@dataclass
class RunResult:
    value: object = None
    brief: str = ""
    degraded: bool = False   # value is a rendering, not the object done() was given
    turns: int = 0
    error: str | None = None
    usage: object = None
    compactions: list = field(default_factory=list)
    seconds: float = 0.0
    finish: str = ""  # done | implicit | max_turns | error

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
        max_turns: int = 40,
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
        self.seed = seed
        self.cwd = str(cwd or Path.cwd())
        self.on_event = on_event or (lambda *_a, **_k: None)
        if budget is not None and budget <= 0:
            raise ValueError(f"budget must be positive, got {budget}")
        self.budget = budget or self.provider.backend.context_budget

        system = SYSTEM + (SUBAGENT_CODA if is_subagent else "")
        self.store = ContextStore(
            task=task,
            system=system,
            replay_thinking=self.provider.backend.replay_thinking and bool(self.provider.thinking),
        )
        self.compressor = Compressor(self.provider, budget=self.budget)
        self.kernel: Kernel | None = None
        self._seed_file: str | None = None
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
        result = RunResult()
        try:
            self.kernel = Kernel(cwd=self.cwd, env=self._kernel_env())
            self._emit("kernel_ready", depth=self.depth)
            result = self._loop()
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            result.finish = "error"
        finally:
            result.seconds = time.monotonic() - started
            result.usage = self.provider.usage
            result.compactions = self.compressor.events
            self._cleanup()
        return result

    def _cleanup(self) -> None:
        if self.kernel:
            self.kernel.shutdown()
        if self._seed_file:
            try:
                os.unlink(self._seed_file)
            except OSError:
                pass

    def _loop(self) -> RunResult:
        result = RunResult()
        assert self.kernel is not None
        empty = 0
        warned_stalled = False

        for turn_no in range(1, self.max_turns + 1):
            self._push_accounting()
            self._refresh_live_vars()

            # Announced before sampling, not after: on a long turn this header is
            # the only thing on screen until the first token arrives.
            self._emit("turn_start", n=turn_no, tokens=self.store.real_tokens)
            turn = self.provider.sample(self.store.system, self.store.messages())
            self.store.observe(turn.input_tokens + turn.cache_read + turn.cache_write)
            self.store.cache_read += turn.cache_read
            self.store.cache_write += turn.cache_write
            self._emit("turn", n=turn_no, text=turn.text, thinking=turn.thinking,
                       code=turn.code, tokens=self.store.real_tokens)

            if turn.tool_use_id is None:
                # No tool call at all: the model stopped acting. Accept its prose as
                # the answer rather than nudging in a loop it cannot escape.
                result.value = turn.text.strip()
                result.brief = result.value[:400]
                result.finish = "implicit"
                if not result.value:
                    result.error = f"model returned neither code nor text (stop={turn.stop_reason})"
                    result.finish = "error"
                return result

            if turn.tool_name and turn.tool_name != "python":
                # Reached only when the turn had no `python` call at all; a stray
                # tool alongside a real one is handled after execution instead.
                self.store.add(
                    code="", output=f"[unknown tool {turn.tool_name!r}. The only tool is "
                                    f"`python`; pass your code as its `code` argument.]",
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
                if empty > MAX_EMPTY_CALLS:
                    result.error = (f"{empty} consecutive empty tool calls from "
                                    f"{self.provider.model}; giving up")
                    result.finish = "error"
                    return result
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
            output = self.kernel.execute(turn.code)
            rendered = output.render()
            if turn.ignored_tools:
                # The python call still ran; say what was dropped so the model
                # stops reaching for a tool it does not have, without costing it
                # the cell it just wrote.
                names = ", ".join(sorted(set(turn.ignored_tools)))
                rendered += (f"\n\n[also called {names} in this turn — ignored. "
                             f"The only tool is `python`.]")
                self._emit("warning", message=f"ignored extra tool call(s): {names}")
            cell = self.store.add(
                code=turn.code,
                output=rendered,
                tool_use_id=turn.tool_use_id or f"tu_{turn_no}",
                thought=turn.text,
                thinking_blocks=turn.thinking_blocks,
            )
            result.turns = len(self.store.cells)
            self._emit("cell", n=cell.n, output=cell.output, ok=output.ok)

            signals = self._drain_signals()
            if signals.get("notes"):
                self.store.notes.update(signals["notes"])

            if request := signals.get("compress"):
                events = self.compressor.apply(
                    self.store, self.kernel,
                    mode=request.get("mode", "evict"), before=request.get("before"),
                )
                for event in events:
                    self._emit("compaction", detail=str(event), requested=True)

            if signals.get("done"):
                result.value = self._pull_done(signals, result)
                result.brief = signals.get("done_brief") or ""
                result.finish = "done"
                return result

            if self.compressor.should_compact(self.store):
                for event in self.compressor.apply(self.store, self.kernel, mode="auto"):
                    self._emit("compaction", detail=str(event), requested=False)
            elif self.compressor.stalled and not warned_stalled:
                warned_stalled = True
                self._emit("warning", message=(
                    f"over the {self.budget:,}-token budget but compaction cannot "
                    f"reclaim more — the floor (system prompt, task, pinned notes, "
                    f"protected tail) is already at the watermark. Raise --budget."))

        result.finish = "max_turns"
        result.error = f"hit the {self.max_turns}-turn limit without calling done()"
        return result

    # ----------------------------------------------------------------- helpers

    def _refresh_live_vars(self) -> None:
        """Re-introspect the namespace into the tail of the context.

        Only once something has been evicted, since before that the cells still
        say what is live. A frozen table inside the compacted-history block would
        describe the namespace as it was at compaction and drift from the kernel
        with every rebinding -- the exact stale-value failure the design set out
        to avoid.
        """
        try:
            # The file ledger is small and always worth carrying: a written file is
            # recoverable by re-reading it, but only if the model still knows it
            # exists. Costs one line per file.
            ledger = self.kernel.probe(
                "import xagent.runtime as _r; _r._emit_file_ledger()", timeout=20)
            # probe() strips the payload, which would eat the first row's indent.
            self.store.live_files = "\n".join(
                "  " + ln.strip() for ln in ledger.splitlines() if ln.strip())
        except Exception:
            pass
        if not self.store.state_report:
            return
        try:
            self.store.live_vars = self.compressor._var_table(self.kernel)
        except Exception:
            self.store.live_vars = ""

    def _push_accounting(self) -> None:
        """Give the model visibility into its own budget, off-transcript."""
        info = self.store.accounting(self.budget)
        if not self.provider.backend.reports_cache:
            info["cache_read"] = None
        try:
            self.kernel.probe(f"import xagent.runtime as _r; _r._set_ctx({info!r})", timeout=20)
        except Exception:
            pass

    def _drain_signals(self) -> dict:
        try:
            return json.loads(self.kernel.probe(SIGNALS, timeout=30) or "{}")
        except Exception:
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
