"""The interactive front end: a conversation with the agent at a terminal.

Two threads and one queue. The Runner works on a worker thread; the input box owns
the main thread's event loop and stays live the whole time, which is what lets a
person type while a turn streams. `patch_stdout` is what makes that safe: the
worker's writes are held and replayed above the box rather than tearing through it.

The queue has two destinations depending on when the text arrives. Mid-response it
goes to the channel, and the Runner delivers it at the next turn boundary as a real
user message. Between responses it is the next prompt. Esc is the third gesture:
it interrupts the cell that is running and ends the response, without ending the
session.

`Runner.run()` -- one prompt, no input layer -- does not come through here at all.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout

HISTORY = Path.home() / ".xagent" / "history"

BANNER = """\
  ⏎ send · esc interrupt · ↑ history · /exit to quit
  Type while it works: your message arrives at the next turn boundary."""


class UserChannel:
    """The person's end of a running response.

    Implements the same protocol `spawn._Link` does, which is the whole point: the
    Runner cannot tell whether the message it is delivering came from a person at a
    terminal or from the agent that spawned it, and neither can anything downstream.

    `post()` and `interrupt()` are called from the prompt_toolkit thread while
    `drain()` and `stopping` are read from the worker, so every field moves under
    one lock.
    """

    stop_label = "interrupted by you"
    stop_finish = "interrupted"

    def __init__(self):
        self._lock = threading.Lock()
        self._messages: list[str] = []
        self._stop = threading.Event()
        self.stop_reason = ""
        self.kernel = None

    # ---------------------------------------------------------- the person's end

    def post(self, text: str) -> int:
        with self._lock:
            self._messages.append(text)
            return len(self._messages)

    def interrupt(self, reason: str = "") -> None:
        """End the current response now.

        The kernel is interrupted from here rather than at the turn boundary. A cell
        may have five minutes left on its timeout, and waiting it out is exactly what
        the person pressing Esc asked not to do.
        """
        with self._lock:
            self.stop_reason = reason or "you pressed esc"
            kernel = self.kernel
        # Set before the kernel is touched, so the boundary the interrupted cell
        # falls through to already knows why it is there.
        self._stop.set()
        if kernel is not None:
            try:
                kernel.interrupt()
            except Exception:
                pass

    # ------------------------------------------------------------ the run's end

    def drain(self) -> list[str]:
        with self._lock:
            pending, self._messages = self._messages, []
        return pending

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def resume(self) -> None:
        """Clear the stop and keep the session. An interrupt is not a kill."""
        self._stop.clear()
        with self._lock:
            self.stop_reason = ""

    def attach(self, kernel) -> None:
        with self._lock:
            self.kernel = kernel

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._messages)


class Session:
    """The loop: read a prompt, run one response, read the next prompt."""

    def __init__(self, runner, *, colour: bool = True, paint=None):
        self.runner = runner
        self.channel = runner.channel
        self.colour = colour
        self.c = paint or (lambda text, _tone: text)
        HISTORY.parent.mkdir(parents=True, exist_ok=True)
        self.prompts = PromptSession(
            history=FileHistory(str(HISTORY)),
            key_bindings=self._keys(),
            multiline=False,
        )
        # Set while a response runs, so a submitted line knows whether it is a
        # message for this response or the prompt that starts the next one.
        self.busy = False

    def _keys(self) -> KeyBindings:
        keys = KeyBindings()

        @keys.add("escape", eager=True)
        def _(event) -> None:
            # `eager` so it fires on its own rather than waiting to see whether this
            # is the start of an escape sequence. The cost is losing Esc as a meta
            # prefix, which nothing here binds.
            if not self.busy:
                return
            self.channel.interrupt()
            print(self.c("  · interrupting — the current cell is being stopped", "yellow"))

        return keys

    # -------------------------------------------------------------------- input

    def _bottom(self) -> HTML:
        if not self.busy:
            return HTML("")
        waiting = self.channel.pending
        note = f"{waiting} message(s) queued · " if waiting else ""
        return HTML(f"<i>  {note}⏎ queue a message · esc interrupt</i>")

    async def _read(self) -> str | None:
        """One line from the person. None means they asked to leave."""
        try:
            text = await self.prompts.prompt_async(
                HTML("<b>&gt;</b> "), bottom_toolbar=self._bottom, refresh_interval=0.5)
        except EOFError:
            return None
        except KeyboardInterrupt:
            # Ctrl-C clears the line rather than ending the session, the way a shell
            # does. Ending the session is Ctrl-D or /exit, both of which say so.
            return ""
        return text.strip()

    def _command(self, text: str) -> bool:
        """Handle a slash command. True means it was one, so it is not a prompt."""
        if not text.startswith("/"):
            return False
        if text in ("/exit", "/quit"):
            raise EOFError
        if text == "/ctx":
            info = self.runner.store.accounting(self.runner.budget)
            pct = 100.0 * info["used"] / max(info["budget"], 1)
            print(self.c(f"  ctx {info['used']:,}/{info['budget']:,} ({pct:.0f}%) · "
                         f"{info['live']} live/{info['folded']} folded/"
                         f"{info['evicted']} evicted · "
                         f"{len(self.runner.store.marks)} message(s)", "dim"))
            return True
        print(self.c(f"  · unknown command {text!r} — /exit, /ctx", "yellow"))
        return True

    # --------------------------------------------------------------------- loop

    async def _main(self, first: str | None) -> None:
        pending = first
        while True:
            if pending is None:
                text = await self._read()
                if text is None:
                    return
                if not text or self._command(text):
                    continue
            else:
                text, pending = pending, None

            await self._respond(text)
            # Anything typed after the response ended is the next prompt, not a
            # message into a response that is already over.
            if leftover := self.channel.drain():
                pending = "\n\n".join(leftover)

    async def _listen(self) -> None:
        """Keep the input box live while a response runs, queueing what is typed.

        This is the half of the design that makes the input box more than decoration:
        the person can steer mid-turn without stopping anything. Each line goes to the
        channel, and the Runner delivers it at the next turn boundary as a real user
        message. Cancelled by `_respond` the moment the response ends.
        """
        while True:
            text = await self._read()
            if text is None:
                # Ctrl-D while it works. Read as "stop": ending the session under a
                # response still in flight would tear the kernel down beneath it.
                self.channel.interrupt("you pressed ctrl-d")
                return
            if not text or self._command(text):
                continue
            waiting = self.channel.post(text)
            print(self.c(f"  · queued ({waiting} waiting) — it arrives at the next "
                         f"turn boundary; esc to interrupt instead", "dim"))

    async def _respond(self, text: str) -> None:
        """Run one response on a worker thread, keeping the input box alive."""
        self.busy = True
        self.channel.resume()
        work = asyncio.create_task(asyncio.to_thread(self.runner.ask, text))
        listening = asyncio.create_task(self._listen())
        try:
            result = await work
        finally:
            self.busy = False
            listening.cancel()
            # Awaited rather than left to be reaped: the next prompt reuses this
            # PromptSession, and starting a second one while the first is still
            # unwinding puts two applications on one terminal.
            try:
                await listening
            except asyncio.CancelledError:
                pass
        if result.finish == "interrupted":
            print(self.c("\n■ stopped. The kernel and everything in it are intact.",
                         "yellow"))
        elif result.error:
            print(self.c(f"\n✗ {result.finish}: {result.error}", "red"))

    def run(self, first: str | None = None) -> int:
        print(self.c(BANNER, "dim"))
        try:
            with patch_stdout(raw=True):
                asyncio.run(self._main(first))
        except (EOFError, KeyboardInterrupt):
            pass
        finally:
            print(self.c("\n● closing the kernel", "dim"))
            self.runner.close()
        usage = self.runner.provider.usage
        print(self.c(f"{self.runner.turns} turns · in {usage.input:,} "
                     f"out {usage.output:,} cache_read {usage.cache_read:,}", "dim"))
        return 0


def interactive(runner, *, colour: bool = True, paint=None,
                first: str | None = None) -> int:
    return Session(runner, colour=colour, paint=paint).run(first)
