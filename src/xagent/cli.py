"""Command-line entry point."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import textwrap
from pathlib import Path

from rich.console import Console
from rich.highlighter import ReprHighlighter
from rich.markdown import Markdown
from rich.pretty import Pretty
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme

from xagent import config
from xagent.audio import AudioAttachment, AudioError, normalize_audio
from xagent.kernel import strip_ansi
from xagent.provider import Provider
from xagent.runner import MAX_TURN_BLOCKS, MAX_TURNS, Runner
from xagent.vision import normalize_images

C = {
    "dim": "\033[2m", "bold": "\033[1m", "reset": "\033[0m",
    "blue": "\033[34m", "green": "\033[32m", "yellow": "\033[33m",
    "red": "\033[31m", "cyan": "\033[36m", "mag": "\033[35m",
}


def paint(enabled: bool):
    if enabled:
        return lambda text, color: f"{C[color]}{text}{C['reset']}"
    return lambda text, color: text


# Prefix and colour for each kind of streamed output.
SECTIONS = {
    "thinking": ("  ┊ ", "dim"),
    "text": ("  ", None),
    "code": ("  │ ", "cyan"),
}
# Tools whose "code" is not python, so it is not lexed as python.
SHELL_TOOLS = {"sh", "bash", "shell"}

# rich does the rendering; this file keeps the gutters. Every renderable is
# captured and then written out line by line behind the same prefix the raw
# stream uses, so markdown and highlighting arrive *inside* the transcript
# rather than as a widget dropped on top of it.
CODE_THEME = "ansi_dark"   # the terminal's own 16 colours, so the user's theme wins
# rich sets inline code on a black background, which is only legible if the
# terminal happens to be dark too. Everything here paints foregrounds and leaves
# the background to whoever chose it.
MD_STYLES = Theme({"markdown.code": "bold cyan", "markdown.item.bullet": "cyan",
                   "markdown.item.number": "cyan"}, inherit=True)
FALLBACK_WIDTH = 100       # when stdout is a file and so has no width of its own

LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s")
FENCES = ("```", "~~~")


class Renderer:
    """Writes a rich renderable into one of the transcript's gutters."""

    def __init__(self, colour: bool):
        self.colour = colour
        self.c = paint(colour)
        self.console = Console(
            # Pinned rather than sniffed: the console renders into a capture
            # buffer, so left to itself it would decide there is no terminal here
            # and drop the colour on the way to one that has it.
            color_system="truecolor" if colour else None,
            force_terminal=colour or None,
            highlight=False, markup=False, emoji=False, theme=MD_STYLES,
        )
        self.highlighter = ReprHighlighter()

    def block(self, renderable, prefix: str, tone: str | None = None, *,
              style: str | None = None, wrap: bool = True) -> None:
        # The gutter is printed rather than rendered, so the width rich wraps to
        # is the terminal's minus the room the gutter takes out of it.
        columns = shutil.get_terminal_size((FALLBACK_WIDTH, 24)).columns
        self.console.width = max(20, columns - len(prefix) - 1)
        with self.console.capture() as capture:
            # soft_wrap also turns cropping off: output that has to survive intact
            # (a cell's, a traceback's) is written at its own width and left to the
            # terminal, rather than cut to the console's.
            self.console.print(renderable, style=style, soft_wrap=not wrap)
        lines = capture.get().split("\n")
        # rich separates its elements with a blank line above and below. Rendering
        # a block at a time turns those into a gap at every seam, on top of the
        # one this printer puts between blocks itself.
        while lines and not strip_ansi(lines[0]).strip():
            lines.pop(0)
        while lines and not strip_ansi(lines[-1]).strip():
            lines.pop()
        if not lines:
            return
        gutter = self.c(prefix, tone) if tone else prefix
        for line in lines:
            sys.stdout.write(gutter + line.rstrip() + "\n")
        sys.stdout.flush()


def split_block(buf: str) -> tuple[str | None, str]:
    """The first *finished* markdown block in `buf`, and what is left after it.

    Finished means a blank line outside a fenced code block. Until one arrives a
    heading is still an ordinary paragraph and a table is still one stray row, so
    rendering early gets the block wrong rather than getting it early.
    """
    lines = buf.split("\n")
    fence = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith(FENCES):
            fence = not fence
            continue
        # The last element is whatever follows the final newline -- a line still
        # being written, not a blank one -- so it never closes a block.
        if fence or line.strip() or not i or i == len(lines) - 1:
            continue
        return "\n".join(lines[:i]), "\n".join(lines[i + 1:])
    return None, buf


class Prose:
    """Buffers streamed prose and renders it a finished markdown block at a time.

    Markdown cannot be rendered a token at a time, and holding the whole turn back
    to render it at the end would undo the streaming. A block is the unit that is
    both complete and soon.
    """

    def __init__(self, renderer: Renderer, prefix: str, tone: str | None = None,
                 style: str | None = None):
        self.renderer, self.prefix, self.tone = renderer, prefix, tone
        self.style = style
        self.buf, self.pending, self.wrote = "", [], False

    def feed(self, text: str) -> None:
        self.buf += text
        while True:
            block, rest = split_block(self.buf)
            if block is None:
                return
            self.buf = rest
            self.take(block)

    def close(self) -> None:
        if self.buf.strip():
            self.take(self.buf)
        self.buf = ""
        self.flush_pending()
        self.wrote = False

    def take(self, block: str) -> None:
        if not block.strip():
            return
        # A list is held whole: rendered a block at a time, every item of a loose
        # ordered list is a list of its own, and they all start again at 1.
        if LIST_ITEM.match(block) or (self.pending and block[:1] in (" ", "\t")):
            self.pending.append(block)
            return
        self.flush_pending()
        self.emit(block)

    def flush_pending(self) -> None:
        if self.pending:
            self.emit("\n\n".join(self.pending))
            self.pending = []

    def emit(self, text: str) -> None:
        if self.wrote:
            sys.stdout.write("\n")   # the gap a whole-document render would leave
        self.renderer.block(Markdown(text.strip("\n"), code_theme=CODE_THEME),
                            self.prefix, self.tone, style=self.style)
        self.wrote = True


class Code:
    """Buffers streamed code and highlights it a finished line at a time.

    The last line in the buffer may still be half-written, so it is held back.
    Everything above it is re-lexed on every delta and only the new lines are
    printed -- so a line is never highlighted against code the model has not
    written yet, and never rewritten once it is on screen.
    """

    def __init__(self, renderer: Renderer, prefix: str, tone: str | None = None):
        self.renderer, self.prefix, self.tone = renderer, prefix, tone
        self.lexer = "python"
        self.buf, self.printed = "", 0

    def feed(self, text: str) -> None:
        self.buf += text
        self.drain(final=False)

    def close(self) -> None:
        self.drain(final=True)
        self.buf, self.printed = "", 0

    def drain(self, final: bool) -> None:
        lines = self.buf.split("\n")
        ready = len(lines) if final else len(lines) - 1
        while final and ready > self.printed and not lines[ready - 1].strip():
            ready -= 1        # a cell's trailing newlines are not lines of code
        if ready <= self.printed:
            return
        self.renderer.block(
            Syntax("\n".join(lines[:ready]), self.lexer, theme=CODE_THEME,
                   background_color="default", word_wrap=True,
                   line_range=(self.printed + 1, ready)),
            self.prefix, self.tone)
        self.printed = ready


# Reasoning is counted in characters as it streams -- the token count is only
# known once the turn ends, and a live counter is worth more than an exact one.
# Same 4-chars-per-token heuristic Provider._clamp budgets with, so the number
# stays comparable with the context figures printed elsewhere.
CHARS_PER_TOKEN = 4
# How often a non-tty (a redirected log, watched with `tail -f`) gets a fresh
# progress line, since it cannot have one rewritten in place.
LOG_TICK_TOKENS = 1_000


def make_printer(colour, verbose: bool, live_updates: bool = True):
    c = paint(colour)
    renderer = Renderer(colour)
    # One buffer per kind of streamed output. Each holds the stream until a piece
    # of it is finished -- a markdown block, a line of code -- and only then hands
    # that piece to rich, because neither can be rendered a token at a time.
    buffers = {
        "text": Prose(renderer, *SECTIONS["text"]),
        "thinking": Prose(renderer, *SECTIONS["thinking"], style="dim"),
        "code": Code(renderer, *SECTIONS["code"]),
    }
    # `section` is the kind currently being streamed, `streamed` whether this turn
    # was shown live (in which case the turn summary must not repeat it).
    # `think_chars` accumulates the reasoning collapsed into a single line.
    live = {"section": None, "streamed": False, "tool": None,
            "think_chars": 0, "logged_at": 0}

    def show(text: str, kind: str, limit: int) -> None:
        """Render a whole block of prose that never streamed -- a turn summary."""
        body = text.strip()
        if len(body) > limit:
            body = body[:limit] + " …"
        prose = Prose(renderer, *SECTIONS[kind],
                      style="dim" if kind == "thinking" else None)
        prose.feed(body)
        prose.close()

    def dump(text: str, prefix: str, tone: str, limit: int, highlight: bool) -> None:
        """Render a cell's output: highlighted, but never reflowed or cut short.

        The output is what the model read. Wrapping it here would put line breaks
        in the transcript that were not in the cell, so it goes out at its own
        width and the terminal deals with it.
        """
        body = text.strip()
        if len(body) > limit:
            body = body[:limit] + " …"
        shown = Text(body)
        renderer.block(renderer.highlighter(shown) if highlight else shown,
                       prefix, tone, wrap=False)

    def think_tokens() -> int:
        return live["think_chars"] // CHARS_PER_TOKEN

    def collapse_thinking(text: str) -> None:
        """Non-verbose: one self-updating line instead of the whole reasoning.

        Hiding reasoning entirely is what made a long turn look like a hang, and
        showing all of it buries the actions. A running count does neither.
        """
        if live["section"] != "thinking":
            close_section()
            live["section"] = "thinking"
            live["logged_at"] = 0
        live["streamed"] = True
        live["think_chars"] += len(text)

        if live_updates:
            # \r rewrites the line; \033[K clears whatever the old one left.
            sys.stdout.write("\r" + c(f"  ┊ thinking… ~{think_tokens():,} tok", "dim")
                             + "\033[K")
            sys.stdout.flush()
        elif think_tokens() - live["logged_at"] >= LOG_TICK_TOKENS:
            live["logged_at"] = think_tokens()
            sys.stdout.write(c(f"  ┊ thinking… ~{think_tokens():,} tok", "dim") + "\n")
            sys.stdout.flush()

    def indent(text: str, prefix: str, limit: int) -> str:
        body = text.strip()
        if len(body) > limit:
            body = body[:limit] + " …"
        return textwrap.indent(body, prefix)

    def close_section() -> None:
        section = live["section"]
        if section == "thinking" and not verbose:
            # Settle the collapsed counter into a final, permanent line.
            done = c(f"  ┊ thought ~{think_tokens():,} tok", "dim")
            sys.stdout.write(("\r" + done + "\033[K\n") if live_updates else (done + "\n"))
            sys.stdout.flush()
            live["section"] = None
            return
        if section is not None:
            # Whatever the buffer was still holding for want of a finished block
            # is finished now: the section it belongs to is over.
            buffers[section].close()
        live["section"] = None

    def stream(kind: str, text: str) -> None:
        """Feed a delta to its section's buffer, opening the section if needed."""
        if not text:
            return
        if kind == "thinking" and not verbose:
            collapse_thinking(text)
            return
        if live["section"] != kind and kind == "text":
            # qwen pads the gap between its reasoning and its tool call with
            # newlines. Opening a section on those prints a blank heading, and
            # keeping them would open the prose on an empty line.
            text = text.lstrip("\n")
            if not text.strip():
                return
        if live["section"] != kind:
            close_section()
            live["section"] = kind
            if kind == "code":
                # Name the tool the model actually called. Printing a fixed
                # "python" here once hid a stray `sh` call behind a correct-
                # looking header -- and would now highlight it as python too.
                tool = live["tool"] or "python"
                print(c(f"  ▸ {tool}", "cyan"))
                buffers["code"].lexer = "bash" if tool in SHELL_TOOLS else "python"
        live["streamed"] = True
        buffers[kind].feed(text)

    def on_event(kind: str, data: dict) -> None:
        if kind == "kernel_ready":
            print(c(f"● kernel ready (depth {data['depth']})", "dim"))
        elif kind == "turn_start":
            live["streamed"] = False
            live["tool"] = None
            live["think_chars"] = 0
            head = c(f"\n─── turn {data['n']}", "dim")
            tokens = c(f"~{data['tokens']:,} tok", "dim")
            print(f"{head} {tokens}")
        elif kind == "delta":
            if data["part"] == "retry":
                close_section()
                print(c(f"  ↻ {data['text']}", "yellow"))
            elif data["part"] == "tool":
                # A new tool block starts its own code section under its own name.
                close_section()
                live["tool"] = data["text"]
            else:
                stream(data["part"], data["text"])
        elif kind == "turn":
            close_section()
            if live["streamed"]:
                return  # already shown as it was generated
            if data.get("thinking"):
                if verbose:
                    show(data["thinking"], "thinking", 900)
                else:
                    approx = len(data["thinking"]) // CHARS_PER_TOKEN
                    print(c(f"  ┊ thought ~{approx:,} tok", "dim"))
            if data.get("text"):
                show(data["text"], "text", 1200)
            # One header per call: a turn that batched three of them ran three
            # cells, and printing only the first would credit the outputs below to
            # code that is not on screen.
            for code in (data.get("codes") or ([data["code"]] if data.get("code") else [])):
                if not (code or "").strip():
                    continue
                print(c("  ▸ python", "cyan"))
                block = Code(renderer, *SECTIONS["code"])
                block.feed(code.strip())
                block.close()
        elif kind == "prompt":
            close_section()
            # A message that reached the run at a turn boundary: the person's own
            # words, or a parent's to its subagent. Printed because the transcript
            # has to show what the model was told and when -- the answer below it is
            # a reply to this, not to the task at the top.
            who = "you" if data.get("tag") == "user-message" else "parent"
            print(c(f"\n▸ {who}", "green"))
            print(indent(data["text"], c("  │ ", "green"), 2000))
        elif kind == "cell":
            close_section()
            colour_name = "dim" if data["ok"] else "red"
            print(c("  ← output", colour_name))
            # A traceback is already coloured by being an error; running the repr
            # highlighter over it would only pick out the numbers inside it.
            dump(data["output"], "  │ ", colour_name, 2000, highlight=data["ok"])
        elif kind == "compaction":
            close_section()
            tag = "requested" if data.get("requested") else "forced"
            print(c(f"\n✂ compaction ({tag}): {data['detail']}", "mag"))
        elif kind == "note":
            close_section()
            print(c(f"  · {data['message']}", "dim"))
        elif kind == "warning":
            close_section()
            print(c(f"\n! {data['message']}", "yellow"))

    return on_event


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="xagent",
        description="An LLM harness whose context window is a Python REPL.",
    )
    parser.add_argument("task", nargs="?",
                        help="what the agent should do. Omit it to open a session "
                             "and type instead")
    parser.add_argument("-f", "--task-file", help="read the task from a file")
    parser.add_argument("-I", "--interactive", action="store_true",
                        help="start a session even with a task on the command line, "
                             "so you can reply to the answer and keep the kernel")
    parser.add_argument("-i", "--image", action="append", dest="images", metavar="PATH",
                        help="attach a local image (repeat for multiple images)")
    parser.add_argument("-a", "--audio", action="append", dest="audio", metavar="PATH",
                        help="attach an audio file as an analysis panel — waveform, "
                             "spectrogram, level and level histogram (repeatable)")
    parser.add_argument("--listen", type=float, default=None, metavar="SECONDS",
                        help="record live audio for this long and attach it the same way")
    parser.add_argument("--source", default="speaker", metavar="WHAT",
                        help="what --listen records: speaker (whatever this machine "
                             "is playing), mic, or a device name (default: speaker)")
    parser.add_argument("-p", "--provider", default=None,
                        choices=sorted(config.BACKENDS), help="which backend to use")
    parser.add_argument("-m", "--model", default=None, help="override the driver model")
    parser.add_argument("-t", "--thinking", default=None, choices=config.THINKING_CHOICES,
                        help=f"thinking depth (default {config.DEFAULT_THINKING}; "
                             f"`off` disables). On current Claude models this is the "
                             f"effort level the model paces its own thinking against; "
                             f"on qwen it maps to a token budget")
    parser.add_argument("-s", "--sampling", default=None, choices=sorted(config.SAMPLING),
                        help="sampling preset (qwen only; the Anthropic API rejects these)")
    parser.add_argument("-C", "--cwd", default=None, help="working directory for the kernel")
    parser.add_argument("-n", "--max-turns", type=int, default=MAX_TURNS,
                        help="turns per block; reaching it compacts and grants "
                             f"another, up to {MAX_TURN_BLOCKS} blocks")
    parser.add_argument("-b", "--budget", type=int, default=None,
                        help="context budget in tokens before compaction is forced")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="stream the reasoning in full (default: one live token count)")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args(argv)
    if args.budget is not None and args.budget <= 0:
        parser.error("--budget must be positive")
    if args.listen is not None and args.listen <= 0:
        parser.error("--listen must be a positive number of seconds")

    if args.task_file:
        task = Path(args.task_file).read_text()
    elif args.task:
        task = args.task
    else:
        task = None

    # A session needs somewhere to read from, so a redirected stdin is always the
    # one-prompt path however it was asked for. That also keeps `xagent "…" | tee`
    # and every script that shells out to it behaving exactly as before.
    interactive = (args.interactive or task is None) and sys.stdin.isatty()
    if task is None and not interactive:
        parser.error("provide a task, or -f/--task-file (stdin is not a terminal, "
                     "so there is nothing to open a session on)")

    try:
        images = normalize_images(args.images)
    except (OSError, ValueError) as e:
        parser.error(str(e))

    try:
        audio = normalize_audio(args.audio)
    except (OSError, ValueError, AudioError) as e:
        parser.error(str(e))

    colour = not args.no_color and sys.stdout.isatty()
    c = paint(colour)
    provider = Provider(backend=args.provider, model=args.model,
                        thinking=args.thinking, sampling=args.sampling)

    tags = f" · thinking={provider.thinking or config.THINKING_OFF}"
    if provider.backend.sampling:
        tags += f" · sampling={provider.sampling}"
    print(c(f"xagent · {provider.backend.name}/{provider.model}{tags}", "bold"))
    print(c(f"max output up to {provider.max_tokens:,} tok"
            f"{f', clamped per request to a {provider.backend.total_window:,} window' if provider.backend.total_window else ''}",
            "dim"))
    print(c(f"cwd {args.cwd or Path.cwd()}", "dim"))

    if args.listen:
        from xagent import capture

        try:
            source = capture.resolve(args.source)
            print(c(f"● recording {source.kind} ({source.device}) for "
                    f"{args.listen:g}s via {source.tool}…", "dim"))
            audio.append(AudioAttachment.from_clip(capture.record(source, args.listen)))
        except AudioError as e:
            print(c(f"! {e}", "yellow"))
    for clip in audio:
        print(c(f"♪ {clip.summary()}", "dim"))

    printer = make_printer(colour, args.verbose, live_updates=sys.stdout.isatty())

    if interactive:
        from xagent.session import Session, UserChannel

        channel = UserChannel()
        runner = Runner(
            # Empty when the person has not typed anything yet: their first message
            # then opens the conversation the way the argument would have, rather than
            # being folded into a `<task>` block that was never written.
            task or "",
            provider=provider, images=images, audio=audio, cwd=args.cwd,
            max_turns=args.max_turns, budget=args.budget,
            on_event=printer, channel=channel,
        )
        return Session(runner, colour=colour, paint=c).run(first=task)

    runner = Runner(
        task,
        provider=provider,
        images=images,
        audio=audio,
        cwd=args.cwd,
        max_turns=args.max_turns,
        budget=args.budget,
        on_event=printer,
    )
    result = runner.run()

    print(c("\n" + "═" * 60, "dim"))
    if result.error:
        print(c(f"✗ {result.finish}: {result.error}", "red"))
    else:
        print(c(f"✓ {result.finish}", "green"))
    if result.degraded:
        print(c("  (value could not be serialized — shown as a rendering)", "yellow"))
    # The answer has already been on screen: it streamed as it was written. Reprinting
    # it here would show the deliverable twice. What is left to print is a value from
    # a run that produced no prose -- a subagent-shaped finish at the top level.
    if not result.answer and result.value is not None:
        print()
        renderer = Renderer(colour)
        if isinstance(result.value, str):
            renderer.block(Markdown(result.value.strip()[:4000], code_theme=CODE_THEME),
                           "  ")
        else:
            renderer.block(Pretty(result.value, max_length=200, max_string=2000), "  ")

    usage = result.usage
    print(c(f"\n{result.turns} turns · {result.seconds:.0f}s · "
            f"in {usage.input:,} out {usage.output:,} "
            f"cache_read {usage.cache_read:,}"
            f"{f' (hit {usage.cache_hit_rate:.0%})' if provider.backend.reports_cache else ''}",
            "dim"))
    if result.compactions:
        for event in result.compactions:
            print(c(f"  ✂ {event}", "dim"))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
