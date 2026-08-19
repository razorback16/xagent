"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from xagent import config
from xagent.provider import Provider
from xagent.runner import Runner

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
    # `section` is the kind currently being streamed, `bol` whether the cursor sits
    # at the start of a line and so still owes a prefix, `streamed` whether this
    # turn was shown live (in which case the turn summary must not repeat it).
    # `think_chars` accumulates the reasoning collapsed into a single line.
    live = {"section": None, "bol": True, "streamed": False, "tool": None,
            "think_chars": 0, "logged_at": 0}

    def think_tokens() -> int:
        return live["think_chars"] // CHARS_PER_TOKEN

    def collapse_thinking(text: str) -> None:
        """Non-verbose: one self-updating line instead of the whole reasoning.

        Hiding reasoning entirely is what made a long turn look like a hang, and
        showing all of it buries the actions. A running count does neither.
        """
        if live["section"] != "thinking":
            close_section()
            live["section"], live["bol"] = "thinking", False
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
        if live["section"] == "thinking" and not verbose:
            # Settle the collapsed counter into a final, permanent line.
            done = c(f"  ┊ thought ~{think_tokens():,} tok", "dim")
            sys.stdout.write(("\r" + done + "\033[K\n") if live_updates else (done + "\n"))
            sys.stdout.flush()
            live["section"], live["bol"] = None, True
            return
        if live["section"] is not None and not live["bol"]:
            sys.stdout.write("\n")
        live["section"], live["bol"] = None, True

    def stream(kind: str, text: str) -> None:
        """Write a delta under its section header, re-prefixing every new line."""
        if not text:
            return
        if kind == "thinking" and not verbose:
            collapse_thinking(text)
            return
        if live["section"] != kind and kind == "text":
            # qwen pads the gap between its reasoning and its tool call with
            # newlines. Opening a section on those prints a blank heading, and
            # keeping them as a prefix indents the prose off its own first line.
            text = text.lstrip("\n")
            if not text.strip():
                return
        if live["section"] != kind:
            close_section()
            live["section"] = kind
            if kind == "code":
                # Name the tool the model actually called. Printing a fixed
                # "python" here once hid a stray `sh` call behind a correct-
                # looking header.
                print(c(f"  ▸ {live['tool'] or 'python'}", "cyan"))
        live["streamed"] = True
        prefix, tone = SECTIONS[kind]
        pieces = text.split("\n")
        for i, piece in enumerate(pieces):
            if i:
                sys.stdout.write("\n")
                live["bol"] = True
            if not piece:
                # Keep the gutter unbroken across blank lines -- but not on a
                # chunk's trailing newline, which would leave a prefix dangling
                # at the end of output that may be the last thing printed.
                if live["bol"] and i < len(pieces) - 1:
                    sys.stdout.write(c(prefix, tone) if tone else prefix)
                    live["bol"] = False
                continue
            if live["bol"]:
                sys.stdout.write(c(prefix, tone) if tone else prefix)
                live["bol"] = False
            sys.stdout.write(c(piece, tone) if tone else piece)
        sys.stdout.flush()

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
                    print(indent(data["thinking"], c("  ┊ ", "dim"), 900))
                else:
                    approx = len(data["thinking"]) // CHARS_PER_TOKEN
                    print(c(f"  ┊ thought ~{approx:,} tok", "dim"))
            if data.get("text"):
                print(indent(data["text"], "  ", 1200))
            if data.get("code"):
                print(c("  ▸ python", "cyan"))
                print(textwrap.indent(data["code"].strip(), c("  │ ", "cyan")))
        elif kind == "cell":
            close_section()
            colour_name = "dim" if data["ok"] else "red"
            print(c("  ← output", colour_name))
            print(indent(data["output"], c("  │ ", colour_name), 2000))
        elif kind == "compaction":
            close_section()
            tag = "requested" if data.get("requested") else "forced"
            print(c(f"\n✂ compaction ({tag}): {data['detail']}", "mag"))
        elif kind == "warning":
            close_section()
            print(c(f"\n! {data['message']}", "yellow"))

    return on_event


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="xagent",
        description="An LLM harness whose context window is a Python REPL.",
    )
    parser.add_argument("task", nargs="?", help="what the agent should do")
    parser.add_argument("-f", "--task-file", help="read the task from a file")
    parser.add_argument("-p", "--provider", default=None,
                        choices=sorted(config.BACKENDS), help="which backend to use")
    parser.add_argument("-m", "--model", default=None, help="override the driver model")
    parser.add_argument("-t", "--thinking", default=None, choices=config.THINKING_CHOICES,
                        help=f"extended thinking level (default {config.DEFAULT_THINKING}; "
                             f"`off` disables)")
    parser.add_argument("-s", "--sampling", default=None, choices=sorted(config.SAMPLING),
                        help="sampling preset (qwen only; the Anthropic API rejects these)")
    parser.add_argument("-C", "--cwd", default=None, help="working directory for the kernel")
    parser.add_argument("-n", "--max-turns", type=int, default=40)
    parser.add_argument("-b", "--budget", type=int, default=None,
                        help="context budget in tokens before compaction is forced")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="stream the reasoning in full (default: one live token count)")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args(argv)
    if args.budget is not None and args.budget <= 0:
        parser.error("--budget must be positive")

    if args.task_file:
        task = Path(args.task_file).read_text()
    elif args.task:
        task = args.task
    else:
        parser.error("provide a task, or -f/--task-file")

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

    runner = Runner(
        task,
        provider=provider,
        cwd=args.cwd,
        max_turns=args.max_turns,
        budget=args.budget,
        on_event=make_printer(colour, args.verbose, live_updates=sys.stdout.isatty()),
    )
    result = runner.run()

    print(c("\n" + "═" * 60, "dim"))
    if result.error:
        print(c(f"✗ {result.finish}: {result.error}", "red"))
    else:
        print(c(f"✓ {result.finish}", "green"))
    if result.degraded:
        print(c("  (value could not be serialized — shown as a rendering)", "yellow"))
    if result.value is not None:
        rendered = result.value if isinstance(result.value, str) else repr(result.value)
        print("\n" + textwrap.indent(rendered.strip()[:4000], "  "))

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
