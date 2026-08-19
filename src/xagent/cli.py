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


def make_printer(colour, verbose: bool):
    c = paint(colour)

    def indent(text: str, prefix: str, limit: int) -> str:
        body = text.strip()
        if len(body) > limit:
            body = body[:limit] + " …"
        return textwrap.indent(body, prefix)

    def on_event(kind: str, data: dict) -> None:
        if kind == "kernel_ready":
            print(c(f"● kernel ready (depth {data['depth']})", "dim"))
        elif kind == "turn":
            head = c(f"\n─── turn {data['n']}", "dim")
            tokens = c(f"~{data['tokens']:,} tok", "dim")
            print(f"{head} {tokens}")
            if verbose and data.get("thinking"):
                print(indent(data["thinking"], c("  ┊ ", "dim"), 900))
            if data.get("text"):
                print(indent(data["text"], "  ", 1200))
            if data.get("code"):
                print(c("  ▸ python", "cyan"))
                print(textwrap.indent(data["code"].strip(), c("  │ ", "cyan")))
        elif kind == "cell":
            colour_name = "dim" if data["ok"] else "red"
            print(c("  ← output", colour_name))
            print(indent(data["output"], c("  │ ", colour_name), 2000))
        elif kind == "compaction":
            tag = "requested" if data.get("requested") else "forced"
            print(c(f"\n✂ compaction ({tag}): {data['detail']}", "mag"))
        elif kind == "warning":
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
    parser.add_argument("-t", "--thinking", default=None,
                        choices=sorted(config.THINKING_BUDGETS), help="extended thinking level")
    parser.add_argument("-s", "--sampling", default=None, choices=sorted(config.SAMPLING),
                        help="sampling preset (qwen only; the Anthropic API rejects these)")
    parser.add_argument("-C", "--cwd", default=None, help="working directory for the kernel")
    parser.add_argument("-n", "--max-turns", type=int, default=40)
    parser.add_argument("-b", "--budget", type=int, default=None,
                        help="context budget in tokens before compaction is forced")
    parser.add_argument("-v", "--verbose", action="store_true", help="show model thinking")
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

    tags = f"{' · thinking=' + args.thinking if args.thinking else ''}"
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
        on_event=make_printer(colour, args.verbose),
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
