"""Run a short session, then print the exact context window that was sent.

    uv run python demo/show_context.py [provider]
"""

from __future__ import annotations

import sys

from xagent.context import est_tokens
from xagent.runner import Runner

TASK = (
    "Read src/xagent/runtime.py into a variable called src. Then, in separate cells, "
    "report (1) how many lines it has, (2) how many top-level 'def ' lines it has, and "
    "(3) the name of the longest top-level function by line count. Then answer in one line and finish."
)


def rule(title: str) -> None:
    print(f"\n\033[1m{'═' * 78}\n {title}\n{'═' * 78}\033[0m")


def main() -> int:
    provider = sys.argv[1] if len(sys.argv) > 1 else "codiv"
    runner = Runner(TASK, backend=provider, max_turns=10)
    runner._cleanup = lambda: None  # keep the kernel so we can introspect it after
    result = runner.run()
    store = runner.store

    rule("THE CONTEXT WINDOW, as the model receives it")

    print("\n\033[36m── system (stable prefix, cached) ──\033[0m")
    print(f"  [~{est_tokens(store.system):,} tokens teaching the REPL discipline]")
    print(f"  {store.system.strip().splitlines()[0]}")
    print("  …")

    for i, msg in enumerate(store.messages()):
        role = msg["role"]
        colour = "33" if role == "user" else "32"
        print(f"\n\033[{colour}m── message {i}: {role} ──\033[0m")
        for block in msg["content"]:
            btype = block["type"]
            cached = " \033[35m← cache breakpoint\033[0m" if "cache_control" in block else ""
            if btype == "text":
                print(f"  \033[2m<text>\033[0m{cached}")
                for line in block["text"].strip().splitlines()[:14]:
                    print(f"    {line[:96]}")
            elif btype == "tool_use":
                print(f"  \033[2m<tool_use name=python>\033[0m{cached}")
                for line in block["input"]["code"].strip().splitlines():
                    print(f"    \033[36m│\033[0m {line[:96]}")
            elif btype == "tool_result":
                body = block["content"]
                print(f"  \033[2m<tool_result>\033[0m{cached}  \033[2m({est_tokens(body)} tokens)\033[0m")
                for line in body.strip().splitlines()[:10]:
                    print(f"    \033[2m│\033[0m {line[:96]}")

    rule("WHAT IT COST")
    live = store.live()
    print(f"  cells               {len(live)}")
    print(f"  system prompt      ~{est_tokens(store.system):,} tokens")
    print(f"  all cells together ~{sum(c.tokens() for c in live):,} tokens")
    print(f"  whole context      ~{store.estimated_tokens():,} tokens")

    rule("WHAT THE KERNEL HOLDS — live, exact, and NOT in the context")
    print(runner.kernel.probe("import xagent.runtime as _r; _r._emit_var_table()"))
    got = runner.kernel.probe("print(len(src), 'characters of source held in `src`')")
    print(f"\n  {got}")
    print(f"  …of which the context above carries almost none.")

    print(f"\n  \033[1mresult:\033[0m {result.value}")
    runner.kernel.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
