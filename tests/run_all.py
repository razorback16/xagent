"""Run every suite. Exits non-zero if anything fails.

    uv run python tests/run_all.py            # all suites (codiv for live ones)
    uv run python tests/run_all.py --offline  # only the suites needing no network
    uv run python tests/run_all.py anthropic  # drive the live suites at Anthropic
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent

# (module, needs a provider)
SUITES = [
    ("test_mechanics", False),
    ("test_sampling", False),
    ("test_vision", False),
    ("test_audio", False),
    ("test_regressions", False),
    ("test_finishing", False),
    ("test_subagents", False),
    ("test_pool", False),
    ("test_compaction", True),
]


def main() -> int:
    args = [a for a in sys.argv[1:]]
    offline = "--offline" in args
    provider = next((a for a in args if not a.startswith("-")), "codiv")

    results, failed = [], 0
    for module, needs_provider in SUITES:
        if needs_provider and offline:
            results.append((module, "skipped", 0.0))
            continue
        cmd = [sys.executable, str(HERE / f"{module}.py")]
        if needs_provider:
            cmd.append(provider)
        print(f"\n\033[1m━━━ {module} ━━━\033[0m", flush=True)
        started = time.monotonic()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.monotonic() - started
        tail = [ln for ln in proc.stdout.splitlines() if "passed" in ln]
        summary = tail[-1].strip() if tail else f"exit {proc.returncode}"
        if proc.returncode != 0:
            failed += 1
            print(proc.stdout[-3000:])
            print(proc.stderr[-2000:], file=sys.stderr)
        results.append((module, summary, elapsed))

    print(f"\n\033[1m{'━' * 56}\033[0m")
    for module, summary, elapsed in results:
        mark = "\033[32m✓\033[0m" if "0 failed" in summary else (
            "\033[2m–\033[0m" if summary == "skipped" else "\033[31m✗\033[0m")
        print(f"  {mark} {module:<18} {summary:<24} {elapsed:5.1f}s")
    print(f"\n{'all suites passed' if not failed else f'{failed} suite(s) FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
