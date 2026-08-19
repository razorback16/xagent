"""Eviction (tier 3) end to end: real kernel for the variable table, real model for
the summary. This is the tier that can lose information, so it is worth pinning down
directly rather than hoping a live session happens to overflow.

Run with:  uv run python tests/test_compaction.py [provider]
"""

from __future__ import annotations

import json
import sys

from xagent.compress import MIN_BATCH, MIN_GAP, Compressor
from xagent.context import ContextStore
from xagent.kernel import Kernel
from xagent.provider import Provider

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32m✓\033[0m" if cond else "\033[31m✗\033[0m"
    print(f"  {mark} {name}" + (f"  — {detail}" if detail and not cond else ""))


def main() -> int:
    provider = Provider(backend=sys.argv[1] if len(sys.argv) > 1 else "codiv")
    print(f"provider {provider.backend.name}/{provider.backend.worker_model}")
    print("starting kernel…")
    k = Kernel()

    try:
        # A session that discovered things and bound them to variables.
        k.execute("marker = 'NEEDLE-7731'")
        k.execute("findings = [{'file': f'mod{i}.py', 'issues': i % 3} for i in range(240)]")
        k.execute("total_issues = sum(f['issues'] for f in findings)")
        k.execute("config_path = 'src/xagent/config.py'")

        store = ContextStore(task="Audit the repo for issues and report the total.",
                             system="SYSTEM PROMPT " * 120)
        narrative = [
            "marker = 'NEEDLE-7731'",
            "findings = scan_repo()",
            "total_issues = sum(f['issues'] for f in findings)",
            "config_path = locate_config()",
        ]
        for i in range(26):
            code = narrative[i] if i < len(narrative) else f"inspect_module({i})"
            store.add(code, f"result of step {i}: " + "detail " * 120, f"tu_{i}",
                      thought=f"Working on step {i}.")

        before = store.estimated_tokens()
        comp = Compressor(provider, budget=6000, keep_recent=8)
        print(f"\nstore: {len(store.cells)} cells, ~{before:,} tokens")

        print("\ntier 3 — eviction")
        event = comp.evict(store, k)
        check("eviction happened", event is not None, "returned None")
        if event is None:
            raise SystemExit("cannot continue without an eviction")
        after = store.estimated_tokens()
        print(f"  {event}")
        check("context actually shrank", after < before, f"{before:,} -> {after:,}")
        check("recent cells are protected", len(store.by_state("live")) == 8,
              f"{len(store.by_state('live'))} live")
        check("older cells are evicted", len(store.by_state("evicted")) == 18,
              f"{len(store.by_state('evicted'))} evicted")

        report = store.state_report
        print("\n  ── state report ──")
        print("\n".join("    " + ln for ln in report.splitlines()[:6]))
        print("    …")

        print("\n  the variable table half")
        check("table names every live variable",
              all(v in report for v in ("marker", "findings", "total_issues", "config_path")),
              report[-900:])
        check("table carries the needle's value", "NEEDLE-7731" in report, report[-900:])
        check("table carries collection sizes", "len=240" in report or "240" in report,
              report[-900:])
        # The report used to call its frozen table "authoritative". It is a
        # snapshot and goes stale the moment the model rebinds anything, so it now
        # says so and defers to the live block rendered in the tail.
        check("report admits the snapshot can go stale",
              "stale" in report.lower() or "may since have changed" in report.lower(),
              report[:400])
        check("report points at the live block as the current truth",
              "live-variables" in report, report[:400])
        check("report says unbound values are gone, not merely summarized",
              "never bound" in report.lower(), report[:400])

        print("\n  the summary half")
        summary = report.split("## What happened")[1].split("## Live variables")[0].strip()
        check("summary is non-trivial", len(summary) > 120, f"{len(summary)} chars")
        check("summary mentions the objective",
              any(w in summary.lower() for w in ("audit", "issue", "repo", "scan")),
              summary[:300])

        print("\n  the evicted span is gone from the wire format")
        wire = json.dumps(store.messages())
        check("evicted cell code is absent", "inspect_module(3)" not in wire)
        check("evicted cell output is absent", "result of step 3:" not in wire)
        check("state report rides in the first message",
              "compacted-history" in json.dumps(store.messages()[0]))
        roles = [m["role"] for m in store.messages()]
        check("messages still alternate legally",
              roles[0] == "user" and all(a != b for a, b in zip(roles, roles[1:])), str(roles[:6]))
        check("cache breakpoints stay within the limit of 4",
              1 <= wire.count("cache_control") <= 4, f"{wire.count('cache_control')}")

        print("\n  recovery after compaction: the kernel is still ground truth")
        out = k.execute("marker")
        check("needle re-derivable from the kernel", "NEEDLE-7731" in out.render(), out.render())
        out = k.execute("len(findings), total_issues")
        check("collections survived exactly", "240" in out.render(), out.render())

        print("\npolicy: batching and cooldown")
        check(f"cooldown blocks an immediate re-compaction (MIN_GAP={MIN_GAP})",
              not comp.should_compact(store))
        for i in range(MIN_GAP + 2):
            store.add(f"more_work({i})", "padding " * 400, f"tu_late_{i}")
        check("compaction resumes once new cells accrue", comp.should_compact(store))

        print("\npolicy: a compaction that would not help is refused")
        tiny = ContextStore(task="t", system="S" * 40)
        for i in range(MIN_BATCH - 1):
            tiny.add(f"f({i})", "out " * 500, f"t_{i}")
        small = Compressor(provider, budget=500, keep_recent=8)
        check("under-sized batch is skipped", not small.should_compact(tiny))
        check("and reported as stalled rather than silently ignored", small.stalled)

    finally:
        k.shutdown()

    print(f"\n{'─' * 60}\n{len(PASS)} passed, {len(FAIL)} failed")
    for name in FAIL:
        print(f"  FAILED: {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
