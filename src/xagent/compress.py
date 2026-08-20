"""Context compaction.

Three tiers, cheapest first:

  1. display caps        always on, enforced in `brief` inside the kernel
  2. fold outputs        drop stdout for old cells, keep the code -- code is dense,
                         high-signal, and cheap, and usually says what happened
  3. evict a span        replace it with one synthetic block holding an LLM summary
                         of the *code* plus a freshly introspected variable table

Tier 3 summarizes code and never values. If cell 12 mutated `df`, a value summary
written at cell 40 lies silently; the variable table is regenerated from the live
kernel at compaction time and cannot.
"""

from __future__ import annotations

from dataclasses import dataclass

from xagent import config
from xagent.context import ContextStore

# Compaction must be rare and batched. Every compaction discards the cached prompt
# prefix and pays for a summary, so a policy that trims a little on every turn is
# slower and more expensive than one that does nothing. These two constants are what
# stop that: a floor on batch size, and a cooldown measured in new cells.
MIN_BATCH = 6
MIN_GAP = 4

SUMMARY_PROMPT = """\
Below is a transcript of Python REPL cells from an agent session. It is about to be
dropped from the agent's context to reclaim space.

Write a compact briefing (200-350 words) that lets the agent continue without it.

Critical constraint: describe what the code DID -- goals pursued, approaches tried,
what worked, what failed and why, decisions made and their reasons, paths and
identifiers that matter. Do NOT describe the *values* of variables: a separate,
freshly-introspected variable table is appended after your summary, and it is
authoritative. If you restate values here you will contradict it as soon as
anything mutates.

Lead with the objective and current position. Be specific about file paths, function
names, and error messages -- those are what the agent cannot reconstruct. Prefer
terse prose over headings. No preamble.

{prior}
<cells>
{body}
</cells>"""


def var_table_failed(exc: Exception) -> str:
    """What stands in for the table when the kernel could not be introspected.

    Shared with the runner so both places that carry a table say the same thing:
    the blocks it lands in are labelled authoritative, and one of them is written
    into the transcript for good.
    """
    return f"  (introspection failed: {type(exc).__name__}: {exc})"


@dataclass
class CompactionEvent:
    kind: str
    cells: int
    before_tokens: int
    after_tokens: int

    @property
    def saved(self) -> int:
        return self.before_tokens - self.after_tokens

    def __str__(self) -> str:
        delta = f"−{self.saved:,}" if self.saved >= 0 else f"+{-self.saved:,}"
        return (f"{self.kind} {self.cells} cell(s): "
                f"{self.before_tokens:,} → {self.after_tokens:,} tokens ({delta})")


class Compressor:
    def __init__(self, provider, budget: int | None = None,
                 watermark: float | None = None, keep_recent: int | None = None):
        self.provider = provider
        if budget is None:
            backend = getattr(provider, "backend", None)
            budget = backend.context_budget if backend else 160_000
        self.budget = budget
        self.watermark = watermark if watermark is not None else config.WATERMARK
        self.keep_recent = keep_recent if keep_recent is not None else config.KEEP_RECENT
        self.events: list[CompactionEvent] = []
        self._last_at = 0     # cell count when we last compacted
        self.stalled = False  # over budget but compaction cannot help

    # ------------------------------------------------------------------ policy

    @property
    def threshold(self) -> int:
        return int(self.budget * self.watermark)

    def should_compact(self, store: ContextStore) -> bool:
        """Over the watermark, with a batch worth compacting, and off cooldown."""
        self.stalled = False  # a query must not leave a stale verdict behind
        if store.used_tokens() < self.threshold:
            return False

        cutoff = self._cutoff(store, None)
        eligible = [c for c in store.cells if c.state != "evicted" and c.n < cutoff]
        if len(eligible) < MIN_BATCH:
            # Nothing compactable is left: the floor (system prompt, task, pinned
            # notes, state report, protected tail) is itself at the threshold.
            # Compacting again would churn the cache for nothing.
            self.stalled = True
            return False
        if len(store.cells) - self._last_at < MIN_GAP:
            return False
        return True

    def _cutoff(self, store: ContextStore, before: int | None) -> int:
        """Highest cell id eligible for compaction.

        The most recent `keep_recent` cells are always off limits, so the tail of
        the prompt -- the part being appended to, and the part the cache is warm
        on -- never shifts.
        """
        live = store.live()
        if self.keep_recent <= 0:
            # Protect nothing: `live[-0]` is `live[0]`, which would protect
            # everything and silently disable compaction.
            natural = live[-1].n + 1 if live else 0
        else:
            natural = live[-self.keep_recent].n if len(live) > self.keep_recent else 0
        return min(natural, before) if before is not None else natural

    # ------------------------------------------------------------------- tiers

    def fold(self, store: ContextStore, before: int | None = None) -> CompactionEvent | None:
        cutoff = self._cutoff(store, before)
        targets = [c for c in store.cells if c.state == "live" and c.n < cutoff]
        if not targets:
            return None
        start = store.used_tokens()
        undo = [(c, c.state) for c in targets]
        for cell in targets:
            cell.state = "folded"

        after = store.used_tokens()
        if after >= start:
            # The fold marker is ~26 tokens; folding cells with shorter output than
            # that grows the context. Measure rather than assume, same as evict().
            for cell, was in undo:
                cell.state = was
            self.stalled = True
            self._last_at = len(store.cells)  # do not retry every turn
            return None

        self._last_at = len(store.cells)
        event = CompactionEvent("folded", len(targets), start, after)
        self.events.append(event)
        return event

    def evict(self, store: ContextStore, kernel, before: int | None = None) -> CompactionEvent | None:
        cutoff = self._cutoff(store, before)
        targets = [c for c in store.cells if c.state != "evicted" and c.n < cutoff]
        if not targets:
            return None
        start = store.used_tokens()
        # Snapshot enough to undo: a summary plus a variable table can easily cost
        # more than the cells it replaces, and a compaction that grows the context
        # is worse than none.
        undo = [(c, c.state) for c in targets]
        undo_report = store.state_report

        var_table = self._var_table(kernel)
        summary = self._summarize(store, targets)

        span = f"{targets[0].n}-{targets[-1].n}"
        store.state_report = (
            f"<compacted-history cells=\"{span}\">\n"
            f"{len(targets)} earlier cells were removed from your context to reclaim space.\n"
            f"Their code and output are gone. Their EFFECTS are not: every variable listed\n"
            f"below is live in your kernel and exactly correct.\n\n"
            f"## What happened\n{summary}\n\n"
            f"## Variables as of this compaction (a snapshot — may since have changed)\n"
            f"{var_table}\n\n"
            f"A <live-variables> block at the END of this conversation is re-introspected\n"
            f"every turn; where the two disagree, that one is correct and this one is stale.\n"
            f"If you need detail neither carries, re-derive it: peek() a variable or re-run\n"
            f"the query. Values the code produced but never bound to a name are gone — the\n"
            f"prose above is a lossy sketch of them, not a record.\n"
            f"</compacted-history>"
        )
        for cell in targets:
            cell.state = "evicted"

        after = store.used_tokens()
        if after >= start:
            for cell, was in undo:
                cell.state = was
            store.state_report = undo_report
            self.stalled = True
            # Arm the cooldown even on a revert. Without this, nothing about the
            # store has changed, should_compact() stays true, and every subsequent
            # turn pays for another discarded summary call.
            self._last_at = len(store.cells)
            return None

        self._last_at = len(store.cells)
        event = CompactionEvent("evicted", len(targets), start, after)
        self.events.append(event)
        return event

    # ------------------------------------------------------------------ driver

    def apply(self, store: ContextStore, kernel, mode: str = "auto",
              before: int | None = None) -> list[CompactionEvent]:
        """Run compaction. `auto` escalates only as far as it must."""
        done: list[CompactionEvent] = []
        if mode in ("fold", "auto"):
            event = self.fold(store, before)
            if event:
                done.append(event)
        needs_more = mode == "evict" or (
            mode == "auto" and store.used_tokens() >= self.threshold
        )
        if needs_more:
            event = self.evict(store, kernel, before)
            if event:
                done.append(event)
        return done

    # ------------------------------------------------------------------ pieces

    MAX_TABLE_ROWS = 60

    def _var_table(self, kernel) -> str:
        try:
            table = kernel.probe(
                "import xagent.runtime as _r; _r._emit_var_table()", timeout=45
            )
        except Exception as e:
            return var_table_failed(e)
        return self._format_var_table(table)

    def _format_var_table(self, table: str) -> str:
        """Cap and indent a raw table.

        Split from the probe that fetches one because the runner already has its
        table in hand: it rides back on the cell itself, in the payload the
        in-kernel hook appends, rather than costing a round-trip of its own.
        Compaction still probes, because it happens between cells.
        """
        if not table:
            return "  (no user variables)"
        rows = [ln.strip() for ln in table.splitlines() if ln.strip()]
        extra = len(rows) - self.MAX_TABLE_ROWS
        if extra > 0:
            rows = rows[: self.MAX_TABLE_ROWS]
            rows.append(f"… {extra:,} more variables (use %whos in a cell)")
        # probe() strips the payload, which would eat the first row's indent.
        return "\n".join("  " + r for r in rows)

    def _summarize(self, store: ContextStore, targets) -> str:
        body: list[str] = []
        for cell in targets:
            body.append(f"--- cell {cell.n} ---")
            if cell.thought.strip():
                body.append(f"[reasoning] {cell.thought.strip()[:600]}")
            body.append(cell.code.strip())
            # Always the real output: auto-compaction folds before it evicts, so
            # keying on cell.state here fed the summarizer nothing but placeholders
            # on the only path production ever takes.
            body.append(f"[output] {cell.output.strip()[:800]}")
        prior = ""
        if store.state_report:
            prior = (
                "This session was compacted before. Fold the briefing below into your "
                "new one so nothing is lost across compactions:\n"
                f"<previous-briefing>\n{store.state_report}\n</previous-briefing>\n"
            )
        prompt = SUMMARY_PROMPT.format(prior=prior, body="\n".join(body)[:120_000])
        try:
            # provider.complete() already defaults to the backend's worker model.
            summary = self.provider.complete(prompt, max_tokens=4096).strip()
            if not summary:
                raise RuntimeError("summariser returned no text")
            return summary
        except Exception as e:
            return (
                f"(summary generation failed: {type(e).__name__}: {e}. "
                f"Cells {targets[0].n}-{targets[-1].n} were dropped; the variable table "
                f"below is still authoritative.)"
            )
