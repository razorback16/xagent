"""The context window as an ordered list of REPL cells.

Not a message list. Cells carry state (live / folded / evicted) and are rendered
into Anthropic messages on each turn, which is what makes compaction a data
operation rather than string surgery on a transcript.

Cache policy is load-bearing and easy to get wrong. An append-only transcript is
the ideal shape for prompt caching -- near-total prefix hits -- and every edit to
an earlier cell throws that away. So: pinned notes are appended at the *tail*
rather than the head (pinned means "survives compaction", not "sits first"), and
the breakpoint that matters rides the second-to-last cell, which no note or new
turn disturbs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CACHE = {"type": "ephemeral"}

FOLDED = "[output folded to reclaim context — the code above ran; its variables are still live]"

# Code is never folded, only evicted, and the newest KEEP_RECENT cells are never
# evicted -- so an uncapped code block is a permanent context cost. Writing a file
# means emitting its whole body as a literal, which is the primary workload for a
# coding agent, so this cap is load-bearing rather than defensive.
MAX_CODE_CHARS = 6000


def est_tokens(text: str) -> int:
    """Cheap character-based estimate, used for budgeting and attribution.

    Systematically low on code, JSON and tracebacks, and 3-4x low on CJK. Callers
    must not mix it with real API counts; `ContextStore.used_tokens()` keeps
    everything in one calibrated unit instead.
    """
    return len(text) // 4 + 1


@dataclass
class Cell:
    n: int
    code: str
    output: str
    tool_use_id: str
    thought: str = ""
    state: str = "live"  # live | folded | evicted
    # Raw thinking blocks, replayed verbatim when the backend demands it (Anthropic
    # rejects a tool-use continuation whose prior thinking blocks are missing).
    thinking_blocks: list[dict] = field(default_factory=list)

    @property
    def shown_output(self) -> str:
        return FOLDED if self.state == "folded" else self.output

    def tokens(self) -> int:
        if self.state == "evicted":
            return 0
        return est_tokens(self.thought) + est_tokens(self.code) + est_tokens(self.shown_output) + 12


@dataclass
class ContextStore:
    task: str
    system: str
    cells: list[Cell] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)
    state_report: str = ""      # replaces every evicted span; there is only ever one
    real_tokens: int = 0        # input_tokens from the most recent response
    cache_read: int = 0
    cache_write: int = 0
    replay_thinking: bool = False
    # Ratio of what the API actually charged to what est_tokens() predicted. Every
    # budget decision is made in calibrated-estimate units so that no comparison
    # ever puts a measured number on one side and a guess on the other -- a mix
    # that biases every "did compaction help?" test toward yes.
    calibration: float = 1.0
    live_vars: str = ""         # refreshed each turn once history has been evicted
    live_files: str = ""        # filesystem mutations, refreshed each turn

    # ------------------------------------------------------------------ cells

    def add(self, code: str, output: str, tool_use_id: str, thought: str = "",
            thinking_blocks: list[dict] | None = None) -> Cell:
        if len(code) > MAX_CODE_CHARS:
            keep = MAX_CODE_CHARS // 2
            dropped = len(code) - 2 * keep
            code = (code[:keep]
                    + f"\n\n# … [{dropped:,} chars of this cell elided from your context; "
                      f"it ran in full, and In[{len(self.cells) + 1}] holds the original] …\n\n"
                    + code[-keep:])
        cell = Cell(n=len(self.cells) + 1, code=code, output=output,
                    tool_use_id=tool_use_id, thought=thought,
                    thinking_blocks=thinking_blocks or [])
        self.cells.append(cell)
        return cell

    def live(self) -> list[Cell]:
        return [c for c in self.cells if c.state != "evicted"]

    def by_state(self, state: str) -> list[Cell]:
        return [c for c in self.cells if c.state == state]

    # --------------------------------------------------------------- budgets

    def estimated_tokens(self) -> int:
        base = est_tokens(self.system) + est_tokens(self.task) + est_tokens(self.state_report)
        base += sum(est_tokens(f"{k}: {v}") for k, v in self.notes.items())
        return base + sum(c.tokens() for c in self.cells)

    def observe(self, real_tokens: int) -> None:
        """Record what the API charged and re-derive the calibration factor."""
        self.real_tokens = real_tokens
        raw = self.estimated_tokens()
        if raw > 0 and real_tokens > 0:
            ratio = real_tokens / raw
            # Clamp against nonsense from a tiny or unusual context.
            self.calibration = max(0.5, min(4.0, ratio))

    def used_tokens(self) -> int:
        """The single unit every budget decision is made in."""
        return int(self.estimated_tokens() * self.calibration)

    def heaviest(self, n: int = 3) -> list[tuple[int, int]]:
        ranked = sorted(
            ((c.n, c.tokens()) for c in self.cells if c.state == "live"),
            key=lambda t: -t[1],
        )
        return ranked[:n]

    def accounting(self, budget: int) -> dict:
        return {
            "used": self.used_tokens(),
            "budget": budget,
            "live": len(self.by_state("live")),
            "folded": len(self.by_state("folded")),
            "evicted": len(self.by_state("evicted")),
            "cache_read": self.cache_read,
            "cache_write": self.cache_write,
            "heaviest": self.heaviest(),
        }

    # ---------------------------------------------------------------- render

    def _notes_block(self) -> str:
        """Pinned notes and the live variable table, both riding the tail.

        The variable table has to be regenerated every turn to stay true. Putting
        it here rather than in the compacted-history block keeps it current without
        rewriting the cached prefix.
        """
        parts = []
        if self.notes:
            body = "\n".join(f"  - {k}: {v}" for k, v in self.notes.items())
            parts.append(f"<pinned-notes>\n{body}\n</pinned-notes>")
        if self.live_files:
            parts.append(
                "<files-you-have-changed note=\"recorded as they happened; complete\">\n"
                f"{self.live_files}\n</files-you-have-changed>"
            )
        if self.live_vars:
            parts.append(
                "<live-variables note=\"introspected just now; authoritative\">\n"
                f"{self.live_vars}\n</live-variables>"
            )
        return ("\n\n" + "\n\n".join(parts)) if parts else ""

    def messages(self) -> list[dict]:
        opening: list[dict] = [{"type": "text", "text": f"<task>\n{self.task}\n</task>"}]
        if self.state_report:
            opening.append({"type": "text", "text": self.state_report})
        opening[-1]["cache_control"] = CACHE

        msgs: list[dict] = [{"role": "user", "content": opening}]
        live = self.live()

        # The breakpoint that earns its keep: the second-to-last cell is stable
        # across the next turn *and* untouched by the notes block, so the whole
        # growing middle of the conversation stays cached.
        anchor = live[-2].n if len(live) >= 2 else None

        for cell in live:
            assistant: list[dict] = []
            if self.replay_thinking:
                assistant.extend(cell.thinking_blocks)
            if cell.thought.strip():
                assistant.append({"type": "text", "text": cell.thought})
            assistant.append({
                "type": "tool_use",
                "id": cell.tool_use_id,
                "name": "python",
                "input": {"code": cell.code},
            })
            msgs.append({"role": "assistant", "content": assistant})

            body = cell.shown_output
            if cell is live[-1]:
                body += self._notes_block()
            result: dict = {
                "type": "tool_result",
                "tool_use_id": cell.tool_use_id,
                "content": body,
            }
            if cell.n == anchor:
                result["cache_control"] = CACHE
            msgs.append({"role": "user", "content": [result]})

        return msgs
