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

from xagent.audio import AudioAttachment
from xagent.vision import IMAGE_TOKEN_ESTIMATE, ImageAttachment

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
    # Which turn ran this cell. A turn that batched several calls produced
    # several cells sharing one id, and they render as one assistant message --
    # see `ContextStore.turns()`. Defaults to the cell's own number, which makes
    # a cell added without one a turn of its own.
    turn: int = 0
    # Raw thinking blocks, replayed verbatim when the backend demands it (Anthropic
    # rejects a tool-use continuation whose prior thinking blocks are missing).
    thinking_blocks: list[dict] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)

    @property
    def shown_output(self) -> str:
        return FOLDED if self.state == "folded" else self.output

    @property
    def shown_images(self) -> list[dict]:
        # Images are cell output too. Tier-2 compaction promises to drop old
        # outputs, so retaining their multimodal blocks would preserve most of
        # the cost while claiming the cell was folded.
        return self.images if self.state == "live" else []

    def tokens(self) -> int:
        if self.state == "evicted":
            return 0
        return (est_tokens(self.thought) + est_tokens(self.code)
                + est_tokens(self.shown_output)
                + len(self.shown_images) * IMAGE_TOKEN_ESTIMATE + 12)


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
    images: list[ImageAttachment] = field(default_factory=list)
    audio: list[AudioAttachment] = field(default_factory=list)

    # ------------------------------------------------------------------ cells

    def add(self, code: str, output: str, tool_use_id: str, thought: str = "",
            thinking_blocks: list[dict] | None = None, turn: int | None = None,
            images: list[dict] | None = None) -> Cell:
        if len(code) > MAX_CODE_CHARS:
            keep = MAX_CODE_CHARS // 2
            dropped = len(code) - 2 * keep
            code = (code[:keep]
                    + f"\n\n# … [{dropped:,} chars of this cell elided from your context; "
                      f"it ran in full, and In[{len(self.cells) + 1}] holds the original] …\n\n"
                    + code[-keep:])
        n = len(self.cells) + 1
        cell = Cell(n=n, code=code, output=output,
                    tool_use_id=tool_use_id, thought=thought,
                    thinking_blocks=thinking_blocks or [],
                    images=images or [],
                    turn=n if turn is None else turn)
        self.cells.append(cell)
        return cell

    def live(self) -> list[Cell]:
        return [c for c in self.cells if c.state != "evicted"]

    def turns(self, cells: list[Cell]) -> list[list[Cell]]:
        """Cells batched back into the turns that ran them.

        A turn that made several `python` calls is one assistant message however
        many cells it ran: the API requires every tool_use block in a message to
        be answered by a tool_result in the single user message that follows, so
        a batch of three renders as three tool_use blocks and three results --
        not as three conversational rounds the model never had.

        Grouped by adjacency rather than by a dict, because cells are appended in
        order and only neighbours can share a turn; a compacted span in the middle
        simply ends the group, which is what it did to the turn as well.
        """
        groups: list[list[Cell]] = []
        for cell in cells:
            if groups and groups[-1][-1].turn == cell.turn:
                groups[-1].append(cell)
            else:
                groups.append([cell])
        return groups

    def by_state(self, state: str) -> list[Cell]:
        return [c for c in self.cells if c.state == state]

    # --------------------------------------------------------------- budgets

    def estimated_tokens(self) -> int:
        base = est_tokens(self.system) + est_tokens(self.task) + est_tokens(self.state_report)
        base += sum(est_tokens(f"{k}: {v}") for k, v in self.notes.items())
        base += len(self.images) * IMAGE_TOKEN_ESTIMATE
        base += sum(clip.tokens() for clip in self.audio)
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
        if self.images:
            labels = "\n".join(f"  - {image.path}" for image in self.images)
            opening.append({
                "type": "text",
                "text": (
                    "<attached-images>\n"
                    f"{labels}\n"
                    "The image content is attached below; use it as visual evidence.\n"
                    "</attached-images>"
                ),
            })
            opening.extend(image.content_block() for image in self.images)
            # Keep the cache breakpoint on text. This is accepted by both the
            # real Messages API and Anthropic-compatible servers, while the image
            # blocks remain ordinary multimodal content.
            opening.append({"type": "text", "text": "<end-attached-images>"})
        if self.audio:
            # Audio enters as a picture of itself. The text block carries the
            # measurements the picture can only be read off approximately --
            # duration, peak, how much of it is silence -- so the model is never
            # squinting at a plot for a number it could be told.
            body = "\n\n".join(clip.text_block() for clip in self.audio)
            opening.append({
                "type": "text",
                "text": (
                    "<attached-audio>\n"
                    f"{body}\n"
                    "An analysis panel for each clip is attached below: waveform, "
                    "log-frequency spectrogram, level over time (all three sharing "
                    "one time axis), and how long the clip spends at each level.\n"
                    "</attached-audio>"
                ),
            })
            opening.extend(clip.image_block() for clip in self.audio)
            opening.append({"type": "text", "text": "<end-attached-audio>"})
        opening[-1]["cache_control"] = CACHE

        msgs: list[dict] = [{"role": "user", "content": opening}]
        live = self.live()

        # The breakpoint that earns its keep: the second-to-last cell is stable
        # across the next turn *and* untouched by the notes block, so the whole
        # growing middle of the conversation stays cached.
        anchor = live[-2].n if len(live) >= 2 else None

        for group in self.turns(live):
            assistant: list[dict] = []
            if self.replay_thinking:
                for cell in group:
                    assistant.extend(cell.thinking_blocks)
            # The prose belongs to the turn, not to any one of its calls, so it is
            # written once above the batch -- which is where the model wrote it.
            thought = "\n".join(c.thought for c in group if c.thought.strip())
            if thought.strip():
                assistant.append({"type": "text", "text": thought})
            for cell in group:
                assistant.append({
                    "type": "tool_use",
                    "id": cell.tool_use_id,
                    "name": "python",
                    "input": {"code": cell.code},
                })
            msgs.append({"role": "assistant", "content": assistant})

            results: list[dict] = []
            for cell in group:
                body = cell.shown_output
                if cell is live[-1]:
                    body += self._notes_block()
                result: dict = {
                    "type": "tool_result",
                    "tool_use_id": cell.tool_use_id,
                    "content": body,
                }
                if cell.shown_images:
                    result["content"] = [
                        {"type": "text", "text": body}, *cell.shown_images
                    ]
                if cell.n == anchor:
                    result["cache_control"] = CACHE
                results.append(result)
            msgs.append({"role": "user", "content": results})

        return msgs
