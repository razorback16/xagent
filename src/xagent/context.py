"""The context window as an ordered list of REPL cells and the messages between them.

Not a message list. Cells carry state (live / folded / evicted) and are rendered
into Anthropic messages on each turn, which is what makes compaction a data
operation rather than string surgery on a transcript.

Two things that are not cells also belong in the order: what a person said, and the
prose a response ended on. Those are `Mark`s. They interleave with the cells and are
never folded or evicted -- an instruction from whoever the agent works for is the
least droppable thing in the window, and the answer it replies to is what makes the
next instruction make sense.

Cache policy is load-bearing and easy to get wrong. An append-only transcript is
the ideal shape for prompt caching -- near-total prefix hits -- and every edit to
an earlier cell throws that away. So: pinned notes are appended at the *tail*
rather than the head (pinned means "survives compaction", not "sits first"), and
the breakpoint that matters rides the second-to-last cell, which no note, mark or
new turn disturbs.
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
class Mark:
    """A message in the context that is not a cell.

    Two kinds. A `prompt` is someone talking to this run -- the person at the
    terminal, or the parent that spawned it -- and renders as a `user` message. An
    `answer` is the prose a response ended on, and renders as an `assistant` one.

    `after` anchors it between the cells: the cell number it follows, 0 for before
    any cell ran. Deliberately a number rather than a position in a list, so that a
    mark keeps its place when the cell it followed is evicted.

    Marks are never folded and never evicted. They are also the reason the store is
    not simply a list of cells: an instruction and the answer before it are the two
    things a compaction has no business reclaiming.

    An answer carries no thinking blocks. Anthropic requires them replayed on an
    assistant turn that holds a tool_use, and this one holds none.
    """
    n: int
    role: str                 # user | assistant
    text: str
    after: int
    # What the text is wrapped in, so the model can tell a person talking from a
    # tool result. Empty for an answer, which is the model's own prose and needs no
    # label -- it is in an assistant message, which says that already.
    tag: str = ""
    images: list[dict] = field(default_factory=list)

    def blocks(self) -> list[dict]:
        body = f"<{self.tag}>\n{self.text}\n</{self.tag}>" if self.tag else self.text
        return [{"type": "text", "text": body}, *self.images]

    def tokens(self) -> int:
        return (est_tokens(self.text) + len(self.tag) // 2
                + len(self.images) * IMAGE_TOKEN_ESTIMATE + 8)


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
    marks: list[Mark] = field(default_factory=list)
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

    # ------------------------------------------------------------------ marks

    def _mark(self, role: str, text: str, tag: str,
              images: list[dict] | None = None) -> Mark:
        mark = Mark(n=len(self.marks) + 1, role=role, text=text,
                    after=len(self.cells), tag=tag, images=images or [])
        self.marks.append(mark)
        return mark

    def add_prompt(self, text: str, tag: str = "user-message",
                   images: list[dict] | None = None) -> Mark:
        """Record someone talking to this run, as a `user` message of its own."""
        return self._mark("user", text, tag, images)

    def add_answer(self, text: str) -> Mark:
        """Record the prose a response ended on, so the next prompt has a reply to
        follow. Without it a conversation reads as a list of questions."""
        return self._mark("assistant", text, "")

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
        base += sum(m.tokens() for m in self.marks)
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
        return "\n\n".join(parts)

    def messages(self) -> list[dict]:
        # An empty task is a session that was opened without one: the person will say
        # what they need, and it arrives as a mark like every message after it. The
        # head still has to exist, because it carries the cache breakpoint.
        head = (f"<task>\n{self.task}\n</task>" if self.task.strip() else
                "<session>\nA person opened this session without a request. What "
                "they need follows as their first message.\n</session>")
        opening: list[dict] = [{"type": "text", "text": head}]
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

        msgs: list[dict] = []

        def push(role: str, blocks: list[dict]) -> None:
            """Add content, merging into the last message when the role matches.

            This is what lets a prompt that arrived at a turn boundary sit in the
            same `user` message as the tool_results it follows. The API requires
            every tool_use to be answered in the message directly after it, so a
            prompt in a message of its own there would be rejected.
            """
            if msgs and msgs[-1]["role"] == role:
                msgs[-1]["content"].extend(blocks)
            else:
                msgs.append({"role": role, "content": list(blocks)})

        push("user", opening)
        live = self.live()

        # The breakpoint that earns its keep: the second-to-last cell is stable
        # across the next turn *and* untouched by the notes block, so the whole
        # growing middle of the conversation stays cached.
        anchor = live[-2].n if len(live) >= 2 else None

        pending = list(self.marks)

        def flush(before: int | None) -> None:
            """Emit every mark that belongs ahead of cell `before`.

            Keyed on `<` rather than `==` on purpose: a mark anchored to a cell that
            eviction has since removed still belongs where it was said.
            """
            while pending and (before is None or pending[0].after < before):
                mark = pending.pop(0)
                push(mark.role, mark.blocks())

        for group in self.turns(live):
            flush(group[0].n)
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
            push("assistant", assistant)

            results: list[dict] = []
            for cell in group:
                body = cell.shown_output
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
            push("user", results)

        flush(None)

        # The notes block goes last of everything, which is the whole point of it:
        # the variable table and the file ledger are regenerated every turn and have
        # to be the newest thing the model reads. It used to ride the final cell's
        # tool_result, and a prompt or an answer after that cell would have left it
        # buried mid-conversation describing a namespace as it was.
        #
        # `messages()` is only ever called to sample, and sampling needs a `user`
        # message last, so in practice this always finds one. A tail that is not
        # `user` means nobody is about to be asked anything, and there is nothing to
        # attach it to.
        if (notes := self._notes_block()) and msgs and msgs[-1]["role"] == "user":
            msgs[-1]["content"].append({"type": "text", "text": notes})

        return msgs
