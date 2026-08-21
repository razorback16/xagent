"""The conversation: prompts, answers, and a kernel that outlives both.

A run used to be one errand. It is a conversation now: `ask()` produces one response
and leaves the kernel and the context standing, so the person can reply and the
namespace the last answer was built from is still there.

What is pinned down here is the part that is easy to get subtly wrong -- where a
message lands in the rendered conversation. The API requires every `tool_use` to be
answered in the message directly after it, so a prompt that arrived at a turn
boundary has to ride the same `user` message as those tool_results rather than sit in
one of its own. The other half is compaction: a person's instruction is the least
droppable thing in the window, and eviction must not touch it or move it.

No network: the provider is a stub playing a fixed script. No kernel either where it
can be avoided, so the rendering half of this suite runs anywhere.

Run with:  uv run python tests/test_session.py
"""

from __future__ import annotations

import sys

from xagent import config
from xagent.cli import make_printer
from xagent.compress import Compressor
from xagent.context import ContextStore
from xagent.prompts import SYSTEM
from xagent.provider import Turn, Usage
from xagent.runner import Runner
from xagent.session import UserChannel

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32m✓\033[0m" if cond else "\033[31m✗\033[0m"
    print(f"  {mark} {name}" + (f"  — {detail}" if detail and not cond else ""))


class StubProvider:
    def __init__(self, turns):
        self.turns, self.n, self.seen = list(turns), 0, []
        self.backend = config.BACKENDS["codiv"]
        self.model, self.thinking, self.sampling = "stub", None, "thinking"
        self.usage, self.on_delta, self.truncated_turns = Usage(), None, 0
        self.is_subagent = False

    def sample(self, system, messages):
        self.seen.append(messages)
        turn = self.turns[self.n] if self.n < len(self.turns) else Turn()
        self.n += 1
        if isinstance(turn, BaseException):
            raise turn
        # A callable stands for something happening while the request is in flight:
        # it runs here, mid-sample, which is where a person types.
        return turn() if callable(turn) else turn

    def system_blocks(self, system):
        return [{"type": "text", "text": system}]

    def complete(self, prompt, model=None, max_tokens=4096):
        return "summary"


def work(code: str, text: str = "") -> Turn:
    return Turn(text=text, code=code, tool_use_id=f"tu_{abs(hash(code)) % 9999}",
                tool_name="python")


def answer(text: str) -> Turn:
    return Turn(text=text)


def roles(msgs) -> list[str]:
    return [m["role"] for m in msgs]


def texts(msg) -> str:
    return " ".join(b.get("text", "") for b in msg["content"])


def faults(msgs) -> list[str]:
    """Every way the rendered conversation would be rejected on the wire."""
    out = []
    if not msgs or msgs[0]["role"] != "user":
        out.append("does not open on a user message")
    for a, b in zip(msgs, msgs[1:]):
        if a["role"] == b["role"]:
            out.append(f"two {a['role']} messages in a row")
    for i, m in enumerate(msgs):
        uses = [b["id"] for b in m["content"] if b.get("type") == "tool_use"]
        if not uses:
            continue
        nxt = msgs[i + 1] if i + 1 < len(msgs) else {"role": "?", "content": []}
        if nxt["role"] != "user":
            out.append(f"tool_use at {i} is not answered by a user message")
            continue
        answered = [b["tool_use_id"] for b in nxt["content"]
                    if b.get("type") == "tool_result"]
        if answered != uses:
            out.append(f"tool_use {uses} answered by {answered}")
    return out


# ------------------------------------------------------------------- rendering


def rendering() -> None:
    print("a prompt that arrives at a turn boundary rides the tool_results")
    store = ContextStore(task="count the files", system="sys")
    store.add(code="a = 1", output="ok", tool_use_id="t1", thought="looking", turn=1)
    store.add_prompt("also count the dirs")
    store.add(code="b = 2", output="ok2", tool_use_id="t2", turn=2)
    msgs = store.messages()
    check("the conversation is valid on the wire", not faults(msgs),
          "; ".join(faults(msgs)))
    check("it renders as user / assistant / user / assistant / user",
          roles(msgs) == ["user", "assistant", "user", "assistant", "user"],
          str(roles(msgs)))
    check("the prompt sits in the same user message as the results it follows",
          "also count the dirs" in texts(msgs[2])
          and any(b.get("type") == "tool_result" for b in msgs[2]["content"]),
          str(msgs[2])[:200])
    check("after them, not before, because that is when it arrived",
          [b["type"] for b in msgs[2]["content"]] == ["tool_result", "text"],
          str([b["type"] for b in msgs[2]["content"]]))
    check("wrapped in a tag that says who is talking",
          "<user-message>\nalso count the dirs\n</user-message>" in texts(msgs[2]), "")

    print("\nan answer is an assistant message, so a reply has something to follow")
    store = ContextStore(task="count the files", system="sys")
    store.add(code="a = 1", output="ok", tool_use_id="t1", turn=1)
    store.add_answer("There are 50.")
    store.add_prompt("and the dirs?")
    msgs = store.messages()
    check("the conversation is valid", not faults(msgs), "; ".join(faults(msgs)))
    check("the answer is its own assistant message",
          msgs[-2]["role"] == "assistant" and texts(msgs[-2]) == "There are 50.",
          str(msgs[-2])[:160])
    check("the prompt after it is its own user message",
          msgs[-1]["role"] == "user" and "and the dirs?" in texts(msgs[-1]),
          str(msgs[-1])[:160])
    check("and an answer carries no tag, because its role says whose words they are",
          store.marks[0].tag == "", repr(store.marks[0].tag))

    print("\ntwo prompts in a row are one user message, not two")
    store = ContextStore(task="t", system="sys")
    store.add(code="a = 1", output="ok", tool_use_id="t1", turn=1)
    store.add_prompt("first")
    store.add_prompt("second")
    msgs = store.messages()
    check("merged into one", not faults(msgs) and roles(msgs).count("user") == 2,
          f"{roles(msgs)} {faults(msgs)}")
    check("both texts present, in order",
          texts(msgs[-1]).index("first") < texts(msgs[-1]).index("second"), "")

    print("\na session opened with no task still has a head to cache against")
    store = ContextStore(task="", system="sys")
    store.add_prompt("what changed in src today?")
    msgs = store.messages()
    check("the head says the request follows rather than pretending to be one",
          "<session>" in texts(msgs[0]) and "<task>" not in texts(msgs[0]),
          texts(msgs[0])[:160])
    check("it still carries the cache breakpoint",
          "cache_control" in msgs[0]["content"][0], str(msgs[0]["content"][0]))
    check("and the first thing the person said is in there",
          "what changed in src today?" in texts(msgs[0]), "")

    print("\nthe notes block is last of everything, or it describes a stale namespace")
    store = ContextStore(task="t", system="sys")
    store.add(code="a = 1", output="ok", tool_use_id="t1", turn=1)
    store.live_vars = "  a  int  1"
    store.add_answer("Done.")
    store.add_prompt("now the dirs")
    msgs = store.messages()
    check("it rides the final user message, past the answer and the prompt",
          "<live-variables" in texts(msgs[-1]), texts(msgs[-1])[:200])
    check("after the prompt, so the freshest thing read is the truest",
          texts(msgs[-1]).index("now the dirs")
          < texts(msgs[-1]).index("<live-variables"), "")
    check("and it appears exactly once", str(msgs).count("<live-variables") == 1, "")
    # The waiting-for-a-reply state: the last thing in the conversation is the
    # answer, and nothing is about to be sampled. There is no user message to hang
    # the block on, and inventing one would be inventing a turn.
    idle = ContextStore(task="t", system="sys", live_vars="  a  int  1")
    idle.add_answer("Done.")
    check("a tail that is not a user message gets none, having nothing to carry it",
          "<live-variables" not in str(idle.messages())
          and idle.messages()[-1]["role"] == "assistant", str(idle.messages())[-200:])

    print("\nmarks count against the budget, or the context is understated")
    store = ContextStore(task="t", system="sys")
    before = store.used_tokens()
    store.add_prompt("a rather long instruction " * 40)
    check("adding a prompt raises the reading", store.used_tokens() > before,
          f"{before} -> {store.used_tokens()}")


def against_compaction() -> None:
    print("\ncompaction may drop a cell, never a thing somebody said")
    store = ContextStore(task="t", system="sys")
    for n in range(1, 13):
        store.add(code=f"x{n} = {n}", output="ok " * 200, tool_use_id=f"t{n}", turn=n)
        if n == 3:
            store.add_prompt("actually, skip the tests")
        if n == 6:
            store.add_answer("Six done.")

    class _P:
        backend = config.BACKENDS["codiv"]

        def complete(self, prompt, model=None, max_tokens=4096):
            return "the agent bound twelve variables"

    compressor = Compressor(_P(), budget=2_000, keep_recent=2)
    events = compressor.apply(store, kernel=None, mode="fold")
    check("folding happens", events, str(events))
    events = compressor.apply(store, kernel=_FakeKernel(), mode="evict")
    check("and eviction after it", events, str(events))
    check("cells were genuinely evicted", store.by_state("evicted"), "")
    check("but both marks survive it", len(store.marks) == 2, str(store.marks))
    msgs = store.messages()
    check("the conversation is still valid", not faults(msgs), "; ".join(faults(msgs)))
    check("the person's instruction is still in the context",
          "skip the tests" in str(msgs), "")
    check("and it still sits ahead of the cells that outlived it",
          str(msgs).index("skip the tests") < str(msgs).index("x12"), "")
    check("the answer survived too", "Six done." in str(msgs), "")


class _FakeKernel:
    def probe(self, code, timeout=None):
        return "  x1  int  1"


# ---------------------------------------------------------------------- runner


def multi_turn() -> None:
    print("\nask() twice: one kernel, one namespace, one turn counter")
    provider = StubProvider([
        work("n = 41", "counting"), answer("There are 41."),
        work("print(n + 1)", "adding"), answer("42, as you asked."),
    ])
    runner = Runner("count the files", provider=provider, max_turns=8)
    try:
        first = runner.ask("count the files")
        check("the first response answers", first.answer == "There are 41.",
              repr(first.answer))
        kernel = runner.kernel
        second = runner.ask("now add one")
        check("the second answers too", second.answer == "42, as you asked.",
              repr(second.answer))
        check("on the same kernel", runner.kernel is kernel, "")
        cells = [c.code for c in runner.store.cells]
        check("both responses' cells are in the one store",
              cells == ["n = 41", "print(n + 1)"], str(cells))
        out = runner.store.cells[1].output
        check("and the second really read the first's variable", "42" in out, out[:120])
        check("the turn counter runs across both, not per response",
              runner.turns == 4, str(runner.turns))

        print("\nthe context is the conversation the person actually had")
        marks = [(m.role, m.tag, m.text) for m in runner.store.marks]
        check("answer, prompt, answer -- the first prompt being the task itself",
              marks == [("assistant", "", "There are 41."),
                        ("user", "user-message", "now add one"),
                        ("assistant", "", "42, as you asked.")], str(marks))
        check("the task is not repeated as a mark as well",
              "count the files" not in [m[2] for m in marks], str(marks))
        msgs = runner.store.messages()
        check("and it renders valid", not faults(msgs), "; ".join(faults(msgs)))
        sent = provider.seen[2]
        check("the second response was sampled with the new prompt in the context",
              "now add one" in str(sent), "")
        check("and with the first answer above it",
              "There are 41." in str(sent), "")
    finally:
        runner.close()


def opened_empty() -> None:
    print("\na session opened with no task: the first thing said is a message")
    provider = StubProvider([answer("Nothing changed today.")])
    runner = Runner("", provider=provider, max_turns=4)
    try:
        result = runner.ask("what changed in src today?")
        check("it answers", result.answer == "Nothing changed today.", repr(result.answer))
        marks = [(m.role, m.text) for m in runner.store.marks]
        check("the question is recorded as a prompt, not swallowed by the head",
              marks[0] == ("user", "what changed in src today?"), str(marks))
        check("with the answer after it",
              marks[1] == ("assistant", "Nothing changed today."), str(marks))
    finally:
        runner.close()


def interrupting() -> None:
    print("\nesc: the response ends, the session does not")
    channel = UserChannel()
    provider = StubProvider([work("a = 1"), work("b = 2"), answer("Stopped early.")])
    runner = Runner("keep going", provider=provider, max_turns=8, channel=channel)
    try:
        runner.start()
        check("the channel was given the kernel, so a stop can reach a live cell",
              channel.kernel is runner.kernel, "")
        channel.interrupt("you pressed esc")
        result = runner.ask("keep going")
        check("the response ends at the first boundary",
              result.finish == "interrupted", result.finish)
        check("said in the person's terms, not as a kill",
              "interrupted by you" in (result.error or ""), repr(result.error))
        check("nothing was sampled, because the stop was already standing",
              provider.n == 0, f"{provider.n} requests")

        print("\nand the next prompt carries on where it stopped")
        channel.resume()
        check("resume clears the stop", not channel.stopping)
        again = runner.ask("carry on then")
        check("the session answers the new prompt",
              again.answer == "Stopped early.", repr(again.answer))
        check("with the interrupted-and-resumed prompt in the context",
              "carry on then" in str(runner.store.messages()), "")
    finally:
        runner.close()


def channel_protocol() -> None:
    print("\nthe user channel and a subagent link are the same protocol")
    from xagent import spawn

    channel, link = UserChannel(), spawn._Link("child")
    for name in ("drain", "resume", "attach", "post", "stop_label", "stop_finish",
                 "stop_reason", "stopping"):
        check(f"both have {name}", hasattr(channel, name) and hasattr(link, name),
              f"channel={hasattr(channel, name)} link={hasattr(link, name)}")
    check("but they describe a stop differently, because they are not the same event",
          channel.stop_label != link.stop_label
          and channel.stop_finish != link.stop_finish,
          f"{channel.stop_label!r} vs {link.stop_label!r}")

    print("\nqueued messages come out in the order they were typed")
    channel.post("one")
    channel.post("two")
    check("both waiting", channel.pending == 2, str(channel.pending))
    check("drained in order", channel.drain() == ["one", "two"], "")
    check("and draining empties it", channel.drain() == [])


def delivery() -> None:
    print("\na queued message lands at the turn boundary, not inside the turn")
    channel = UserChannel()
    posted = []

    def midway():
        channel.post("actually, just count them")
        posted.append(True)
        return work("b = 2", "still working")

    provider = StubProvider([work("a = 1", "starting"), midway,
                             work("c = 3", "counting"), answer("Counted.")])
    events = []
    runner = Runner("audit everything", provider=provider, max_turns=8,
                    channel=channel,
                    on_event=lambda kind, data: events.append((kind, data)))
    try:
        result = runner.ask("audit everything")
        check("the response still ends normally", result.answer == "Counted.",
              repr(result.answer))
        marks = [m for m in runner.store.marks if m.role == "user"]
        check("the message arrived as one user mark", len(marks) == 1, str(marks))
        check("tagged as the person, not as a parent",
              marks[0].tag == "user-message", marks[0].tag)
        check("anchored after the cell of the turn it arrived in",
              marks[0].after == 2,
              f"after={marks[0].after} cells={[c.code for c in runner.store.cells]}")
        check("so the turn it interrupted ran its own cell first",
              [c.code for c in runner.store.cells][:2] == ["a = 1", "b = 2"],
              str([c.code for c in runner.store.cells]))
        prompts = [d for kind, d in events if kind == "prompt"]
        check("the reader is shown it, because the answer is a reply to it",
              len(prompts) == 1 and prompts[0]["text"] == "actually, just count them",
              str(prompts))
        check("no cell was fabricated to carry it",
              "inbox" not in str([c.code for c in runner.store.cells]), "")
        out = _rendered(events)
        check("and the printer puts it on screen under its own header",
              "▸ you" in out and "actually, just count them" in out, repr(out[:400]))
    finally:
        runner.close()


def _rendered(events) -> str:
    import contextlib
    import io

    printer = make_printer(colour=False, verbose=False, live_updates=False)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        for kind, data in events:
            printer(kind, data)
    return buffer.getvalue()


def prompt_says_so() -> None:
    print("\nthe model is told a person can reach it, or it will not expect one")
    check("there is a section about them",
          "# The person you work for" in SYSTEM, "")
    check("naming the tag their message arrives under",
          "<user-message>" in SYSTEM, "")
    check("saying the newest message wins",
          "newest message wins" in SYSTEM, "")
    check("and that an interrupted cell was stopped on purpose",
          "stopped by them" in SYSTEM, "")
    check("it is framed as a conversation rather than an errand",
          "a conversation, not an errand" in SYSTEM, "")


def main() -> int:
    rendering()
    against_compaction()
    channel_protocol()
    prompt_says_so()
    multi_turn()
    opened_empty()
    interrupting()
    delivery()
    print(f"\n{'─' * 60}\n{len(PASS)} passed, {len(FAIL)} failed")
    for name in FAIL:
        print(f"  \033[31mFAILED:\033[0m {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
