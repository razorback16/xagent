"""The answering contract, driven end to end against a scripted provider.

The top-level agent ends a response by writing its answer as ordinary text and making
no tool call in that turn. There is no finish tool at any depth. Every defect this
suite pins down is a way that reading came apart in a real run -- prose stranded a
turn away from the turn that ended, a response that ended having written nothing, a
turn ceiling that threw away the paragraph it had -- and each one showed up as the
same paragraph reaching the person twice, or not at all.

A subagent finishes the other way and keeps the in-kernel `done(value)`: its result
is an object in its own kernel, often one it never saw whole, so it hands back a
name rather than transcribing a value into a tool argument. The last block here
pins that asymmetry.

The answer is also recorded in the context as an assistant message, because a
conversation whose replies are missing reads as a list of unanswered questions. That
is checked here too: it is the thing that makes the next prompt make sense.

No network: the provider is a stub that plays a fixed sequence of turns and records
what it was asked for, so the paths that depend on the model doing something unusual
are testable without waiting for it to happen. A real kernel does run, which is what
makes the in-kernel half of the contract testable too.

Run with:  uv run python tests/test_finishing.py
"""

from __future__ import annotations

import contextlib
import io
import re
import sys

from xagent import config
from xagent.cli import make_printer
from xagent.prompts import SYSTEM
from xagent.provider import Call, Turn, Usage
from xagent.runner import Runner, strip_trailing_done

PASS, FAIL = [], []

ANSWER = "**50 project files, 9,305 lines.** Python dominates at 4,952 of them."


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32m✓\033[0m" if cond else "\033[31m✗\033[0m"
    print(f"  {mark} {name}" + (f"  — {detail}" if detail and not cond else ""))


class StubProvider:
    """Plays a fixed sequence of turns; counts the requests each response cost."""

    def __init__(self, turns):
        self.turns, self.n = list(turns), 0
        self.backend = config.BACKENDS["codiv"]
        self.model, self.thinking, self.sampling = "stub", None, "thinking"
        self.usage, self.on_delta, self.truncated_turns = Usage(), None, 0
        # Set by Runner from the role. Recorded so the suite can prove the two
        # renderings of the python tool go to the two roles they were written for.
        self.is_subagent = False

    def sample(self, system, messages):
        self.seen = messages
        turn = self.turns[self.n] if self.n < len(self.turns) else Turn()
        self.n += 1
        # A scripted exception stands for a request that died on the wire after
        # the run had already done real work.
        if isinstance(turn, BaseException):
            raise turn
        return turn

    def system_blocks(self, system):
        return [{"type": "text", "text": system}]

    def complete(self, prompt, model=None, max_tokens=4096):
        return "summary"


def build(turns, is_subagent: bool = False, max_turns: int = 8) -> Runner:
    return Runner("count the files", provider=StubProvider(turns),
                  max_turns=max_turns, is_subagent=is_subagent)


def drive_recorded(turns, is_subagent: bool = False, max_turns: int = 8):
    """Run the script and hand back the runner and every event it emitted."""
    events = []
    runner = build(turns, is_subagent, max_turns)
    runner.on_event = lambda kind, data: events.append((kind, data))
    result = runner.run()
    return result, runner, events


def drive(turns, is_subagent: bool = False, max_turns: int = 8):
    result, runner, _ = drive_recorded(turns, is_subagent, max_turns)
    return result, runner.provider.n


def notes(events) -> str:
    return "\n".join(d["message"] for kind, d in events if kind == "note")


def warnings(events) -> str:
    return "\n".join(d["message"] for kind, d in events if kind == "warning")


def work(text: str = "") -> Turn:
    return Turn(text=text, code="x = 1", tool_use_id="tu_w", tool_name="python")


def answer(text: str = "", **kw) -> Turn:
    """A response ending as it arrives from the provider: prose, and no tool call."""
    return Turn(text=text, **kw)


def cells(events) -> list:
    return [d for kind, d in events if kind == "cell"]


def rendered(events) -> str:
    """Drive the real CLI printer over synthesised events; return what it wrote.

    The runner can emit an honest record and still have nobody show it, so the
    printer is checked here rather than assumed from the events alone.
    """
    printer = make_printer(colour=False, verbose=False, live_updates=False)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        for kind, data in events:
            printer(kind, data)
    return buffer.getvalue()


def main() -> int:
    print("the contract: a turn with no call is the answer")
    result, runner, events = drive_recorded([work(), answer(ANSWER)])
    check("it ends on the prose of the turn that called nothing",
          result.answer == ANSWER,
          f"finish={result.finish!r} answer={result.answer[:80]!r}")
    check("without paying for the answer a second time", runner.provider.n == 2,
          f"{runner.provider.n} requests")
    check("recorded as answered", result.finish == "answered", result.finish)
    check("and nothing is corrected, because nothing was wrong",
          not warnings(events), warnings(events))

    print("\nthe answer is recorded in the context, so a reply has something to follow")
    marks = runner.store.marks
    check("one mark, holding the answer", len(marks) == 1 and marks[0].text == ANSWER,
          repr(marks))
    check("as an assistant message, which is whose words they are",
          marks[0].role == "assistant", marks[0].role)
    check("and untagged, because an assistant message needs no label",
          marks[0].tag == "", repr(marks[0].tag))
    msgs = runner.store.messages()
    check("it renders as the last assistant message of the conversation",
          msgs[-1]["role"] == "assistant"
          and msgs[-1]["content"][-1]["text"] == ANSWER, repr(msgs[-1])[:200])

    print("\nthere is no finish tool, at either depth")
    check("the top level is told the only tool is python",
          "exactly one tool, `python`" in SYSTEM, "")
    check("and told how a response ends instead",
          "no tool call" in SYSTEM and "# Answering" in SYSTEM, "")
    check("a `done` tool is never described to it", "`done` tool" not in SYSTEM, "")

    print("\nthe old spelling is a redirect, not a finish")
    result, runner, events = drive_recorded(
        [work(), Turn(text=ANSWER, code="done()", tool_use_id="tu_c",
                      tool_name="python"), answer("Real answer.")])
    check("done() in a cell does not end a top-level response", runner.provider.n == 3,
          f"{runner.provider.n} requests")
    check("the kernel says so where the model will read it",
          "does not end your response" in cells(events)[1]["output"],
          cells(events)[1]["output"][:160])
    check("so the response ends on the turn that really ended it",
          result.answer == "Real answer.", repr(result.answer))
    hallucinated = Turn(text="", tool_use_id="tu_hd", tool_name="done")
    result, runner, events = drive_recorded([hallucinated, answer(ANSWER)])
    check("a `done` tool call it invented is corrected rather than honoured",
          result.answer == ANSWER and runner.provider.n == 2, repr(result.answer))
    check("and it is told the one tool it has, where it will read it",
          "only tool is `python`" in runner.store.cells[0].output,
          runner.store.cells[0].output[:160])

    print("\nprose the harness refused to run is still the answer")
    stray = Turn(text=ANSWER, tool_use_id="tu_s", tool_name="bash")
    result, n = drive([work(), stray, answer()])
    check("a rejected turn's answer survives to the turn that ends after it",
          result.answer == ANSWER, f"answer={result.answer[:80]!r}")
    check("and costs no request of its own", n == 3, f"{n} requests")

    print("\nbut it goes stale the moment work resumes")
    result, n = drive([work(), stray, work(), answer("Fresh answer.")])
    check("a response ending after more work uses the newer prose",
          result.answer == "Fresh answer.", repr(result.answer))

    print("\nending having written nothing at all")
    result, runner, events = drive_recorded([work(), answer("")])
    check("reported as an error rather than as an empty answer",
          result.finish == "error" and not result.ok, f"{result.finish} {result.error!r}")
    check("naming what the model produced instead",
          "neither code nor text" in (result.error or ""), repr(result.error))
    check("nothing is asked for a second time", runner.provider.n == 2,
          f"{runner.provider.n} requests")
    check("and no answer mark is recorded, because there was no answer",
          not runner.store.marks, repr(runner.store.marks))

    print("\na cell that names its own timeout runs under it, and every cell")
    print("        output reports the clock, the cost and the context")
    slow = Turn(text="", code="import time; time.sleep(3)", tool_use_id="tu_s",
                tool_name="python", timeout=1)
    _, _, events = drive_recorded([slow, answer(ANSWER)])
    out = cells(events)[0]["output"]
    check("the model's timeout is what the cell ran under", "timed out after 1s" in out,
          repr(out))
    check("and the notice says how to ask for longer", "`timeout`" in out, repr(out))
    check("the reader is told the cell was given a longer leash",
          "cell timeout 1s" in notes(events), notes(events))
    check("an interrupted cell is still reported as not ok",
          cells(events)[0]["ok"] is False)
    _, _, events = drive_recorded([work(), answer(ANSWER)])
    status = cells(events)[0]["output"].splitlines()[-1]
    check("every cell output ends with a status line",
          status.startswith("[") and status.endswith("]"), repr(status))
    for field in ("cell ", "run ", "ctx "):
        check(f"it carries {field.strip()}", field in status, repr(status))
    check("the context figure is quoted against the budget",
          re.search(r"ctx [\d,]+/[\d,]+ \(\d+%", status) is not None, repr(status))
    check("a turn with no timeout of its own says nothing about one",
          "cell timeout" not in notes(events), notes(events))
    check("the model is told the status line exists, so it need not call ctx()",
          "never need to call `ctx()`" in SYSTEM, "")

    print("\nthe turn limit is a compaction trigger, not a wall: a long job")
    print("        that ran out of turns used to lose every cell it had built")
    result, runner, events = drive_recorded([work()] * 40, max_turns=2)
    extensions = [m for m in notes(events).splitlines() if "turn limit" in m]
    check("the response does not stop at the limit", result.turns > 2, str(result.turns))
    check("it is extended a block at a time", len(extensions) == 3,
          f"{len(extensions)}: {extensions}")
    check("each extension says where it now stops",
          all("continuing to" in m and "of 8" in m for m in extensions),
          repr(extensions))
    check("and compaction is what it does at the boundary",
          all("compact" in m for m in extensions), repr(extensions))
    check("the hard ceiling is what finally ends it", result.turns == 8,
          str(result.turns))
    check("reported as the ceiling it is, not as the block size",
          "hard ceiling of 8 turns" in (result.error or ""), repr(result.error))
    check("recorded as max_turns, so a caller can tell it apart from a crash",
          result.finish == "max_turns", result.finish)
    check("a response that ends early is never extended",
          not [m for m in notes(drive_recorded([work(), answer(ANSWER)])[2]
                                ).splitlines() if "turn limit" in m])

    print("\na request that dies mid-run used to take the whole run with it:")
    print("        sixty turns of work reported as `0 turns` and nothing kept")
    partial = "Found the bug: the parser binds `^` too tightly."
    result, runner, events = drive_recorded(
        [work(), work(partial), RuntimeError("peer closed connection")])
    check("the failure is reported", result.error and not result.ok, repr(result.error))
    check("named by its type, so the wire error is recognisable",
          "RuntimeError" in (result.error or ""), repr(result.error))
    check("the turns it did complete are counted, not zeroed",
          result.turns == 2, str(result.turns))
    check("and the cells it ran are still in the store", len(cells(events)) == 2,
          str(len(cells(events))))
    check("the last prose it wrote survives as a partial answer",
          result.answer == partial, repr(result.answer))
    check("recorded in the context too, so a reply still has something to follow",
          [m.text for m in runner.store.marks] == [partial],
          repr(runner.store.marks))
    sub, _, _ = drive_recorded([work(), RuntimeError("closed")], is_subagent=True)
    check("a subagent is handed no prose even here, because nobody reads it",
          sub.answer == "" and sub.error, repr(sub.answer))

    print("\na narrated done() is punctuation, not part of what the person reads")
    result, n = drive([work(), answer(ANSWER + "\n\ndone()")])
    check("a done() the model typed into its answer is dropped",
          result.answer == ANSWER, repr(result.answer[-40:]))
    for src, want in (("A.\n\n`done()`", "A."),
                      ("A.\n```python\ndone()\n```", "A."),
                      ("Call done() when finished.", "Call done() when finished."),
                      ("A.\ndone()\ndone()", "A.")):
        check(f"stripping {src!r} leaves {want!r}", strip_trailing_done(src) == want,
              repr(strip_trailing_done(src)))
    stray2 = Turn(text=ANSWER, tool_use_id="tu_s2", tool_name="bash")
    result, n = drive([work(), stray2, answer("")])
    check("prose refused a turn earlier reaches the ending turn too",
          result.answer == ANSWER, f"finish={result.finish!r} answer={result.answer!r}")
    check("and is recorded as answered rather than as the error it nearly was",
          result.finish == "answered", result.finish)

    print("\na lone stray tool is corrected, not read as the end of the response")
    result, runner, events = drive_recorded(
        [work(), Turn(text=ANSWER, tool_use_id="tu_z", tool_name="bash"),
         answer("Corrected and answered.")])
    check("the model is told the tool does not exist",
          "unknown tool 'bash'" in warnings(events), warnings(events))
    check("and named the one it has, in the cell output it reads",
          "only tool is `python`" in runner.store.cells[1].output,
          runner.store.cells[1].output[:160])
    check("so the response ends on the turn after the correction",
          result.answer == "Corrected and answered.", repr(result.answer))

    print("\nhitting the ceiling threw away prose the model had already written")
    # Driven to the hard ceiling, not the block limit: the block limit compacts
    # and carries on now, so the salvage path is what happens at the very end.
    result, n = drive([work("Half the files counted."),
                       work("50 files, 9,305 lines so far.\n\ndone()")] * 4,
                      max_turns=2)
    check("the last prose is carried out as the answer",
          result.answer == "50 files, 9,305 lines so far.", repr(result.answer))
    check("the run is still reported as failed", result.error is not None
          and result.finish == "max_turns", f"{result.finish} {result.error!r}")
    check("and the error says the text above is only partial",
          "partial answer" in (result.error or ""), result.error)
    result, n = drive([work(), Turn(text=ANSWER, tool_use_id="tu_s3",
                                    tool_name="bash")] * 4, max_turns=2)
    check("prose from a refused turn is salvaged the same way",
          result.answer == ANSWER, repr(result.answer[:60]))

    print("\na subagent answers a program, so it finishes by value instead")
    value = Turn(text="", code="done({'n': 12})", tool_use_id="tu_v", tool_name="python")
    result, runner, events = drive_recorded([work(), value], is_subagent=True)
    check("its done() value crosses back", result.value == {"n": 12}, repr(result.value))
    check("it is handed no prose to write", result.answer == "", repr(result.answer))
    check("and records no answer mark, because nobody would read one",
          not runner.store.marks, repr(runner.store.marks))
    check("it gets the subagent rendering of the python tool",
          runner.provider.is_subagent is True, repr(runner.provider.is_subagent))
    hallucinated = Turn(text="", tool_use_id="tu_h", tool_name="done")
    result, runner, events = drive_recorded([hallucinated, value], is_subagent=True)
    check("a `done` block it hallucinated is corrected, not guessed at",
          result.value == {"n": 12} and runner.provider.n == 2, repr(result.value))
    check("and it is told to send the value as Python instead",
          "unknown tool 'done'" in warnings(events), warnings(events))
    both_sub = Turn(text="", code="x = 2", tool_use_id="tu_sb", tool_name="python",
                    ignored_tools=["done"])
    result, runner, events = drive_recorded([both_sub, value], is_subagent=True)
    check("a `done` block beside its cell does not finish it either",
          result.value == {"n": 12} and runner.provider.n == 2, repr(result.value))
    check("it is told the block was ignored, so it sends the value as Python",
          "ignored extra tool call(s): done" in warnings(events), warnings(events))
    check("and the correction names the one tool it has",
          "only tool is `python`" in str(events), "")
    result, n = drive([work("Partial."), work("More.")] * 4, is_subagent=True,
                      max_turns=2)
    check("and the turn limit hands it no prose either",
          result.finish == "max_turns" and result.answer == "",
          f"{result.finish} {result.answer!r}")

    print("\nparallel calls: one turn, several cells, run in order in the one kernel")
    batch = Turn(text="", calls=[Call(code="p = 41", tool_use_id="tu_p1"),
                                 Call(code="print(p + 1)", tool_use_id="tu_p2")])
    result, runner, events = drive_recorded([batch, answer(ANSWER)])
    check("every call ran, each with an output of its own",
          len(cells(events)) == 2, f"{len(cells(events))} cell events")
    check("in order, in the same namespace",
          "42" in cells(events)[1]["output"], cells(events)[1]["output"][:120])
    check("and nothing is reported as ignored, because nothing was",
          "ignored" not in warnings(events), warnings(events))
    check("the reader is told the turn batched rather than looping",
          "2 calls in this turn" in notes(events), notes(events))
    check("the response still ends on the prose of the turn after it",
          result.answer == ANSWER, repr(result.answer[:80]))

    stray = Turn(text="", calls=[Call(code="c = 1", tool_use_id="tu_s1"),
                                 Call(code="d = 2", tool_use_id="tu_s2")],
                 ignored_tools=["bash"])
    _, _, events = drive_recorded([stray, answer(ANSWER)])
    check("a stray tool beside a batch is reported once, on the last cell",
          warnings(events).count("ignored extra tool call(s): bash") == 1,
          warnings(events))
    check("and not against a call that ran perfectly well",
          "ignored" not in cells(events)[0]["output"], cells(events)[0]["output"][:120])

    half = Turn(text="", calls=[Call(code="e = 1", tool_use_id="tu_h1"),
                                Call(code="   ", tool_use_id="tu_h2")])
    _, _, events = drive_recorded([half, answer(ANSWER)])
    check("an empty call beside a real one costs the real one nothing",
          len(cells(events)) == 1, f"{len(cells(events))} cell events")
    check("and the model is told one of its calls carried no code",
          "arrived with no code" in cells(events)[0]["output"],
          cells(events)[0]["output"][:160])

    stop = Turn(text="", calls=[Call(code="done({'n': 7})", tool_use_id="tu_d1"),
                                Call(code="never = True", tool_use_id="tu_d2")])
    result, _, events = drive_recorded([stop], is_subagent=True)
    check("done() inside a batch ends the run at that call",
          result.value == {"n": 7} and len(cells(events)) == 1,
          f"value={result.value!r} cells={len(cells(events))}")
    check("and the reader is told what the turn had queued behind it",
          "were not run" in notes(events), notes(events))

    out = rendered([("turn", {"n": 1, "text": "", "thinking": "",
                              "code": "p = 41", "codes": ["p = 41", "print(p + 1)"]})])
    check("a turn that did not stream prints every call it made, not just the first",
          out.count("▸ python") == 2, repr(out))

    print(f"\n{'─' * 60}\n{len(PASS)} passed, {len(FAIL)} failed")
    for name in FAIL:
        print(f"  \033[31mFAILED:\033[0m {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
