"""The finishing contract, driven end to end against a scripted provider.

The top-level agent finishes by writing its answer as ordinary text and calling the
`done` tool in the same turn. Every defect this suite pins down is a way that
pairing came apart in a real run -- prose stranded a turn away from the call, a
finish with nothing written, a finish reported as a stray -- and each one showed up
as the same paragraph reaching the user twice, or not at all.

A subagent finishes the other way and keeps the in-kernel `done(value)`: its result
is an object in its own kernel, often one it never saw whole, so it hands back a
name rather than transcribing a value into a tool argument. The last block here
pins that asymmetry.

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
from xagent.provider import Turn, Usage
from xagent.runner import Runner, strip_trailing_done

PASS, FAIL = [], []

ANSWER = "**50 project files, 9,305 lines.** Python dominates at 4,952 of them."


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32m✓\033[0m" if cond else "\033[31m✗\033[0m"
    print(f"  {mark} {name}" + (f"  — {detail}" if detail and not cond else ""))


class StubProvider:
    """Plays a fixed sequence of turns; records the `tools` flag of each request."""

    def __init__(self, turns):
        self.turns, self.n, self.asked = list(turns), 0, []
        self.backend = config.BACKENDS["codiv"]
        self.model, self.thinking, self.sampling = "stub", None, "thinking"
        self.usage, self.on_delta, self.truncated_turns = Usage(), None, 0
        # Set by Runner from the role. Recorded so the suite can prove a subagent
        # is never offered the tool in the first place.
        self.offer_done = True

    def sample(self, system, messages, *, tools=True):
        self.asked.append(tools)
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


def drive_recorded(turns, is_subagent: bool = False, max_turns: int = 8):
    """Run the script and hand back the provider and every event it emitted.

    The provider is the only witness to how many requests a finish cost, and the
    warnings say which of the turn's tool blocks the harness acted on.
    """
    provider = StubProvider(turns)
    events = []
    result = Runner("count the files", provider=provider, max_turns=max_turns,
                    is_subagent=is_subagent,
                    on_event=lambda kind, data: events.append((kind, data))).run()
    return result, provider, events


def drive(turns, is_subagent: bool = False, max_turns: int = 8):
    result, provider, _ = drive_recorded(turns, is_subagent, max_turns)
    # A request made with the tools withheld is the fallback asking for an answer.
    return result, provider.asked.count(False)


def notes(events) -> str:
    return "\n".join(d["message"] for kind, d in events if kind == "note")


def warnings(events) -> str:
    return "\n".join(d["message"] for kind, d in events if kind == "warning")


def work(text: str = "") -> Turn:
    return Turn(text=text, code="x = 1", tool_use_id="tu_w", tool_name="python")


def finish(text: str = "", **kw) -> Turn:
    """The finish as it arrives from the provider: a `done` block, no arguments."""
    return Turn(text=text, done=True, done_id="tu_d", **kw)


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
    print("the contract: prose beside the call, and the call ends the run")
    result, provider, events = drive_recorded([work(), finish(ANSWER)])
    check("it finishes on the prose written beside it", result.answer == ANSWER,
          f"finish={result.finish!r} answer={result.answer[:80]!r}")
    check("without paying for the answer a second time",
          provider.asked.count(False) == 0, str(provider.asked))
    check("recorded as done", result.finish == "done", result.finish)
    check("the finish carries no value of its own", result.value is None,
          repr(result.value))
    check("and nothing is corrected, because nothing was wrong",
          not warnings(events), warnings(events))

    print("\nthe old spelling is a redirect now, not a finish")
    result, provider, events = drive_recorded(
        [work(), Turn(text=ANSWER, code="done()", tool_use_id="tu_c",
                      tool_name="python"), finish("Real answer.")])
    check("done() in a cell does not end a top-level run", provider.n == 3,
          f"{provider.n} requests")
    check("the kernel says so where the model will read it",
          "`done` tool" in cells(events)[1]["output"], cells(events)[1]["output"][:160])
    check("so the run finishes on the turn that really finished",
          result.answer == "Real answer.", repr(result.answer))

    print("\nprose the harness refused to run is still the answer")
    stray = Turn(text=ANSWER, tool_use_id="tu_s", tool_name="bash")
    silent = finish()
    result, extra = drive([work(), stray, silent])
    check("a rejected turn's answer survives to the finish after it",
          result.answer == ANSWER, f"answer={result.answer[:80]!r}")
    check("so the fallback never fires", extra == 0, f"{extra} extra")

    print("\nbut it goes stale the moment work resumes")
    result, extra = drive([work(), stray, work(), silent, Turn(text="Fresh answer.")])
    check("a finish after more work asks again rather than reusing it",
          result.answer == "Fresh answer." and extra == 1,
          f"answer={result.answer[:60]!r} extra={extra}")

    print("\nfinishing with nothing written")
    result, provider, events = drive_recorded([work(), silent, Turn(text="Here it is.")])
    check("the fallback asks, with the tools withheld",
          result.answer == "Here it is." and provider.asked.count(False) == 1,
          f"answer={result.answer[:60]!r} asked={provider.asked}")
    check("and the finish that ran nothing shows the reader no cell",
          len(cells(events)) == 1, f"{len(cells(events))} cell events")
    # The recovery cell is normally appended after real ones. On turn 1 it is the
    # only thing in the store, which is the case that would break if the ask
    # needed a cell that had actually run.
    result, provider, events = drive_recorded([silent, Turn(text="Recovered.")])
    check("a run that finishes silently on its first turn still answers",
          result.answer == "Recovered." and provider.asked.count(False) == 1,
          f"answer={result.answer!r} asked={provider.asked}")
    check("and shows the reader no cell for it either", not cells(events),
          f"{len(cells(events))} cell events")

    print("\nthe transcript credited the answer to a cell output the fallback had")
    print("already overwritten: `← output │ '[done]'` on screen, a longer ask sent")
    run_then_finish = Turn(text="", code="x = 2", tool_use_id="tu_rf",
                           tool_name="python", done=True, done_id="tu_d2")
    result, provider, events = drive_recorded(
        [work(), run_then_finish, Turn(text="Answered.")])
    starts = [d for kind, d in events if kind == "answer_start"]
    check("the answer header carries the ask the model was actually given",
          len(starts) == 1 and "no answer written" in starts[0].get("ask", ""),
          repr(starts))
    shown = cells(events)[-1]["output"]
    check("which is not what the cell event went out with — that is the defect",
          starts and starts[0].get("ask") != shown, repr(shown))
    check("and no second cell event reprints the cell to say so",
          len(cells(events)) == 2, f"{len(cells(events))} cell events")
    _, _, clean = drive_recorded([work(), Turn(text=ANSWER)])
    check("a run that answered on its own substitutes nothing, so discloses nothing",
          not [1 for kind, _ in clean if kind == "answer_start"])

    print("\nand the printer has to actually put it on screen")
    # The ask the run above really emitted, so the two halves cannot drift apart.
    ask = starts[0]["ask"] if starts else ""
    out = rendered([("cell", {"n": 2, "output": shown, "ok": True}),
                    ("answer_start", {"ask": ask}),
                    ("delta", {"part": "text", "text": "Answered."}),
                    ("answer", {"text": "Answered."})])
    check("the ask is rendered under the cell it replaced", ask and ask in out,
          repr(out))
    check("labelled as a rewrite rather than passed off as the cell's own output",
          "output rewritten for the answer turn" in out, repr(out))
    # The printer prefixes every line, and a cell output is now several lines --
    # code, then the harness status line -- so the raw string is not a substring
    # of what came out. Count its lines instead; the defect this guards against
    # (the cell reprinted under the rewrite) would double every one of them.
    check("the cell output is not printed a second time",
          all(out.count(ln) == 1 for ln in shown.splitlines() if ln.strip()),
          repr(out))
    check("the answer still streams under its header, and only once",
          "─── answer" in out and out.count("Answered.") == 1, repr(out))
    out = rendered([("answer_start", {}), ("answer", {"text": "Answered."})])
    check("an answer that never streamed is still printed by the turn summary",
          out.count("Answered.") == 1, repr(out))
    check("and a header with no ask prints no rewrite line",
          "rewritten" not in out, repr(out))

    print("\na cell that names its own timeout runs under it, and every cell")
    print("        output reports the clock, the cost and the context")
    slow = Turn(text="", code="import time; time.sleep(3)", tool_use_id="tu_s",
                tool_name="python", timeout=1)
    _, _, events = drive_recorded([slow, finish(text=ANSWER)])
    out = cells(events)[0]["output"]
    check("the model's timeout is what the cell ran under", "timed out after 1s" in out,
          repr(out))
    check("and the notice says how to ask for longer", "`timeout`" in out, repr(out))
    check("the reader is told the cell was given a longer leash",
          "cell timeout 1s" in notes(events), notes(events))
    check("an interrupted cell is still reported as not ok",
          cells(events)[0]["ok"] is False)
    _, _, events = drive_recorded([work(), finish(text=ANSWER)])
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
    result, provider, events = drive_recorded([work()] * 40, max_turns=2)
    extensions = [m for m in notes(events).splitlines() if "turn limit" in m]
    check("the run does not stop at the limit", result.turns > 2, str(result.turns))
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
    check("a run that finishes early is never extended",
          not [m for m in notes(drive_recorded([work(), finish(text=ANSWER)])[2]
                                ).splitlines() if "turn limit" in m])

    print("\na request that dies mid-run used to take the whole run with it:")
    print("        sixty turns of work reported as `0 turns` and nothing kept")
    partial = "Found the bug: the parser binds `^` too tightly."
    result, _, events = drive_recorded(
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
    sub, _, _ = drive_recorded([work(), RuntimeError("closed")], is_subagent=True)
    check("a subagent is handed no prose even here, because nobody reads it",
          sub.answer == "" and sub.error, repr(sub.answer))

    print("\nanswering without calling done at all")
    result, extra = drive([work(), Turn(text=ANSWER)])
    check("prose with no tool call finishes the run", result.answer == ANSWER)
    check("recorded as answered, not as an error", result.finish == "answered",
          result.finish)
    check("and asks for nothing more", extra == 0, f"{extra} extra")

    print("\nthe call is punctuation, so it is not part of what the user reads")
    result, extra = drive([work(), finish(ANSWER + "\n\ndone()")])
    check("a done() the model narrated beside the call is dropped",
          result.answer == ANSWER, repr(result.answer[-40:]))
    for src, want in (("A.\n\n`done()`", "A."),
                      ("A.\n```python\ndone()\n```", "A."),
                      ("Call done() when finished.", "Call done() when finished."),
                      ("A.\ndone()\ndone()", "A.")):
        check(f"stripping {src!r} leaves {want!r}", strip_trailing_done(src) == want,
              repr(strip_trailing_done(src)))

    print("\nthe same text finished differently depending on which exit it took:")
    print("a narrated done() survived into the answer when no call went with it")
    result, extra = drive([work(), Turn(text=ANSWER + "\n\ndone()")])
    check("a text-only finish is cleaned like every other one",
          result.answer == ANSWER, repr(result.answer[-40:]))
    check("still recorded as answered", result.finish == "answered", result.finish)
    check("and still costs nothing extra", extra == 0, f"{extra} extra")
    stray2 = Turn(text=ANSWER, tool_use_id="tu_s2", tool_name="bash")
    result, extra = drive([work(), stray2, Turn(text="")])
    check("prose refused a turn earlier reaches a text-only finish too",
          result.answer == ANSWER, f"finish={result.finish!r} answer={result.answer!r}")

    print("\na finish beside a real python call means both halves")
    both = Turn(text=ANSWER, code="x = 2", tool_use_id="tu_b", tool_name="python",
                done=True, done_id="tu_d3")
    result, provider, events = drive_recorded([work(), both, Turn(text="Second answer.")])
    check("the turn that ran the cell also ends the run", result.finish == "done",
          result.finish)
    check("on the prose written beside it", result.answer == ANSWER,
          repr(result.answer[:80]))
    check("nothing is sampled after it", provider.n == 2, f"{provider.n} requests")
    check("and nothing extra is asked for", provider.asked.count(False) == 0)
    check("the finish carries no argument, so the value is None", result.value is None,
          repr(result.value))
    check("the cell beside it really ran", len(cells(events)) == 2,
          f"{len(cells(events))} cell events")
    check("the finish is never reported to the model as ignored",
          "ignored extra tool call" not in warnings(events), warnings(events))
    result, provider, events = drive_recorded(
        [work(), Turn(text=ANSWER, code="x = 2", tool_use_id="tu_b2",
                      tool_name="python", ignored_tools=["sh"], done=True)])
    check("a genuinely unknown tool alongside it is still reported",
          "ignored extra tool call(s): sh" in warnings(events), warnings(events))
    check("and the run still finishes on that turn", result.finish == "done",
          result.finish)
    ran = Turn(text=ANSWER, code="done({'n': 7})", tool_use_id="tu_b3",
               tool_name="python", done=True)
    result, extra = drive([work(), ran])
    check("a done() the cell ran cannot smuggle a value into a top-level finish",
          result.value is None, repr(result.value))

    print("\na finish with no cell to run, in the company it sometimes keeps")
    result, provider, events = drive_recorded(
        [work(), Turn(text=ANSWER, tool_use_id="tu_z", tool_name="bash", done=True)])
    check("a stray tool beside the finish does not delay it",
          result.finish == "done" and provider.n == 2, f"{result.finish} n={provider.n}")
    check("the stray is reported all the same",
          "ignored extra tool call(s): bash" in warnings(events), warnings(events))
    result, provider, _ = drive_recorded(
        [work(), Turn(text=ANSWER, code="   ", tool_use_id="tu_y",
                      tool_name="python", done=True)])
    check("an empty cell beside the finish is a finish, not a retry",
          result.answer == ANSWER and provider.n == 2,
          f"{result.finish} n={provider.n}")

    print("\nhitting the ceiling threw away prose the model had already written")
    # Driven to the hard ceiling, not the block limit: the block limit compacts
    # and carries on now, so the salvage path is what happens at the very end.
    result, extra = drive([work("Half the files counted."),
                           work("50 files, 9,305 lines so far.\n\ndone()")] * 4,
                          max_turns=2)
    check("the last prose is carried out as the answer",
          result.answer == "50 files, 9,305 lines so far.", repr(result.answer))
    check("the run is still reported as failed", result.error is not None
          and result.finish == "max_turns", f"{result.finish} {result.error!r}")
    check("and the error says the text above is only partial",
          "partial answer" in (result.error or ""), result.error)
    check("salvaging it costs no request", extra == 0, f"{extra} extra")
    result, extra = drive([work(), Turn(text=ANSWER, tool_use_id="tu_s3",
                                        tool_name="bash")] * 4, max_turns=2)
    check("prose from a refused turn is salvaged the same way",
          result.answer == ANSWER, repr(result.answer[:60]))

    print("\na subagent answers a program, so it finishes by value instead")
    value = Turn(text="", code="done({'n': 12})", tool_use_id="tu_v", tool_name="python")
    result, provider, events = drive_recorded([work(), value], is_subagent=True)
    check("its done() value crosses back", result.value == {"n": 12}, repr(result.value))
    check("it is handed no prose to write", result.answer == "", repr(result.answer))
    check("and never spends a turn on one", provider.asked.count(False) == 0)
    check("it is never offered the tool in the first place",
          provider.offer_done is False, repr(provider.offer_done))
    # What a hallucinated finish really looks like on this side: the provider
    # offers a subagent no `done` tool, so the block arrives as a stray.
    hallucinated = Turn(text="", tool_use_id="tu_h", tool_name="done")
    result, provider, events = drive_recorded([hallucinated, value], is_subagent=True)
    check("a `done` block it hallucinated is corrected, not guessed at",
          result.value == {"n": 12} and provider.n == 2, repr(result.value))
    check("and it is told to send the value as Python instead",
          "unknown tool 'done'" in warnings(events), warnings(events))
    # The runner guard, on a turn shaped the way a real one could be: even if the
    # flag reached a subagent, its result is the argument and must not be guessed.
    guarded = Turn(text="", code="y = 1", tool_use_id="tu_g", tool_name="python",
                   done=True, done_id="tu_g2")
    result, provider, _ = drive_recorded([guarded, value], is_subagent=True)
    check("and the flag alone cannot finish a subagent either",
          result.value == {"n": 12} and provider.n == 2, repr(result.value))
    both_sub = Turn(text="", code="x = 2", tool_use_id="tu_sb", tool_name="python",
                    ignored_tools=["done"])
    result, provider, events = drive_recorded([both_sub, value], is_subagent=True)
    check("a `done` block beside its cell does not finish it either",
          result.value == {"n": 12} and provider.n == 2, repr(result.value))
    check("it is told the block was ignored, so it sends the value as Python",
          "ignored extra tool call(s): done" in warnings(events), warnings(events))
    check("and the correction names the one tool it has",
          "only tool is `python`" in str(events), "")
    result, extra = drive([work("Partial."), work("More.")] * 4, is_subagent=True,
                          max_turns=2)
    check("and the turn limit hands it no prose either",
          result.finish == "max_turns" and result.answer == "",
          f"{result.finish} {result.answer!r}")

    print(f"\n{'─' * 60}\n{len(PASS)} passed, {len(FAIL)} failed")
    for name in FAIL:
        print(f"  \033[31mFAILED:\033[0m {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
