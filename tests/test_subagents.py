"""Supervising a subagent: watching it, talking to it, and stopping it.

A subagent has no wall clock. It runs for as long as its job takes, so a parent
that could only wait was a parent that found out how it went and nothing else.
The supervision surface -- `poll()`, `send()`, `kill()` -- is what replaces the
timeout, and this suite is the contract for all three.

The failure that motivated it is the one at the top of the file. A parent ran
`gather(...)` in a cell; the 180s *cell* timeout fired and raised KeyboardInterrupt
into the waiting thread; `Handle.result()` caught it as though it were an ordinary
failure, turned it into an AgentError, and handed it back to the loop -- which went
straight on to block on the next handle. The harness had already reported the cell
as timed out, but the kernel was still sitting in that wait, and every later cell
queued behind it forever. So an interrupt must come *out* of `result()` and out of
`gather()`, untouched, and gather must not go on waiting for the rest. Half this
suite is about that single BaseException/Exception distinction.

The other half is the supervision surface itself: that no default deadline crept
back in, that a message really becomes a cell the child ran, that a kill keeps the
work the child had already done, and that a parent reading `poll()` is reading
something small enough to print into a context window.

No network and no credentials: providers are scripted stubs, exactly as
tests/test_finishing does it. Real kernels do run -- that is what makes the
injected-cell half a real test rather than an assertion about a string.

Run with:  uv run python tests/test_subagents.py
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

# No warm pool: this suite starts a handful of kernels of its own and must leave
# nothing behind, and a background warmer would both cost time and show up in the
# thread census at the end.
os.environ["XAGENT_KERNEL_POOL"] = "0"

import xagent.runner as runner_module  # noqa: E402
from xagent import config, runtime, spawn  # noqa: E402
from xagent.kernel import Kernel  # noqa: E402
from xagent.prompts import SUBAGENT_CODA  # noqa: E402
from xagent.provider import Call, Turn, Usage  # noqa: E402
from xagent.runner import RunResult  # noqa: E402
from xagent.runner import Runner as RealRunner  # noqa: E402
from xagent.spawn import AgentError, Handle, status_table  # noqa: E402

PASS, FAIL = [], []

# Every wait in this suite is bounded by one of these. Generous, because the point
# of a bound here is that a broken build fails instead of hanging -- not that a
# loaded machine is measured.
PATIENCE = 20.0


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32m✓\033[0m" if cond else "\033[31m✗\033[0m"
    print(f"  {mark} {name}" + (f"  — {detail}" if detail and not cond else ""))


def wait_for(predicate, timeout: float = PATIENCE) -> bool:
    """Poll until true or out of patience. Never the assertion itself -- the thing
    that stops a broken build from hanging the suite."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def capture(fn):
    """Run `fn`, returning (value, raised, stdout). Nothing escapes but the value."""
    buffer = io.StringIO()
    raised = None
    value = None
    with contextlib.redirect_stdout(buffer):
        try:
            value = fn()
        except BaseException as e:  # the whole point: BaseException, not Exception
            raised = e
    return value, raised, buffer.getvalue()


# --------------------------------------------------------------- the fake model


class StubProvider:
    """A scripted model. Each entry is a Turn, a BaseException to raise, or a
    callable returning a Turn -- the callable is where this suite gets to act
    while the child is mid-request, with no sleeping and no race."""

    def __init__(self, turns):
        self.turns, self.n, self.asked = list(turns), 0, []
        self.backend = config.BACKENDS["codiv"]
        self.model, self.thinking, self.sampling = "stub", None, "thinking"
        self.usage, self.on_delta, self.truncated_turns = Usage(), None, 0
        self.is_subagent = False
        # Set to the runner's store by `child()`, so every request can record what
        # the transcript held at the moment it went out. That is how "delivered
        # before the first sample" is asserted rather than timed.
        self.store = None
        self.seen: list[list[str]] = []

    def sample(self, system, messages, *, tools=True):
        self.asked.append(tools)
        self.seen.append([c.code for c in self.store.cells] if self.store else [])
        turn = self.turns[self.n] if self.n < len(self.turns) else Turn()
        self.n += 1
        if callable(turn):
            turn = turn()
        if isinstance(turn, BaseException):
            raise turn
        return turn

    def system_blocks(self, system):
        return [{"type": "text", "text": system}]

    def complete(self, prompt, model=None, max_tokens=4096):
        return "summary"


_ids = iter(range(1, 10_000))


def work(code: str, text: str = "") -> Turn:
    # A non-zero `input_tokens`, because a parent watching a real run reads a
    # token count off the handle and a turn that reported none would leave it at
    # zero for the whole run.
    return Turn(text=text, input_tokens=1200,
                calls=[Call(code=code, tool_use_id=f"tu_{next(_ids)}")])


# ------------------------------------------------- driving a real child, offline

# `spawn._work` builds its own Runner, which would reach for a provider and a
# network. The class is swapped for a lookup into this table so that the worker
# under test is the real one -- link.begin(), link.finish(), the lot -- while the
# Runner it drives is the scripted one prepared here. Keyed by task, so two
# children starting at once cannot take each other's.
_PREPARED: dict[str, RealRunner] = {}
# Every real child this suite started, so teardown can stop any that an early
# failure left running rather than joining a thread that is sleeping for ten
# minutes.
_LIVE: list[Handle] = []


def _handoff(task, **kwargs):
    return _PREPARED.pop(task)


def child(task: str, turns, executor: ThreadPoolExecutor, label: str | None = None,
          link: "spawn._Link | None" = None) -> tuple[Handle, RealRunner, StubProvider]:
    """A real subagent: real _Link, real Runner, real kernel, scripted model."""
    link = link if link is not None else spawn._Link(label or task)
    provider = StubProvider(turns)
    runner = RealRunner(task, provider=provider, is_subagent=True,
                        on_event=link.on_event, channel=link)
    provider.store = runner.store
    _PREPARED[task] = runner
    future = executor.submit(spawn._work, link, task, None, None, 8, None, None)
    handle = Handle(future, label or task, task, link)
    _LIVE.append(handle)
    return handle, runner, provider


def settled(exc: BaseException) -> Future:
    """A future that already carries an exception, with no thread behind it."""
    future: Future = Future()
    future.set_running_or_notify_cancel()
    future.set_exception(exc)
    return future


def handle_over(future: Future, label: str = "fake") -> Handle:
    return Handle(future, label, "a prompt", spawn._Link(label))


class Spy:
    """A handle-shaped stand-in for gather(), which records being waited on."""

    def __init__(self, label: str, outcome, done: bool = False):
        self.label, self.outcome, self.done = label, outcome, done
        self.waited = 0

    def result(self, timeout=None):
        self.waited += 1
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def reset_registry() -> None:
    """So one section's handles are invisible to the next section's poll()."""
    with spawn._lock:
        spawn._handles.clear()
        spawn._spawned = 0


def conversation_faults(msgs: list[dict]) -> list[str]:
    """Everything wrong with a rendered conversation, as the API would see it."""
    faults = []
    if not msgs or msgs[0]["role"] != "user":
        faults.append("does not open with the user")
    for i, m in enumerate(msgs):
        want = "user" if i % 2 == 0 else "assistant"
        if m["role"] != want:
            faults.append(f"message {i} is {m['role']}, expected {want}")
    if msgs and msgs[-1]["role"] != "user":
        faults.append("does not end on a user message")
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        uses = [b["id"] for b in m["content"] if b.get("type") == "tool_use"]
        nxt = msgs[i + 1] if i + 1 < len(msgs) else None
        if not uses:
            faults.append(f"assistant message {i} calls nothing")
            continue
        if nxt is None:
            faults.append(f"tool_use {uses} is never answered")
            continue
        results = [b["tool_use_id"] for b in nxt["content"]
                   if b.get("type") == "tool_result"]
        if results != uses:
            faults.append(f"message {i}: {uses} answered by {results}")
    return faults


# ------------------------------------------------------------------------ suites


def interrupts() -> None:
    print("an interrupt is the cell being cut short, not a subagent failing:")
    print("        swallowing it left the kernel waiting after the cell was dead")
    for exc in (KeyboardInterrupt(), SystemExit(2)):
        name = type(exc).__name__
        h = handle_over(settled(exc), "interrupted")
        value, raised, _ = capture(h.result)
        check(f"{name} from the child comes back out of result()",
              type(raised) is type(exc), f"raised={raised!r} returned={value!r}")
        check(f"and {name} is not quietly converted into an AgentError",
              not isinstance(value, AgentError), repr(value))

    class Interrupted(Future):
        """The real shape of the bug: the wait itself is interrupted."""

        def result(self, timeout=None):
            raise KeyboardInterrupt("cell timeout")

    h = handle_over(Interrupted(), "still-working")
    _, raised, _ = capture(h.result)
    check("an interrupt raised in the waiting thread escapes result() too",
          isinstance(raised, KeyboardInterrupt), repr(raised))

    with ThreadPoolExecutor(max_workers=1) as ex:
        def die():
            raise KeyboardInterrupt("from inside the child")
        h = handle_over(ex.submit(die), "real-thread")
        _, raised, _ = capture(h.result)
        check("and so does one that really crossed a worker thread",
              isinstance(raised, KeyboardInterrupt), repr(raised))

    print("\nan ordinary failure is still a value in a slot, not an exception")
    h = handle_over(settled(RuntimeError("peer closed connection")), "crashed")
    out = h.result()
    check("an Exception becomes an AgentError", isinstance(out, AgentError), repr(out))
    check("named by its type, so a wire error stays recognisable",
          "RuntimeError: peer closed connection" in out.message, out.message)
    check("carrying the label of the child it belongs to", out.label == "crashed",
          out.label)
    check("and it is falsy, so a batch can be filtered", not out and bool(out) is False)

    print("\ngather() hands the interrupt straight back out")
    boom, sibling = Spy("boom", KeyboardInterrupt()), Spy("sibling", "unreached")
    _, raised, printed = capture(lambda: runtime.gather([boom, sibling]))
    check("the interrupt propagates out of gather()",
          isinstance(raised, KeyboardInterrupt), repr(raised))
    check("it does not go on to wait on the handles behind it",
          sibling.waited == 0, f"sibling waited {sibling.waited}x")
    check("and it says how many children are still running",
          "[gather] interrupted with 2 subagent(s) still running" in printed,
          repr(printed))
    check("naming them, so the parent knows what to poll", "boom" in printed
          and "sibling" in printed, repr(printed))
    check("and saying they are untouched rather than lost",
          "still working" in printed and "gather() again" in printed, repr(printed))
    check("the whole notice is one paragraph, not a dump",
          len(printed) < 400, f"{len(printed)} chars")

    done_already = Spy("finished", "kept", done=True)
    later = [Spy(f"pending{i}", "unreached") for i in range(4)]
    _, raised, printed = capture(
        lambda: runtime.gather([done_already, Spy("boom", KeyboardInterrupt()), *later]))
    check("results already collected are not what the count is about",
          "5 subagent(s) still running" in printed, repr(printed))
    check("only the first few are named, so the line cannot grow with the fan-out",
          printed.count("pending") <= 3 and "…" in printed, repr(printed))
    check("and the earlier handles really were waited on",
          done_already.waited == 1 and all(s.waited == 0 for s in later))

    _, raised, _ = capture(lambda: runtime.gather([Spy("exit", SystemExit(1))]))
    check("SystemExit is not absorbed into a slot either",
          isinstance(raised, SystemExit), repr(raised))

    print("\nbut one child's ordinary failure must not discard its siblings")
    out = runtime.gather([Spy("a", "first"),
                          handle_over(settled(ValueError("bad input")), "b"),
                          Spy("c", "third")])
    check("the batch comes back whole and in order",
          len(out) == 3 and out[0] == "first" and out[2] == "third", repr(out))
    check("with an AgentError only in the slot that failed",
          isinstance(out[1], AgentError) and "ValueError: bad input" in out[1].message,
          repr(out[1]))
    check("so the survivors filter out cleanly",
          [r for r in out if r] == ["first", "third"], repr(out))


def no_wall_clock() -> None:
    gate = threading.Event()
    ex = ThreadPoolExecutor(max_workers=8, thread_name_prefix="xa-test-clock")

    def blocked(value="eventually"):
        # The bounded fallback: if the gate is never set, this still ends.
        gate.wait(PATIENCE)
        return RunResult(value=value, finish="done", turns=3, last_text="all done")

    try:
        print("a subagent runs for as long as its job takes")
        slow = handle_over(ex.submit(lambda: (time.sleep(0.9),
                                              RunResult(value="slow value",
                                                        finish="done"))[1]), "slow")
        started = time.monotonic()
        out = runtime.gather([slow])
        elapsed = time.monotonic() - started
        check("gather() with no timeout waits past any default there might be",
              out == ["slow value"] and elapsed >= 0.85, f"{elapsed:.2f}s {out!r}")

        print("\nand timeout=0 means poll once — the reading nobody could have wanted")
        finished = handle_over(ex.submit(lambda: RunResult(value=7, finish="done")))
        check("the finished handle resolves", finished.result(timeout=PATIENCE) == 7)
        running = handle_over(ex.submit(blocked), "still-going")
        check("it really is running", wait_for(lambda: running.running))
        started = time.monotonic()
        out = runtime.gather([finished, running], timeout=0)
        elapsed = time.monotonic() - started
        check("gather(timeout=0) returns at once rather than waiting forever",
              elapsed < 2.0, f"{elapsed:.2f}s")
        check("a child that had finished still yields its value", out[0] == 7, repr(out))
        caught = out[1]
        check("and the one it caught mid-flight is an AgentError",
              isinstance(caught, AgentError), repr(caught))
        check("flagged as still running, not as a failure", caught.still_running is True)
        check("and as the deadline it was", caught.is_timeout is True)
        check("said in words too, since this is read by a model",
              "this is the gather deadline, not a failure" in caught.message,
              caught.message)
        check("with the way to pick it back up", "gather() it again" in caught.message
              and "poll()" in caught.message, caught.message)
        check("it is falsy like every AgentError, so `if r` is still safe",
              not caught)

        failure = runtime.gather([handle_over(settled(OSError("disk gone")), "died")])[0]
        check("a real failure is distinguishable from it, field by field",
              isinstance(failure, AgentError) and failure.still_running is False
              and failure.is_timeout is False, repr(failure))

        print("\nthe handle it gave up on is untouched, and gathering it works")
        check("the handle is still running afterwards", running.running is True)
        gate.set()
        out = runtime.gather([running], timeout=PATIENCE)
        check("gathering it again returns the real value", out == ["eventually"],
              repr(out))
        check("and the deadline left no mark on the run",
              running.state == "done" and running.turns == 3,
              f"{running.state} turns={running.turns}")

        print("\nthe deadline covers the batch, not each handle in turn")
        gate2 = threading.Event()
        try:
            stuck = [handle_over(ex.submit(lambda: gate2.wait(PATIENCE)), f"s{i}")
                     for i in range(6)]
            started = time.monotonic()
            out = runtime.gather(stuck, timeout=0.4)
            elapsed = time.monotonic() - started
            check("six handles and one 0.4s deadline cost 0.4s, not 2.4s",
                  elapsed < 1.5, f"{elapsed:.2f}s")
            check("every slot says the same thing: still running",
                  all(isinstance(r, AgentError) and r.still_running for r in out),
                  repr(out))
        finally:
            gate2.set()

        print("\nand a negative deadline is a mistake, not a shorthand")
        _, raised, _ = capture(lambda: runtime.gather([finished], timeout=-1))
        check("a negative timeout raises ValueError", isinstance(raised, ValueError),
              repr(raised))
        check("naming the value it was given", "-1" in str(raised), str(raised))
        spy = Spy("untouched", "never")
        _, raised, _ = capture(lambda: runtime.gather([spy], timeout=-0.001))
        check("however slightly negative", isinstance(raised, ValueError), repr(raised))
        check("and it is raised before anything is waited on, so nothing half-gathers",
              spy.waited == 0, f"waited {spy.waited}x")
        check("gather([]) is an empty list, not an error", runtime.gather([]) == [])
        check("gather accepts any iterable, not only a list",
              runtime.gather(iter([finished])) == [7])
    finally:
        gate.set()
        ex.shutdown(wait=True)


def messaging_unit() -> None:
    gate = threading.Event()
    ex = ThreadPoolExecutor(max_workers=2, thread_name_prefix="xa-test-msg")
    try:
        h = handle_over(ex.submit(lambda: (gate.wait(PATIENCE),
                                           RunResult(finish="done"))[1]), "listener")
        check("nothing is queued before anything is sent", h.messages_sent == 0)
        said = h.send("switch to plan B")
        check("send() reports the queue depth", "(1 waiting)" in said, said)
        check("and says when it will arrive",
              "message from you at the end of its current turn" in said, said)
        check("it is one line, because it lands in a context window",
              "\n" not in said and len(said) < 200, repr(said))
        check("the count moves", h.messages_sent == 1, str(h.messages_sent))
        check("a second message queues behind the first",
              "(2 waiting)" in h.send("and then plan C"), "")
        check("runtime.send() is the same call by another name",
              "(3 waiting)" in runtime.send(h, "plan D"), "")
        check("counted once each", h.messages_sent == 3, str(h.messages_sent))

        for empty in ("", "   ", "\n\t "):
            said = h.send(empty)
            check(f"an empty message ({empty!r}) is refused",
                  "nothing queued" in said and "empty" in said, said)
        check("and refusing one costs nothing", h.messages_sent == 3,
              str(h.messages_sent))
        check("the mailbox holds exactly what was queued",
              h._link.drain() == ["switch to plan B", "and then plan C", "plan D"], "")
        check("draining it empties it", h._link.drain() == [])

        gate.set()
        check("the child finishes", wait_for(lambda: h.done))
        said = h.send("too late")
        check("sending to a finished child is a sentence, not an exception",
              "already finished" in said and "nothing was queued" in said, said)
        check("and nothing is queued behind its back", h._link.drain() == [])
        check("nor counted", h.messages_sent == 3, str(h.messages_sent))
    finally:
        gate.set()
        ex.shutdown(wait=True)


def messaging_live(ex: ThreadPoolExecutor) -> None:
    print("a message becomes a real user message in the child's conversation")
    task = "audit the parser"
    mid = "stop auditing, just count the files"
    link = spawn._Link("auditor")
    # Two messages posted before the run starts, as a parent that spawned and
    # immediately thought of something does it.
    check("posting reports the queue depth", link.post("read the README first") == 1)
    check("and again for the second", link.post("and the licence") == 2)

    def midway():
        # Posted while the child is mid-request. Delivery is at the *end* of the
        # turn it lands in -- the cell of that turn runs first, then the boundary
        # above the next request delivers. No sleeping, and nothing to race.
        link.post(mid)
        return work("b = 2", "still auditing")

    handle, runner, provider = child(
        task, [work("a = 1", "starting the audit"), midway,
               work("done({'files': 3})", "finished")], ex, label="auditor", link=link)
    run = handle._future.result(timeout=PATIENCE)

    marks = runner.store.marks
    codes = [c.code for c in runner.store.cells]
    check("every message arrived as a mark of its own", len(marks) == 3,
          f"{len(marks)} marks, cells={codes}")
    check("and not as a cell the child never wrote", "inbox()" not in codes, str(codes))
    check("each is a user message, which is what somebody talking is",
          all(m.role == "user" for m in marks), str([m.role for m in marks]))
    check("labelled as the parent talking, so the model knows who it is",
          all(m.tag == "message-from-parent" for m in marks),
          str([m.tag for m in marks]))
    check("it carries the parent's words verbatim", marks[2].text == mid,
          repr(marks[2].text))
    check("the two queued before the run started arrived before any cell ran",
          [m.after for m in marks[:2]] == [0, 0], str([m.after for m in marks]))
    check("the one sent mid-run landed after the cell of the turn it arrived in",
          marks[2].after == 2, f"after={marks[2].after} cells={codes}")
    check("so the turn it interrupted still ran its own cell first",
          codes[:2] == ["a = 1", "b = 2"], str(codes))
    check("each message was delivered exactly once",
          sum(m.text == mid for m in marks) == 1, str([m.text for m in marks]))
    check("and the mailbox is empty afterwards", link.drain() == [])

    msgs = runner.store.messages()
    faults = conversation_faults(msgs)
    check("the transcript still renders as a valid conversation", not faults,
          "; ".join(faults))
    check("with the message in it", mid in str(msgs))
    said = " ".join(b.get("text", "") for m in msgs for b in m["content"])
    check("wrapped in the tag the model is told to read it by",
          f"<message-from-parent>\n{mid}\n</message-from-parent>" in said, said[:300])
    check("the two posted before the run ride the opening user message",
          msgs[0]["role"] == "user"
          and "read the README first" in str(msgs[0]["content"]), str(msgs[0])[:300])
    check("every tool_use in the transcript is still `python`",
          all(b["name"] == "python" for m in msgs if m["role"] == "assistant"
              for b in m["content"] if b.get("type") == "tool_use"), "")
    check("and no assistant message claims the child asked to be messaged",
          "message-from-parent" not in str([m for m in msgs
                                            if m["role"] == "assistant"]), "")

    check("the run itself is untouched by any of it",
          run.finish == "done" and run.value == {"files": 3},
          f"{run.finish} {run.value!r}")
    check("and the parent's gather() gets the value",
          runtime.gather([handle]) == [{"files": 3}], "")
    check("the link counted every message that went out",
          handle.messages_sent == 3, str(handle.messages_sent))


def no_inbox_left() -> None:
    print("\ninbox() is gone: a message is a message, not a cell to run")
    check("it is not in the model-facing namespace",
          not any(getattr(o, "__name__", "") == "inbox" for o in runtime.PUBLIC),
          str([getattr(o, "__name__", "") for o in runtime.PUBLIC]))
    check("nor is it defined at all", not hasattr(runtime, "inbox"))
    check("and nothing pushes a message into a kernel any more",
          not hasattr(runtime, "_deliver"))
    check("the subagent prompt describes a message instead of a cell",
          "<message-from-parent>" in SUBAGENT_CODA
          and "inbox()" not in SUBAGENT_CODA, "")
    check("and says it outranks the task the parent spawned it with",
          "outranks the task" in SUBAGENT_CODA, "")


def killing(ex: ThreadPoolExecutor) -> None:
    print("kill() during a long cell: the cell dies now, not at its own timeout")
    handle, runner, _ = child(
        "sleep forever",
        [work("import time; time.sleep(600)", "about to sleep for ten minutes"),
         work("done('never')", "unreachable")], ex, label="sleeper")
    # The cell never completes, so there is no "cell" event to wait for: what says
    # the child is in it is a published kernel and a turn under way.
    check("the child has a kernel of its own and a turn under way",
          wait_for(lambda: handle._link.kernel is not None and handle.turns >= 1),
          f"turns={handle.turns} kernel={handle._link.kernel!r}")
    time.sleep(0.3)  # long enough to be well inside the sleep(600)
    started = time.monotonic()
    said = handle.kill("changed my mind")
    out = handle.result(timeout=PATIENCE)
    latency = time.monotonic() - started
    check("kill() itself returns one line and does not wait",
          "\n" not in said and "is stopping" in said and "changed my mind" in said, said)
    check("the child is gone in seconds, not in the cell's remaining ten minutes",
          latency < 10.0, f"{latency:.2f}s")
    check("its parent gets an AgentError", isinstance(out, AgentError), repr(out))
    check("naming the kill and the reason given",
          "killed by parent: changed my mind" in out.message, out.message)
    check("the turn it reached", "stopped at turn 1" in out.message, out.message)
    check("and the last thing it wrote",
          "ten minutes" in out.message and "ten minutes" in (out.last_text or ""),
          f"{out.message} | {out.last_text!r}")
    check("the AgentError carries the turn count as a field too", out.turns == 1,
          str(out.turns))
    check("it is not flagged as still running — this one really is over",
          out.still_running is False and out.is_timeout is False, repr(out))
    run = handle._future.result(timeout=PATIENCE)
    check("the run itself is recorded as killed", run.finish == "killed", run.finish)
    check("keeping the turns it had done", run.turns >= 1, str(run.turns))
    check("and the prose it had written",
          "ten minutes" in run.last_text, repr(run.last_text))
    check("the work it did before the kill is still in the store",
          any("sleep(600)" in c.code for c in runner.store.cells),
          str([c.code for c in runner.store.cells]))
    check("a subagent is still handed no prose to speak", run.answer == "",
          repr(run.answer))
    check("the handle settles on killed", handle.state == "killed", handle.state)
    check("done, and not running", handle.done and not handle.running)

    print("\nkill() between turns, where there is no cell to interrupt")
    resume = threading.Event()
    handle2, runner2, _ = child(
        "between turns",
        [work("x = 1", "first pass"),
         lambda: (resume.wait(PATIENCE), work("y = 2", "second pass"))[1],
         work("done('never')")], ex, label="thinker")
    check("it reaches the second request", wait_for(lambda: handle2.turns >= 2),
          f"turns={handle2.turns}")
    started = time.monotonic()
    said = handle2.kill()
    resume.set()
    out2 = handle2.result(timeout=PATIENCE)
    check("kill with no reason is still one line", "\n" not in said
          and "is stopping" in said, said)
    check("it takes effect at the turn boundary", time.monotonic() - started < 10.0)
    check("and the AgentError says no reason was given",
          isinstance(out2, AgentError) and "no reason given" in out2.message,
          repr(out2))
    check("the cell that had already run survives",
          [c.code for c in runner2.store.cells] == ["x = 1"],
          str([c.code for c in runner2.store.cells]))
    check("and the turn the kill landed on never ran its cell",
          not any(c.code == "y = 2" for c in runner2.store.cells), "")
    run2 = handle2._future.result(timeout=PATIENCE)
    check("recorded as killed all the same", run2.finish == "killed", run2.finish)
    check("with an error naming the parent", "killed by parent" in (run2.error or ""),
          repr(run2.error))

    print("\nkilling is never allowed to raise, whatever it is aimed at")
    check("killing a finished child says so",
          "already finished" in handle.kill(), handle.kill())
    check("killing twice says the same thing, and does nothing",
          "already finished" in handle.kill("again"), "")
    check("killing an already-killed handle is idempotent",
          handle2.kill() == handle2.kill(), "")
    check("every one of them is one line",
          all("\n" not in s for s in (handle.kill(), handle2.kill("x"))), "")

    gate = threading.Event()
    queue = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xa-test-queued")
    try:
        queue.submit(gate.wait, PATIENCE)
        never = queue.submit(lambda: RunResult(value="never ran", finish="done"))
        h = handle_over(never, "queued")
        said = h.kill("not needed after all")
        check("a child killed before it started says it was cancelled",
              "cancelled before it started" in said, said)
        out = h.result(timeout=PATIENCE)
        check("and its result is an AgentError, not a dangling wait",
              isinstance(out, AgentError), repr(out))
        check("naming the kill and the reason",
              "killed by parent: not needed after all" in out.message, out.message)
        check("and saying there was nothing to salvage",
              "cancelled before it started" in out.message, out.message)
        check("the handle reports killed", h.state == "killed", h.state)
        check("killing it again is a no-op", "already finished" in h.kill(), "")
    finally:
        gate.set()
        queue.shutdown(wait=True)

    class Hostile:
        def interrupt(self):
            raise OSError("the channel is gone")

    stuck = threading.Event()
    ex2 = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xa-test-hostile")
    try:
        h = handle_over(ex2.submit(lambda: (stuck.wait(PATIENCE),
                                            RunResult(finish="done"))[1]), "hostile")
        check("the child is running", wait_for(lambda: h.running))
        h._link.attach(Hostile())
        said = h.kill("stop")
        check("a kill whose interrupt blows up still returns a sentence",
              said.startswith("[kill] could not stop") and "OSError" in said, said)
        check("still one line", "\n" not in said, repr(said))
    finally:
        stuck.set()
        ex2.shutdown(wait=True)


def progress(ex: ThreadPoolExecutor) -> None:
    print("what a parent can read while the child is still working")
    resume = threading.Event()
    handle, runner, _ = child(
        "a long job",
        [work("a = 1", "reading the source"),
         work("b = 2", "found the parser"),
         lambda: (resume.wait(PATIENCE), work("done('finished')"))[1]],
        ex, label="worker")

    check("a handle starts out running, not done",
          wait_for(lambda: handle.state == "running"), handle.state)
    check("it reaches the third request",
          wait_for(lambda: handle.turns >= 3), f"turns={handle.turns}")
    check("done and running are opposites, always",
          handle.done is (not handle.running))
    check("cells are counted as they run", handle.cells >= 2, str(handle.cells))
    check("turns are counted as they are sampled", handle.turns == 3,
          str(handle.turns))
    check("tokens are reported once there is a reading",
          handle.tokens > 0, str(handle.tokens))
    check("the clock runs", handle.seconds > 0, str(handle.seconds))
    check("and the last thing the child said is readable mid-run",
          "found the parser" in handle.last_text, repr(handle.last_text))
    check("sending to it counts", handle.send("nearly there?")
          and handle.messages_sent == 1, str(handle.messages_sent))

    line = repr(handle)
    check("repr is a single line", "\n" not in line, repr(line))
    check("short enough to print into a context window", len(line) < 140,
          f"{len(line)}: {line}")
    check("naming the label and the state", "worker" in line and "running" in line,
          line)
    check("and the turn it is on", "turn 3" in line, line)

    table = status_table([handle])
    check("poll() names the agent without waiting for it", "worker" in table, table)
    check("and says how many are still running", "1 still running" in table, table)

    resume.set()
    check("the value comes back", handle.result(timeout=PATIENCE) == "finished")
    check("the state settles on done", handle.state == "done", handle.state)
    stopped = handle.seconds
    time.sleep(0.05)
    check("the clock stops when the run does", handle.seconds == stopped,
          f"{stopped} -> {handle.seconds}")
    check("and repr says done too", "done" in repr(handle), repr(handle))

    print("\nthe state a handle ends in, for each way a run can end")
    failed = handle_over(settled(RuntimeError("crashed")), "crasher")
    check("a worker that raised reads as failed", failed.state == "failed",
          failed.state)
    ex2 = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xa-test-states")
    try:
        link = spawn._Link("errored")
        run = RunResult(error="model refused", finish="error", turns=4,
                        last_text="I got four turns in.")
        link.begin()
        link.record(turns=4, last_text=run.last_text)
        link.finish("failed")
        h = Handle(ex2.submit(lambda: run), "errored", "p", link)
        out = h.result(timeout=PATIENCE)
        check("a run that came back not-ok is failed too", h.state == "failed", h.state)
        check("and its AgentError carries how far it got",
              isinstance(out, AgentError) and out.turns == 4
              and "stopped at turn 4" in out.message, repr(out))
        check("and the last prose, so the work is not reported as nothing",
              "four turns in" in (out.last_text or ""), repr(out.last_text))

        cancelled: Future = Future()
        cancelled.cancel()
        check("a cancelled future reads as killed",
              handle_over(cancelled, "gone").state == "killed", "")
        out = handle_over(cancelled, "gone").result()
        check("and resolves to an AgentError rather than raising",
              isinstance(out, AgentError)
              and "cancelled before it started" in out.message, repr(out))
    finally:
        ex2.shutdown(wait=True)

    print("\nthe turn a parent is shown is a turn, not a cell count")
    # RunResult.turns counts cells, which is not the same number. Taking it at the
    # end made poll()'s turn column jump the moment the child finished, and reported
    # a kill as landing at a turn that never happened.
    counted = spawn._Link("counted")
    counted.begin()
    for n in (1, 2, 3):
        counted.on_event("turn_start", {"n": n, "tokens": 100 * n})
    counted.record(turns=5, tokens=900, last_text="three turns, five cells")
    check("a run's cell count never overwrites the turns the events counted",
          counted.turns == 3, str(counted.turns))
    check("but it still fills the gap when no event ever arrived",
          (lambda link: (link.record(turns=7), link.turns)[1])(spawn._Link("silent")) == 7)
    killed_link = spawn._Link("mid-message")
    killed_link.begin()
    killed_link.on_event("turn_start", {"n": 2, "tokens": 40})
    killed_link.stop("enough")
    ex3 = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xa-test-turns")
    try:
        run = RunResult(error="killed by parent: enough", finish="killed", turns=6,
                        last_text="halfway through")
        killed_link.record(turns=6, last_text=run.last_text)
        killed_link.finish("killed")
        out = Handle(ex3.submit(lambda: run), "mid-message", "p",
                     killed_link).result(timeout=PATIENCE)
        check("so a kill is reported at the turn the child actually reached",
              isinstance(out, AgentError) and out.turns == 2
              and "stopped at turn 2" in out.message, repr(out))
    finally:
        ex3.shutdown(wait=True)

    print("\nrepr and the poll table stay small however wide the run gets")
    wide = spawn._Link("l" * 80)
    wide.begin()
    wide.record(turns=128, tokens=1_234_567, last_text="p" * 400)
    h = Handle(settled(RuntimeError("x")), "l" * 80, "p", wide)
    line = repr(h)
    check("a monstrous label and a monstrous sentence still make one line",
          "\n" not in line, repr(line))
    # 144 chars at its very widest: a 40-char label and a 48-char sentence, both
    # already clipped, plus turn/duration/tokens. Wider than the ~120 a terminal
    # gives it, so the widest possible repr does wrap on screen -- but it is a
    # bounded overshoot rather than an unbounded one, which is what matters here.
    check("clipped hard on both halves", len(line) < 160, f"{len(line)}: {line}")
    row = status_table([h]).splitlines()[1]
    check("and the poll row is clipped too", len(row) < 160, f"{len(row)}: {row}")


def polling() -> None:
    print("poll() with no arguments is the whole session, and never blocks")
    reset_registry()
    real_work = spawn._work
    gate = threading.Event()

    def fake_work(link, prompt, seed, model, max_turns, thinking, sampling):
        link.begin()
        link.record(turns=2, tokens=4321, last_text=f"worked on {prompt}")
        if "slow" in prompt:
            gate.wait(PATIENCE)
        link.finish("done")
        return RunResult(value=f"result of {prompt}", finish="done", turns=2)

    spawn._work = fake_work
    try:
        handles = [spawn.spawn(f"job {i}", label=f"job-{i}") for i in range(3)]
        slow = spawn.spawn("slow job", label="slow-job")
        check("every spawn is registered", len(spawn.registry()) == 4,
              str(len(spawn.registry())))
        check("registry() hands out a copy, not the list itself",
              (spawn.registry().clear(), len(spawn.registry()))[1] == 4, "")
        check("the slow one is still running", wait_for(lambda: slow.running))

        started = time.monotonic()
        _, _, printed = capture(runtime.poll)
        elapsed = time.monotonic() - started
        check("poll() returns immediately even with a child mid-run",
              elapsed < 2.0, f"{elapsed:.2f}s")
        check("it covers every handle in the registry",
              all(f"job-{i}" in printed for i in range(3)) and "slow-job" in printed,
              printed)
        check("it counts them, and how many are still going",
              "[4 subagent(s), 1 still running]" in printed, printed.splitlines()[0])
        check("it stays small enough to print into a context window",
              len(printed) < 1200, f"{len(printed)} chars")
        check("every row is one line under 160 chars",
              all(len(r) < 160 for r in printed.splitlines()),
              str(max(len(r) for r in printed.splitlines())))
        check("each row carries the state, the turn and the cost",
              all(("turn" in r and "tok" in r) for r in printed.splitlines()[1:]),
              printed)
        slow.send("are you nearly there?")
        check("a message sent is shown as an arrow, so it is not silently pending",
              "1↑" in status_table([slow]), status_table([slow]))

        _, _, only = capture(lambda: runtime.poll(handles[:1]))
        check("poll(handles) narrows to what it was given",
              "[1 subagent(s)" in only and "job-1" not in only, only)
        check("an empty session says so rather than printing a blank table",
              status_table([]) == "[no subagents in this session]", status_table([]))

        gate.set()
        check("the slow child finishes",
              slow.result(timeout=PATIENCE) == "result of slow job")
        check("and the values crossed back unchanged",
              runtime.gather(handles) == [f"result of job {i}" for i in range(3)],
              repr(runtime.gather(handles)))
        check("poll() afterwards reports nothing running",
              "0 still running" in status_table(), status_table().splitlines()[0])

        print("\nthe guards that stop a fan-out from spawning forever")
        depth = os.environ.get("XAGENT_DEPTH")
        os.environ["XAGENT_DEPTH"] = str(config.MAX_AGENT_DEPTH)
        try:
            h = spawn.spawn("too deep")
        finally:
            if depth is None:
                os.environ.pop("XAGENT_DEPTH", None)
            else:
                os.environ["XAGENT_DEPTH"] = depth
        out = h.result(timeout=PATIENCE)
        check("a spawn past the depth limit fails immediately",
              isinstance(out, AgentError) and "depth limit" in out.message, repr(out))
        check("without starting anything", h.done and h.state == "failed", h.state)

        with spawn._lock:
            spawn._spawned = config.MAX_TOTAL_AGENTS
        h = spawn.spawn("one too many")
        out = h.result(timeout=PATIENCE)
        check("and one past the session cap fails the same way",
              isinstance(out, AgentError) and "agent cap" in out.message, repr(out))
    finally:
        gate.set()
        spawn._work = real_work
        if spawn._pool is not None:
            spawn._pool.shutdown(wait=True)
            spawn._pool = None
        reset_registry()

    print("\nseeds are checked at spawn time, naming the key that is wrong")
    for seed, exc, needle in (({"gather": 1}, ValueError, "shadow"),
                              ({"not an identifier": 1}, ValueError, "identifier"),
                              ({"lock": threading.Lock()}, TypeError, "seed[")):
        _, raised, _ = capture(lambda s=seed: spawn._check_seed(s))
        check(f"seed {list(seed)[0]!r} is refused as {exc.__name__}",
              isinstance(raised, exc) and needle in str(raised), repr(raised))
    _, raised, _ = capture(lambda: spawn._check_seed(["not", "a", "dict"]))
    check("a seed that is not a dict is refused too", isinstance(raised, TypeError),
          repr(raised))


def non_regression(ex: ThreadPoolExecutor) -> None:
    print("nothing above changed what a subagent that simply works does")
    handle, runner, provider = child(
        "count the files", [work("n = 50", "counting"), work("done({'n': n})", "done")],
        ex, label="counter")
    out = runtime.gather([handle])
    check("gather() returns the done() value, unchanged and unwrapped",
          out == [{"n": 50}], repr(out))
    check("it is a real object, not a rendering", isinstance(out[0], dict))
    check("no message was ever delivered, so no cell was injected",
          [c.code for c in runner.store.cells] == ["n = 50", "done({'n': n})"],
          str([c.code for c in runner.store.cells]))
    check("the transcript is valid", not conversation_faults(runner.store.messages()),
          "; ".join(conversation_faults(runner.store.messages())))
    check("a subagent gets the subagent rendering of the python tool",
          provider.is_subagent is True, repr(provider.is_subagent))
    check("gathering the same handle twice gives the same value",
          runtime.gather([handle, handle]) == [{"n": 50}, {"n": 50}], "")
    check("and the handle reads done", handle.state == "done" and handle.done)

    print("\nAgentError's own shape, which parent code destructures")
    e = AgentError("label", "message")
    check("it still constructs positionally", e.label == "label"
          and e.message == "message", repr(e))
    check("with the quiet defaults a caller relies on",
          (e.turns, e.last_text, e.still_running, e.is_timeout) == (0, "", False, False),
          repr((e.turns, e.last_text, e.still_running, e.is_timeout)))
    check("it is falsy", not e and bool(e) is False)
    check("and its repr names the child and what went wrong",
          repr(e) == "<AgentError label: message>", repr(e))
    check("it is a plain value, not an exception class",
          not isinstance(e, BaseException))
    check("and it is in the namespace the model programs against",
          AgentError in runtime.PUBLIC)
    for tool in (runtime.gather, runtime.poll, runtime.send, runtime.kill,
                 runtime.agent):
        check(f"{tool.__name__}() is offered to the model", tool in runtime.PUBLIC)
    check("the docstring the model reads says there is no wall clock",
          "no wall clock" in (runtime.agent.__doc__ or ""), "")
    check("and that gather re-raises an interrupt instead of absorbing it",
          "comes straight back out" in (runtime.gather.__doc__ or ""), "")


def main() -> int:
    base_threads = threading.active_count()
    ex = ThreadPoolExecutor(max_workers=4, thread_name_prefix="xa-test-child")
    # `spawn._work` is the worker under test, so it stays real; the Runner it
    # builds would reach for a network, so the class it reaches for is swapped for
    # a lookup into the table `child()` fills. Restored below, whatever happens.
    real_runner_class = runner_module.Runner
    runner_module.Runner = _handoff
    try:
        interrupts()
        reset_registry()
        print("\na deadline is something a parent chose, never something it inherited")
        no_wall_clock()
        reset_registry()
        print("\ntalking to a running subagent")
        messaging_unit()
        messaging_live(ex)
        no_inbox_left()
        reset_registry()
        print("\nstopping one")
        killing(ex)
        reset_registry()
        print("\nwatching one")
        progress(ex)
        polling()
        reset_registry()
        print("")
        non_regression(ex)
    finally:
        runner_module.Runner = real_runner_class
        # Whatever went wrong above, nothing this suite started outlives it: stop
        # every child first, then join. The other order would wait out a sleep(600).
        for h in _LIVE:
            if not h.done:
                h.kill("suite teardown")
        for h in _LIVE:
            with contextlib.suppress(BaseException):
                h._future.result(timeout=PATIENCE)
        _LIVE.clear()
        ex.shutdown(wait=True)
        _PREPARED.clear()
        if spawn._pool is not None:
            spawn._pool.shutdown(wait=True)
            spawn._pool = None
        reset_registry()

    print("\nand nothing is left running behind it")
    settled_threads = wait_for(lambda: threading.active_count() <= base_threads, 10)
    check("every thread this suite started has been joined", settled_threads,
          f"{threading.active_count()} vs {base_threads}: "
          f"{[t.name for t in threading.enumerate()]}")
    check("and no handle is still running",
          all(h.done for h in spawn.registry()), str(spawn.registry()))

    print(f"\n{'─' * 60}\n{len(PASS)} passed, {len(FAIL)} failed")
    for name in FAIL:
        print(f"  \033[31mFAILED:\033[0m {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
