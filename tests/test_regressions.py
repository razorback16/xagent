"""Regressions for defects found in review. Each check names the bug it pins down.

Run with:  uv run python tests/test_regressions.py
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import keyword
import os
import re
import sys
import textwrap
import time
from pathlib import Path

from xagent import config, runtime, spawn
from xagent.compress import Compressor
from xagent.context import MAX_CODE_CHARS, ContextStore
from xagent.kernel import Kernel, _ControlSplitter
from xagent.runtime import CTL_BEGIN, CTL_END
from xagent.prompts import SUBAGENT_CODA, SYSTEM, SYSTEM_SUBAGENT
from xagent.provider import (PYTHON_TOOL, PYTHON_TOOL_SUBAGENT,
                             Provider, Usage)
from xagent.runner import DONE_EXPR, RunResult, Runner

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32m✓\033[0m" if cond else "\033[31m✗\033[0m"
    print(f"  {mark} {name}" + (f"  — {detail}" if detail and not cond else ""))


def main() -> int:
    k = Kernel()
    # The in-kernel done() is the subagent's finish and only signals for that
    # role, so everything about the value channel is exercised in a kernel that
    # really has the role. `k` stays top-level, where done() is a redirect.
    sub = Kernel(env={**os.environ, "XAGENT_ROLE": "subagent"})
    try:
        print("handles: _N named a slot one ahead of the real one, so every")
        print("         advertised handle was a NameError")
        for i in range(3):
            out = k.execute(f"'v{i}' + 'x' * 2000")
            advertised = re.findall(r"_(\d+)", out.render())
            live = json.loads(k.probe(
                "import re, json; print(json.dumps(sorted("
                "n for n in globals() if re.fullmatch(r'_\\d+', n))))"))
            check(f"cell {i}: advertised handle exists",
                  bool(advertised) and f"_{advertised[0]}" in live,
                  f"advertised _{advertised[0] if advertised else '?'}, live {live}")
        check("peek() on an advertised handle resolves",
              k.execute("peek(_3, n=1)").ok)

        print("\nhandles: no slot was invented outside the display path")
        k.execute("blob = 'q' * 5000")
        table = k.probe("import xagent.runtime as _r; _r._emit_var_table()")
        check("variable table advertises no phantom _N", not re.search(r"_\d+", table),
              table[:160])

        print("\ncontrol channel: the budgeted stream corrupted JSON and pickles")
        sub.execute("note('big', 'X' * 8000)")
        sub.execute("done({'answer': 42, 'blob': 'B' * 50000})")
        signals = json.loads(sub.probe("import xagent.runtime as _r; _r._emit_signals()"))
        check("a large note leaves signals parseable", signals["done"] is True)
        check("the note survives whole", len(signals["notes"]["big"]) == 8000,
              str(len(signals["notes"]["big"])))
        value = sub.probe_pickle(DONE_EXPR)
        check("a large done() value crosses intact",
              value["answer"] == 42 and len(value["blob"]) == 50000)
        check("shadowing print does not break the control channel",
              (sub.execute("print = lambda *a, **kw: None"),
               json.loads(sub.probe("import xagent.runtime as _r; _r._emit_signals()"))["done"])[1])
        sub.execute("del print")

        print("\ncontrol payload: a truncated one must be refused, never half-read")
        NONCE = "c0ffee"
        good = json.dumps({"v": 1, "signals": {"done": False}})

        def split(chunks, nonce=NONCE):
            sp = _ControlSplitter(nonce)
            seen = "".join(sp.feed(c) for c in chunks)
            seen += sp.finish()
            return seen, sp.result()

        body = CTL_BEGIN.format(NONCE) + good + CTL_END.format(NONCE)
        seen, (payload, err) = split(["out ", body, " more"])
        check("a complete payload is lifted out of the stream",
              payload == json.loads(good) and err is None, f"{payload} {err}")
        check("and the model's own output survives it intact", seen == "out  more",
              repr(seen))
        # The markers can land split across two stream messages.
        seen, (payload, err) = split([("out " + body)[:12], ("out " + body)[12:]])
        check("a marker split across chunks is still recognised",
              payload == json.loads(good) and seen == "out ", f"{payload} {seen!r}")
        seen, (payload, err) = split(["x", CTL_BEGIN.format(NONCE), good[:20]])
        check("a payload cut off mid-flight is refused, not half-read",
              payload is None and "truncated" in (err or ""), f"{payload} {err}")
        check("and its fragment never reaches the model's output", seen == "x", repr(seen))
        seen, (payload, err) = split([CTL_BEGIN.format("other") + good
                                      + CTL_END.format("other")])
        check("a payload under another nonce is left alone as output",
              payload is None and good in seen, f"{payload} {seen[:60]!r}")
        stale = CTL_BEGIN.format(NONCE) + json.dumps({"v": 1, "files": "OLD"}) \
            + CTL_END.format(NONCE)
        seen, (payload, err) = split([stale, body])
        check("when two arrive, the one the hook wrote last wins",
              payload == json.loads(good), str(payload))
        seen, (payload, err) = split([CTL_BEGIN.format(NONCE) + "{not json"
                                      + CTL_END.format(NONCE)])
        check("an unparseable payload is an error, not an empty dict",
              payload is None and "unparseable" in (err or ""), f"{payload} {err}")

        print("\ncontrol payload: a failed one must not overwrite what it could not read")
        runner = Runner.__new__(Runner)
        runner.store = ContextStore(task="t", system="s")
        runner.store.live_vars = "  findings  list  len=160"
        runner.store.live_files = "  created  a.py"
        runner.compressor = Compressor(provider=None, budget=1000)
        signals = runner._apply_payload({}, want_vars=True)
        check("an empty payload leaves the last known variables standing",
              runner.store.live_vars == "  findings  list  len=160", runner.store.live_vars)
        check("and the last known file ledger too",
              runner.store.live_files == "  created  a.py", runner.store.live_files)
        check("with no signals invented for it", signals == {}, str(signals))

        print("\nsecurity: a subagent return value is not an execution vector")
        sub.execute(
            "class Evil:\n"
            "    def __reduce__(self):\n"
            "        return (__import__('os').system, ('touch /tmp/claude-1000/XAGENT_PWNED',))\n"
            "done(Evil())"
        )
        refused = False
        try:
            sub.probe_pickle(DONE_EXPR)
        except Exception:
            refused = True
        check("malicious __reduce__ is refused", refused)
        check("its payload never ran", not os.path.exists("/tmp/claude-1000/XAGENT_PWNED"))
        check("plain data still crosses", sub.probe_pickle("{'a': [1, 2], 'b': 'x'}") == {"a": [1, 2], "b": "x"})

        print("\nsecurity: credentials are out of reach")
        check("reading a .env is refused", "PermissionError" in k.execute("read('.env')").render())
        leaked = k.probe(
            "import os, json; print(json.dumps([k for k in os.environ "
            "if any(h in k.upper() for h in ('TOKEN','SECRET','API_KEY','PASSWORD','CREDENTIAL'))]))")
        check("no credential-shaped vars in the kernel env", json.loads(leaked) == [],
              leaked)

        print("\nvariable table: dataclass instances were filtered out as 'tools'")
        k.execute("res = sh('echo hi'); hits = grep('def ', '*.py', 'src/xagent'); one = hits[0]")
        table = k.probe("import xagent.runtime as _r; _r._emit_var_table()")
        check("a Result instance is listed", "res" in table, table[:200])
        check("a Hit instance is listed", re.search(r"^\s*one\s", table, re.M) is not None,
              table[:200])
        check("the injected tools are still hidden", not re.search(r"^\s*grep\s", table, re.M))

        print("\ntool behaviour")
        crlf = Path("/tmp/claude-1000/xagent-crlf.txt")
        crlf.write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")
        k.execute(f"edit({str(crlf)!r}, 'beta', 'BETA')")
        check("edit() preserves CRLF line endings",
              crlf.read_bytes() == b"alpha\r\nBETA\r\ngamma\r\n", repr(crlf.read_bytes()))
        crlf.unlink(missing_ok=True)
        check("read(lines=(0, n)) is rejected rather than silently wrong",
              "ValueError" in k.execute("read('src/xagent/spawn.py', (0, 5))").render())
        # A generator cannot be inspected without advancing it; the bug was that
        # peek() drained it entirely. Now it takes only what it shows.
        check("peek() consumes only what it shows from a generator",
              "2" in k.execute("g = (i for i in range(3))\npeek(g, n=1)\nlen(list(g))").render())
        check("seed keys that would shadow a tool are refused",
              "shadow" in k.execute("agent('x', seed={'read': 'text'})").render())

        print("\nkernel: earlier display_data was overwritten by the last")
        out = k.execute("from IPython.display import display\nfor s in ['ALPHA','BETA','GAMMA']: display(s)")
        check("every display() survives", all(w in out.render() for w in ("ALPHA", "BETA", "GAMMA")),
              out.render()[:160])

        print("\ncompaction: decisions were made in mixed units")
        store = ContextStore(task="t", system="S" * 100)
        for i in range(20):
            store.add(f"f{i}()", "x", f"tu{i}")
        store.observe(30_000)          # real usage far above the raw estimate
        comp = Compressor(provider=None, budget=5000, keep_recent=8)
        comp._var_table = lambda kern: "(table)"
        comp._summarize = lambda st, t: "summary " * 40
        check("a harmful eviction is refused, not accepted on a unit mismatch",
              comp.evict(store, None) is None)
        check("the cooldown is armed even on a revert",
              comp._last_at == len(store.cells))

        tiny = ContextStore(task="t", system="S" * 100)
        for i in range(20):
            tiny.add(f"f{i}()", "x", f"tu{i}")
        check("folding cells with tiny output is refused", Compressor(
            provider=None, budget=100, keep_recent=8).fold(tiny) is None)

        big = ContextStore(task="t", system="S" * 100)
        for i in range(30):
            big.add(f"f{i}()", "out " * 400, f"tu{i}")
        c2 = Compressor(provider=None, budget=500_000, keep_recent=8)
        c2.stalled = True
        check("stalled is cleared when the budget is fine",
              not c2.should_compact(big) and not c2.stalled)

        print("\ncontext: code was permanent and entirely uncapped")
        store2 = ContextStore(task="t", system="s")
        cell = store2.add("x = '''" + "L" * 60_000 + "'''", "ok", "tu_1")
        check("an oversized code cell is elided", len(cell.code) <= MAX_CODE_CHARS + 400,
              f"{len(cell.code):,} chars")
        check("the elision says where the original lives", "In[1]" in cell.code)

        print("\nfilesystem mutations are tracked as facts, not left to summary prose")
        k.execute("import shutil, os; shutil.rmtree('/tmp/claude-1000/led', ignore_errors=True)")
        k.execute("import xagent.runtime as _r; _r._STATE['files'].clear()")
        k.execute(r"write('/tmp/claude-1000/led/a.py', 'def a(): pass\n')")
        k.execute(r"write('/tmp/claude-1000/led/b.py', 'def b(): pass\n')")
        k.execute("edit('/tmp/claude-1000/led/a.py', 'def a()', 'def alpha()')")
        ledger = k.probe("import xagent.runtime as _r; _r._emit_file_ledger()")
        check("every written file is listed",
              "a.py" in ledger and "b.py" in ledger, ledger)
        # The ledger the runner actually reads now rides the cell, not a probe.
        k.push("import xagent.runtime as _r; _r._arm_turn({}, False, 'led1')")
        carried = (k.execute("1", control_nonce="led1").control or {}).get("files", "")
        check("and the same ledger rides the turn payload",
              "a.py" in carried and "b.py" in carried, carried[:200])
        check("creation is distinguished from modification", "created" in ledger, ledger)
        check("repeat writes are counted", "2 writes" in ledger, ledger)
        check("the ledger is queryable from code",
              len(k.probe_pickle("__import__('xagent.runtime', fromlist=['x']).files_touched()")) == 2)
        check("compaction never touches the files themselves",
              k.execute("read('/tmp/claude-1000/led/a.py')").ok)

        print("\ncontext: the variable table is refreshed in the tail, not frozen")
        store3 = ContextStore(task="t", system="s")
        store3.add("a()", "o", "tu_1")
        store3.live_vars = "  findings  list  len=160"
        store3.live_files = "  created   src/new_mod.py  (900 bytes)"
        wire = json.dumps(store3.messages())
        check("live variables ride the last message",
              "len=160" in json.dumps(store3.messages()[-1]))
        check("and not the cached prefix", "len=160" not in json.dumps(store3.messages()[0]))
        check("marked as introspected now", "authoritative" in wire)
        check("the file ledger rides the tail too",
              "new_mod.py" in json.dumps(store3.messages()[-1])
              and "new_mod.py" not in json.dumps(store3.messages()[0]))

        print("\ncontext: a turn that made several calls is one assistant message,")
        print("         because the API answers a message's tool_use blocks in the")
        print("         single user message after it -- not one round each")

        def unanswered(msgs):
            """Every tool_use id in the transcript that no tool_result answers."""
            missing = []
            for i, msg in enumerate(msgs):
                if msg["role"] != "assistant":
                    continue
                asked = [b["id"] for b in msg["content"] if b["type"] == "tool_use"]
                nxt = msgs[i + 1] if i + 1 < len(msgs) else {"content": []}
                answered = {b.get("tool_use_id") for b in nxt["content"]}
                missing += [a for a in asked if a not in answered]
            return missing

        store4 = ContextStore(task="t", system="s")
        store4.add("a()", "A", "tu_a", thought="both at once", turn=1)
        store4.add("b()", "B", "tu_b", turn=1)
        store4.add("c()", "C", "tu_c", turn=2)
        msgs = store4.messages()
        check("two turns and an opening, not three exchanges", len(msgs) == 5,
              f"{len(msgs)} messages")
        batched = msgs[1]["content"]
        check("the batch carries both tool_use blocks, in order",
              [b["id"] for b in batched if b["type"] == "tool_use"] == ["tu_a", "tu_b"],
              str(batched))
        check("with the prose written once above them, not per call",
              sum(1 for b in batched if b["type"] == "text") == 1, str(batched))
        check("and both results arrive in the one user message that follows",
              [r["tool_use_id"] for r in msgs[2]["content"]] == ["tu_a", "tu_b"],
              str(msgs[2]))
        check("no tool_use goes unanswered, which the API rejects outright",
              not unanswered(msgs), str(unanswered(msgs)))
        store4.cells[0].state = "evicted"
        check("an evicted call takes its own result with it",
              not unanswered(store4.messages()), str(unanswered(store4.messages())))
        store4.cells[1].state = "folded"
        check("a folded one keeps its pair", not unanswered(store4.messages()),
              str(unanswered(store4.messages())))
        legacy = ContextStore(task="t", system="s")
        legacy.add("f1()", "o", "t1")
        legacy.add("f2()", "o", "t2")
        check("a cell added without a turn is a turn of its own, as it always was",
              len(legacy.turns(legacy.live())) == 2,
              str([c.turn for c in legacy.cells]))

        print("\nmulti-tool turn: a stray `sh` block arriving after a good `python`")
        print("                 call overwrote it, so the cell was silently discarded")

        class _Block:
            def __init__(self, name, ident, code, timeout=None):
                self.type, self.name, self.id = "tool_use", name, ident
                # The finish carries no input at all, which is what lets it be a
                # tool in the first place.
                self.input = {} if code is None else {"code": code}
                if timeout is not None:
                    self.input["timeout"] = timeout

        class _Usage:
            input_tokens = 1
            cache_read_input_tokens = 0
            cache_creation_input_tokens = 0

        class _Resp:
            stop_reason = "tool_use"
            usage = _Usage()

            def __init__(self, blocks):
                self.content = blocks

        def _prov(is_subagent=False):
            prov = Provider.__new__(Provider)
            prov.usage = Usage()
            prov.calls = 0
            prov.on_delta = None
            prov._code_seen = ""
            prov.backend = config.BACKENDS["codiv"]
            prov.model = "m"
            prov.thinking = None
            prov.sampling = "thinking"
            prov.is_subagent = is_subagent
            return prov

        def assemble(blocks, is_subagent=False):
            prov = _prov(is_subagent)
            prov._create = lambda **kw: _Resp(blocks)
            return prov._sample_once("sys", [], 1024)

        py, sh = _Block("python", "tu_py", "x = 1"), _Block("sh", "tu_sh", "ls")
        turn = assemble([py, sh])
        check("the python call is the one acted on", turn.tool_name == "python", str(turn.tool_name))
        check("its code survives the stray block", turn.code == "x = 1", repr(turn.code))
        check("its own tool_use_id is kept", turn.tool_use_id == "tu_py", str(turn.tool_use_id))
        check("the stray tool is reported, not acted on", turn.ignored_tools == ["sh"],
              str(turn.ignored_tools))

        turn = assemble([sh, py])
        check("order does not matter: python still wins", turn.code == "x = 1", repr(turn.code))
        check("and sh is still reported", turn.ignored_tools == ["sh"], str(turn.ignored_tools))

        turn = assemble([sh])
        check("a lone unknown tool is still surfaced for correction",
              turn.tool_name == "sh" and not turn.ignored_tools, str(turn.tool_name))

        print("\nparallel calls: a second `python` block was reported as a stray and")
        print("                its code discarded, so a turn that batched three")
        print("                independent steps ran one and paid for three")
        py2 = _Block("python", "tu_py2", "y = 2")
        turn = assemble([py, py2])
        check("both python calls are acted on",
              [c.code for c in turn.calls] == ["x = 1", "y = 2"],
              str([c.code for c in turn.calls]))
        check("each keeps its own id, which is what the results are matched by",
              [c.tool_use_id for c in turn.calls] == ["tu_py", "tu_py2"],
              str([c.tool_use_id for c in turn.calls]))
        check("neither is reported as a stray", turn.ignored_tools == [],
              str(turn.ignored_tools))
        check("and the scalar view still describes the first, as it always did",
              turn.code == "x = 1" and turn.tool_use_id == "tu_py", repr(turn.code))

        turn = assemble([py, sh, py2])
        check("a stray between two calls costs neither of them",
              [c.code for c in turn.calls] == ["x = 1", "y = 2"],
              str([c.code for c in turn.calls]))
        check("and is still reported once", turn.ignored_tools == ["sh"],
              str(turn.ignored_tools))

        slow = _Block("python", "tu_slow", "build()", timeout=45)
        turn = assemble([py, slow])
        check("a timeout is per call, not per turn",
              [c.timeout for c in turn.calls] == [None, 45.0],
              str([c.timeout for c in turn.calls]))

        turn = assemble([py, _Block("done", "tu_d", None), py2])
        check("a `done` block beside a batch keeps every call in it, and is a stray",
              turn.ignored_tools == ["done"]
              and [c.code for c in turn.calls] == ["x = 1", "y = 2"],
              f"ignored={turn.ignored_tools} calls={[c.code for c in turn.calls]}")
        turn = assemble([py, py2], is_subagent=True)
        check("a subagent batches the same way",
              [c.code for c in turn.calls] == ["x = 1", "y = 2"],
              str([c.code for c in turn.calls]))

        print("\nprompt: the tool contract belongs in the tool description, which")
        print("        the chat template renders inside the <tools> block")
        DESC = PYTHON_TOOL["description"]
        SUB_DESC = PYTHON_TOOL_SUBAGENT["description"]
        check("the description names python as the only tool the top level has",
              "only tool that exists" in DESC, DESC[:200])
        check("and says there is no tool for finishing either",
              "no tool for finishing" in DESC, DESC[:300])
        check("the subagent's names `python` as its only one too",
              "only tool that exists" in SUB_DESC, SUB_DESC[:200])
        check("it says the helpers are functions, not tools", "not tools" in DESC)
        check("it warns that any other tool name wastes the turn",
              "other than `python` runs nothing" in DESC)
        check("and the subagent's warns about anything but python",
              "other than `python` runs nothing" in SUB_DESC)
        check("the top level is told a turn with no call is the answer",
              "no `python` call is the answer" in DESC, DESC[-400:])
        check("the prompt does not restate the contract at length",
              "How a cell runs" not in SYSTEM)
        check("no heading offers the helpers as tools", "Tools available" not in SYSTEM)

        print("\nprompt: `AgentError` was documented but never injected, so the")
        print("        idiom the prompt taught would have raised NameError")
        PLACEHOLDERS = {
            "doc", "hits", "parts", "fs", "h", "h1", "h2", "hs", "r", "results",
            "src", "key", "text", "value", "code", "python", "_7", "_N", "seed",
            "prompt", "f", "live-variables", "files-you-have-changed", "out",
            "id_rsa", "env", "ssh", "aws", "None", "PermissionError", "datetime",
            "Decimal", "old", "new",
        }
        named = sorted(n for n in set(re.findall(r"`([A-Za-z_]\w*)`", SYSTEM))
                       if n not in PLACEHOLDERS and not keyword.iskeyword(n))
        missing = json.loads(k.probe(
            "import json, builtins; print(json.dumps([n for n in "
            f"{named!r} if n not in globals() and not hasattr(builtins, n)]))"))
        check("every symbol the prompt names in backticks is really bound",
              missing == [], f"documented but absent: {missing}")
        check("isinstance() on a failed slot works now",
              k.execute("isinstance(AgentError('l', 'm'), AgentError)").render().strip()
              == "True")
        check("the prompt teaches isinstance, not truthiness",
              "isinstance(r, AgentError)" in SYSTEM)

        print("\nprompt: invented keyword arguments (edit/grep/sh/peek/compress)")
        real = {f.__name__: f for f in runtime.PUBLIC}
        for fn, args in re.findall(r"(?<![\w.])([a-z_]\w*)\(([^)\n]*)\)", SYSTEM + DESC):
            if fn not in real:
                continue
            params = set(inspect.signature(real[fn]).parameters)
            invented = set(re.findall(r"(\w+)\s*=", args)) - params
            check(f"{fn}() is documented with real parameter names", not invented,
                  f"{sorted(invented)} are not parameters of {fn}")

        print("\nprompt: compress() was described as immediate; it is queued")
        check("compress() reports itself as queued",
              "queued" in k.execute("compress()").render())
        check("the prompt says so too",
              "queues a compaction" in SYSTEM and "after the current cell" in SYSTEM)

        print("\nprompt: done() was told to accept a dataclass, which the process")
        print("        boundary refuses -- the answer silently degrades to a string")
        sub.execute("from dataclasses import dataclass as _dc\n"
                    "@_dc\nclass Finding:\n    path: str\n"
                    "done(Finding('a.py'))")
        refused = False
        try:
            sub.probe_pickle(DONE_EXPR)
        except Exception:
            refused = True
        check("a locally defined dataclass really is refused", refused)
        check("no prompt recommends returning one",
              not any("dataclass" in p for p in (SYSTEM, SYSTEM_SUBAGENT, SUBAGENT_CODA)))
        sub.execute("done({'path': 'a.py'})")
        check("plain data still crosses", sub.probe_pickle(DONE_EXPR) == {"path": "a.py"})

        print("\nprompt: the answer to a person was a done() value, so a dict repr")
        print("        reached the terminal where sentences belonged")
        fin = SYSTEM.split("# Answering")[-1]
        flat = " ".join(fin.split())
        check("the top-level agent is told a turn with no call is the answer",
              "no tool call in that turn" in flat, flat[:400])
        check("and that a turn with a call is it still working",
              "still working" in flat and "handing back" in flat, flat[:400])
        check("it rules out handing a person a data structure",
              "not an answer to a person" in flat)
        check("it says the person may reply, so this is not a one-shot",
              "they will reply if there is more" in flat, flat[:800])
        check("and that the kernel survives the answer, so a reply resumes",
              "every variable still bound" in flat, flat[:800])
        check("the coda tells a subagent there is no turn after its done()",
              "no turn after your" in SUBAGENT_CODA, SUBAGENT_CODA[-400:])
        check("no finish tool is described to the top level",
              "`done` tool" not in flat, flat[:400])
        check("and tells the model not to type done() into the answer",
              "Never type `done()` into the answer" in flat, flat[:600])
        check("and never as a block the model can echo into its answer",
              "\n    done()\n" not in SYSTEM,
              "a standalone done() block invites the model to type it into the prose")
        subfin = " ".join(SYSTEM_SUBAGENT.split("# Finishing")[-1].split())  # noqa
        check("the subagent's prompt sends done(value) as Python instead",
              "one more line of code in the `python` tool" in subfin, subfin[:400])
        check("and explains that naming a value beats transcribing it",
              "not the capped view" in subfin, subfin[:600])
        rendered = k.execute("done()").render()
        check("a top-level done() redirects rather than ending the response",
              "no tool call" in rendered and "does not end your response" in rendered,
              rendered[:250])
        check("passing a value anyway is answered, not silently kept",
              "read by nobody" in k.execute("done({'n': 1})").render())
        rendered = sub.execute("done({'n': 1})").render()
        check("a subagent's done() reports the value it hands back",
              "'n': 1" in rendered and "does not end" not in rendered, rendered[:200])
        check("the result carries the prose separately from the value",
              "answer" in {f.name for f in dataclasses.fields(RunResult)})

        print("\nfinishing: the `done` tool cost a turn to correct whenever a model")
        print("           narrated it, so there is no such tool at either depth now")
        dn = _Block("done", "tu_d", None)
        turn = assemble([py, dn])
        check("a `done` block never competes with the cell beside it",
              turn.code == "x = 1" and turn.ignored_tools == ["done"],
              f"code={turn.code!r} ignored={turn.ignored_tools}")
        turn = assemble([dn])
        check("a lone `done` block is promoted so there is an id to answer",
              turn.tool_name == "done" and turn.tool_use_id == "tu_d",
              str(turn.tool_use_id))
        check("neither role is offered anything but python",
              [t["name"] for t in _prov(False)._tools()] == ["python"]
              and [t["name"] for t in _prov(True)._tools()] == ["python"])
        check("but the two roles get different descriptions of it",
              _prov(False)._tools()[0] is PYTHON_TOOL
              and _prov(True)._tools()[0] is PYTHON_TOOL_SUBAGENT)
        check("a turn is produced when it has a call or text, so an answer counts",
              "turn.tool_use_id or turn.text.strip()"
              in inspect.getsource(Provider.sample))
        check("prose from a refused turn is carried to the turn that ends after it",
              "carried, stranded = stranded" in inspect.getsource(Runner._loop))

        print("\ncell timeout: a fixed 180s wall meant a slow build could only")
        print("              ever be discovered by being cut off at it")
        from xagent.kernel import CELL_TIMEOUT, MAX_CELL_TIMEOUT
        from xagent.provider import _cell_timeout
        check("the tool advertises the timeout argument",
              "timeout" in PYTHON_TOOL["input_schema"]["properties"],
              str(PYTHON_TOOL["input_schema"]))
        check("and both roles advertise it, from the one shared schema",
              PYTHON_TOOL["input_schema"] is PYTHON_TOOL_SUBAGENT["input_schema"])
        check("only `code` is required, so a plain call is unchanged",
              PYTHON_TOOL["input_schema"]["required"] == ["code"])
        check("the default quoted to the model is the kernel's own",
              f"{CELL_TIMEOUT:g} seconds" in PYTHON_TOOL["description"]
              and inspect.signature(Kernel.execute).parameters["timeout"].default
                  == CELL_TIMEOUT)
        check("nothing is asked for when nothing was passed", _cell_timeout(None) is None)
        check("a request above the cap is clamped, not refused",
              _cell_timeout(10_000) == MAX_CELL_TIMEOUT)
        for junk in ("soon", -5, 0, float("nan"), [30]):
            check(f"{junk!r} is dropped, so the cell still runs under the default",
                  _cell_timeout(junk) is None)
        check("a number sent as a string is still honoured", _cell_timeout("300") == 300)
        out = k.execute("import time\ntime.sleep(2)", timeout=0.5)
        check("the notice names the limit that fired",
              "timed out after 0.5s" in out.render(), out.render())
        check("and points at the argument that raises it", "`timeout`" in out.render())

        print("\nprovider: a malformed stream event killed a subagent 1.7s into its run")
        # Seen live against SGLang: a `message_start` the SDK could not model-construct
        # left `event.message` a plain dict, and its own accumulator then called
        # `.to_dict()` on it. The subagent died with an AttributeError before it had
        # done anything, which read as a bug in the harness rather than as weather.
        import xagent.provider as provider_mod
        from xagent.provider import _sdk_stream_fault

        sdk_frame = compile("raise AttributeError(\"'dict' object has no attribute 'to_dict'\")",
                            "/site-packages/anthropic/lib/streaming/_messages.py", "exec")

        class _Stream:
            def __init__(self, fail):
                self.fail = fail

            def __enter__(self):
                if self.fail:
                    exec(sdk_frame)
                return self

            def __exit__(self, *exc):
                return False

            def __iter__(self):
                return iter(())

            def get_final_message(self):
                return "final"

        class _Messages:
            def __init__(self, failures):
                self.calls, self.failures = 0, failures

            def stream(self, **kw):
                self.calls += 1
                return _Stream(self.calls <= self.failures)

        class _Client:
            def __init__(self, failures):
                self.messages = _Messages(failures)

        class _NoSleep:
            monotonic = staticmethod(time.monotonic)
            sleep = staticmethod(lambda _s: None)

        try:
            exec(sdk_frame)
        except AttributeError as e:
            check("a fault raised inside the SDK is recognised", _sdk_stream_fault(e))
        try:
            raise AttributeError("raised right here")
        except AttributeError as e:
            check("one raised in our own code is not", not _sdk_stream_fault(e))

        real_time = provider_mod.time
        provider_mod.time = _NoSleep
        try:
            prov = Provider(backend="codiv")
            prov.client = _Client(failures=1)
            check("a malformed stream event is retried, not fatal",
                  prov._create(model="m") == "final")
            check("and it took a second request to get there", prov.client.messages.calls == 2)

            prov = Provider(backend="codiv")
            prov.client = _Client(failures=9)
            try:
                prov._create(model="m")
            except AttributeError:
                check("a stream that never recovers still surfaces", True)
            else:
                check("a stream that never recovers still surfaces", False, "swallowed")

            # An AttributeError from our own delta handling must arrive as itself,
            # on the first attempt, rather than five retries later.
            prov = Provider(backend="codiv")
            prov.client = _Client(failures=0)
            prov._relay = lambda event: (_ for _ in ()).throw(AttributeError("ours"))
            hit = _Stream(False)
            hit.__iter__ = lambda self=None: iter([object()])
            calls_before = prov.client.messages.calls
            try:
                prov.client.messages.stream = lambda **kw: hit
                prov._create(model="m")
            except AttributeError as e:
                check("our own AttributeError is not retried away", not _sdk_stream_fault(e))
            else:
                check("our own AttributeError is not retried away", True, "no fault raised")
        finally:
            provider_mod.time = real_time

        print("\ncell timeout: a cell that swallowed the interrupt kept the harness")
        print("              reading long after it had reported the cell as over")
        # Measured at 15 minutes inside zmq_poll. The drain deadline was only
        # consulted where the message queue ran dry, and a cell that goes on
        # printing never lets it -- so the bound existed and was never reached.
        import xagent.kernel as kernel_mod
        from xagent.kernel import Kernel as _Kernel
        grace = kernel_mod.DRAIN_GRACE
        kernel_mod.DRAIN_GRACE = 1.0
        stubborn = _Kernel(cwd=Path.cwd())
        try:
            started = time.monotonic()
            out = stubborn.execute(
                "import time\n"
                "while True:\n"
                "    try:\n"
                "        print('still here')\n"
                "        time.sleep(0.02)\n"
                "    except KeyboardInterrupt:\n"
                "        pass\n",
                timeout=1,
            )
            took = time.monotonic() - started
            check("the drain is a hard bound, not one checked only when idle",
                  took < 8, f"{took:.1f}s")
            check("the cell is still reported as timed out", out.timed_out)
            check("and the harness knows the interrupt did not take", not out.stopped)
            check("so the notice does not promise a quiescent namespace",
                  "did not stop it" in out.render() and "namespace survived" not in out.render(),
                  out.render()[:200])
        finally:
            kernel_mod.DRAIN_GRACE = grace
            stubborn.shutdown()
        check("a cell that does stop is still reported as stopped",
              k.execute("1 + 1").stopped)

        print("\nprompt: constants quoted in prose must match the code")
        check("the cell-elision figure matches MAX_CODE_CHARS", str(MAX_CODE_CHARS) in SYSTEM,
              f"prompt should quote {MAX_CODE_CHARS}")
        subs = SYSTEM.split("# Subagents")[-1]
        for label, value in (("depth", config.MAX_AGENT_DEPTH),
                             ("session cap", config.MAX_TOTAL_AGENTS),
                             ("concurrency", spawn.MAX_CONCURRENCY)):
            check(f"the {label} limit in the prompt matches the code", str(value) in subs,
                  f"{value} missing from the subagent section")

        print("\nprompt: an unescaped \\n once split a code example in place, so")
        print("        every indented example is compiled rather than eyeballed")
        # Strip the generated listing first: it is indented like an example but is
        # a signature table, not Python, and compiling it would always fail.
        listing = textwrap.indent(runtime.inventory(include_done=False), "    ")
        check("the generated listing is embedded verbatim", listing in SYSTEM,
              "prompts.SYSTEM no longer carries runtime.inventory()")
        check("the top-level listing does not offer done() as a helper",
              "done(" not in listing, listing[-300:])
        sub_listing = textwrap.indent(runtime.inventory(include_done=True), "    ")
        check("the subagent's listing does, because that is how it finishes",
              "done(" in sub_listing and sub_listing in SYSTEM_SUBAGENT,
              "prompts.SYSTEM_SUBAGENT no longer carries the full listing")
        body = SYSTEM.replace(listing, "")
        blocks = [textwrap.dedent(b) for b in
                  re.findall(r"\n\n((?:(?:    .*)?\n)+)", body) if b.strip()]
        check("the prompt still carries worked examples", len(blocks) >= 2, str(len(blocks)))
        for i, src in enumerate(blocks, 1):
            try:
                compile(src, f"<prompt example {i}>", "exec")
                ok, why = True, ""
            except SyntaxError as e:
                ok, why = False, f"{e.msg} at line {e.lineno}"
            check(f"code example {i} is valid Python", ok, why)
        check("triple quotes in examples are balanced", SYSTEM.count('\"\"\"') % 2 == 0)

        print("\nnamespace listing: hand-written, it was already wrong -- missing")
        print("                   optional parameters and naming an unbound class")
        listing = runtime.inventory()
        check("it lists every installed helper",
              all(o.__name__ in listing for o in runtime.PUBLIC), listing[:200])
        check("it lists nothing that is not installed",
              not (set(re.findall(r"^(\w+)\(", listing, re.M))
                   - {o.__name__ for o in runtime.PUBLIC}))
        check("no row is blank for want of a docstring",
              not re.search(r"^\s+\.\.\.$", listing, re.M), listing[:200])
        check("optional parameters are present because they are introspected",
              "max_hits=5000" in listing and "count=1" in listing and "cwd=None" in listing)
        check("helpers() prints the same listing the prompt carries",
              k.execute("helpers()").render().strip().startswith(listing.split("\n")[0]))
        check("the prompt says plainly that none of it is a tool",
              "None of them is a tool" in SYSTEM)

    finally:
        k.shutdown()
        sub.shutdown()

    print(f"\n{'─' * 60}\n{len(PASS)} passed, {len(FAIL)} failed")
    for name in FAIL:
        print(f"  FAILED: {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
