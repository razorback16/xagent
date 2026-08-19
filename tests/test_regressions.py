"""Regressions for defects found in review. Each check names the bug it pins down.

Run with:  uv run python tests/test_regressions.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from xagent.compress import Compressor
from xagent.context import MAX_CODE_CHARS, ContextStore
from xagent.kernel import Kernel

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32m✓\033[0m" if cond else "\033[31m✗\033[0m"
    print(f"  {mark} {name}" + (f"  — {detail}" if detail and not cond else ""))


def main() -> int:
    k = Kernel()
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
        k.execute("note('big', 'X' * 8000)")
        k.execute("done({'answer': 42, 'blob': 'B' * 50000})")
        signals = json.loads(k.probe("import xagent.runtime as _r; _r._emit_signals()"))
        check("a large note leaves signals parseable", signals["done"] is True)
        check("the note survives whole", len(signals["notes"]["big"]) == 8000,
              str(len(signals["notes"]["big"])))
        from xagent.runner import DONE_EXPR

        value = k.probe_pickle(DONE_EXPR)
        check("a large done() value crosses intact",
              value["answer"] == 42 and len(value["blob"]) == 50000)
        check("shadowing print does not break the control channel",
              (k.execute("print = lambda *a, **kw: None"),
               json.loads(k.probe("import xagent.runtime as _r; _r._emit_signals()"))["done"])[1])
        k.execute("del print")

        print("\nsecurity: a subagent return value is not an execution vector")
        k.execute(
            "class Evil:\n"
            "    def __reduce__(self):\n"
            "        return (__import__('os').system, ('touch /tmp/claude-1000/XAGENT_PWNED',))\n"
            "done(Evil())"
        )
        refused = False
        try:
            k.probe_pickle(DONE_EXPR)
        except Exception:
            refused = True
        check("malicious __reduce__ is refused", refused)
        check("its payload never ran", not os.path.exists("/tmp/claude-1000/XAGENT_PWNED"))
        check("plain data still crosses", k.probe_pickle("{'a': [1, 2], 'b': 'x'}") == {"a": [1, 2], "b": "x"})

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

    finally:
        k.shutdown()

    print(f"\n{'─' * 60}\n{len(PASS)} passed, {len(FAIL)} failed")
    for name in FAIL:
        print(f"  FAILED: {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
