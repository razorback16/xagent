"""The warm kernel pool: that an adopted kernel is indistinguishable from a cold
one, and that no kernel outlives the process that started it.

Adoption is where this can go quietly wrong. A prewarmed kernel carries the
environment of whatever started the pool, and the per-run half -- depth, role, and
the seed file `runtime.install()` reads at bootstrap -- only lands at acquire time.
Get that ordering wrong and a subagent silently inherits its parent's data, or
finishes as though a person were waiting. Each check below names one of those.

No API calls. Run with:  uv run python tests/test_pool.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Force the pool on before anything reads the knob, and keep it small.
os.environ["XAGENT_KERNEL_POOL"] = "2"

import cloudpickle  # noqa: E402

from xagent import pool  # noqa: E402
from xagent.kernel import Kernel  # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32m✓\033[0m" if cond else "\033[31m✗\033[0m"
    print(f"  {mark} {name}" + (f"  — {detail}" if detail and not cond else ""))


def wait_for(predicate, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


def pid_of(kernel: Kernel) -> int:
    return int(kernel.probe("import os; print(os.getpid())"))


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main() -> int:
    seed_path = Path("/tmp/claude-1000/xagent-pool-seed.pkl")
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(seed_path, "wb") as fh:
        cloudpickle.dump({"seeded_value": [1, 2, 3]}, fh)

    print("warming")
    pool.ensure_started()
    check("the pool fills to its target",
          wait_for(lambda: pool.stats()["warm"] == 2), str(pool.stats()))

    print("\nan adopted kernel is a finished kernel")
    env = {**os.environ, "XAGENT_DEPTH": "1", "XAGENT_ROLE": "subagent",
           "XAGENT_SEED": str(seed_path), "XAGENT_PROVIDER": "codiv"}
    started = time.monotonic()
    k = pool.acquire(os.getcwd(), env)
    took = time.monotonic() - started
    try:
        check("acquiring a warm kernel does not pay for a cold start", took < 0.4,
              f"{took:.2f}s")
        check("the bootstrap ran, so the model surface is bound",
              k.execute("helpers()").ok)
        check("the seed was read, which only happens if env landed first",
              k.execute("seeded_value").render().strip() == "[1, 2, 3]",
              k.execute("seeded_value").render()[:120])
        check("its depth is the one it was given",
              k.probe("import xagent.spawn as s; print(s._depth())") == "1")
        check("and its role, so it finishes by value",
              k.probe("import os; print(os.environ['XAGENT_ROLE'])") == "subagent")
        check("done() really signals in it",
              "signals" in k.execute("done(1)").render()
              or k.execute("done(1)").ok)
        check("it runs in the directory it was handed",
              k.probe("import os; print(os.getcwd())") == os.getcwd())
        check("no credential-shaped variable survived the adoption",
              k.probe("import os; print([v for v in os.environ "
                      "if 'TOKEN' in v.upper() or 'SECRET' in v.upper()])") == "[]")
        check("and it is marked adopted, so it cannot be adopted twice", k.adopted)
    finally:
        k.shutdown()

    print("\nthe pool refills behind an acquire")
    check("a taken kernel is replaced", wait_for(lambda: pool.stats()["warm"] == 2),
          str(pool.stats()))

    print("\nstale state from the pool's own process is scrubbed, not inherited")
    warm_env = {k_: v for k_, v in os.environ.items() if not k_.startswith("XAGENT_SEED")}
    k = pool.acquire(os.getcwd(), {**warm_env, "XAGENT_DEPTH": "0",
                                   "XAGENT_ROLE": "top"})
    try:
        check("a seed the run did not set is gone",
              k.probe("import os; print(os.environ.get('XAGENT_SEED'))") == "None")
        check("so nothing of the previous run is bound",
              "NameError" in k.execute("seeded_value").render())
        check("and a top-level kernel's done() only redirects",
              "does not end your response" in k.execute("done()").render(),
              k.execute("done()").render()[:120])
    finally:
        k.shutdown()

    print("\nnothing outlives the pool")
    wait_for(lambda: pool.stats()["warm"] == 2)
    with pool._lock:
        pids = [pid_of(x) for x in pool._warm]
    check("warm kernels are real processes", len(pids) == 2 and all(alive(p) for p in pids),
          str(pids))
    pool.shutdown_pool()
    check("shutdown_pool drains the pool", pool.stats()["warm"] == 0, str(pool.stats()))
    check("and every warm kernel is reaped",
          wait_for(lambda: not any(alive(p) for p in pids), timeout=15),
          f"still alive: {[p for p in pids if alive(p)]}")

    print("\na kernel that warmed a pool of its own does not leak it")
    # This is the depth>0 case: the pool lives inside a kernel process, and only
    # a graceful shutdown gives its atexit a chance to run.
    pool._closing = False
    parent = Kernel(env={**os.environ, "XAGENT_KERNEL_POOL": "1"})
    try:
        parent.execute("import xagent.pool as p; p.ensure_started()")
        got = wait_for(lambda: parent.probe(
            "import xagent.pool as p; print(p.stats()['warm'])") == "1", timeout=30)
        check("the child process warms its own pool", got,
              parent.probe("import xagent.pool as p; print(p.stats())"))
        child_pids = [int(x) for x in parent.probe(
            "import xagent.pool as p; "
            "print(' '.join(str(k.probe('import os; print(os.getpid())')) "
            "for k in p._warm))").split()]
        check("whose kernels are live processes",
              bool(child_pids) and all(alive(p) for p in child_pids), str(child_pids))
    finally:
        parent.shutdown()
    check("and they die with the process that started them",
          wait_for(lambda: not any(alive(p) for p in child_pids), timeout=20),
          f"orphaned: {[p for p in child_pids if alive(p)]}")

    print("\nthe pool can be turned off")
    os.environ["XAGENT_KERNEL_POOL"] = "0"
    check("the target drops to zero", pool.target_size() == 0)
    k = pool.acquire(os.getcwd(), dict(os.environ))
    try:
        check("acquire still returns a working kernel", k.execute("1 + 1").ok)
        check("which was cold-started, not taken from a pool",
              pool.stats()["warm"] == 0, str(pool.stats()))
    finally:
        k.shutdown()
    os.environ["XAGENT_KERNEL_POOL"] = "2"
    check("and 'off' is honoured as well as 0",
          (os.environ.__setitem__("XAGENT_KERNEL_POOL", "off"), pool.target_size())[1] == 0)

    seed_path.unlink(missing_ok=True)
    print(f"\n{'─' * 60}\n{len(PASS)} passed, {len(FAIL)} failed")
    for name in FAIL:
        print(f"  \033[31mFAILED:\033[0m {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
