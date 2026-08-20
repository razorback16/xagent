"""Sampling presets and the output-length clamp.

No API calls: the request is intercepted before it leaves.

Run with:  uv run python tests/test_sampling.py
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import httpx

from xagent import config, provider as provider_mod
from xagent.provider import Provider

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32m✓\033[0m" if cond else "\033[31m✗\033[0m"
    print(f"  {mark} {name}" + (f"  — {detail}" if detail and not cond else ""))


def capture(**kw) -> dict:
    """Build a Provider and return the kwargs its next request would carry."""
    seen: dict = {}

    def spy(self, **request):
        seen.update(request)
        raise _Stop

    original, Provider._create = Provider._create, spy
    try:
        p = Provider(**kw)
        try:
            p.sample("system", [{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
        except _Stop:
            pass
    finally:
        Provider._create = original
    return seen


class _Stop(Exception):
    pass


class _FakeStream:
    """One streamed response: relays a few events, then dies or completes."""

    def __init__(self, events, boom):
        self.events, self.boom = events, boom

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        yield from self.events
        if self.boom is not None:
            raise self.boom

    def get_final_message(self):
        return "final"


def _events(code: str):
    return [SimpleNamespace(type="text", text="thinking out loud"),
            SimpleNamespace(type="input_json", snapshot={"code": code})]


def main() -> int:
    print("qwen sampling presets reach the request")
    thinking = capture(backend="codiv", sampling="thinking")
    check("thinking: temperature 1.0", thinking.get("temperature") == 1.0, str(thinking.get("temperature")))
    check("thinking: top_p 0.95", thinking.get("top_p") == 0.95)
    check("thinking: top_k 20", thinking.get("top_k") == 20)
    check("thinking: min_p 0.0 via extra_body", (thinking.get("extra_body") or {}).get("min_p") == 0.0)
    check("thinking: presence_penalty 0.0",
          (thinking.get("extra_body") or {}).get("presence_penalty") == 0.0)
    check("thinking: repetition_penalty 1.0",
          (thinking.get("extra_body") or {}).get("repetition_penalty") == 1.0)

    instruct = capture(backend="codiv", sampling="instruct")
    check("instruct: temperature 0.7", instruct.get("temperature") == 0.7)
    check("instruct: top_p 0.80", instruct.get("top_p") == 0.80)
    check("instruct: top_k 20", instruct.get("top_k") == 20)
    check("instruct: presence_penalty 1.5",
          (instruct.get("extra_body") or {}).get("presence_penalty") == 1.5)

    check("thinking is the default preset",
          capture(backend="codiv").get("temperature") == 1.0)

    print("\nthe Anthropic API gets none of them (it rejects every one)")
    an = capture(backend="anthropic")
    check("no temperature", "temperature" not in an, str(an.get("temperature")))
    check("no top_p", "top_p" not in an)
    check("no top_k", "top_k" not in an)
    check("no extra_body payload", not an.get("extra_body"), str(an.get("extra_body")))

    print("\nextended thinking rides alongside the sampling preset")
    et = capture(backend="codiv", thinking="high", sampling="thinking")
    # SGLang accepts both together (verified: 200 on thinking + top_p/top_k), and it
    # is the only backend that is sent sampling at all -- so thinking being on by
    # default must not cost it its preset.
    check("temperature survives thinking", et.get("temperature") == 1.0, str(et.get("temperature")))
    check("top_p survives thinking", et.get("top_p") == 0.95)
    check("top_k survives thinking", et.get("top_k") == 20)
    check("thinking block sent", et.get("thinking", {}).get("type") == "adaptive")
    check("carrying the level as effort",
          (et.get("output_config") or {}).get("effort") == "high",
          str(et.get("output_config")))
    check("extra_body survives alongside thinking",
          (et.get("extra_body") or {}).get("min_p") == 0.0)

    print("\nthinking is on by default, and `off` is the only way out")
    # Each backend states the same two things in its own shape: an unspecified
    # level is the default level, and `off` really means none.
    for backend, level_of, off_is in (
        ("anthropic", lambda kw: (kw.get("output_config") or {}).get("effort"),
         lambda kw: kw.get("thinking") == {"type": "disabled"}),
        ("codiv", lambda kw: (kw.get("output_config") or {}).get("effort"),
         lambda kw: kw.get("thinking") == {"type": "disabled"}),
    ):
        default = capture(backend=backend)
        check(f"{backend}: unspecified → the default level",
              level_of(default) == config.DEFAULT_THINKING,
              f"{default.get('thinking')} {default.get('output_config')}")
        off = capture(backend=backend, thinking="off")
        check(f"{backend}: `off` asks for no thinking", off_is(off),
              str(off.get("thinking")))
    check("default level is medium", config.DEFAULT_THINKING == "medium")
    check("`off` resolves to no level", config.resolve_thinking("off") is None)
    check("unspecified resolves to the default",
          config.resolve_thinking(None) == config.DEFAULT_THINKING)
    check("a named level resolves to itself", config.resolve_thinking("low") == "low")
    check("`off` is offered on the command line", "off" in config.THINKING_CHOICES)

    print("\n128k output ceiling")
    for backend in ("anthropic", "codiv"):
        check(f"{backend}: configured at exactly 128,000",
              config.BACKENDS[backend].max_output == 128_000,
              f"{config.BACKENDS[backend].max_output:,}")
    check("anthropic requests 128,000", an.get("max_tokens") == 128_000, str(an.get("max_tokens")))
    check("codiv requests 128,000 on a small context",
          thinking.get("max_tokens") == 128_000, str(thinking.get("max_tokens")))

    print("\nthe clamp keeps input + output inside a hard window")
    p_codiv, p_an = Provider(backend="codiv"), Provider(backend="anthropic")
    window = config.BACKENDS["codiv"].total_window
    for ctx_chars in (4_000, 400_000, 800_000):
        approx_input = ctx_chars // 4
        clamped = p_codiv._clamp(128_000, "sys", [{"role": "user", "content": "x" * ctx_chars}])
        check(f"codiv: {approx_input:,}-token context → {clamped:,} output, total under window",
              approx_input + clamped <= window, f"{approx_input + clamped:,} > {window:,}")

    try:
        p_codiv._clamp(128_000, "s", [{"role": "user", "content": "x" * 4_000_000}])
        check("codiv: a context with no room raises instead of overflowing", False,
              "returned a budget instead of raising")
    except RuntimeError as e:
        check("codiv: a context with no room raises instead of overflowing",
              "too large" in str(e), str(e)[:80])
    check("anthropic: no window, so no clamping",
          p_an._clamp(128_000, "s", [{"role": "user", "content": "x" * 800_000}]) == 128_000)
    check("clamp never exceeds the backend ceiling",
          p_codiv._clamp(999_999, "s", [{"role": "user", "content": "hi"}]) == 128_000)

    print("\nthinking depth: a token budget is silently ignored by the current")
    print("        models, so every level ran at the server default")
    from typing import get_args, get_type_hints
    from anthropic.types.output_config_param import OutputConfigParam

    allowed = set(get_args(get_args(get_type_hints(OutputConfigParam)["effort"])[0]))
    check("every level we offer is an effort level the SDK accepts",
          set(config.THINKING_LEVELS) <= allowed,
          f"{set(config.THINKING_LEVELS) - allowed} unknown to the SDK")
    for level in config.THINKING_LEVELS:
        kw = capture(backend="anthropic", thinking=level)
        ok = (kw.get("thinking", {}).get("type") == "adaptive"
              and (kw.get("output_config") or {}).get("effort") == level)
        check(f"anthropic/{level}: adaptive, with effort carrying the level", ok,
              f"thinking={kw.get('thinking')} output_config={kw.get('output_config')}")
        check(f"anthropic/{level}: no budget_tokens, which these models dropped",
              "budget_tokens" not in kw.get("thinking", {}), str(kw.get("thinking")))
    check("thinking is streamed back readably, not omitted",
          capture(backend="anthropic")["thinking"].get("display") == "summarized")
    off = capture(backend="anthropic", thinking="off")
    check("`off` is said out loud, because these models think by default",
          off.get("thinking") == {"type": "disabled"}, str(off.get("thinking")))
    check("and carries no effort, which a disabled turn may not pair with",
          off.get("output_config") is None, str(off.get("output_config")))
    # qwen takes the same shape: SGLang maps output_config.effort onto
    # reasoning_effort, and logs-and-drops budget_tokens as unenforceable.
    for level in config.THINKING_LEVELS:
        kw = capture(backend="codiv", thinking=level)
        check(f"codiv/{level}: the same effort shape, which SGLang does read",
              kw.get("thinking", {}).get("type") == "adaptive"
              and (kw.get("output_config") or {}).get("effort") == level,
              f"thinking={kw.get('thinking')} output_config={kw.get('output_config')}")
    check("no backend is sent a budget it would silently drop",
          not any("budget_tokens" in str(capture(backend=b, thinking=lv).get("thinking"))
                  for b in ("anthropic", "codiv") for lv in config.THINKING_LEVELS))

    print("\na stream that dies while being read: the SDK wraps what happens")
    print("        sending a request, not what happens reading one back")
    p = Provider(backend="codiv")
    relayed, attempts = [], []
    p.on_delta = lambda part, text: relayed.append((part, text))
    booms = [httpx.RemoteProtocolError("peer closed connection without sending "
                                       "complete message body"), None]

    def fake_stream(**kw):
        boom = booms[len(attempts)]
        attempts.append(kw)
        return _FakeStream(_events("x = 1"), boom)

    p.client = SimpleNamespace(messages=SimpleNamespace(stream=fake_stream))
    slept, provider_mod.time.sleep = [], lambda s: slept.append(s)
    try:
        got = p._create(model="m", max_tokens=10, system=[], messages=[])
    finally:
        provider_mod.time.sleep = __import__("time").sleep
    check("an incomplete chunked read is retried, not raised", got == "final",
          repr(got))
    check("and the second attempt reissues the same request", len(attempts) == 2,
          str(len(attempts)))
    check("the operator is told, so a retry does not look like a hang",
          any(part == "retry" for part, _ in relayed), repr(relayed))
    check("it waited before reissuing", slept == [2.0], str(slept))
    code = "".join(text for part, text in relayed if part == "code")
    check("the retried turn relays its code in full, not diffed against the "
          "dead attempt's", code == "x = 1x = 1", repr(code))

    p2 = Provider(backend="codiv")
    dead = [httpx.RemoteProtocolError("closed") for _ in range(5)]
    tries = []

    def always_fail(**kw):
        tries.append(kw)
        return _FakeStream([], dead[len(tries) - 1])

    p2.client = SimpleNamespace(messages=SimpleNamespace(stream=always_fail))
    provider_mod.time.sleep = lambda s: None
    try:
        p2._create(model="m", max_tokens=10, system=[], messages=[])
        check("a stream that never comes back is raised, not retried forever",
              False, "returned instead of raising")
    except httpx.RemoteProtocolError:
        check("a stream that never comes back is raised, not retried forever",
              len(tries) == 5, f"{len(tries)} attempts")
    finally:
        provider_mod.time.sleep = __import__("time").sleep

    print(f"\n{'─' * 60}\n{len(PASS)} passed, {len(FAIL)} failed")
    for name in FAIL:
        print(f"  FAILED: {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
