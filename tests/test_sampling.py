"""Sampling presets and the output-length clamp.

No API calls: the request is intercepted before it leaves.

Run with:  uv run python tests/test_sampling.py
"""

from __future__ import annotations

import sys

from xagent import config
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

    print("\nextended thinking displaces the sampling distribution")
    et = capture(backend="codiv", thinking="high", sampling="thinking")
    check("temperature dropped when thinking is on", "temperature" not in et)
    check("top_p dropped when thinking is on", "top_p" not in et)
    check("top_k dropped when thinking is on", "top_k" not in et)
    check("thinking block sent", et.get("thinking", {}).get("type") == "enabled")
    check("budget_tokens is the 'high' budget",
          et["thinking"]["budget_tokens"] == config.THINKING_BUDGETS["high"],
          str(et.get("thinking")))
    check("budget_tokens stays under max_tokens",
          et["thinking"]["budget_tokens"] < et["max_tokens"])
    check("extra_body survives alongside thinking", bool(et.get("extra_body")))

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

    print(f"\n{'─' * 60}\n{len(PASS)} passed, {len(FAIL)} failed")
    for name in FAIL:
        print(f"  FAILED: {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
