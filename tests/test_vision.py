"""Vision input formatting and budgeting, with no API calls."""

from __future__ import annotations

import base64
import sys
import tempfile
from pathlib import Path

from xagent.context import ContextStore
from xagent.kernel import CellOutput, Kernel
from xagent.provider import Provider
from xagent.vision import IMAGE_TOKEN_ESTIMATE, ImageAttachment, normalize_images

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32m✓\033[0m" if cond else "\033[31m✗\033[0m"
    print(f"  {mark} {name}" + (f"  — {detail}" if detail and not cond else ""))


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "screen.PNG"
        # A real 1x1 PNG lets the integration assertion below exercise IPython's
        # display_data path rather than only testing a hand-built message.
        raw = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
        path.write_bytes(raw)

        image = ImageAttachment.from_path(path)
        block = image.content_block()
        check("loads a supported image", image.media_type == "image/png")
        check("encodes an Anthropic image block", block["type"] == "image")
        check("preserves the original bytes",
              base64.b64decode(block["source"]["data"]) == raw)
        check("normalizes a single path", normalize_images(path) == [image])

        store = ContextStore(task="inspect the screenshot", system="system", images=[image])
        messages = store.messages()
        opening = messages[0]["content"]
        check("puts the image in the opening user message",
              sum(block.get("type") == "image" for block in opening) == 1)
        check("keeps the task text alongside the image",
              "inspect the screenshot" in opening[0]["text"])
        check("keeps the cache marker on a text block",
              opening[-1]["type"] == "text" and "cache_control" in opening[-1])
        check("accounts for image tokens without base64 inflation",
              store.estimated_tokens() >= IMAGE_TOKEN_ESTIMATE)

        provider = Provider(backend="codiv")
        huge = [{"role": "user", "content": [image.content_block()]}]
        check("request budgeting does not count base64 as text",
              provider._clamp(128_000, "system", huge) == 128_000)

        displayed = CellOutput(images=[block])
        rendered = displayed.render()
        check("REPL image output gets a textual marker", "1 image displayed" in rendered)
        check("REPL rendering never leaks base64 into text", block["source"]["data"] not in rendered)
        store.add("display(image)", rendered, "tu-image", images=displayed.images)
        result_content = store.messages()[-1]["content"][0]["content"]
        check("REPL image output becomes tool-result content",
              isinstance(result_content, list)
              and any(item.get("type") == "image" for item in result_content))

        kernel = Kernel(cwd=Path.cwd())
        try:
            code = (
                "from IPython.display import Image, display\n"
                f"display(Image(data={raw!r}, format='png'))"
            )
            output = kernel.execute(code)
            check("IPython display_data captures raster output", len(output.images) == 1,
                  str(output.images))
            check("captured display data is a PNG block",
                  output.images and output.images[0]["source"]["media_type"] == "image/png")
        finally:
            kernel.shutdown()

        bad = Path(td) / "not-an-image.png"
        bad.write_text("not an image")
        try:
            ImageAttachment.from_path(bad)
        except ValueError as e:
            check("rejects invalid image bytes", "valid image" in str(e))
        else:
            check("rejects invalid image bytes", False, "accepted invalid bytes")

        try:
            ImageAttachment.from_path(Path(td) / "missing.jpg")
        except FileNotFoundError:
            check("reports a missing image path", True)
        else:
            check("reports a missing image path", False, "did not raise")

    print(f"\n{'─' * 60}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print(f"  FAILED: {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
