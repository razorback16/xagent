"""Audio analysis, rendering and attachment, with no API calls and no model.

Everything here runs on synthesized sound, so the suite proves the parts the
harness owns: decoding, the numbers, the picture, and the path a clip takes into
a request. Transcription is not exercised -- it needs weights, and the visual
path deliberately does not depend on it.
"""

from __future__ import annotations

import array
import base64
import math
import struct
import sys
import tempfile
from pathlib import Path

from xagent import asr, capture, plot
from xagent.audio import (
    AudioAttachment, AudioError, Clip, Sound, SAMPLE_RATE, decode, envelope,
    frame_levels, from_pcm16, normalize_audio, resample, spectrogram, to_wav,
)
from xagent.context import ContextStore
from xagent.kernel import Kernel
from xagent.vision import IMAGE_TOKEN_ESTIMATE

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32m✓\033[0m" if cond else "\033[31m✗\033[0m"
    print(f"  {mark} {name}" + (f"  — {detail}" if detail and not cond else ""))


def tone(hz: float, seconds: float, amp: float = 0.3, rate: int = SAMPLE_RATE):
    return array.array("f", (amp * math.sin(2 * math.pi * hz * i / rate)
                             for i in range(int(rate * seconds))))


def png_size(data: bytes) -> tuple[int, int]:
    return struct.unpack(">II", data[16:24])


def main() -> int:
    print("\nraster canvas")
    c = plot.Canvas(120, 40)
    c.text(2, 2, "hz 1k -20 dbfs")
    c.set(-5, -5, plot.FG)                       # off-canvas writes are dropped
    c.set(999, 999, plot.FG)
    png = c.to_png()
    check("canvas emits a PNG", png.startswith(b"\x89PNG\r\n\x1a\n"))
    check("PNG header carries the size", png_size(png) == (120, 40), str(png_size(png)))
    check("text width is measurable", plot.Canvas.text_width("abc", 2) == 34,
          str(plot.Canvas.text_width("abc", 2)))
    check("the ramp is ordered light-to-dark", sum(plot.heat(1.0)) > sum(plot.heat(0.0)))
    check("the ramp clamps outside 0..1", plot.heat(-3) == plot.heat(0.0)
          and plot.heat(9) == plot.heat(1.0))

    print("\ndecoding")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "tone.wav"
        path.write_bytes(to_wav(tone(1000, 1.0), SAMPLE_RATE))
        samples = decode(path)
        check("decodes a wav to the model's rate", abs(len(samples) - SAMPLE_RATE) <= 2,
              str(len(samples)))
        check("decoded amplitude survives the round trip",
              0.28 < max(samples) < 0.32, str(max(samples)))
        check("pcm16 bytes decode to floats",
              abs(max(from_pcm16(to_wav(tone(440, 0.1))[44:])) - 0.3) < 0.01)
        check("resampling halves the sample count",
              abs(len(resample(tone(440, 1.0), 16000, 8000)) - 8000) <= 1)

        print("\nmeasurement")
        loud = Clip(samples=tone(1000, 2.0, amp=0.5), rate=SAMPLE_RATE, label="loud")
        quiet = Clip(samples=array.array("f", [0.0] * SAMPLE_RATE), rate=SAMPLE_RATE,
                     label="silence")
        s, q = loud.stats(), quiet.stats()
        check("duration comes from the sample count", abs(s["seconds"] - 2.0) < 0.01)
        check("peak is reported in dBFS", abs(s["peak_dbfs"] - (-6.0)) < 0.3,
              str(s["peak_dbfs"]))
        check("a tone is entirely active", s["active"] > 0.99, str(s["active"]))
        check("silence is entirely silent", q["silence"] > 0.99 and q["active"] < 0.01,
              str(q))
        check("clipping is detected",
              Clip(array.array("f", [1.0] * 1000), SAMPLE_RATE).stats()["clipped"] > 0.9)

        print("\nspectrum")
        cols = spectrogram(tone(1000, 1.0), cols=8)
        check("one spectrum per requested column", len(cols) == 8, str(len(cols)))
        loudest = max(range(len(cols[0])), key=lambda i: cols[0][i])
        peak_hz = loudest * (SAMPLE_RATE / 2) / (len(cols[0]) - 1)
        check("a 1 kHz tone peaks at 1 kHz", abs(peak_hz - 1000) < 60, f"{peak_hz:.0f} Hz")
        check("the floor is far below the peak",
              cols[0][loudest] - min(cols[0]) > 40, str(cols[0][loudest] - min(cols[0])))
        check("envelope has one entry per column", len(envelope(tone(440, 1.0), 30)) == 30)
        check("envelope brackets the signal",
              all(lo <= 0 <= hi for lo, hi, _ in envelope(tone(440, 1.0), 30)))

        print("\nthe picture")
        image = loud.png(width=700)
        check("the panel is a PNG", image.startswith(b"\x89PNG\r\n\x1a\n"))
        check("the panel is as wide as asked", png_size(image)[0] == 700, str(png_size(image)))
        check("a very short clip still renders", Clip(array.array("f", [0.1] * 5)).png())
        check("an empty clip does not crash", Clip(array.array("f", [])).png())

        print("\nattachment")
        att = AudioAttachment.from_path(path)
        block = att.image_block()
        check("attaches an image block", block["type"] == "image"
              and block["source"]["media_type"] == "image/png")
        check("the block carries decodable bytes",
              base64.b64decode(block["source"]["data"]).startswith(b"\x89PNG"))
        check("the text block states the measurements",
              "1.0s" in att.text_block() and "dBFS" in att.text_block(), att.text_block())
        check("no transcript is claimed when ASR did not run",
              "transcript" not in att.text_block().lower(), att.text_block())
        check("budgeting counts the panel as an image", att.tokens() > IMAGE_TOKEN_ESTIMATE)
        check("normalizes a single path", len(normalize_audio(path)) == 1)
        check("passes an attachment through", normalize_audio([att]) == [att])
        check("normalizes nothing to nothing", normalize_audio(None) == [])

        try:
            AudioAttachment.from_path(Path(td) / "clip.txt")
        except ValueError as e:
            check("rejects a non-audio extension", "unsupported audio type" in str(e))
        else:
            check("rejects a non-audio extension", False, "accepted .txt")
        try:
            AudioAttachment.from_path(Path(td) / "missing.wav")
        except FileNotFoundError:
            check("reports a missing audio path", True)
        else:
            check("reports a missing audio path", False, "did not raise")
        empty = Path(td) / "empty.wav"
        empty.write_bytes(to_wav(array.array("f", []), SAMPLE_RATE))
        try:
            AudioAttachment.from_path(empty)
        except AudioError:
            check("rejects a file with no audio in it", True)
        else:
            check("rejects a file with no audio in it", False, "accepted an empty clip")

        print("\nthe request")
        store = ContextStore(task="what is playing?", system="system", audio=[att])
        opening = store.messages()[0]["content"]
        check("puts the panel in the opening user message",
              sum(b.get("type") == "image" for b in opening) == 1)
        check("describes the panels in text",
              any("spectrogram" in b.get("text", "") for b in opening))
        check("keeps the cache marker on a text block",
              opening[-1]["type"] == "text" and "cache_control" in opening[-1])
        check("budget includes the attached clip",
              store.estimated_tokens() >= IMAGE_TOKEN_ESTIMATE)

        print("\nthe model's handle")
        sound = Sound(clip=loud, source="speaker")
        cut = sound.zoom(0.5, 1.0, show=False)
        check("zoom slices the samples", abs(cut.seconds - 0.5) < 0.01, str(cut.seconds))
        check("zoom clamps past the end", sound.zoom(1.9, 99, show=False).seconds <= 0.2)
        check("zoom keeps its offset", abs(cut.started - 0.5) < 0.01)
        check("repr stays one line and short", "\n" not in repr(sound) and len(repr(sound)) < 120,
              repr(sound))
        check("saves audio as wav", "wrote" in sound.save(Path(td) / "out.wav")
              and (Path(td) / "out.wav").stat().st_size > 1000)
        check("saves the panel as png",
              (sound.save(Path(td) / "out.png"),
               (Path(td) / "out.png").read_bytes().startswith(b"\x89PNG"))[1])
        check("show() outside a kernel is a no-op that returns itself",
              sound.show() is sound)

        print("\nin the kernel")
        kernel = Kernel(cwd=Path.cwd())
        try:
            out = kernel.execute(f"s = listen({str(path)!r})\ns")
            check("listen() attaches its panel to the cell", len(out.images) == 1,
                  out.render()[:200])
            check("the attached panel is a PNG",
                  out.images and out.images[0]["source"]["media_type"] == "image/png")
            check("the cell reports the sound in one line", "<Sound" in out.render(),
                  out.render()[:200])
            check("the samples stay in the kernel",
                  "16000" in kernel.execute("len(s.samples)").render())
            out = kernel.execute("s.zoom(0.2, 0.4)")
            check("zoom renders again without re-reading the file", len(out.images) == 1)
        finally:
            kernel.shutdown()

    print("\ncapture")
    for kind in ("speaker", "mic"):
        try:
            src = capture.resolve(kind)
            ok = (src.kind == kind and isinstance(src.argv, list)
                  and all(isinstance(a, str) for a in src.argv)
                  and str(SAMPLE_RATE) in src.argv)
            check(f"resolves a {kind} source to a command", ok, repr(src.argv))
        except AudioError as e:
            # A headless box has no sound server, and saying so is the correct
            # outcome -- but it must say so rather than record nothing.
            check(f"resolves a {kind} source to a command", "install" in str(e).lower()
                  or "device" in str(e).lower(), str(e))
    try:
        next(capture.stream("speaker", seconds=0))
    except ValueError as e:
        check("rejects a zero-length capture", "positive" in str(e))
    except AudioError:
        check("rejects a zero-length capture", True)
    else:
        check("rejects a zero-length capture", False, "captured for no time")
    check("device report names the tools it found",
          isinstance(capture.devices().get("tools"), list))

    print("\nthe shapes the refactor pinned down")
    clip = Clip(tone(440, 1.0), SAMPLE_RATE, "shared.wav")
    first = clip.levels()
    check("frame levels are computed once and shared",
          clip.levels() is first and first == frame_levels(clip.samples, clip.rate))
    s = clip.stats()
    counted = sum(1 for lv in first if lv >= s["gate_dbfs"]) / len(first)
    check("the active fraction is counted against those same frames",
          abs(s["active"] - counted) < 1e-12, f"{s['active']} vs {counted}")
    try:
        from_pcm16(b"\x00\x00", SAMPLE_RATE)                    # type: ignore[call-arg]
    except TypeError:
        check("from_pcm16 takes no rate it would ignore", True)
    else:
        check("from_pcm16 takes no rate it would ignore", False, "accepted one")
    argv = capture._ffmpeg_argv("pulse", "sink.monitor", 16000)
    check("every ffmpeg capture writes raw mono PCM16 to stdout",
          argv[-7:] == ["-ar", "16000", "-ac", "1", "-f", "s16le", "-"]
          and argv[argv.index("-i") + 1] == "sink.monitor", str(argv))
    mac = capture._darwin("mic", "", 16000)
    check("the macOS default device is not doubly prefixed",
          "::default" not in mac.argv and ":default" in mac.argv, str(mac.argv))

    print("\nspeech recognition (postponed — the module must still behave)")
    check("no backend is claimed when none is installed",
          asr.pick_backend() in (None, "transformers", "gguf"), str(asr.pick_backend()))
    feed = asr.SampleFeed(SAMPLE_RATE)
    feed.push([0.1] * 100)
    check("a feed serves what it holds", len(feed.get(0, 50)) == 50)
    feed.close()
    check("a closed feed serves a short tail", len(feed.get(90, 200)) == 10)
    check("a closed feed ends with None", feed.get(100, 110) is None)
    text, langs = asr._clean("Hello there.<en-US> ok")
    check("language tags are read and removed",
          langs == ["en-US"] and "<" not in text, f"{text!r} {langs}")
    check("importing asr does not import torch", "torch" not in sys.modules)

    print(f"\n{'─' * 60}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print(f"  FAILED: {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
