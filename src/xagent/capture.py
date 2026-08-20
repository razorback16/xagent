"""Capturing sound as it happens -- from the microphone, or from the speakers.

"What is playing right now" is the more useful of the two and the harder one to
get: it means recording the *output* of the sound server rather than an input
device. Every server calls that something different (PipeWire targets the sink
node, PulseAudio exposes a `.monitor` source), so this module resolves a working
argv for whichever one is running and hands back plain 16 kHz mono PCM16 chunks.

The chunks are the point. They go straight into `asr.SampleFeed`, which is what
lets the transcript arrive while the audio is still playing rather than after it.
"""

from __future__ import annotations

import collections
import platform
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from xagent.audio import SAMPLE_RATE, AudioError, Clip

# Aliases people actually type, mapped onto the two things a machine has.
SPEAKER = {"speaker", "speakers", "system", "output", "loopback", "monitor", "desktop"}
MIC = {"mic", "microphone", "input", "voice"}

CHUNK_MS = 200
# Any longer than this and a forgotten `listen()` is a memory leak with a UI.
MAX_SECONDS = 3600


@dataclass
class Source:
    """A resolved capture command: what it records, and how."""

    kind: str                      # speaker | mic
    device: str                    # what the sound server calls it
    tool: str                      # pw-record | parec | ffmpeg | arecord
    argv: list[str] = field(default_factory=list)
    rate: int = SAMPLE_RATE

    def __repr__(self) -> str:
        return f"<Source {self.kind} via {self.tool}: {self.device}>"


def _run(argv: list[str], timeout: float = 5.0) -> str:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout if proc.returncode == 0 else ""


def _ffmpeg_argv(fmt: str, source: str, rate: int) -> list[str]:
    """ffmpeg reading one device and writing raw mono PCM16 to stdout.

    The three platforms differ only in the input format and how the device is
    named; everything after `-i` is the contract with `stream()`.
    """
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", fmt, "-i", source, "-ar", str(rate), "-ac", "1",
            "-f", "s16le", "-"]


def _pipewire_node(kind: str) -> str:
    """The default sink's or source's node name, as PipeWire knows it.

    The class is checked rather than trusted. WirePlumber answers
    `@DEFAULT_AUDIO_SOURCE@` with the *sink* on a machine that has no capture
    device configured, and a `listen("mic")` that quietly recorded the speakers
    instead would be a wrong answer rather than a missing one.
    """
    target = "@DEFAULT_AUDIO_SINK@" if kind == "speaker" else "@DEFAULT_AUDIO_SOURCE@"
    want = "Audio/Sink" if kind == "speaker" else "Audio/Source"
    name, klass = "", ""
    for line in _run(["wpctl", "inspect", target]).splitlines():
        stripped = line.strip().lstrip("* ")
        if stripped.startswith("node.name"):
            name = stripped.split("=", 1)[1].strip().strip('"')
        elif stripped.startswith("media.class"):
            klass = stripped.split("=", 1)[1].strip().strip('"')
    return name if klass == want else ""


def _pulse_device(kind: str) -> str:
    """The default sink's monitor, or the default source."""
    if kind == "speaker":
        sink = _run(["pactl", "get-default-sink"]).strip()
        return f"{sink}.monitor" if sink else ""
    return _run(["pactl", "get-default-source"]).strip()


def resolve(source: str = "speaker", rate: int = SAMPLE_RATE) -> Source:
    """Work out how to record `source` on this machine.

    `source` is "speaker", "mic", or a device name the sound server knows -- in
    which case it is passed through to whichever tool is available.
    """
    name = (source or "speaker").strip()
    lowered = name.lower()
    kind = "mic" if lowered in MIC else "speaker"
    explicit = "" if (lowered in MIC or lowered in SPEAKER) else name

    system = platform.system()
    if system == "Darwin":
        return _darwin(kind, explicit, rate)
    if system == "Windows":
        return _windows(kind, explicit, rate)

    if shutil.which("pw-record"):
        # Targeting a *sink* node records what is being played to it, which is
        # the whole trick: no loopback device to install, no monitor to find.
        device = explicit or _pipewire_node(kind)
        if device:
            return Source(kind, device, "pw-record", [
                "pw-record", "--target", device, "--rate", str(rate),
                "--channels", "1", "--format", "s16", "--latency", f"{CHUNK_MS}ms", "-",
            ], rate)
    if shutil.which("parec"):
        device = explicit or _pulse_device(kind)
        if device:
            return Source(kind, device, "parec", [
                "parec", f"--device={device}", "--rate", str(rate),
                "--channels=1", "--format=s16le", "--latency-msec", str(CHUNK_MS),
            ], rate)
    if shutil.which("ffmpeg"):
        device = explicit or _pulse_device(kind) or _pipewire_node(kind) or "default"
        if kind == "speaker" and device != "default" and not device.endswith(".monitor"):
            device += ".monitor"
        return Source(kind, device, "ffmpeg",
                      _ffmpeg_argv("pulse", device, rate), rate)
    if kind == "mic" and shutil.which("arecord"):
        device = explicit or "default"
        return Source(kind, device, "arecord", [
            "arecord", "-D", device, "-f", "S16_LE", "-r", str(rate), "-c", "1",
            "-t", "raw", "-q",
        ], rate)
    raise AudioError(
        "no capture backend found. On Linux install one of pipewire-utils "
        "(pw-record), pulseaudio-utils (parec) or ffmpeg"
        + ("" if kind == "mic" else "; recording the speakers needs a running "
                                    "PipeWire or PulseAudio server")
    )


def _darwin(kind: str, explicit: str, rate: int) -> Source:
    if not shutil.which("ffmpeg"):
        raise AudioError("capturing audio on macOS needs ffmpeg (brew install ffmpeg)")
    if kind == "speaker" and not explicit:
        raise AudioError(
            "macOS has no system-audio device of its own. Install a loopback "
            "driver (BlackHole: `brew install blackhole-2ch`), make it the output "
            "device, and pass its name: listen('BlackHole 2ch')"
        )
    # avfoundation names its input "[video]:[audio]", so the leading colon is
    # added below and must not be in the device too -- `:default` here produced
    # `-i ::default`, which ffmpeg rejects.
    device = explicit or "default"
    return Source(kind, device, "ffmpeg",
                  _ffmpeg_argv("avfoundation", f":{device}", rate), rate)


def _windows(kind: str, explicit: str, rate: int) -> Source:
    if not shutil.which("ffmpeg"):
        raise AudioError("capturing audio on Windows needs ffmpeg on PATH")
    if not explicit:
        raise AudioError(
            "name the device to capture: a WASAPI loopback (\"Stereo Mix\") for "
            "the speakers, or the microphone. `ffmpeg -list_devices true -f "
            "dshow -i dummy` lists them"
        )
    return Source(kind, explicit, "ffmpeg",
                  _ffmpeg_argv("dshow", f"audio={explicit}", rate), rate)


def stream(source: str | Source = "speaker", *, seconds: float | None = 15,
           rate: int = SAMPLE_RATE, chunk_ms: int = CHUNK_MS,
           stop: threading.Event | None = None) -> Iterator[bytes]:
    """Yield PCM16 chunks until `seconds` elapse, `stop` is set, or the tool dies."""
    src = source if isinstance(source, Source) else resolve(source, rate)
    if seconds is not None and seconds <= 0:
        raise ValueError(f"seconds must be positive, got {seconds}")
    limit = min(seconds, MAX_SECONDS) if seconds is not None else None
    size = max(2, int(src.rate * 2 * chunk_ms / 1000)) & ~1
    try:
        proc = subprocess.Popen(src.argv, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, bufsize=0)
    except OSError as e:
        raise AudioError(f"could not start {src.tool}: {e}") from e

    # stderr is drained on a thread and kept only as a tail: a capture tool that
    # complains once a second would otherwise fill its pipe and wedge the capture.
    errors: collections.deque[bytes] = collections.deque(maxlen=32)
    threading.Thread(target=errors.extend, args=(proc.stderr,), daemon=True).start()

    started = time.monotonic()
    got = 0
    try:
        while True:
            if stop is not None and stop.is_set():
                break
            if limit is not None and time.monotonic() - started >= limit:
                break
            block = proc.stdout.read(size)
            if not block:
                break
            got += len(block)
            yield block
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        proc.stdout.close()
        proc.stderr.close()
    if not got:
        detail = b"".join(errors).decode("utf-8", "replace").strip()
        raise AudioError(
            f"{src.tool} captured nothing from {src.device}"
            + (f": {detail[-300:]}" if detail else
               " (is anything playing, and does this session have audio access?)")
        )


def record(source: str | Source = "speaker", seconds: float = 15, *,
           rate: int = SAMPLE_RATE, stop: threading.Event | None = None) -> Clip:
    """Capture into memory and return it as a clip."""
    src = source if isinstance(source, Source) else resolve(source, rate)
    raw = b"".join(stream(src, seconds=seconds, rate=rate, stop=stop))
    return Clip.from_pcm16(raw, rate=src.rate, label=f"{src.kind}:{src.device}")


def devices() -> dict:
    """What this machine can be asked to record, for a clear error message."""
    found = {"tools": [t for t in ("pw-record", "parec", "ffmpeg", "arecord")
                       if shutil.which(t)]}
    for kind in ("speaker", "mic"):
        try:
            src = resolve(kind)
            found[kind] = {"device": src.device, "tool": src.tool}
        except AudioError as e:
            found[kind] = {"error": str(e)}
    return found
