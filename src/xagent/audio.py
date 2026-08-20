"""Audio as something the model can both see and read.

A vision model cannot listen. What it can do is look, so a clip enters the context
window twice: once as a picture of the sound -- waveform, spectrogram, and the
distribution of levels in it -- and once as the text a local ASR model streamed out
of it. The picture carries what a transcript throws away (silence, music, noise,
clipping, who is loud and when); the transcript carries the words.

Decoding is ffmpeg's job when ffmpeg is there and the standard library's when the
file is a plain WAV, so the common case needs nothing installed. Everything below
works on 16 kHz mono float samples, which is what the ASR model wants anyway.
"""

from __future__ import annotations

import array
import io
import math
import shutil
import subprocess
import wave
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from xagent import plot
from xagent.plot import ACCENT, BG, DIM, FG, GRID, PANEL, WARM, Canvas

# What the ASR model is trained on, so resampling happens once, here.
SAMPLE_RATE = 16_000

SUPPORTED_AUDIO_TYPES = {
    ".aac", ".aif", ".aiff", ".au", ".caf", ".flac", ".m4a", ".mka", ".mp3",
    ".mp4", ".oga", ".ogg", ".opus", ".wav", ".webm", ".wma",
}

# Silence is not zero, it is quiet. Anything under this is floor.
SILENCE_DBFS = -55.0
# A frame is "active" if it stands out of this clip's own floor, rather than
# against a fixed threshold that would call a quiet recording empty.
ACTIVE_MARGIN_DB = 25.0

FRAME_MS = 20.0
MAX_STAT_FRAMES = 6000


class AudioError(RuntimeError):
    """Decoding or capture failed in a way the caller should see whole."""


def db(x: float, floor: float = -120.0) -> float:
    return floor if x <= 1e-12 else max(floor, 20.0 * math.log10(x))


# ----------------------------------------------------------------- decoding


def _ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def decode(path: str | Path, rate: int = SAMPLE_RATE) -> array.array:
    """Decode any audio file to mono float32 at `rate`."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"audio file does not exist: {p}")
    if not p.is_file():
        raise AudioError(f"audio path is not a file: {p}")
    exe = _ffmpeg()
    if exe is None:
        return _decode_wav(p, rate)
    proc = subprocess.run(
        [exe, "-v", "error", "-nostdin", "-i", str(p), "-map", "0:a:0",
         "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(rate), "-"],
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        hint = detail[-1] if detail else f"exit {proc.returncode}, no audio decoded"
        raise AudioError(f"ffmpeg could not decode {p}: {hint}")
    return array.array("f", proc.stdout)


def _decode_wav(p: Path, rate: int) -> array.array:
    """The no-ffmpeg path: uncompressed WAV only, resampled by nearest neighbour."""
    try:
        with wave.open(str(p), "rb") as wf:
            channels, width = wf.getnchannels(), wf.getsampwidth()
            src_rate, frames = wf.getframerate(), wf.getnframes()
            raw = wf.readframes(frames)
    except wave.Error as e:
        raise AudioError(
            f"cannot decode {p} without ffmpeg ({e}); install ffmpeg for "
            f"compressed formats"
        ) from e
    mono = _pcm_to_float(raw, width, channels)
    return resample(mono, src_rate, rate)


def _pcm_to_float(raw: bytes, width: int, channels: int) -> array.array:
    if width == 1:                                    # WAV 8-bit is unsigned
        vals = array.array("f", ((b - 128) / 128.0 for b in raw))
    elif width == 2:
        ints = array.array("h")
        ints.frombytes(raw[:len(raw) - len(raw) % 2])
        vals = array.array("f", (v / 32768.0 for v in ints))
    elif width == 3:
        vals = array.array("f")
        for i in range(0, len(raw) - 2, 3):
            v = int.from_bytes(raw[i:i + 3], "little", signed=True)
            vals.append(v / 8388608.0)
    elif width == 4:
        ints = array.array("i")
        ints.frombytes(raw[:len(raw) - len(raw) % 4])
        vals = array.array("f", (v / 2147483648.0 for v in ints))
    else:
        raise AudioError(f"unsupported WAV sample width: {width} bytes")
    if channels > 1:
        n = len(vals) // channels
        vals = array.array("f", (
            sum(vals[i * channels:(i + 1) * channels]) / channels for i in range(n)
        ))
    return vals


def from_pcm16(raw: bytes) -> array.array:
    """Signed 16-bit little-endian PCM -- what every capture backend emits.

    Rate-free on purpose: this is a change of units, not of timebase, and a
    `rate` argument here could only ever be ignored or believed wrongly.
    """
    ints = array.array("h")
    ints.frombytes(raw[:len(raw) - len(raw) % 2])
    return array.array("f", (v / 32768.0 for v in ints))


def resample(samples: Sequence[float], src: int, dst: int) -> array.array:
    if src == dst or not len(samples):
        return samples if isinstance(samples, array.array) else array.array("f", samples)
    n = int(len(samples) * dst / src)
    step = src / dst
    return array.array("f", (samples[min(len(samples) - 1, int(i * step))] for i in range(n)))


def to_wav(samples: Sequence[float], rate: int = SAMPLE_RATE) -> bytes:
    """A WAV container, for the backends that want a file rather than an array."""
    pcm = array.array("h", (int(max(-1.0, min(1.0, s)) * 32767) for s in samples))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------- dsp


def _numpy():
    try:
        import numpy

        return numpy
    except ImportError:
        return None


def _fft(re: list[float], im: list[float]) -> None:
    """In-place iterative radix-2 FFT, for when numpy is not installed.

    Only ever called on one window at a time and on at most `cols` windows, which
    is what keeps the no-numpy path a second rather than a minute: the number of
    transforms is bounded by the width of the picture, not by the length of the
    clip.
    """
    n = len(re)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            re[i], re[j] = re[j], re[i]
            im[i], im[j] = im[j], im[i]
    length = 2
    while length <= n:
        ang = -2 * math.pi / length
        wr, wi = math.cos(ang), math.sin(ang)
        for i in range(0, n, length):
            cr, ci = 1.0, 0.0
            half = length >> 1
            for k in range(i, i + half):
                ur, ui = re[k], im[k]
                vr = re[k + half] * cr - im[k + half] * ci
                vi = re[k + half] * ci + im[k + half] * cr
                re[k], im[k] = ur + vr, ui + vi
                re[k + half], im[k + half] = ur - vr, ui - vi
                cr, ci = cr * wr - ci * wi, cr * wi + ci * wr
        length <<= 1


def _hann(n: int) -> list[float]:
    return [0.5 - 0.5 * math.cos(2 * math.pi * i / (n - 1)) for i in range(n)]


def spectrogram(samples: Sequence[float], cols: int, n_fft: int = 512) -> list[list[float]]:
    """`cols` magnitude spectra spread across the clip, in dB.

    One transform per column rather than a fixed hop: the picture is `cols` pixels
    wide, so computing thousands of frames only to average them back down to that
    is work nobody sees.
    """
    total = len(samples)
    if total < 2 or cols <= 0:
        return []
    n_fft = min(n_fft, 1 << max(3, (max(2, total)).bit_length() - 1))
    window = _hann(n_fft)
    span = max(1, (total - n_fft) // max(1, cols - 1)) if total > n_fft else 0
    starts = [min(max(0, total - n_fft), c * span) for c in range(cols)]

    np = _numpy()
    if np is not None:
        a = np.frombuffer(memoryview(array.array("f", samples)), dtype=np.float32)
        if len(a) < n_fft:
            a = np.pad(a, (0, n_fft - len(a)))
        idx = np.asarray(starts)[:, None] + np.arange(n_fft)[None, :]
        frames = a[np.clip(idx, 0, len(a) - 1)] * np.asarray(window, dtype=np.float32)
        mags = np.abs(np.fft.rfft(frames, axis=1)) / (n_fft / 2)
        with np.errstate(divide="ignore"):
            return (20 * np.log10(np.maximum(mags, 1e-6))).tolist()

    out: list[list[float]] = []
    for start in starts:
        chunk = list(samples[start:start + n_fft])
        chunk.extend([0.0] * (n_fft - len(chunk)))
        re = [chunk[i] * window[i] for i in range(n_fft)]
        im = [0.0] * n_fft
        _fft(re, im)
        scale = n_fft / 2
        out.append([
            db(math.hypot(re[k], im[k]) / scale, floor=-120.0)
            for k in range(n_fft // 2 + 1)
        ])
    return out


def envelope(samples: Sequence[float], cols: int) -> list[tuple[float, float, float]]:
    """Per-column (min, max, rms), sampled rather than scanned.

    A ten-minute clip is ten million numbers and the picture has a thousand
    columns; reading every sample to draw each one costs seconds and changes no
    pixel, so each column is stride-sampled to a bounded number of points.
    """
    n = len(samples)
    if n == 0 or cols <= 0:
        return []
    per = max(1, n // cols)
    step = max(1, per // 1200)
    out = []
    for c in range(cols):
        a = (c * n) // cols
        b = max(a + 1, ((c + 1) * n) // cols)
        chunk = samples[a:b:step]
        if not len(chunk):
            out.append((0.0, 0.0, 0.0))
            continue
        rms = math.sqrt(sum(v * v for v in chunk) / len(chunk))
        out.append((min(chunk), max(chunk), rms))
    return out


def frame_levels(samples: Sequence[float], rate: int) -> list[float]:
    """Per-frame RMS in dBFS, the unit the histogram and the stats agree in."""
    n = len(samples)
    if n == 0:
        return []
    size = max(1, int(rate * FRAME_MS / 1000))
    count = max(1, n // size)
    stride = max(1, count // MAX_STAT_FRAMES)
    step = max(1, size // 400)
    levels = []
    for i in range(0, count, stride):
        chunk = samples[i * size:(i + 1) * size:step]
        if not len(chunk):
            continue
        levels.append(db(math.sqrt(sum(v * v for v in chunk) / len(chunk))))
    return levels


# --------------------------------------------------------------------- clip


@dataclass
class Clip:
    """Decoded audio plus the numbers a picture of it is drawn from."""

    samples: array.array
    rate: int = SAMPLE_RATE
    label: str = "audio"
    # Filled by `levels()` on first use. A Clip is read, never edited: replacing
    # `samples` in place would leave this stale, so make a new Clip instead.
    _levels: list[float] | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_path(cls, path: str | Path, rate: int = SAMPLE_RATE) -> "Clip":
        p = Path(path).expanduser()
        return cls(samples=decode(p, rate), rate=rate, label=p.name)

    @classmethod
    def from_pcm16(cls, raw: bytes, rate: int = SAMPLE_RATE, label: str = "capture") -> "Clip":
        return cls(samples=from_pcm16(raw), rate=rate, label=label)

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.rate if self.rate else 0.0

    def levels(self) -> list[float]:
        """Per-frame dBFS, computed once.

        Both the numbers and the picture are readings of these frames -- the
        `active %` in the title, the gate line, the histogram -- so they share
        one list rather than agreeing by coincidence.
        """
        if self._levels is None:
            self._levels = frame_levels(self.samples, self.rate)
        return self._levels

    def stats(self) -> dict:
        n = len(self.samples)
        if not n:
            return {"seconds": 0.0, "rate": self.rate, "samples": 0, "peak_dbfs": -120.0,
                    "rms_dbfs": -120.0, "active": 0.0, "silence": 1.0, "clipped": 0.0,
                    "floor_dbfs": -120.0, "gate_dbfs": SILENCE_DBFS}
        levels = self.levels()
        peak = max(max(self.samples), -min(self.samples))
        # Peak and rms disagree about what they need: the peak has to see every
        # sample or it is not a peak, while an average over a fifth of a million
        # of them is the same number to more decimal places than are printed.
        sparse = self.samples[::max(1, n // 200_000)]
        rms = math.sqrt(sum(v * v for v in sparse) / len(sparse))
        peak_db, rms_db = db(peak), db(rms)
        gate = max(SILENCE_DBFS, peak_db - ACTIVE_MARGIN_DB)
        active = sum(1 for lv in levels if lv >= gate) / len(levels) if levels else 0.0
        quiet = sum(1 for lv in levels if lv < SILENCE_DBFS) / len(levels) if levels else 0.0
        clipped = sum(1 for v in sparse if abs(v) >= 0.999)
        return {
            "seconds": self.seconds,
            "rate": self.rate,
            "samples": n,
            "peak_dbfs": peak_db,
            "rms_dbfs": rms_db,
            "active": active,
            "silence": quiet,
            "clipped": clipped / len(sparse),
            "floor_dbfs": min(levels) if levels else -120.0,
            "gate_dbfs": gate,
        }

    def png(self, width: int = 960) -> bytes:
        return render(self, width=width)

    def wav(self) -> bytes:
        return to_wav(self.samples, self.rate)

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        s = self.stats()
        return (f"<Clip {self.label} {s['seconds']:.1f}s {self.rate // 1000}kHz "
                f"peak {s['peak_dbfs']:.0f}dBFS active {s['active']:.0%}>")


# ------------------------------------------------------------------ drawing

TITLE_H, PAD = 30, 10
WAVE_H, SPEC_H, LEVEL_H, HIST_H = 118, 196, 104, 92
AXIS_L, AXIS_R, AXIS_B = 58, 14, 24
# The spectrogram's floor. Anything quieter is background, and stretching the ramp
# down to -120 dB would spend most of it on the noise nobody is looking at.
SPEC_FLOOR, SPEC_CEIL = -85.0, -8.0
# The range the level curve and the histogram both work in, so a bar and a dip
# can be read against each other.
LEVEL_LO, LEVEL_HI = -70.0, 0.0
HIST_BUCKETS = 28


def _time_ticks(seconds: float) -> list[float]:
    if seconds <= 0:
        return [0.0]
    for step in (0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800):
        if seconds / step <= 8:
            break
    ticks, t = [], 0.0
    while t <= seconds + 1e-9:
        ticks.append(round(t, 3))
        t += step
    return ticks


def _fmt_time(t: float) -> str:
    if t >= 60:
        return f"{int(t) // 60}:{int(t) % 60:02d}"
    if t >= 10:
        return f"{t:.0f}s"
    return f"{t:.2f}".rstrip("0").rstrip(".") + "s"


def _fmt_hz(hz: float) -> str:
    return f"{hz / 1000:g}k" if hz >= 1000 else f"{hz:.0f}"


def _elide(s: str, room: int, scale: int) -> str:
    """Trim a label to the pixels it has, since the facts beside it are fixed."""
    fits = max(0, room // (Canvas.text_width("M", scale) + scale))
    return s if len(s) <= fits else (s[:max(1, fits - 1)] + "~")


# Each panel draws into the box it is handed and knows nothing about the ones
# above it, so `render` below is a table of contents rather than four hundred
# pixels of arithmetic in a row.


def _draw_title(c: Canvas, clip: Clip, s: dict, width: int) -> None:
    facts = (f"{s['seconds']:.2f}s   {clip.rate / 1000:g} khz mono   "
             f"peak {s['peak_dbfs']:.1f} dbfs   rms {s['rms_dbfs']:.1f} dbfs   "
             f"active {s['active']:.0%}   silence {s['silence']:.0%}")
    if s["clipped"] > 0.0005:
        facts += f"   clipped {s['clipped']:.1%}"
    facts_w = Canvas.text_width(facts, 1)
    c.text(AXIS_L, 8, _elide(clip.label, width - AXIS_L - AXIS_R - facts_w - 16, 2),
           FG, scale=2)
    c.text_right(width - AXIS_R, 12, facts, DIM, scale=1)


def _draw_wave(c: Canvas, samples: Sequence[float], top: int, w: int) -> None:
    """Peak-to-peak in the pale colour, rms in the solid one."""
    c.fill(AXIS_L, top, w, WAVE_H, PANEL)
    mid = top + WAVE_H // 2
    half = WAVE_H // 2 - 2
    for frac, tag in ((1.0, ""), (0.316, "-10"), (0.1, "-20")):
        for sign in (1, -1):
            c.dashed_hline(mid - int(sign * frac * half), AXIS_L, AXIS_L + w - 1, GRID)
        # The full-scale line is drawn but not labelled: its label would land in
        # the same pixels as the panel's own name.
        if tag:
            c.text_right(AXIS_L - 5, mid - int(frac * half) - 3, tag, DIM)
    peaks = plot.mix(ACCENT, BG, 0.45)
    for i, (lo, hi, rms) in enumerate(envelope(samples, w)):
        x = AXIS_L + i
        c.vline(x, mid - int(max(-1.0, min(1.0, hi)) * half),
                mid - int(max(-1.0, min(1.0, lo)) * half), peaks)
        r = int(min(1.0, rms) * half)
        c.vline(x, mid - r, mid + r, ACCENT)
    c.hline(mid, AXIS_L, AXIS_L + w - 1, plot.mix(GRID, FG, 0.2))
    c.frame(AXIS_L, top, w, WAVE_H, GRID)
    c.text(6, top + 4, "wave", DIM)


def _draw_spec(c: Canvas, samples: Sequence[float], rate: int, top: int, w: int) -> None:
    """Log-frequency, so speech does not end up squeezed into the bottom tenth."""
    c.fill(AXIS_L, top, w, SPEC_H, (0, 0, 4))
    spec = spectrogram(samples, w)
    nyquist = rate / 2
    if spec:
        bins = len(spec[0])
        log_lo, log_hi = math.log10(60.0), math.log10(nyquist)
        # Precompute the bin each row reads, so the log axis costs one pass.
        rows = []
        for row in range(SPEC_H):
            frac = 1 - row / max(1, SPEC_H - 1)
            hz = 10 ** (log_lo + frac * (log_hi - log_lo))
            rows.append(min(bins - 1, max(0, int(hz / nyquist * (bins - 1)))))
        span = SPEC_CEIL - SPEC_FLOOR
        for i, column in enumerate(spec):
            x = AXIS_L + i
            for row, b in enumerate(rows):
                c.set(x, top + row, plot.heat((column[b] - SPEC_FLOOR) / span))
        for hz in (100, 250, 500, 1000, 2000, 4000, 8000, 16000):
            if hz >= nyquist:
                continue
            frac = (math.log10(hz) - log_lo) / (log_hi - log_lo)
            y = top + int((1 - frac) * (SPEC_H - 1))
            c.dashed_hline(y, AXIS_L, AXIS_L + w - 1, (70, 70, 80), on=2, off=9)
            c.text_right(AXIS_L - 5, y - 3, _fmt_hz(hz), DIM)
    c.frame(AXIS_L, top, w, SPEC_H, GRID)
    c.text(6, top + 4, "spec", DIM)
    c.text(6, top + 16, "hz", DIM)
    # The ramp, so a bright pixel can be read back as a number.
    ramp_w, ramp_x, ramp_y = 90, AXIS_L + w - 96, top + SPEC_H - 12
    c.fill(ramp_x - 30, ramp_y - 3, ramp_w + 76, 12, (0, 0, 4))
    for i in range(ramp_w):
        c.vline(ramp_x + i, ramp_y, ramp_y + 5, plot.heat(i / ramp_w))
    c.text_right(ramp_x - 4, ramp_y - 1, f"{SPEC_FLOOR:.0f}", DIM)
    c.text(ramp_x + ramp_w + 4, ramp_y - 1, f"{SPEC_CEIL:.0f} db", DIM)


def _level_y(v: float, top: int) -> int:
    f = (max(LEVEL_LO, min(LEVEL_HI, v)) - LEVEL_LO) / (LEVEL_HI - LEVEL_LO)
    return top + int((1 - f) * (LEVEL_H - 3)) + 1


def _draw_level(c: Canvas, levels: list[float], gate: float, top: int, w: int) -> None:
    """Loudness against time, with the gate the `active %` was counted against."""
    c.fill(AXIS_L, top, w, LEVEL_H, PANEL)
    for mark in (0, -20, -40, -60):
        y = _level_y(mark, top)
        c.dashed_hline(y, AXIS_L, AXIS_L + w - 1, GRID)
        c.text_right(AXIS_L - 5, y - 3, f"{mark}", DIM)
    if levels:
        gate_y = _level_y(gate, top)
        c.dashed_hline(gate_y, AXIS_L, AXIS_L + w - 1, plot.mix(WARM, PANEL, 0.4),
                       on=6, off=6)
        c.text(AXIS_L + 3, gate_y - 9, "voice gate", plot.mix(WARM, PANEL, 0.25))
        prev = None
        for i in range(w):
            lv = levels[min(len(levels) - 1, i * len(levels) // max(1, w))]
            y, x = _level_y(lv, top), AXIS_L + i
            colour = ACCENT if lv >= gate else plot.mix(DIM, PANEL, 0.35)
            c.vline(x, y, top + LEVEL_H - 2, plot.mix(colour, PANEL, 0.78))
            if prev is not None:
                c.vline(x, min(prev, y), max(prev, y), colour)
            c.set(x, y, colour)
            prev = y
    c.frame(AXIS_L, top, w, LEVEL_H, GRID)
    c.text(6, top + 4, "level", DIM)
    c.text(6, top + 16, "dbfs", DIM)


def _draw_time_axis(c: Canvas, base: int, w: int, seconds: float) -> None:
    """One axis for the three panels above it, so they can be read against
    each other: a bright band in the spectrogram sits directly above the moment
    it happened in the waveform."""
    c.hline(base, AXIS_L, AXIS_L + w - 1, GRID)
    for t in _time_ticks(seconds):
        x = AXIS_L + int(min(1.0, t / seconds) * (w - 1))
        c.vline(x, base, base + 3, GRID)
        c.text_center(x, base + 7, _fmt_time(t), DIM)


def _draw_hist(c: Canvas, levels: list[float], gate: float, top: int, w: int) -> None:
    """How long the clip spends at each level. Off the time axis entirely -- its
    x is dBFS -- which is why it lives below the axis rather than beside a panel
    that answers to it."""
    hist_h = HIST_H - 40
    c.fill(AXIS_L, top, w, hist_h, PANEL)
    counts = [0] * HIST_BUCKETS
    for lv in levels:
        f = (max(LEVEL_LO, min(LEVEL_HI, lv)) - LEVEL_LO) / (LEVEL_HI - LEVEL_LO)
        counts[min(HIST_BUCKETS - 1, int(f * HIST_BUCKETS))] += 1
    tallest = max(counts) or 1
    bar_w = w / HIST_BUCKETS
    for b, count in enumerate(counts):
        x = AXIS_L + int(b * bar_w)
        bw = max(1, int((b + 1) * bar_w) - int(b * bar_w) - 1)
        h = int((count / tallest) * (hist_h - 3))
        edge = LEVEL_LO + (b / HIST_BUCKETS) * (LEVEL_HI - LEVEL_LO)
        colour = ACCENT if edge >= gate else plot.mix(DIM, PANEL, 0.45)
        c.fill(x, top + hist_h - 1 - h, bw, h, colour)
    c.frame(AXIS_L, top, w, hist_h, GRID)
    c.text(6, top + 4, "how", DIM)
    c.text(6, top + 16, "long", DIM)
    c.text(AXIS_L + 4, top - 11, f"time spent at each level, dbfs "
                                 f"({len(levels)} frames of {FRAME_MS:g}ms)", DIM)
    for mark in range(int(LEVEL_LO), 1, 10):
        f = (mark - LEVEL_LO) / (LEVEL_HI - LEVEL_LO)
        x = AXIS_L + int(f * (w - 1))
        c.vline(x, top + hist_h, top + hist_h + 3, GRID)
        c.text_center(x, top + hist_h + 7, f"{mark}", DIM)


def render(clip: Clip, width: int = 960) -> bytes:
    """The picture: waveform, spectrogram, level over time, level histogram.

    The three time panels share one x axis and one width, which is the only
    reason the picture can be read as a whole rather than as four charts.
    """
    width = max(420, int(width))
    height = TITLE_H + WAVE_H + PAD + SPEC_H + PAD + LEVEL_H + AXIS_B + HIST_H
    c = Canvas(width, height)
    s = clip.stats()
    # One list, shared: the gate line, the histogram and the `active %` in the
    # title are three readings of the same frames, and computing them twice
    # would leave that agreement to luck.
    levels = clip.levels()
    gate = s["gate_dbfs"]
    plot_w = width - AXIS_L - AXIS_R

    wave_top = TITLE_H
    spec_top = wave_top + WAVE_H + PAD
    level_top = spec_top + SPEC_H + PAD
    axis_y = level_top + LEVEL_H + AXIS_B - 8
    hist_top = height - HIST_H + 22          # clear of the time labels above

    _draw_title(c, clip, s, width)
    _draw_wave(c, clip.samples, wave_top, plot_w)
    _draw_spec(c, clip.samples, clip.rate, spec_top, plot_w)
    _draw_level(c, levels, gate, level_top, plot_w)
    _draw_time_axis(c, axis_y, plot_w, max(s["seconds"], 1e-6))
    _draw_hist(c, levels, gate, hist_top, plot_w)
    return c.to_png()


# ------------------------------------------------------------- attachments


@dataclass
class AudioAttachment:
    """A validated clip, its picture, and whatever the ASR model heard in it."""

    path: Path
    clip: Clip
    transcript: str = ""
    language: str = "auto"
    detected: list[str] = field(default_factory=list)
    asr_error: str = ""
    _png: bytes = b""

    @classmethod
    def from_path(cls, path: str | Path, *, lang: str = "auto", asr: bool = False,
                  on_text=None, width: int = 960) -> "AudioAttachment":
        p = Path(path).expanduser()
        suffix = p.suffix.lower()
        if suffix and suffix not in SUPPORTED_AUDIO_TYPES:
            supported = ", ".join(sorted(SUPPORTED_AUDIO_TYPES))
            raise ValueError(f"unsupported audio type for {p}; use one of {supported}")
        clip = Clip.from_path(p)
        if not len(clip):
            raise AudioError(f"audio file decoded to no samples: {p}")
        return cls.from_clip(clip, path=p, lang=lang, asr=asr, on_text=on_text, width=width)

    @classmethod
    def from_clip(cls, clip: Clip, *, path: str | Path | None = None, lang: str = "auto",
                  asr: bool = False, on_text=None, width: int = 960) -> "AudioAttachment":
        att = cls(path=Path(path) if path else Path(clip.label), clip=clip, language=lang)
        att._png = clip.png(width=width)
        if asr:
            from xagent.asr import transcribe_clip

            try:
                heard = transcribe_clip(clip, lang=lang, on_text=on_text)
                att.transcript, att.detected = heard.text, heard.languages
            except Exception as e:                    # ASR is optional; the picture is not
                att.asr_error = f"{type(e).__name__}: {e}"
        return att

    # ------------------------------------------------------------ rendering

    def image_block(self) -> dict:
        import base64

        return {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png",
                       "data": base64.b64encode(self._png).decode("ascii")},
        }

    @property
    def png(self) -> bytes:
        return self._png

    def summary(self) -> str:
        s = self.clip.stats()
        tag = f" [{', '.join(self.detected)}]" if self.detected else ""
        return (f"{self.path} — {s['seconds']:.1f}s, {s['rate'] // 1000} kHz mono, "
                f"peak {s['peak_dbfs']:.1f} dBFS, rms {s['rms_dbfs']:.1f} dBFS, "
                f"{s['active']:.0%} active / {s['silence']:.0%} silent{tag}")

    def text_block(self) -> str:
        parts = [self.summary()]
        if self.transcript.strip():
            parts.append(f"transcript:\n{self.transcript.strip()}")
        elif self.asr_error:
            parts.append(f"transcript unavailable — {self.asr_error}")
        return "\n".join(parts)

    def tokens(self) -> int:
        from xagent.vision import IMAGE_TOKEN_ESTIMATE

        return IMAGE_TOKEN_ESTIMATE + len(self.text_block()) // 4 + 1


AudioInput = str | Path | AudioAttachment


def normalize_audio(
    clips: AudioInput | Iterable[AudioInput] | None, *, lang: str = "auto",
    asr: bool = False, on_text=None,
) -> list[AudioAttachment]:
    """Load and validate one clip or a sequence of them."""
    if clips is None:
        return []
    if isinstance(clips, (str, Path, AudioAttachment)):
        clips = [clips]
    return [
        clip if isinstance(clip, AudioAttachment)
        else AudioAttachment.from_path(clip, lang=lang, asr=asr, on_text=on_text)
        for clip in clips
    ]


# ------------------------------------------------------- the model's handle


@dataclass
class Sound:
    """A clip the model is holding: the picture, the numbers, the samples.

    Returned by `listen()`. The samples stay in the kernel like every other large
    value -- what enters the context window is the panel image and a one-line
    repr, and `zoom()` re-renders any span of it without re-capturing.
    """

    clip: Clip
    source: str = ""
    text: str = ""
    started: float = 0.0          # offset into the original capture, for zoom()

    @property
    def seconds(self) -> float:
        return self.clip.seconds

    @property
    def rate(self) -> int:
        return self.clip.rate

    @property
    def samples(self) -> array.array:
        return self.clip.samples

    def stats(self) -> dict:
        return self.clip.stats()

    def png(self, width: int = 960) -> bytes:
        return self.clip.png(width=width)

    def show(self, width: int = 960) -> "Sound":
        """Display the panels, so the harness attaches them to this cell."""
        try:
            from IPython import get_ipython
            from IPython.display import Image, display
        except ImportError:                       # outside a kernel: nothing to show
            return self
        if get_ipython() is None:
            # Called from a plain script, where `display` would print a repr line
            # and no picture. Silence is the honest answer.
            return self
        display(Image(data=self.png(width=width), format="png"))
        return self

    def zoom(self, start: float = 0.0, end: float | None = None, *,
             show: bool = True, width: int = 960) -> "Sound":
        """A new Sound over [start, end) seconds of this one."""
        n = len(self.clip.samples)
        a = max(0, min(n, int(start * self.rate)))
        b = n if end is None else max(a + 1, min(n, int(end * self.rate)))
        label = f"{self.clip.label} {self.started + start:.2f}-{self.started + b / self.rate:.2f}s"
        cut = Sound(Clip(samples=self.clip.samples[a:b], rate=self.rate, label=label),
                    source=self.source, started=self.started + a / self.rate)
        return cut.show(width=width) if show else cut

    def save(self, path: str | Path) -> str:
        """Write the audio (.wav) or the panels (.png) to disk."""
        p = Path(path).expanduser()
        p.write_bytes(self.png() if p.suffix.lower() == ".png" else self.clip.wav())
        return f"wrote {p} ({p.stat().st_size:,} bytes)"

    def __repr__(self) -> str:
        s = self.stats()
        head = f" {self.text.strip()[:80]}" if self.text.strip() else ""
        return (f"<Sound {self.source or self.clip.label} {s['seconds']:.1f}s "
                f"{self.rate // 1000}kHz peak {s['peak_dbfs']:.0f}dBFS "
                f"active {s['active']:.0%}>{head}")
