"""A minimal raster canvas, so the audio panels need no plotting library.

Rendering a waveform and a spectrogram is a few thousand rectangles and a few
dozen labels. Reaching for matplotlib to draw them would put numpy, a font stack
and a C extension between `xagent --audio clip.wav` and a picture of the sound --
for a picture this file draws in a hundred lines of the standard library.

The output is a PNG because that is what the vision path already accepts: raw RGB
rows, one zlib stream, three chunks.
"""

from __future__ import annotations

import struct
import zlib

RGB = tuple[int, int, int]

# A dark ground, because a spectrogram is mostly floor and a light one glares.
BG: RGB = (14, 15, 20)
PANEL: RGB = (21, 23, 30)
GRID: RGB = (44, 48, 60)
FG: RGB = (226, 230, 240)
DIM: RGB = (129, 138, 158)
ACCENT: RGB = (118, 185, 0)      # the NVIDIA green the model card wears
WARM: RGB = (240, 140, 70)

# inferno, sampled at eight stops and interpolated between them. Perceptually
# ordered, so a brighter pixel really is a louder one -- which a hue ramp such as
# jet does not promise, and a reader cannot check.
_RAMP: tuple[RGB, ...] = (
    (0, 0, 4), (40, 11, 84), (101, 21, 110), (159, 42, 99),
    (212, 72, 66), (245, 125, 21), (250, 193, 39), (252, 255, 164),
)


def mix(a: RGB, b: RGB, t: float) -> RGB:
    """`t` of the way from `a` to `b`."""
    return (int(a[0] + (b[0] - a[0]) * t),
            int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t))


def heat(t: float) -> RGB:
    """Map 0..1 onto the ramp."""
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    pos = t * (len(_RAMP) - 1)
    i = int(pos)
    if i >= len(_RAMP) - 1:
        return _RAMP[-1]
    return mix(_RAMP[i], _RAMP[i + 1], pos - i)


# A 5x7 bitmap font, written as pixels rather than as hex so a wrong glyph is
# visible in the diff. Uppercase only: labels on a plot are short, and carrying
# lowercase would double the table for characters an axis never uses.
_GLYPHS: dict[str, tuple[str, ...]] = {
    "A": (".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "B": ("####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."),
    "C": (".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."),
    "D": ("####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."),
    "E": ("#####", "#....", "#....", "####.", "#....", "#....", "#####"),
    "F": ("#####", "#....", "#....", "####.", "#....", "#....", "#...."),
    "G": (".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".###."),
    "H": ("#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "I": (".###.", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "J": ("..###", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."),
    "K": ("#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"),
    "L": ("#....", "#....", "#....", "#....", "#....", "#....", "#####"),
    "M": ("#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"),
    "N": ("#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"),
    "O": (".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "P": ("####.", "#...#", "#...#", "####.", "#....", "#....", "#...."),
    "Q": (".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"),
    "R": ("####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"),
    "S": (".####", "#....", "#....", ".###.", "....#", "....#", "####."),
    "T": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."),
    "U": ("#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "V": ("#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."),
    "W": ("#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"),
    "X": ("#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"),
    "Y": ("#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."),
    "Z": ("#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"),
    "0": (".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."),
    "1": ("..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "2": (".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"),
    "3": ("#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."),
    "4": ("...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."),
    "5": ("#####", "#....", "####.", "....#", "....#", "#...#", ".###."),
    "6": ("..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."),
    "7": ("#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."),
    "8": (".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."),
    "9": (".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."),
    " ": (".....", ".....", ".....", ".....", ".....", ".....", "....."),
    ".": (".....", ".....", ".....", ".....", ".....", ".##..", ".##.."),
    ",": (".....", ".....", ".....", ".....", ".##..", ".##..", ".#..."),
    ":": (".....", ".##..", ".##..", ".....", ".##..", ".##..", "....."),
    "-": (".....", ".....", ".....", "#####", ".....", ".....", "....."),
    "+": (".....", "..#..", "..#..", "#####", "..#..", "..#..", "....."),
    "/": ("....#", "...#.", "...#.", "..#..", ".#...", ".#...", "#...."),
    "%": ("##..#", "##..#", "...#.", "..#..", ".#...", "#..##", "#..##"),
    "(": ("..##.", ".#...", ".#...", ".#...", ".#...", ".#...", "..##."),
    ")": (".##..", "...#.", "...#.", "...#.", "...#.", "...#.", ".##.."),
    "[": (".###.", ".#...", ".#...", ".#...", ".#...", ".#...", ".###."),
    "]": (".###.", "...#.", "...#.", "...#.", "...#.", "...#.", ".###."),
    "<": ("...#.", "..#..", ".#...", "#....", ".#...", "..#..", "...#."),
    ">": (".#...", "..#..", "...#.", "....#", "...#.", "..#..", ".#..."),
    "=": (".....", ".....", "#####", ".....", "#####", ".....", "....."),
    "_": (".....", ".....", ".....", ".....", ".....", ".....", "#####"),
    "#": (".#.#.", ".#.#.", "#####", ".#.#.", "#####", ".#.#.", ".#.#."),
    "*": (".....", "#.#.#", ".###.", "#####", ".###.", "#.#.#", "....."),
    "?": (".###.", "#...#", "....#", "...#.", "..#..", ".....", "..#.."),
    "!": ("..#..", "..#..", "..#..", "..#..", "..#..", ".....", "..#.."),
    "'": ("..#..", "..#..", ".....", ".....", ".....", ".....", "....."),
    "~": (".....", ".....", ".##.#", "#.#.#", "#.##.", ".....", "....."),
}
_MISSING = ("#####", "#...#", "#...#", "#...#", "#...#", "#...#", "#####")

GLYPH_W, GLYPH_H, TRACK = 5, 7, 1


class Canvas:
    """An RGB pixel buffer that knows how to become a PNG."""

    def __init__(self, width: int, height: int, bg: RGB = BG):
        if width <= 0 or height <= 0:
            raise ValueError(f"canvas must have positive size, got {width}x{height}")
        self.width, self.height = width, height
        self.buf = bytearray(bytes(bg) * (width * height))

    # ------------------------------------------------------------- primitives

    def set(self, x: int, y: int, rgb: RGB) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            i = (y * self.width + x) * 3
            self.buf[i:i + 3] = bytes(rgb)

    def fill(self, x: int, y: int, w: int, h: int, rgb: RGB) -> None:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.width, x + w), min(self.height, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        row = bytes(rgb) * (x1 - x0)
        for yy in range(y0, y1):
            i = (yy * self.width + x0) * 3
            self.buf[i:i + len(row)] = row

    def vline(self, x: int, y0: int, y1: int, rgb: RGB) -> None:
        if y1 < y0:
            y0, y1 = y1, y0
        self.fill(x, y0, 1, y1 - y0 + 1, rgb)

    def hline(self, y: int, x0: int, x1: int, rgb: RGB) -> None:
        if x1 < x0:
            x0, x1 = x1, x0
        self.fill(x0, y, x1 - x0 + 1, 1, rgb)

    def dashed_hline(self, y: int, x0: int, x1: int, rgb: RGB, on: int = 3, off: int = 4) -> None:
        x = x0
        while x <= x1:
            self.fill(x, y, min(on, x1 - x + 1), 1, rgb)
            x += on + off

    def frame(self, x: int, y: int, w: int, h: int, rgb: RGB) -> None:
        self.hline(y, x, x + w - 1, rgb)
        self.hline(y + h - 1, x, x + w - 1, rgb)
        self.vline(x, y, y + h - 1, rgb)
        self.vline(x + w - 1, y, y + h - 1, rgb)

    # ------------------------------------------------------------------- text

    @staticmethod
    def text_width(s: str, scale: int = 1) -> int:
        if not s:
            return 0
        return (len(s) * (GLYPH_W + TRACK) - TRACK) * scale

    def text(self, x: int, y: int, s: str, rgb: RGB = FG, scale: int = 1) -> int:
        """Draw uppercase text, returning the width it took."""
        cx = x
        for ch in s.upper():
            glyph = _GLYPHS.get(ch, _MISSING)
            for row, bits in enumerate(glyph):
                for col, bit in enumerate(bits):
                    if bit == "#":
                        self.fill(cx + col * scale, y + row * scale, scale, scale, rgb)
            cx += (GLYPH_W + TRACK) * scale
        return cx - TRACK * scale - x

    def text_right(self, x: int, y: int, s: str, rgb: RGB = FG, scale: int = 1) -> None:
        self.text(x - self.text_width(s, scale), y, s, rgb, scale)

    def text_center(self, x: int, y: int, s: str, rgb: RGB = FG, scale: int = 1) -> None:
        self.text(x - self.text_width(s, scale) // 2, y, s, rgb, scale)

    # -------------------------------------------------------------------- png

    def to_png(self) -> bytes:
        stride = self.width * 3
        raw = bytearray()
        for y in range(self.height):
            raw.append(0)                       # filter type 0: none
            raw += self.buf[y * stride:(y + 1) * stride]
        return _png(self.width, self.height, bytes(raw))


def _chunk(tag: bytes, data: bytes) -> bytes:
    body = tag + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def _png(width: int, height: int, raw: bytes) -> bytes:
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit truecolour
    return (b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", header)
            + _chunk(b"IDAT", zlib.compress(raw, 6))
            + _chunk(b"IEND", b""))
