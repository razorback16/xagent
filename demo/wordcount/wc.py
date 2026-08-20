#!/usr/bin/env python3
"""Count words, lines and characters in a file.

Usage:
    wc.py [-w] [-l] [-c] [--top N] FILE

With no selection flag, all three counts are printed on one line:
    lines words characters FILE
With one or more of -l/-w/-c, only the selected counts are printed,
one per line (in the order lines, words, characters):
    count FILE
--top N additionally prints the N most common words (case-insensitive),
one per line as "count word", most frequent first, ties alphabetical.

Exits 0 on success, 1 if the file cannot be read, 2 on bad usage.
"""

import argparse
import sys
from collections import Counter


def analyze(text):
    """Return a dict with lines, words and characters counts for *text*."""
    return {
        "lines": len(text.splitlines()),
        "words": len(text.split()),
        "chars": len(text),
    }


def top_words(text, n):
    """Return the N most common (lowercased) words as (word, count) pairs."""
    counts = Counter(w.lower() for w in text.split())
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:n]


def build_parser():
    p = argparse.ArgumentParser(
        prog="wc.py",
        description="Count words, lines and characters in a file.",
    )
    p.add_argument("file", help="file to read")
    p.add_argument("-l", action="store_true", help="print the line count")
    p.add_argument("-w", action="store_true", help="print the word count")
    p.add_argument("-c", action="store_true", help="print the character count")
    p.add_argument(
        "--top",
        type=int,
        metavar="N",
        help="show the N most common words",
    )
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.top is not None and args.top < 0:
        parser.error("--top N must be a non-negative integer")

    try:
        with open(args.file, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as e:
        print(f"wc.py: {args.file}: {e.strerror}", file=sys.stderr)
        return 1

    counts = analyze(text)

    if args.l or args.w or args.c:
        for flag, key in ((args.l, "lines"), (args.w, "words"), (args.c, "chars")):
            if flag:
                print(f"{counts[key]} {args.file}")
    else:
        print(
            f"{counts['lines']} {counts['words']} {counts['chars']} {args.file}"
        )

    if args.top:
        for word, n in top_words(text, args.top):
            print(f"{n} {word}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
