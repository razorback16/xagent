#!/usr/bin/env python3
"""Tests for wc.py."""

import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

import wc


class TestAnalyze(unittest.TestCase):
    def test_basic_counts(self):
        text = "hello world\nfoo bar\n"
        c = wc.analyze(text)
        self.assertEqual(c["lines"], 2)
        self.assertEqual(c["words"], 4)
        self.assertEqual(c["chars"], 20)

    def test_empty(self):
        c = wc.analyze("")
        self.assertEqual(c, {"lines": 0, "words": 0, "chars": 0})

    def test_no_trailing_newline(self):
        c = wc.analyze("one two three")
        self.assertEqual(c["lines"], 1)
        self.assertEqual(c["words"], 3)
        self.assertEqual(c["chars"], 13)

    def test_blank_lines_and_whitespace(self):
        text = "a\n\n\nb  c\t\n"
        c = wc.analyze(text)
        self.assertEqual(c["lines"], 4)
        self.assertEqual(c["words"], 3)
        self.assertEqual(c["chars"], 10)


class TestTopWords(unittest.TestCase):
    def test_counts_and_order(self):
        text = "the quick brown the fox the lazy dog the"
        top = wc.top_words(text, 3)
        self.assertEqual(top, [("the", 4), ("brown", 1), ("dog", 1)])

    def test_case_insensitive(self):
        top = wc.top_words("The the THE", 1)
        self.assertEqual(top, [("the", 3)])

    def test_limit(self):
        text = "a b c d e"
        self.assertEqual(wc.top_words(text, 2), [("a", 1), ("b", 1)])

    def test_zero_returns_empty(self):
        self.assertEqual(wc.top_words("x y", 0), [])

    def test_empty_text(self):
        self.assertEqual(wc.top_words("", 5), [])


class TestMain(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        with open(self.path, "w") as f:
            f.write("alpha beta alpha\ngamma alpha\n")

    def tearDown(self):
        os.unlink(self.path)

    def call(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = wc.main([self.path, *argv])
        return code, out.getvalue(), err.getvalue()

    def test_all_counts_default(self):
        code, out, err = self.call()
        self.assertEqual(code, 0)
        self.assertEqual(out, f"2 5 29 {self.path}\n")
        self.assertEqual(err, "")

    def test_single_flag_w(self):
        code, out, _ = self.call("-w")
        self.assertEqual(code, 0)
        self.assertEqual(out, f"5 {self.path}\n")

    def test_single_flag_l(self):
        code, out, _ = self.call("-l")
        self.assertEqual(out, f"2 {self.path}\n")

    def test_single_flag_c(self):
        code, out, _ = self.call("-c")
        self.assertEqual(out, f"29 {self.path}\n")

    def test_multiple_flags_order(self):
        code, out, _ = self.call("-c", "-w", "-l")
        self.assertEqual(code, 0)
        self.assertEqual(out, f"2 {self.path}\n5 {self.path}\n29 {self.path}\n")

    def test_top_flag(self):
        code, out, _ = self.call("--top", "2")
        self.assertEqual(code, 0)
        expected = f"2 5 29 {self.path}\n3 alpha\n1 beta\n"
        self.assertEqual(out, expected)

    def test_top_with_only_w(self):
        code, out, _ = self.call("-w", "--top", "1")
        self.assertEqual(code, 0)
        self.assertEqual(out, f"5 {self.path}\n3 alpha\n")

    def test_top_zero_prints_nothing_extra(self):
        code, out, _ = self.call("--top", "0")
        self.assertEqual(code, 0)
        self.assertEqual(out, f"2 5 29 {self.path}\n")

    def test_missing_file(self):
        out2, err2 = io.StringIO(), io.StringIO()
        with redirect_stdout(out2), redirect_stderr(err2):
            code = wc.main(["definitely_missing.txt"])
        self.assertEqual(code, 1)
        self.assertIn("definitely_missing.txt", err2.getvalue())
        self.assertEqual(out2.getvalue(), "")

    def test_negative_top_rejected(self):
        with self.assertRaises(SystemExit) as cm:
            with redirect_stderr(io.StringIO()):
                wc.main([self.path, "--top", "-1"])
        self.assertEqual(cm.exception.code, 2)


class TestCLI(unittest.TestCase):
    """Run wc.py as a real subprocess."""

    FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wc.py")

    def cli(self, *argv):
        return subprocess.run(
            [sys.executable, self.FILE, *argv],
            capture_output=True, text=True,
        )

    def test_default_output(self):
        r = self.cli("-l", "-w", "-c", self.FILE)
        self.assertEqual(r.returncode, 0)
        lines = r.stdout.splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].endswith(self.FILE))
        # the file itself is 2,588 bytes of ASCII, so counts must be plausible
        self.assertGreaterEqual(int(lines[1].split()[0]), 100)

    def test_missing_file_exit_code(self):
        r = self.cli("no_such_file_here.txt")
        self.assertEqual(r.returncode, 1)
        self.assertIn("no_such_file_here.txt", r.stderr)

    def test_top_flag(self):
        r = self.cli("-w", "--top", "1", self.FILE)
        self.assertEqual(r.returncode, 0)
        self.assertEqual(len(r.stdout.splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
