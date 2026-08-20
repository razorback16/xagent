"""Tests for todo.py.

Each test runs against a throwaway SQLite database in a temporary directory,
so the real todo.db (if any) is never touched.
"""

import os
import tempfile
import unittest

import todo


class TodoTestCase(unittest.TestCase):
    """Base class that gives each test its own temporary todo.db."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = os.path.join(self._tmp.name, "todo.db")
        self.con = todo.connect(self.db_path)
        self.addCleanup(self.con.close)

    # Helpers ----------------------------------------------------------------
    def main(self, *argv):
        """Invoke the CLI against the temp database; returns the exit code."""
        return todo.main(list(argv), db_path=self.db_path)

    def add(self, text, priority=None):
        cmd = ["add", text]
        if priority is not None:
            cmd += ["--priority", priority]
        rc = self.main(*cmd)
        self.assertEqual(rc, 0)
        return self.last_id()

    def last_id(self):
        row = self.con.execute(
            "SELECT id FROM tasks ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None

    def ids(self, include_done=False, priority=None):
        rows = todo.list_tasks(self.con, include_done, priority)
        return [row[0] for row in rows]

    def seed(self, mapping):
        """Create tasks from {id: (text, priority, done)}."""
        for task_id, (text, priority, done) in mapping.items():
            self.con.execute(
                "INSERT INTO tasks (id, text, priority, done) VALUES (?, ?, ?, ?)",
                (task_id, text, priority, int(done)),
            )
        self.con.commit()

    def capture_output(self, fn):
        """Run fn(), capturing combined stdout+stderr as a string."""
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            fn()
        return buf.getvalue()


class TestAddList(TodoTestCase):
    def test_add_creates_row(self):
        self.add("buy milk")
        row = self.con.execute("SELECT id, text, priority, done FROM tasks").fetchone()
        self.assertEqual(row, (1, "buy milk", "med", 0))

    def test_add_explicit_priority(self):
        self.add("ship release", priority="high")
        row = self.con.execute("SELECT priority FROM tasks").fetchone()
        self.assertEqual(row[0], "high")

    def test_ids_are_sequential(self):
        self.add("a")
        self.add("b")
        self.add("c")
        self.assertEqual(self.last_id(), 3)

    def test_list_shows_only_open_by_default(self):
        a = self.add("open one")
        b = self.add("open two")
        self.assertEqual(self.ids(), [a, b])
        self.main("done", str(a))
        self.assertEqual(self.ids(), [b])

    def test_list_all_includes_completed(self):
        a = self.add("first")
        b = self.add("second")
        self.main("done", str(b))
        self.assertEqual(self.ids(include_done=True), [a, b])

    def test_list_empty_database(self):
        self.assertEqual(self.ids(), [])

    def test_list_output_formatting(self):
        a = self.add("urgent thing", priority="high")
        b = self.add("chore", priority="low")
        out = self.capture_output(lambda: self.main("list"))
        lines = out.strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn(str(a), lines[0])
        self.assertIn("[ ]", lines[0])
        self.assertIn(str(b), lines[1])
        self.main("done", str(a))
        out = self.capture_output(lambda: self.main("list", "--all"))
        first = out.strip().splitlines()[0]
        self.assertIn(str(a), first)
        self.assertIn("[x]", first)


class TestSortingAndFiltering(TodoTestCase):
    def test_sorted_by_priority_then_id(self):
        self.add("low1", priority="low")
        self.add("high1", priority="high")
        self.add("med1", priority="med")
        self.add("high2", priority="high")
        self.add("low2", priority="low")
        self.assertEqual(self.ids(), [2, 4, 3, 1, 5])

    def test_priority_filter_high(self):
        self.add("l", priority="low")
        self.add("h1", priority="high")
        self.add("m", priority="med")
        self.add("h2", priority="high")
        self.assertEqual(self.ids(priority="high"), [2, 4])

    def test_priority_filter_no_matches(self):
        self.add("m", priority="med")
        self.assertEqual(self.ids(priority="high"), [])

    def test_filter_with_all(self):
        self.add("done high", priority="high")
        self.add("open high", priority="high")
        self.main("done", "1")
        self.assertEqual(self.ids(include_done=False, priority="high"), [2])
        self.assertEqual(self.ids(include_done=True, priority="high"), [1, 2])


class TestDoneRm(TodoTestCase):
    def test_done_marks_task(self):
        a = self.add("to complete")
        self.assertEqual(self.main("done", str(a)), 0)
        self.assertEqual(self.ids(), [])
        self.assertEqual(self.ids(include_done=True), [a])

    def test_done_unknown_id(self):
        rc = self.main("done", "999")
        self.assertNotEqual(rc, 0)
        err = self.capture_output(lambda: self.main("done", "999"))
        self.assertIn("no task with id 999", err)

    def test_done_already_done_is_ok(self):
        a = self.add("x")
        self.main("done", str(a))
        self.assertEqual(self.main("done", str(a)), 0)
        self.assertEqual(self.ids(include_done=True), [a])

    def test_done_non_numeric_id(self):
        self.assertNotEqual(self.main("done", "abc"), 0)

    def test_rm_deletes_task(self):
        self.add("one")
        b = self.add("two")
        self.assertEqual(self.main("rm", str(b)), 0)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM tasks").fetchone()[0], 1)

    def test_rm_unknown_id(self):
        rc = self.main("rm", "42")
        self.assertNotEqual(rc, 0)
        err = self.capture_output(lambda: self.main("rm", "42"))
        self.assertIn("no task with id 42", err)

    def test_rm_open_task_then_stats(self):
        self.add("kept")
        self.add("gone")
        self.main("rm", "2")
        stats = todo.get_stats(self.con)
        self.assertEqual(stats["total"], 1)
        self.assertEqual(stats["open"], 1)


class TestStats(TodoTestCase):
    def test_stats_empty(self):
        stats = todo.get_stats(self.con)
        self.assertEqual(stats["total"], 0)
        self.assertEqual(stats["open"], 0)
        self.assertEqual(stats["completed"], 0)
        self.assertEqual(stats["open_by_priority"], {"high": 0, "med": 0, "low": 0})

    def test_stats_counts(self):
        self.add("a", priority="high")
        self.add("b", priority="high")
        self.add("c", priority="med")
        self.add("d", priority="low")
        self.add("e", priority="low")
        self.main("done", "3")   # the med task
        self.main("done", "5")   # a low task
        stats = todo.get_stats(self.con)
        self.assertEqual(stats["total"], 5)
        self.assertEqual(stats["open"], 3)
        self.assertEqual(stats["completed"], 2)
        self.assertEqual(stats["open_by_priority"], {"high": 2, "med": 0, "low": 1})

    def test_stats_command_prints(self):
        self.add("a", priority="high")
        self.add("b")
        self.main("done", "1")
        out = self.capture_output(lambda: self.main("stats"))
        self.assertIn("Tasks:     2", out)        # total 2
        self.assertIn("Open:      1", out)
        self.assertIn("Completed: 1", out)
        self.assertIn("high 0", out)


class TestErrors(TodoTestCase):
    def test_bad_priority(self):
        rc = self.main("add", "x", "--priority", "urgent")
        self.assertNotEqual(rc, 0)
        self.assertEqual(self.last_id(), None)

    def test_bad_priority_on_list(self):
        rc = self.main("list", "--priority", "urgent")
        self.assertNotEqual(rc, 0)

    def test_missing_text(self):
        self.assertNotEqual(self.main("add"), 0)

    def test_missing_id(self):
        self.assertNotEqual(self.main("done"), 0)

    def test_no_command(self):
        self.assertNotEqual(self.main(), 0)

    def test_bad_input_does_not_modify_data(self):
        self.add("safe task")
        self.main("done", "123")
        self.main("rm", "123")
        self.assertEqual(self.ids(), [1])


if __name__ == "__main__":
    unittest.main()
