#!/usr/bin/env python3
"""A simple command-line todo manager backed by SQLite.

Usage:
    python todo.py add <text> [--priority low|med|high]
    python todo.py list [--all] [--priority low|med|high]
    python todo.py done <id>
    python todo.py rm <id>
    python todo.py stats

Tasks are stored in todo.db (SQLite) in the current working directory,
created on demand. Standard library only.
"""

import argparse
import sqlite3
import sys

DEFAULT_DB = "todo.db"
PRIORITIES = ("low", "med", "high")

# Lower rank sorts first, so "high" priority tasks appear at the top.
PRIORITY_ORDER = "CASE priority WHEN 'high' THEN 0 WHEN 'med' THEN 1 WHEN 'low' THEN 2 END"

EXIT_OK = 0
EXIT_ERROR = 1


class TodoError(Exception):
    """A user-facing error (unknown id, etc.). Reported on stderr, exit code 1."""


def connect(db_path=DEFAULT_DB):
    """Open (and if necessary create) the todo database."""
    con = sqlite3.connect(db_path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            text     TEXT NOT NULL,
            priority TEXT NOT NULL DEFAULT 'med'
                     CHECK (priority IN ('low', 'med', 'high')),
            done     INTEGER NOT NULL DEFAULT 0 CHECK (done IN (0, 1))
        )
        """
    )
    con.commit()
    return con


def list_tasks(con, include_done=False, priority=None):
    """Return tasks sorted by priority (high first), then id."""
    sql = "SELECT id, text, priority, done FROM tasks"
    clauses, params = [], []
    if not include_done:
        clauses.append("done = 0")
    if priority is not None:
        clauses.append("priority = ?")
        params.append(priority)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY " + PRIORITY_ORDER + ", id"
    return con.execute(sql, params).fetchall()


def format_task(row):
    task_id, text, priority, done = row
    mark = "x" if done else " "
    return f"{task_id:>4}  [{mark}] {priority:<4} {text}"


def get_stats(con):
    """Return a dict of counts for the stats command."""
    (total,) = con.execute("SELECT COUNT(*) FROM tasks").fetchone()
    (open_count,) = con.execute("SELECT COUNT(*) FROM tasks WHERE done = 0").fetchone()
    by_priority = {"high": 0, "med": 0, "low": 0}
    for prio, count in con.execute(
        "SELECT priority, COUNT(*) FROM tasks WHERE done = 0 GROUP BY priority"
    ):
        by_priority[prio] = count
    return {
        "total": total,
        "open": open_count,
        "completed": total - open_count,
        "open_by_priority": by_priority,
    }


def cmd_add(con, args):
    cur = con.execute(
        "INSERT INTO tasks (text, priority) VALUES (?, ?)",
        (args.text, args.priority),
    )
    con.commit()
    print(f"Added task {cur.lastrowid} ({args.priority}): {args.text}")


def cmd_list(con, args):
    rows = list_tasks(con, include_done=args.all, priority=args.priority)
    if not rows:
        what = "tasks" if args.all else "open tasks"
        print(f"No {what} found.")
        return
    for row in rows:
        print(format_task(row))


def cmd_done(con, args):
    row = con.execute(
        "SELECT text, done FROM tasks WHERE id = ?", (args.id,)
    ).fetchone()
    if row is None:
        raise TodoError(f"no task with id {args.id}")
    if row[1]:
        print(f"Task {args.id} is already done.")
        return
    con.execute("UPDATE tasks SET done = 1 WHERE id = ?", (args.id,))
    con.commit()
    print(f"Completed task {args.id}: {row[0]}")


def cmd_rm(con, args):
    cur = con.execute("DELETE FROM tasks WHERE id = ?", (args.id,))
    con.commit()
    if cur.rowcount == 0:
        raise TodoError(f"no task with id {args.id}")
    print(f"Removed task {args.id}")


def cmd_stats(con, args):
    stats = get_stats(con)
    print(f"Tasks:     {stats['total']}")
    print(f"Open:      {stats['open']}")
    print(f"Completed: {stats['completed']}")
    print()
    print("Open by priority:")
    for prio in PRIORITIES:
        print(f"  {prio:<4} {stats['open_by_priority'][prio]}")


HANDLERS = {
    "add": cmd_add,
    "list": cmd_list,
    "done": cmd_done,
    "rm": cmd_rm,
    "stats": cmd_stats,
}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="todo", description="A simple command-line todo manager."
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    p_add = sub.add_parser("add", help="add a new task")
    p_add.add_argument("text", help="task description")
    p_add.add_argument(
        "--priority",
        choices=PRIORITIES,
        default="med",
        help="task priority (default: med)",
    )

    p_list = sub.add_parser("list", help="list tasks (open ones by default)")
    p_list.add_argument("--all", action="store_true", help="include completed tasks")
    p_list.add_argument(
        "--priority",
        choices=PRIORITIES,
        help="show only tasks with this priority",
    )

    p_done = sub.add_parser("done", help="mark a task as done")
    p_done.add_argument("id", type=int, help="task id")

    p_rm = sub.add_parser("rm", help="remove a task")
    p_rm.add_argument("id", type=int, help="task id")

    sub.add_parser("stats", help="show task statistics")

    return parser


def main(argv=None, db_path=DEFAULT_DB):
    """Run the CLI. Returns a process exit code (0 on success, non-zero on error)."""
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        # argparse prints a usage message and exits(2) on bad input
        # (bad choice, bad int, missing argument). Map that to a returned code
        # so callers (tests, scripts) see a status instead of an exception.
        code = exc.code
        if code is None:
            return EXIT_OK
        try:
            return int(code)
        except (TypeError, ValueError):
            return EXIT_ERROR

    con = connect(db_path)
    try:
        HANDLERS[args.command](con, args)
        return EXIT_OK
    except TodoError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
