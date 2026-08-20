# todo

A tiny command-line todo manager in a single Python file. Tasks are persisted
in a SQLite database (`todo.db` in the current directory, created on demand).
Standard library only — no third-party dependencies.

## Usage

```
python todo.py <command>
```

## Commands

### add <text> [--priority low|med|high]

Add a new task. The default priority is `med`.

```sh
$ python todo.py add "buy groceries"
Added task 1 (med): buy groceries
$ python todo.py add "ship release" --priority high
Added task 2 (high): ship release
```

### list [--all] [--priority P]

List tasks, sorted by priority (high, then med, then low) and then by id.
Only open tasks are shown by default.

```sh
$ python todo.py list
   2  [ ] high ship release
   1  [ ] med  buy groceries

$ python todo.py list --all          # include completed tasks
$ python todo.py list --priority high   # only high-priority open tasks
```

### done <id>

Mark a task as completed.

```sh
$ python todo.py done 1
Completed task 1: buy groceries
```

### rm <id>

Delete a task (open or completed).

```sh
$ python todo.py rm 1
Removed task 1
```

### stats

Show task counts.

```sh
$ python todo.py stats
Tasks:     4
Open:      3
Completed: 1

Open by priority:
  low  1
  med  2
  high 0
```

## Errors

Bad input (an unknown id, a bad priority, a missing argument, …) prints an
`error:` message to stderr and exits with a non-zero status:

```sh
$ python todo.py done 999
error: no task with id 999
$ python todo.py add "x" --priority urgent
error: argument --priority: invalid choice: 'urgent' (choose from low, med, high)
```

## Tests

```sh
python -m unittest test_todo.py -v
```

Each test uses its own temporary database; your real `todo.db` is never touched.
