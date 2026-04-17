"""Terminal-assistant subcommand modules.

Each module defines one or more Typer commands that are registered on the
top-level ``app`` in ``gemma/main.py``.  All commands in this package are
stateless by default (no memory read/write) and behave correctly in a pipe.

Modules
-------
shell   -- ``sh`` (natural-language → shell command),
           ``why`` (explain last failed command),
           ``install-shell`` (print/append the shell hook snippet).
explain -- ``explain`` (stdin / file / --cmd / --error modes).
git     -- ``commit`` (generate conventional-commit message),
           ``diff`` (plain-English diff summary).
"""
