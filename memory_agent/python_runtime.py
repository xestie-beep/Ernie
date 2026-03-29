from __future__ import annotations

import os
import shutil


def available_python_commands() -> list[str]:
    """Return likely Python launcher names, preferring ones present on PATH."""
    if os.name == "nt":
        base = ["python", "py", "python3"]
    else:
        base = ["python", "python3", "py"]

    available = [command for command in base if shutil.which(command)]
    ordered = available + [command for command in base if command not in available]
    return list(dict.fromkeys(ordered))


def preferred_python_command() -> str:
    commands = available_python_commands()
    return commands[0] if commands else ("python" if os.name == "nt" else "python3")
