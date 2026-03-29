from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import SHELL_COMMAND_TIMEOUT_SECONDS, SHELL_OUTPUT_CHAR_LIMIT

ALLOWED_COMMAND_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("python3", "-m", "memory_agent.cli"),
    ("python3", "-m", "unittest"),
    ("python3", "-m", "pytest"),
    ("python", "-m", "memory_agent.cli"),
    ("python", "-m", "unittest"),
    ("python", "-m", "pytest"),
    ("py", "-m", "memory_agent.cli"),
    ("py", "-m", "unittest"),
    ("py", "-m", "pytest"),
    ("pytest",),
    ("git", "status"),
    ("git", "diff"),
    ("git", "log"),
    ("git", "show"),
    ("git", "branch", "--show-current"),
    ("git", "rev-parse", "--abbrev-ref", "HEAD"),
    ("ruff", "check"),
    ("ruff", "format", "--check"),
)

SHELL_META_TOKENS = ("&&", "||", "|", ";", ">", "<")


@dataclass(slots=True)
class ShellExecutionResult:
    status: str
    command_text: str
    argv: list[str] = field(default_factory=list)
    cwd: str = ""
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "command_text": self.command_text,
            "argv": self.argv,
            "cwd": self.cwd,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "reason": self.reason,
        }


class GuardedShellAdapter:
    def __init__(
        self,
        *,
        workspace_root: Path | None = None,
        timeout_seconds: float = SHELL_COMMAND_TIMEOUT_SECONDS,
        output_char_limit: int = SHELL_OUTPUT_CHAR_LIMIT,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ):
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.timeout_seconds = timeout_seconds
        self.output_char_limit = output_char_limit
        self.runner = runner or subprocess.run

    def execute(
        self,
        command_text: str,
        *,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ShellExecutionResult:
        stripped = command_text.strip()
        if not stripped:
            return ShellExecutionResult(
                status="blocked",
                command_text=command_text,
                reason="empty_command",
            )
        if any(token in stripped for token in SHELL_META_TOKENS):
            return ShellExecutionResult(
                status="blocked",
                command_text=command_text,
                reason="shell_metacharacters_not_allowed",
            )

        try:
            argv = shlex.split(stripped, posix=os.name != "nt")
        except ValueError as exc:
            return ShellExecutionResult(
                status="blocked",
                command_text=command_text,
                reason=f"parse_error:{exc}",
            )
        if not argv:
            return ShellExecutionResult(
                status="blocked",
                command_text=command_text,
                reason="empty_argv",
            )
        if not self._is_allowed_prefix(argv):
            return ShellExecutionResult(
                status="blocked",
                command_text=command_text,
                argv=argv,
                reason="command_prefix_not_allowed",
            )

        resolved_cwd = self._resolve_cwd(cwd)
        if resolved_cwd is None:
            return ShellExecutionResult(
                status="blocked",
                command_text=command_text,
                argv=argv,
                reason="cwd_outside_workspace",
            )

        try:
            completed = self.runner(
                argv,
                cwd=str(resolved_cwd),
                capture_output=True,
                text=True,
                timeout=timeout_seconds or self.timeout_seconds,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ShellExecutionResult(
                status="error",
                command_text=command_text,
                argv=argv,
                cwd=str(resolved_cwd),
                reason="timeout",
                stdout=self._trim_output(exc.stdout or ""),
                stderr=self._trim_output(exc.stderr or ""),
            )
        except OSError as exc:
            return ShellExecutionResult(
                status="error",
                command_text=command_text,
                argv=argv,
                cwd=str(resolved_cwd),
                reason=f"oserror:{exc}",
            )

        return ShellExecutionResult(
            status="success" if completed.returncode == 0 else "error",
            command_text=command_text,
            argv=argv,
            cwd=str(resolved_cwd),
            exit_code=int(completed.returncode),
            stdout=self._trim_output(completed.stdout or ""),
            stderr=self._trim_output(completed.stderr or ""),
            reason="ok" if completed.returncode == 0 else "nonzero_exit",
        )

    def _is_allowed_prefix(self, argv: list[str]) -> bool:
        lowered = [item.lower() for item in argv]
        return any(
            len(lowered) >= len(prefix) and lowered[: len(prefix)] == [item.lower() for item in prefix]
            for prefix in ALLOWED_COMMAND_PREFIXES
        )

    def _resolve_cwd(self, cwd: str | None) -> Path | None:
        candidate = self.workspace_root if not cwd else (self.workspace_root / cwd).resolve() if not Path(cwd).is_absolute() else Path(cwd).resolve()
        try:
            candidate.relative_to(self.workspace_root)
        except ValueError:
            return None
        return candidate

    def _trim_output(self, text: str) -> str:
        stripped = text.strip()
        if len(stripped) <= self.output_char_limit:
            return stripped
        return stripped[: self.output_char_limit - 3] + "..."
