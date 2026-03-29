from __future__ import annotations

import difflib
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import (
    DEFAULT_PATCH_GIT_BRANCH_PREFIX,
    DEFAULT_PATCH_RUNS_DIR,
    PATCH_RUNNER_OUTPUT_CHAR_LIMIT,
    SHELL_COMMAND_TIMEOUT_SECONDS,
)
from .file_adapter import FileOperationResult, WorkspaceFileAdapter
from .memory import MemoryStore
from .models import MemoryRecord
from .shell_adapter import GuardedShellAdapter, ShellExecutionResult

PATCH_IGNORE_NAMES = {
    ".agent",
    ".eval_tmp",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".test_tmp",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
}
PATCH_IGNORE_PREFIXES = ("tmp",)


@dataclass(slots=True)
class PatchOperation:
    operation: str
    path: str
    text: str | None = None
    find_text: str | None = None
    symbol_name: str | None = None
    replace_all: bool = False
    cwd: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "path": self.path,
            "text": self.text,
            "find_text": self.find_text,
            "symbol_name": self.symbol_name,
            "replace_all": self.replace_all,
            "cwd": self.cwd,
        }


@dataclass(slots=True)
class PatchValidationResult:
    kind: str
    name: str
    status: str
    passed: bool
    details: str
    command_text: str = ""
    result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "status": self.status,
            "passed": self.passed,
            "details": self.details,
            "command_text": self.command_text,
            "result": self.result,
        }


@dataclass(slots=True)
class PatchRunReport:
    run_name: str
    suite_name: str
    status: str
    task_title: str | None = None
    run_id: int | None = None
    apply_on_success: bool = False
    applied: bool = False
    temp_workspace: str = ""
    changed_files: list[str] = field(default_factory=list)
    diff_preview: str = ""
    operations: list[PatchOperation] = field(default_factory=list)
    operation_results: list[FileOperationResult] = field(default_factory=list)
    validations: list[PatchValidationResult] = field(default_factory=list)
    baseline_evaluation: dict[str, Any] | None = None
    candidate_evaluation: dict[str, Any] | None = None
    rejection_reason: str = ""
    tool_outcome: MemoryRecord | None = None
    task_update: MemoryRecord | None = None
    preview_only: bool = False
    git_apply: GitApplyResult | None = None

    def render(self) -> str:
        lines = [f"Patch run [{self.status}] {self.run_name}", f"Suite: {self.suite_name}"]
        if self.task_title:
            lines.append(f"Task: {self.task_title}")
        if self.temp_workspace:
            lines.append(f"Workspace: {self.temp_workspace}")
        if self.baseline_evaluation is not None:
            lines.append(
                f"Baseline score: {float(self.baseline_evaluation.get('score', 0.0) or 0.0):.1%}"
            )
        if self.candidate_evaluation is not None:
            lines.append(
                "Candidate score: "
                f"{float(self.candidate_evaluation.get('score', 0.0) or 0.0):.1%}"
            )
        if self.changed_files:
            lines.append("Changed files: " + ", ".join(self.changed_files))
        if self.git_apply is not None and self.git_apply.status != "not_requested":
            lines.append(
                f"Git apply: [{self.git_apply.status}] "
                f"{self.git_apply.message or 'available'}"
            )
            if self.git_apply.branch_name:
                lines.append(f"Git branch: {self.git_apply.branch_name}")
            if self.git_apply.commit:
                lines.append(f"Git commit: {self.git_apply.commit}")
            if self.git_apply.rollback_hint:
                lines.append(f"Rollback: {self.git_apply.rollback_hint}")
        if self.diff_preview:
            lines.extend(["", "Diff preview:", self.diff_preview])
        if self.rejection_reason:
            lines.append("Reason: " + self.rejection_reason)
        if self.validations:
            lines.append("")
            lines.append("Validations:")
            for validation in self.validations:
                lines.append(
                    f"- [{validation.status}] {validation.name}: {validation.details}"
                )
        return "\n".join(lines)


@dataclass(slots=True)
class GitApplyResult:
    status: str = "not_requested"
    repo_root: str = ""
    branch_name: str | None = None
    original_branch: str | None = None
    commit: str | None = None
    message: str = ""
    rollback_ready: bool = False
    rollback_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "repo_root": self.repo_root,
            "branch_name": self.branch_name,
            "original_branch": self.original_branch,
            "commit": self.commit,
            "message": self.message,
            "rollback_ready": self.rollback_ready,
            "rollback_hint": self.rollback_hint,
        }


@dataclass(slots=True)
class PatchRollbackReport:
    status: str
    patch_run_id: int | None = None
    repo_root: str = ""
    branch_name: str | None = None
    original_branch: str | None = None
    commit: str | None = None
    switched_to: str | None = None
    deleted_branch: bool = False
    reason: str = ""
    tool_outcome: MemoryRecord | None = None

    def render(self) -> str:
        lines = [f"Patch rollback [{self.status}]"]
        if self.patch_run_id is not None:
            lines.append(f"Patch run id: {self.patch_run_id}")
        if self.branch_name:
            lines.append(f"Branch: {self.branch_name}")
        if self.original_branch:
            lines.append(f"Original branch: {self.original_branch}")
        if self.switched_to:
            lines.append(f"Switched to: {self.switched_to}")
        if self.commit:
            lines.append(f"Commit: {self.commit}")
        if self.deleted_branch:
            lines.append("Deleted disposable branch.")
        if self.reason:
            lines.append(f"Reason: {self.reason}")
        return "\n".join(lines)


@dataclass(slots=True)
class GitRepoState:
    repo_root: Path
    current_branch: str
    head_commit: str
    clean_worktree: bool


class WorkspacePatchRunner:
    def __init__(
        self,
        memory_store: MemoryStore,
        *,
        workspace_root: Path | None = None,
        runs_root: Path | None = None,
        suite_name: str = "builtin",
        output_char_limit: int = PATCH_RUNNER_OUTPUT_CHAR_LIMIT,
        shell_runner: Callable[..., Any] | None = None,
        git_mode: str = "auto",
        git_branch_prefix: str = DEFAULT_PATCH_GIT_BRANCH_PREFIX,
    ):
        self.memory_store = memory_store
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        configured_runs_root = runs_root or DEFAULT_PATCH_RUNS_DIR
        self.runs_root = (
            configured_runs_root
            if Path(configured_runs_root).is_absolute()
            else (self.workspace_root / configured_runs_root)
        ).resolve()
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.suite_name = suite_name
        self.output_char_limit = output_char_limit
        self.shell_runner = shell_runner
        self.git_mode = git_mode
        self.git_branch_prefix = str(git_branch_prefix or DEFAULT_PATCH_GIT_BRANCH_PREFIX).strip() or DEFAULT_PATCH_GIT_BRANCH_PREFIX

    def preview(
        self,
        run_name: str,
        *,
        operations: list[PatchOperation],
        task_title: str | None = None,
        suite_name: str | None = None,
    ) -> PatchRunReport:
        selected_suite = str(suite_name or self.suite_name or "builtin").strip() or "builtin"
        run_slug = self._slugify(run_name)
        run_dir = self.runs_root / f"{run_slug}_{uuid.uuid4().hex[:10]}"
        temp_workspace = run_dir / "workspace"
        self._copy_workspace(temp_workspace)

        file_adapter = WorkspaceFileAdapter(
            workspace_root=temp_workspace,
            preview_char_limit=self.output_char_limit,
        )

        operation_results: list[FileOperationResult] = []
        changed_files: list[str] = []
        rejection_reason = ""
        status = "accepted"

        if not operations:
            status = "rejected"
            rejection_reason = "no_operations"
        else:
            for operation in operations:
                result = file_adapter.execute(
                    operation.operation,
                    operation.path,
                    text=operation.text,
                    find_text=operation.find_text,
                    symbol_name=operation.symbol_name,
                    replace_all=operation.replace_all,
                    cwd=operation.cwd,
                )
                operation_results.append(result)
                if result.status != "success":
                    status = "rejected"
                    rejection_reason = (
                        f"operation_{result.status}:{operation.operation}:{result.reason}"
                    )
                    break
                if result.changed:
                    changed_result_paths = list(result.changed_paths) or [result.path]
                    for changed_path in changed_result_paths:
                        relative_path = self._relative_to_workspace(
                            Path(changed_path),
                            temp_workspace,
                        )
                        if relative_path and relative_path not in changed_files:
                            changed_files.append(relative_path)

        if status == "accepted" and not changed_files:
            status = "rejected"
            rejection_reason = "no_changed_files"

        return PatchRunReport(
            run_name=run_name,
            suite_name=selected_suite,
            status=status,
            task_title=task_title,
            apply_on_success=False,
            applied=False,
            temp_workspace=str(temp_workspace),
            changed_files=changed_files,
            diff_preview=self._build_diff_preview(temp_workspace, changed_files),
            operations=list(operations),
            operation_results=operation_results,
            rejection_reason=rejection_reason,
            preview_only=True,
        )

    def run(
        self,
        run_name: str,
        *,
        operations: list[PatchOperation],
        validation_commands: list[str] | None = None,
        apply_on_success: bool = False,
        task_title: str | None = None,
        task_area: str = "self_improvement",
        suite_name: str | None = None,
        git_mode: str | None = None,
    ) -> PatchRunReport:
        selected_suite = str(suite_name or self.suite_name or "builtin").strip() or "builtin"
        baseline_evaluation = self._ensure_baseline_evaluation(selected_suite)
        run_slug = self._slugify(run_name)
        run_dir = self.runs_root / f"{run_slug}_{uuid.uuid4().hex[:10]}"
        temp_workspace = run_dir / "workspace"
        self._copy_workspace(temp_workspace)

        file_adapter = WorkspaceFileAdapter(
            workspace_root=temp_workspace,
            preview_char_limit=self.output_char_limit,
        )
        shell_adapter = GuardedShellAdapter(
            workspace_root=temp_workspace,
            output_char_limit=self.output_char_limit,
            runner=self.shell_runner,
        )

        operation_results: list[FileOperationResult] = []
        validations: list[PatchValidationResult] = []
        changed_files: list[str] = []
        candidate_evaluation: dict[str, Any] | None = None
        applied = False
        task_update: MemoryRecord | None = None
        rejection_reason = ""
        status = "accepted"
        git_apply = GitApplyResult(status="not_requested", message="git branch apply disabled")

        if not operations:
            status = "rejected"
            rejection_reason = "no_operations"
        else:
            for operation in operations:
                result = file_adapter.execute(
                    operation.operation,
                    operation.path,
                    text=operation.text,
                    find_text=operation.find_text,
                    symbol_name=operation.symbol_name,
                    replace_all=operation.replace_all,
                    cwd=operation.cwd,
                )
                operation_results.append(result)
                if result.status != "success":
                    status = "rejected"
                    rejection_reason = (
                        f"operation_{result.status}:{operation.operation}:{result.reason}"
                    )
                    break
                if result.changed:
                    changed_result_paths = list(result.changed_paths) or [result.path]
                    for changed_path in changed_result_paths:
                        relative_path = self._relative_to_workspace(
                            Path(changed_path),
                            temp_workspace,
                        )
                        if relative_path and relative_path not in changed_files:
                            changed_files.append(relative_path)

        if status == "accepted" and not changed_files:
            status = "rejected"
            rejection_reason = "no_changed_files"

        commands = (
            list(validation_commands)
            if validation_commands is not None
            else self._default_validation_commands()
        )
        if status == "accepted":
            for command in commands:
                shell_result = shell_adapter.execute(command)
                validation = self._shell_validation_result(command, shell_result)
                validations.append(validation)
                if not validation.passed:
                    status = "rejected"
                    rejection_reason = f"validation_failed:{command}"
                    break

        if status == "accepted":
            evaluation_validation, candidate_evaluation = self._run_candidate_evaluation(
                shell_adapter=shell_adapter,
                suite_name=selected_suite,
            )
            validations.append(evaluation_validation)
            if not evaluation_validation.passed:
                status = "rejected"
                rejection_reason = evaluation_validation.details or "candidate_evaluation_failed"

        if (
            status == "accepted"
            and baseline_evaluation is not None
            and candidate_evaluation is not None
            and float(candidate_evaluation.get("score", 0.0) or 0.0)
            + 1e-9
            < float(baseline_evaluation.get("score", 0.0) or 0.0)
        ):
            status = "rejected"
            rejection_reason = "candidate_score_below_baseline"

        if status == "accepted" and apply_on_success:
            try:
                git_apply = self._apply_candidate(
                    temp_workspace,
                    changed_files,
                    run_name=run_name,
                    backups_root=run_dir / "backups",
                    git_mode=git_mode,
                )
                applied = True
                status = "applied"
                if task_title:
                    active_task = self.memory_store.find_active_task(task_title, area=task_area)
                    if active_task is not None:
                        completion = self.memory_store.complete_task(task_title, area=task_area)
                        task_update = completion["completed"]
            except OSError as exc:
                status = "error"
                rejection_reason = f"apply_failed:{exc}"

        report = PatchRunReport(
            run_name=run_name,
            suite_name=selected_suite,
            status=status,
            task_title=task_title,
            apply_on_success=apply_on_success,
            applied=applied,
            temp_workspace=str(temp_workspace),
            changed_files=changed_files,
            diff_preview=self._build_diff_preview(temp_workspace, changed_files),
            operations=list(operations),
            operation_results=operation_results,
            validations=validations,
            baseline_evaluation=baseline_evaluation,
            candidate_evaluation=candidate_evaluation,
            rejection_reason=rejection_reason,
            task_update=task_update,
            git_apply=git_apply,
        )
        patch_run = self.memory_store.record_patch_run(
            run_name=run_name,
            suite_name=selected_suite,
            task_title=task_title,
            status=status,
            baseline_evaluation=baseline_evaluation,
            candidate_evaluation=candidate_evaluation,
            apply_on_success=apply_on_success,
            applied=applied,
            workspace_path=str(temp_workspace),
            changed_files=changed_files,
            operation_results=[item.to_dict() for item in operation_results],
            validation_results=[item.to_dict() for item in validations],
            summary={
                "rejection_reason": rejection_reason,
                "task_area": task_area,
                "task_completed": task_update is not None,
                "operation_count": len(operations),
                "git": git_apply.to_dict(),
            },
        )
        report.run_id = int(patch_run["id"])
        report.tool_outcome = self.memory_store.record_tool_outcome(
            "patch-run",
            self._tool_outcome_text(report),
            status=self._tool_outcome_status(report.status),
            subject="self_improvement",
            tags=["patch-run", selected_suite, report.status],
            metadata={
                "patch_run_id": report.run_id,
                "task_title": task_title,
                "changed_files": list(changed_files),
                "candidate_score": (
                    float(candidate_evaluation.get("score", 0.0) or 0.0)
                    if candidate_evaluation is not None
                    else None
                ),
                "baseline_score": (
                    float(baseline_evaluation.get("score", 0.0) or 0.0)
                    if baseline_evaluation is not None
                    else None
                ),
                "rejection_reason": rejection_reason,
                "git_apply": git_apply.to_dict(),
            },
        )
        return report

    def apply_preview(
        self,
        preview_report: PatchRunReport,
        *,
        validation_commands: list[str] | None = None,
        task_title: str | None = None,
        task_area: str = "execution",
        suite_name: str | None = None,
        git_mode: str | None = None,
    ) -> PatchRunReport:
        if preview_report.status != "accepted":
            raise ValueError("Only accepted preview reports can be applied.")
        temp_workspace = Path(preview_report.temp_workspace).resolve()
        if not temp_workspace.exists():
            raise FileNotFoundError(f"Missing preview workspace: {temp_workspace}")

        selected_suite = str(suite_name or preview_report.suite_name or self.suite_name or "builtin").strip() or "builtin"
        baseline_evaluation = self._ensure_baseline_evaluation(selected_suite)
        shell_adapter = GuardedShellAdapter(
            workspace_root=temp_workspace,
            output_char_limit=self.output_char_limit,
            runner=self.shell_runner,
        )
        validations: list[PatchValidationResult] = []
        candidate_evaluation: dict[str, Any] | None = None
        rejection_reason = ""
        status = "accepted"
        applied = False
        task_update: MemoryRecord | None = None
        resolved_task_title = task_title or preview_report.task_title
        git_apply = GitApplyResult(status="not_requested", message="git branch apply disabled")

        commands = (
            list(validation_commands)
            if validation_commands is not None
            else self._default_validation_commands()
        )
        for command in commands:
            shell_result = shell_adapter.execute(command)
            validation = self._shell_validation_result(command, shell_result)
            validations.append(validation)
            if not validation.passed:
                status = "rejected"
                rejection_reason = f"validation_failed:{command}"
                break

        if status == "accepted":
            evaluation_validation, candidate_evaluation = self._run_candidate_evaluation(
                shell_adapter=shell_adapter,
                suite_name=selected_suite,
            )
            validations.append(evaluation_validation)
            if not evaluation_validation.passed:
                status = "rejected"
                rejection_reason = evaluation_validation.details or "candidate_evaluation_failed"

        if (
            status == "accepted"
            and baseline_evaluation is not None
            and candidate_evaluation is not None
            and float(candidate_evaluation.get("score", 0.0) or 0.0)
            + 1e-9
            < float(baseline_evaluation.get("score", 0.0) or 0.0)
        ):
            status = "rejected"
            rejection_reason = "candidate_score_below_baseline"

        if status == "accepted":
            try:
                git_apply = self._apply_candidate(
                    temp_workspace,
                    list(preview_report.changed_files),
                    run_name=preview_report.run_name,
                    backups_root=temp_workspace.parent / "backups_apply",
                    git_mode=git_mode,
                )
                applied = True
                status = "applied"
                if resolved_task_title:
                    active_task = self.memory_store.find_active_task(
                        resolved_task_title,
                        area=task_area,
                    )
                    if active_task is not None:
                        completion = self.memory_store.complete_task(
                            resolved_task_title,
                            area=task_area,
                        )
                        task_update = completion["completed"]
            except OSError as exc:
                status = "error"
                rejection_reason = f"apply_failed:{exc}"

        report = PatchRunReport(
            run_name=preview_report.run_name,
            suite_name=selected_suite,
            status=status,
            task_title=resolved_task_title,
            apply_on_success=True,
            applied=applied,
            temp_workspace=preview_report.temp_workspace,
            changed_files=list(preview_report.changed_files),
            diff_preview=preview_report.diff_preview,
            operations=list(preview_report.operations),
            operation_results=list(preview_report.operation_results),
            validations=validations,
            baseline_evaluation=baseline_evaluation,
            candidate_evaluation=candidate_evaluation,
            rejection_reason=rejection_reason,
            task_update=task_update,
            git_apply=git_apply,
        )
        patch_run = self.memory_store.record_patch_run(
            run_name=report.run_name,
            suite_name=selected_suite,
            task_title=resolved_task_title,
            status=status,
            baseline_evaluation=baseline_evaluation,
            candidate_evaluation=candidate_evaluation,
            apply_on_success=True,
            applied=applied,
            workspace_path=str(temp_workspace),
            changed_files=report.changed_files,
            operation_results=[item.to_dict() for item in report.operation_results],
            validation_results=[item.to_dict() for item in validations],
            summary={
                "rejection_reason": rejection_reason,
                "task_area": task_area,
                "task_completed": task_update is not None,
                "operation_count": len(report.operations),
                "preview_only": False,
                "preview_diff": report.diff_preview,
                "git": git_apply.to_dict(),
            },
        )
        report.run_id = int(patch_run["id"])
        report.tool_outcome = self.memory_store.record_tool_outcome(
            "patch-run",
            self._tool_outcome_text(report),
            status=self._tool_outcome_status(report.status),
            subject="self_improvement",
            tags=["patch-run", selected_suite, report.status],
            metadata={
                "patch_run_id": report.run_id,
                "task_title": resolved_task_title,
                "changed_files": list(report.changed_files),
                "candidate_score": (
                    float(candidate_evaluation.get("score", 0.0) or 0.0)
                    if candidate_evaluation is not None
                    else None
                ),
                "baseline_score": (
                    float(baseline_evaluation.get("score", 0.0) or 0.0)
                    if baseline_evaluation is not None
                    else None
                ),
                "rejection_reason": rejection_reason,
                "git_apply": git_apply.to_dict(),
            },
        )
        return report

    def _ensure_baseline_evaluation(self, suite_name: str) -> dict[str, Any] | None:
        baseline = self.memory_store.latest_evaluation_run(suite_name=suite_name)
        if baseline is not None:
            return baseline
        from .evaluation import MemoryEvaluator

        report = MemoryEvaluator(self.workspace_root).run_builtin_suite()
        return self.memory_store.record_evaluation_run(suite_name, report)

    def _copy_workspace(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            self.workspace_root,
            destination,
            ignore=self._copytree_ignore,
        )

    def _shell_validation_result(
        self,
        command: str,
        shell_result: ShellExecutionResult,
    ) -> PatchValidationResult:
        details = shell_result.reason or shell_result.status
        if shell_result.exit_code is not None:
            details = f"{details}; exit={shell_result.exit_code}"
        if shell_result.stderr:
            details = f"{details}; stderr={shell_result.stderr}"
        elif shell_result.stdout:
            details = f"{details}; stdout={shell_result.stdout}"
        return PatchValidationResult(
            kind="shell",
            name=f"validate:{command}",
            status=shell_result.status,
            passed=shell_result.status == "success",
            details=details,
            command_text=command,
            result=shell_result.to_dict(),
        )

    def _run_candidate_evaluation(
        self,
        *,
        shell_adapter: GuardedShellAdapter,
        suite_name: str,
    ) -> tuple[PatchValidationResult, dict[str, Any] | None]:
        command = (
            f"{self._default_python_command()} -m memory_agent.cli "
            "--db .agent/patch_candidate.sqlite3 evaluate --json"
        )
        shell_result = shell_adapter.execute(command)
        if shell_result.status != "success":
            validation = PatchValidationResult(
                kind="evaluation",
                name=f"evaluate:{suite_name}",
                status=shell_result.status,
                passed=False,
                details=shell_result.reason or "candidate_evaluation_command_failed",
                command_text=command,
                result={"shell": shell_result.to_dict()},
            )
            return validation, None
        try:
            payload = json.loads(shell_result.stdout)
        except json.JSONDecodeError as exc:
            validation = PatchValidationResult(
                kind="evaluation",
                name=f"evaluate:{suite_name}",
                status="error",
                passed=False,
                details=f"candidate_evaluation_parse_error:{exc}",
                command_text=command,
                result={"shell": shell_result.to_dict()},
            )
            return validation, None
        evaluation = self._normalize_evaluation_payload(payload, suite_name=suite_name)
        score = float(evaluation.get("score", 0.0) or 0.0)
        scenarios_passed = int(evaluation.get("scenarios_passed", 0) or 0)
        scenarios_total = int(evaluation.get("scenarios_total", 0) or 0)
        details = f"score={score:.1%}; scenarios={scenarios_passed}/{scenarios_total}"
        validation = PatchValidationResult(
            kind="evaluation",
            name=f"evaluate:{suite_name}",
            status="success" if bool(evaluation.get("passed")) else "error",
            passed=bool(evaluation.get("passed")),
            details=details,
            command_text=command,
            result={"shell": shell_result.to_dict(), "evaluation": evaluation},
        )
        return validation, evaluation

    def _normalize_evaluation_payload(
        self,
        payload: dict[str, Any],
        *,
        suite_name: str,
    ) -> dict[str, Any]:
        scenario_results = list(payload.get("scenario_results", []))
        scenarios_total = len(scenario_results)
        scenarios_passed = sum(1 for item in scenario_results if bool(item.get("passed")))
        checks_total = sum(len(item.get("checks", [])) for item in scenario_results)
        checks_passed = sum(
            1
            for item in scenario_results
            for check in item.get("checks", [])
            if bool(check.get("passed"))
        )
        return {
            "suite_name": suite_name,
            "score": float(payload.get("score", 0.0) or 0.0),
            "passed": bool(payload.get("passed")),
            "scenarios_passed": scenarios_passed,
            "scenarios_total": scenarios_total,
            "checks_passed": checks_passed,
            "checks_total": checks_total,
            "summary": payload,
        }

    def _apply_changed_files(
        self,
        temp_workspace: Path,
        changed_files: list[str],
        backups_root: Path,
    ) -> None:
        backups: list[tuple[Path, bool, Path | None]] = []
        try:
            for relative_path in changed_files:
                source = temp_workspace / relative_path
                target = self.workspace_root / relative_path
                if not source.exists():
                    raise OSError(f"missing_changed_file:{relative_path}")
                target.parent.mkdir(parents=True, exist_ok=True)
                backup_path: Path | None = None
                existed = target.exists()
                if existed:
                    backup_path = backups_root / relative_path
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target, backup_path)
                shutil.copy2(source, target)
                backups.append((target, existed, backup_path))
        except OSError:
            for target, existed, backup_path in reversed(backups):
                if existed and backup_path is not None and backup_path.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_path, target)
                elif not existed and target.exists():
                    target.unlink()
            raise

    def _apply_candidate(
        self,
        temp_workspace: Path,
        changed_files: list[str],
        *,
        run_name: str,
        backups_root: Path,
        git_mode: str | None,
    ) -> GitApplyResult:
        resolved_git_mode = self._resolve_git_mode(git_mode)
        if resolved_git_mode == "off":
            self._apply_changed_files(temp_workspace, changed_files, backups_root)
            return GitApplyResult(
                status="not_requested",
                message="git branch apply disabled",
            )

        git_state = self._inspect_git_repo()
        if git_state is None:
            if resolved_git_mode == "branch":
                raise OSError("git_branch_required:not_a_git_repo")
            self._apply_changed_files(temp_workspace, changed_files, backups_root)
            return GitApplyResult(
                status="direct_apply",
                message="git branch apply skipped because no git repo was detected",
            )

        if not git_state.clean_worktree:
            if resolved_git_mode == "branch":
                raise OSError("git_branch_required:dirty_worktree")
            self._apply_changed_files(temp_workspace, changed_files, backups_root)
            return GitApplyResult(
                status="direct_apply",
                repo_root=str(git_state.repo_root),
                original_branch=git_state.current_branch,
                commit=git_state.head_commit,
                message="git branch apply skipped because the worktree is not clean",
            )

        if git_state.current_branch == "HEAD":
            if resolved_git_mode == "branch":
                raise OSError("git_branch_required:detached_head")
            self._apply_changed_files(temp_workspace, changed_files, backups_root)
            return GitApplyResult(
                status="direct_apply",
                repo_root=str(git_state.repo_root),
                commit=git_state.head_commit,
                message="git branch apply skipped because the repo is in detached HEAD state",
            )

        return self._apply_changed_files_on_git_branch(
            temp_workspace,
            changed_files,
            run_name=run_name,
            git_state=git_state,
            backups_root=backups_root,
        )

    def _apply_changed_files_on_git_branch(
        self,
        temp_workspace: Path,
        changed_files: list[str],
        *,
        run_name: str,
        git_state: GitRepoState,
        backups_root: Path,
    ) -> GitApplyResult:
        branch_name = self._build_git_branch_name(run_name)
        repo_relative_files = self._repo_relative_changed_files(
            changed_files,
            repo_root=git_state.repo_root,
        )
        checkout_result = self._run_process(
            ["git", "checkout", "-b", branch_name],
            cwd=git_state.repo_root,
        )
        if checkout_result.status != "success":
            raise OSError(
                "git_branch_create_failed:"
                f"{checkout_result.stderr or checkout_result.reason or checkout_result.status}"
            )

        try:
            self._apply_changed_files(
                temp_workspace,
                changed_files,
                backups_root / "git_branch_backups" / uuid.uuid4().hex,
            )
            add_result = self._run_process(
                ["git", "add", "--", *repo_relative_files],
                cwd=git_state.repo_root,
            )
            if add_result.status != "success":
                raise OSError(
                    "git_add_failed:"
                    f"{add_result.stderr or add_result.reason or add_result.status}"
                )
            commit_message = f"codex: apply patch run {run_name}"
            commit_result = self._run_process(
                [
                    "git",
                    "-c",
                    "user.name=Codex",
                    "-c",
                    "user.email=codex@local.invalid",
                    "commit",
                    "--no-verify",
                    "-m",
                    commit_message,
                ],
                cwd=git_state.repo_root,
            )
            if commit_result.status != "success":
                raise OSError(
                    "git_commit_failed:"
                    f"{commit_result.stderr or commit_result.reason or commit_result.status}"
                )
            head_result = self._run_process(
                ["git", "rev-parse", "HEAD"],
                cwd=git_state.repo_root,
            )
            commit = head_result.stdout.strip() if head_result.status == "success" else ""
            rollback_hint = self._build_git_rollback_hint(
                original_branch=git_state.current_branch,
                branch_name=branch_name,
            )
            return GitApplyResult(
                status="applied",
                repo_root=str(git_state.repo_root),
                branch_name=branch_name,
                original_branch=git_state.current_branch,
                commit=commit or None,
                message="candidate committed to a disposable git branch",
                rollback_ready=bool(commit),
                rollback_hint=rollback_hint,
            )
        except OSError:
            self._restore_git_branch_state(
                repo_root=git_state.repo_root,
                original_branch=git_state.current_branch,
                branch_name=branch_name,
            )
            raise

    def rollback(
        self,
        patch_run_id: int | None = None,
        *,
        force: bool = False,
    ) -> PatchRollbackReport:
        patch_run = (
            self.memory_store.get_patch_run(int(patch_run_id))
            if patch_run_id is not None
            else self._latest_rollback_ready_patch_run()
        )
        if patch_run is None:
            return self._record_rollback_outcome(
                PatchRollbackReport(
                    status="unavailable",
                    reason="no_git_backed_patch_run_found",
                )
            )
        git_payload = dict((patch_run.get("summary") or {}).get("git") or {})
        branch_name = str(git_payload.get("branch_name") or "").strip() or None
        original_branch = str(git_payload.get("original_branch") or "").strip() or None
        repo_root_text = str(git_payload.get("repo_root") or "").strip()
        commit = str(git_payload.get("commit") or "").strip() or None
        if not branch_name or not repo_root_text:
            return self._record_rollback_outcome(
                PatchRollbackReport(
                    status="unavailable",
                    patch_run_id=int(patch_run["id"]),
                    reason="patch_run_has_no_git_branch_metadata",
                )
            )

        repo_root = Path(repo_root_text).resolve()
        repo_check = self._run_process(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=repo_root,
        )
        if repo_check.status != "success":
            return self._record_rollback_outcome(
                PatchRollbackReport(
                    status="error",
                    patch_run_id=int(patch_run["id"]),
                    repo_root=str(repo_root),
                    branch_name=branch_name,
                    original_branch=original_branch,
                    commit=commit,
                    reason="git_repo_unavailable_for_rollback",
                )
            )

        current_branch_result = self._run_process(
            ["git", "branch", "--show-current"],
            cwd=repo_root,
        )
        current_branch = current_branch_result.stdout.strip()
        branch_head_result = self._run_process(
            ["git", "rev-parse", branch_name],
            cwd=repo_root,
        )
        if branch_head_result.status != "success":
            return self._record_rollback_outcome(
                PatchRollbackReport(
                    status="unavailable",
                    patch_run_id=int(patch_run["id"]),
                    repo_root=str(repo_root),
                    branch_name=branch_name,
                    original_branch=original_branch,
                    commit=commit,
                    reason="git_branch_missing",
                )
            )
        branch_head = branch_head_result.stdout.strip()
        if commit and branch_head and branch_head != commit and not force:
            return self._record_rollback_outcome(
                PatchRollbackReport(
                    status="blocked",
                    patch_run_id=int(patch_run["id"]),
                    repo_root=str(repo_root),
                    branch_name=branch_name,
                    original_branch=original_branch,
                    commit=commit,
                    reason="git_branch_head_changed",
                )
            )

        switched_to: str | None = None
        if current_branch == branch_name:
            worktree_result = self._run_process(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
            )
            if worktree_result.status != "success":
                return self._record_rollback_outcome(
                    PatchRollbackReport(
                        status="error",
                        patch_run_id=int(patch_run["id"]),
                        repo_root=str(repo_root),
                        branch_name=branch_name,
                        original_branch=original_branch,
                        commit=commit,
                        reason="git_status_failed_before_rollback",
                    )
                )
            if worktree_result.stdout.strip():
                return self._record_rollback_outcome(
                    PatchRollbackReport(
                        status="blocked",
                        patch_run_id=int(patch_run["id"]),
                        repo_root=str(repo_root),
                        branch_name=branch_name,
                        original_branch=original_branch,
                        commit=commit,
                        reason="git_worktree_dirty_on_branch_to_rollback",
                    )
                )
            if not original_branch:
                return self._record_rollback_outcome(
                    PatchRollbackReport(
                        status="error",
                        patch_run_id=int(patch_run["id"]),
                        repo_root=str(repo_root),
                        branch_name=branch_name,
                        commit=commit,
                        reason="missing_original_branch_for_rollback",
                    )
                )
            switch_result = self._run_process(
                ["git", "checkout", original_branch],
                cwd=repo_root,
            )
            if switch_result.status != "success":
                return self._record_rollback_outcome(
                    PatchRollbackReport(
                        status="error",
                        patch_run_id=int(patch_run["id"]),
                        repo_root=str(repo_root),
                        branch_name=branch_name,
                        original_branch=original_branch,
                        commit=commit,
                        reason="git_checkout_original_branch_failed",
                    )
                )
            switched_to = original_branch

        delete_result = self._run_process(
            ["git", "branch", "-D", branch_name],
            cwd=repo_root,
        )
        if delete_result.status != "success":
            return self._record_rollback_outcome(
                PatchRollbackReport(
                    status="error",
                    patch_run_id=int(patch_run["id"]),
                    repo_root=str(repo_root),
                    branch_name=branch_name,
                    original_branch=original_branch,
                    commit=commit,
                    switched_to=switched_to,
                    reason="git_delete_branch_failed",
                )
            )
        return self._record_rollback_outcome(
            PatchRollbackReport(
                status="rolled_back",
                patch_run_id=int(patch_run["id"]),
                repo_root=str(repo_root),
                branch_name=branch_name,
                original_branch=original_branch,
                commit=commit,
                switched_to=switched_to,
                deleted_branch=True,
                reason="git_branch_deleted",
            )
        )

    def _inspect_git_repo(self) -> GitRepoState | None:
        repo_result = self._run_process(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=self.workspace_root,
        )
        if repo_result.status != "success":
            return None
        repo_root = Path(repo_result.stdout.strip()).resolve()
        try:
            self.workspace_root.relative_to(repo_root)
        except ValueError:
            return None
        branch_result = self._run_process(
            ["git", "branch", "--show-current"],
            cwd=repo_root,
        )
        if branch_result.status != "success":
            return None
        head_result = self._run_process(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
        )
        if head_result.status != "success":
            return None
        status_result = self._run_process(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
        )
        if status_result.status != "success":
            return None
        return GitRepoState(
            repo_root=repo_root,
            current_branch=branch_result.stdout.strip() or "HEAD",
            head_commit=head_result.stdout.strip(),
            clean_worktree=not status_result.stdout.strip(),
        )

    def _restore_git_branch_state(
        self,
        *,
        repo_root: Path,
        original_branch: str,
        branch_name: str,
    ) -> None:
        self._run_process(["git", "reset", "--hard", "HEAD"], cwd=repo_root)
        self._run_process(["git", "checkout", original_branch], cwd=repo_root)
        self._run_process(["git", "branch", "-D", branch_name], cwd=repo_root)

    def _build_git_branch_name(self, run_name: str) -> str:
        suffix = uuid.uuid4().hex[:8]
        slug = self._slugify(run_name)
        prefix = self.git_branch_prefix.rstrip("/") or DEFAULT_PATCH_GIT_BRANCH_PREFIX
        return f"{prefix}/{slug}-{suffix}"

    def _build_git_rollback_hint(
        self,
        *,
        original_branch: str,
        branch_name: str,
    ) -> str:
        return (
            f"git checkout {original_branch} && git branch -D {branch_name}"
        )

    def _repo_relative_changed_files(
        self,
        changed_files: list[str],
        *,
        repo_root: Path,
    ) -> list[str]:
        try:
            workspace_prefix = self.workspace_root.relative_to(repo_root)
        except ValueError:
            workspace_prefix = Path()
        return [
            (workspace_prefix / relative_path).as_posix()
            if str(workspace_prefix) not in {"", "."}
            else relative_path
            for relative_path in changed_files
        ]

    def _run_process(
        self,
        argv: list[str],
        *,
        cwd: Path,
    ) -> ShellExecutionResult:
        command_text = " ".join(str(part) for part in argv)
        runner = self.shell_runner or subprocess.run
        try:
            completed = runner(
                argv,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=SHELL_COMMAND_TIMEOUT_SECONDS,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ShellExecutionResult(
                status="error",
                command_text=command_text,
                argv=[str(part) for part in argv],
                cwd=str(cwd),
                reason="timeout",
                stdout=self._trim_preview(exc.stdout or ""),
                stderr=self._trim_preview(exc.stderr or ""),
            )
        except OSError as exc:
            return ShellExecutionResult(
                status="error",
                command_text=command_text,
                argv=[str(part) for part in argv],
                cwd=str(cwd),
                reason=f"oserror:{exc}",
            )
        return ShellExecutionResult(
            status="success" if completed.returncode == 0 else "error",
            command_text=command_text,
            argv=[str(part) for part in argv],
            cwd=str(cwd),
            exit_code=int(completed.returncode),
            stdout=self._trim_preview(completed.stdout or ""),
            stderr=self._trim_preview(completed.stderr or ""),
            reason="ok" if completed.returncode == 0 else "nonzero_exit",
        )

    def _resolve_git_mode(self, git_mode: str | None) -> str:
        candidate = str(git_mode or self.git_mode or "auto").strip().lower()
        if candidate not in {"auto", "branch", "off"}:
            return "auto"
        return candidate

    def _latest_rollback_ready_patch_run(self) -> dict[str, Any] | None:
        offset = 0
        while True:
            candidate = self.memory_store.latest_patch_run(offset=offset)
            if candidate is None:
                return None
            git_payload = dict((candidate.get("summary") or {}).get("git") or {})
            if bool(git_payload.get("rollback_ready")):
                return candidate
            offset += 1

    def _record_rollback_outcome(self, report: PatchRollbackReport) -> PatchRollbackReport:
        report.tool_outcome = self.memory_store.record_tool_outcome(
            "patch-rollback",
            self._rollback_outcome_text(report),
            status="success" if report.status == "rolled_back" else "blocked" if report.status in {"blocked", "unavailable"} else "error",
            subject="self_improvement",
            tags=["patch-rollback", report.status],
            metadata={
                "patch_run_id": report.patch_run_id,
                "repo_root": report.repo_root,
                "branch_name": report.branch_name,
                "original_branch": report.original_branch,
                "commit": report.commit,
                "deleted_branch": report.deleted_branch,
                "switched_to": report.switched_to,
                "reason": report.reason,
            },
        )
        return report

    def _rollback_outcome_text(self, report: PatchRollbackReport) -> str:
        if report.status == "rolled_back":
            return (
                f"Rolled back disposable patch branch '{report.branch_name}'"
                + (f" and returned to '{report.switched_to}'." if report.switched_to else ".")
            )
        if report.status == "blocked":
            return (
                f"Patch rollback for '{report.branch_name or 'unknown branch'}' was blocked. "
                f"Reason: {report.reason or 'blocked'}"
            )
        if report.status == "unavailable":
            return (
                "No rollback-ready git-backed patch run was available. "
                f"Reason: {report.reason or 'unavailable'}"
            )
        return (
            f"Patch rollback for '{report.branch_name or 'unknown branch'}' failed. "
            f"Reason: {report.reason or 'error'}"
        )

    def _relative_to_workspace(self, path: Path, workspace_root: Path) -> str:
        return path.resolve().relative_to(workspace_root).as_posix()

    def _build_diff_preview(self, temp_workspace: Path, changed_files: list[str]) -> str:
        if not changed_files:
            return ""
        parts: list[str] = []
        for relative_path in changed_files[:5]:
            source_path = self.workspace_root / relative_path
            candidate_path = temp_workspace / relative_path
            before_text = self._safe_read_for_diff(source_path)
            after_text = self._safe_read_for_diff(candidate_path)
            diff_lines = list(
                difflib.unified_diff(
                    before_text.splitlines(),
                    after_text.splitlines(),
                    fromfile=f"a/{relative_path}",
                    tofile=f"b/{relative_path}",
                    lineterm="",
                )
            )
            if diff_lines:
                parts.append("\n".join(diff_lines))
        if not parts:
            return ""
        joined = "\n\n".join(parts)
        return self._trim_preview(joined)

    def _safe_read_for_diff(self, path: Path) -> str:
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def _trim_preview(self, text: str) -> str:
        stripped = str(text or "").strip()
        if len(stripped) <= self.output_char_limit:
            return stripped
        return stripped[: self.output_char_limit - 3].rstrip() + "..."

    def _default_validation_commands(self) -> list[str]:
        python_command = self._default_python_command()
        return [f"{python_command} -m unittest discover -s tests -v"]

    def _default_python_command(self) -> str:
        return "python" if os.name == "nt" else "python3"

    def _slugify(self, text: str) -> str:
        lowered = "".join(
            char.lower() if char.isalnum() else "-"
            for char in str(text or "patch-run").strip()
        )
        compacted = "-".join(part for part in lowered.split("-") if part)
        return compacted or "patch-run"

    def _copytree_ignore(self, directory: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for name in names:
            if name in PATCH_IGNORE_NAMES or any(
                name.lower().startswith(prefix) for prefix in PATCH_IGNORE_PREFIXES
            ):
                ignored.add(name)
                continue
            candidate = Path(directory) / name
            try:
                if candidate.is_dir():
                    next(candidate.iterdir(), None)
            except OSError:
                ignored.add(name)
        return ignored

    def _tool_outcome_status(self, report_status: str) -> str:
        if report_status in {"accepted", "applied"}:
            return "success"
        if report_status == "rejected":
            return "blocked"
        return "error"

    def _tool_outcome_text(self, report: PatchRunReport) -> str:
        if report.status == "applied":
            if report.git_apply is not None and report.git_apply.status == "applied":
                return (
                    f"Patch run '{report.run_name}' passed validation, matched the baseline, "
                    f"and committed {len(report.changed_files)} file updates on "
                    f"'{report.git_apply.branch_name}'"
                )
            return (
                f"Patch run '{report.run_name}' passed validation, matched the baseline, "
                f"and applied {len(report.changed_files)} file updates"
            )
        if report.status == "accepted":
            return (
                f"Patch run '{report.run_name}' passed validation and matched the baseline "
                "without applying the candidate back to the main workspace"
            )
        if report.status == "rejected":
            return (
                f"Patch run '{report.run_name}' was rejected. "
                f"Reason: {report.rejection_reason or 'validation_failed'}"
            )
        return (
            f"Patch run '{report.run_name}' hit an apply error. "
            f"Reason: {report.rejection_reason or 'apply_failed'}"
        )
