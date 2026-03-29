from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .file_adapter import FileOperationResult, WorkspaceFileAdapter
from .memory import MemoryStore
from .models import MemoryRecord
from .patch_runner import PatchRunReport
from .planner import MemoryPlanner, PlannerAction, PlannerSnapshot
from .shell_adapter import GuardedShellAdapter, ShellExecutionResult


@dataclass(slots=True)
class ExecutorResult:
    requested_action: PlannerAction | None
    executed_kind: str
    status: str
    summary: str
    reasons: list[str] = field(default_factory=list)
    tool_outcome: MemoryRecord | None = None
    task_update: MemoryRecord | None = None
    related_task: MemoryRecord | None = None
    maintenance_report: dict[str, Any] | None = None
    shell_result: ShellExecutionResult | None = None
    file_result: FileOperationResult | None = None
    patch_run: PatchRunReport | None = None
    prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        lines = [f"Execution result: [{self.status}] {self.summary}"]
        if self.requested_action is not None:
            lines.append(f"Requested action: [{self.requested_action.kind}] {self.requested_action.title}")
        lines.append(f"Executed kind: {self.executed_kind}")
        if self.reasons:
            lines.append("Reasons: " + ", ".join(self.reasons))
        if self.shell_result is not None:
            lines.append(
                f"Shell: [{self.shell_result.status}] {self.shell_result.command_text}"
            )
        if self.file_result is not None:
            lines.append(
                f"File: [{self.file_result.status}] {self.file_result.operation} {self.file_result.path}"
            )
        if self.patch_run is not None:
            lines.append(
                f"Patch: [{self.patch_run.status}] {len(self.patch_run.changed_files)} changed file(s)"
            )
        if self.prompt:
            lines.append("Prompt: " + self.prompt)
        return "\n".join(lines)


@dataclass(slots=True)
class ExecutionCycle:
    query: str
    before_plan: PlannerSnapshot
    result: ExecutorResult
    after_plan: PlannerSnapshot

    def render(self) -> str:
        lines = [self.before_plan.render(), "", self.result.render(), "", "Updated plan:", self.after_plan.render()]
        return "\n".join(lines)


class MemoryExecutor:
    def __init__(
        self,
        memory_store: MemoryStore,
        *,
        shell_adapter: GuardedShellAdapter | None = None,
        file_adapter: WorkspaceFileAdapter | None = None,
    ):
        self.memory_store = memory_store
        self.planner = MemoryPlanner(memory_store)
        self.shell_adapter = shell_adapter or GuardedShellAdapter()
        self.file_adapter = file_adapter or WorkspaceFileAdapter()

    def execute_next(
        self,
        query: str = "next best action",
        *,
        action_limit: int = 5,
    ) -> ExecutionCycle:
        plan_query = query.strip() or "next best action"
        before_plan = self.planner.build_plan(plan_query, action_limit=action_limit)
        result = self.execute_action(before_plan.recommendation)
        after_plan = self.planner.build_plan(plan_query, action_limit=action_limit)
        return ExecutionCycle(
            query=plan_query,
            before_plan=before_plan,
            result=result,
            after_plan=after_plan,
        )

    def execute_action(self, action: PlannerAction | None) -> ExecutorResult:
        if action is None:
            return ExecutorResult(
                requested_action=None,
                executed_kind="noop",
                status="noop",
                summary="No action was available to execute.",
            )
        if action.kind == "work_task":
            return self._execute_work_task(action)
        if action.kind == "prepare_task":
            return self._execute_prepare_task(action)
        if action.kind == "resolve_blocker":
            return self._execute_resolve_blocker(action)
        if action.kind == "run_maintenance":
            return self._execute_run_maintenance(action)
        if action.kind == "ask_user":
            return self._execute_ask_user(action)
        if action.kind == "noop":
            return ExecutorResult(
                requested_action=action,
                executed_kind="noop",
                status="noop",
                summary=action.summary,
                reasons=list(action.reasons),
                metadata=dict(action.metadata),
            )
        return self._execute_ask_user(
            PlannerAction(
                kind="ask_user",
                title="Clarify unsupported action",
                summary=(
                    f"The planner proposed unsupported action kind '{action.kind}'. "
                    "Please clarify how this should be executed."
                ),
                score=0.0,
                reasons=list(action.reasons),
                metadata={"unsupported_kind": action.kind, **dict(action.metadata)},
            )
        )

    def _execute_work_task(self, action: PlannerAction) -> ExecutorResult:
        task = self._resolve_task(action)
        if task is None:
            return self._execute_ask_user(
                PlannerAction(
                    kind="ask_user",
                    title=f"Missing task for {action.title}",
                    summary=(
                        f"I could not find the active task '{action.title}' to start work on."
                    ),
                    score=action.score,
                    reasons=["task_missing", *action.reasons],
                    metadata=dict(action.metadata),
                )
            )

        command = str(task.metadata.get("command") or "").strip()
        file_operation = str(task.metadata.get("file_operation") or "").strip()
        if command and file_operation:
            return self._execute_ambiguous_execution_metadata(action, task)
        if file_operation:
            return self._execute_task_file_operation(action, task, file_operation)
        if command:
            return self._execute_task_command(action, task, command)

        status = str(task.metadata.get("status", "open"))
        if status == "in_progress":
            updated_task = task
            summary = f"'{action.title}' is already in progress."
            outcome_text = f"Confirmed task '{action.title}' is already in progress"
        else:
            updated_task = self.memory_store.record_task(
                action.title,
                status="in_progress",
                area=task.subject,
            )
            summary = f"Started task '{action.title}' and moved it to in-progress."
            outcome_text = f"Started task '{action.title}'"

        tool_outcome = self.memory_store.record_tool_outcome(
            "executor",
            outcome_text,
            status="success",
            subject=task.subject,
            tags=["executor", "work_task"],
        )
        return ExecutorResult(
            requested_action=action,
            executed_kind="work_task",
            status="success",
            summary=summary,
            reasons=list(action.reasons),
            tool_outcome=tool_outcome,
            task_update=updated_task,
            metadata={"task_status": updated_task.metadata.get("status")},
        )

    def _execute_prepare_task(self, action: PlannerAction) -> ExecutorResult:
        target_task = self._resolve_task(action)
        if target_task is None:
            missing_title = str(action.metadata.get("target_task_title") or action.title)
            return self._execute_ask_user(
                PlannerAction(
                    kind="ask_user",
                    title=f"Missing task for {missing_title}",
                    summary=(
                        f"I could not find the risky task '{missing_title}' to prepare safely."
                    ),
                    score=action.score,
                    reasons=["task_missing", *action.reasons],
                    metadata=dict(action.metadata),
                )
            )

        target_title = str(target_task.metadata.get("title") or action.metadata.get("target_task_title") or action.title)
        prep_title = str(action.metadata.get("prep_task_title") or f"Prepare safer execution for {target_title}")
        prep_details = str(action.metadata.get("prep_task_details") or "").strip()
        existing_prep = self.memory_store.find_active_task(prep_title, area=target_task.subject, decorate=True)

        if existing_prep is not None and str(existing_prep.metadata.get("status", "open")) != "done":
            prep_task = existing_prep
            created_prep = False
        else:
            prep_task = self.memory_store.record_task(
                prep_title,
                status="open",
                area=target_task.subject,
                owner="agent",
                details=prep_details or None,
                due_date=target_task.metadata.get("due_date"),
                tags=["pilot-prep", "safer-execution"],
                importance=min(max(target_task.importance + 0.02, 0.78), 0.96),
                confidence=0.9,
            )
            created_prep = True

        blocked_by = list(
            dict.fromkeys(
                [
                    str(item).strip()
                    for item in target_task.metadata.get("blocked_by", [])
                    if str(item).strip()
                ]
                + [prep_title]
            )
        )
        depends_on = list(
            dict.fromkeys(
                [
                    str(item).strip()
                    for item in target_task.metadata.get("depends_on", [])
                    if str(item).strip()
                ]
                + [prep_title]
            )
        )

        if (
            str(target_task.metadata.get("status", "open")) == "blocked"
            and blocked_by == list(target_task.metadata.get("blocked_by", []))
            and depends_on == list(target_task.metadata.get("depends_on", []))
        ):
            updated_target = target_task
        else:
            updated_target = self.memory_store.record_task(
                target_title,
                status="blocked",
                area=target_task.subject,
                depends_on=depends_on,
                blocked_by=blocked_by,
                importance=target_task.importance,
                confidence=target_task.confidence,
            )

        if created_prep:
            outcome_text = (
                f"Created safer prep task '{prep_title}' for '{target_title}' based on "
                "recurring pilot approval friction"
            )
            summary = (
                f"Created '{prep_title}' and routed '{target_title}' through it so the "
                "pilot can take a safer step first."
            )
        else:
            outcome_text = (
                f"Kept safer prep task '{prep_title}' linked to '{target_title}' because "
                "recurring pilot approval friction still applies"
            )
            summary = (
                f"'{target_title}' already has safer prep task '{prep_title}', so the risky "
                "work stays routed through that step first."
            )

        tool_outcome = self.memory_store.record_tool_outcome(
            "executor",
            outcome_text,
            status="success",
            subject=target_task.subject,
            tags=["executor", "prepare_task", "pilot-history"],
            metadata={
                "target_task_title": target_title,
                "prep_task_title": prep_title,
            },
        )
        return ExecutorResult(
            requested_action=action,
            executed_kind="prepare_task",
            status="success",
            summary=summary,
            reasons=list(action.reasons),
            tool_outcome=tool_outcome,
            task_update=updated_target,
            related_task=prep_task,
            metadata={
                "target_task_title": target_title,
                "prep_task_title": prep_title,
                "created_prep": created_prep,
            },
        )

    def _execute_ambiguous_execution_metadata(
        self,
        action: PlannerAction,
        task: MemoryRecord,
    ) -> ExecutorResult:
        tool_outcome = self.memory_store.record_tool_outcome(
            "executor",
            (
                f"Task '{action.title}' has both shell and file execution metadata, "
                "so execution was blocked pending clarification"
            ),
            status="blocked",
            subject=task.subject,
            tags=["executor", "ambiguous_execution"],
        )
        return ExecutorResult(
            requested_action=action,
            executed_kind="ask_user",
            status="blocked",
            summary=(
                f"'{action.title}' defines both a shell command and a file operation, "
                "so I need clarification before executing it."
            ),
            reasons=["ambiguous_execution_metadata", *action.reasons],
            tool_outcome=tool_outcome,
            related_task=task,
            prompt=(
                f"'{action.title}' has both a shell command and a file operation configured. "
                "Please keep one execution mode for that task."
            ),
        )

    def _execute_task_command(
        self,
        action: PlannerAction,
        task: MemoryRecord,
        command: str,
    ) -> ExecutorResult:
        shell_result = self.shell_adapter.execute(
            command,
            cwd=str(task.metadata.get("cwd") or "").strip() or None,
        )
        if shell_result.status == "blocked":
            tool_outcome = self.memory_store.record_tool_outcome(
                "shell",
                (
                    f"Blocked command for '{action.title}': {command}. "
                    f"Reason: {shell_result.reason}"
                ),
                status="blocked",
                subject=task.subject,
                tags=["executor", "shell", "blocked"],
            )
            return ExecutorResult(
                requested_action=action,
                executed_kind="run_shell",
                status="blocked",
                summary=(
                    f"Command for '{action.title}' was blocked by shell policy."
                ),
                reasons=["shell_blocked", shell_result.reason, *action.reasons],
                tool_outcome=tool_outcome,
                related_task=task,
                shell_result=shell_result,
                prompt=(
                    f"The task command for '{action.title}' is blocked by policy "
                    f"({shell_result.reason})."
                ),
                metadata={"command": command},
            )

        if shell_result.status == "error":
            updated_task = (
                task
                if str(task.metadata.get("status", "open")) == "in_progress"
                else self.memory_store.record_task(
                    action.title,
                    status="in_progress",
                    area=task.subject,
                )
            )
            tool_outcome = self.memory_store.record_tool_outcome(
                "shell",
                self._shell_outcome_text(action.title, shell_result),
                status="error",
                subject=task.subject,
                tags=["executor", "shell", "error"],
            )
            return ExecutorResult(
                requested_action=action,
                executed_kind="run_shell",
                status="error",
                summary=(
                    f"Command for '{action.title}' failed with "
                    f"{shell_result.reason or 'an error'}."
                ),
                reasons=["shell_error", shell_result.reason, *action.reasons],
                tool_outcome=tool_outcome,
                task_update=updated_task,
                related_task=task,
                shell_result=shell_result,
                metadata={
                    "command": command,
                    "exit_code": shell_result.exit_code,
                    "task_status": updated_task.metadata.get("status"),
                },
            )

        updated_task: MemoryRecord | None
        next_occurrence: MemoryRecord | None = None
        if bool(task.metadata.get("complete_on_success", False)):
            completion = self.memory_store.complete_task(action.title, area=task.subject)
            updated_task = completion["completed"]
            next_occurrence = completion["next_occurrence"]
            summary = f"Ran command for '{action.title}' and completed the task."
        else:
            current_status = str(task.metadata.get("status", "open"))
            updated_task = (
                task
                if current_status == "in_progress"
                else self.memory_store.record_task(
                    action.title,
                    status="in_progress",
                    area=task.subject,
                )
            )
            summary = f"Ran command for '{action.title}' and kept it in progress."
        tool_outcome = self.memory_store.record_tool_outcome(
            "shell",
            self._shell_outcome_text(action.title, shell_result),
            status="success",
            subject=task.subject,
            tags=["executor", "shell", "success"],
        )
        return ExecutorResult(
            requested_action=action,
            executed_kind="run_shell",
            status="success",
            summary=summary,
            reasons=["shell_success", *action.reasons],
            tool_outcome=tool_outcome,
            task_update=updated_task,
            related_task=next_occurrence,
            shell_result=shell_result,
            metadata={
                "command": command,
                "exit_code": shell_result.exit_code,
                "task_status": updated_task.metadata.get("status") if updated_task else None,
            },
        )

    def _execute_task_file_operation(
        self,
        action: PlannerAction,
        task: MemoryRecord,
        file_operation: str,
    ) -> ExecutorResult:
        file_result = self.file_adapter.execute(
            file_operation,
            str(task.metadata.get("file_path") or ""),
            text=task.metadata.get("file_text"),
            find_text=task.metadata.get("find_text"),
            symbol_name=task.metadata.get("symbol_name"),
            replace_all=bool(task.metadata.get("replace_all", False)),
            cwd=str(task.metadata.get("cwd") or "").strip() or None,
        )
        if file_result.status == "blocked":
            tool_outcome = self.memory_store.record_tool_outcome(
                "file",
                (
                    f"Blocked file operation for '{action.title}': {file_operation} "
                    f"{str(task.metadata.get('file_path') or '').strip()}. "
                    f"Reason: {file_result.reason}"
                ),
                status="blocked",
                subject=task.subject,
                tags=["executor", "file", "blocked"],
            )
            return ExecutorResult(
                requested_action=action,
                executed_kind="run_file_operation",
                status="blocked",
                summary=(f"File operation for '{action.title}' was blocked by file policy."),
                reasons=["file_blocked", file_result.reason, *action.reasons],
                tool_outcome=tool_outcome,
                related_task=task,
                file_result=file_result,
                prompt=(
                    f"The file task for '{action.title}' is blocked by policy "
                    f"({file_result.reason})."
                ),
                metadata={
                    "file_operation": file_operation,
                    "file_path": task.metadata.get("file_path"),
                    "symbol_name": task.metadata.get("symbol_name"),
                },
            )

        if file_result.status == "error":
            updated_task = (
                task
                if str(task.metadata.get("status", "open")) == "in_progress"
                else self.memory_store.record_task(
                    action.title,
                    status="in_progress",
                    area=task.subject,
                )
            )
            tool_outcome = self.memory_store.record_tool_outcome(
                "file",
                self._file_outcome_text(action.title, file_result),
                status="error",
                subject=task.subject,
                tags=["executor", "file", "error"],
            )
            return ExecutorResult(
                requested_action=action,
                executed_kind="run_file_operation",
                status="error",
                summary=(
                    f"File operation for '{action.title}' failed with "
                    f"{file_result.reason or 'an error'}."
                ),
                reasons=["file_error", file_result.reason, *action.reasons],
                tool_outcome=tool_outcome,
                task_update=updated_task,
                related_task=task,
                file_result=file_result,
                metadata={
                    "file_operation": file_operation,
                    "file_path": task.metadata.get("file_path"),
                    "symbol_name": task.metadata.get("symbol_name"),
                    "task_status": updated_task.metadata.get("status"),
                },
            )

        updated_task: MemoryRecord | None
        next_occurrence: MemoryRecord | None = None
        if bool(task.metadata.get("complete_on_success", False)):
            completion = self.memory_store.complete_task(action.title, area=task.subject)
            updated_task = completion["completed"]
            next_occurrence = completion["next_occurrence"]
            summary = f"Ran file operation for '{action.title}' and completed the task."
        else:
            current_status = str(task.metadata.get("status", "open"))
            updated_task = (
                task
                if current_status == "in_progress"
                else self.memory_store.record_task(
                    action.title,
                    status="in_progress",
                    area=task.subject,
                )
            )
            summary = f"Ran file operation for '{action.title}' and kept it in progress."
        tool_outcome = self.memory_store.record_tool_outcome(
            "file",
            self._file_outcome_text(action.title, file_result),
            status="success",
            subject=task.subject,
            tags=["executor", "file", "success"],
        )
        return ExecutorResult(
            requested_action=action,
            executed_kind="run_file_operation",
            status="success",
            summary=summary,
            reasons=["file_success", *action.reasons],
            tool_outcome=tool_outcome,
            task_update=updated_task,
            related_task=next_occurrence,
            file_result=file_result,
            metadata={
                "file_operation": file_operation,
                "file_path": task.metadata.get("file_path"),
                "symbol_name": task.metadata.get("symbol_name"),
                "task_status": updated_task.metadata.get("status") if updated_task else None,
            },
        )

    def _execute_resolve_blocker(self, action: PlannerAction) -> ExecutorResult:
        blocked_task = self._resolve_task(action)
        blockers = [
            str(item).strip()
            for item in action.metadata.get("blocked_by", [])
            if str(item).strip()
        ]
        if blocked_task is None:
            return self._execute_ask_user(
                PlannerAction(
                    kind="ask_user",
                    title=f"Missing blocked task for {action.title}",
                    summary=(
                        f"I could not find the blocked task '{action.title}' to resolve next."
                    ),
                    score=action.score,
                    reasons=["task_missing", *action.reasons],
                    metadata={"blocked_by": blockers, **dict(action.metadata)},
                )
            )

        if not blockers:
            follow_up = PlannerAction(
                kind="work_task",
                title=action.title,
                summary=(
                    f"No active blocker was left on '{action.title}', so start the task directly."
                ),
                score=action.score,
                reasons=["blocker_cleared", *action.reasons],
                task_id=blocked_task.id,
                evidence_memory_ids=list(action.evidence_memory_ids),
                metadata={"area": blocked_task.subject},
            )
            return self._execute_work_task(follow_up)

        for blocker_title in blockers:
            blocker_task = self.memory_store.find_active_task(blocker_title, decorate=True)
            if blocker_task is None:
                continue
            if bool(blocker_task.metadata.get("blocked_now")):
                continue
            follow_up = PlannerAction(
                kind="work_task",
                title=blocker_title,
                summary=(
                    f"Work on blocker task '{blocker_title}' to unblock '{action.title}'."
                ),
                score=action.score,
                reasons=["rerouted_to_blocker_task", *action.reasons],
                task_id=blocker_task.id,
                evidence_memory_ids=list(action.evidence_memory_ids),
                metadata={"area": blocker_task.subject},
            )
            rerouted = self._execute_work_task(follow_up)
            rerouted.requested_action = action
            rerouted.executed_kind = "resolve_blocker"
            rerouted.reasons = ["rerouted_to_blocker_task", *action.reasons, *rerouted.reasons]
            rerouted.related_task = blocked_task
            rerouted.summary = (
                f"Shifted execution to blocker task '{blocker_title}' to unblock '{action.title}'. "
                + rerouted.summary
            )
            rerouted.metadata = {
                **rerouted.metadata,
                "blocked_task": action.title,
                "blocker_task": blocker_title,
            }
            return rerouted

        prompt = (
            f"'{action.title}' is blocked by {', '.join(blockers[:2])}. "
            "I need user input or an external update before I can unblock it."
        )
        tool_outcome = self.memory_store.record_tool_outcome(
            "executor",
            f"Need user input to unblock '{action.title}': {', '.join(blockers[:2])}",
            status="blocked",
            subject=blocked_task.subject,
            tags=["executor", "ask_user", "resolve_blocker"],
        )
        return ExecutorResult(
            requested_action=action,
            executed_kind="ask_user",
            status="blocked",
            summary=(
                f"'{action.title}' is still blocked by an external or blocked dependency."
            ),
            reasons=["needs_user_input", *action.reasons],
            tool_outcome=tool_outcome,
            related_task=blocked_task,
            prompt=prompt,
            metadata={"blocked_by": blockers},
        )

    def _execute_run_maintenance(self, action: PlannerAction) -> ExecutorResult:
        report = self.memory_store.run_maintenance(force=False)
        executed = sorted(report.get("executed", {}).keys())
        if executed:
            summary = f"Ran maintenance tasks: {', '.join(executed)}."
            status = "success"
            outcome_text = f"Ran maintenance tasks: {', '.join(executed)}"
        else:
            summary = "No memory maintenance was due right now."
            status = "noop"
            outcome_text = "No maintenance tasks were due"
        tool_outcome = self.memory_store.record_tool_outcome(
            "executor",
            outcome_text,
            status=status,
            subject="memory",
            tags=["executor", "run_maintenance"],
        )
        return ExecutorResult(
            requested_action=action,
            executed_kind="run_maintenance",
            status=status,
            summary=summary,
            reasons=list(action.reasons),
            tool_outcome=tool_outcome,
            maintenance_report=report,
            metadata={"executed": executed},
        )

    def _execute_ask_user(self, action: PlannerAction) -> ExecutorResult:
        tool_outcome = self.memory_store.record_tool_outcome(
            "executor",
            f"Need user input: {action.summary}",
            status="blocked",
            subject=str(action.metadata.get("area") or "execution"),
            tags=["executor", "ask_user"],
        )
        return ExecutorResult(
            requested_action=action,
            executed_kind="ask_user",
            status="blocked",
            summary=action.summary,
            reasons=list(action.reasons),
            tool_outcome=tool_outcome,
            prompt=action.summary,
            metadata=dict(action.metadata),
        )

    def _resolve_task(self, action: PlannerAction) -> MemoryRecord | None:
        task_id = action.task_id
        if task_id is not None:
            try:
                candidate = self.memory_store.get_memory(task_id)
            except KeyError:
                candidate = None
            if candidate is not None and candidate.archived_at is None:
                return self.memory_store.find_active_task(
                    action.title,
                    area=str(action.metadata.get("area") or candidate.subject),
                    decorate=True,
                ) or candidate
        return self.memory_store.find_active_task(
            action.title,
            area=str(action.metadata.get("area") or "") or None,
            decorate=True,
        )

    def _shell_outcome_text(self, task_title: str, shell_result: ShellExecutionResult) -> str:
        snippet = shell_result.stdout or shell_result.stderr
        if snippet:
            first_line = snippet.splitlines()[0].strip()
            return (
                f"Ran shell command for '{task_title}' "
                f"(exit={shell_result.exit_code}): {first_line}"
            )
        return (
            f"Ran shell command for '{task_title}' "
            f"(exit={shell_result.exit_code})"
        )

    def _file_outcome_text(self, task_title: str, file_result: FileOperationResult) -> str:
        if file_result.preview:
            first_line = file_result.preview.splitlines()[0].strip()
            return (
                f"Ran file operation for '{task_title}' "
                f"({file_result.operation} {file_result.path}): {first_line}"
            )
        return (
            f"Ran file operation for '{task_title}' "
            f"({file_result.operation} {file_result.path})"
        )
