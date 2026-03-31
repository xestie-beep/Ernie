from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from .file_adapter import FileOperationResult, WorkspaceFileAdapter
from .memory import MemoryStore
from .models import MemoryRecord
from .patch_runner import PatchRunReport
from .planner import MemoryPlanner, PlannerAction, PlannerSnapshot
from .service_manager import CockpitServiceManager
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
        service_manager: CockpitServiceManager | None = None,
    ):
        self.memory_store = memory_store
        self.service_manager = service_manager or CockpitServiceManager()
        self.planner = MemoryPlanner(memory_store, service_manager=self.service_manager)
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
        if action.kind == "batch_ready_tasks":
            return self._execute_batch_ready_tasks(action)
        if action.kind == "delegate_task":
            return self._execute_delegate_task(action)
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
        service_action = str(task.metadata.get("service_action") or "").strip()
        service_inspection = str(task.metadata.get("service_inspection") or "").strip()
        file_operation = str(task.metadata.get("file_operation") or "").strip()
        configured_modes = [
            bool(command),
            bool(service_action),
            bool(service_inspection),
            bool(file_operation),
        ]
        if sum(1 for item in configured_modes if item) > 1:
            return self._execute_ambiguous_execution_metadata(action, task)
        if file_operation:
            return self._execute_task_file_operation(action, task, file_operation)
        if service_action:
            return self._execute_task_service_action(action, task, service_action)
        if service_inspection:
            return self._execute_task_service_inspection(action, task, service_inspection)
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

    def _execute_batch_ready_tasks(self, action: PlannerAction) -> ExecutorResult:
        batch_tasks = self._resolve_batch_tasks(action)
        if not batch_tasks:
            return self._execute_ask_user(
                PlannerAction(
                    kind="ask_user",
                    title="Missing tasks for ready batch",
                    summary=(
                        "I could not resolve the ready-task batch that was selected for execution."
                    ),
                    score=action.score,
                    reasons=["task_batch_missing", *action.reasons],
                    metadata=dict(action.metadata),
                )
            )

        batch_results: list[dict[str, Any]] = []
        last_task_update: MemoryRecord | None = None
        last_related_task: MemoryRecord | None = None

        for task in batch_tasks:
            title = str(task.metadata.get("title") or task.content)
            follow_up = PlannerAction(
                kind="work_task",
                title=title,
                summary=f"Work on '{title}' now as part of the current safe ready batch.",
                score=action.score,
                reasons=["batched_execution", *action.reasons],
                task_id=task.id,
                evidence_memory_ids=list(action.evidence_memory_ids),
                metadata={"area": task.subject},
            )
            result = self._execute_work_task(follow_up)
            batch_results.append(
                {
                    "title": title,
                    "executed_kind": result.executed_kind,
                    "status": result.status,
                    "summary": result.summary,
                }
            )
            if result.task_update is not None:
                last_task_update = result.task_update
            if result.related_task is not None:
                last_related_task = result.related_task
            if result.status not in {"success", "noop"}:
                tool_outcome = self.memory_store.record_tool_outcome(
                    "executor",
                    (
                        f"Ready-task batch stalled on '{title}' with status "
                        f"{result.status}: {result.summary}"
                    ),
                    status=result.status,
                    subject=task.subject,
                    tags=["executor", "batch_ready_tasks", result.status],
                    metadata={
                        "batch_results": batch_results,
                        "stalled_task_title": title,
                    },
                )
                return ExecutorResult(
                    requested_action=action,
                    executed_kind="batch_ready_tasks",
                    status=result.status,
                    summary=(
                        f"Ready-task batch stalled on '{title}' after {len(batch_results) - 1} "
                        "successful step(s)."
                    ),
                    reasons=list(action.reasons),
                    tool_outcome=tool_outcome,
                    task_update=last_task_update,
                    related_task=last_related_task or task,
                    prompt=result.prompt,
                    metadata={"batch_results": batch_results},
                )

        tool_outcome = self.memory_store.record_tool_outcome(
            "executor",
            (
                f"Ran ready-task batch: {', '.join(item['title'] for item in batch_results)}"
            ),
            status="success",
            subject="execution",
            tags=["executor", "batch_ready_tasks", "success"],
            metadata={"batch_results": batch_results},
        )
        return ExecutorResult(
            requested_action=action,
            executed_kind="batch_ready_tasks",
            status="success",
            summary=(
                f"Ran a safe batch of {len(batch_results)} ready task(s): "
                f"{', '.join(item['title'] for item in batch_results)}."
            ),
            reasons=list(action.reasons),
            tool_outcome=tool_outcome,
            task_update=last_task_update,
            related_task=last_related_task,
            metadata={"batch_results": batch_results},
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
                service_inspection=target_task.metadata.get("service_action"),
                service_label=target_task.metadata.get("service_label"),
                service_requires_confirmation=bool(
                    target_task.metadata.get("service_requires_confirmation", False)
                ),
                service_confirmation_message=target_task.metadata.get(
                    "service_confirmation_message"
                ),
                service_success_message=target_task.metadata.get("service_success_message"),
                complete_on_success=True,
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

    def _execute_delegate_task(self, action: PlannerAction) -> ExecutorResult:
        target_task = self._resolve_task(action)
        if target_task is None:
            missing_title = str(action.metadata.get("target_task_title") or action.title)
            return self._execute_ask_user(
                PlannerAction(
                    kind="ask_user",
                    title=f"Missing task for {missing_title}",
                    summary=f"I could not find the task '{missing_title}' to delegate.",
                    score=action.score,
                    reasons=["task_missing", *action.reasons],
                    metadata=dict(action.metadata),
                )
            )

        target_title = str(
            target_task.metadata.get("title")
            or action.metadata.get("target_task_title")
            or action.title
        )
        delegate_title = str(
            action.metadata.get("delegate_task_title")
            or f"Delegate work for {target_title}"
        )
        delegate_details = str(action.metadata.get("delegate_task_details") or "").strip()
        existing_delegate = self.memory_store.find_active_task(
            delegate_title,
            area=target_task.subject,
            decorate=True,
        )

        if existing_delegate is not None and str(existing_delegate.metadata.get("status", "open")) != "done":
            delegate_task = existing_delegate
            created_delegate = False
        else:
            delegate_task = self.memory_store.record_task(
                delegate_title,
                status="open",
                area=target_task.subject,
                owner="delegate",
                details=delegate_details or None,
                due_date=target_task.metadata.get("due_date"),
                tags=["delegated", "handoff"],
                importance=min(max(target_task.importance, 0.78), 0.95),
                confidence=0.9,
            )
            created_delegate = True

        blocked_by = list(
            dict.fromkeys(
                [
                    str(item).strip()
                    for item in target_task.metadata.get("blocked_by", [])
                    if str(item).strip()
                ]
                + [delegate_title]
            )
        )
        depends_on = list(
            dict.fromkeys(
                [
                    str(item).strip()
                    for item in target_task.metadata.get("depends_on", [])
                    if str(item).strip()
                ]
                + [delegate_title]
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
                owner=str(target_task.metadata.get("owner") or "agent"),
                details=target_task.metadata.get("details"),
                depends_on=depends_on,
                blocked_by=blocked_by,
                due_date=target_task.metadata.get("due_date"),
                recurrence_days=target_task.metadata.get("recurrence_days"),
                snoozed_until=target_task.metadata.get("snoozed_until"),
                command=target_task.metadata.get("command"),
                cwd=target_task.metadata.get("cwd"),
                service_action=target_task.metadata.get("service_action"),
                file_operation=target_task.metadata.get("file_operation"),
                file_path=target_task.metadata.get("file_path"),
                file_text=target_task.metadata.get("file_text"),
                find_text=target_task.metadata.get("find_text"),
                symbol_name=target_task.metadata.get("symbol_name"),
                replace_all=bool(target_task.metadata.get("replace_all", False)),
                complete_on_success=bool(target_task.metadata.get("complete_on_success", False)),
                retry_limit=target_task.metadata.get("retry_limit"),
                retry_count=target_task.metadata.get("retry_count"),
                retry_cooldown_minutes=target_task.metadata.get("retry_cooldown_minutes"),
                last_retry_at=target_task.metadata.get("last_retry_at"),
                last_failure_at=target_task.metadata.get("last_failure_at"),
                importance=target_task.importance,
                confidence=target_task.confidence,
            )

        if created_delegate:
            outcome_text = (
                f"Created delegated task '{delegate_title}' for '{target_title}' and routed the parent through it"
            )
            summary = (
                f"Created delegated task '{delegate_title}' and routed '{target_title}' through that handoff."
            )
        else:
            outcome_text = f"Kept delegated task '{delegate_title}' linked to '{target_title}'"
            summary = (
                f"'{target_title}' already has delegated child task '{delegate_title}', so the parent stays routed through it."
            )

        tool_outcome = self.memory_store.record_tool_outcome(
            "executor",
            outcome_text,
            status="success",
            subject=target_task.subject,
            tags=["executor", "delegate_task"],
            metadata={
                "target_task_title": target_title,
                "delegate_task_title": delegate_title,
            },
        )
        return ExecutorResult(
            requested_action=action,
            executed_kind="delegate_task",
            status="success",
            summary=summary,
            reasons=list(action.reasons),
            tool_outcome=tool_outcome,
            task_update=updated_target,
            related_task=delegate_task,
            metadata={
                "target_task_title": target_title,
                "delegate_task_title": delegate_title,
                "created_delegate": created_delegate,
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
            updated_task, retry_scheduled, retry_metadata = self._record_task_error_state(task)
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
                summary=self._retry_error_summary(
                    action.title,
                    failure_kind="Command",
                    failure_reason=shell_result.reason,
                    retry_scheduled=retry_scheduled,
                    retry_metadata=retry_metadata,
                ),
                reasons=[
                    "shell_error",
                    shell_result.reason,
                    *(["retry_scheduled"] if retry_scheduled else []),
                    *action.reasons,
                ],
                tool_outcome=tool_outcome,
                task_update=updated_task,
                related_task=task,
                shell_result=shell_result,
                metadata={
                    "command": command,
                    "exit_code": shell_result.exit_code,
                    "task_status": updated_task.metadata.get("status"),
                    **retry_metadata,
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
            updated_task, retry_scheduled, retry_metadata = self._record_task_error_state(task)
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
                summary=self._retry_error_summary(
                    action.title,
                    failure_kind="File operation",
                    failure_reason=file_result.reason,
                    retry_scheduled=retry_scheduled,
                    retry_metadata=retry_metadata,
                ),
                reasons=[
                    "file_error",
                    file_result.reason,
                    *(["retry_scheduled"] if retry_scheduled else []),
                    *action.reasons,
                ],
                tool_outcome=tool_outcome,
                task_update=updated_task,
                related_task=task,
                file_result=file_result,
                metadata={
                    "file_operation": file_operation,
                    "file_path": task.metadata.get("file_path"),
                    "symbol_name": task.metadata.get("symbol_name"),
                    "task_status": updated_task.metadata.get("status"),
                    **retry_metadata,
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

    def _execute_task_service_action(
        self,
        action: PlannerAction,
        task: MemoryRecord,
        service_action: str,
    ) -> ExecutorResult:
        try:
            service_result = self.service_manager.perform_action(service_action)
        except Exception as exc:
            updated_task, retry_scheduled, retry_metadata = self._record_task_error_state(task)
            reason = str(exc).strip() or "service_action_failed"
            tool_outcome = self.memory_store.record_tool_outcome(
                "service_manager",
                (
                    f"Service action for '{action.title}' failed: {service_action}. "
                    f"Reason: {reason}"
                ),
                status="error",
                subject=task.subject,
                tags=["executor", "service_action", "error"],
                metadata={"service_action": service_action},
            )
            return ExecutorResult(
                requested_action=action,
                executed_kind="run_service_action",
                status="error",
                summary=self._retry_error_summary(
                    action.title,
                    failure_kind="Service action",
                    failure_reason=reason,
                    retry_scheduled=retry_scheduled,
                    retry_metadata=retry_metadata,
                ),
                reasons=[
                    "service_action_error",
                    reason,
                    *(["retry_scheduled"] if retry_scheduled else []),
                    *action.reasons,
                ],
                tool_outcome=tool_outcome,
                task_update=updated_task,
                related_task=task,
                metadata={
                    "service_action": service_action,
                    "task_status": updated_task.metadata.get("status"),
                    **retry_metadata,
                },
            )

        updated_task: MemoryRecord | None
        next_occurrence: MemoryRecord | None = None
        if bool(task.metadata.get("complete_on_success", False)):
            completion = self.memory_store.complete_task(action.title, area=task.subject)
            updated_task = completion["completed"]
            next_occurrence = completion["next_occurrence"]
            summary = (
                f"Ran service action '{service_action}' for '{action.title}' and completed the task."
            )
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
            summary = (
                f"Ran service action '{service_action}' for '{action.title}' and kept it in progress."
            )
        message = str(service_result.get("message") or "").strip()
        verification_task = self._ensure_service_verification_task(
            action=action,
            task=task,
            service_action=service_action,
            service_result=service_result,
        )
        tool_outcome = self.memory_store.record_tool_outcome(
            "service_manager",
            (
                f"Ran service action '{service_action}' for '{action.title}'."
                + (f" {message}" if message else "")
            ),
            status="success",
            subject=task.subject,
            tags=["executor", "service_action", "success"],
            metadata={"service_action": service_action},
        )
        return ExecutorResult(
            requested_action=action,
            executed_kind="run_service_action",
            status="success",
            summary=summary,
            reasons=["service_action_success", *action.reasons],
            tool_outcome=tool_outcome,
            task_update=updated_task,
            related_task=verification_task or next_occurrence,
            metadata={
                "service_action": service_action,
                "service_result": service_result,
                "verification_task_title": (
                    verification_task.metadata.get("title") if verification_task else None
                ),
                "task_status": updated_task.metadata.get("status") if updated_task else None,
            },
        )

    def _execute_task_service_inspection(
        self,
        action: PlannerAction,
        task: MemoryRecord,
        service_inspection: str,
    ) -> ExecutorResult:
        try:
            inspection_result = self.service_manager.inspect_action(service_inspection)
        except Exception as exc:
            updated_task, retry_scheduled, retry_metadata = self._record_task_error_state(task)
            reason = str(exc).strip() or "service_inspection_failed"
            tool_outcome = self.memory_store.record_tool_outcome(
                "service_manager",
                (
                    f"Service inspection for '{action.title}' failed: {service_inspection}. "
                    f"Reason: {reason}"
                ),
                status="error",
                subject=task.subject,
                tags=["executor", "service_inspection", "error"],
                metadata={"service_inspection": service_inspection},
            )
            return ExecutorResult(
                requested_action=action,
                executed_kind="run_service_inspection",
                status="error",
                summary=self._retry_error_summary(
                    action.title,
                    failure_kind="Service inspection",
                    failure_reason=reason,
                    retry_scheduled=retry_scheduled,
                    retry_metadata=retry_metadata,
                ),
                reasons=[
                    "service_inspection_error",
                    reason,
                    *(["retry_scheduled"] if retry_scheduled else []),
                    *action.reasons,
                ],
                tool_outcome=tool_outcome,
                task_update=updated_task,
                related_task=task,
                metadata={
                    "service_inspection": service_inspection,
                    "task_status": updated_task.metadata.get("status"),
                    **retry_metadata,
                },
            )

        updated_task: MemoryRecord | None
        next_occurrence: MemoryRecord | None = None
        if bool(task.metadata.get("complete_on_success", False)):
            completion = self.memory_store.complete_task(action.title, area=task.subject)
            updated_task = completion["completed"]
            next_occurrence = completion["next_occurrence"]
            summary = (
                f"Ran service inspection '{service_inspection}' for '{action.title}' and completed the task."
            )
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
            summary = (
                f"Ran service inspection '{service_inspection}' for '{action.title}' and kept it in progress."
            )
        verification_target = str(inspection_result.get("verification_target") or "").strip()
        parent_update = self._apply_service_inspection_followup(
            prep_title=action.title,
            inspection_result=inspection_result,
        )
        resolved_service_sync_tasks = self._resolve_service_sync_followups(
            area=task.subject,
            service_inspection=service_inspection,
            inspection_result=inspection_result,
        )
        tool_outcome = self.memory_store.record_tool_outcome(
            "service_manager",
            (
                f"Inspected service state for '{action.title}' via '{service_inspection}'."
                + (f" Verification target: {verification_target}" if verification_target else "")
            ),
            status="success",
            subject=task.subject,
            tags=["executor", "service_inspection", "success"],
            metadata={
                "service_inspection": service_inspection,
                "service_inspection_healthy": self._service_inspection_is_healthy(
                    service_inspection,
                    inspection_result,
                ),
                "resolved_service_sync_titles": [
                    str(item.metadata.get("title") or item.content)
                    for item in resolved_service_sync_tasks
                ],
            },
        )
        return ExecutorResult(
            requested_action=action,
            executed_kind="run_service_inspection",
            status="success",
            summary=summary,
            reasons=["service_inspection_success", *action.reasons],
            tool_outcome=tool_outcome,
            task_update=updated_task,
            related_task=parent_update or next_occurrence,
            metadata={
                "service_inspection": service_inspection,
                "inspection_result": inspection_result,
                "parent_task_updated": parent_update is not None,
                "resolved_service_sync_titles": [
                    str(item.metadata.get("title") or item.content)
                    for item in resolved_service_sync_tasks
                ],
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
        service_sync = self.memory_store.sync_service_tasks(self.service_manager.settings())
        if service_sync["created"] or service_sync["updated"] or service_sync["resolved"]:
            report.setdefault("executed", {})["service_sync"] = service_sync
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

    def _apply_service_inspection_followup(
        self,
        *,
        prep_title: str,
        inspection_result: dict[str, Any],
    ) -> MemoryRecord | None:
        target_title = self._target_title_from_prep_title(prep_title)
        if not target_title:
            return None
        target_task = self.memory_store.find_active_task(target_title, decorate=True)
        if target_task is None:
            return None
        inspection = inspection_result.get("inspection")
        if not isinstance(inspection, dict):
            inspection = {}
        verification_target = str(inspection_result.get("verification_target") or "").strip()
        status_value = str(inspection.get("status") or "unknown").strip() or "unknown"
        active_value = inspection.get("active")
        summary = f"Latest prep inspection: status={status_value}"
        if isinstance(active_value, bool):
            summary += f", active={'yes' if active_value else 'no'}"
        if verification_target:
            summary += f". Verification target: {verification_target}"
        current_details = str(target_task.metadata.get("details") or "").strip()
        summary_prefix = "Latest prep inspection:"
        detail_lines = [
            line
            for line in current_details.splitlines()
            if not line.strip().startswith(summary_prefix)
        ]
        updated_details = "\n".join([*detail_lines, summary]).strip()
        current_status = str(target_task.metadata.get("status", "open"))
        blocker_titles = [
            str(item).strip()
            for item in target_task.metadata.get("blocked_by", [])
            if str(item).strip()
        ]
        dependency_titles = [
            str(item).strip()
            for item in target_task.metadata.get("depends_on", [])
            if str(item).strip()
        ]
        remaining_blockers = [
            title for title in blocker_titles if not self.memory_store._task_title_is_done(title)
        ]
        unresolved_dependencies = [
            title for title in dependency_titles if not self.memory_store._task_title_is_done(title)
        ]
        updated_status = (
            "open"
            if current_status == "blocked" and not remaining_blockers and not unresolved_dependencies
            else current_status
        )
        if updated_details == current_details and updated_status == current_status:
            return target_task
        return self.memory_store.record_task(
            target_title,
            status=updated_status,
            area=target_task.subject,
            details=updated_details,
        )

    def _ensure_service_verification_task(
        self,
        *,
        action: PlannerAction,
        task: MemoryRecord,
        service_action: str,
        service_result: dict[str, Any],
    ) -> MemoryRecord | None:
        verification_target = str(service_result.get("verification_target") or "").strip()
        if not verification_target:
            return None
        verification_title = f"Verify {action.title}"
        action_label = str(task.metadata.get("service_label") or service_action).strip() or service_action
        verification_scope = self._service_verification_scope(service_action)
        superseded_tasks = self._suppress_stale_service_verification_tasks(
            area=task.subject,
            verification_title=verification_title,
            verification_scope=verification_scope,
        )
        success_message = str(
            service_result.get("message")
            or task.metadata.get("service_success_message")
            or ""
        ).strip()
        verification_details = (
            f"Confirm the post-action state after '{action_label}'. "
            f"Verification target: {verification_target}."
        )
        if success_message:
            verification_details += f" Expected result: {success_message}"
        if superseded_tasks:
            verification_details += (
                " Superseded older verification task(s): "
                + ", ".join(str(item.metadata.get("title") or item.content) for item in superseded_tasks)
                + "."
            )
        return self.memory_store.record_task(
            verification_title,
            status="open",
            area=task.subject,
            owner="agent",
            details=verification_details,
            due_date=task.metadata.get("due_date"),
            service_inspection=service_action,
            service_label=action_label,
            complete_on_success=True,
            tags=["service-verification", "post-action"],
            importance=min(max(task.importance + 0.01, 0.8), 0.94),
            confidence=0.9,
        )

    def _suppress_stale_service_verification_tasks(
        self,
        *,
        area: str,
        verification_title: str,
        verification_scope: str,
    ) -> list[MemoryRecord]:
        if not verification_scope:
            return []
        suppressed: list[MemoryRecord] = []
        for candidate in self.memory_store._active_task_memories():
            if candidate.subject != area:
                continue
            if str(candidate.metadata.get("status", "open")) == "done":
                continue
            candidate_title = str(candidate.metadata.get("title") or candidate.content).strip()
            if candidate_title == verification_title:
                continue
            tags = {str(tag).strip() for tag in candidate.tags if str(tag).strip()}
            if "service-verification" not in tags:
                continue
            candidate_scope = self._service_verification_scope(
                str(candidate.metadata.get("service_inspection") or "")
            )
            if candidate_scope != verification_scope:
                continue
            current_details = str(candidate.metadata.get("details") or "").strip()
            superseded_note = f"Superseded by '{verification_title}'."
            updated_details = (
                f"{current_details}\n{superseded_note}".strip()
                if current_details and superseded_note not in current_details
                else current_details or superseded_note
            )
            suppressed.append(
                self.memory_store.record_task(
                    candidate_title,
                    status="done",
                    area=area,
                    details=updated_details,
                )
            )
        return suppressed

    def _service_verification_scope(self, action_name: str) -> str:
        normalized = str(action_name or "").strip().lower()
        if normalized in {"install_local_service", "restart_local_service"}:
            return "local_service"
        if normalized in {"install_remote_service", "restart_remote_service"}:
            return "remote_service"
        if normalized == "install_desktop_launcher":
            return "desktop_launcher"
        return normalized

    def _resolve_service_sync_followups(
        self,
        *,
        area: str,
        service_inspection: str,
        inspection_result: dict[str, Any],
    ) -> list[MemoryRecord]:
        if not self._service_inspection_is_healthy(service_inspection, inspection_result):
            return []
        verification_scope = self._service_verification_scope(service_inspection)
        if not verification_scope:
            return []
        resolved: list[MemoryRecord] = []
        for candidate in self.memory_store._reviewable_task_views():
            if candidate.subject != area:
                continue
            tags = {str(tag).strip() for tag in candidate.tags if str(tag).strip()}
            if "service-sync" not in tags:
                continue
            candidate_action = str(candidate.metadata.get("service_action") or "").strip()
            if self._service_verification_scope(candidate_action) != verification_scope:
                continue
            title = str(candidate.metadata.get("title") or candidate.content)
            completion = self.memory_store.complete_task(title, area=candidate.subject)
            completed = completion["completed"]
            if completed is not None:
                resolved.append(completed)
        return resolved

    def _service_inspection_is_healthy(
        self,
        service_inspection: str,
        inspection_result: dict[str, Any],
    ) -> bool:
        inspection = inspection_result.get("inspection")
        if not isinstance(inspection, dict):
            return False
        scope = self._service_verification_scope(service_inspection)
        if scope in {"local_service", "remote_service"}:
            active = inspection.get("active")
            status = str(inspection.get("status") or "").strip().lower()
            return bool(active) or status == "active"
        if scope == "desktop_launcher":
            return bool(
                inspection.get("desktop_entry_installed")
                or inspection.get("launcher_installed")
            )
        return False

    def _target_title_from_prep_title(self, prep_title: str) -> str:
        prefix = "Prepare safer execution for "
        clean_title = str(prep_title or "").strip()
        if not clean_title.startswith(prefix):
            return ""
        return clean_title[len(prefix):].strip()

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

    def _resolve_batch_tasks(self, action: PlannerAction) -> list[MemoryRecord]:
        resolved: list[MemoryRecord] = []
        seen_ids: set[int] = set()
        for task_id in action.metadata.get("task_ids", []):
            try:
                candidate = self.memory_store.get_memory(int(task_id))
            except (KeyError, TypeError, ValueError):
                continue
            if candidate.archived_at is not None:
                continue
            decorated = self.memory_store.find_active_task(
                str(candidate.metadata.get("title") or candidate.content),
                area=candidate.subject,
                decorate=True,
            ) or candidate
            if decorated.id in seen_ids:
                continue
            seen_ids.add(decorated.id)
            resolved.append(decorated)

        if resolved:
            return resolved

        for title in action.metadata.get("task_titles", []):
            candidate = self.memory_store.find_active_task(
                str(title).strip(),
                decorate=True,
            )
            if candidate is None or candidate.id in seen_ids:
                continue
            seen_ids.add(candidate.id)
            resolved.append(candidate)
        return resolved

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

    def _record_task_error_state(
        self,
        task: MemoryRecord,
    ) -> tuple[MemoryRecord, bool, dict[str, Any]]:
        retry_limit = int(task.metadata.get("retry_limit", 0) or 0)
        retry_count = int(task.metadata.get("retry_count", 0) or 0)
        retry_cooldown_minutes = int(task.metadata.get("retry_cooldown_minutes", 0) or 0)
        next_retry_count = retry_count + 1
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        task_title = str(task.metadata.get("title") or task.content)
        common_kwargs = {
            "owner": str(task.metadata.get("owner") or "agent"),
            "details": task.metadata.get("details"),
            "depends_on": list(task.metadata.get("depends_on", [])),
            "blocked_by": list(task.metadata.get("blocked_by", [])),
            "due_date": task.metadata.get("due_date"),
            "recurrence_days": task.metadata.get("recurrence_days"),
            "command": task.metadata.get("command"),
            "cwd": task.metadata.get("cwd"),
            "file_operation": task.metadata.get("file_operation"),
            "file_path": task.metadata.get("file_path"),
            "file_text": task.metadata.get("file_text"),
            "find_text": task.metadata.get("find_text"),
            "symbol_name": task.metadata.get("symbol_name"),
            "replace_all": bool(task.metadata.get("replace_all", False)),
            "complete_on_success": bool(task.metadata.get("complete_on_success", False)),
            "retry_limit": retry_limit,
            "retry_count": next_retry_count if retry_limit > 0 else retry_count,
            "retry_cooldown_minutes": retry_cooldown_minutes,
            "last_retry_at": (
                now_iso if retry_limit > 0 and next_retry_count <= retry_limit else task.metadata.get("last_retry_at")
            ),
            "last_failure_at": now_iso if retry_limit > 0 else task.metadata.get("last_failure_at"),
            "tags": [tag for tag in task.tags if tag not in {"open", "in_progress", "blocked"}],
            "importance": task.importance,
            "confidence": task.confidence,
        }
        if retry_limit > 0 and next_retry_count <= retry_limit:
            snoozed_until = None
            if retry_cooldown_minutes > 0:
                snoozed_until = (now + timedelta(minutes=retry_cooldown_minutes)).isoformat()
            updated_task = self.memory_store.record_task(
                task_title,
                status="open",
                area=task.subject,
                snoozed_until=snoozed_until,
                **common_kwargs,
            )
            return updated_task, True, {
                "retry_limit": retry_limit,
                "retry_count": next_retry_count,
                "retry_cooldown_minutes": retry_cooldown_minutes,
                "snoozed_until": snoozed_until,
            }
        updated_task = (
            task
            if str(task.metadata.get("status", "open")) == "in_progress" and retry_limit <= 0
            else self.memory_store.record_task(
                task_title,
                status="in_progress",
                area=task.subject,
                snoozed_until=None,
                **common_kwargs,
            )
        )
        return updated_task, False, {
            "retry_limit": retry_limit,
            "retry_count": next_retry_count if retry_limit > 0 else retry_count,
            "retry_cooldown_minutes": retry_cooldown_minutes,
        }

    def _retry_error_summary(
        self,
        task_title: str,
        *,
        failure_kind: str,
        failure_reason: str | None,
        retry_scheduled: bool,
        retry_metadata: dict[str, Any],
    ) -> str:
        detail = failure_reason or "an error"
        if not retry_scheduled:
            return f"{failure_kind} for '{task_title}' failed with {detail}."
        retry_count = int(retry_metadata.get("retry_count", 0) or 0)
        retry_limit = int(retry_metadata.get("retry_limit", 0) or 0)
        snoozed_until = str(retry_metadata.get("snoozed_until") or "").strip()
        summary = (
            f"{failure_kind} for '{task_title}' failed with {detail}, and retry "
            f"{retry_count}/{retry_limit} was scheduled."
        )
        if snoozed_until:
            summary += f" Next retry after {snoozed_until}."
        return summary
