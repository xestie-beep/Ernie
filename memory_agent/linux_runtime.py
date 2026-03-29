from __future__ import annotations

import json
import os
import shlex
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.11+ should provide tomllib
    tomllib = None

from .action_contract import (
    ACTION_TYPE_ASK_USER,
    ACTION_TYPE_EXECUTE_PLAN_ACTION,
    ACTION_TYPE_NOOP,
    ACTION_TYPE_REPLY_ONLY,
    ValidatedModelAction,
    build_action_options,
    parse_model_action_response,
    validate_model_action,
)
from .agent import MemoryFirstAgent
from .config import (
    DEFAULT_PILOT_POLICY_PATH,
    DEFAULT_PILOT_TRACE_DIR,
    PILOT_DEFAULT_ACTION_LIMIT,
)
from .executor import ExecutorResult, MemoryExecutor
from .file_adapter import WorkspaceFileAdapter
from .model_adapter import (
    BaseModelAdapter,
    ModelResponse,
    build_default_model_adapter,
)
from .models import MemoryRecord, utc_now_iso
from .patch_runner import PatchOperation, PatchRunReport, WorkspacePatchRunner
from .planner import MemoryPlanner, PlannerAction, PlannerSnapshot
from .shell_adapter import GuardedShellAdapter

DEFAULT_AUTO_APPROVE_ACTION_KINDS = {"ask_user", "noop", "prepare_task", "run_maintenance"}
DEFAULT_AUTO_APPROVE_FILE_OPERATIONS = {"read_text"}
DEFAULT_AUTO_APPROVE_SHELL_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("python3", "-m", "unittest"),
    ("python3", "-m", "pytest"),
    ("python", "-m", "unittest"),
    ("python", "-m", "pytest"),
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


@dataclass(slots=True)
class LinuxPilotPolicy:
    name: str
    workspace_root: Path
    trace_dir: Path
    default_action_limit: int = PILOT_DEFAULT_ACTION_LIMIT
    prefer_model: bool = True
    git_write_mode: str = "auto"
    auto_approve_action_kinds: set[str] = field(default_factory=set)
    auto_approve_file_operations: set[str] = field(default_factory=set)
    auto_approve_shell_prefixes: tuple[tuple[str, ...], ...] = field(default_factory=tuple)
    loaded_from: str | None = None

    @classmethod
    def default(cls, *, workspace_root: Path | None = None) -> LinuxPilotPolicy:
        root = (workspace_root or Path.cwd()).resolve()
        return cls(
            name="linux-pilot",
            workspace_root=root,
            trace_dir=(root / DEFAULT_PILOT_TRACE_DIR).resolve(),
            default_action_limit=PILOT_DEFAULT_ACTION_LIMIT,
            prefer_model=True,
            git_write_mode="auto",
            auto_approve_action_kinds=set(DEFAULT_AUTO_APPROVE_ACTION_KINDS),
            auto_approve_file_operations=set(DEFAULT_AUTO_APPROVE_FILE_OPERATIONS),
            auto_approve_shell_prefixes=tuple(DEFAULT_AUTO_APPROVE_SHELL_PREFIXES),
        )

    @classmethod
    def load(
        cls,
        path: Path | None = None,
        *,
        workspace_root: Path | None = None,
    ) -> LinuxPilotPolicy:
        policy = cls.default(workspace_root=workspace_root)
        policy_path = (path or (policy.workspace_root / DEFAULT_PILOT_POLICY_PATH)).resolve()
        if not policy_path.exists():
            return policy
        if tomllib is None:
            raise RuntimeError("tomllib is required to load pilot policy files.")

        payload = tomllib.loads(policy_path.read_text(encoding="utf-8"))
        general = payload.get("general", {})
        approvals = payload.get("approvals", {})
        if isinstance(general, dict):
            name = str(general.get("name") or "").strip()
            if name:
                policy.name = name
            trace_dir = str(general.get("trace_dir") or "").strip()
            if trace_dir:
                policy.trace_dir = (
                    Path(trace_dir)
                    if Path(trace_dir).is_absolute()
                    else (policy.workspace_root / trace_dir)
                ).resolve()
            action_limit = general.get("default_action_limit")
            if isinstance(action_limit, int) and action_limit > 0:
                policy.default_action_limit = action_limit
            prefer_model = general.get("prefer_model")
            if isinstance(prefer_model, bool):
                policy.prefer_model = prefer_model
            git_write_mode = str(general.get("git_write_mode") or "").strip().lower()
            if git_write_mode in {"auto", "branch", "off"}:
                policy.git_write_mode = git_write_mode

        if isinstance(approvals, dict):
            action_kinds = approvals.get("auto_approve_action_kinds")
            file_operations = approvals.get("auto_approve_file_operations")
            shell_prefixes = approvals.get("auto_approve_shell_prefixes")
            if isinstance(action_kinds, list):
                policy.auto_approve_action_kinds = {
                    str(item).strip()
                    for item in action_kinds
                    if str(item).strip()
                }
            if isinstance(file_operations, list):
                policy.auto_approve_file_operations = {
                    str(item).strip()
                    for item in file_operations
                    if str(item).strip()
                }
            if isinstance(shell_prefixes, list):
                parsed_prefixes: list[tuple[str, ...]] = []
                for item in shell_prefixes:
                    clean_item = str(item).strip()
                    if not clean_item:
                        continue
                    parsed = tuple(shlex.split(clean_item, posix=True))
                    if parsed:
                        parsed_prefixes.append(parsed)
                policy.auto_approve_shell_prefixes = tuple(parsed_prefixes)

        policy.loaded_from = str(policy_path)
        return policy

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "workspace_root": str(self.workspace_root),
            "trace_dir": str(self.trace_dir),
            "default_action_limit": self.default_action_limit,
            "prefer_model": self.prefer_model,
            "git_write_mode": self.git_write_mode,
            "auto_approve_action_kinds": sorted(self.auto_approve_action_kinds),
            "auto_approve_file_operations": sorted(self.auto_approve_file_operations),
            "auto_approve_shell_prefixes": [
                " ".join(prefix) for prefix in self.auto_approve_shell_prefixes
            ],
            "loaded_from": self.loaded_from,
        }

    def render_template(self) -> str:
        shell_lines = "\n".join(
            f"{self._toml_string(' '.join(prefix))},"
            for prefix in self.auto_approve_shell_prefixes
        )
        action_lines = "\n".join(
            f"{self._toml_string(item)},"
            for item in sorted(self.auto_approve_action_kinds)
        )
        file_lines = "\n".join(
            f"{self._toml_string(item)},"
            for item in sorted(self.auto_approve_file_operations)
        )
        return (
            "# Supervised Linux pilot policy for the memory-first agent\n"
            "[general]\n"
            f"name = {self._toml_string(self.name)}\n"
            f"trace_dir = {self._toml_string(self._relative_or_absolute(self.trace_dir))}\n"
            f"default_action_limit = {self.default_action_limit}\n"
            f"prefer_model = {'true' if self.prefer_model else 'false'}\n"
            f"git_write_mode = {self._toml_string(self.git_write_mode)}\n\n"
            "[approvals]\n"
            "auto_approve_action_kinds = [\n"
            f"{self._indent_block(action_lines)}\n"
            "]\n"
            "auto_approve_file_operations = [\n"
            f"{self._indent_block(file_lines)}\n"
            "]\n"
            "auto_approve_shell_prefixes = [\n"
            f"{self._indent_block(shell_lines)}\n"
            "]\n"
        )

    def is_auto_approved_shell(self, command_text: str) -> bool:
        stripped = str(command_text or "").strip()
        if not stripped:
            return False
        try:
            argv = shlex.split(stripped, posix=os.name != "nt")
        except ValueError:
            return False
        lowered = [part.lower() for part in argv]
        return any(
            len(lowered) >= len(prefix)
            and lowered[: len(prefix)] == [part.lower() for part in prefix]
            for prefix in self.auto_approve_shell_prefixes
        )

    def _relative_or_absolute(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.workspace_root))
        except ValueError:
            return str(path)

    def _indent_block(self, text: str) -> str:
        return "\n".join(f"  {line}" for line in text.splitlines()) if text else ""

    def _toml_string(self, value: str) -> str:
        return json.dumps(str(value))


@dataclass(slots=True)
class ApprovalDecision:
    status: str
    category: str
    reason: str
    prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    preview_patch: PatchRunReport | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "category": self.category,
            "reason": self.reason,
            "prompt": self.prompt,
            "metadata": self.metadata,
            "preview_patch": (
                _patch_run_to_json(self.preview_patch)
                if self.preview_patch is not None
                else None
            ),
        }


@dataclass(slots=True)
class PilotTurnReport:
    user_event_id: int
    user_text: str
    action_limit_used: int
    context_render: str
    plan: PlannerSnapshot
    policy_status: dict[str, Any]
    model_status: dict[str, Any]
    selected_action: PlannerAction | None = None
    selected_action_source: str = "none"
    approval: ApprovalDecision | None = None
    execution_result: ExecutorResult | None = None
    after_plan: PlannerSnapshot | None = None
    model_response: ModelResponse | None = None
    model_action: ValidatedModelAction | None = None
    assistant_event_id: int | None = None
    assistant_message: str | None = None
    trace_path: str | None = None
    error: str | None = None

    def render(self) -> str:
        lines = [f"Pilot turn for event #{self.user_event_id}"]
        if self.selected_action is not None:
            lines.append(
                f"Selected action: [{self.selected_action.kind}] {self.selected_action.title}"
                f" ({self.selected_action_source})"
            )
        if self.approval is not None:
            lines.append(
                f"Approval: [{self.approval.status}] {self.approval.reason}"
            )
            if self.approval.prompt:
                lines.append(f"Approval prompt: {self.approval.prompt}")
            if self.approval.preview_patch is not None:
                preview = self.approval.preview_patch
                lines.append(
                    "Preview packet: "
                    + ", ".join(preview.changed_files[:3])
                    if preview.changed_files
                    else "Preview packet: no changed files"
                )
                if preview.diff_preview:
                    lines.extend(["Preview diff:", preview.diff_preview])
        if self.assistant_message:
            lines.extend(["", "Assistant:", self.assistant_message])
        if self.execution_result is not None:
            lines.extend(["", self.execution_result.render()])
        if self.after_plan is not None:
            lines.extend(["", "Updated plan:", self.after_plan.render()])
        if self.trace_path:
            lines.extend(["", f"Trace: {self.trace_path}"])
        if self.error:
            lines.extend(["", f"Error: {self.error}"])
        return "\n".join(lines)


@dataclass(slots=True)
class PilotRunReport:
    goal_text: str
    root_user_event_id: int | None
    max_steps: int
    action_limit_used: int
    auto_approve: bool
    use_model: bool
    steps: list[PilotTurnReport] = field(default_factory=list)
    stop_reason: str = "not_started"
    stop_summary: str = ""
    review: Any | None = None

    @property
    def executed_steps(self) -> int:
        return sum(1 for step in self.steps if step.execution_result is not None)

    @property
    def approval_requests(self) -> int:
        return sum(
            1
            for step in self.steps
            if step.approval is not None and step.approval.status == "needs_approval"
        )

    @property
    def approvals_granted(self) -> int:
        return sum(
            1
            for step in self.steps
            if step.approval is not None and step.approval.status == "approved"
        )

    @property
    def trace_paths(self) -> list[str]:
        return [
            str(step.trace_path)
            for step in self.steps
            if step.trace_path
        ]

    def render(self) -> str:
        lines = [
            f"Pilot run for goal: {self.goal_text}",
            (
                f"Stop: [{self.stop_reason}] {self.stop_summary}"
                if self.stop_summary
                else f"Stop: [{self.stop_reason}]"
            ),
            (
                f"Steps: {len(self.steps)}/{self.max_steps} | "
                f"Executed: {self.executed_steps} | "
                f"Approval requests: {self.approval_requests} | "
                f"Approvals granted: {self.approvals_granted}"
            ),
        ]
        for index, step in enumerate(self.steps, start=1):
            lines.extend(["", f"Step {index}:", step.render()])
        if self.review is not None:
            lines.extend(["", self.review.render()])
        if self.trace_paths:
            lines.extend(["", "Trace files:"])
            lines.extend(f"- {path}" for path in self.trace_paths)
        return "\n".join(lines)


class LinuxPilotRuntime:
    def __init__(
        self,
        memory_store,
        *,
        policy: LinuxPilotPolicy | None = None,
        model_adapter: BaseModelAdapter | None = None,
        shell_adapter: GuardedShellAdapter | None = None,
        file_adapter: WorkspaceFileAdapter | None = None,
        patch_runner: WorkspacePatchRunner | None = None,
    ):
        self.memory_store = memory_store
        self.policy = policy or LinuxPilotPolicy.load(workspace_root=Path.cwd())
        self.model_adapter = model_adapter or build_default_model_adapter()
        self.planner = MemoryPlanner(memory_store)
        self.executor = MemoryExecutor(
            memory_store,
            shell_adapter=shell_adapter
            or GuardedShellAdapter(workspace_root=self.policy.workspace_root),
            file_adapter=file_adapter
            or WorkspaceFileAdapter(workspace_root=self.policy.workspace_root),
        )
        self.patch_runner = patch_runner or WorkspacePatchRunner(
            memory_store,
            workspace_root=self.policy.workspace_root,
            git_mode=self.policy.git_write_mode,
        )
        self.agent = MemoryFirstAgent(memory_store, model_adapter=self.model_adapter)

    def run_turn(
        self,
        text: str,
        *,
        approve: bool = False,
        use_model: bool | None = None,
        action_limit: int | None = None,
    ) -> PilotTurnReport:
        resolved_limit = max(1, int(action_limit or self.policy.default_action_limit))
        resolved_use_model = self.policy.prefer_model if use_model is None else use_model
        event_id, _stored_memories, context, plan, model_status = self.agent._prepare_turn(
            text,
            action_limit=resolved_limit,
        )
        return self._evaluate_planned_turn(
            user_event_id=event_id,
            user_text=text,
            context_render=context.render(),
            plan=plan,
            model_status=model_status,
            approve=approve,
            use_model=resolved_use_model,
            action_limit=resolved_limit,
        )

    def run_session(
        self,
        text: str,
        *,
        max_steps: int = 5,
        auto_approve: bool = False,
        use_model: bool | None = None,
        action_limit: int | None = None,
    ) -> PilotRunReport:
        resolved_limit = max(1, int(action_limit or self.policy.default_action_limit))
        resolved_steps = max(1, int(max_steps))
        resolved_use_model = self.policy.prefer_model if use_model is None else use_model
        root_user_event_id, _stored_memories, context, plan, model_status = self.agent._prepare_turn(
            text,
            action_limit=resolved_limit,
        )

        report = PilotRunReport(
            goal_text=text,
            root_user_event_id=root_user_event_id,
            max_steps=resolved_steps,
            action_limit_used=resolved_limit,
            auto_approve=auto_approve,
            use_model=resolved_use_model,
        )

        current_context_render = context.render()
        current_plan = plan
        current_model_status = model_status
        previous_signature: tuple[str, str] | None = None
        repeated_signature_count = 0

        for step_index in range(resolved_steps):
            step = self._evaluate_planned_turn(
                user_event_id=root_user_event_id,
                user_text=text,
                context_render=current_context_render,
                plan=current_plan,
                model_status=current_model_status,
                approve=auto_approve,
                use_model=resolved_use_model,
                action_limit=resolved_limit,
            )
            report.steps.append(step)

            signature = (
                step.selected_action.kind if step.selected_action is not None else "none",
                step.selected_action.title if step.selected_action is not None else "none",
            )
            if signature == previous_signature:
                repeated_signature_count += 1
            else:
                previous_signature = signature
                repeated_signature_count = 1

            stop_reason, stop_summary = self._session_stop_for_step(
                step,
                step_index=step_index,
                max_steps=resolved_steps,
                repeated_signature_count=repeated_signature_count,
            )
            if stop_reason is not None:
                report.stop_reason = stop_reason
                report.stop_summary = stop_summary
                break

            current_plan = (
                step.after_plan
                if step.after_plan is not None
                else self.planner.build_plan(text, action_limit=resolved_limit)
            )
            current_context_render = self.memory_store.build_context(query=text).render()
            current_model_status = self.agent.model_status()
        else:
            report.stop_reason = "max_steps"
            report.stop_summary = (
                f"Stopped after reaching the session step budget of {resolved_steps}."
            )

        if not report.steps and report.stop_reason == "not_started":
            report.stop_reason = "not_started"
            report.stop_summary = "The pilot run did not execute any steps."
        return report

    def _evaluate_planned_turn(
        self,
        *,
        user_event_id: int,
        user_text: str,
        context_render: str,
        plan: PlannerSnapshot,
        model_status: dict[str, object],
        approve: bool,
        use_model: bool,
        action_limit: int,
    ) -> PilotTurnReport:
        selected_action = plan.recommendation
        selected_action_source = "planner" if selected_action is not None else "none"
        model_response = None
        model_action = None
        assistant_message = None
        error = None

        if use_model and bool(model_status.get("enabled")):
            context = self.memory_store.build_context(query=user_text)
            options = build_action_options(plan, limit=action_limit)
            messages = self.agent._build_decision_messages(
                text=user_text,
                context=context,
                plan=plan,
                options=options,
            )
            try:
                model_response = self.model_adapter.chat(messages)
                proposal = parse_model_action_response(model_response.content)
                model_action = validate_model_action(proposal, options)
            except Exception as exc:  # pragma: no cover - covered through report behavior
                error = str(exc)
            else:
                assistant_message = model_action.assistant_message
                if (
                    model_action.action_type == ACTION_TYPE_EXECUTE_PLAN_ACTION
                    and model_action.chosen_action is not None
                ):
                    selected_action = model_action.chosen_action
                    selected_action_source = "model"
                elif model_action.action_type == ACTION_TYPE_ASK_USER:
                    selected_action = PlannerAction(
                        kind="ask_user",
                        title="Ask the user for clarification",
                        summary=assistant_message
                        or "I need one detail before I can safely continue.",
                        score=0.0,
                        reasons=["model_requested_clarification"],
                        metadata={"area": "execution"},
                    )
                    selected_action_source = "model"
                elif model_action.action_type == ACTION_TYPE_NOOP:
                    selected_action = PlannerAction(
                        kind="noop",
                        title="No action required",
                        summary=assistant_message or "No immediate action is required.",
                        score=0.0,
                        reasons=["model_noop"],
                        metadata={},
                    )
                    selected_action_source = "model"
                elif model_action.fallback_to_reply and plan.recommendation is not None:
                    selected_action = plan.recommendation
                    selected_action_source = "planner_fallback"
                    assistant_message = None
                elif model_action.action_type == ACTION_TYPE_REPLY_ONLY:
                    selected_action = None
                    selected_action_source = "model_reply"

        approval = self._approval_for_action(selected_action)
        execution_result = None
        after_plan = None
        assistant_event_id = None

        if selected_action is None:
            if assistant_message:
                assistant_event_id, _ = self.agent._store_assistant_message(
                    assistant_message,
                    metadata={
                        "response_mode": "pilot_reply",
                        "pilot_policy": self.policy.name,
                        "selected_action_source": selected_action_source,
                    },
                )
        elif approval.status in {"needs_approval", "blocked"} and not approve:
            assistant_message = None
        else:
            if approval.status == "needs_approval" and approve:
                approval = self._granted_approval(approval, selected_action)
            if approval.status == "blocked":
                execution_result = None
            else:
                execution_result = self._execute_selected_action(selected_action, approval)
            after_plan = self.planner.build_plan(query=user_text, action_limit=action_limit)
            message_to_store = self._execution_message(
                selected_action,
                execution_result,
                assistant_message=assistant_message,
            )
            if message_to_store:
                assistant_message = message_to_store
                assistant_event_id, _ = self.agent._store_assistant_message(
                    message_to_store,
                    metadata={
                        "response_mode": "pilot_execution",
                        "pilot_policy": self.policy.name,
                        "selected_action_source": selected_action_source,
                        "approval_status": approval.status,
                        "execution_kind": execution_result.executed_kind,
                        "execution_status": execution_result.status,
                    },
                )

        report = PilotTurnReport(
            user_event_id=user_event_id,
            user_text=user_text,
            action_limit_used=action_limit,
            context_render=context_render,
            plan=plan,
            policy_status=self.policy.status(),
            model_status=model_status,
            selected_action=selected_action,
            selected_action_source=selected_action_source,
            approval=approval,
            execution_result=execution_result,
            after_plan=after_plan,
            model_response=model_response,
            model_action=model_action,
            assistant_event_id=assistant_event_id,
            assistant_message=assistant_message,
            error=error,
        )
        report.trace_path = self._persist_trace(report)
        return report

    def approve_turn(self, report: PilotTurnReport) -> PilotTurnReport:
        if report.selected_action is None:
            raise ValueError("Cannot approve a pilot turn with no selected action.")
        if report.execution_result is not None:
            return report

        current_approval = report.approval or self._approval_for_action(report.selected_action)
        if current_approval.status == "blocked":
            raise ValueError("Cannot approve a blocked pilot turn.")
        report.approval = self._granted_approval(current_approval, report.selected_action)
        execution_result = self._execute_selected_action(report.selected_action, report.approval)
        after_plan = self.planner.build_plan(
            query=report.user_text,
            action_limit=report.action_limit_used,
        )
        message_to_store = self._execution_message(
            report.selected_action,
            execution_result,
            assistant_message=report.assistant_message,
        )
        assistant_event_id = report.assistant_event_id
        if message_to_store:
            assistant_event_id, _ = self.agent._store_assistant_message(
                message_to_store,
                metadata={
                    "response_mode": "pilot_execution",
                    "pilot_policy": self.policy.name,
                    "selected_action_source": report.selected_action_source,
                    "approval_status": report.approval.status,
                    "execution_kind": execution_result.executed_kind,
                    "execution_status": execution_result.status,
                },
            )
        report.execution_result = execution_result
        report.after_plan = after_plan
        report.assistant_message = message_to_store
        report.assistant_event_id = assistant_event_id
        report.trace_path = self._persist_trace(report)
        return report

    def _session_stop_for_step(
        self,
        step: PilotTurnReport,
        *,
        step_index: int,
        max_steps: int,
        repeated_signature_count: int,
    ) -> tuple[str | None, str]:
        if step.error:
            return "model_error", f"Stopped because the model step failed: {step.error}"
        if (
            step.approval is not None
            and step.approval.status in {"needs_approval", "blocked"}
            and step.execution_result is None
        ):
            return (
                "needs_approval" if step.approval.status == "needs_approval" else "blocked",
                step.approval.prompt or step.approval.reason,
            )
        if step.selected_action is None:
            if step.assistant_message:
                return "reply_only", "Stopped after producing an assistant reply."
            return "no_action", "Stopped because no executor action was selected."
        if step.execution_result is None:
            return "no_execution", f"Stopped before executing '{step.selected_action.title}'."
        if step.execution_result.status == "error":
            return "execution_error", step.execution_result.summary
        if step.execution_result.status == "blocked":
            return "blocked", step.execution_result.prompt or step.execution_result.summary
        if step.selected_action.kind == "ask_user":
            return "needs_user_input", step.execution_result.prompt or step.execution_result.summary
        if step.selected_action.kind == "noop" or step.execution_result.status == "noop":
            return "noop", step.execution_result.summary
        if step.after_plan is not None and step.after_plan.recommendation is None:
            return "completed", "Stopped because there was no follow-up action to run."
        if repeated_signature_count >= 2:
            return (
                "stalled",
                f"Stopped because the same action kept recurring: '{step.selected_action.title}'.",
            )
        if step_index + 1 >= max_steps:
            return (
                "max_steps",
                f"Stopped after reaching the session step budget of {max_steps}.",
            )
        return None, ""

    def _approval_for_action(self, action: PlannerAction | None) -> ApprovalDecision:
        if action is None:
            return ApprovalDecision(
                status="no_action",
                category="none",
                reason="No executor action was selected for this turn.",
            )
        if action.kind in self.policy.auto_approve_action_kinds:
            return ApprovalDecision(
                status="auto_approved",
                category="action_kind",
                reason=f"Action kind '{action.kind}' is auto-approved by pilot policy.",
            )
        if action.kind == "prepare_task":
            return ApprovalDecision(
                status="auto_approved",
                category="prepare_task",
                reason=(
                    f"Preparation action '{action.title}' only restructures work into a safer step."
                ),
            )
        if action.kind == "work_task":
            task = self._resolve_task(action)
            return self._approval_for_task_record(task, action=action, source="work_task")
        if action.kind == "resolve_blocker":
            reroute_target = self._resolve_blocker_target(action)
            if reroute_target is None:
                return ApprovalDecision(
                    status="auto_approved",
                    category="resolve_blocker",
                    reason="Blocked task currently routes to clarification, which is auto-approved.",
                )
            return self._approval_for_task_record(
                reroute_target,
                action=action,
                source="resolve_blocker",
            )
        return ApprovalDecision(
            status="needs_approval",
            category="unknown_action",
            reason=f"Action kind '{action.kind}' is not auto-approved.",
            prompt=f"Approval required before running '{action.title}'.",
        )

    def _resolve_task(self, action: PlannerAction) -> MemoryRecord | None:
        if action.task_id is not None:
            try:
                task = self.memory_store.get_memory(action.task_id)
            except KeyError:
                task = None
            if task is not None and task.archived_at is None:
                return self.memory_store.find_active_task(
                    str(task.metadata.get("title") or action.title),
                    area=task.subject,
                    decorate=True,
                ) or task
        area = str(action.metadata.get("area") or "").strip() or None
        return self.memory_store.find_active_task(action.title, area=area, decorate=True)

    def _resolve_blocker_target(self, action: PlannerAction) -> MemoryRecord | None:
        blockers = [
            str(item).strip()
            for item in action.metadata.get("blocked_by", [])
            if str(item).strip()
        ]
        for blocker_title in blockers:
            blocker_task = self.memory_store.find_active_task(
                blocker_title,
                decorate=True,
            )
            if blocker_task is None:
                continue
            if bool(blocker_task.metadata.get("blocked_now")):
                continue
            return blocker_task
        return None

    def _approval_for_task_record(
        self,
        task: MemoryRecord | None,
        *,
        action: PlannerAction,
        source: str,
    ) -> ApprovalDecision:
        if task is None:
            return ApprovalDecision(
                status="auto_approved",
                category="missing_task",
                reason=f"Task '{action.title}' could not be resolved ahead of execution.",
            )
        command = str(task.metadata.get("command") or "").strip()
        file_operation = str(task.metadata.get("file_operation") or "").strip()
        file_path = str(task.metadata.get("file_path") or "").strip()
        task_title = str(task.metadata.get("title") or action.title).strip() or action.title
        if command and file_operation:
            return ApprovalDecision(
                status="needs_approval",
                category="ambiguous_execution",
                reason=(
                    f"Task '{action.title}' defines both shell and file execution metadata."
                ),
                prompt=(
                    f"Approval required because '{action.title}' needs execution metadata cleaned up "
                    "before it can run safely."
                ),
                metadata={"source": source},
            )
        if file_operation:
            if file_operation in self.policy.auto_approve_file_operations:
                return ApprovalDecision(
                    status="auto_approved",
                    category="file_operation",
                    reason=(
                        f"File operation '{file_operation}' on '{file_path}' is auto-approved."
                    ),
                    metadata={"source": source, "file_path": file_path},
                )
            preview = self._build_patch_preview(task, action=action)
            if preview is not None:
                if preview.status != "accepted":
                    return ApprovalDecision(
                        status="blocked",
                        category="file_operation_preview",
                        reason=(
                        f"Preview for file operation '{file_operation}' on '{file_path}' failed."
                    ),
                        prompt=(
                            "The proposed change could not produce a clean preview packet. "
                            f"Reason: {preview.rejection_reason or preview.status}."
                        ),
                        metadata={
                            "source": source,
                            "file_path": file_path,
                            "execution_task_title": task_title,
                        },
                        preview_patch=preview,
                    )
                changed_count = len(preview.changed_files)
                return ApprovalDecision(
                    status="needs_approval",
                    category="file_operation",
                    reason=(
                        f"File operation '{file_operation}' on '{file_path}' requires approval."
                    ),
                    prompt=(
                        f"Approval required before applying {changed_count} previewed file "
                        f"change(s) for '{task_title}'."
                    ),
                    metadata={
                        "source": source,
                        "file_path": file_path,
                        "execution_task_title": task_title,
                    },
                    preview_patch=preview,
                )
            return ApprovalDecision(
                status="needs_approval",
                category="file_operation",
                reason=(
                    f"File operation '{file_operation}' on '{file_path}' requires approval."
                ),
                prompt=(
                    f"Approval required before running file operation '{file_operation}' "
                    f"on '{file_path}'."
                ),
                metadata={"source": source, "file_path": file_path},
            )
        if command:
            if self.policy.is_auto_approved_shell(command):
                return ApprovalDecision(
                    status="auto_approved",
                    category="shell_command",
                    reason=f"Shell command '{command}' matches an auto-approved prefix.",
                    metadata={"source": source, "command": command},
                )
            return ApprovalDecision(
                status="needs_approval",
                category="shell_command",
                reason=f"Shell command '{command}' requires approval.",
                prompt=f"Approval required before running shell command '{command}'.",
                metadata={"source": source, "command": command},
            )
        return ApprovalDecision(
            status="auto_approved",
            category="task_state",
            reason=f"Task '{action.title}' only changes task state and is auto-approved.",
            metadata={"source": source},
        )

    def _build_patch_preview(
        self,
        task: MemoryRecord,
        *,
        action: PlannerAction,
    ) -> PatchRunReport | None:
        file_operation = str(task.metadata.get("file_operation") or "").strip()
        file_path = str(task.metadata.get("file_path") or "").strip()
        if not file_operation or not file_path:
            return None
        operation = PatchOperation(
            operation=file_operation,
            path=file_path,
            text=task.metadata.get("file_text"),
            find_text=task.metadata.get("find_text"),
            symbol_name=task.metadata.get("symbol_name"),
            replace_all=bool(task.metadata.get("replace_all", False)),
            cwd=str(task.metadata.get("cwd") or "").strip() or None,
        )
        return self.patch_runner.preview(
            f"pilot preview {str(task.metadata.get('title') or action.title)}",
            operations=[operation],
            task_title=str(task.metadata.get("title") or action.title),
        )

    def _execution_message(
        self,
        action: PlannerAction,
        result: ExecutorResult,
        *,
        assistant_message: str | None,
    ) -> str | None:
        if result is None:
            return None
        if result.status in {"blocked", "error"}:
            return result.prompt or result.summary
        if assistant_message and assistant_message.strip():
            return assistant_message
        if result.prompt:
            return result.prompt
        return result.summary

    def _execute_selected_action(
        self,
        action: PlannerAction,
        approval: ApprovalDecision,
    ) -> ExecutorResult:
        if (
            approval.preview_patch is not None
            and action.kind in {"work_task", "resolve_blocker"}
        ):
            return self._execute_preview_patch_action(action, approval)
        return self.executor.execute_action(action)

    def _execute_preview_patch_action(
        self,
        action: PlannerAction,
        approval: ApprovalDecision,
    ) -> ExecutorResult:
        preview = approval.preview_patch
        assert preview is not None
        area = str(action.metadata.get("area") or "execution")
        execution_title = str(
            approval.metadata.get("execution_task_title")
            or preview.task_title
            or action.title
        )
        report = self.patch_runner.apply_preview(
            preview,
            task_title=execution_title,
            task_area=area,
            git_mode=self.policy.git_write_mode,
        )
        status = "success" if report.status == "applied" else "error"
        summary = (
            f"Applied previewed patch for '{execution_title}' across "
            f"{len(report.changed_files)} file(s)."
            if report.status == "applied"
            else f"Previewed patch for '{execution_title}' was rejected before apply."
        )
        prompt = None
        if report.status != "applied":
            prompt = (
                f"Previewed patch for '{execution_title}' did not pass validation: "
                f"{report.rejection_reason or report.status}."
            )
        return ExecutorResult(
            requested_action=action,
            executed_kind="run_patch_preview",
            status=status,
            summary=summary,
            reasons=list(action.reasons),
            tool_outcome=report.tool_outcome,
            task_update=report.task_update,
            prompt=prompt,
            metadata={
                "patch_run_id": report.run_id,
                "changed_files": list(report.changed_files),
                "patch_status": report.status,
            },
            patch_run=report,
        )

    def _granted_approval(
        self,
        approval: ApprovalDecision,
        action: PlannerAction,
    ) -> ApprovalDecision:
        if approval.status == "approved":
            return approval
        return ApprovalDecision(
            status="approved",
            category=approval.category,
            reason=f"Approval granted for '{action.title}'.",
            prompt=approval.prompt,
            metadata={
                **approval.metadata,
                "original_status": approval.status,
            },
            preview_patch=approval.preview_patch,
        )

    def _persist_trace(self, report: PilotTurnReport) -> str:
        self.policy.trace_dir.mkdir(parents=True, exist_ok=True)
        if report.trace_path:
            trace_path = Path(report.trace_path)
        else:
            trace_name = (
                f"turn_{utc_now_iso().replace(':', '-').replace('.', '-')}_"
                f"{uuid.uuid4().hex[:8]}.json"
            )
            trace_path = self.policy.trace_dir / trace_name
        trace_path.write_text(
            json.dumps(self._report_to_dict(report), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return str(trace_path)

    def _report_to_dict(self, report: PilotTurnReport) -> dict[str, Any]:
        return {
            "user_event_id": report.user_event_id,
            "user_text": report.user_text,
            "action_limit_used": report.action_limit_used,
            "context": report.context_render,
            "plan": _plan_to_json(report.plan),
            "policy_status": report.policy_status,
            "model_status": report.model_status,
            "selected_action": (
                _plan_action_to_json(report.selected_action)
                if report.selected_action is not None
                else None
            ),
            "selected_action_source": report.selected_action_source,
            "approval": report.approval.to_dict() if report.approval is not None else None,
            "execution_result": (
                _execution_result_to_json(report.execution_result)
                if report.execution_result is not None
                else None
            ),
            "after_plan": _plan_to_json(report.after_plan) if report.after_plan else None,
            "model_response": (
                _model_response_to_json(report.model_response)
                if report.model_response is not None
                else None
            ),
            "model_action": (
                _model_action_to_json(report.model_action)
                if report.model_action is not None
                else None
            ),
            "assistant_event_id": report.assistant_event_id,
            "assistant_message": report.assistant_message,
            "trace_path": report.trace_path,
            "error": report.error,
        }


def _plan_action_to_json(action: PlannerAction) -> dict[str, Any]:
    return {
        "kind": action.kind,
        "title": action.title,
        "summary": action.summary,
        "score": round(action.score, 4),
        "reasons": list(action.reasons),
        "task_id": action.task_id,
        "evidence_memory_ids": list(action.evidence_memory_ids),
        "metadata": dict(action.metadata),
    }


def _plan_to_json(snapshot: PlannerSnapshot | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return {
        "query": snapshot.query,
        "recommendation": (
            _plan_action_to_json(snapshot.recommendation)
            if snapshot.recommendation is not None
            else None
        ),
        "alternatives": [_plan_action_to_json(action) for action in snapshot.alternatives],
        "maintenance": snapshot.maintenance,
        "pilot_history": snapshot.pilot_history,
    }


def _execution_result_to_json(result: ExecutorResult) -> dict[str, Any]:
    return {
        "requested_action": (
            _plan_action_to_json(result.requested_action)
            if result.requested_action is not None
            else None
        ),
        "executed_kind": result.executed_kind,
        "status": result.status,
        "summary": result.summary,
        "reasons": list(result.reasons),
        "tool_outcome": (
            _memory_to_json(result.tool_outcome)
            if result.tool_outcome is not None
            else None
        ),
        "task_update": (
            _memory_to_json(result.task_update)
            if result.task_update is not None
            else None
        ),
        "related_task": (
            _memory_to_json(result.related_task)
            if result.related_task is not None
            else None
        ),
        "maintenance_report": result.maintenance_report,
        "shell_result": result.shell_result.to_dict() if result.shell_result else None,
        "file_result": result.file_result.to_dict() if result.file_result else None,
        "patch_run": _patch_run_to_json(result.patch_run) if result.patch_run else None,
        "prompt": result.prompt,
        "metadata": dict(result.metadata),
    }


def _patch_run_to_json(report: PatchRunReport) -> dict[str, Any]:
    return {
        "run_id": report.run_id,
        "run_name": report.run_name,
        "suite_name": report.suite_name,
        "status": report.status,
        "task_title": report.task_title,
        "apply_on_success": report.apply_on_success,
        "applied": report.applied,
        "temp_workspace": report.temp_workspace,
        "changed_files": list(report.changed_files),
        "diff_preview": report.diff_preview,
        "operations": [operation.to_dict() for operation in report.operations],
        "operation_results": [result.to_dict() for result in report.operation_results],
        "validations": [
            validation.to_dict() for validation in report.validations
        ],
        "baseline_evaluation": report.baseline_evaluation,
        "candidate_evaluation": report.candidate_evaluation,
        "rejection_reason": report.rejection_reason,
        "preview_only": report.preview_only,
    }


def _memory_to_json(record: MemoryRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "kind": record.kind,
        "subject": record.subject,
        "content": record.content,
        "layer": record.layer,
        "tags": list(record.tags),
        "importance": record.importance,
        "confidence": record.confidence,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "archived_at": record.archived_at,
        "metadata": dict(record.metadata),
    }


def _model_response_to_json(response: ModelResponse) -> dict[str, Any]:
    return {
        "content": response.content,
        "model": response.model,
        "role": response.role,
        "thinking": response.thinking,
        "done_reason": response.done_reason,
        "prompt_eval_count": response.prompt_eval_count,
        "eval_count": response.eval_count,
    }


def _model_action_to_json(action: ValidatedModelAction) -> dict[str, Any]:
    return {
        "action_type": action.action_type,
        "assistant_message": action.assistant_message,
        "chosen_option": (
            {
                "option_id": action.chosen_option.option_id,
                "source": action.chosen_option.source,
                "action": _plan_action_to_json(action.chosen_option.action),
            }
            if action.chosen_option is not None
            else None
        ),
        "rationale": action.rationale,
        "parse_error": action.parse_error,
        "validation_error": action.validation_error,
        "fallback_to_reply": action.fallback_to_reply,
        "raw_text": action.raw_text,
    }
