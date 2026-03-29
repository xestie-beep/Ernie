from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .improvement import PilotHistoryReporter
from .memory import MemoryStore
from .models import ContextWindow, MemoryRecord

GENERIC_QUERY_TOKENS = {
    "a",
    "action",
    "best",
    "do",
    "for",
    "i",
    "me",
    "my",
    "next",
    "now",
    "plan",
    "should",
    "step",
    "the",
    "to",
    "what",
    "work",
}


@dataclass(slots=True)
class PlannerAction:
    kind: str
    title: str
    summary: str
    score: float
    reasons: list[str] = field(default_factory=list)
    task_id: int | None = None
    evidence_memory_ids: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PlannerSnapshot:
    query: str
    recommendation: PlannerAction | None = None
    alternatives: list[PlannerAction] = field(default_factory=list)
    context: ContextWindow | None = None
    recent_nudges: list[MemoryRecord] = field(default_factory=list)
    maintenance: dict[str, dict[str, Any]] = field(default_factory=dict)
    pilot_history: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        lines = [f"Plan query: {self.query}", "", "Recommended next action:"]
        if self.recommendation is None:
            lines.append("- No action surfaced. Capture a fresh goal, task, or blocker.")
            return "\n".join(lines)

        lines.extend(self._render_action(self.recommendation))
        if self.alternatives:
            lines.extend(["", "Alternatives:"])
            for action in self.alternatives:
                lines.extend(self._render_action(action))
        if self.pilot_history:
            lines.extend(["", "Pilot history signals:"])
            approval_count = int(self.pilot_history.get("approval_friction_count", 0) or 0)
            blocked_count = int(self.pilot_history.get("blocker_friction_count", 0) or 0)
            review_count = int(self.pilot_history.get("review_count", 0) or 0)
            if review_count:
                lines.append(f"- reviews considered: {review_count}")
            if approval_count:
                lines.append(f"- recurring approval friction: {approval_count}")
            if blocked_count:
                lines.append(f"- recurring blocker friction: {blocked_count}")
        return "\n".join(lines)

    def _render_action(self, action: PlannerAction) -> list[str]:
        lines = [f"- [{action.kind}] {action.summary} (score={action.score:.3f})"]
        if action.reasons:
            lines.append("  reasons: " + ", ".join(action.reasons))
        return lines


class MemoryPlanner:
    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store

    def build_plan(
        self,
        query: str = "next best action",
        *,
        action_limit: int = 5,
        context: ContextWindow | None = None,
    ) -> PlannerSnapshot:
        plan_query = query.strip() or "next best action"
        resolved_context = context or self.memory_store.build_context(plan_query)
        recent_nudges = self.memory_store.get_recent_nudges(limit=max(action_limit * 3, 8))
        maintenance = self.memory_store.maintenance_status()
        pilot_history = self._pilot_history_signals(limit=max(action_limit * 3, 8))

        candidates: list[PlannerAction] = []
        candidates.extend(
            self._build_blocker_actions(
                plan_query,
                resolved_context,
                recent_nudges,
                pilot_history,
            )
        )
        candidates.extend(
            self._build_preparation_actions(
                plan_query,
                resolved_context,
                recent_nudges,
                pilot_history,
            )
        )
        candidates.extend(
            self._build_ready_actions(
                plan_query,
                resolved_context,
                recent_nudges,
                pilot_history,
            )
        )
        candidates.extend(
            self._build_maintenance_actions(
                plan_query,
                resolved_context,
                maintenance,
                pilot_history,
            )
        )

        if not candidates:
            candidates.append(
                self._build_capture_action(
                    plan_query,
                    resolved_context,
                    maintenance,
                    pilot_history,
                )
            )

        candidates.sort(
            key=lambda action: (
                action.score,
                len(action.reasons),
                action.title,
            ),
            reverse=True,
        )
        recommendation = candidates[0] if candidates else None
        alternatives = candidates[1:action_limit]
        return PlannerSnapshot(
            query=plan_query,
            recommendation=recommendation,
            alternatives=alternatives,
            context=resolved_context,
            recent_nudges=recent_nudges,
            maintenance=maintenance,
            pilot_history=pilot_history,
        )

    def _build_preparation_actions(
        self,
        query: str,
        context: ContextWindow,
        recent_nudges: list[MemoryRecord],
        pilot_history: dict[str, Any],
    ) -> list[PlannerAction]:
        approval_friction_count = int(pilot_history.get("approval_friction_count", 0) or 0)
        if approval_friction_count < 2:
            return []

        actions: list[PlannerAction] = []
        for task in context.ready_tasks:
            if not self._likely_pilot_gated_task(task):
                continue
            if self._has_active_preparation_task(task):
                continue

            task_title = str(task.metadata.get("title") or task.content)
            prep_title = self._preparation_task_title(task_title)
            reasons = [
                "prepare_for_safer_execution",
                "approval_prone_task",
                f"pilot_history=approval_friction({approval_friction_count})",
            ]
            score = 0.78 + min(task.importance, 1.0) * 0.06
            evidence_ids = [task.id]

            if bool(task.metadata.get("overdue")):
                score += 0.08
                reasons.append("overdue")

            due_bonus, due_reason = self._due_soon_bonus(task)
            if due_bonus > 0.0 and not bool(task.metadata.get("overdue")):
                score += min(0.05, due_bonus)
                reasons.append(due_reason)

            nudge = self._match_task_nudge(task, recent_nudges)
            nudge_type = str(nudge.metadata.get("nudge_type", "")) if nudge is not None else ""
            if nudge is not None and nudge_type in {"overdue_ready", "stale_ready"}:
                score += 0.05
                reasons.append(f"nudge={nudge_type}")
                evidence_ids.append(nudge.id)

            execution_mode = self._execution_mode(task)
            query_bonus, query_reason = self._query_bonus(
                query,
                [
                    task_title,
                    prep_title,
                    task.subject,
                    str(task.metadata.get("details") or ""),
                    str(task.metadata.get("command") or ""),
                    str(task.metadata.get("file_operation") or ""),
                    str(task.metadata.get("file_path") or ""),
                    "safer preparation split decomposition approval friction",
                ],
            )
            if query_bonus > 0.0:
                score += min(0.12, query_bonus)
                reasons.append(query_reason)

            actions.append(
                PlannerAction(
                    kind="prepare_task",
                    title=prep_title,
                    summary=(
                        f"Prepare a safer execution step for '{task_title}' before running the "
                        f"{execution_mode.replace('_', ' ')} path again."
                    ),
                    score=score,
                    reasons=reasons,
                    task_id=task.id,
                    evidence_memory_ids=list(dict.fromkeys(evidence_ids)),
                    metadata={
                        "area": task.subject,
                        "status": task.metadata.get("status"),
                        "due_date": task.metadata.get("due_date"),
                        "target_task_title": task_title,
                        "prep_task_title": prep_title,
                        "prep_task_details": self._preparation_task_details(task),
                        "execution_mode": execution_mode,
                    },
                )
            )
        return actions

    def _build_ready_actions(
        self,
        query: str,
        context: ContextWindow,
        recent_nudges: list[MemoryRecord],
        pilot_history: dict[str, Any],
    ) -> list[PlannerAction]:
        actions: list[PlannerAction] = []
        approval_friction_count = int(pilot_history.get("approval_friction_count", 0) or 0)
        for task in context.ready_tasks:
            title = str(task.metadata.get("title") or task.content)
            reasons = ["ready_now"]
            score = 0.82 + min(task.importance, 1.0) * 0.08
            evidence_ids = [task.id]

            if bool(task.metadata.get("overdue")):
                score += 0.26
                reasons.append("overdue")

            due_bonus, due_reason = self._due_soon_bonus(task)
            if due_bonus > 0.0 and not bool(task.metadata.get("overdue")):
                score += due_bonus
                reasons.append(due_reason)

            escalation_level = int(task.metadata.get("escalation_level", 0) or 0)
            if escalation_level:
                score += min(0.12, escalation_level * 0.03)
                reasons.append(f"escalation_level={escalation_level}")

            nudge = self._match_task_nudge(task, recent_nudges)
            nudge_type = str(nudge.metadata.get("nudge_type", "")) if nudge is not None else ""
            if nudge is not None and nudge_type in {"overdue_ready", "stale_ready"}:
                score += 0.1
                reasons.append(f"nudge={nudge_type}")
                evidence_ids.append(nudge.id)

            query_bonus, query_reason = self._query_bonus(
                query,
                [
                    title,
                    task.subject,
                    str(task.metadata.get("details") or ""),
                    str(task.metadata.get("command") or ""),
                    str(task.metadata.get("file_operation") or ""),
                    str(task.metadata.get("file_path") or ""),
                    " ".join(str(tag) for tag in task.tags),
                ],
            )
            if query_bonus > 0.0:
                score += query_bonus
                reasons.append(query_reason)

            if approval_friction_count:
                if self._likely_pilot_gated_task(task):
                    score -= min(0.28, 0.08 + (0.04 * min(approval_friction_count, 5)))
                    reasons.append(
                        f"pilot_history=approval_friction({approval_friction_count})"
                    )
                else:
                    score += min(0.08, 0.02 * min(approval_friction_count, 4))
                    reasons.append(
                        f"pilot_history_prefers_safe_execution({approval_friction_count})"
                    )

            summary = f"Work on '{title}' now"
            if bool(task.metadata.get("overdue")):
                summary += "; it is overdue and already ready to execute."
            elif due_reason:
                summary += f"; it is ready and {due_reason.replace('_', ' ')}."
            else:
                summary += "; it is ready without active blockers."

            actions.append(
                PlannerAction(
                    kind="work_task",
                    title=title,
                    summary=summary,
                    score=score,
                    reasons=reasons,
                    task_id=task.id,
                    evidence_memory_ids=list(dict.fromkeys(evidence_ids)),
                    metadata={
                        "area": task.subject,
                        "status": task.metadata.get("status"),
                        "due_date": task.metadata.get("due_date"),
                    },
                )
            )
        return actions

    def _build_blocker_actions(
        self,
        query: str,
        context: ContextWindow,
        recent_nudges: list[MemoryRecord],
        pilot_history: dict[str, Any],
    ) -> list[PlannerAction]:
        actions: list[PlannerAction] = []
        blocker_friction_count = int(pilot_history.get("blocker_friction_count", 0) or 0)
        for task in context.open_tasks:
            if not bool(task.metadata.get("blocked_now")):
                continue
            title = str(task.metadata.get("title") or task.content)
            blockers = [
                str(item)
                for item in task.metadata.get("active_blockers", [])
                if str(item).strip()
            ] or [
                str(item)
                for item in task.metadata.get("unresolved_dependencies", [])
                if str(item).strip()
            ]
            if any(
                self._is_direct_preparation_blocker(blocker_title, task.subject)
                for blocker_title in blockers
            ):
                continue
            reasons = ["blocked"]
            score = 0.9 + min(task.importance, 1.0) * 0.08
            evidence_ids = [task.id]

            if bool(task.metadata.get("overdue")):
                score += 0.18
                reasons.append("overdue")

            escalation_level = int(task.metadata.get("escalation_level", 0) or 0)
            if escalation_level:
                score += min(0.12, escalation_level * 0.04)
                reasons.append(f"escalation_level={escalation_level}")

            nudge = self._match_task_nudge(task, recent_nudges)
            nudge_type = str(nudge.metadata.get("nudge_type", "")) if nudge is not None else ""
            if nudge is not None:
                evidence_ids.append(nudge.id)
                if nudge_type == "escalation":
                    score += 0.28
                    reasons.append("nudge=escalation")
                elif nudge_type in {"overdue_blocked", "stale_blocked"}:
                    score += 0.1
                    reasons.append(f"nudge={nudge_type}")

            if blockers:
                score += 0.03
                reasons.append("blockers=" + ",".join(blockers[:2]))

            query_bonus, query_reason = self._query_bonus(
                query,
                [
                    title,
                    task.subject,
                    " ".join(blockers),
                    str(task.metadata.get("details") or ""),
                    str(task.metadata.get("command") or ""),
                    str(task.metadata.get("file_operation") or ""),
                    str(task.metadata.get("file_path") or ""),
                    " ".join(str(tag) for tag in task.tags),
                ],
            )
            if query_bonus > 0.0:
                score += query_bonus
                reasons.append(query_reason)

            if blocker_friction_count:
                score += min(0.16, 0.04 * min(blocker_friction_count, 4))
                reasons.append(
                    f"pilot_history=blocker_friction({blocker_friction_count})"
                )

            summary = f"Resolve blocker for '{title}' next"
            if nudge_type == "escalation":
                summary += "; it has already escalated after earlier nudges."
            elif bool(task.metadata.get("overdue")):
                summary += "; it is overdue and still blocked."
            else:
                summary += "; it cannot move until the blocker is cleared."
            if blockers:
                summary += f" Current blocker: {', '.join(blockers[:2])}."

            actions.append(
                PlannerAction(
                    kind="resolve_blocker",
                    title=title,
                    summary=summary,
                    score=score,
                    reasons=reasons,
                    task_id=task.id,
                    evidence_memory_ids=list(dict.fromkeys(evidence_ids)),
                    metadata={
                        "area": task.subject,
                        "status": task.metadata.get("status"),
                        "blocked_by": blockers,
                        "due_date": task.metadata.get("due_date"),
                    },
                )
            )
        return actions

    def _build_maintenance_actions(
        self,
        query: str,
        context: ContextWindow,
        maintenance: dict[str, dict[str, Any]],
        pilot_history: dict[str, Any],
    ) -> list[PlannerAction]:
        due_names = [
            task_name
            for task_name, state in maintenance.items()
            if bool(state.get("due"))
        ]
        if not due_names:
            return []

        reasons = ["maintenance_due=" + ",".join(due_names)]
        score = 0.56 + len(due_names) * 0.05
        if not context.ready_tasks and not any(
            bool(task.metadata.get("blocked_now")) for task in context.open_tasks
        ):
            score += 0.16
            reasons.append("no_active_execution")

        pending_nudges = int(maintenance.get("task_review", {}).get("pending_nudges", 0) or 0)
        if pending_nudges:
            score += min(0.12, pending_nudges * 0.03)
            reasons.append(f"pending_nudges={pending_nudges}")

        query_bonus, query_reason = self._query_bonus(
            query,
            [" ".join(due_names), "maintenance memory upkeep reflection profile contradictions"],
        )
        if query_bonus > 0.0:
            score += query_bonus
            reasons.append(query_reason)

        approval_friction_count = int(pilot_history.get("approval_friction_count", 0) or 0)
        if approval_friction_count and not context.ready_tasks:
            score += min(0.1, 0.03 * min(approval_friction_count, 4))
            reasons.append(
                f"pilot_history_prefers_safe_maintenance({approval_friction_count})"
            )

        due_labels = ", ".join(name.replace("_", " ") for name in due_names[:3])
        return [
            PlannerAction(
                kind="run_maintenance",
                title="Run due memory maintenance",
                summary=f"Run due memory maintenance next to refresh {due_labels}.",
                score=score,
                reasons=reasons,
                metadata={"due_tasks": due_names},
            )
        ]

    def _build_capture_action(
        self,
        query: str,
        context: ContextWindow,
        maintenance: dict[str, dict[str, Any]],
        pilot_history: dict[str, Any],
    ) -> PlannerAction:
        reasons = ["no_ready_tasks", "no_blocked_priorities"]
        if any(bool(state.get("due")) for state in maintenance.values()):
            reasons.append("maintenance_pending")
        if context.memories:
            reasons.append("memory_context_available")
        approval_friction_count = int(pilot_history.get("approval_friction_count", 0) or 0)
        blocker_friction_count = int(pilot_history.get("blocker_friction_count", 0) or 0)
        if approval_friction_count:
            reasons.append(f"pilot_history=approval_friction({approval_friction_count})")
        if blocker_friction_count:
            reasons.append(f"pilot_history=blocker_friction({blocker_friction_count})")
        return PlannerAction(
            kind="ask_user",
            title="Ask the user for the next concrete goal",
            summary=(
                "No strong execution target surfaced, so ask the user for the next goal, "
                "decision, or blocker before taking action."
            ),
            score=0.25,
            reasons=reasons,
        )

    def _pilot_history_signals(self, *, limit: int) -> dict[str, Any]:
        report = PilotHistoryReporter(self.memory_store).build(limit=limit)
        stop_reason_counts = {
            str(item.get("key") or ""): int(item.get("count") or 0)
            for item in report.stop_reasons
        }
        category_counts = {
            str(item.get("key") or ""): int(item.get("count") or 0)
            for item in report.opportunity_categories
        }
        approval_friction_count = max(
            stop_reason_counts.get("needs_approval", 0),
            category_counts.get("approval_friction", 0),
        )
        blocker_friction_count = max(
            stop_reason_counts.get("blocked", 0),
            stop_reason_counts.get("needs_user_input", 0),
            category_counts.get("pilot_blocker", 0),
            category_counts.get("pilot_blocked_outcome", 0),
        )
        return {
            "review_count": report.total_reviews,
            "approval_friction_count": approval_friction_count,
            "blocker_friction_count": blocker_friction_count,
            "recurring_patterns": report.recurring_patterns,
        }

    def _execution_mode(self, task: MemoryRecord) -> str:
        command = str(task.metadata.get("command") or "").strip()
        if command:
            return "shell_command"
        file_operation = str(task.metadata.get("file_operation") or "").strip()
        if file_operation:
            return f"file_operation:{file_operation}"
        return "task_state"

    def _likely_pilot_gated_task(self, task: MemoryRecord) -> bool:
        command = str(task.metadata.get("command") or "").strip()
        file_operation = str(task.metadata.get("file_operation") or "").strip()
        if command:
            return True
        if not file_operation:
            return False
        return file_operation != "read_text"

    def _has_active_preparation_task(self, task: MemoryRecord) -> bool:
        task_title = str(task.metadata.get("title") or task.content)
        prep_title = self._preparation_task_title(task_title)
        prep_task = self.memory_store.find_active_task(prep_title, area=task.subject, decorate=True)
        if prep_task is None:
            return False
        return str(prep_task.metadata.get("status", "open")) != "done"

    def _is_direct_preparation_blocker(self, blocker_title: str, area: str) -> bool:
        clean_title = str(blocker_title).strip()
        if not clean_title.startswith("Prepare safer execution for "):
            return False
        blocker_task = self.memory_store.find_active_task(clean_title, area=area, decorate=True)
        if blocker_task is None:
            return False
        return not bool(blocker_task.metadata.get("blocked_now"))

    def _preparation_task_title(self, task_title: str) -> str:
        return f"Prepare safer execution for {task_title}"

    def _preparation_task_details(self, task: MemoryRecord) -> str:
        task_title = str(task.metadata.get("title") or task.content)
        command = str(task.metadata.get("command") or "").strip()
        file_operation = str(task.metadata.get("file_operation") or "").strip()
        file_path = str(task.metadata.get("file_path") or "").strip()
        if command:
            return (
                f"Recurring pilot approval friction suggests splitting '{task_title}' into "
                "a smaller, safer command path first. Review the command, isolate the minimum "
                "read/test step, and document the approval-ready execution batch before rerunning it."
            )
        if file_operation:
            return (
                f"Recurring pilot approval friction suggests splitting '{task_title}' into "
                "a smaller, safer file workflow first. Inspect the target, narrow the "
                f"'{file_operation}' operation on '{file_path}' into the smallest reviewed change, "
                "and stage a read-only or low-risk prep step before the main edit."
            )
        return (
            f"Recurring pilot approval friction suggests breaking '{task_title}' into a "
            "safer preparation step before the main execution path runs again."
        )

    def _match_task_nudge(
        self,
        task: MemoryRecord,
        recent_nudges: list[MemoryRecord],
    ) -> MemoryRecord | None:
        cycle_key = str(task.metadata.get("cycle_key") or "")
        title = str(task.metadata.get("title") or "").strip()
        for nudge in recent_nudges:
            nudge_cycle = str(nudge.metadata.get("task_cycle_key") or "")
            nudge_title = str(nudge.metadata.get("task_title") or "").strip()
            if cycle_key and nudge_cycle == cycle_key:
                return nudge
            if nudge_title == title:
                return nudge
        return None

    def _query_bonus(self, query: str, fields: list[str]) -> tuple[float, str]:
        query_tokens = self._query_tokens(query)
        if not query_tokens:
            return 0.0, ""
        field_tokens = {
            token
            for field in fields
            for token in self._tokenize(field)
        }
        overlap = sorted(query_tokens & field_tokens)
        if not overlap:
            return 0.0, ""
        return min(0.18, 0.06 * len(overlap)), "query_match=" + ",".join(overlap)

    def _query_tokens(self, query: str) -> set[str]:
        return {
            token
            for token in self._tokenize(query)
            if token not in GENERIC_QUERY_TOKENS
        }

    def _tokenize(self, text: str) -> set[str]:
        return {match.group(0).lower() for match in re.finditer(r"[a-z0-9_]+", text.lower())}

    def _due_soon_bonus(self, task: MemoryRecord) -> tuple[float, str]:
        due_value = str(task.metadata.get("due_date") or "").strip()
        if not due_value:
            return 0.0, ""
        due_at = self._parse_temporal_value(due_value)
        if due_at is None:
            return 0.0, ""
        delta_seconds = (due_at - datetime.now(timezone.utc)).total_seconds()
        if delta_seconds <= 0:
            return 0.0, ""
        delta_days = delta_seconds / 86400.0
        if delta_days <= 1.0:
            return 0.11, "due_within_24h"
        if delta_days <= 3.0:
            return 0.07, "due_within_3d"
        return 0.0, ""

    def _parse_temporal_value(self, value: str) -> datetime | None:
        try:
            if "T" in value:
                parsed = datetime.fromisoformat(value)
            else:
                parsed = datetime.fromisoformat(f"{value}T00:00:00+00:00")
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
