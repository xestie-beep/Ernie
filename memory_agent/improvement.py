from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .memory import MemoryStore
from .models import MemoryRecord
from .patch_runner import PatchOperation, PatchRunReport, WorkspacePatchRunner


@dataclass(slots=True)
class ImprovementOpportunity:
    title: str
    summary: str
    score: float
    category: str
    details: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ImprovementReviewReport:
    suite_name: str
    current_evaluation: dict[str, Any]
    previous_evaluation: dict[str, Any] | None = None
    best_evaluation: dict[str, Any] | None = None
    opportunities: list[ImprovementOpportunity] = field(default_factory=list)
    promoted_tasks: list[MemoryRecord] = field(default_factory=list)
    review_outcome: MemoryRecord | None = None

    @property
    def passed(self) -> bool:
        return bool(self.current_evaluation.get("passed"))

    def render(self) -> str:
        current_score = float(self.current_evaluation.get("score", 0.0) or 0.0)
        lines = [
            f"Self-improvement review: {current_score:.1%}",
            f"Suite: {self.suite_name}",
        ]
        if self.previous_evaluation is not None:
            previous_score = float(self.previous_evaluation.get("score", 0.0) or 0.0)
            lines.append(f"Previous score: {previous_score:.1%}")
        if self.best_evaluation is not None:
            best_score = float(self.best_evaluation.get("score", 0.0) or 0.0)
            lines.append(f"Best score: {best_score:.1%}")
        lines.extend(["", "Top opportunities:"])
        if not self.opportunities:
            lines.append("- none")
        else:
            for opportunity in self.opportunities[:5]:
                lines.append(
                    f"- [{opportunity.category}] {opportunity.title} "
                    f"(score={opportunity.score:.2f})"
                )
                if opportunity.details:
                    lines.append(f"  {opportunity.details}")
        if self.promoted_tasks:
            lines.extend(["", "Promoted tasks:"])
            for task in self.promoted_tasks:
                lines.append(
                    f"- [{task.subject}] {task.metadata.get('title') or task.content}"
                )
        return "\n".join(lines)


@dataclass(slots=True)
class PilotRunReviewReport:
    goal_text: str
    stop_reason: str
    stop_summary: str
    opportunities: list[ImprovementOpportunity] = field(default_factory=list)
    recurring_patterns: list[dict[str, Any]] = field(default_factory=list)
    promoted_tasks: list[MemoryRecord] = field(default_factory=list)
    review_outcome: MemoryRecord | None = None
    approval_requests: int = 0
    approvals_granted: int = 0
    executed_steps: int = 0

    def render(self) -> str:
        lines = [
            f"Pilot run review: {self.goal_text}",
            f"Stop: [{self.stop_reason}] {self.stop_summary}",
            (
                f"Executed steps: {self.executed_steps} | "
                f"Approval requests: {self.approval_requests} | "
                f"Approvals granted: {self.approvals_granted}"
            ),
            "",
            "Top opportunities:",
        ]
        if not self.opportunities:
            lines.append("- none")
        else:
            for opportunity in self.opportunities[:5]:
                lines.append(
                    f"- [{opportunity.category}] {opportunity.title} "
                    f"(score={opportunity.score:.2f})"
                )
                if opportunity.details:
                    lines.append(f"  {opportunity.details}")
        if self.recurring_patterns:
            lines.extend(["", "Recurring patterns:"])
            for pattern in self.recurring_patterns[:5]:
                lines.append(
                    f"- [{pattern.get('kind')}] {pattern.get('label')} "
                    f"(seen {pattern.get('count')} times)"
                )
        if self.promoted_tasks:
            lines.extend(["", "Promoted tasks:"])
            for task in self.promoted_tasks:
                lines.append(
                    f"- [{task.subject}] {task.metadata.get('title') or task.content}"
                )
        return "\n".join(lines)


@dataclass(slots=True)
class PilotHistoryEntry:
    review_outcome_id: int
    created_at: str
    goal_text: str
    stop_reason: str
    executed_steps: int
    approval_requests: int
    approvals_granted: int
    opportunity_count: int
    opportunity_categories: list[str] = field(default_factory=list)
    recurring_patterns: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class PilotHistoryReport:
    total_reviews: int
    window_size: int
    recent_runs: list[PilotHistoryEntry] = field(default_factory=list)
    stop_reasons: list[dict[str, Any]] = field(default_factory=list)
    opportunity_categories: list[dict[str, Any]] = field(default_factory=list)
    recurring_patterns: list[dict[str, Any]] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"Pilot report: {self.total_reviews} review(s) in the last {self.window_size} run(s)",
            "",
            "Top stop reasons:",
        ]
        if not self.stop_reasons:
            lines.append("- none")
        else:
            for item in self.stop_reasons[:5]:
                lines.append(f"- {item['label']} ({item['count']})")
        lines.extend(["", "Top friction categories:"])
        if not self.opportunity_categories:
            lines.append("- none")
        else:
            for item in self.opportunity_categories[:5]:
                lines.append(f"- {item['label']} ({item['count']})")
        lines.extend(["", "Recurring patterns:"])
        if not self.recurring_patterns:
            lines.append("- none")
        else:
            for item in self.recurring_patterns[:5]:
                lines.append(
                    f"- [{item['kind']}] {item['label']} "
                    f"(seen {item['count']} times)"
                )
        lines.extend(["", "Recent pilot reviews:"])
        if not self.recent_runs:
            lines.append("- none")
        else:
            for run in self.recent_runs[:5]:
                lines.append(
                    f"- [{run.stop_reason}] {run.goal_text} | "
                    f"steps={run.executed_steps} approvals={run.approval_requests}/{run.approvals_granted} "
                    f"opportunities={run.opportunity_count}"
                )
        return "\n".join(lines)


class PilotRunReviewer:
    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store

    def review(
        self,
        run_report: Any,
        *,
        promote_limit: int = 0,
    ) -> PilotRunReviewReport:
        recurring_patterns = self._historical_patterns(run_report)
        trusted_write_candidates = self._current_trusted_write_candidates(run_report)
        opportunities: list[ImprovementOpportunity] = []
        opportunities.extend(self._stop_reason_opportunities(run_report))
        opportunities.extend(self._approval_opportunities(run_report))
        opportunities.extend(self._failure_opportunities(run_report))
        opportunities.extend(self._historical_pattern_opportunities(run_report, recurring_patterns))

        deduped = self._dedupe_opportunities(opportunities)
        deduped.sort(
            key=lambda item: (item.score, item.category, item.title),
            reverse=True,
        )
        promoted_tasks = self._promote_opportunities(deduped, limit=promote_limit)
        review_outcome = self.memory_store.record_tool_outcome(
            "pilot-review",
            self._review_outcome_text(run_report, deduped, promoted_tasks),
            status="success" if not deduped else "blocked",
            subject="self_improvement",
            tags=["pilot", "review", "self-improvement"],
            metadata={
                "goal_text": str(getattr(run_report, "goal_text", "") or ""),
                "stop_reason": str(getattr(run_report, "stop_reason", "") or ""),
                "executed_steps": int(getattr(run_report, "executed_steps", 0) or 0),
                "approval_requests": int(getattr(run_report, "approval_requests", 0) or 0),
                "approvals_granted": int(getattr(run_report, "approvals_granted", 0) or 0),
                "recurring_patterns": recurring_patterns,
                "trusted_write_candidates": trusted_write_candidates,
                "opportunity_categories": [
                    opportunity.category for opportunity in deduped
                ],
                "opportunity_titles": [
                    opportunity.title for opportunity in deduped
                ],
                "promoted_task_ids": [task.id for task in promoted_tasks],
                "opportunity_count": len(deduped),
            },
        )
        return PilotRunReviewReport(
            goal_text=str(getattr(run_report, "goal_text", "") or ""),
            stop_reason=str(getattr(run_report, "stop_reason", "") or ""),
            stop_summary=str(getattr(run_report, "stop_summary", "") or ""),
            opportunities=deduped,
            recurring_patterns=recurring_patterns,
            promoted_tasks=promoted_tasks,
            review_outcome=review_outcome,
            approval_requests=int(getattr(run_report, "approval_requests", 0) or 0),
            approvals_granted=int(getattr(run_report, "approvals_granted", 0) or 0),
            executed_steps=int(getattr(run_report, "executed_steps", 0) or 0),
        )

    def _stop_reason_opportunities(self, run_report: Any) -> list[ImprovementOpportunity]:
        stop_reason = str(getattr(run_report, "stop_reason", "") or "")
        stop_summary = str(getattr(run_report, "stop_summary", "") or "")
        last_step = self._last_step(run_report)
        action_title = self._step_action_title(last_step) or str(
            getattr(run_report, "goal_text", "") or "the pilot run"
        )
        if stop_reason == "needs_approval":
            return [
                ImprovementOpportunity(
                    title=f"Reduce pilot approval friction for {action_title}",
                    summary=(
                        "Split the task into safer sub-steps, narrow the execution scope, "
                        "or formalize a reviewed allowlist entry so the pilot does not stall here."
                    ),
                    score=0.87,
                    category="approval_friction",
                    details=stop_summary,
                    source="pilot:needs_approval",
                    metadata={"action_title": action_title},
                )
            ]
        if stop_reason in {"blocked", "needs_user_input"}:
            return [
                ImprovementOpportunity(
                    title=f"Unblock pilot workflow for {action_title}",
                    summary=(
                        "Turn the blocking condition into a clearer prerequisite, "
                        "follow-up task, or planner rule so the pilot can make forward progress."
                    ),
                    score=0.9,
                    category="pilot_blocker",
                    details=stop_summary,
                    source=f"pilot:{stop_reason}",
                    metadata={"action_title": action_title},
                )
            ]
        if stop_reason == "execution_error":
            return [
                ImprovementOpportunity(
                    title=f"Fix pilot execution failure for {action_title}",
                    summary=(
                        "Investigate the executor or tool path that failed during the pilot run "
                        "and make that route reliable before extending capability."
                    ),
                    score=0.94,
                    category="pilot_execution_error",
                    details=stop_summary,
                    source="pilot:execution_error",
                    metadata={"action_title": action_title},
                )
            ]
        if stop_reason == "model_error":
            return [
                ImprovementOpportunity(
                    title="Stabilize pilot model decision path",
                    summary=(
                        "Harden the model decision contract or backend configuration so pilot runs "
                        "do not stop on model-side failures."
                    ),
                    score=0.91,
                    category="pilot_model_error",
                    details=stop_summary,
                    source="pilot:model_error",
                )
            ]
        if stop_reason == "stalled":
            return [
                ImprovementOpportunity(
                    title=f"Break recurring pilot loop for {action_title}",
                    summary=(
                        "Adjust planning, stop conditions, or task state transitions so the pilot "
                        "does not keep selecting the same action."
                    ),
                    score=0.83,
                    category="pilot_stall",
                    details=stop_summary,
                    source="pilot:stalled",
                    metadata={"action_title": action_title},
                )
            ]
        if stop_reason == "max_steps":
            last_after_plan = getattr(last_step, "after_plan", None)
            remaining = getattr(last_after_plan, "recommendation", None)
            if remaining is not None:
                return [
                    ImprovementOpportunity(
                        title="Improve pilot run chunking or step budget",
                        summary=(
                            "The pilot exhausted its step budget with more work still ready. "
                            "Decide whether this workflow should use smaller goals or a higher budget."
                        ),
                        score=0.68,
                        category="pilot_budget",
                        details=stop_summary,
                        source="pilot:max_steps",
                        metadata={"next_action_title": getattr(remaining, 'title', None)},
                    )
                ]
        return []

    def _approval_opportunities(self, run_report: Any) -> list[ImprovementOpportunity]:
        approvals_granted = int(getattr(run_report, "approvals_granted", 0) or 0)
        if approvals_granted <= 0:
            return []
        approved_steps = [
            step
            for step in getattr(run_report, "steps", [])
            if getattr(getattr(step, "approval", None), "status", None) == "approved"
        ]
        approved_actions = [
            self._step_action_title(step)
            for step in approved_steps
            if self._step_action_title(step)
        ]
        unique_actions = sorted(dict.fromkeys(approved_actions))
        return [
            ImprovementOpportunity(
                title="Review approved pilot actions for policy promotion",
                summary=(
                    "Inspect which manually approved actions were safe in practice and decide "
                    "whether they should become structured safe patterns or reviewed policy entries."
                ),
                score=min(0.72 + (0.03 * approvals_granted), 0.84),
                category="pilot_policy_review",
                details=", ".join(unique_actions[:4]) if unique_actions else "",
                source="pilot:approved_actions",
                metadata={"approved_action_titles": unique_actions},
            )
        ]

    def _failure_opportunities(self, run_report: Any) -> list[ImprovementOpportunity]:
        opportunities: list[ImprovementOpportunity] = []
        for step in getattr(run_report, "steps", []):
            execution_result = getattr(step, "execution_result", None)
            if execution_result is None:
                continue
            if str(getattr(execution_result, "status", "")) != "blocked":
                continue
            prompt = str(getattr(execution_result, "prompt", "") or getattr(execution_result, "summary", "")).strip()
            action_title = self._step_action_title(step) or str(
                getattr(run_report, "goal_text", "") or "the pilot run"
            )
            opportunities.append(
                ImprovementOpportunity(
                    title=f"Reduce blocked pilot outcomes for {action_title}",
                    summary=(
                        "Translate this blocked outcome into a clearer prerequisite, "
                        "routing rule, or preparation step so the pilot can continue safely."
                    ),
                    score=0.82,
                    category="pilot_blocked_outcome",
                    details=prompt,
                    source="pilot:blocked_step",
                    metadata={"action_title": action_title},
                )
            )
        return opportunities

    def _historical_patterns(self, run_report: Any, *, limit: int = 12) -> list[dict[str, Any]]:
        outcomes = self.memory_store.get_recent_tool_outcomes(
            limit=limit,
            subject="self_improvement",
            tool_name="pilot-review",
        )
        category_counts: dict[str, int] = {}
        stop_reason_counts: dict[str, int] = {}
        trusted_write_counts: dict[str, dict[str, Any]] = {}
        current_stop_reason = str(getattr(run_report, "stop_reason", "") or "")
        for outcome in outcomes:
            stop_reason = str(outcome.metadata.get("stop_reason") or "").strip()
            if stop_reason:
                stop_reason_counts[stop_reason] = stop_reason_counts.get(stop_reason, 0) + 1
            for category in outcome.metadata.get("opportunity_categories", []):
                clean_category = str(category).strip()
                if not clean_category:
                    continue
                category_counts[clean_category] = category_counts.get(clean_category, 0) + 1
            for candidate in outcome.metadata.get("trusted_write_candidates", []):
                if not isinstance(candidate, dict):
                    continue
                key = str(candidate.get("key") or "").strip()
                if not key:
                    continue
                existing = trusted_write_counts.get(key)
                count = int(candidate.get("count") or 1)
                file_operation = str(candidate.get("file_operation") or "").strip()
                file_path = str(candidate.get("file_path") or "").strip()
                if (not file_operation or not file_path) and ":" in key:
                    file_operation, file_path = key.split(":", 1)
                if existing is None or count > int(existing.get("count") or 0):
                    trusted_write_counts[key] = {
                        "kind": "trusted_write_candidate",
                        "key": key,
                        "label": str(
                            candidate.get("label")
                            or f"{file_operation} on {file_path}".strip()
                        ),
                        "count": count,
                        "file_operation": file_operation,
                        "file_path": file_path,
                    }
            for pattern in outcome.metadata.get("recurring_patterns", []):
                if not isinstance(pattern, dict):
                    continue
                if str(pattern.get("kind") or "").strip() != "trusted_write_candidate":
                    continue
                key = str(pattern.get("key") or "").strip()
                if not key:
                    continue
                existing = trusted_write_counts.get(key)
                count = int(pattern.get("count") or 0)
                file_operation = str(pattern.get("file_operation") or "").strip()
                file_path = str(pattern.get("file_path") or "").strip()
                if (not file_operation or not file_path) and ":" in key:
                    file_operation, file_path = key.split(":", 1)
                if existing is None or count > int(existing.get("count") or 0):
                    trusted_write_counts[key] = {
                        "kind": "trusted_write_candidate",
                        "key": key,
                        "label": str(
                            pattern.get("label")
                            or f"{file_operation} on {file_path}".strip()
                        ),
                        "count": count,
                        "file_operation": file_operation,
                        "file_path": file_path,
                    }

        patterns: list[dict[str, Any]] = []
        current_stop_reason_count = stop_reason_counts.get(current_stop_reason, 0) + (
            1 if current_stop_reason else 0
        )
        if current_stop_reason and current_stop_reason_count >= 2:
            patterns.append(
                {
                    "kind": "stop_reason",
                    "key": current_stop_reason,
                    "label": current_stop_reason.replace("_", " "),
                    "count": current_stop_reason_count,
                }
            )

        current_categories = {
            opportunity.category
            for opportunity in self._stop_reason_opportunities(run_report)
            + self._approval_opportunities(run_report)
            + self._failure_opportunities(run_report)
        }
        for category in sorted(current_categories):
            count = category_counts.get(category, 0) + 1
            if count >= 2:
                patterns.append(
                    {
                        "kind": "category",
                        "key": category,
                        "label": category.replace("_", " "),
                        "count": count,
                    }
                )
        for candidate in self._current_trusted_write_candidates(run_report):
            key = str(candidate.get("key") or "").strip()
            if not key:
                continue
            historical = trusted_write_counts.get(key)
            count = int(historical.get("count") or 0) if historical is not None else 0
            count += 1
            if count < 2:
                continue
            patterns.append(
                {
                    "kind": "trusted_write_candidate",
                    "key": key,
                    "label": candidate["label"],
                    "count": count,
                    "file_operation": candidate["file_operation"],
                    "file_path": candidate["file_path"],
                }
            )
        return patterns

    def _historical_pattern_opportunities(
        self,
        run_report: Any,
        patterns: list[dict[str, Any]],
    ) -> list[ImprovementOpportunity]:
        opportunities: list[ImprovementOpportunity] = []
        for pattern in patterns:
            if pattern.get("kind") == "category":
                category = str(pattern.get("key") or "").strip()
                count = int(pattern.get("count") or 0)
                if not category or count < 2:
                    continue
                title = (
                    f"Reduce recurring {category.replace('_', ' ')} across pilot runs"
                )
                opportunities.append(
                    ImprovementOpportunity(
                        title=title,
                        summary=(
                            "This friction pattern keeps showing up across recent supervised runs. "
                            "Address it at the planner, policy, or task-design level instead of "
                            "treating each occurrence as isolated."
                        ),
                        score=min(0.78 + (0.03 * min(count, 4)), 0.93),
                        category="pilot_history_pattern",
                        details=(
                            f"Matched recurring category '{category}' in {count} recent pilot reviews."
                        ),
                        source=f"pilot-history:{category}",
                        metadata={
                            "history_kind": "category",
                            "history_key": category,
                            "history_count": count,
                            "goal_text": str(getattr(run_report, 'goal_text', '') or ''),
                        },
                    )
                )
            elif pattern.get("kind") == "stop_reason":
                stop_reason = str(pattern.get("key") or "").strip()
                count = int(pattern.get("count") or 0)
                if not stop_reason or count < 2:
                    continue
                opportunities.append(
                    ImprovementOpportunity(
                        title=(
                            f"Reduce recurring pilot stop reason: "
                            f"{stop_reason.replace('_', ' ')}"
                        ),
                        summary=(
                            "Recent pilot runs are stopping for the same reason. "
                            "Tune the workflow so this stop condition becomes rarer over time."
                        ),
                        score=min(0.76 + (0.03 * min(count, 4)), 0.91),
                        category="pilot_history_pattern",
                        details=(
                            f"Pilot runs stopped with '{stop_reason}' {count} times in recent history."
                        ),
                        source=f"pilot-history-stop:{stop_reason}",
                        metadata={
                            "history_kind": "stop_reason",
                            "history_key": stop_reason,
                            "history_count": count,
                            "goal_text": str(getattr(run_report, 'goal_text', '') or ''),
                        },
                    )
                )
            elif pattern.get("kind") == "trusted_write_candidate":
                file_operation = str(pattern.get("file_operation") or "").strip()
                file_path = str(pattern.get("file_path") or "").strip()
                count = int(pattern.get("count") or 0)
                if not file_operation or not file_path or count < 2:
                    continue
                opportunities.append(
                    ImprovementOpportunity(
                        title=(
                            f"Review trusted pilot write candidate for {file_path}"
                        ),
                        summary=(
                            "The same low-risk single-file write keeps hitting approval review. "
                            "Check whether the current threshold or trusted-write settings are too strict "
                            "for this path and operation."
                        ),
                        score=min(0.8 + (0.03 * min(count, 4)), 0.92),
                        category="pilot_trusted_write_candidate",
                        details=(
                            f"Observed repeated approval friction for {file_operation} on "
                            f"{file_path} across {count} pilot reviews."
                        ),
                        source=f"pilot-history-trusted:{file_operation}:{file_path}",
                        metadata={
                            "history_kind": "trusted_write_candidate",
                            "history_key": str(pattern.get("key") or ""),
                            "history_count": count,
                            "file_operation": file_operation,
                            "file_path": file_path,
                            "goal_text": str(getattr(run_report, 'goal_text', '') or ''),
                        },
                    )
                )
        return opportunities

    def _current_trusted_write_candidates(self, run_report: Any) -> list[dict[str, Any]]:
        candidates: dict[str, dict[str, Any]] = {}
        for step in getattr(run_report, "steps", []) or []:
            approval = getattr(step, "approval", None)
            if approval is None or str(getattr(approval, "status", "")).strip() != "needs_approval":
                continue
            if str(getattr(approval, "category", "")).strip() != "file_operation":
                continue
            metadata = dict(getattr(approval, "metadata", {}) or {})
            file_operation = str(metadata.get("file_operation") or "").strip()
            file_path = str(metadata.get("file_path") or "").strip()
            preview = getattr(approval, "preview_patch", None)
            if not file_operation or not file_path or preview is None:
                continue
            if file_operation not in {"write_text", "replace_text", "append_text"}:
                continue
            if str(getattr(preview, "status", "")).strip() != "accepted":
                continue
            changed_files = list(getattr(preview, "changed_files", []) or [])
            if changed_files != [file_path]:
                continue
            key = f"{file_operation}:{file_path}"
            candidates[key] = {
                "kind": "trusted_write_candidate",
                "key": key,
                "label": f"{file_operation} on {file_path}",
                "count": 1,
                "file_operation": file_operation,
                "file_path": file_path,
            }
        return list(candidates.values())

    def _last_step(self, run_report: Any) -> Any | None:
        steps = list(getattr(run_report, "steps", []) or [])
        return steps[-1] if steps else None

    def _step_action_title(self, step: Any | None) -> str | None:
        if step is None:
            return None
        action = getattr(step, "selected_action", None)
        if action is None:
            return None
        return str(getattr(action, "title", "") or "").strip() or None

    def _promote_opportunities(
        self,
        opportunities: list[ImprovementOpportunity],
        *,
        limit: int,
    ) -> list[MemoryRecord]:
        promoted: list[MemoryRecord] = []
        for opportunity in opportunities[: max(limit, 0)]:
            details = opportunity.summary
            if opportunity.details:
                details += f" {opportunity.details}"
            task = self.memory_store.record_task(
                opportunity.title,
                status="open",
                area="self_improvement",
                owner="agent",
                details=details,
                tags=["self-improvement", "pilot-review", opportunity.category],
                importance=min(max(opportunity.score, 0.5), 0.98),
                confidence=0.9,
            )
            promoted.append(task)
        return promoted

    def _dedupe_opportunities(
        self,
        opportunities: list[ImprovementOpportunity],
    ) -> list[ImprovementOpportunity]:
        deduped: dict[str, ImprovementOpportunity] = {}
        for opportunity in opportunities:
            existing = deduped.get(opportunity.title)
            if existing is None or opportunity.score > existing.score:
                deduped[opportunity.title] = opportunity
        return list(deduped.values())

    def _review_outcome_text(
        self,
        run_report: Any,
        opportunities: list[ImprovementOpportunity],
        promoted_tasks: list[MemoryRecord],
    ) -> str:
        goal_text = str(getattr(run_report, "goal_text", "") or "pilot run")
        stop_reason = str(getattr(run_report, "stop_reason", "") or "unknown")
        if promoted_tasks:
            promoted_titles = [
                str(task.metadata.get("title") or task.content)
                for task in promoted_tasks
            ]
            return (
                f"Pilot review for '{goal_text}' stopped at {stop_reason} and promoted tasks: "
                + ", ".join(promoted_titles[:3])
            )
        if opportunities:
            return (
                f"Pilot review for '{goal_text}' stopped at {stop_reason} and found "
                f"{len(opportunities)} improvement opportunities"
            )
        return (
            f"Pilot review for '{goal_text}' stopped at {stop_reason} and found no new "
            "follow-up opportunities"
        )


class PilotHistoryReporter:
    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store

    def build(self, *, limit: int = 12) -> PilotHistoryReport:
        outcomes = self.memory_store.get_recent_tool_outcomes(
            limit=max(limit, 1),
            subject="self_improvement",
            tool_name="pilot-review",
        )
        recent_runs = [self._entry_from_outcome(outcome) for outcome in outcomes]
        stop_counts: dict[str, int] = {}
        category_counts: dict[str, int] = {}
        pattern_counts: dict[tuple[str, str], dict[str, Any]] = {}
        trusted_write_counts: dict[str, dict[str, Any]] = {}

        for run in recent_runs:
            if run.stop_reason:
                stop_counts[run.stop_reason] = stop_counts.get(run.stop_reason, 0) + 1
            for category in run.opportunity_categories:
                category_counts[category] = category_counts.get(category, 0) + 1
            for pattern in run.recurring_patterns:
                kind = str(pattern.get("kind") or "").strip()
                key = str(pattern.get("key") or "").strip()
                if not kind or not key:
                    continue
                if kind == "trusted_write_candidate":
                    existing_candidate = trusted_write_counts.get(key)
                    count = int(pattern.get("count") or 0)
                    file_operation = str(pattern.get("file_operation") or "").strip()
                    file_path = str(pattern.get("file_path") or "").strip()
                    if (not file_operation or not file_path) and ":" in key:
                        file_operation, file_path = key.split(":", 1)
                    if existing_candidate is None or count > int(existing_candidate.get("count") or 0):
                        trusted_write_counts[key] = {
                            "kind": kind,
                            "key": key,
                            "label": str(pattern.get("label") or f"{file_operation} on {file_path}"),
                            "count": count,
                            "file_operation": file_operation,
                            "file_path": file_path,
                        }
                    continue
                compound = (kind, key)
                existing = pattern_counts.get(compound)
                count = int(pattern.get("count") or 0)
                if existing is None or count > int(existing.get("count") or 0):
                    pattern_counts[compound] = {
                        "kind": kind,
                        "key": key,
                        "label": str(pattern.get("label") or key.replace("_", " ")),
                        "count": count,
                    }

        stop_reasons = [
            {
                "key": key,
                "label": key.replace("_", " "),
                "count": count,
            }
            for key, count in sorted(
                stop_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        opportunity_categories = [
            {
                "key": key,
                "label": key.replace("_", " "),
                "count": count,
            }
            for key, count in sorted(
                category_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        recurring_patterns = sorted(
            [*pattern_counts.values(), *trusted_write_counts.values()],
            key=lambda item: (-int(item.get("count") or 0), str(item.get("label") or "")),
        )

        return PilotHistoryReport(
            total_reviews=len(recent_runs),
            window_size=max(limit, 1),
            recent_runs=recent_runs,
            stop_reasons=stop_reasons,
            opportunity_categories=opportunity_categories,
            recurring_patterns=recurring_patterns,
        )

    def _entry_from_outcome(self, outcome: MemoryRecord) -> PilotHistoryEntry:
        metadata = outcome.metadata
        categories = [
            str(item).strip()
            for item in metadata.get("opportunity_categories", [])
            if str(item).strip()
        ]
        patterns = [
            pattern
            for pattern in metadata.get("recurring_patterns", [])
            if isinstance(pattern, dict)
        ]
        return PilotHistoryEntry(
            review_outcome_id=outcome.id,
            created_at=outcome.created_at,
            goal_text=str(metadata.get("goal_text") or "").strip(),
            stop_reason=str(metadata.get("stop_reason") or "").strip(),
            executed_steps=int(metadata.get("executed_steps", 0) or 0),
            approval_requests=int(metadata.get("approval_requests", 0) or 0),
            approvals_granted=int(metadata.get("approvals_granted", 0) or 0),
            opportunity_count=int(metadata.get("opportunity_count", 0) or 0),
            opportunity_categories=categories,
            recurring_patterns=patterns,
        )


class MemoryImprovementEngine:
    def __init__(
        self,
        memory_store: MemoryStore,
        evaluator: Any,
        *,
        suite_name: str = "builtin",
    ):
        self.memory_store = memory_store
        self.evaluator = evaluator
        self.suite_name = suite_name

    def review(
        self,
        *,
        promote_limit: int = 3,
        include_strategic_backlog: bool = True,
    ) -> ImprovementReviewReport:
        previous_evaluation = self.memory_store.latest_evaluation_run(suite_name=self.suite_name)
        eval_report = self.evaluator.run_builtin_suite()
        current_evaluation = self.memory_store.record_evaluation_run(self.suite_name, eval_report)
        best_evaluation = self.memory_store.best_evaluation_run(suite_name=self.suite_name)

        opportunities: list[ImprovementOpportunity] = []
        opportunities.extend(self._failing_evaluation_opportunities(current_evaluation))
        opportunities.extend(
            self._regression_opportunities(
                current=current_evaluation,
                previous=previous_evaluation,
                best=best_evaluation,
            )
        )
        opportunities.extend(self._recent_failure_opportunities())
        if include_strategic_backlog:
            opportunities.extend(self._strategic_backlog_opportunities())

        deduped = self._dedupe_opportunities(opportunities)
        deduped.sort(
            key=lambda item: (item.score, item.category, item.title),
            reverse=True,
        )
        promoted_tasks = self._promote_opportunities(deduped, limit=promote_limit)

        review_outcome = self.memory_store.record_tool_outcome(
            "improvement-review",
            self._review_outcome_text(current_evaluation, promoted_tasks),
            status="success" if bool(current_evaluation.get("passed")) else "error",
            subject="self_improvement",
            tags=["self-improvement", "review", self.suite_name],
            metadata={
                "suite_name": self.suite_name,
                "evaluation_run_id": current_evaluation["id"],
                "score": current_evaluation["score"],
                "promoted_task_ids": [task.id for task in promoted_tasks],
                "opportunity_count": len(deduped),
            },
        )
        return ImprovementReviewReport(
            suite_name=self.suite_name,
            current_evaluation=current_evaluation,
            previous_evaluation=previous_evaluation,
            best_evaluation=best_evaluation,
            opportunities=deduped,
            promoted_tasks=promoted_tasks,
            review_outcome=review_outcome,
        )

    def run_patch_candidate(
        self,
        run_name: str,
        *,
        operations: list[PatchOperation],
        validation_commands: list[str] | None = None,
        apply_on_success: bool = False,
        task_title: str | None = None,
        task_area: str = "self_improvement",
        patch_runner: WorkspacePatchRunner | None = None,
    ) -> PatchRunReport:
        runner = patch_runner or WorkspacePatchRunner(
            self.memory_store,
            suite_name=self.suite_name,
        )
        return runner.run(
            run_name,
            operations=operations,
            validation_commands=validation_commands,
            apply_on_success=apply_on_success,
            task_title=task_title,
            task_area=task_area,
            suite_name=self.suite_name,
        )

    def _failing_evaluation_opportunities(
        self,
        evaluation: dict[str, Any],
    ) -> list[ImprovementOpportunity]:
        if bool(evaluation.get("passed")):
            return []
        opportunities: list[ImprovementOpportunity] = []
        summary = evaluation.get("summary", {})
        for scenario in summary.get("scenario_results", []):
            failing_checks = [
                check for check in scenario.get("checks", []) if not bool(check.get("passed"))
            ]
            if not failing_checks:
                continue
            first_check = failing_checks[0]
            details = "; ".join(
                f"{check.get('name')}: {check.get('details')}" for check in failing_checks[:2]
            )
            opportunities.append(
                ImprovementOpportunity(
                    title=f"Fix evaluation scenario: {scenario.get('name')}",
                    summary=(
                        f"Resolve failing evaluation scenario '{scenario.get('name')}' "
                        f"so the {self.suite_name} suite returns to green."
                    ),
                    score=0.98,
                    category="evaluation_failure",
                    details=details or str(first_check.get("name") or ""),
                    source=f"evaluation:{scenario.get('name')}",
                    metadata={
                        "scenario_name": scenario.get("name"),
                        "failing_check_names": [
                            check.get("name") for check in failing_checks
                        ],
                    },
                )
            )
        return opportunities

    def _regression_opportunities(
        self,
        *,
        current: dict[str, Any],
        previous: dict[str, Any] | None,
        best: dict[str, Any] | None,
    ) -> list[ImprovementOpportunity]:
        opportunities: list[ImprovementOpportunity] = []
        current_score = float(current.get("score", 0.0) or 0.0)
        if previous is not None:
            previous_score = float(previous.get("score", 0.0) or 0.0)
            if current_score + 1e-9 < previous_score:
                opportunities.append(
                    ImprovementOpportunity(
                        title="Recover recent evaluation regression",
                        summary=(
                            "Recover the evaluation score regression before adding more capability."
                        ),
                        score=0.95,
                        category="regression",
                        details=(
                            f"Current score {current_score:.1%} is below previous score "
                            f"{previous_score:.1%}."
                        ),
                        source="evaluation:previous",
                        metadata={
                            "current_score": current_score,
                            "previous_score": previous_score,
                        },
                    )
                )
        if best is not None:
            best_score = float(best.get("score", 0.0) or 0.0)
            if current_score + 1e-9 < best_score:
                opportunities.append(
                    ImprovementOpportunity(
                        title="Close the gap to the best known evaluation score",
                        summary=(
                            "Investigate why the current build is below the best recorded evaluation score."
                        ),
                        score=0.83,
                        category="regression",
                        details=(
                            f"Current score {current_score:.1%} is below best score "
                            f"{best_score:.1%}."
                        ),
                        source="evaluation:best",
                        metadata={
                            "current_score": current_score,
                            "best_score": best_score,
                        },
                    )
                )
        return opportunities

    def _recent_failure_opportunities(self) -> list[ImprovementOpportunity]:
        outcomes = self.memory_store.get_recent_tool_outcomes(
            limit=6,
            statuses=("error", "blocked"),
        )
        opportunities: list[ImprovementOpportunity] = []
        for outcome in outcomes:
            tool_name = str(outcome.metadata.get("tool_name") or "tooling").strip()
            status = str(outcome.metadata.get("status") or "").strip() or "error"
            clean_outcome = str(outcome.metadata.get("outcome") or outcome.content).strip()
            title = f"Reduce {status} outcomes from {tool_name}"
            opportunities.append(
                ImprovementOpportunity(
                    title=title,
                    summary=(
                        f"Investigate recurring {status} outcomes from {tool_name} "
                        "and reduce that failure mode."
                    ),
                    score=0.78 if status == "blocked" else 0.8,
                    category="operational_failure",
                    details=clean_outcome,
                    source=f"tool_outcome:{outcome.id}",
                    metadata={
                        "tool_outcome_id": outcome.id,
                        "tool_name": tool_name,
                        "status": status,
                    },
                )
            )
        return opportunities

    def _strategic_backlog_opportunities(self) -> list[ImprovementOpportunity]:
        opportunities: list[ImprovementOpportunity] = []
        if self.memory_store.find_active_task(
            "Add richer task orchestration",
            area="self_improvement",
        ) is None:
            opportunities.append(
                ImprovementOpportunity(
                    title="Add richer task orchestration",
                    summary=(
                        "Extend the planner and executor with richer orchestration flows such as "
                        "delegation, batching, retry policies, and broader tool adapters."
                    ),
                    score=0.74,
                    category="strategic_backlog",
                    details=(
                        "This keeps the self-improvement backlog aligned with the current roadmap "
                        "instead of resurfacing capabilities that already shipped."
                    ),
                    source="roadmap:task_orchestration",
                    metadata={"roadmap_item": "task_orchestration"},
                )
            )
        return opportunities

    def _promote_opportunities(
        self,
        opportunities: list[ImprovementOpportunity],
        *,
        limit: int,
    ) -> list[MemoryRecord]:
        promoted: list[MemoryRecord] = []
        for opportunity in opportunities[: max(limit, 0)]:
            details = opportunity.summary
            if opportunity.details:
                details += f" {opportunity.details}"
            task = self.memory_store.record_task(
                opportunity.title,
                status="open",
                area="self_improvement",
                owner="agent",
                details=details,
                tags=["self-improvement", opportunity.category],
                importance=min(max(opportunity.score, 0.5), 0.98),
                confidence=0.9,
            )
            promoted.append(task)
        return promoted

    def _dedupe_opportunities(
        self,
        opportunities: list[ImprovementOpportunity],
    ) -> list[ImprovementOpportunity]:
        deduped: dict[str, ImprovementOpportunity] = {}
        for opportunity in opportunities:
            existing = deduped.get(opportunity.title)
            if existing is None or opportunity.score > existing.score:
                deduped[opportunity.title] = opportunity
        return list(deduped.values())

    def _review_outcome_text(
        self,
        evaluation: dict[str, Any],
        promoted_tasks: list[MemoryRecord],
    ) -> str:
        score = float(evaluation.get("score", 0.0) or 0.0)
        promoted_titles = [str(task.metadata.get("title") or task.content) for task in promoted_tasks]
        if promoted_titles:
            return (
                f"Improvement review scored {score:.1%} and promoted tasks: "
                + ", ".join(promoted_titles[:3])
            )
        return f"Improvement review scored {score:.1%} and found no new tasks to promote"
