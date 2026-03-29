from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .agent import MemoryFirstAgent
from .executor import MemoryExecutor
from .memory import MemoryStore
from .model_adapter import ModelResponse
from .models import ContextWindow, MemoryDraft, SearchResult
from .planner import MemoryPlanner


@dataclass(slots=True)
class EvalAction:
    op: str
    params: dict[str, Any]
    alias: str | None = None


@dataclass(slots=True)
class SearchExpectation:
    name: str
    query: str
    top_contains: str | None = None
    required_contains: list[str] = field(default_factory=list)
    min_results: int = 1


@dataclass(slots=True)
class ContextExpectation:
    name: str
    query: str
    required_profile_contains: list[str] = field(default_factory=list)
    required_memory_contains: list[str] = field(default_factory=list)
    required_ready_tasks: list[str] = field(default_factory=list)
    forbidden_ready_tasks: list[str] = field(default_factory=list)
    required_overdue_tasks: list[str] = field(default_factory=list)
    required_open_tasks: list[str] = field(default_factory=list)
    forbidden_open_tasks: list[str] = field(default_factory=list)
    required_task_dependencies: list[tuple[str, str]] = field(default_factory=list)
    required_task_blockers: list[tuple[str, str]] = field(default_factory=list)
    required_task_due_dates: list[tuple[str, str]] = field(default_factory=list)
    required_evidence: list[tuple[str, str]] = field(default_factory=list)
    required_contradictions: list[tuple[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class HistoryExpectation:
    name: str
    alias: str
    required_edges_to_aliases: list[tuple[str, str]] = field(default_factory=list)
    must_be_archived: bool = False


@dataclass(slots=True)
class PlanExpectation:
    name: str
    query: str
    expected_kind: str | None = None
    top_title: str | None = None
    summary_contains: str | None = None
    required_reason_fragments: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvalScenario:
    name: str
    description: str
    actions: list[EvalAction]
    search_expectations: list[SearchExpectation] = field(default_factory=list)
    context_expectations: list[ContextExpectation] = field(default_factory=list)
    history_expectations: list[HistoryExpectation] = field(default_factory=list)
    plan_expectations: list[PlanExpectation] = field(default_factory=list)


@dataclass(slots=True)
class EvalCheckResult:
    name: str
    passed: bool
    details: str


@dataclass(slots=True)
class EvalScenarioResult:
    name: str
    description: str
    passed: bool
    score: float
    checks: list[EvalCheckResult] = field(default_factory=list)


@dataclass(slots=True)
class EvalSuiteResult:
    passed: bool
    score: float
    scenario_results: list[EvalScenarioResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": round(self.score, 4),
            "scenario_results": [
                {
                    "name": result.name,
                    "description": result.description,
                    "passed": result.passed,
                    "score": round(result.score, 4),
                    "checks": [
                        {
                            "name": check.name,
                            "passed": check.passed,
                            "details": check.details,
                        }
                        for check in result.checks
                    ],
                }
                for result in self.scenario_results
            ],
        }

    def render(self) -> str:
        passed_checks = sum(
            1 for scenario in self.scenario_results for check in scenario.checks if check.passed
        )
        total_checks = sum(len(scenario.checks) for scenario in self.scenario_results)
        lines = [
            f"Memory evaluation score: {self.score:.1%}",
            f"Scenarios: {sum(1 for result in self.scenario_results if result.passed)}/{len(self.scenario_results)} passed",
            f"Checks: {passed_checks}/{total_checks} passed",
            "",
        ]
        for result in self.scenario_results:
            status = "PASS" if result.passed else "FAIL"
            lines.append(f"{status} {result.name} ({result.score:.0%})")
            lines.append(f"  {result.description}")
            for check in result.checks:
                prefix = "ok" if check.passed else "x"
                lines.append(f"  - {prefix} {check.name}: {check.details}")
            lines.append("")
        return "\n".join(lines).rstrip()


class MemoryEvaluator:
    def __init__(self, workspace_root: Path | None = None):
        self.workspace_root = Path.cwd() if workspace_root is None else workspace_root
        self.temp_root = self.workspace_root / ".eval_tmp"
        self.temp_root.mkdir(exist_ok=True)

    def run_builtin_suite(self) -> EvalSuiteResult:
        scenario_results = [self._run_scenario(scenario) for scenario in self._builtin_scenarios()]
        if not scenario_results:
            return EvalSuiteResult(passed=True, score=1.0, scenario_results=[])
        total_score = sum(result.score for result in scenario_results) / len(scenario_results)
        return EvalSuiteResult(
            passed=all(result.passed for result in scenario_results),
            score=total_score,
            scenario_results=scenario_results,
        )

    def _run_scenario(self, scenario: EvalScenario) -> EvalScenarioResult:
        db_path = self.temp_root / f"{scenario.name}_{uuid.uuid4().hex}.sqlite3"
        store = MemoryStore(db_path)
        aliases: dict[str, int] = {}
        try:
            self._execute_actions(store, scenario.actions, aliases)
            checks: list[EvalCheckResult] = []
            for expectation in scenario.search_expectations:
                checks.append(self._check_search(store, expectation))
            for expectation in scenario.context_expectations:
                checks.append(self._check_context(store, expectation))
            for expectation in scenario.history_expectations:
                checks.append(self._check_history(store, expectation, aliases))
            for expectation in scenario.plan_expectations:
                checks.append(self._check_plan(store, expectation))
            score = (
                sum(1 for check in checks if check.passed) / len(checks)
                if checks
                else 1.0
            )
            return EvalScenarioResult(
                name=scenario.name,
                description=scenario.description,
                passed=all(check.passed for check in checks),
                score=score,
                checks=checks,
            )
        finally:
            store.close()
            for candidate in (
                db_path,
                Path(f"{db_path}-wal"),
                Path(f"{db_path}-shm"),
            ):
                if candidate.exists():
                    candidate.unlink()

    def _execute_actions(
        self,
        store: MemoryStore,
        actions: list[EvalAction],
        aliases: dict[str, int],
    ) -> None:
        for action in actions:
            alias = action.alias
            params = action.params
            if action.op == "observe":
                _, stored = store.observe(
                    role=params.get("role", "user"),
                    content=params["text"],
                )
                if alias and stored:
                    aliases[alias] = stored[0].id
                continue

            if action.op == "remember":
                record = store.remember(
                    MemoryDraft(
                        kind=params["kind"],
                        subject=params["subject"],
                        content=params["content"],
                        tags=params.get("tags", []),
                        importance=params.get("importance", 0.7),
                        confidence=params.get("confidence", 0.8),
                    )
                )
                if alias:
                    aliases[alias] = record.id
                continue

            if action.op == "revise":
                record = store.revise_memory(
                    aliases[params["target_alias"]],
                    params["content"],
                )
                if alias:
                    aliases[alias] = record.id
                continue

            if action.op == "task":
                record = store.record_task(
                    params["title"],
                    status=params.get("status", "open"),
                    area=params.get("area", "execution"),
                    owner=params.get("owner", "agent"),
                    details=params.get("details"),
                    depends_on=params.get("depends_on"),
                    blocked_by=params.get("blocked_by"),
                    due_date=params.get("due_date"),
                    recurrence_days=params.get("recurrence_days"),
                    snoozed_until=params.get("snoozed_until"),
                    command=params.get("command"),
                    cwd=params.get("cwd"),
                    file_operation=params.get("file_operation"),
                    file_path=params.get("file_path"),
                    file_text=params.get("file_text"),
                    find_text=params.get("find_text"),
                    symbol_name=params.get("symbol_name"),
                    replace_all=params.get("replace_all"),
                    complete_on_success=params.get("complete_on_success"),
                    tags=params.get("tags", []),
                )
                if alias:
                    aliases[alias] = record.id
                continue

            if action.op == "complete_task":
                result = store.complete_task(
                    params["title"],
                    area=params.get("area", "execution"),
                    completed_at=params.get("completed_at"),
                )
                if alias and result["completed"] is not None:
                    aliases[alias] = result["completed"].id
                next_alias = params.get("next_alias")
                if next_alias and result["next_occurrence"] is not None:
                    aliases[next_alias] = result["next_occurrence"].id
                continue

            if action.op == "snooze_task":
                record = store.snooze_task(
                    params["title"],
                    until=params["until"],
                    area=params.get("area", "execution"),
                )
                if alias:
                    aliases[alias] = record.id
                continue

            if action.op == "resume_task":
                record = store.resume_task(
                    params["title"],
                    area=params.get("area", "execution"),
                )
                if alias:
                    aliases[alias] = record.id
                continue

            if action.op == "unblock_task":
                record = store.unblock_task(
                    params["title"],
                    area=params.get("area", "execution"),
                )
                if alias:
                    aliases[alias] = record.id
                continue

            if action.op == "backdate":
                memory_id = aliases[params["target_alias"]]
                updated_at = (
                    datetime.now(timezone.utc) - timedelta(days=int(params["days_ago"]))
                ).isoformat()
                store.connection.execute(
                    "update memories set updated_at = ? where id = ?",
                    (updated_at, memory_id),
                )
                store.connection.commit()
                continue

            if action.op == "backdate_latest_nudge":
                title = params["title"].strip()
                rows = store.connection.execute(
                    """
                    select id, metadata_json
                    from memories
                    where archived_at is null and kind = 'nudge' and layer = 'atomic'
                    order by updated_at desc, id desc
                    """
                ).fetchall()
                for row in rows:
                    metadata = json.loads(row["metadata_json"])
                    if str(metadata.get("task_title", "")).strip() != title:
                        continue
                    updated_at = (
                        datetime.now(timezone.utc) - timedelta(days=int(params["days_ago"]))
                    ).isoformat()
                    store.connection.execute(
                        "update memories set updated_at = ? where id = ?",
                        (updated_at, int(row["id"])),
                    )
                    store.connection.commit()
                    break
                continue

            if action.op == "seed_file":
                path = self.workspace_root / params["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(params["text"], encoding="utf-8")
                continue

            if action.op == "decision":
                record = store.record_decision(
                    params["topic"],
                    params["decision"],
                    rationale=params.get("rationale"),
                    tags=params.get("tags", []),
                )
                if alias:
                    aliases[alias] = record.id
                continue

            if action.op == "tool_outcome":
                record = store.record_tool_outcome(
                    params["tool_name"],
                    params["outcome"],
                    status=params.get("status", "success"),
                    subject=params.get("subject", "tooling"),
                    tags=params.get("tags", []),
                    metadata=params.get("metadata"),
                )
                if alias:
                    aliases[alias] = record.id
                continue

            if action.op == "maintain":
                store.run_maintenance(force=params.get("force", False))
                continue

            if action.op == "review_tasks":
                store.review_tasks(limit=params.get("limit", 10))
                continue

            if action.op == "execute":
                MemoryExecutor(store).execute_next(
                    query=params.get("query", "next best action"),
                    action_limit=params.get("limit", 5),
                )
                continue

            if action.op == "model_decide":
                response_text = params["response"]

                class EvalModelAdapter:
                    @property
                    def enabled(self) -> bool:
                        return True

                    def status(self) -> dict[str, Any]:
                        return {
                            "enabled": True,
                            "backend": "eval",
                            "model": "eval-main",
                        }

                    def chat(self, messages: list[Any]) -> ModelResponse:
                        return ModelResponse(
                            content=response_text,
                            model="eval-main",
                        )

                report = MemoryFirstAgent(
                    store,
                    model_adapter=EvalModelAdapter(),
                ).decide(
                    params.get("query", "what should I do next"),
                    execute_actions=params.get("execute", True),
                    action_limit=params.get("limit", 5),
                )
                if alias and report.assistant_event_id is not None:
                    aliases[alias] = report.assistant_event_id
                continue

            raise ValueError(f"Unsupported evaluation action: {action.op}")

    def _check_search(self, store: MemoryStore, expectation: SearchExpectation) -> EvalCheckResult:
        results = store.search(expectation.query, limit=max(expectation.min_results, 5))
        contents = [result.memory.content.lower() for result in results]
        passed = len(results) >= expectation.min_results
        details: list[str] = [f"{len(results)} results"]

        if expectation.top_contains is not None:
            top_match = bool(results) and expectation.top_contains.lower() in results[0].memory.content.lower()
            passed = passed and top_match
            details.append(f"top contains '{expectation.top_contains}' = {top_match}")

        for fragment in expectation.required_contains:
            found = any(fragment.lower() in content for content in contents)
            passed = passed and found
            details.append(f"contains '{fragment}' = {found}")

        return EvalCheckResult(
            name=expectation.name,
            passed=passed,
            details="; ".join(details),
        )

    def _check_context(self, store: MemoryStore, expectation: ContextExpectation) -> EvalCheckResult:
        context = store.build_context(expectation.query, memory_limit=5, recent_event_count=3)
        passed = True
        details: list[str] = []

        profile_text = " ".join(item.memory.content.lower() for item in context.profiles)
        memory_text = " ".join(item.memory.content.lower() for item in context.memories)
        ready_task_text = " ".join(
            str(task.metadata.get("title") or task.content).lower() for task in context.ready_tasks
        )
        overdue_task_text = " ".join(
            str(task.metadata.get("title") or task.content).lower() for task in context.overdue_tasks
        )
        open_task_text = " ".join(
            str(task.metadata.get("title") or task.content).lower() for task in context.open_tasks
        )

        for fragment in expectation.required_profile_contains:
            found = fragment.lower() in profile_text
            passed = passed and found
            details.append(f"profile has '{fragment}' = {found}")

        for fragment in expectation.required_memory_contains:
            found = fragment.lower() in memory_text
            passed = passed and found
            details.append(f"memory has '{fragment}' = {found}")

        for fragment in expectation.required_ready_tasks:
            found = fragment.lower() in ready_task_text
            passed = passed and found
            details.append(f"ready tasks have '{fragment}' = {found}")

        for fragment in expectation.forbidden_ready_tasks:
            found = fragment.lower() not in ready_task_text
            passed = passed and found
            details.append(f"ready tasks exclude '{fragment}' = {found}")

        for fragment in expectation.required_overdue_tasks:
            found = fragment.lower() in overdue_task_text
            passed = passed and found
            details.append(f"overdue tasks have '{fragment}' = {found}")

        for fragment in expectation.required_open_tasks:
            found = fragment.lower() in open_task_text
            passed = passed and found
            details.append(f"open loops have '{fragment}' = {found}")

        for fragment in expectation.forbidden_open_tasks:
            found = fragment.lower() not in open_task_text
            passed = passed and found
            details.append(f"open loops exclude '{fragment}' = {found}")

        for task_title, dependency_title in expectation.required_task_dependencies:
            found = self._task_metadata_contains(
                context,
                task_title,
                "depends_on",
                dependency_title,
            )
            passed = passed and found
            details.append(
                f"task '{task_title}' depends on '{dependency_title}' = {found}"
            )

        for task_title, blocker_title in expectation.required_task_blockers:
            found = self._task_metadata_contains(
                context,
                task_title,
                "blocked_by",
                blocker_title,
            )
            passed = passed and found
            details.append(
                f"task '{task_title}' blocked by '{blocker_title}' = {found}"
            )

        for task_title, due_date in expectation.required_task_due_dates:
            found = self._task_due_date_matches(context, task_title, due_date)
            passed = passed and found
            details.append(f"task '{task_title}' due '{due_date}' = {found}")

        for anchor_fragment, evidence_fragment in expectation.required_evidence:
            found = self._bundle_has_evidence(context, anchor_fragment, evidence_fragment)
            passed = passed and found
            details.append(
                f"bundle '{anchor_fragment}' has evidence '{evidence_fragment}' = {found}"
            )

        for anchor_fragment, contradiction_fragment in expectation.required_contradictions:
            found = self._bundle_has_contradiction(context, anchor_fragment, contradiction_fragment)
            passed = passed and found
            details.append(
                f"bundle '{anchor_fragment}' has contradiction '{contradiction_fragment}' = {found}"
            )

        return EvalCheckResult(
            name=expectation.name,
            passed=passed,
            details="; ".join(details) if details else "context rendered",
        )

    def _task_metadata_contains(
        self,
        context: ContextWindow,
        task_title: str,
        key: str,
        expected_value: str,
    ) -> bool:
        task_title = task_title.lower()
        expected_value = expected_value.lower()
        for task in context.open_tasks:
            title = str(task.metadata.get("title") or task.content).lower()
            if task_title != title:
                continue
            values = [
                str(item).lower()
                for item in task.metadata.get(key, [])
                if str(item).strip()
            ]
            return expected_value in values
        return False

    def _task_due_date_matches(
        self,
        context: ContextWindow,
        task_title: str,
        expected_due_date: str,
    ) -> bool:
        task_title = task_title.lower()
        for task in context.open_tasks:
            title = str(task.metadata.get("title") or task.content).lower()
            if task_title != title:
                continue
            return str(task.metadata.get("due_date") or "") == expected_due_date
        return False

    def _check_history(
        self,
        store: MemoryStore,
        expectation: HistoryExpectation,
        aliases: dict[str, int],
    ) -> EvalCheckResult:
        memory_id = aliases[expectation.alias]
        memory = store.get_memory(memory_id)
        edges = store.get_memory_edges(memory_id, direction="outgoing")
        passed = True
        details: list[str] = []

        if expectation.must_be_archived:
            archived = memory.archived_at is not None
            passed = passed and archived
            details.append(f"archived = {archived}")

        for edge_type, target_alias in expectation.required_edges_to_aliases:
            target_id = aliases[target_alias]
            found = any(edge.edge_type == edge_type and edge.to_memory_id == target_id for edge in edges)
            passed = passed and found
            details.append(f"edge {edge_type}->{target_alias} = {found}")

        return EvalCheckResult(
            name=expectation.name,
            passed=passed,
            details="; ".join(details) if details else "history verified",
        )

    def _check_plan(self, store: MemoryStore, expectation: PlanExpectation) -> EvalCheckResult:
        snapshot = MemoryPlanner(store).build_plan(expectation.query, action_limit=5)
        recommendation = snapshot.recommendation
        passed = recommendation is not None
        details: list[str] = []

        if recommendation is None:
            return EvalCheckResult(
                name=expectation.name,
                passed=False,
                details="no recommendation returned",
            )

        details.append(f"top action kind={recommendation.kind}")
        details.append(f"title='{recommendation.title}'")

        if expectation.expected_kind is not None:
            kind_match = recommendation.kind == expectation.expected_kind
            passed = passed and kind_match
            details.append(f"kind matches '{expectation.expected_kind}' = {kind_match}")

        if expectation.top_title is not None:
            title_match = expectation.top_title.lower() == recommendation.title.lower()
            passed = passed and title_match
            details.append(f"title matches '{expectation.top_title}' = {title_match}")

        if expectation.summary_contains is not None:
            summary_match = expectation.summary_contains.lower() in recommendation.summary.lower()
            passed = passed and summary_match
            details.append(
                f"summary contains '{expectation.summary_contains}' = {summary_match}"
            )

        reason_text = " ".join(recommendation.reasons).lower()
        for fragment in expectation.required_reason_fragments:
            found = fragment.lower() in reason_text
            passed = passed and found
            details.append(f"reasons contain '{fragment}' = {found}")

        return EvalCheckResult(
            name=expectation.name,
            passed=passed,
            details="; ".join(details),
        )

    def _bundle_has_evidence(
        self,
        context: ContextWindow,
        anchor_fragment: str,
        evidence_fragment: str,
    ) -> bool:
        anchor_fragment = anchor_fragment.lower()
        evidence_fragment = evidence_fragment.lower()
        for bundle in context.bundles:
            if anchor_fragment in bundle.anchor.memory.content.lower():
                if any(evidence_fragment in memory.content.lower() for memory in bundle.evidence):
                    return True
        return False

    def _bundle_has_contradiction(
        self,
        context: ContextWindow,
        anchor_fragment: str,
        contradiction_fragment: str,
    ) -> bool:
        anchor_fragment = anchor_fragment.lower()
        contradiction_fragment = contradiction_fragment.lower()
        for bundle in context.bundles:
            if anchor_fragment in bundle.anchor.memory.content.lower():
                if any(
                    contradiction_fragment in memory.content.lower()
                    for memory in bundle.contradictions
                ):
                    return True
        return False

    def _builtin_scenarios(self) -> list[EvalScenario]:
        return [
            EvalScenario(
                name="local_runtime_recall",
                description="Finds the core local runtime constraint and low-cost preference.",
                actions=[
                    EvalAction(
                        "observe",
                        {
                            "role": "user",
                            "text": (
                                "We are building from scratch. The agent must run locally on my "
                                "main pc, and low ongoing cost matters."
                            ),
                        },
                    ),
                ],
                search_expectations=[
                    SearchExpectation(
                        name="runtime search",
                        query="local runtime",
                        top_contains="run locally",
                        required_contains=["run locally"],
                    ),
                    SearchExpectation(
                        name="cost search",
                        query="low cost",
                        required_contains=["low ongoing cost"],
                    ),
                ],
            ),
            EvalScenario(
                name="latest_truth_revision",
                description="Prefers the revised latest truth and keeps history through supersession.",
                actions=[
                    EvalAction(
                        "remember",
                        {
                            "kind": "constraint",
                            "subject": "runtime",
                            "content": "The agent should run locally on the user's main PC.",
                            "tags": ["runtime", "local"],
                            "importance": 0.95,
                            "confidence": 0.95,
                        },
                        alias="runtime_original",
                    ),
                    EvalAction(
                        "revise",
                        {
                            "target_alias": "runtime_original",
                            "content": "The agent should run locally on the user's main PC and laptop.",
                        },
                        alias="runtime_current",
                    ),
                ],
                search_expectations=[
                    SearchExpectation(
                        name="latest truth search",
                        query="laptop runtime",
                        top_contains="laptop",
                    ),
                ],
                history_expectations=[
                    HistoryExpectation(
                        name="revision edge",
                        alias="runtime_current",
                        required_edges_to_aliases=[("supersedes", "runtime_original")],
                    ),
                    HistoryExpectation(
                        name="old version archived",
                        alias="runtime_original",
                        must_be_archived=True,
                    ),
                ],
            ),
            EvalScenario(
                name="entity_alias_recall",
                description="Recalls the same concept through canonical entities and aliases, not just lexical overlap.",
                actions=[
                    EvalAction(
                        "observe",
                        {
                            "role": "user",
                            "text": "My priority is implementing the memory system first.",
                        },
                    ),
                    EvalAction(
                        "decision",
                        {
                            "topic": "storage",
                            "decision": "Use SQLite as the source of truth for memory storage",
                        },
                    ),
                ],
                search_expectations=[
                    SearchExpectation(
                        name="memory alias search",
                        query="retrieval stack",
                        top_contains="memory system first",
                    ),
                    SearchExpectation(
                        name="database alias search",
                        query="database choice",
                        top_contains="SQLite",
                    ),
                ],
            ),
            EvalScenario(
                name="contradiction_surface",
                description="Surfaces contradictory memories instead of hiding or overwriting them.",
                actions=[
                    EvalAction(
                        "remember",
                        {
                            "kind": "constraint",
                            "subject": "runtime",
                            "content": "The agent should run locally on the user's main PC.",
                            "tags": ["runtime", "local"],
                            "importance": 0.95,
                            "confidence": 0.95,
                        },
                        alias="local_runtime",
                    ),
                    EvalAction(
                        "remember",
                        {
                            "kind": "constraint",
                            "subject": "runtime",
                            "content": "The agent should run in the cloud instead of locally.",
                            "tags": ["runtime", "cloud"],
                            "importance": 0.9,
                            "confidence": 0.9,
                        },
                        alias="cloud_runtime",
                    ),
                ],
                search_expectations=[
                    SearchExpectation(
                        name="cloud search",
                        query="cloud runtime",
                        top_contains="cloud",
                    ),
                ],
                context_expectations=[
                    ContextExpectation(
                        name="contradiction bundle",
                        query="cloud runtime",
                        required_contradictions=[
                            ("cloud", "locally on the user's main PC"),
                        ],
                    ),
                ],
            ),
            EvalScenario(
                name="structured_execution_profile",
                description="Builds an operational profile from decisions, tasks, and tool outcomes.",
                actions=[
                    EvalAction(
                        "decision",
                        {
                            "topic": "architecture",
                            "decision": "Use SQLite as the source of truth for memory storage",
                            "rationale": "It is cheap and local-first",
                        },
                    ),
                    EvalAction(
                        "task",
                        {
                            "title": "Add contradiction handling",
                            "status": "open",
                            "area": "architecture",
                        },
                    ),
                    EvalAction(
                        "tool_outcome",
                        {
                            "tool_name": "tests",
                            "outcome": "All verification checks passed",
                            "subject": "architecture",
                        },
                    ),
                    EvalAction("maintain", {"force": True}),
                ],
                context_expectations=[
                    ContextExpectation(
                        name="architecture profile",
                        query="architecture tasks decisions",
                        required_profile_contains=["decisions:", "tasks:", "tool outcomes:"],
                        required_evidence=[
                            (
                                "Decision: Use SQLite as the source of truth",
                                "Architecture summary",
                            ),
                        ],
                    ),
                ],
            ),
            EvalScenario(
                name="task_lifecycle_resurfacing",
                description="Keeps only the current task version active and resurfaces it as an open loop.",
                actions=[
                    EvalAction(
                        "task",
                        {
                            "title": "Ship the memory benchmark harness",
                            "status": "open",
                            "area": "execution",
                        },
                        alias="task_open",
                    ),
                    EvalAction(
                        "task",
                        {
                            "title": "Ship the memory benchmark harness",
                            "status": "in_progress",
                            "area": "execution",
                        },
                        alias="task_current",
                    ),
                    EvalAction(
                        "task",
                        {
                            "title": "Archive outdated benchmark fixtures",
                            "status": "done",
                            "area": "execution",
                        },
                        alias="task_done",
                    ),
                ],
                context_expectations=[
                    ContextExpectation(
                        name="open loop context",
                        query="current execution tasks",
                        required_open_tasks=["Ship the memory benchmark harness"],
                        forbidden_open_tasks=["Archive outdated benchmark fixtures"],
                    ),
                ],
                history_expectations=[
                    HistoryExpectation(
                        name="task supersedes prior version",
                        alias="task_current",
                        required_edges_to_aliases=[("supersedes", "task_open")],
                    ),
                    HistoryExpectation(
                        name="old task archived",
                        alias="task_open",
                        must_be_archived=True,
                    ),
                ],
            ),
            EvalScenario(
                name="task_dependency_resurfacing",
                description="Preserves dependency, blocker, and due-date state on active open loops.",
                actions=[
                    EvalAction(
                        "task",
                        {
                            "title": "Finish entity resolution",
                            "status": "in_progress",
                            "area": "execution",
                            "due_date": "2026-04-01",
                        },
                    ),
                    EvalAction(
                        "task",
                        {
                            "title": "Build task graph maintenance",
                            "status": "blocked",
                            "area": "execution",
                            "depends_on": ["Finish entity resolution"],
                            "blocked_by": ["Finish entity resolution"],
                            "due_date": "2026-04-02",
                        },
                        alias="dependent_task",
                    ),
                ],
                context_expectations=[
                    ContextExpectation(
                        name="dependency-aware open loop context",
                        query="blocked execution tasks",
                        required_open_tasks=["Build task graph maintenance"],
                        required_task_dependencies=[
                            ("Build task graph maintenance", "Finish entity resolution"),
                        ],
                        required_task_blockers=[
                            ("Build task graph maintenance", "Finish entity resolution"),
                        ],
                        required_task_due_dates=[
                            ("Build task graph maintenance", "2026-04-02"),
                        ],
                    ),
                ],
            ),
            EvalScenario(
                name="execution_priority_views",
                description="Surfaces ready and overdue work while keeping blocked tasks out of the ready queue.",
                actions=[
                    EvalAction(
                        "task",
                        {
                            "title": "Ship overdue ready memory loop",
                            "status": "open",
                            "area": "execution",
                            "due_date": "2000-01-01",
                        },
                    ),
                    EvalAction(
                        "task",
                        {
                            "title": "Ship future ready memory loop",
                            "status": "in_progress",
                            "area": "execution",
                            "due_date": "2999-01-01",
                        },
                    ),
                    EvalAction(
                        "task",
                        {
                            "title": "Blocked on external API approval",
                            "status": "blocked",
                            "area": "execution",
                            "blocked_by": ["Vendor approval"],
                            "due_date": "2000-01-01",
                        },
                    ),
                ],
                context_expectations=[
                    ContextExpectation(
                        name="ready and overdue task views",
                        query="ready overdue execution work",
                        required_ready_tasks=[
                            "Ship overdue ready memory loop",
                            "Ship future ready memory loop",
                        ],
                        forbidden_ready_tasks=["Blocked on external API approval"],
                        required_overdue_tasks=[
                            "Ship overdue ready memory loop",
                            "Blocked on external API approval",
                        ],
                    ),
                ],
            ),
            EvalScenario(
                name="task_maintenance_nudges",
                description="Generates nudges for overdue and stale tasks without needing manual review.",
                actions=[
                    EvalAction(
                        "task",
                        {
                            "title": "Ship overdue ready memory loop",
                            "status": "open",
                            "area": "execution",
                            "due_date": "2000-01-01",
                        },
                        alias="overdue_ready_task",
                    ),
                    EvalAction(
                        "task",
                        {
                            "title": "Unblock vendor approval",
                            "status": "blocked",
                            "area": "execution",
                            "blocked_by": ["Vendor approval"],
                            "due_date": "2999-01-01",
                        },
                        alias="stale_blocked_task",
                    ),
                    EvalAction(
                        "backdate",
                        {
                            "target_alias": "stale_blocked_task",
                            "days_ago": 5,
                        },
                    ),
                    EvalAction("maintain", {"force": True}),
                ],
                search_expectations=[
                    SearchExpectation(
                        name="overdue nudge search",
                        query="overdue nudge",
                        required_contains=["Ship overdue ready memory loop"],
                    ),
                    SearchExpectation(
                        name="blocked nudge search",
                        query="blocked nudge vendor approval",
                        required_contains=["Unblock vendor approval"],
                    ),
                ],
            ),
            EvalScenario(
                name="recurring_and_snoozed_execution",
                description="Rolls recurring tasks forward and keeps snoozed work out of the ready queue until resumed.",
                actions=[
                    EvalAction(
                        "task",
                        {
                            "title": "Weekly memory audit",
                            "status": "open",
                            "area": "execution",
                            "due_date": "2026-04-01",
                            "recurrence_days": 7,
                        },
                        alias="weekly_audit_open",
                    ),
                    EvalAction(
                        "complete_task",
                        {
                            "title": "Weekly memory audit",
                            "area": "execution",
                            "next_alias": "weekly_audit_next",
                        },
                        alias="weekly_audit_done",
                    ),
                    EvalAction(
                        "snooze_task",
                        {
                            "title": "Weekly memory audit",
                            "area": "execution",
                            "until": "2999-01-01",
                        },
                    ),
                    EvalAction(
                        "resume_task",
                        {
                            "title": "Weekly memory audit",
                            "area": "execution",
                        },
                    ),
                ],
                context_expectations=[
                    ContextExpectation(
                        name="recurring task context",
                        query="weekly memory audit ready work",
                        required_ready_tasks=["Weekly memory audit"],
                        required_task_due_dates=[("Weekly memory audit", "2026-04-08")],
                    ),
                ],
                history_expectations=[
                    HistoryExpectation(
                        name="recurrence edge",
                        alias="weekly_audit_done",
                        required_edges_to_aliases=[("recurs_to", "weekly_audit_next")],
                    ),
                ],
            ),
            EvalScenario(
                name="unblock_and_escalation_flow",
                description="Unblocks actionable work and escalates tasks that stay overdue after earlier nudges.",
                actions=[
                    EvalAction(
                        "task",
                        {
                            "title": "Finalize vendor migration",
                            "status": "blocked",
                            "area": "execution",
                            "blocked_by": ["Vendor approval"],
                        },
                    ),
                    EvalAction(
                        "unblock_task",
                        {
                            "title": "Finalize vendor migration",
                            "area": "execution",
                        },
                    ),
                    EvalAction(
                        "task",
                        {
                            "title": "Escalate vendor access",
                            "status": "blocked",
                            "area": "execution",
                            "blocked_by": ["Vendor approval"],
                            "due_date": "2000-01-01",
                        },
                    ),
                    EvalAction("review_tasks", {"limit": 5}),
                    EvalAction(
                        "backdate_latest_nudge",
                        {
                            "title": "Escalate vendor access",
                            "days_ago": 2,
                        },
                    ),
                    EvalAction("review_tasks", {"limit": 5}),
                ],
                context_expectations=[
                    ContextExpectation(
                        name="unblocked task ready view",
                        query="vendor migration ready work",
                        required_ready_tasks=["Finalize vendor migration"],
                    ),
                ],
                search_expectations=[
                    SearchExpectation(
                        name="escalation search",
                        query="escalation vendor access",
                        top_contains="Escalation:",
                        required_contains=["Escalate vendor access"],
                    ),
                ],
            ),
            EvalScenario(
                name="planner_next_action",
                description="Chooses the next action from ready work, blocked escalations, and maintenance state.",
                actions=[
                    EvalAction(
                        "task",
                        {
                            "title": "Ship overdue ready memory loop",
                            "status": "open",
                            "area": "execution",
                            "due_date": "2000-01-01",
                        },
                    ),
                    EvalAction(
                        "task",
                        {
                            "title": "Escalate vendor access",
                            "status": "blocked",
                            "area": "execution",
                            "blocked_by": ["Vendor approval"],
                            "due_date": "2000-01-01",
                        },
                    ),
                    EvalAction("review_tasks", {"limit": 5}),
                    EvalAction(
                        "backdate_latest_nudge",
                        {
                            "title": "Escalate vendor access",
                            "days_ago": 2,
                        },
                    ),
                    EvalAction("review_tasks", {"limit": 5}),
                ],
                plan_expectations=[
                    PlanExpectation(
                        name="planner recommendation",
                        query="what should I do next",
                        expected_kind="resolve_blocker",
                        top_title="Escalate vendor access",
                        summary_contains="escalated",
                        required_reason_fragments=["nudge=escalation", "blocked"],
                    ),
                ],
            ),
            EvalScenario(
                name="executor_progress_loop",
                description="Executes the chosen action, records the outcome, and updates task state.",
                actions=[
                    EvalAction(
                        "task",
                        {
                            "title": "Ship overdue ready memory loop",
                            "status": "open",
                            "area": "execution",
                            "due_date": "2000-01-01",
                        },
                    ),
                    EvalAction(
                        "execute",
                        {
                            "query": "what should I do next",
                            "limit": 3,
                        },
                    ),
                ],
                search_expectations=[
                    SearchExpectation(
                        name="executor outcome search",
                        query="executor started task",
                        required_contains=["Started task 'Ship overdue ready memory loop'"],
                    ),
                ],
                context_expectations=[
                    ContextExpectation(
                        name="task moved in progress",
                        query="ship overdue ready memory loop",
                        required_open_tasks=["Ship overdue ready memory loop"],
                    ),
                ],
            ),
            EvalScenario(
                name="pilot_history_preparation",
                description="Recurring approval friction can surface a safer preparation task before risky work runs again.",
                actions=[
                    EvalAction(
                        "tool_outcome",
                        {
                            "tool_name": "pilot-review",
                            "outcome": "Pilot review 0 stopped on approval friction",
                            "status": "blocked",
                            "subject": "self_improvement",
                            "tags": ["pilot", "review", "self-improvement"],
                            "metadata": {
                                "goal_text": "pilot history seed 0",
                                "stop_reason": "needs_approval",
                                "executed_steps": 0,
                                "approval_requests": 1,
                                "approvals_granted": 0,
                                "opportunity_count": 1,
                                "opportunity_categories": ["approval_friction"],
                                "recurring_patterns": [],
                            },
                        },
                    ),
                    EvalAction(
                        "tool_outcome",
                        {
                            "tool_name": "pilot-review",
                            "outcome": "Pilot review 1 stopped on approval friction",
                            "status": "blocked",
                            "subject": "self_improvement",
                            "tags": ["pilot", "review", "self-improvement"],
                            "metadata": {
                                "goal_text": "pilot history seed 1",
                                "stop_reason": "needs_approval",
                                "executed_steps": 0,
                                "approval_requests": 1,
                                "approvals_granted": 0,
                                "opportunity_count": 1,
                                "opportunity_categories": ["approval_friction"],
                                "recurring_patterns": [],
                            },
                        },
                    ),
                    EvalAction(
                        "task",
                        {
                            "title": "Pilot risky write",
                            "status": "open",
                            "area": "execution",
                            "file_operation": "write_text",
                            "file_path": ".test_tmp/pilot_risky_write_eval.txt",
                            "file_text": "updated\n",
                            "complete_on_success": True,
                        },
                    ),
                ],
                plan_expectations=[
                    PlanExpectation(
                        name="planner recommends prep task",
                        query="what should I do next",
                        expected_kind="prepare_task",
                        top_title="Prepare safer execution for Pilot risky write",
                        summary_contains="safer execution step",
                        required_reason_fragments=[
                            "prepare_for_safer_execution",
                            "pilot_history=approval_friction(2)",
                        ],
                    ),
                ],
            ),
            EvalScenario(
                name="model_structured_action_contract",
                description="Lets the main model choose a planner-approved action and routes it through the executor safely.",
                actions=[
                    EvalAction(
                        "task",
                        {
                            "title": "Ship overdue ready memory loop",
                            "status": "open",
                            "area": "execution",
                            "due_date": "2000-01-01",
                        },
                    ),
                    EvalAction(
                        "model_decide",
                        {
                            "query": "What should I do next?",
                            "execute": True,
                            "limit": 3,
                            "response": (
                                '{"assistant_message":"I am starting the overdue ready task now.",'
                                '"action":{"type":"execute_plan_action","option_id":"A1"}}'
                            ),
                        },
                    ),
                ],
                search_expectations=[
                    SearchExpectation(
                        name="model decision executor outcome",
                        query="executor started overdue ready task",
                        required_contains=["Started task 'Ship overdue ready memory loop'"],
                    ),
                ],
                context_expectations=[
                    ContextExpectation(
                        name="model decision updated task state",
                        query="ship overdue ready memory loop",
                        required_open_tasks=["Ship overdue ready memory loop"],
                    ),
                ],
            ),
            EvalScenario(
                name="shell_command_execution",
                description="Runs an allowed shell command for a task and closes the task on success.",
                actions=[
                    EvalAction(
                        "task",
                        {
                            "title": "Check CLI help",
                            "status": "open",
                            "area": "execution",
                            "command": "python -m memory_agent.cli --help",
                            "complete_on_success": True,
                        },
                    ),
                    EvalAction(
                        "execute",
                        {
                            "query": "check cli help",
                            "limit": 3,
                        },
                    ),
                ],
                search_expectations=[
                    SearchExpectation(
                        name="shell outcome search",
                        query="shell cli help",
                        required_contains=["Check CLI help"],
                    ),
                ],
                context_expectations=[
                    ContextExpectation(
                        name="shell task completed",
                        query="check cli help",
                        forbidden_open_tasks=["Check CLI help"],
                    ),
                ],
            ),
            EvalScenario(
                name="file_operation_execution",
                description="Runs a bounded workspace file edit for a task and closes the task on success.",
                actions=[
                    EvalAction(
                        "seed_file",
                        {
                            "path": ".eval_tmp/eval_file_task.txt",
                            "text": "alpha\nbeta\n",
                        },
                    ),
                    EvalAction(
                        "task",
                        {
                            "title": "Update eval file text",
                            "status": "open",
                            "area": "execution",
                            "file_operation": "replace_text",
                            "file_path": ".eval_tmp/eval_file_task.txt",
                            "find_text": "beta",
                            "file_text": "gamma",
                            "complete_on_success": True,
                        },
                    ),
                    EvalAction(
                        "execute",
                        {
                            "query": "update eval file text",
                            "limit": 3,
                        },
                    ),
                ],
                search_expectations=[
                    SearchExpectation(
                        name="file outcome search",
                        query="file operation eval file",
                        required_contains=["Update eval file text"],
                    ),
                ],
                context_expectations=[
                    ContextExpectation(
                        name="file task completed",
                        query="update eval file text",
                        forbidden_open_tasks=["Update eval file text"],
                    ),
                ],
            ),
            EvalScenario(
                name="assistant_tool_outcome_extraction",
                description="Extracts useful tool outcomes from assistant progress messages.",
                actions=[
                    EvalAction(
                        "observe",
                        {
                            "role": "assistant",
                            "text": "Implemented local semantic reranking and tests passed.",
                        },
                    ),
                ],
                search_expectations=[
                    SearchExpectation(
                        name="assistant outcome search",
                        query="verification tests",
                        required_contains=["Tool outcome [success] tests"],
                    ),
                ],
            ),
        ]
