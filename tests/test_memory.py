from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import uuid
import zipfile
from unittest.mock import patch
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
from argparse import Namespace

import memory_agent.cli as cli_module
from memory_agent.cockpit import COCKPIT_HTML, CockpitService
from memory_agent.agent import MemoryFirstAgent
from memory_agent.cli import _resolve_patch_run_args, _resolve_serve_config, _run_pilot_chat, build_parser
from memory_agent.evaluation import (
    EvalCheckResult,
    EvalScenarioResult,
    EvalSuiteResult,
    MemoryEvaluator,
)
from memory_agent.executor import MemoryExecutor
from memory_agent.file_adapter import WorkspaceFileAdapter
from memory_agent.improvement import MemoryImprovementEngine, PilotHistoryReporter, PilotRunReviewer
from memory_agent.linux_runtime import LinuxPilotPolicy, LinuxPilotRuntime
from memory_agent.memory import MemoryStore
from memory_agent.migration import ProjectHandoffManager
from memory_agent.model_adapter import BaseModelAdapter, ModelMessage, ModelResponse, OllamaChatAdapter
from memory_agent.models import MemoryDraft
from memory_agent.patch_runner import PatchOperation, WorkspacePatchRunner
from memory_agent.planner import MemoryPlanner, PlannerAction
from memory_agent.reranker import OptionalSemanticReranker
from memory_agent.service_manager import CockpitServiceManager
from memory_agent.shell_adapter import GuardedShellAdapter


class MemoryStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = Path.cwd() / ".test_tmp"
        self.temp_root.mkdir(exist_ok=True)
        self.db_path = self.temp_root / f"{uuid.uuid4().hex}.sqlite3"
        self.extra_paths: list[Path] = []
        self.extra_dirs: list[Path] = []
        self.store = MemoryStore(self.db_path)

    def tearDown(self) -> None:
        self.store.close()
        for candidate in self.extra_paths:
            if candidate.exists():
                candidate.unlink()
        for candidate in reversed(self.extra_dirs):
            if candidate.exists():
                shutil.rmtree(candidate)
        for candidate in (
            self.db_path,
            Path(f"{self.db_path}-wal"),
            Path(f"{self.db_path}-shm"),
        ):
            if candidate.exists():
                candidate.unlink()

    def _record_green_baseline(self, *, score: float = 1.0) -> None:
        report = EvalSuiteResult(
            passed=score >= 1.0,
            score=score,
            scenario_results=[
                EvalScenarioResult(
                    name="baseline",
                    description="Baseline evaluation.",
                    passed=score >= 1.0,
                    score=score,
                    checks=[
                        EvalCheckResult(
                            name="baseline-check",
                            passed=score >= 1.0,
                            details="Baseline is available.",
                        )
                    ],
                )
            ],
        )
        self.store.record_evaluation_run("builtin", report)

    def _make_workspace(self) -> Path:
        workspace = self.temp_root / f"workspace_{uuid.uuid4().hex}"
        workspace.mkdir(parents=True, exist_ok=True)
        self.extra_dirs.append(workspace)
        return workspace

    def _fake_patch_shell_runner(self, argv, cwd, capture_output, text, timeout, shell):
        command_text = " ".join(str(part) for part in argv)
        if "evaluate" in command_text and "--json" in command_text:
            payload = {
                "passed": True,
                "score": 1.0,
                "scenario_results": [
                    {
                        "name": "preview",
                        "description": "preview validation",
                        "passed": True,
                        "score": 1.0,
                        "checks": [
                            {
                                "name": "preview-check",
                                "passed": True,
                                "details": "ok",
                            }
                        ],
                    }
                ],
            }
            return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")
        return subprocess.CompletedProcess(argv, 0, "ok\n", "")

    def _make_runtime_patch_runner(
        self,
        *,
        workspace_root: Path | None = None,
    ) -> WorkspacePatchRunner:
        return WorkspacePatchRunner(
            self.store,
            workspace_root=workspace_root or Path.cwd(),
            shell_runner=self._fake_patch_shell_runner,
        )

    def test_observe_extracts_local_runtime_and_priority(self) -> None:
        _, stored = self.store.observe(
            role="user",
            content=(
                "Lets make an agent from scratch. It should run locally on my main pc. "
                "My priority is implementing the memory system first."
            ),
        )
        contents = {item.content for item in stored}
        self.assertIn("Build the agent from scratch.", contents)
        self.assertIn("The agent should run locally on the user's main PC.", contents)
        self.assertIn("Implement the memory system first.", contents)

    def test_observe_extracts_tasks_and_decisions(self) -> None:
        _, stored = self.store.observe(
            role="user",
            content=(
                "Next step: add contradiction handling. "
                "Decision: use SQLite as the source of truth for memory storage."
            ),
        )
        kinds = {(item.kind, item.subject) for item in stored}
        self.assertIn(("task", "execution"), kinds)
        self.assertIn(("decision", "storage"), kinds)

    def test_assistant_observe_extracts_tool_outcome(self) -> None:
        _, stored = self.store.observe(
            role="assistant",
            content="Implemented local semantic reranking and tests passed.",
        )
        self.assertTrue(any(item.kind == "tool_outcome" for item in stored))

    def test_search_returns_relevant_memory(self) -> None:
        self.store.remember(
            MemoryDraft(
                kind="constraint",
                subject="runtime",
                content="The agent should run locally on the user's main PC.",
                tags=["runtime", "local"],
                importance=0.95,
                confidence=0.95,
            )
        )
        results = self.store.search("local main pc", limit=3)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(
            results[0].memory.content,
            "The agent should run locally on the user's main PC.",
        )

    def test_entity_aliases_bridge_database_and_retrieval_queries(self) -> None:
        self.store.observe(
            role="user",
            content="My priority is implementing the memory system first.",
        )
        sqlite_decision = self.store.record_decision(
            "storage",
            "Use SQLite as the source of truth for memory storage",
        )
        database_results = self.store.search("database choice", limit=3)
        self.assertGreaterEqual(len(database_results), 1)
        self.assertEqual(database_results[0].memory.id, sqlite_decision.id)
        self.assertTrue(
            any(reason.startswith("entities=") for reason in database_results[0].reasons)
        )

        retrieval_results = self.store.search("retrieval stack", limit=3)
        self.assertGreaterEqual(len(retrieval_results), 1)
        self.assertIn("memory system first", retrieval_results[0].memory.content.lower())
        resolved = self.store.resolve_entities("database choice for the retrieval stack")
        canonical_names = {entity.canonical_name for entity in resolved}
        self.assertIn("sqlite", canonical_names)
        self.assertIn("memory_system", canonical_names)

    def test_context_includes_recent_events(self) -> None:
        self.store.observe(role="user", content="We are building from scratch.")
        self.store.observe(role="user", content="Please optimize for low ongoing cost.")
        self.store.remember(
            MemoryDraft(
                kind="preference",
                subject="optimization",
                content="Optimize the agent for efficient, low-latency operation.",
                tags=["performance", "optimization"],
                importance=0.92,
                confidence=0.9,
            )
        )
        self.store.consolidate_recent()
        context = self.store.build_context("cost", memory_limit=3, recent_event_count=2)
        self.assertEqual(len(context.recent_events), 2)
        self.assertIn("Please optimize for low ongoing cost.", context.render())
        self.assertTrue(any(profile.memory.layer == "profile" for profile in context.profiles))
        self.assertTrue(any(bundle.evidence for bundle in context.bundles))

    def test_task_updates_supersede_previous_versions_and_resurface_open_loops(self) -> None:
        original = self.store.record_task(
            "Ship the memory benchmark harness",
            status="open",
            area="execution",
        )
        current = self.store.record_task(
            "Ship the memory benchmark harness",
            status="in_progress",
            area="execution",
        )
        done = self.store.record_task(
            "Archive outdated benchmark fixtures",
            status="done",
            area="execution",
        )
        original_after = self.store.get_memory(original.id)
        self.assertIsNotNone(original_after.archived_at)
        edges = self.store.get_memory_edges(current.id, direction="outgoing")
        self.assertTrue(
            any(edge.edge_type == "supersedes" and edge.to_memory_id == original.id for edge in edges)
        )
        open_tasks = self.store.get_open_tasks(limit=5)
        titles = [str(task.metadata.get("title")) for task in open_tasks]
        self.assertIn("Ship the memory benchmark harness", titles)
        self.assertNotIn("Archive outdated benchmark fixtures", titles)
        context = self.store.build_context("execution tasks", memory_limit=5, recent_event_count=1)
        context_titles = [str(task.metadata.get("title")) for task in context.open_tasks]
        self.assertIn("Ship the memory benchmark harness", context_titles)
        self.assertNotIn("Archive outdated benchmark fixtures", context_titles)

    def test_task_dependencies_blockers_and_due_dates_persist_across_updates(self) -> None:
        prerequisite = self.store.record_task(
            "Finish entity resolution",
            status="in_progress",
            area="execution",
            due_date="2026-04-01",
        )
        blocked = self.store.record_task(
            "Build task graph maintenance",
            status="blocked",
            area="execution",
            depends_on=["Finish entity resolution"],
            blocked_by=["Finish entity resolution"],
            due_date="2026-04-02",
        )
        blocked_current = self.store.record_task(
            "Build task graph maintenance",
            status="in_progress",
            area="execution",
        )
        self.assertEqual(blocked_current.metadata.get("due_date"), "2026-04-02")
        self.assertEqual(
            blocked_current.metadata.get("depends_on"),
            ["Finish entity resolution"],
        )
        self.assertEqual(
            blocked_current.metadata.get("blocked_by"),
            ["Finish entity resolution"],
        )
        context = self.store.build_context("blocked execution tasks", memory_limit=5, recent_event_count=1)
        surfaced = next(
            task for task in context.open_tasks if task.id == blocked_current.id
        )
        self.assertEqual(surfaced.metadata.get("due_date"), "2026-04-02")
        task_entity = next(
            link.entity
            for link in self.store.get_memory_entities(blocked_current.id)
            if link.entity.entity_type == "task"
        )
        prerequisite_entity = next(
            link.entity
            for link in self.store.get_memory_entities(prerequisite.id)
            if link.entity.entity_type == "task"
        )
        entity_edges = self.store.get_entity_edges(task_entity.id, direction="outgoing")
        self.assertTrue(
            any(
                edge.edge_type == "depends_on" and edge.to_entity_id == prerequisite_entity.id
                for edge in entity_edges
            )
        )
        self.assertTrue(
            any(
                edge.edge_type == "blocked_by" and edge.to_entity_id == prerequisite_entity.id
                for edge in entity_edges
            )
        )

    def test_ready_and_overdue_task_views_prioritize_actionable_work(self) -> None:
        overdue_ready = self.store.record_task(
            "Ship overdue ready memory loop",
            status="open",
            area="execution",
            due_date="2000-01-01",
        )
        future_ready = self.store.record_task(
            "Ship future ready memory loop",
            status="in_progress",
            area="execution",
            due_date="2999-01-01",
        )
        blocked_overdue = self.store.record_task(
            "Blocked on external API approval",
            status="blocked",
            area="execution",
            blocked_by=["Vendor approval"],
            due_date="2000-01-01",
        )
        ready_tasks = self.store.get_ready_tasks(limit=5)
        ready_titles = [str(task.metadata.get("title")) for task in ready_tasks]
        self.assertEqual(ready_titles[0], "Ship overdue ready memory loop")
        self.assertIn("Ship future ready memory loop", ready_titles)
        self.assertNotIn("Blocked on external API approval", ready_titles)

        overdue_tasks = self.store.get_overdue_tasks(limit=5)
        overdue_titles = [str(task.metadata.get("title")) for task in overdue_tasks]
        self.assertIn("Ship overdue ready memory loop", overdue_titles)
        self.assertIn("Blocked on external API approval", overdue_titles)

        open_tasks = self.store.get_open_tasks(limit=5)
        open_titles = [str(task.metadata.get("title")) for task in open_tasks]
        self.assertLess(
            open_titles.index("Ship future ready memory loop"),
            open_titles.index("Blocked on external API approval"),
        )

        context = self.store.build_context("ready overdue execution work", memory_limit=5, recent_event_count=1)
        ready_context_titles = [str(task.metadata.get("title")) for task in context.ready_tasks]
        overdue_context_titles = [str(task.metadata.get("title")) for task in context.overdue_tasks]
        self.assertIn("Ship overdue ready memory loop", ready_context_titles)
        self.assertIn("Ship future ready memory loop", ready_context_titles)
        self.assertNotIn("Blocked on external API approval", ready_context_titles)
        self.assertIn("Blocked on external API approval", overdue_context_titles)
        rendered = context.render()
        self.assertIn("Ready now:", rendered)
        self.assertIn("Overdue:", rendered)
        self.assertIn("overdue", rendered.lower())

    def test_review_tasks_generates_nudges_for_overdue_and_stale_work(self) -> None:
        overdue_ready = self.store.record_task(
            "Ship overdue ready memory loop",
            status="open",
            area="execution",
            due_date="2000-01-01",
        )
        stale_blocked = self.store.record_task(
            "Unblock vendor approval",
            status="blocked",
            area="execution",
            blocked_by=["Vendor approval"],
            due_date="2999-01-01",
        )
        backdated = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        self.store.connection.execute(
            "update memories set updated_at = ? where id = ?",
            (backdated, stale_blocked.id),
        )
        self.store.connection.commit()

        nudges = self.store.review_tasks(limit=5)
        self.assertEqual(len(nudges), 2)
        nudge_contents = [nudge.content for nudge in nudges]
        self.assertTrue(any("overdue and ready to work on now" in content for content in nudge_contents))
        self.assertTrue(any("has been blocked for" in content for content in nudge_contents))
        self.assertTrue(all(nudge.kind == "nudge" for nudge in nudges))

        search_results = self.store.search("overdue nudge", limit=5)
        self.assertTrue(any(result.memory.kind == "nudge" for result in search_results))

        overdue_nudge = next(
            nudge for nudge in nudges if "Ship overdue ready memory loop" in nudge.content
        )
        edges = self.store.get_memory_edges(overdue_nudge.id, direction="outgoing")
        self.assertTrue(
            any(edge.edge_type == "nudges" and edge.to_memory_id == overdue_ready.id for edge in edges)
        )

        second_pass = self.store.review_tasks(limit=5)
        self.assertEqual(second_pass, [])

    def test_recurring_tasks_roll_forward_and_respect_snooze_resume(self) -> None:
        recurring = self.store.record_task(
            "Weekly memory audit",
            status="open",
            area="execution",
            due_date="2026-04-01",
            recurrence_days=7,
        )
        completion = self.store.complete_task("Weekly memory audit", area="execution")
        completed = completion["completed"]
        next_occurrence = completion["next_occurrence"]
        self.assertIsNotNone(completed)
        self.assertIsNotNone(next_occurrence)
        self.assertEqual(next_occurrence.metadata.get("due_date"), "2026-04-08")
        self.assertEqual(next_occurrence.metadata.get("recurrence_days"), 7)
        recurrence_edges = self.store.get_memory_edges(completed.id, direction="outgoing")
        self.assertTrue(
            any(edge.edge_type == "recurs_to" and edge.to_memory_id == next_occurrence.id for edge in recurrence_edges)
        )

        snoozed = self.store.snooze_task(
            "Weekly memory audit",
            until="2999-01-01",
            area="execution",
        )
        self.assertEqual(snoozed.metadata.get("snoozed_until"), "2999-01-01")
        ready_titles = [str(task.metadata.get("title")) for task in self.store.get_ready_tasks(limit=5)]
        self.assertNotIn("Weekly memory audit", ready_titles)

        resumed = self.store.resume_task("Weekly memory audit", area="execution")
        self.assertIsNone(resumed.metadata.get("snoozed_until"))
        ready_titles = [str(task.metadata.get("title")) for task in self.store.get_ready_tasks(limit=5)]
        self.assertIn("Weekly memory audit", ready_titles)

    def test_unblock_task_clears_blockers_and_enables_ready_state(self) -> None:
        blocked = self.store.record_task(
            "Finalize vendor migration",
            status="blocked",
            area="execution",
            blocked_by=["Vendor approval"],
        )
        unblocked = self.store.unblock_task("Finalize vendor migration", area="execution")
        self.assertEqual(unblocked.metadata.get("blocked_by"), [])
        self.assertEqual(unblocked.metadata.get("status"), "open")
        ready_titles = [str(task.metadata.get("title")) for task in self.store.get_ready_tasks(limit=5)]
        self.assertIn("Finalize vendor migration", ready_titles)

    def test_review_tasks_escalates_after_repeated_nudges(self) -> None:
        blocked = self.store.record_task(
            "Escalate vendor access",
            status="blocked",
            area="execution",
            blocked_by=["Vendor approval"],
            due_date="2000-01-01",
        )
        first_nudges = self.store.review_tasks(limit=5)
        self.assertTrue(any("Escalate vendor access" in nudge.content for nudge in first_nudges))
        first_nudge = next(nudge for nudge in first_nudges if "Escalate vendor access" in nudge.content)
        backdated = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        self.store.connection.execute(
            "update memories set updated_at = ? where id = ?",
            (backdated, first_nudge.id),
        )
        self.store.connection.commit()

        second_nudges = self.store.review_tasks(limit=5)
        self.assertEqual(len(second_nudges), 1)
        self.assertIn("Escalation:", second_nudges[0].content)
        self.assertIn("Escalate vendor access", second_nudges[0].content)

    def test_planner_prioritizes_escalated_blockers_over_ready_work(self) -> None:
        self.store.record_task(
            "Ship overdue ready memory loop",
            status="open",
            area="execution",
            due_date="2000-01-01",
        )
        self.store.record_task(
            "Escalate vendor access",
            status="blocked",
            area="execution",
            blocked_by=["Vendor approval"],
            due_date="2000-01-01",
        )
        first_nudges = self.store.review_tasks(limit=5)
        first_nudge = next(nudge for nudge in first_nudges if "Escalate vendor access" in nudge.content)
        backdated = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        self.store.connection.execute(
            "update memories set updated_at = ? where id = ?",
            (backdated, first_nudge.id),
        )
        self.store.connection.commit()
        self.store.review_tasks(limit=5)

        planner = MemoryPlanner(self.store)
        snapshot = planner.build_plan("what should I do next", action_limit=3)
        self.assertIsNotNone(snapshot.recommendation)
        self.assertEqual(snapshot.recommendation.kind, "resolve_blocker")
        self.assertEqual(snapshot.recommendation.title, "Escalate vendor access")
        self.assertTrue(
            any(reason == "nudge=escalation" for reason in snapshot.recommendation.reasons)
        )
        alternative_titles = [action.title for action in snapshot.alternatives]
        self.assertIn("Ship overdue ready memory loop", alternative_titles)

    def test_planner_recommends_maintenance_when_no_active_execution_exists(self) -> None:
        self.store.remember(
            MemoryDraft(
                kind="preference",
                subject="optimization",
                content="Optimize the agent for low ongoing cost.",
                tags=["cost", "optimization"],
                importance=0.95,
                confidence=0.9,
            )
        )
        self.store.remember(
            MemoryDraft(
                kind="preference",
                subject="optimization",
                content="Optimize the agent for efficient, low-latency operation.",
                tags=["performance", "optimization"],
                importance=0.92,
                confidence=0.9,
            )
        )

        planner = MemoryPlanner(self.store)
        snapshot = planner.build_plan("memory maintenance", action_limit=3)
        self.assertIsNotNone(snapshot.recommendation)
        self.assertEqual(snapshot.recommendation.kind, "run_maintenance")
        self.assertIn("maintenance_due=", " ".join(snapshot.recommendation.reasons))

    def test_service_sync_status_marks_missing_recommended_service_tasks_due(self) -> None:
        status = self.store.service_sync_status(
            {
                "onboarding": {
                    "actions": [
                        {
                            "action": "install_local_service",
                            "label": "Install local service",
                            "enabled": True,
                        }
                    ]
                }
            }
        )

        self.assertTrue(status["due"])
        self.assertEqual(status["recommended_actions"], ["install_local_service"])
        self.assertEqual(
            status["missing_titles"],
            ["Cockpit setup: Install local service"],
        )

    def test_planner_recommends_maintenance_when_service_sync_is_due(self) -> None:
        class FakeServiceManager:
            def settings(self) -> dict[str, object]:
                return {
                    "onboarding": {
                        "actions": [
                            {
                                "action": "install_remote_service",
                                "label": "Install remote service",
                                "description": "Install or repair remote access.",
                                "enabled": True,
                            }
                        ]
                    }
                }

        planner = MemoryPlanner(self.store, service_manager=FakeServiceManager())
        snapshot = planner.build_plan("what should I do next", action_limit=3)

        self.assertIsNotNone(snapshot.recommendation)
        self.assertEqual(snapshot.recommendation.kind, "run_maintenance")
        self.assertIn("maintenance_due=service_sync", " ".join(snapshot.recommendation.reasons))
        self.assertIn("service_sync_recommended=1", snapshot.recommendation.reasons)

    def test_planner_suppresses_service_sync_after_recent_healthy_verification(self) -> None:
        class FakeServiceManager:
            def settings(self) -> dict[str, object]:
                return {
                    "onboarding": {
                        "actions": [
                            {
                                "action": "install_remote_service",
                                "label": "Install remote service",
                                "description": "Install or repair remote access.",
                                "enabled": True,
                            }
                        ]
                    }
                }

        self.store.record_tool_outcome(
            "service_manager",
            "Healthy remote verification just completed.",
            status="success",
            subject="execution",
            tags=["executor", "service_inspection", "success"],
            metadata={
                "service_inspection": "restart_remote_service",
                "service_inspection_healthy": True,
                "resolved_service_sync_titles": ["Cockpit setup: Install remote service"],
            },
        )

        planner = MemoryPlanner(self.store, service_manager=FakeServiceManager())
        snapshot = planner.build_plan("what should I do next", action_limit=3)

        self.assertFalse(snapshot.maintenance["service_sync"]["due"])
        self.assertEqual(snapshot.maintenance["service_sync"]["recommended_actions"], [])
        self.assertEqual(
            snapshot.maintenance["service_sync"]["suppressed_recent_verification_actions"],
            ["install_remote_service"],
        )
        self.assertEqual(
            snapshot.maintenance["service_sync"]["suppressed_recent_verification_titles"],
            ["Cockpit setup: Install remote service"],
        )
        self.assertIsNotNone(
            snapshot.maintenance["service_sync"]["suppressed_recent_verification_updated_at"]
        )
        self.assertGreaterEqual(
            snapshot.maintenance["service_sync"]["suppressed_recent_verification_age_seconds"],
            0,
        )
        self.assertGreater(
            snapshot.maintenance["service_sync"]["suppressed_recent_verification_expires_in_seconds"],
            0,
        )
        self.assertTrue(
            snapshot.recommendation is None
            or snapshot.recommendation.kind != "run_maintenance"
        )

    def test_planner_uses_scope_specific_service_sync_suppression_windows(self) -> None:
        class FakeServiceManager:
            def settings(self) -> dict[str, object]:
                return {
                    "onboarding": {
                        "actions": [
                            {
                                "action": "install_local_service",
                                "label": "Install local service",
                                "description": "Install the local cockpit service.",
                                "enabled": True,
                            },
                            {
                                "action": "install_remote_service",
                                "label": "Install remote service",
                                "description": "Install or repair remote access.",
                                "enabled": True,
                            },
                        ]
                    }
                }

        local_verification = self.store.record_tool_outcome(
            "service_manager",
            "Healthy local verification completed.",
            status="success",
            subject="execution",
            tags=["executor", "service_inspection", "success"],
            metadata={
                "service_inspection": "restart_local_service",
                "service_inspection_healthy": True,
                "resolved_service_sync_titles": ["Cockpit setup: Install local service"],
            },
        )
        remote_verification = self.store.record_tool_outcome(
            "service_manager",
            "Healthy remote verification completed.",
            status="success",
            subject="execution",
            tags=["executor", "service_inspection", "success"],
            metadata={
                "service_inspection": "restart_remote_service",
                "service_inspection_healthy": True,
                "resolved_service_sync_titles": ["Cockpit setup: Install remote service"],
            },
        )
        backdated = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
        self.store.connection.execute(
            "update memories set updated_at = ? where id in (?, ?)",
            (backdated, local_verification.id, remote_verification.id),
        )
        self.store.connection.commit()

        planner = MemoryPlanner(self.store, service_manager=FakeServiceManager())
        status = planner.service_sync_status()

        self.assertTrue(status["due"])
        self.assertEqual(status["recommended_actions"], ["install_local_service"])
        self.assertEqual(status["missing_titles"], ["Cockpit setup: Install local service"])
        self.assertEqual(status["suppressed_recent_verification_scopes"], ["remote_service"])
        self.assertEqual(
            status["suppressed_recent_verification_actions"],
            ["install_remote_service"],
        )
        self.assertEqual(
            status["suppressed_recent_verification_titles"],
            ["Cockpit setup: Install remote service"],
        )
        self.assertGreater(status["suppressed_recent_verification_age_seconds"], 0)
        self.assertGreater(
            status["suppressed_recent_verification_expires_in_seconds"],
            0,
        )

    def test_planner_routes_confirmation_heavy_service_tasks_to_preparation(self) -> None:
        self.store.record_task(
            "Restart local cockpit service",
            status="open",
            area="execution",
            details=(
                "Confirm reconnect timing.\n"
                "Latest prep inspection: status=inactive, active=no. Verification target: http://127.0.0.1:8765/"
            ),
            service_action="restart_local_service",
            service_label="Restart local service",
            complete_on_success=True,
        )
        self.store.record_task(
            "Restart remote cockpit service",
            status="open",
            area="execution",
            service_action="restart_remote_service",
            service_label="Restart remote service",
            service_requires_confirmation=True,
            service_confirmation_message="Restart remote access for this machine?",
            complete_on_success=True,
        )

        planner = MemoryPlanner(self.store)
        snapshot = planner.build_plan("restart service", action_limit=3)

        self.assertIsNotNone(snapshot.recommendation)
        self.assertEqual(snapshot.recommendation.kind, "prepare_task")
        self.assertEqual(
            snapshot.recommendation.title,
            "Prepare safer execution for Restart remote cockpit service",
        )
        self.assertIn("service_confirmation_required", snapshot.recommendation.reasons)
        local_action = next(
            action
            for action in snapshot.alternatives
            if action.title == "Restart local cockpit service"
        )
        self.assertIn("service_low_friction", local_action.reasons)

    def test_planner_uses_pilot_history_to_prefer_safe_ready_work(self) -> None:
        for index in range(2):
            self.store.record_tool_outcome(
                "pilot-review",
                f"Pilot review {index} stopped on approval friction",
                status="blocked",
                subject="self_improvement",
                tags=["pilot", "review", "self-improvement"],
                metadata={
                    "goal_text": f"pilot history seed {index}",
                    "stop_reason": "needs_approval",
                    "executed_steps": 0,
                    "approval_requests": 1,
                    "approvals_granted": 0,
                    "opportunity_count": 1,
                    "opportunity_categories": ["approval_friction"],
                    "recurring_patterns": [],
                },
            )

        self.store.record_task(
            "Pilot safe read",
            status="open",
            area="execution",
            file_operation="read_text",
            file_path=".test_tmp/pilot_safe_read.txt",
            complete_on_success=True,
        )
        self.store.record_task(
            "Pilot risky write",
            status="open",
            area="execution",
            file_operation="write_text",
            file_path=".test_tmp/pilot_risky_write.txt",
            file_text="updated\n",
            complete_on_success=True,
        )

        planner = MemoryPlanner(self.store)
        snapshot = planner.build_plan("what should I do next", action_limit=3)

        self.assertIsNotNone(snapshot.recommendation)
        self.assertEqual(snapshot.recommendation.title, "Pilot safe read")
        self.assertEqual(snapshot.pilot_history.get("approval_friction_count"), 2)
        self.assertTrue(
            any(
                reason.startswith("pilot_history_prefers_safe_execution")
                for reason in snapshot.recommendation.reasons
            )
        )
        risky = next(action for action in snapshot.alternatives if action.title == "Pilot risky write")
        self.assertTrue(
            any(
                reason.startswith("pilot_history=approval_friction")
                for reason in risky.reasons
            )
        )

    def test_planner_can_propose_preparation_for_risky_ready_work(self) -> None:
        for index in range(2):
            self.store.record_tool_outcome(
                "pilot-review",
                f"Pilot review {index} stopped on approval friction",
                status="blocked",
                subject="self_improvement",
                tags=["pilot", "review", "self-improvement"],
                metadata={
                    "goal_text": f"pilot history prep seed {index}",
                    "stop_reason": "needs_approval",
                    "executed_steps": 0,
                    "approval_requests": 1,
                    "approvals_granted": 0,
                    "opportunity_count": 1,
                    "opportunity_categories": ["approval_friction"],
                    "recurring_patterns": [],
                },
            )

        self.store.record_task(
            "Pilot risky write",
            status="open",
            area="execution",
            file_operation="write_text",
            file_path=".test_tmp/pilot_risky_write.txt",
            file_text="updated\n",
            complete_on_success=True,
        )

        planner = MemoryPlanner(self.store)
        snapshot = planner.build_plan("what should I do next", action_limit=3)

        self.assertIsNotNone(snapshot.recommendation)
        self.assertEqual(snapshot.recommendation.kind, "prepare_task")
        self.assertEqual(
            snapshot.recommendation.title,
            "Prepare safer execution for Pilot risky write",
        )
        self.assertEqual(
            snapshot.recommendation.metadata.get("target_task_title"),
            "Pilot risky write",
        )
        self.assertTrue(
            any(
                reason.startswith("pilot_history=approval_friction")
                for reason in snapshot.recommendation.reasons
            )
        )

    def test_planner_can_propose_preparation_for_confirmation_heavy_service_task(self) -> None:
        self.store.record_task(
            "Restart remote cockpit service",
            status="open",
            area="execution",
            service_action="restart_remote_service",
            service_label="Restart remote service",
            service_requires_confirmation=True,
            service_confirmation_message=(
                "Restart remote access for this machine? Existing remote browser sessions may need to reconnect afterward."
            ),
            complete_on_success=True,
        )

        planner = MemoryPlanner(self.store)
        snapshot = planner.build_plan("what should I do next", action_limit=3)

        self.assertIsNotNone(snapshot.recommendation)
        self.assertEqual(snapshot.recommendation.kind, "prepare_task")
        self.assertEqual(
            snapshot.recommendation.title,
            "Prepare safer execution for Restart remote cockpit service",
        )
        self.assertIn("service_confirmation_required", snapshot.recommendation.reasons)
        prep_details = str(snapshot.recommendation.metadata.get("prep_task_details") or "")
        self.assertIn("1. Inspect current service state", prep_details)
        self.assertIn("2. Capture the approval-sensitive change", prep_details)
        self.assertIn("3. Define the post-action verification", prep_details)
        self.assertIn("Restart remote access for this machine?", prep_details)

    def test_planner_can_propose_safe_ready_batch_when_query_requests_it(self) -> None:
        first = self.temp_root / f"{uuid.uuid4().hex}_planner_batch_first.txt"
        second = self.temp_root / f"{uuid.uuid4().hex}_planner_batch_second.txt"
        self.extra_paths.extend([first, second])
        first.write_text("one\n", encoding="utf-8")
        second.write_text("two\n", encoding="utf-8")

        self.store.record_task(
            "Batch read first file",
            status="open",
            area="execution",
            file_operation="read_text",
            file_path=str(first.relative_to(Path.cwd())),
            complete_on_success=True,
        )
        self.store.record_task(
            "Batch read second file",
            status="open",
            area="execution",
            file_operation="read_text",
            file_path=str(second.relative_to(Path.cwd())),
            complete_on_success=True,
        )

        planner = MemoryPlanner(self.store)
        snapshot = planner.build_plan("batch the safe ready tasks", action_limit=3)

        self.assertIsNotNone(snapshot.recommendation)
        self.assertEqual(snapshot.recommendation.kind, "batch_ready_tasks")
        self.assertEqual(
            snapshot.recommendation.metadata.get("task_titles"),
            ["Batch read first file", "Batch read second file"],
        )
        self.assertIn("batch_requested", snapshot.recommendation.reasons)

    def test_planner_prioritizes_post_action_service_verification(self) -> None:
        self.store.record_task(
            "Routine safe read",
            status="open",
            area="execution",
            file_operation="read_text",
            file_path=".test_tmp/routine_safe_read.txt",
            complete_on_success=True,
        )
        self.store.record_task(
            "Verify Restart local cockpit service",
            status="open",
            area="execution",
            details=(
                "Confirm the post-action state after 'Restart local service'. "
                "Verification target: http://127.0.0.1:8765/."
            ),
            service_inspection="restart_local_service",
            service_label="Restart local service",
            complete_on_success=True,
            tags=["service-verification", "post-action"],
        )

        planner = MemoryPlanner(self.store)
        snapshot = planner.build_plan("what should I do next", action_limit=3)

        self.assertIsNotNone(snapshot.recommendation)
        self.assertEqual(snapshot.recommendation.title, "Verify Restart local cockpit service")
        self.assertIn("post_action_verification", snapshot.recommendation.reasons)
        self.assertIn("verification_follow_up", snapshot.recommendation.reasons)

    def test_planner_can_propose_delegation_when_query_requests_it(self) -> None:
        self.store.record_task(
            "Delegate quarterly report",
            status="open",
            area="execution",
        )

        planner = MemoryPlanner(self.store)
        snapshot = planner.build_plan("delegate the quarterly report task", action_limit=3)

        self.assertIsNotNone(snapshot.recommendation)
        self.assertEqual(snapshot.recommendation.kind, "delegate_task")
        self.assertEqual(
            snapshot.recommendation.metadata.get("target_task_title"),
            "Delegate quarterly report",
        )
        self.assertIn("delegation_requested", snapshot.recommendation.reasons)

    def test_executor_runs_safe_ready_batch(self) -> None:
        first = self.temp_root / f"{uuid.uuid4().hex}_executor_batch_first.txt"
        second = self.temp_root / f"{uuid.uuid4().hex}_executor_batch_second.txt"
        self.extra_paths.extend([first, second])
        first.write_text("one\n", encoding="utf-8")
        second.write_text("two\n", encoding="utf-8")

        self.store.record_task(
            "Batch read first file",
            status="open",
            area="execution",
            file_operation="read_text",
            file_path=str(first.relative_to(Path.cwd())),
            complete_on_success=True,
        )
        self.store.record_task(
            "Batch read second file",
            status="open",
            area="execution",
            file_operation="read_text",
            file_path=str(second.relative_to(Path.cwd())),
            complete_on_success=True,
        )

        executor = MemoryExecutor(self.store)
        cycle = executor.execute_next("batch the safe ready tasks", action_limit=3)

        self.assertEqual(cycle.result.executed_kind, "batch_ready_tasks")
        self.assertEqual(cycle.result.status, "success")
        batch_results = cycle.result.metadata.get("batch_results") or []
        self.assertEqual(len(batch_results), 2)
        self.assertEqual(
            [item.get("title") for item in batch_results],
            ["Batch read first file", "Batch read second file"],
        )
        ready_titles = [
            task.metadata.get("title") for task in self.store.get_ready_tasks(limit=5)
        ]
        self.assertNotIn("Batch read first file", ready_titles)
        self.assertNotIn("Batch read second file", ready_titles)

    def test_executor_creates_delegated_child_task(self) -> None:
        self.store.record_task(
            "Delegate quarterly report",
            status="open",
            area="execution",
        )

        executor = MemoryExecutor(self.store)
        cycle = executor.execute_next("delegate the quarterly report task", action_limit=3)

        self.assertEqual(cycle.result.executed_kind, "delegate_task")
        self.assertEqual(cycle.result.status, "success")
        self.assertIsNotNone(cycle.result.related_task)
        self.assertEqual(
            cycle.result.related_task.metadata.get("title"),
            "Delegate work for Delegate quarterly report",
        )
        self.assertEqual(cycle.result.related_task.metadata.get("owner"), "delegate")
        self.assertEqual(cycle.result.task_update.metadata.get("status"), "blocked")
        self.assertIn(
            "Delegate work for Delegate quarterly report",
            cycle.result.task_update.metadata.get("blocked_by", []),
        )

    def test_executor_starts_ready_task_and_logs_outcome(self) -> None:
        self.store.record_task(
            "Ship overdue ready memory loop",
            status="open",
            area="execution",
            due_date="2000-01-01",
        )
        executor = MemoryExecutor(self.store)
        cycle = executor.execute_next("what should I do next", action_limit=3)

        self.assertEqual(cycle.result.executed_kind, "work_task")
        self.assertEqual(cycle.result.status, "success")
        self.assertEqual(cycle.result.task_update.metadata.get("status"), "in_progress")
        self.assertIsNotNone(cycle.result.tool_outcome)
        self.assertIn("Started task 'Ship overdue ready memory loop'", cycle.result.tool_outcome.content)

    def test_executor_reroutes_blocked_task_to_ready_dependency(self) -> None:
        self.store.record_task(
            "Finish entity resolution",
            status="open",
            area="execution",
        )
        self.store.record_task(
            "Build task graph maintenance",
            status="blocked",
            area="execution",
            blocked_by=["Finish entity resolution"],
            due_date="2000-01-01",
        )

        executor = MemoryExecutor(self.store)
        cycle = executor.execute_next("what should I do next", action_limit=3)

        self.assertEqual(cycle.result.executed_kind, "resolve_blocker")
        self.assertEqual(cycle.result.status, "success")
        self.assertIsNotNone(cycle.result.task_update)
        self.assertEqual(cycle.result.task_update.metadata.get("title"), "Finish entity resolution")
        self.assertEqual(cycle.result.task_update.metadata.get("status"), "in_progress")
        self.assertIn("Finish entity resolution", cycle.result.summary)

    def test_executor_prepares_safer_step_for_risky_task_after_pilot_friction(self) -> None:
        for index in range(2):
            self.store.record_tool_outcome(
                "pilot-review",
                f"Pilot review {index} stopped on approval friction",
                status="blocked",
                subject="self_improvement",
                tags=["pilot", "review", "self-improvement"],
                metadata={
                    "goal_text": f"pilot history executor seed {index}",
                    "stop_reason": "needs_approval",
                    "executed_steps": 0,
                    "approval_requests": 1,
                    "approvals_granted": 0,
                    "opportunity_count": 1,
                    "opportunity_categories": ["approval_friction"],
                    "recurring_patterns": [],
                },
            )

        self.store.record_task(
            "Pilot risky write",
            status="open",
            area="execution",
            file_operation="write_text",
            file_path=".test_tmp/pilot_risky_write_executor.txt",
            file_text="updated\n",
            complete_on_success=True,
        )

        executor = MemoryExecutor(self.store)
        cycle = executor.execute_next("what should I do next", action_limit=3)

        self.assertEqual(cycle.result.executed_kind, "prepare_task")
        self.assertEqual(cycle.result.status, "success")
        self.assertIsNotNone(cycle.result.related_task)
        self.assertEqual(
            cycle.result.related_task.metadata.get("title"),
            "Prepare safer execution for Pilot risky write",
        )
        self.assertEqual(cycle.result.related_task.metadata.get("status"), "open")
        self.assertEqual(cycle.result.task_update.metadata.get("status"), "blocked")
        self.assertIn(
            "Prepare safer execution for Pilot risky write",
            cycle.result.task_update.metadata.get("blocked_by", []),
        )
        self.assertIsNotNone(cycle.after_plan.recommendation)
        self.assertEqual(cycle.after_plan.recommendation.kind, "work_task")
        self.assertEqual(
            cycle.after_plan.recommendation.title,
            "Prepare safer execution for Pilot risky write",
        )

    def test_agent_execute_next_replans_after_running_executor(self) -> None:
        self.store.record_task(
            "Ship overdue ready memory loop",
            status="open",
            area="execution",
            due_date="2000-01-01",
        )
        agent = MemoryFirstAgent(self.store)

        cycle = agent.execute_next("what should I do next", action_limit=3)

        self.assertEqual(cycle.before_plan.recommendation.title, "Ship overdue ready memory loop")
        self.assertEqual(cycle.result.executed_kind, "work_task")
        self.assertIsNotNone(cycle.after_plan.recommendation)
        self.assertEqual(cycle.after_plan.recommendation.title, "Ship overdue ready memory loop")

    def test_executor_runs_safe_shell_command_and_completes_task(self) -> None:
        self.store.record_task(
            "Check CLI help",
            status="open",
            area="execution",
            command="python -m memory_agent.cli --help",
            complete_on_success=True,
        )

        def fake_runner(argv, cwd, capture_output, text, timeout, shell):
            self.assertEqual(argv[:3], ["python", "-m", "memory_agent.cli"])
            self.assertFalse(shell)
            return subprocess.CompletedProcess(argv, 0, "usage: cli.py\n", "")

        executor = MemoryExecutor(
            self.store,
            shell_adapter=GuardedShellAdapter(
                workspace_root=Path.cwd(),
                runner=fake_runner,
            ),
        )
        cycle = executor.execute_next("check cli help", action_limit=3)

        self.assertEqual(cycle.result.executed_kind, "run_shell")
        self.assertEqual(cycle.result.status, "success")
        self.assertIsNotNone(cycle.result.shell_result)
        self.assertEqual(cycle.result.task_update.metadata.get("status"), "done")
        self.assertIn("Ran shell command for 'Check CLI help'", cycle.result.tool_outcome.content)
        open_titles = [str(task.metadata.get("title")) for task in self.store.get_open_tasks(limit=5)]
        self.assertNotIn("Check CLI help", open_titles)

    def test_shell_adapter_blocks_disallowed_command_prefix(self) -> None:
        self.store.record_task(
            "Run arbitrary python one-liner",
            status="open",
            area="execution",
            command="python -c print('hi')",
        )
        executor = MemoryExecutor(self.store)
        cycle = executor.execute_next("run arbitrary python one-liner", action_limit=3)

        self.assertEqual(cycle.result.executed_kind, "run_shell")
        self.assertEqual(cycle.result.status, "blocked")
        self.assertIsNotNone(cycle.result.shell_result)
        self.assertEqual(cycle.result.shell_result.reason, "command_prefix_not_allowed")
        self.assertIn("blocked by shell policy", cycle.result.summary.lower())

    def test_executor_runs_workspace_file_replace_and_completes_task(self) -> None:
        target = self.temp_root / f"{uuid.uuid4().hex}.txt"
        self.extra_paths.append(target)
        target.write_text("alpha\nbeta\n", encoding="utf-8")
        self.store.record_task(
            "Update local fixture",
            status="open",
            area="execution",
            file_operation="replace_text",
            file_path=str(target.relative_to(Path.cwd())),
            find_text="beta",
            file_text="gamma",
            complete_on_success=True,
        )

        executor = MemoryExecutor(
            self.store,
            file_adapter=WorkspaceFileAdapter(workspace_root=Path.cwd()),
        )
        cycle = executor.execute_next("update local fixture", action_limit=3)

        self.assertEqual(cycle.result.executed_kind, "run_file_operation")
        self.assertEqual(cycle.result.status, "success")
        self.assertIsNotNone(cycle.result.file_result)
        self.assertEqual(cycle.result.task_update.metadata.get("status"), "done")
        self.assertEqual(target.read_text(encoding="utf-8"), "alpha\ngamma\n")
        self.assertIn("Ran file operation for 'Update local fixture'", cycle.result.tool_outcome.content)

    def test_executor_runs_service_action_and_completes_task(self) -> None:
        self.store.record_task(
            "Verify Install local cockpit service",
            status="open",
            area="execution",
            details="Confirm old local service install state.",
            service_inspection="install_local_service",
            service_label="Install local service",
            complete_on_success=True,
            tags=["service-verification", "post-action"],
        )
        self.store.record_task(
            "Restart local cockpit service",
            status="open",
            area="execution",
            service_action="restart_local_service",
            service_label="Restart local service",
            complete_on_success=True,
        )

        class FakeServiceManager:
            def perform_action(self, action: str) -> dict[str, object]:
                self.last_action = action
                return {
                    "action": action,
                    "message": "Local cockpit service restarted.",
                    "result": {"ok": True},
                    "verification_target": "http://127.0.0.1:8765/",
                    "settings": {"local_service": {"active": True}},
                }

        service_manager = FakeServiceManager()
        executor = MemoryExecutor(self.store, service_manager=service_manager)
        cycle = executor.execute_action(
            PlannerAction(
                kind="work_task",
                title="Restart local cockpit service",
                summary="Run the local service restart task.",
                score=0.9,
                metadata={"area": "execution"},
            )
        )

        self.assertEqual(service_manager.last_action, "restart_local_service")
        self.assertEqual(cycle.executed_kind, "run_service_action")
        self.assertEqual(cycle.status, "success")
        self.assertEqual(cycle.task_update.metadata.get("status"), "done")
        self.assertEqual(cycle.metadata.get("service_action"), "restart_local_service")
        self.assertEqual(
            cycle.metadata.get("verification_task_title"),
            "Verify Restart local cockpit service",
        )
        self.assertIsNotNone(cycle.related_task)
        self.assertEqual(
            cycle.related_task.metadata.get("title"),
            "Verify Restart local cockpit service",
        )
        self.assertEqual(
            cycle.related_task.metadata.get("service_inspection"),
            "restart_local_service",
        )
        self.assertTrue(cycle.related_task.metadata.get("complete_on_success"))
        self.assertIn(
            "Superseded older verification task(s): Verify Install local cockpit service.",
            str(cycle.related_task.metadata.get("details") or ""),
        )
        self.assertIn(
            "Verification target: http://127.0.0.1:8765/",
            str(cycle.related_task.metadata.get("details") or ""),
        )
        superseded = next(
            task
            for task in self.store._active_task_memories()
            if task.subject == "execution"
            and str(task.metadata.get("title") or "") == "Verify Install local cockpit service"
            and str(task.metadata.get("status") or "") == "done"
        )
        self.assertIsNotNone(superseded)
        self.assertEqual(superseded.metadata.get("status"), "done")
        self.assertIn(
            "Superseded by 'Verify Restart local cockpit service'.",
            str(superseded.metadata.get("details") or ""),
        )
        self.assertIn(
            "Ran service action 'restart_local_service' for 'Restart local cockpit service'",
            cycle.tool_outcome.content,
        )

    def test_executor_runs_service_inspection_and_completes_task(self) -> None:
        self.store.record_task(
            "Restart remote cockpit service",
            status="blocked",
            area="execution",
            details=(
                "Capture remote reconnect plan.\n"
                "Latest prep inspection: status=active, active=yes. Verification target: http://old.example/"
            ),
            service_action="restart_remote_service",
            service_label="Restart remote service",
            service_requires_confirmation=True,
            blocked_by=["Prepare safer execution for Restart remote cockpit service"],
            complete_on_success=True,
        )
        self.store.record_task(
            "Prepare safer execution for Restart remote cockpit service",
            status="open",
            area="execution",
            service_inspection="restart_remote_service",
            service_label="Restart remote service",
            complete_on_success=True,
        )

        class FakeServiceManager:
            def inspect_action(self, action: str) -> dict[str, object]:
                self.last_action = action
                return {
                    "action": action,
                    "inspection": {"active": False, "status": "inactive"},
                    "verification_target": "http://100.1.2.3:8766/",
                    "settings": {},
                }

        service_manager = FakeServiceManager()
        executor = MemoryExecutor(self.store, service_manager=service_manager)
        cycle = executor.execute_next("inspect remote cockpit service", action_limit=3)

        self.assertEqual(service_manager.last_action, "restart_remote_service")
        self.assertEqual(cycle.result.executed_kind, "run_service_inspection")
        self.assertEqual(cycle.result.status, "success")
        self.assertEqual(cycle.result.task_update.metadata.get("status"), "done")
        self.assertIsNotNone(cycle.result.related_task)
        self.assertEqual(
            cycle.result.related_task.metadata.get("title"),
            "Restart remote cockpit service",
        )
        self.assertEqual(cycle.result.related_task.metadata.get("status"), "open")
        self.assertFalse(cycle.result.related_task.metadata.get("blocked_now"))
        parent_details = str(cycle.result.related_task.metadata.get("details") or "")
        self.assertIn("Capture remote reconnect plan.", parent_details)
        self.assertIn(
            "Latest prep inspection: status=inactive, active=no. Verification target: http://100.1.2.3:8766/",
            parent_details,
        )
        self.assertNotIn("http://old.example/", parent_details)
        self.assertEqual(parent_details.count("Latest prep inspection:"), 1)
        self.assertEqual(
            cycle.result.metadata.get("service_inspection"),
            "restart_remote_service",
        )
        self.assertIn("Verification target", cycle.result.tool_outcome.content)

    def test_executor_resolves_matching_service_sync_task_after_healthy_verification(self) -> None:
        self.store.record_task(
            "Cockpit setup: Install local service",
            status="open",
            area="execution",
            service_action="install_local_service",
            service_label="Install local service",
            complete_on_success=True,
            tags=["cockpit", "service-action", "service-sync"],
        )
        self.store.record_task(
            "Verify Restart local cockpit service",
            status="open",
            area="execution",
            service_inspection="restart_local_service",
            service_label="Restart local service",
            complete_on_success=True,
            tags=["service-verification", "post-action"],
        )

        class FakeServiceManager:
            def inspect_action(self, action: str) -> dict[str, object]:
                self.last_action = action
                return {
                    "action": action,
                    "inspection": {"active": True, "status": "active"},
                    "verification_target": "http://127.0.0.1:8765/",
                    "settings": {},
                }

        service_manager = FakeServiceManager()
        executor = MemoryExecutor(self.store, service_manager=service_manager)
        cycle = executor.execute_action(
            PlannerAction(
                kind="work_task",
                title="Verify Restart local cockpit service",
                summary="Verify the local cockpit service after restart.",
                score=0.9,
                metadata={"area": "execution"},
            )
        )

        self.assertEqual(cycle.executed_kind, "run_service_inspection")
        self.assertEqual(cycle.status, "success")
        self.assertEqual(
            cycle.metadata.get("resolved_service_sync_titles"),
            ["Cockpit setup: Install local service"],
        )
        resolved_sync = next(
            task
            for task in self.store._active_task_memories()
            if task.subject == "execution"
            and str(task.metadata.get("title") or "") == "Cockpit setup: Install local service"
            and str(task.metadata.get("status") or "") == "done"
        )
        self.assertEqual(resolved_sync.metadata.get("status"), "done")

    def test_executor_run_maintenance_syncs_recommended_service_tasks(self) -> None:
        class FakeServiceManager:
            def settings(self) -> dict[str, object]:
                return {
                    "onboarding": {
                        "actions": [
                            {
                                "action": "install_local_service",
                                "label": "Install local service",
                                "description": "Install or repair the local cockpit background service.",
                                "enabled": True,
                                "success_message": "Local cockpit service installed or repaired.",
                            }
                        ]
                    }
                }

        executor = MemoryExecutor(self.store, service_manager=FakeServiceManager())
        result = executor.execute_action(
            PlannerAction(
                kind="run_maintenance",
                title="Run due memory maintenance",
                summary="Run maintenance.",
                score=0.5,
            )
        )

        self.assertEqual(result.executed_kind, "run_maintenance")
        self.assertEqual(result.status, "success")
        self.assertIn("service_sync", result.metadata.get("executed", []))
        self.assertIn("service_sync", result.maintenance_report.get("executed", {}))
        synced_task = self.store.find_active_task(
            "Cockpit setup: Install local service",
            area="execution",
        )
        self.assertIsNotNone(synced_task)
        self.assertEqual(synced_task.metadata.get("service_action"), "install_local_service")

    def test_linux_pilot_runtime_requires_approval_for_service_action_tasks(self) -> None:
        trace_dir = self.temp_root / f"pilot_traces_{uuid.uuid4().hex}"
        self.extra_dirs.append(trace_dir)

        self.store.record_task(
            "Restart local cockpit service",
            status="open",
            area="execution",
            service_action="restart_local_service",
            service_label="Restart local service",
            complete_on_success=True,
        )
        policy = LinuxPilotPolicy.default(workspace_root=Path.cwd())
        policy.trace_dir = trace_dir
        runtime = LinuxPilotRuntime(self.store, policy=policy)

        report = runtime.run_turn("restart local cockpit service", use_model=False)

        self.assertEqual(report.approval.status, "needs_approval")
        self.assertEqual(report.approval.category, "service_action")
        self.assertIn("restart_local_service", report.approval.prompt)
        self.assertEqual(report.approval.metadata.get("service_label"), "Restart local service")
        self.assertFalse(report.approval.metadata.get("service_requires_confirmation"))
        self.assertIsNone(report.execution_result)
        self.assertIsNone(report.assistant_event_id)
        self.assertTrue(Path(str(report.trace_path)).exists())

    def test_planner_carries_latest_prep_inspection_on_service_actions(self) -> None:
        self.store.record_task(
            "Restart local cockpit service",
            status="open",
            area="execution",
            details=(
                "Confirm reconnect timing.\n"
                "Latest prep inspection: status=inactive, active=no. Verification target: http://127.0.0.1:8765/"
            ),
            service_action="restart_local_service",
            service_label="Restart local service",
            complete_on_success=True,
        )

        planner = MemoryPlanner(self.store)
        snapshot = planner.build_plan("restart local cockpit service", action_limit=3)

        self.assertIsNotNone(snapshot.recommendation)
        self.assertEqual(snapshot.recommendation.kind, "work_task")
        self.assertEqual(
            snapshot.recommendation.metadata.get("latest_prep_inspection"),
            "status=inactive, active=no. Verification target: http://127.0.0.1:8765/",
        )

    def test_linux_pilot_runtime_approval_uses_latest_prep_inspection_metadata(self) -> None:
        self.store.record_task(
            "Restart local cockpit service",
            status="open",
            area="execution",
            service_action="restart_local_service",
            service_label="Restart local service",
            complete_on_success=True,
        )
        policy = LinuxPilotPolicy.default(workspace_root=Path.cwd())
        runtime = LinuxPilotRuntime(self.store, policy=policy)
        task = self.store.find_active_task(
            "Restart local cockpit service",
            area="execution",
            decorate=True,
        )

        approval = runtime._approval_for_task_record(
            task,
            action=PlannerAction(
                kind="work_task",
                title="Restart local cockpit service",
                summary="Work on the service task now.",
                score=0.9,
                task_id=task.id if task is not None else None,
                metadata={
                    "area": "execution",
                    "latest_prep_inspection": (
                        "status=inactive, active=no. Verification target: http://127.0.0.1:8765/"
                    ),
                },
            ),
            source="test",
        )

        self.assertEqual(approval.status, "needs_approval")
        self.assertIn("Latest prep inspection: status=inactive, active=no.", approval.prompt)
        self.assertEqual(
            approval.metadata.get("latest_prep_inspection"),
            "status=inactive, active=no. Verification target: http://127.0.0.1:8765/",
        )

    def test_linux_pilot_runtime_auto_approves_service_inspection_tasks(self) -> None:
        trace_dir = self.temp_root / f"pilot_traces_{uuid.uuid4().hex}"
        self.extra_dirs.append(trace_dir)
        self.store.record_task(
            "Inspect local cockpit service",
            status="open",
            area="execution",
            service_inspection="restart_local_service",
            service_label="Restart local service",
            complete_on_success=True,
        )

        class FakeServiceManager:
            def inspect_action(self, action: str) -> dict[str, object]:
                return {
                    "action": action,
                    "inspection": {"active": True, "status": "active"},
                    "verification_target": "http://127.0.0.1:8765/",
                    "settings": {},
                }

        policy = LinuxPilotPolicy.default(workspace_root=Path.cwd())
        policy.trace_dir = trace_dir
        runtime = LinuxPilotRuntime(self.store, policy=policy)
        runtime.executor.service_manager = FakeServiceManager()

        report = runtime.run_turn("inspect local cockpit service", use_model=False)

        self.assertEqual(report.approval.status, "auto_approved")
        self.assertEqual(report.approval.category, "service_inspection")
        self.assertIsNotNone(report.execution_result)
        self.assertEqual(report.execution_result.executed_kind, "run_service_inspection")
        self.assertEqual(report.execution_result.status, "success")

    def test_executor_schedules_retry_for_failed_shell_task(self) -> None:
        self.store.record_task(
            "Retry CLI help",
            status="open",
            area="execution",
            command="python -m memory_agent.cli --help",
            retry_limit=2,
            retry_cooldown_minutes=5,
        )

        def fake_runner(argv, cwd, capture_output, text, timeout, shell):
            self.assertEqual(argv[:3], ["python", "-m", "memory_agent.cli"])
            self.assertFalse(shell)
            return subprocess.CompletedProcess(argv, 1, "", "boom\n")

        executor = MemoryExecutor(
            self.store,
            shell_adapter=GuardedShellAdapter(
                workspace_root=Path.cwd(),
                runner=fake_runner,
            ),
        )
        cycle = executor.execute_next("retry cli help", action_limit=3)

        self.assertEqual(cycle.result.executed_kind, "run_shell")
        self.assertEqual(cycle.result.status, "error")
        self.assertIn("retry 1/2 was scheduled", cycle.result.summary.lower())
        self.assertIsNotNone(cycle.result.task_update)
        self.assertEqual(cycle.result.task_update.metadata.get("status"), "open")
        self.assertEqual(cycle.result.task_update.metadata.get("retry_count"), 1)
        self.assertEqual(cycle.result.task_update.metadata.get("retry_limit"), 2)
        self.assertEqual(cycle.result.task_update.metadata.get("retry_cooldown_minutes"), 5)
        self.assertTrue(cycle.result.task_update.metadata.get("snoozed_until"))

    def test_planner_surfaces_retry_ready_reason_after_cooldown(self) -> None:
        self.store.record_task(
            "Retry ready task",
            status="open",
            area="execution",
            command="python -m memory_agent.cli --help",
            retry_limit=2,
            retry_count=1,
            retry_cooldown_minutes=0,
            last_retry_at="2026-03-29T00:00:00+00:00",
            last_failure_at="2026-03-29T00:00:00+00:00",
        )

        planner = MemoryPlanner(self.store)
        snapshot = planner.build_plan("what should I do next", action_limit=3)

        self.assertIsNotNone(snapshot.recommendation)
        self.assertEqual(snapshot.recommendation.title, "Retry ready task")
        self.assertIn("retry_ready=1/2", snapshot.recommendation.reasons)

    def test_linux_pilot_runtime_requires_approval_for_file_write_tasks(self) -> None:
        target = self.temp_root / f"{uuid.uuid4().hex}_pilot_write.txt"
        self.extra_paths.append(target)
        target.write_text("alpha\n", encoding="utf-8")
        trace_dir = self.temp_root / f"pilot_traces_{uuid.uuid4().hex}"
        self.extra_dirs.append(trace_dir)

        self.store.record_task(
            "Write pilot file",
            status="open",
            area="execution",
            file_operation="write_text",
            file_path=str(target.relative_to(Path.cwd())),
            file_text="updated\n",
            complete_on_success=True,
        )
        policy = LinuxPilotPolicy.default(workspace_root=Path.cwd())
        policy.trace_dir = trace_dir
        runtime = LinuxPilotRuntime(
            self.store,
            policy=policy,
            patch_runner=self._make_runtime_patch_runner(),
        )

        report = runtime.run_turn("write pilot file", use_model=False)

        self.assertEqual(report.approval.status, "needs_approval")
        self.assertEqual(report.approval.category, "file_operation")
        self.assertIsNotNone(report.approval.preview_patch)
        self.assertIn(target.relative_to(Path.cwd()).as_posix(), report.approval.preview_patch.changed_files)
        self.assertIn("+updated", report.approval.preview_patch.diff_preview)
        self.assertIsNone(report.execution_result)
        self.assertIsNone(report.assistant_event_id)
        self.assertEqual(target.read_text(encoding="utf-8"), "alpha\n")
        self.assertIsNotNone(report.trace_path)
        self.assertTrue(Path(str(report.trace_path)).exists())

    def test_linux_pilot_runtime_auto_executes_read_only_file_tasks(self) -> None:
        target = self.temp_root / f"{uuid.uuid4().hex}_pilot_read.txt"
        self.extra_paths.append(target)
        target.write_text("alpha\n", encoding="utf-8")
        trace_dir = self.temp_root / f"pilot_traces_{uuid.uuid4().hex}"
        self.extra_dirs.append(trace_dir)

        self.store.record_task(
            "Read pilot file",
            status="open",
            area="execution",
            file_operation="read_text",
            file_path=str(target.relative_to(Path.cwd())),
            complete_on_success=True,
        )
        policy = LinuxPilotPolicy.default(workspace_root=Path.cwd())
        policy.trace_dir = trace_dir
        runtime = LinuxPilotRuntime(self.store, policy=policy)

        report = runtime.run_turn("read pilot file", use_model=False)

        self.assertEqual(report.approval.status, "auto_approved")
        self.assertIsNotNone(report.execution_result)
        self.assertEqual(report.execution_result.status, "success")
        self.assertIsNotNone(report.execution_result.file_result)
        self.assertEqual(report.execution_result.file_result.operation, "read_text")
        self.assertIsNotNone(report.assistant_event_id)
        self.assertTrue(Path(str(report.trace_path)).exists())

    def test_linux_pilot_runtime_can_execute_write_after_explicit_approval(self) -> None:
        target = self.temp_root / f"{uuid.uuid4().hex}_pilot_approve.txt"
        self.extra_paths.append(target)
        target.write_text("alpha\n", encoding="utf-8")
        trace_dir = self.temp_root / f"pilot_traces_{uuid.uuid4().hex}"
        self.extra_dirs.append(trace_dir)

        self.store.record_task(
            "Approve pilot write",
            status="open",
            area="execution",
            file_operation="write_text",
            file_path=str(target.relative_to(Path.cwd())),
            file_text="approved\n",
            complete_on_success=True,
        )
        policy = LinuxPilotPolicy.default(workspace_root=Path.cwd())
        policy.trace_dir = trace_dir
        self._record_green_baseline()
        runtime = LinuxPilotRuntime(
            self.store,
            policy=policy,
            patch_runner=self._make_runtime_patch_runner(),
        )

        report = runtime.run_turn("approve pilot write", use_model=False, approve=True)

        self.assertEqual(report.approval.status, "approved")
        self.assertIsNotNone(report.execution_result)
        self.assertEqual(report.execution_result.status, "success")
        self.assertEqual(report.execution_result.executed_kind, "run_patch_preview")
        self.assertIsNotNone(report.execution_result.patch_run)
        self.assertEqual(report.execution_result.patch_run.status, "applied")
        self.assertEqual(target.read_text(encoding="utf-8"), "approved\n")

    def test_linux_pilot_runtime_auto_approves_trusted_low_risk_write(self) -> None:
        target = self.temp_root / f"{uuid.uuid4().hex}_pilot_trusted.txt"
        self.extra_paths.append(target)
        target.write_text("alpha\n", encoding="utf-8")
        trace_dir = self.temp_root / f"pilot_traces_{uuid.uuid4().hex}"
        self.extra_dirs.append(trace_dir)
        rel_path = str(target.relative_to(Path.cwd()))

        for index in range(2):
            self.store.record_patch_run(
                run_name=f"trusted preview {index}",
                suite_name="builtin",
                task_title="Trusted pilot write",
                status="applied",
                baseline_evaluation={"score": 1.0},
                candidate_evaluation={"score": 1.0},
                apply_on_success=True,
                applied=True,
                workspace_path=str(Path.cwd()),
                changed_files=[rel_path],
                operation_results=[
                    {
                        "operation": "write_text",
                        "path": rel_path,
                        "status": "success",
                    }
                ],
                validation_results=[],
                summary={"git": {"status": "not_applicable", "rollback_ready": False}},
            )

        self.store.record_task(
            "Trusted pilot write",
            status="open",
            area="execution",
            file_operation="write_text",
            file_path=rel_path,
            file_text="trusted\n",
            complete_on_success=True,
        )
        policy = LinuxPilotPolicy.default(workspace_root=Path.cwd())
        policy.trace_dir = trace_dir
        self._record_green_baseline()
        runtime = LinuxPilotRuntime(
            self.store,
            policy=policy,
            patch_runner=self._make_runtime_patch_runner(),
        )

        report = runtime.run_turn("trusted pilot write", use_model=False)

        self.assertEqual(report.approval.status, "auto_approved")
        self.assertEqual(report.approval.category, "trusted_file_operation")
        self.assertEqual(report.approval.metadata.get("trusted_operation"), "write_text")
        self.assertEqual(report.approval.metadata.get("trusted_path"), rel_path)
        self.assertEqual(report.approval.metadata.get("matched_successes"), 2)
        self.assertEqual(report.approval.metadata.get("required_successes"), 2)
        self.assertIsNotNone(report.execution_result)
        self.assertEqual(report.execution_result.status, "success")
        self.assertEqual(report.execution_result.executed_kind, "run_patch_preview")
        self.assertEqual(target.read_text(encoding="utf-8"), "trusted\n")

    def test_linux_pilot_runtime_respects_trusted_write_threshold(self) -> None:
        target = self.temp_root / f"{uuid.uuid4().hex}_pilot_trusted_threshold.txt"
        self.extra_paths.append(target)
        target.write_text("alpha\n", encoding="utf-8")
        trace_dir = self.temp_root / f"pilot_traces_{uuid.uuid4().hex}"
        self.extra_dirs.append(trace_dir)
        rel_path = str(target.relative_to(Path.cwd()))

        for index in range(2):
            self.store.record_patch_run(
                run_name=f"trusted threshold {index}",
                suite_name="builtin",
                task_title="Trusted threshold write",
                status="applied",
                baseline_evaluation={"score": 1.0},
                candidate_evaluation={"score": 1.0},
                apply_on_success=True,
                applied=True,
                workspace_path=str(Path.cwd()),
                changed_files=[rel_path],
                operation_results=[
                    {
                        "operation": "write_text",
                        "path": rel_path,
                        "status": "success",
                    }
                ],
                validation_results=[],
                summary={"git": {"status": "not_applicable", "rollback_ready": False}},
            )

        self.store.record_task(
            "Trusted threshold write",
            status="open",
            area="execution",
            file_operation="write_text",
            file_path=rel_path,
            file_text="threshold\n",
            complete_on_success=True,
        )
        policy = LinuxPilotPolicy.default(workspace_root=Path.cwd())
        policy.trace_dir = trace_dir
        policy.trusted_auto_approve_required_successes = 3
        self._record_green_baseline()
        runtime = LinuxPilotRuntime(
            self.store,
            policy=policy,
            patch_runner=self._make_runtime_patch_runner(),
        )

        report = runtime.run_turn("trusted threshold write", use_model=False)

        self.assertEqual(report.approval.status, "needs_approval")
        self.assertEqual(report.approval.category, "file_operation")
        self.assertIsNotNone(report.approval.preview_patch)
        self.assertIsNone(report.execution_result)

    def test_linux_pilot_policy_round_trips_trusted_write_settings(self) -> None:
        policy = LinuxPilotPolicy.default(workspace_root=Path.cwd())
        policy.trusted_auto_approve_file_operations = {"append_text"}
        policy.trusted_auto_approve_required_successes = 4
        policy.service_sync_suppression_window_seconds = {
            "local_service": 300,
            "remote_service": 2400,
            "desktop_launcher": 900,
        }
        policy_path = self.temp_root / f"{uuid.uuid4().hex}_pilot_policy.toml"
        self.extra_paths.append(policy_path)
        policy_path.write_text(policy.render_template(), encoding="utf-8")

        loaded = LinuxPilotPolicy.load(policy_path, workspace_root=Path.cwd())

        self.assertEqual(loaded.trusted_auto_approve_file_operations, {"append_text"})
        self.assertEqual(loaded.trusted_auto_approve_required_successes, 4)
        self.assertEqual(
            loaded.service_sync_suppression_window_seconds,
            {
                "local_service": 300,
                "remote_service": 2400,
                "desktop_launcher": 900,
            },
        )

    def test_linux_pilot_runtime_can_approve_pending_turn_without_duplicate_user_event(self) -> None:
        target = self.temp_root / f"{uuid.uuid4().hex}_pilot_pending.txt"
        self.extra_paths.append(target)
        target.write_text("alpha\n", encoding="utf-8")
        trace_dir = self.temp_root / f"pilot_traces_{uuid.uuid4().hex}"
        self.extra_dirs.append(trace_dir)

        self.store.record_task(
            "Queued pilot write",
            status="open",
            area="execution",
            file_operation="write_text",
            file_path=str(target.relative_to(Path.cwd())),
            file_text="queued\n",
            complete_on_success=True,
        )
        baseline_events = self.store.stats()["events"]
        policy = LinuxPilotPolicy.default(workspace_root=Path.cwd())
        policy.trace_dir = trace_dir
        self._record_green_baseline()
        runtime = LinuxPilotRuntime(
            self.store,
            policy=policy,
            patch_runner=self._make_runtime_patch_runner(),
        )

        report = runtime.run_turn("queued pilot write", use_model=False)

        self.assertEqual(self.store.stats()["events"], baseline_events + 1)
        self.assertEqual(report.approval.status, "needs_approval")
        trace_path = report.trace_path

        approved = runtime.approve_turn(report)

        self.assertEqual(approved.user_event_id, report.user_event_id)
        self.assertEqual(approved.approval.status, "approved")
        self.assertIsNotNone(approved.execution_result)
        self.assertEqual(approved.execution_result.status, "success")
        self.assertEqual(approved.execution_result.executed_kind, "run_patch_preview")
        self.assertEqual(target.read_text(encoding="utf-8"), "queued\n")
        self.assertEqual(self.store.stats()["events"], baseline_events + 2)
        self.assertEqual(approved.trace_path, trace_path)

    def test_linux_pilot_runtime_uses_model_selected_action_before_execution(self) -> None:
        trace_dir = self.temp_root / f"pilot_traces_{uuid.uuid4().hex}"
        self.extra_dirs.append(trace_dir)
        self.store.record_task(
            "Pilot model selected task",
            status="open",
            area="execution",
        )

        class FakeModelAdapter(BaseModelAdapter):
            @property
            def enabled(self) -> bool:
                return True

            def chat(self, messages: list[ModelMessage]) -> ModelResponse:
                return ModelResponse(
                    content=(
                        '{"assistant_message":"I am starting the selected task now.",'
                        '"action":{"type":"execute_plan_action","option_id":"A1"}}'
                    ),
                    model="fake-model",
                )

            def status(self) -> dict[str, object]:
                return {
                    "enabled": True,
                    "backend": "fake",
                    "model": "fake-model",
                }

        policy = LinuxPilotPolicy.default(workspace_root=Path.cwd())
        policy.trace_dir = trace_dir
        runtime = LinuxPilotRuntime(
            self.store,
            policy=policy,
            model_adapter=FakeModelAdapter(),
        )

        report = runtime.run_turn("What should I do next?", use_model=True)

        self.assertEqual(report.selected_action_source, "model")
        self.assertIsNotNone(report.model_action)
        self.assertEqual(report.model_action.action_type, "execute_plan_action")
        self.assertIsNotNone(report.execution_result)
        self.assertEqual(report.execution_result.status, "success")
        self.assertIn("selected task", str(report.assistant_message).lower())

    def test_linux_pilot_runtime_session_stops_when_approval_is_required(self) -> None:
        target = self.temp_root / f"{uuid.uuid4().hex}_pilot_run_gate.txt"
        self.extra_paths.append(target)
        target.write_text("alpha\n", encoding="utf-8")
        trace_dir = self.temp_root / f"pilot_traces_{uuid.uuid4().hex}"
        self.extra_dirs.append(trace_dir)

        self.store.record_task(
            "Pilot run gated write",
            status="open",
            area="execution",
            file_operation="write_text",
            file_path=str(target.relative_to(Path.cwd())),
            file_text="run-gated\n",
            complete_on_success=True,
        )
        policy = LinuxPilotPolicy.default(workspace_root=Path.cwd())
        policy.trace_dir = trace_dir
        runtime = LinuxPilotRuntime(self.store, policy=policy)

        run = runtime.run_session(
            "pilot run gated write",
            max_steps=3,
            auto_approve=False,
            use_model=False,
        )

        self.assertEqual(run.stop_reason, "needs_approval")
        self.assertEqual(len(run.steps), 1)
        self.assertEqual(run.executed_steps, 0)
        self.assertEqual(target.read_text(encoding="utf-8"), "alpha\n")

    def test_linux_pilot_runtime_session_executes_multiple_safe_steps(self) -> None:
        first = self.temp_root / f"{uuid.uuid4().hex}_pilot_run_first.txt"
        second = self.temp_root / f"{uuid.uuid4().hex}_pilot_run_second.txt"
        self.extra_paths.extend([first, second])
        first.write_text("one\n", encoding="utf-8")
        second.write_text("two\n", encoding="utf-8")
        trace_dir = self.temp_root / f"pilot_traces_{uuid.uuid4().hex}"
        self.extra_dirs.append(trace_dir)

        self.store.record_task(
            "Pilot run first read",
            status="open",
            area="execution",
            file_operation="read_text",
            file_path=str(first.relative_to(Path.cwd())),
            complete_on_success=True,
        )
        self.store.record_task(
            "Pilot run second read",
            status="open",
            area="execution",
            file_operation="read_text",
            file_path=str(second.relative_to(Path.cwd())),
            complete_on_success=True,
        )
        policy = LinuxPilotPolicy.default(workspace_root=Path.cwd())
        policy.trace_dir = trace_dir
        runtime = LinuxPilotRuntime(self.store, policy=policy)

        run = runtime.run_session(
            "pilot run read",
            max_steps=2,
            auto_approve=True,
            use_model=False,
        )

        self.assertEqual(len(run.steps), 2)
        self.assertEqual(run.executed_steps, 2)
        self.assertEqual(run.stop_reason, "max_steps")
        self.assertEqual(run.approval_requests, 0)
        self.assertEqual(run.approvals_granted, 0)
        self.assertEqual(len(run.trace_paths), 2)

    def test_linux_pilot_runtime_auto_executes_preparation_actions(self) -> None:
        trace_dir = self.temp_root / f"pilot_traces_{uuid.uuid4().hex}"
        self.extra_dirs.append(trace_dir)
        for index in range(2):
            self.store.record_tool_outcome(
                "pilot-review",
                f"Pilot review {index} stopped on approval friction",
                status="blocked",
                subject="self_improvement",
                tags=["pilot", "review", "self-improvement"],
                metadata={
                    "goal_text": f"pilot history runtime seed {index}",
                    "stop_reason": "needs_approval",
                    "executed_steps": 0,
                    "approval_requests": 1,
                    "approvals_granted": 0,
                    "opportunity_count": 1,
                    "opportunity_categories": ["approval_friction"],
                    "recurring_patterns": [],
                },
            )
        self.store.record_task(
            "Pilot runtime risky write",
            status="open",
            area="execution",
            file_operation="write_text",
            file_path=".test_tmp/pilot_runtime_risky_write.txt",
            file_text="updated\n",
            complete_on_success=True,
        )
        policy = LinuxPilotPolicy.default(workspace_root=Path.cwd())
        policy.trace_dir = trace_dir
        runtime = LinuxPilotRuntime(self.store, policy=policy)

        report = runtime.run_turn("what should I do next", use_model=False)

        self.assertIsNotNone(report.selected_action)
        self.assertEqual(report.selected_action.kind, "prepare_task")
        self.assertEqual(report.approval.status, "auto_approved")
        self.assertIsNotNone(report.execution_result)
        self.assertEqual(report.execution_result.executed_kind, "prepare_task")
        self.assertEqual(report.execution_result.status, "success")
        self.assertIsNotNone(report.after_plan)
        self.assertEqual(
            report.after_plan.recommendation.title,
            "Prepare safer execution for Pilot runtime risky write",
        )

    def test_linux_pilot_runtime_auto_executes_preparation_for_confirmation_heavy_service_task(self) -> None:
        trace_dir = self.temp_root / f"pilot_traces_{uuid.uuid4().hex}"
        self.extra_dirs.append(trace_dir)
        self.store.record_task(
            "Restart remote cockpit service",
            status="open",
            area="execution",
            service_action="restart_remote_service",
            service_label="Restart remote service",
            service_requires_confirmation=True,
            service_confirmation_message=(
                "Restart remote access for this machine? Existing remote browser sessions may need to reconnect afterward."
            ),
            complete_on_success=True,
        )
        policy = LinuxPilotPolicy.default(workspace_root=Path.cwd())
        policy.trace_dir = trace_dir
        runtime = LinuxPilotRuntime(self.store, policy=policy)

        report = runtime.run_turn("what should I do next", use_model=False)

        self.assertIsNotNone(report.selected_action)
        self.assertEqual(report.selected_action.kind, "prepare_task")
        self.assertEqual(report.approval.status, "auto_approved")
        self.assertIsNotNone(report.execution_result)
        self.assertEqual(report.execution_result.executed_kind, "prepare_task")
        self.assertEqual(report.execution_result.status, "success")
        self.assertEqual(
            report.execution_result.related_task.metadata.get("title"),
            "Prepare safer execution for Restart remote cockpit service",
        )
        prep_details = str(report.execution_result.related_task.metadata.get("details") or "")
        self.assertIn("1. Inspect current service state", prep_details)
        self.assertIn("3. Define the post-action verification", prep_details)
        self.assertEqual(
            report.execution_result.related_task.metadata.get("service_inspection"),
            "restart_remote_service",
        )
        self.assertTrue(
            report.execution_result.related_task.metadata.get("complete_on_success")
        )

    def test_pilot_run_review_flags_approval_friction_and_promotes_task(self) -> None:
        target = self.temp_root / f"{uuid.uuid4().hex}_pilot_review_gate.txt"
        self.extra_paths.append(target)
        target.write_text("alpha\n", encoding="utf-8")
        trace_dir = self.temp_root / f"pilot_traces_{uuid.uuid4().hex}"
        self.extra_dirs.append(trace_dir)

        self.store.record_task(
            "Pilot review gated write",
            status="open",
            area="execution",
            file_operation="write_text",
            file_path=str(target.relative_to(Path.cwd())),
            file_text="reviewed\n",
            complete_on_success=True,
        )
        policy = LinuxPilotPolicy.default(workspace_root=Path.cwd())
        policy.trace_dir = trace_dir
        runtime = LinuxPilotRuntime(self.store, policy=policy)
        run = runtime.run_session(
            "pilot review gated write",
            max_steps=2,
            auto_approve=False,
            use_model=False,
        )

        reviewer = PilotRunReviewer(self.store)
        review = reviewer.review(run, promote_limit=1)

        self.assertEqual(review.stop_reason, "needs_approval")
        self.assertTrue(review.opportunities)
        self.assertEqual(review.opportunities[0].category, "approval_friction")
        self.assertEqual(len(review.promoted_tasks), 1)
        self.assertIn("Reduce pilot approval friction", review.promoted_tasks[0].content)
        self.assertIsNotNone(review.review_outcome)

    def test_pilot_run_review_recognizes_recurring_cross_run_patterns(self) -> None:
        policy = LinuxPilotPolicy.default(workspace_root=Path.cwd())
        reviewer = PilotRunReviewer(self.store)
        for index in range(2):
            target = self.temp_root / f"{uuid.uuid4().hex}_pilot_history_{index}.txt"
            self.extra_paths.append(target)
            target.write_text("alpha\n", encoding="utf-8")
            trace_dir = self.temp_root / f"pilot_traces_{uuid.uuid4().hex}"
            self.extra_dirs.append(trace_dir)
            policy.trace_dir = trace_dir
            self.store.record_task(
                f"Pilot recurring write {index}",
                status="open",
                area="execution",
                file_operation="write_text",
                file_path=str(target.relative_to(Path.cwd())),
                file_text="beta\n",
                complete_on_success=True,
            )
            runtime = LinuxPilotRuntime(self.store, policy=policy)
            run = runtime.run_session(
                f"pilot recurring write {index}",
                max_steps=2,
                auto_approve=False,
                use_model=False,
            )
            review = reviewer.review(run, promote_limit=0)

        self.assertTrue(review.recurring_patterns)
        pattern_kinds = {pattern["kind"] for pattern in review.recurring_patterns}
        self.assertIn("category", pattern_kinds)
        self.assertIn("stop_reason", pattern_kinds)
        history_opportunities = [
            item for item in review.opportunities if item.category == "pilot_history_pattern"
        ]
        self.assertTrue(history_opportunities)
        self.assertIn("recurring", history_opportunities[0].title.lower())

    def test_pilot_history_report_aggregates_recent_pilot_reviews(self) -> None:
        policy = LinuxPilotPolicy.default(workspace_root=Path.cwd())
        reviewer = PilotRunReviewer(self.store)
        for index in range(2):
            target = self.temp_root / f"{uuid.uuid4().hex}_pilot_report_{index}.txt"
            self.extra_paths.append(target)
            target.write_text("alpha\n", encoding="utf-8")
            trace_dir = self.temp_root / f"pilot_traces_{uuid.uuid4().hex}"
            self.extra_dirs.append(trace_dir)
            policy.trace_dir = trace_dir
            self.store.record_task(
                f"Pilot report write {index}",
                status="open",
                area="execution",
                file_operation="write_text",
                file_path=str(target.relative_to(Path.cwd())),
                file_text="beta\n",
                complete_on_success=True,
            )
            runtime = LinuxPilotRuntime(self.store, policy=policy)
            run = runtime.run_session(
                f"pilot report write {index}",
                max_steps=2,
                auto_approve=False,
                use_model=False,
            )
            reviewer.review(run, promote_limit=0)

        report = PilotHistoryReporter(self.store).build(limit=10)

        self.assertEqual(report.total_reviews, 2)
        self.assertTrue(report.stop_reasons)
        self.assertEqual(report.stop_reasons[0]["key"], "needs_approval")
        self.assertTrue(report.opportunity_categories)
        self.assertEqual(report.opportunity_categories[0]["key"], "approval_friction")
        self.assertTrue(report.recurring_patterns)
        self.assertEqual(report.recurring_patterns[0]["kind"], "category")

    def test_pilot_history_report_surfaces_trusted_write_candidates(self) -> None:
        policy = LinuxPilotPolicy.default(workspace_root=Path.cwd())
        reviewer = PilotRunReviewer(self.store)
        target = self.temp_root / f"{uuid.uuid4().hex}_pilot_trusted_candidate.txt"
        self.extra_paths.append(target)
        target.write_text("alpha\n", encoding="utf-8")
        rel_path = str(target.relative_to(Path.cwd()))

        for index in range(2):
            trace_dir = self.temp_root / f"pilot_traces_{uuid.uuid4().hex}"
            self.extra_dirs.append(trace_dir)
            policy.trace_dir = trace_dir
            self.store.record_task(
                f"Pilot trusted candidate {index}",
                status="open",
                area="execution",
                file_operation="write_text",
                file_path=rel_path,
                file_text=f"beta {index}\n",
                complete_on_success=True,
            )
            runtime = LinuxPilotRuntime(
                self.store,
                policy=policy,
                patch_runner=self._make_runtime_patch_runner(),
            )
            run = runtime.run_session(
                f"pilot trusted candidate {index}",
                max_steps=2,
                auto_approve=False,
                use_model=False,
            )
            reviewer.review(run, promote_limit=0)

        report = PilotHistoryReporter(self.store).build(limit=10)

        trusted_patterns = [
            item for item in report.recurring_patterns if item.get("kind") == "trusted_write_candidate"
        ]
        self.assertTrue(trusted_patterns)
        self.assertEqual(trusted_patterns[0]["file_operation"], "write_text")
        self.assertEqual(trusted_patterns[0]["file_path"], rel_path)
        self.assertGreaterEqual(int(trusted_patterns[0]["count"] or 0), 2)

    def test_pilot_chat_prompts_and_executes_approved_turn(self) -> None:
        target = self.temp_root / f"{uuid.uuid4().hex}_pilot_chat.txt"
        self.extra_paths.append(target)
        target.write_text("alpha\n", encoding="utf-8")
        trace_dir = self.temp_root / f"pilot_traces_{uuid.uuid4().hex}"
        self.extra_dirs.append(trace_dir)
        policy_path = self.temp_root / f"{uuid.uuid4().hex}_pilot_policy.toml"
        self.extra_paths.append(policy_path)

        self.store.record_task(
            "Pilot chat write",
            status="open",
            area="execution",
            file_operation="write_text",
            file_path=str(target.relative_to(Path.cwd())),
            file_text="chat-approved\n",
            complete_on_success=True,
        )
        policy = LinuxPilotPolicy.default(workspace_root=Path.cwd())
        policy.trace_dir = trace_dir
        policy_path.write_text(policy.render_template(), encoding="utf-8")
        self._record_green_baseline()

        inputs = iter(["pilot chat write", "yes", ":quit"])
        outputs: list[str] = []

        def fake_input(_prompt: str) -> str:
            return next(inputs)

        def fake_output(text: str) -> None:
            outputs.append(text)

        original_runtime = cli_module.LinuxPilotRuntime

        class TestLinuxPilotRuntime(original_runtime):
            def __init__(self, memory_store, **kwargs):
                kwargs.setdefault(
                    "patch_runner",
                    self_outer._make_runtime_patch_runner(),
                )
                super().__init__(memory_store, **kwargs)

        self_outer = self
        cli_module.LinuxPilotRuntime = TestLinuxPilotRuntime
        try:
            result = _run_pilot_chat(
                self.store,
                policy_file=policy_path,
                use_model=False,
                input_fn=fake_input,
                output_fn=fake_output,
            )
        finally:
            cli_module.LinuxPilotRuntime = original_runtime

        self.assertEqual(result, 0)
        self.assertEqual(target.read_text(encoding="utf-8"), "chat-approved\n")
        rendered = "\n".join(outputs)
        self.assertIn("Supervised Linux pilot session", rendered)
        self.assertIn("Approval: [needs_approval]", rendered)
        self.assertIn("Preview packet:", rendered)
        self.assertIn("Approved and executed queued action.", rendered)

    def test_executor_runs_python_symbol_replace_and_completes_task(self) -> None:
        target = self.temp_root / f"{uuid.uuid4().hex}.py"
        self.extra_paths.append(target)
        target.write_text(
            "def greet(name: str) -> str:\n"
            "    return f'hello {name}'\n\n"
            "def untouched() -> str:\n"
            "    return 'still here'\n",
            encoding="utf-8",
        )
        self.store.record_task(
            "Upgrade greet helper",
            status="open",
            area="execution",
            file_operation="replace_python_function",
            file_path=str(target.relative_to(Path.cwd())),
            symbol_name="greet",
            file_text=(
                "def greet(name: str) -> str:\n"
                "    return f'hello there {name}'\n"
            ),
            complete_on_success=True,
        )

        executor = MemoryExecutor(
            self.store,
            file_adapter=WorkspaceFileAdapter(workspace_root=Path.cwd()),
        )
        cycle = executor.execute_next("upgrade greet helper", action_limit=3)

        self.assertEqual(cycle.result.executed_kind, "run_file_operation")
        self.assertEqual(cycle.result.status, "success")
        self.assertIsNotNone(cycle.result.file_result)
        self.assertEqual(cycle.result.task_update.metadata.get("status"), "done")
        updated = target.read_text(encoding="utf-8")
        self.assertIn("hello there", updated)
        self.assertIn("still here", updated)
        self.assertIn("greet", str(cycle.result.metadata.get("symbol_name")))

    def test_executor_runs_python_symbol_replace_on_utf8_bom_file(self) -> None:
        target = self.temp_root / f"{uuid.uuid4().hex}.py"
        self.extra_paths.append(target)
        target.write_text(
            "def greet() -> str:\n    return 'old'\n",
            encoding="utf-8-sig",
        )
        self.store.record_task(
            "Upgrade BOM greet helper",
            status="open",
            area="execution",
            file_operation="replace_python_function",
            file_path=str(target.relative_to(Path.cwd())),
            symbol_name="greet",
            file_text="def greet() -> str:\n    return 'new'\n",
            complete_on_success=True,
        )

        executor = MemoryExecutor(
            self.store,
            file_adapter=WorkspaceFileAdapter(workspace_root=Path.cwd()),
        )
        cycle = executor.execute_next("upgrade bom greet helper", action_limit=3)

        self.assertEqual(cycle.result.status, "success")
        self.assertIn("return 'new'", target.read_text(encoding="utf-8"))

    def test_executor_inserts_python_symbol_after_target_and_completes_task(self) -> None:
        target = self.temp_root / f"{uuid.uuid4().hex}.py"
        self.extra_paths.append(target)
        target.write_text(
            "def greet() -> str:\n"
            "    return 'hello'\n",
            encoding="utf-8",
        )
        self.store.record_task(
            "Add helper after greet",
            status="open",
            area="execution",
            file_operation="insert_python_after_symbol",
            file_path=str(target.relative_to(Path.cwd())),
            symbol_name="greet",
            file_text="def helper() -> str:\n    return 'helper'\n",
            complete_on_success=True,
        )

        executor = MemoryExecutor(
            self.store,
            file_adapter=WorkspaceFileAdapter(workspace_root=Path.cwd()),
        )
        cycle = executor.execute_next("add helper after greet", action_limit=3)

        self.assertEqual(cycle.result.status, "success")
        updated = target.read_text(encoding="utf-8")
        self.assertIn("def greet()", updated)
        self.assertIn("def helper()", updated)
        self.assertLess(updated.index("def greet()"), updated.index("def helper()"))

    def test_executor_renames_python_identifier_and_completes_task(self) -> None:
        target = self.temp_root / f"{uuid.uuid4().hex}.py"
        self.extra_paths.append(target)
        target.write_text(
            "def greet() -> str:\n"
            "    return 'hello'\n\n"
            "value = greet()\n"
            "label = 'greet should stay in strings'\n",
            encoding="utf-8",
        )
        self.store.record_task(
            "Rename greet helper",
            status="open",
            area="execution",
            file_operation="rename_python_identifier",
            file_path=str(target.relative_to(Path.cwd())),
            symbol_name="greet",
            file_text="salute",
            complete_on_success=True,
        )

        executor = MemoryExecutor(
            self.store,
            file_adapter=WorkspaceFileAdapter(workspace_root=Path.cwd()),
        )
        cycle = executor.execute_next("rename greet helper", action_limit=3)

        self.assertEqual(cycle.result.status, "success")
        updated = target.read_text(encoding="utf-8")
        self.assertIn("def salute()", updated)
        self.assertIn("value = salute()", updated)
        self.assertIn("'greet should stay in strings'", updated)

    def test_executor_renames_python_method_and_completes_task(self) -> None:
        target = self.temp_root / f"{uuid.uuid4().hex}.py"
        self.extra_paths.append(target)
        target.write_text(
            "class Greeter:\n"
            "    def speak(self) -> str:\n"
            "        return 'hello'\n\n"
            "greeter = Greeter()\n"
            "value = greeter.speak()\n"
            "method_ref = Greeter.speak\n"
            "label = 'speak should stay in strings'\n",
            encoding="utf-8",
        )
        self.store.record_task(
            "Rename speak method",
            status="open",
            area="execution",
            file_operation="rename_python_method",
            file_path=str(target.relative_to(Path.cwd())),
            symbol_name="Greeter.speak",
            file_text="salute",
            complete_on_success=True,
        )

        executor = MemoryExecutor(
            self.store,
            file_adapter=WorkspaceFileAdapter(workspace_root=Path.cwd()),
        )
        cycle = executor.execute_next("rename speak method", action_limit=3)

        self.assertEqual(cycle.result.status, "success")
        updated = target.read_text(encoding="utf-8")
        self.assertIn("def salute(self)", updated)
        self.assertIn("value = greeter.salute()", updated)
        self.assertIn("method_ref = Greeter.salute", updated)
        self.assertIn("'speak should stay in strings'", updated)

    def test_executor_adds_python_import_and_completes_task(self) -> None:
        target = self.temp_root / f"{uuid.uuid4().hex}.py"
        self.extra_paths.append(target)
        target.write_text(
            "\"\"\"Module docstring.\"\"\"\n\n"
            "from __future__ import annotations\n\n"
            "class Greeter:\n"
            "    pass\n",
            encoding="utf-8",
        )
        self.store.record_task(
            "Add pathlib import",
            status="open",
            area="execution",
            file_operation="add_python_import",
            file_path=str(target.relative_to(Path.cwd())),
            file_text="from pathlib import Path",
            complete_on_success=True,
        )

        executor = MemoryExecutor(
            self.store,
            file_adapter=WorkspaceFileAdapter(workspace_root=Path.cwd()),
        )
        cycle = executor.execute_next("add pathlib import", action_limit=3)

        self.assertEqual(cycle.result.status, "success")
        updated = target.read_text(encoding="utf-8")
        self.assertIn("from __future__ import annotations\nfrom pathlib import Path\n", updated)
        self.assertIn("class Greeter", updated)

    def test_executor_removes_python_import_and_completes_task(self) -> None:
        target = self.temp_root / f"{uuid.uuid4().hex}.py"
        self.extra_paths.append(target)
        target.write_text(
            "from pathlib import Path\n"
            "import json\n\n"
            "VALUE = 1\n",
            encoding="utf-8",
        )
        self.store.record_task(
            "Remove pathlib import",
            status="open",
            area="execution",
            file_operation="remove_python_import",
            file_path=str(target.relative_to(Path.cwd())),
            file_text="from pathlib import Path",
            complete_on_success=True,
        )

        executor = MemoryExecutor(
            self.store,
            file_adapter=WorkspaceFileAdapter(workspace_root=Path.cwd()),
        )
        cycle = executor.execute_next("remove pathlib import", action_limit=3)

        self.assertEqual(cycle.result.status, "success")
        updated = target.read_text(encoding="utf-8")
        self.assertNotIn("from pathlib import Path", updated)
        self.assertIn("import json", updated)
        self.assertIn("VALUE = 1", updated)

    def test_executor_adds_python_function_parameter_and_rewrites_calls(self) -> None:
        target = self.temp_root / f"{uuid.uuid4().hex}.py"
        self.extra_paths.append(target)
        target.write_text(
            "def greet(name: str) -> str:\n"
            "    return name.upper()\n\n"
            "value = greet('sam')\n",
            encoding="utf-8",
        )
        self.store.record_task(
            "Add excited parameter to greet",
            status="open",
            area="execution",
            file_operation="add_python_function_parameter",
            file_path=str(target.relative_to(Path.cwd())),
            symbol_name="greet",
            file_text=json.dumps(
                {
                    "parameter_name": "excited",
                    "call_argument": "True",
                }
            ),
            complete_on_success=True,
        )

        executor = MemoryExecutor(
            self.store,
            file_adapter=WorkspaceFileAdapter(workspace_root=Path.cwd()),
        )
        cycle = executor.execute_next("add excited parameter to greet", action_limit=3)

        self.assertEqual(cycle.result.status, "success")
        updated = target.read_text(encoding="utf-8")
        self.assertIn("def greet(name: str, excited)", updated)
        self.assertIn("value = greet('sam', excited=True)", updated)

    def test_executor_adds_python_method_parameter_and_rewrites_calls(self) -> None:
        target = self.temp_root / f"{uuid.uuid4().hex}.py"
        self.extra_paths.append(target)
        target.write_text(
            "class Greeter:\n"
            "    def speak(self, name: str) -> str:\n"
            "        return name.upper()\n\n"
            "greeter = Greeter()\n"
            "value = greeter.speak('sam')\n",
            encoding="utf-8",
        )
        self.store.record_task(
            "Add excited parameter to Greeter.speak",
            status="open",
            area="execution",
            file_operation="add_python_method_parameter",
            file_path=str(target.relative_to(Path.cwd())),
            symbol_name="Greeter.speak",
            file_text=json.dumps(
                {
                    "parameter_name": "excited",
                    "call_argument": "True",
                }
            ),
            complete_on_success=True,
        )

        executor = MemoryExecutor(
            self.store,
            file_adapter=WorkspaceFileAdapter(workspace_root=Path.cwd()),
        )
        cycle = executor.execute_next("add excited parameter to greeter.speak", action_limit=3)

        self.assertEqual(cycle.result.status, "success")
        updated = target.read_text(encoding="utf-8")
        self.assertIn("def speak(self, name: str, excited)", updated)
        self.assertIn("value = greeter.speak('sam', excited=True)", updated)

    def test_executor_blocks_signature_refactor_for_multiline_callsites_without_default(self) -> None:
        target = self.temp_root / f"{uuid.uuid4().hex}.py"
        self.extra_paths.append(target)
        original = (
            "def greet(name: str) -> str:\n"
            "    return name.upper()\n\n"
            "value = greet(\n"
            "    'sam',\n"
            ")\n"
        )
        target.write_text(original, encoding="utf-8")
        self.store.record_task(
            "Add parameter with multiline calls",
            status="open",
            area="execution",
            file_operation="add_python_function_parameter",
            file_path=str(target.relative_to(Path.cwd())),
            symbol_name="greet",
            file_text=json.dumps(
                {
                    "parameter_name": "excited",
                    "call_argument": "True",
                }
            ),
            complete_on_success=True,
        )

        executor = MemoryExecutor(
            self.store,
            file_adapter=WorkspaceFileAdapter(workspace_root=Path.cwd()),
        )
        cycle = executor.execute_next("add parameter with multiline calls", action_limit=3)

        self.assertEqual(cycle.result.status, "blocked")
        self.assertIsNotNone(cycle.result.file_result)
        self.assertEqual(
            cycle.result.file_result.reason,
            "multiline_call_sites_not_supported",
        )
        self.assertEqual(target.read_text(encoding="utf-8"), original)

    def test_executor_renames_python_export_across_imports_and_completes_task(self) -> None:
        workspace = Path.cwd() / f"refactorws_{uuid.uuid4().hex[:8]}"
        package = workspace / f"pkgrefactor_{uuid.uuid4().hex[:8]}"
        package.mkdir(parents=True, exist_ok=True)
        self.extra_dirs.append(workspace)
        (package / "__init__.py").write_text("", encoding="utf-8")
        source = package / "greeter.py"
        consumer = workspace / "consumer.py"
        module_consumer = workspace / "module_consumer.py"
        source.write_text(
            "def greet(name: str) -> str:\n"
            "    return name.upper()\n\n"
            "value = greet('sam')\n",
            encoding="utf-8",
        )
        consumer.write_text(
            f"from {package.name}.greeter import greet\n"
            "result = greet('zoe')\n",
            encoding="utf-8",
        )
        module_consumer.write_text(
            f"import {package.name}.greeter as greeter_mod\n"
            "other = greeter_mod.greet('amy')\n",
            encoding="utf-8",
        )
        self.store.record_task(
            "Rename greet export across imports",
            status="open",
            area="execution",
            file_operation="rename_python_export_across_imports",
            file_path=str(source.relative_to(Path.cwd())),
            symbol_name="greet",
            file_text="salute",
            complete_on_success=True,
        )

        executor = MemoryExecutor(
            self.store,
            file_adapter=WorkspaceFileAdapter(workspace_root=Path.cwd()),
        )
        cycle = executor.execute_next("rename greet export across imports", action_limit=3)

        self.assertEqual(cycle.result.status, "success")
        self.assertIsNotNone(cycle.result.file_result)
        source_text = source.read_text(encoding="utf-8")
        consumer_text = consumer.read_text(encoding="utf-8")
        module_consumer_text = module_consumer.read_text(encoding="utf-8")
        self.assertIn("def salute(name: str)", source_text)
        self.assertIn("value = salute('sam')", source_text)
        self.assertIn(f"from {package.name}.greeter import salute", consumer_text)
        self.assertIn("result = salute('zoe')", consumer_text)
        self.assertIn("greeter_mod.salute('amy')", module_consumer_text)
        changed_paths = set(cycle.result.file_result.changed_paths)
        self.assertIn(str(source.resolve()), changed_paths)
        self.assertIn(str(consumer.resolve()), changed_paths)
        self.assertIn(str(module_consumer.resolve()), changed_paths)

    def test_executor_blocks_cross_import_rename_when_consumer_binding_conflicts(self) -> None:
        workspace = Path.cwd() / f"refactorws_{uuid.uuid4().hex[:8]}"
        package = workspace / f"pkgrefactor_{uuid.uuid4().hex[:8]}"
        package.mkdir(parents=True, exist_ok=True)
        self.extra_dirs.append(workspace)
        (package / "__init__.py").write_text("", encoding="utf-8")
        source = package / "greeter.py"
        consumer = workspace / "consumer.py"
        original_source = "def greet(name: str) -> str:\n    return name.upper()\n"
        original_consumer = (
            f"from {package.name}.greeter import greet\n"
            "greet = 'local shadow'\n"
            "result = greet\n"
        )
        source.write_text(original_source, encoding="utf-8")
        consumer.write_text(original_consumer, encoding="utf-8")
        self.store.record_task(
            "Rename greet export with conflicting consumer",
            status="open",
            area="execution",
            file_operation="rename_python_export_across_imports",
            file_path=str(source.relative_to(Path.cwd())),
            symbol_name="greet",
            file_text="salute",
            complete_on_success=True,
        )

        executor = MemoryExecutor(
            self.store,
            file_adapter=WorkspaceFileAdapter(workspace_root=Path.cwd()),
        )
        cycle = executor.execute_next("rename greet export with conflicting consumer", action_limit=3)

        self.assertEqual(cycle.result.status, "blocked")
        self.assertIsNotNone(cycle.result.file_result)
        self.assertEqual(cycle.result.file_result.reason, "consumer_binding_conflict")
        self.assertEqual(source.read_text(encoding="utf-8"), original_source)
        self.assertEqual(consumer.read_text(encoding="utf-8"), original_consumer)

    def test_executor_moves_python_export_to_module_and_updates_import_consumers(self) -> None:
        workspace = Path.cwd() / f"movews_{uuid.uuid4().hex[:8]}"
        package = workspace / f"pkgmove_{uuid.uuid4().hex[:8]}"
        package.mkdir(parents=True, exist_ok=True)
        self.extra_dirs.append(workspace)
        (package / "__init__.py").write_text("", encoding="utf-8")
        source = package / "legacy.py"
        destination = package / "core.py"
        consumer = workspace / "consumer.py"
        module_consumer = workspace / "module_consumer.py"
        source.write_text(
            "from pathlib import Path\n\n"
            "def greet(name: str) -> str:\n"
            "    return Path(name).name.upper()\n\n"
            "value = greet('sam.txt')\n",
            encoding="utf-8",
        )
        consumer.write_text(
            f"from {package.name}.legacy import greet\n"
            "result = greet('zoe.txt')\n",
            encoding="utf-8",
        )
        module_consumer.write_text(
            f"import {package.name}.legacy as legacy_mod\n"
            "other = legacy_mod.greet('amy.txt')\n",
            encoding="utf-8",
        )
        self.store.record_task(
            "Move greet export to core module",
            status="open",
            area="execution",
            file_operation="move_python_export_to_module",
            file_path=str(source.relative_to(Path.cwd())),
            symbol_name="greet",
            file_text=json.dumps(
                {
                    "destination_path": str(destination.relative_to(Path.cwd())),
                }
            ),
            complete_on_success=True,
        )

        executor = MemoryExecutor(
            self.store,
            file_adapter=WorkspaceFileAdapter(workspace_root=Path.cwd()),
        )
        cycle = executor.execute_next("move greet export to core module", action_limit=3)

        self.assertEqual(cycle.result.status, "success")
        self.assertIsNotNone(cycle.result.file_result)
        source_text = source.read_text(encoding="utf-8")
        destination_text = destination.read_text(encoding="utf-8")
        consumer_text = consumer.read_text(encoding="utf-8")
        module_consumer_text = module_consumer.read_text(encoding="utf-8")
        self.assertIn(f"from {package.name}.core import greet", source_text)
        self.assertNotIn("def greet(name: str)", source_text)
        self.assertIn("value = greet('sam.txt')", source_text)
        self.assertIn("from pathlib import Path", destination_text)
        self.assertIn("def greet(name: str)", destination_text)
        self.assertIn("return Path(name).name.upper()", destination_text)
        self.assertIn(f"from {package.name}.core import greet", consumer_text)
        self.assertIn("result = greet('zoe.txt')", consumer_text)
        self.assertIn(f"import {package.name}.legacy as legacy_mod", module_consumer_text)
        self.assertIn("legacy_mod.greet('amy.txt')", module_consumer_text)
        changed_paths = set(cycle.result.file_result.changed_paths)
        self.assertIn(str(source.resolve()), changed_paths)
        self.assertIn(str(destination.resolve()), changed_paths)
        self.assertIn(str(consumer.resolve()), changed_paths)
        self.assertNotIn(str(module_consumer.resolve()), changed_paths)

    def test_executor_blocks_move_python_export_when_consumer_import_split_is_required(self) -> None:
        workspace = Path.cwd() / f"movews_{uuid.uuid4().hex[:8]}"
        package = workspace / f"pkgmove_{uuid.uuid4().hex[:8]}"
        package.mkdir(parents=True, exist_ok=True)
        self.extra_dirs.append(workspace)
        (package / "__init__.py").write_text("", encoding="utf-8")
        source = package / "legacy.py"
        destination = package / "core.py"
        consumer = workspace / "consumer.py"
        original_source = (
            "def greet(name: str) -> str:\n"
            "    return name.upper()\n\n"
            "def helper(name: str) -> str:\n"
            "    return name.lower()\n"
        )
        source.write_text(original_source, encoding="utf-8")
        original_consumer = (
            f"from {package.name}.legacy import greet, helper\n"
            "value = greet('sam')\n"
        )
        consumer.write_text(original_consumer, encoding="utf-8")
        self.store.record_task(
            "Move greet export with split import consumer",
            status="open",
            area="execution",
            file_operation="move_python_export_to_module",
            file_path=str(source.relative_to(Path.cwd())),
            symbol_name="greet",
            file_text=json.dumps(
                {
                    "destination_path": str(destination.relative_to(Path.cwd())),
                }
            ),
            complete_on_success=True,
        )

        executor = MemoryExecutor(
            self.store,
            file_adapter=WorkspaceFileAdapter(workspace_root=Path.cwd()),
        )
        cycle = executor.execute_next("move greet export with split import consumer", action_limit=3)

        self.assertEqual(cycle.result.status, "blocked")
        self.assertIsNotNone(cycle.result.file_result)
        self.assertEqual(cycle.result.file_result.reason, "consumer_import_split_required")
        self.assertEqual(source.read_text(encoding="utf-8"), original_source)
        self.assertEqual(consumer.read_text(encoding="utf-8"), original_consumer)
        self.assertFalse(destination.exists())

    def test_file_adapter_blocks_outside_workspace_path(self) -> None:
        outside_path = Path.cwd().parent / f"{uuid.uuid4().hex}.txt"
        self.store.record_task(
            "Write outside workspace",
            status="open",
            area="execution",
            file_operation="write_text",
            file_path=str(outside_path),
            file_text="outside",
        )

        executor = MemoryExecutor(self.store)
        cycle = executor.execute_next("write outside workspace", action_limit=3)

        self.assertEqual(cycle.result.executed_kind, "run_file_operation")
        self.assertEqual(cycle.result.status, "blocked")
        self.assertIsNotNone(cycle.result.file_result)
        self.assertEqual(cycle.result.file_result.reason, "path_outside_workspace")
        self.assertIn("blocked by file policy", cycle.result.summary.lower())

    def test_improvement_review_records_evaluation_and_promotes_current_strategic_backlog(self) -> None:
        class FakeEvaluator:
            def run_builtin_suite(self):
                return EvalSuiteResult(
                    passed=True,
                    score=1.0,
                    scenario_results=[
                        EvalScenarioResult(
                            name="all_green",
                            description="Everything passed.",
                            passed=True,
                            score=1.0,
                            checks=[
                                EvalCheckResult(
                                    name="green",
                                    passed=True,
                                    details="All checks passed.",
                                )
                            ],
                        )
                    ],
                )

        engine = MemoryImprovementEngine(self.store, FakeEvaluator())
        review = engine.review(promote_limit=2)

        self.assertTrue(review.passed)
        self.assertIsNotNone(review.current_evaluation)
        self.assertEqual(review.current_evaluation["checks_passed"], 1)
        self.assertIsNotNone(self.store.latest_evaluation_run())
        self.assertTrue(
            any(
                opportunity.title == "Add richer task orchestration"
                for opportunity in review.opportunities
            )
        )
        self.assertTrue(
            any(
                str(task.metadata.get("title")) == "Add richer task orchestration"
                for task in review.promoted_tasks
            )
        )
        self.assertIsNotNone(
            self.store.find_active_task(
                "Add richer task orchestration",
                area="self_improvement",
            )
        )

    def test_improvement_review_turns_failed_eval_checks_into_self_improvement_tasks(self) -> None:
        class FakeEvaluator:
            def run_builtin_suite(self):
                return EvalSuiteResult(
                    passed=False,
                    score=0.5,
                    scenario_results=[
                        EvalScenarioResult(
                            name="broken_path",
                            description="A key scenario failed.",
                            passed=False,
                            score=0.0,
                            checks=[
                                EvalCheckResult(
                                    name="missing guardrail",
                                    passed=False,
                                    details="The execution path skipped validation.",
                                )
                            ],
                        )
                    ],
                )

        engine = MemoryImprovementEngine(self.store, FakeEvaluator())
        review = engine.review(promote_limit=1, include_strategic_backlog=False)

        self.assertFalse(review.passed)
        self.assertTrue(
            any(
                opportunity.category == "evaluation_failure"
                and "broken_path" in opportunity.title
                for opportunity in review.opportunities
            )
        )
        self.assertEqual(len(review.promoted_tasks), 1)
        self.assertEqual(
            review.promoted_tasks[0].subject,
            "self_improvement",
        )
        self.assertIn(
            "Resolve failing evaluation scenario",
            str(review.promoted_tasks[0].metadata.get("details") or ""),
        )
        latest_outcome = self.store.get_recent_tool_outcomes(limit=1)[0]
        self.assertEqual(latest_outcome.subject, "self_improvement")

    def test_patch_runner_applies_validated_candidate_and_records_patch_run(self) -> None:
        self._record_green_baseline()
        workspace = self._make_workspace()
        target = workspace / "target.txt"
        target.write_text("alpha\nbeta\n", encoding="utf-8")

        def fake_runner(argv, cwd, capture_output, text, timeout, shell):
            self.assertFalse(shell)
            if argv[:3] == ["python", "-m", "memory_agent.cli"]:
                payload = {
                    "passed": True,
                    "score": 1.0,
                    "scenario_results": [
                        {
                            "name": "candidate",
                            "description": "Candidate eval passed.",
                            "passed": True,
                            "score": 1.0,
                            "checks": [
                                {
                                    "name": "candidate-check",
                                    "passed": True,
                                    "details": "All checks passed.",
                                }
                            ],
                        }
                    ],
                }
                return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")
            return subprocess.CompletedProcess(argv, 0, "ok", "")

        runner = WorkspacePatchRunner(
            self.store,
            workspace_root=workspace,
            shell_runner=fake_runner,
        )
        report = runner.run(
            "promote target update",
            operations=[
                PatchOperation(
                    operation="replace_text",
                    path="target.txt",
                    find_text="beta",
                    text="gamma",
                )
            ],
            apply_on_success=True,
        )

        self.assertEqual(report.status, "applied")
        self.assertTrue(report.applied)
        self.assertEqual(target.read_text(encoding="utf-8"), "alpha\ngamma\n")
        self.assertEqual(report.changed_files, ["target.txt"])
        self.assertIsNotNone(report.tool_outcome)
        self.assertEqual(report.tool_outcome.metadata.get("patch_run_id"), report.run_id)
        latest_patch_run = self.store.latest_patch_run()
        self.assertIsNotNone(latest_patch_run)
        self.assertEqual(latest_patch_run["status"], "applied")
        self.assertEqual(latest_patch_run["changed_files"], ["target.txt"])

    def test_patch_runner_commits_successful_candidate_to_disposable_git_branch(self) -> None:
        self._record_green_baseline()
        workspace = self._make_workspace()
        target = workspace / "target.txt"
        target.write_text("alpha\nbeta\n", encoding="utf-8")
        repo_root = str(workspace.resolve())
        calls: list[list[str]] = []
        state = {
            "current_branch": "main",
            "branches": {"main": "main-head"},
            "commit_counter": 0,
        }

        def fake_runner(argv, cwd, capture_output, text, timeout, shell):
            self.assertFalse(shell)
            calls.append([str(part) for part in argv])
            command = [str(part) for part in argv]
            if command[:3] == ["git", "rev-parse", "--show-toplevel"]:
                return subprocess.CompletedProcess(argv, 0, repo_root, "")
            if command[:3] == ["git", "branch", "--show-current"]:
                return subprocess.CompletedProcess(argv, 0, state["current_branch"], "")
            if command[:3] == ["git", "status", "--porcelain"]:
                return subprocess.CompletedProcess(argv, 0, "", "")
            if command[:3] == ["git", "rev-parse", "HEAD"]:
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    state["branches"][state["current_branch"]],
                    "",
                )
            if len(command) == 3 and command[:2] == ["git", "rev-parse"]:
                branch = command[2]
                if branch in state["branches"]:
                    return subprocess.CompletedProcess(argv, 0, state["branches"][branch], "")
                return subprocess.CompletedProcess(argv, 1, "", "unknown branch")
            if command[:3] == ["git", "checkout", "-b"]:
                branch = command[3]
                state["branches"][branch] = state["branches"][state["current_branch"]]
                state["current_branch"] = branch
                return subprocess.CompletedProcess(argv, 0, f"Switched to {branch}", "")
            if command[:2] == ["git", "add"]:
                return subprocess.CompletedProcess(argv, 0, "", "")
            if "commit" in command:
                state["commit_counter"] += 1
                state["branches"][state["current_branch"]] = (
                    f"commit-{state['commit_counter']}"
                )
                return subprocess.CompletedProcess(argv, 0, "[branch] commit", "")
            if command[:3] == ["python", "-m", "memory_agent.cli"]:
                payload = {
                    "passed": True,
                    "score": 1.0,
                    "scenario_results": [
                        {
                            "name": "candidate",
                            "description": "Candidate eval passed.",
                            "passed": True,
                            "score": 1.0,
                            "checks": [
                                {
                                    "name": "candidate-check",
                                    "passed": True,
                                    "details": "All checks passed.",
                                }
                            ],
                        }
                    ],
                }
                return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")
            return subprocess.CompletedProcess(argv, 0, "ok", "")

        runner = WorkspacePatchRunner(
            self.store,
            workspace_root=workspace,
            shell_runner=fake_runner,
            git_mode="auto",
        )
        report = runner.run(
            "promote target update",
            operations=[
                PatchOperation(
                    operation="replace_text",
                    path="target.txt",
                    find_text="beta",
                    text="gamma",
                )
            ],
            apply_on_success=True,
        )

        self.assertEqual(report.status, "applied")
        self.assertEqual(target.read_text(encoding="utf-8"), "alpha\ngamma\n")
        self.assertIsNotNone(report.git_apply)
        self.assertEqual(report.git_apply.status, "applied")
        self.assertTrue(str(report.git_apply.branch_name).startswith("codex/patch-run/"))
        self.assertEqual(report.git_apply.original_branch, "main")
        self.assertTrue(report.git_apply.rollback_ready)
        self.assertIn("git checkout main", report.git_apply.rollback_hint)
        latest_patch_run = self.store.latest_patch_run()
        self.assertIsNotNone(latest_patch_run)
        self.assertEqual(latest_patch_run["summary"]["git"]["status"], "applied")
        self.assertEqual(latest_patch_run["summary"]["git"]["original_branch"], "main")
        joined_calls = "\n".join(" ".join(call) for call in calls)
        self.assertIn("git checkout -b", joined_calls)
        self.assertIn("git add -- target.txt", joined_calls)
        self.assertIn("git -c user.name=Codex -c user.email=codex@local.invalid commit --no-verify -m codex: apply patch run promote target update", joined_calls)

    def test_patch_runner_rolls_back_disposable_git_branch(self) -> None:
        self._record_green_baseline()
        workspace = self._make_workspace()
        target = workspace / "target.txt"
        target.write_text("alpha\nbeta\n", encoding="utf-8")
        repo_root = str(workspace.resolve())
        state = {
            "current_branch": "main",
            "branches": {"main": "main-head"},
            "commit_counter": 0,
        }

        def fake_runner(argv, cwd, capture_output, text, timeout, shell):
            self.assertFalse(shell)
            command = [str(part) for part in argv]
            if command[:3] == ["git", "rev-parse", "--show-toplevel"]:
                return subprocess.CompletedProcess(argv, 0, repo_root, "")
            if command[:3] == ["git", "branch", "--show-current"]:
                return subprocess.CompletedProcess(argv, 0, state["current_branch"], "")
            if command[:3] == ["git", "status", "--porcelain"]:
                return subprocess.CompletedProcess(argv, 0, "", "")
            if command[:3] == ["git", "rev-parse", "HEAD"]:
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    state["branches"][state["current_branch"]],
                    "",
                )
            if len(command) == 3 and command[:2] == ["git", "rev-parse"]:
                branch = command[2]
                if branch in state["branches"]:
                    return subprocess.CompletedProcess(argv, 0, state["branches"][branch], "")
                return subprocess.CompletedProcess(argv, 1, "", "unknown branch")
            if command[:3] == ["git", "checkout", "-b"]:
                branch = command[3]
                state["branches"][branch] = state["branches"][state["current_branch"]]
                state["current_branch"] = branch
                return subprocess.CompletedProcess(argv, 0, "", "")
            if command[:2] == ["git", "checkout"]:
                branch = command[2]
                if branch not in state["branches"]:
                    return subprocess.CompletedProcess(argv, 1, "", "unknown branch")
                state["current_branch"] = branch
                return subprocess.CompletedProcess(argv, 0, "", "")
            if command[:2] == ["git", "add"]:
                return subprocess.CompletedProcess(argv, 0, "", "")
            if command[:3] == ["git", "branch", "-D"]:
                branch = command[3]
                if branch in state["branches"] and branch != state["current_branch"]:
                    del state["branches"][branch]
                    return subprocess.CompletedProcess(argv, 0, "", "")
                return subprocess.CompletedProcess(argv, 1, "", "cannot delete")
            if "commit" in command:
                state["commit_counter"] += 1
                state["branches"][state["current_branch"]] = (
                    f"commit-{state['commit_counter']}"
                )
                return subprocess.CompletedProcess(argv, 0, "", "")
            if command[:3] == ["python", "-m", "memory_agent.cli"]:
                payload = {
                    "passed": True,
                    "score": 1.0,
                    "scenario_results": [
                        {
                            "name": "candidate",
                            "description": "Candidate eval passed.",
                            "passed": True,
                            "score": 1.0,
                            "checks": [
                                {
                                    "name": "candidate-check",
                                    "passed": True,
                                    "details": "All checks passed.",
                                }
                            ],
                        }
                    ],
                }
                return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")
            return subprocess.CompletedProcess(argv, 0, "ok", "")

        runner = WorkspacePatchRunner(
            self.store,
            workspace_root=workspace,
            shell_runner=fake_runner,
            git_mode="auto",
        )
        applied = runner.run(
            "promote target update",
            operations=[
                PatchOperation(
                    operation="replace_text",
                    path="target.txt",
                    find_text="beta",
                    text="gamma",
                )
            ],
            apply_on_success=True,
        )

        rollback = runner.rollback(applied.run_id)

        self.assertEqual(rollback.status, "rolled_back")
        self.assertTrue(rollback.deleted_branch)
        self.assertEqual(rollback.switched_to, "main")
        self.assertEqual(state["current_branch"], "main")
        self.assertNotIn(applied.git_apply.branch_name, state["branches"])
        self.assertIsNotNone(rollback.tool_outcome)

    def test_handoff_pack_creates_linux_bundle_with_state(self) -> None:
        workspace = self._make_workspace()
        agent_dir = workspace / ".agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        export_db = agent_dir / "agent_memory.sqlite3"
        traces_dir = agent_dir / "pilot_traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "pilot_policy.toml").write_text(
            "[general]\nname = \"linux-pilot\"\n",
            encoding="utf-8",
        )
        (traces_dir / "trace-1.json").write_text(
            json.dumps({"status": "ok"}, indent=2),
            encoding="utf-8",
        )
        export_store = MemoryStore(export_db)
        try:
            export_store.record_task(
                "Bring up Linux agent",
                status="open",
                area="execution",
            )
            manager = ProjectHandoffManager(export_store, workspace_root=workspace)
            report = manager.create_bundle()
        finally:
            export_store.close()

        bundle_path = Path(report.bundle_path)
        self.assertTrue(bundle_path.exists())
        self.assertTrue(Path(report.manifest_path).exists())
        self.assertTrue(Path(report.summary_path).exists())
        with zipfile.ZipFile(bundle_path) as archive:
            names = set(archive.namelist())
        self.assertIn(".agent/agent_memory.sqlite3", names)
        self.assertIn(".agent/pilot_policy.toml", names)
        self.assertIn(".agent/pilot_traces/trace-1.json", names)
        self.assertIn("_handoff/handoff_manifest.json", names)
        self.assertIn("_handoff/HANDOFF.md", names)

    def test_handoff_restore_restores_bundle_into_target_workspace(self) -> None:
        source_workspace = self._make_workspace()
        source_agent_dir = source_workspace / ".agent"
        source_agent_dir.mkdir(parents=True, exist_ok=True)
        export_db = source_agent_dir / "agent_memory.sqlite3"
        (source_agent_dir / "pilot_policy.toml").write_text(
            "[general]\nname = \"linux-pilot\"\n",
            encoding="utf-8",
        )
        export_store = MemoryStore(export_db)
        try:
            export_store.remember(
                MemoryDraft(
                    kind="constraint",
                    subject="runtime",
                    content="The agent should run locally on the Linux box.",
                )
            )
            bundle_report = ProjectHandoffManager(
                export_store,
                workspace_root=source_workspace,
            ).create_bundle(include_traces=False)
        finally:
            export_store.close()

        target_workspace = self._make_workspace()
        (target_workspace / "pyproject.toml").write_text(
            "[project]\nname = \"target\"\nversion = \"0.0.1\"\n",
            encoding="utf-8",
        )
        restore_report = ProjectHandoffManager(
            self.store,
            workspace_root=target_workspace,
        ).restore_bundle(Path(bundle_report.bundle_path))

        restored_db = target_workspace / ".agent" / "agent_memory.sqlite3"
        self.assertTrue(restored_db.exists())
        self.assertTrue((target_workspace / ".agent" / "pilot_policy.toml").exists())
        self.assertIn(".agent/agent_memory.sqlite3", restore_report.restored_files)
        restored_store = MemoryStore(restored_db)
        try:
            results = restored_store.search("Linux box", limit=3)
        finally:
            restored_store.close()
        self.assertTrue(results)
        self.assertIn("Linux box", results[0].memory.content)

    def test_handoff_restore_manager_does_not_require_memory_store(self) -> None:
        source_workspace = self._make_workspace()
        source_agent_dir = source_workspace / ".agent"
        source_agent_dir.mkdir(parents=True, exist_ok=True)
        export_db = source_agent_dir / "agent_memory.sqlite3"
        export_store = MemoryStore(export_db)
        try:
            export_store.remember(
                MemoryDraft(
                    kind="constraint",
                    subject="runtime",
                    content="The agent should restore without opening sqlite first.",
                )
            )
            bundle_report = ProjectHandoffManager(
                export_store,
                workspace_root=source_workspace,
            ).create_bundle(include_traces=False)
        finally:
            export_store.close()

        target_workspace = self._make_workspace()
        (target_workspace / "pyproject.toml").write_text(
            "[project]\nname = \"target\"\nversion = \"0.0.1\"\n",
            encoding="utf-8",
        )
        restore_report = ProjectHandoffManager(
            workspace_root=target_workspace,
        ).restore_bundle(Path(bundle_report.bundle_path))

        self.assertIn(".agent/agent_memory.sqlite3", restore_report.restored_files)
        restored_store = MemoryStore(target_workspace / ".agent" / "agent_memory.sqlite3")
        try:
            results = restored_store.search("restore without opening sqlite", limit=3)
        finally:
            restored_store.close()
        self.assertTrue(results)

    def test_patch_run_args_accept_utf8_bom_spec_file(self) -> None:
        spec_path = self.temp_root / f"{uuid.uuid4().hex}_patch_spec.json"
        self.extra_paths.append(spec_path)
        spec_path.write_text(
            "\ufeff"
            + json.dumps(
                {
                    "operations": [
                        {
                            "operation": "replace_text",
                            "path": "README.md",
                            "find_text": "Memory-First Agent",
                            "text": "Memory-First Agent",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        operations, validation_commands, apply_on_success, task_title = _resolve_patch_run_args(
            Namespace(
                spec_file=spec_path,
                file_op=None,
                file_path=None,
                file_text=None,
                find_text=None,
                symbol_name=None,
                replace_all=False,
                validate=None,
                apply_on_success=False,
                task_title=None,
            )
        )

        self.assertEqual(len(operations), 1)
        self.assertEqual(operations[0].operation, "replace_text")
        self.assertIsNone(validation_commands)
        self.assertFalse(apply_on_success)
        self.assertIsNone(task_title)

    def test_patch_runner_applies_multi_file_python_symbol_batch(self) -> None:
        self._record_green_baseline()
        workspace = self._make_workspace()
        module_a = workspace / "module_a.py"
        module_b = workspace / "module_b.py"
        module_a.write_text(
            "def greet() -> str:\n"
            "    return 'old-a'\n",
            encoding="utf-8",
        )
        module_b.write_text(
            "class Greeter:\n"
            "    def speak(self) -> str:\n"
            "        return 'old-b'\n",
            encoding="utf-8",
        )

        def fake_runner(argv, cwd, capture_output, text, timeout, shell):
            self.assertFalse(shell)
            if argv[:3] == ["python", "-m", "memory_agent.cli"]:
                payload = {
                    "passed": True,
                    "score": 1.0,
                    "scenario_results": [
                        {
                            "name": "candidate",
                            "description": "Candidate eval passed.",
                            "passed": True,
                            "score": 1.0,
                            "checks": [
                                {
                                    "name": "candidate-check",
                                    "passed": True,
                                    "details": "All checks passed.",
                                }
                            ],
                        }
                    ],
                }
                return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")
            return subprocess.CompletedProcess(argv, 0, "ok", "")

        runner = WorkspacePatchRunner(
            self.store,
            workspace_root=workspace,
            shell_runner=fake_runner,
        )
        report = runner.run(
            "multi file symbol patch",
            operations=[
                PatchOperation(
                    operation="replace_python_function",
                    path="module_a.py",
                    symbol_name="greet",
                    text="def greet() -> str:\n    return 'new-a'\n",
                ),
                PatchOperation(
                    operation="replace_python_class",
                    path="module_b.py",
                    symbol_name="Greeter",
                    text=(
                        "class Greeter:\n"
                        "    def speak(self) -> str:\n"
                        "        return 'new-b'\n"
                    ),
                ),
            ],
            apply_on_success=True,
        )

        self.assertEqual(report.status, "applied")
        self.assertEqual(sorted(report.changed_files), ["module_a.py", "module_b.py"])
        self.assertIn("new-a", module_a.read_text(encoding="utf-8"))
        self.assertIn("new-b", module_b.read_text(encoding="utf-8"))

    def test_patch_runner_applies_python_delete_and_rename_batch(self) -> None:
        self._record_green_baseline()
        workspace = self._make_workspace()
        module = workspace / "module_ops.py"
        module.write_text(
            "def helper() -> str:\n"
            "    return 'helper'\n\n"
            "def greet() -> str:\n"
            "    return helper()\n\n"
            "value = greet()\n",
            encoding="utf-8",
        )

        def fake_runner(argv, cwd, capture_output, text, timeout, shell):
            self.assertFalse(shell)
            if argv[:3] == ["python", "-m", "memory_agent.cli"]:
                payload = {
                    "passed": True,
                    "score": 1.0,
                    "scenario_results": [
                        {
                            "name": "candidate",
                            "description": "Candidate eval passed.",
                            "passed": True,
                            "score": 1.0,
                            "checks": [
                                {
                                    "name": "candidate-check",
                                    "passed": True,
                                    "details": "All checks passed.",
                                }
                            ],
                        }
                    ],
                }
                return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")
            return subprocess.CompletedProcess(argv, 0, "ok", "")

        runner = WorkspacePatchRunner(
            self.store,
            workspace_root=workspace,
            shell_runner=fake_runner,
        )
        report = runner.run(
            "delete and rename python symbols",
            operations=[
                PatchOperation(
                    operation="replace_python_function",
                    path="module_ops.py",
                    symbol_name="greet",
                    text="def greet() -> str:\n    return 'hello'\n",
                ),
                PatchOperation(
                    operation="delete_python_symbol",
                    path="module_ops.py",
                    symbol_name="helper",
                ),
                PatchOperation(
                    operation="rename_python_identifier",
                    path="module_ops.py",
                    symbol_name="greet",
                    text="salute",
                ),
            ],
            apply_on_success=True,
        )

        self.assertEqual(report.status, "applied")
        updated = module.read_text(encoding="utf-8")
        self.assertNotIn("def helper()", updated)
        self.assertIn("def salute()", updated)
        self.assertIn("value = salute()", updated)

    def test_patch_runner_applies_import_and_method_refactor_batch(self) -> None:
        self._record_green_baseline()
        workspace = self._make_workspace()
        module = workspace / "module_refactor.py"
        module.write_text(
            "\"\"\"Refactor target.\"\"\"\n\n"
            "from __future__ import annotations\n\n"
            "class Greeter:\n"
            "    def speak(self) -> str:\n"
            "        return 'hello'\n\n"
            "greeter = Greeter()\n"
            "VALUE = greeter.speak()\n",
            encoding="utf-8",
        )

        def fake_runner(argv, cwd, capture_output, text, timeout, shell):
            self.assertFalse(shell)
            if argv[:3] == ["python", "-m", "memory_agent.cli"]:
                payload = {
                    "passed": True,
                    "score": 1.0,
                    "scenario_results": [
                        {
                            "name": "candidate",
                            "description": "Candidate eval passed.",
                            "passed": True,
                            "score": 1.0,
                            "checks": [
                                {
                                    "name": "candidate-check",
                                    "passed": True,
                                    "details": "All checks passed.",
                                }
                            ],
                        }
                    ],
                }
                return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")
            return subprocess.CompletedProcess(argv, 0, "ok", "")

        runner = WorkspacePatchRunner(
            self.store,
            workspace_root=workspace,
            shell_runner=fake_runner,
        )
        report = runner.run(
            "import and method refactor batch",
            operations=[
                PatchOperation(
                    operation="add_python_import",
                    path="module_refactor.py",
                    text="from pathlib import Path",
                ),
                PatchOperation(
                    operation="rename_python_method",
                    path="module_refactor.py",
                    symbol_name="Greeter.speak",
                    text="salute",
                ),
            ],
            apply_on_success=True,
        )

        self.assertEqual(report.status, "applied")
        updated = module.read_text(encoding="utf-8")
        self.assertIn("from pathlib import Path", updated)
        self.assertIn("def salute(self)", updated)
        self.assertIn("VALUE = greeter.salute()", updated)

    def test_patch_runner_applies_signature_refactor_batch(self) -> None:
        self._record_green_baseline()
        workspace = self._make_workspace()
        module = workspace / "module_signature.py"
        module.write_text(
            "def greet(name: str) -> str:\n"
            "    return name.upper()\n\n"
            "value = greet('sam')\n",
            encoding="utf-8",
        )

        def fake_runner(argv, cwd, capture_output, text, timeout, shell):
            self.assertFalse(shell)
            if argv[:3] == ["python", "-m", "memory_agent.cli"]:
                payload = {
                    "passed": True,
                    "score": 1.0,
                    "scenario_results": [
                        {
                            "name": "candidate",
                            "description": "Candidate eval passed.",
                            "passed": True,
                            "score": 1.0,
                            "checks": [
                                {
                                    "name": "candidate-check",
                                    "passed": True,
                                    "details": "All checks passed.",
                                }
                            ],
                        }
                    ],
                }
                return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")
            return subprocess.CompletedProcess(argv, 0, "ok", "")

        runner = WorkspacePatchRunner(
            self.store,
            workspace_root=workspace,
            shell_runner=fake_runner,
        )
        report = runner.run(
            "signature refactor batch",
            operations=[
                PatchOperation(
                    operation="add_python_function_parameter",
                    path="module_signature.py",
                    symbol_name="greet",
                    text=json.dumps(
                        {
                            "parameter_name": "excited",
                            "call_argument": "True",
                        }
                    ),
                )
            ],
            apply_on_success=True,
        )

        self.assertEqual(report.status, "applied")
        updated = module.read_text(encoding="utf-8")
        self.assertIn("def greet(name: str, excited)", updated)
        self.assertIn("value = greet('sam', excited=True)", updated)

    def test_patch_runner_applies_cross_import_rename_and_reports_multi_file_changes(self) -> None:
        self._record_green_baseline()
        workspace = self._make_workspace()
        package = workspace / "pkgdemo"
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        source = package / "greeter.py"
        consumer = workspace / "consumer.py"
        module_consumer = workspace / "module_consumer.py"
        source.write_text(
            "def greet(name: str) -> str:\n"
            "    return name.upper()\n",
            encoding="utf-8",
        )
        consumer.write_text(
            "from pkgdemo.greeter import greet\n"
            "result = greet('zoe')\n",
            encoding="utf-8",
        )
        module_consumer.write_text(
            "import pkgdemo.greeter as greeter_mod\n"
            "other = greeter_mod.greet('amy')\n",
            encoding="utf-8",
        )

        def fake_runner(argv, cwd, capture_output, text, timeout, shell):
            self.assertFalse(shell)
            if argv[:3] == ["python", "-m", "memory_agent.cli"]:
                payload = {
                    "passed": True,
                    "score": 1.0,
                    "scenario_results": [
                        {
                            "name": "candidate",
                            "description": "Candidate eval passed.",
                            "passed": True,
                            "score": 1.0,
                            "checks": [
                                {
                                    "name": "candidate-check",
                                    "passed": True,
                                    "details": "All checks passed.",
                                }
                            ],
                        }
                    ],
                }
                return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")
            return subprocess.CompletedProcess(argv, 0, "ok", "")

        runner = WorkspacePatchRunner(
            self.store,
            workspace_root=workspace,
            shell_runner=fake_runner,
        )
        report = runner.run(
            "cross import rename",
            operations=[
                PatchOperation(
                    operation="rename_python_export_across_imports",
                    path="pkgdemo/greeter.py",
                    symbol_name="greet",
                    text="salute",
                )
            ],
            apply_on_success=True,
        )

        self.assertEqual(report.status, "applied")
        self.assertEqual(
            sorted(report.changed_files),
            ["consumer.py", "module_consumer.py", "pkgdemo/greeter.py"],
        )
        self.assertIn("def salute(name: str)", source.read_text(encoding="utf-8"))
        self.assertIn("import salute", consumer.read_text(encoding="utf-8"))
        self.assertIn("greeter_mod.salute('amy')", module_consumer.read_text(encoding="utf-8"))

    def test_patch_runner_moves_python_export_to_module_and_reports_changed_files(self) -> None:
        self._record_green_baseline()
        workspace = self._make_workspace()
        package = workspace / "pkgmove"
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        source = package / "legacy.py"
        destination = package / "core.py"
        consumer = workspace / "consumer.py"
        source.write_text(
            "from pathlib import Path\n\n"
            "def greet(name: str) -> str:\n"
            "    return Path(name).name.upper()\n",
            encoding="utf-8",
        )
        consumer.write_text(
            "from pkgmove.legacy import greet\n"
            "value = greet('sam.txt')\n",
            encoding="utf-8",
        )

        def fake_runner(argv, cwd, capture_output, text, timeout, shell):
            self.assertFalse(shell)
            if argv[:3] == ["python", "-m", "memory_agent.cli"]:
                payload = {
                    "passed": True,
                    "score": 1.0,
                    "scenario_results": [
                        {
                            "name": "candidate",
                            "description": "Candidate eval passed.",
                            "passed": True,
                            "score": 1.0,
                            "checks": [
                                {
                                    "name": "candidate-check",
                                    "passed": True,
                                    "details": "All checks passed.",
                                }
                            ],
                        }
                    ],
                }
                return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")
            return subprocess.CompletedProcess(argv, 0, "ok", "")

        runner = WorkspacePatchRunner(
            self.store,
            workspace_root=workspace,
            shell_runner=fake_runner,
        )
        report = runner.run(
            "move export to module",
            operations=[
                PatchOperation(
                    operation="move_python_export_to_module",
                    path="pkgmove/legacy.py",
                    symbol_name="greet",
                    text=json.dumps(
                        {
                            "destination_path": "pkgmove/core.py",
                        }
                    ),
                )
            ],
            apply_on_success=True,
        )

        self.assertEqual(report.status, "applied")
        self.assertEqual(
            sorted(report.changed_files),
            ["consumer.py", "pkgmove/core.py", "pkgmove/legacy.py"],
        )
        self.assertIn("from pathlib import Path", destination.read_text(encoding="utf-8"))
        self.assertIn("def greet(name: str)", destination.read_text(encoding="utf-8"))
        self.assertIn("from pkgmove.core import greet", source.read_text(encoding="utf-8"))
        self.assertIn("from pkgmove.core import greet", consumer.read_text(encoding="utf-8"))

    def test_patch_runner_rejects_regressing_candidate_without_touching_workspace(self) -> None:
        self._record_green_baseline()
        workspace = self._make_workspace()
        target = workspace / "target.txt"
        target.write_text("alpha\nbeta\n", encoding="utf-8")

        def fake_runner(argv, cwd, capture_output, text, timeout, shell):
            self.assertFalse(shell)
            if argv[:3] == ["python", "-m", "memory_agent.cli"]:
                payload = {
                    "passed": False,
                    "score": 0.5,
                    "scenario_results": [
                        {
                            "name": "candidate",
                            "description": "Candidate eval failed.",
                            "passed": False,
                            "score": 0.0,
                            "checks": [
                                {
                                    "name": "candidate-check",
                                    "passed": False,
                                    "details": "Regression detected.",
                                }
                            ],
                        }
                    ],
                }
                return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")
            return subprocess.CompletedProcess(argv, 0, "ok", "")

        runner = WorkspacePatchRunner(
            self.store,
            workspace_root=workspace,
            shell_runner=fake_runner,
        )
        report = runner.run(
            "reject target update",
            operations=[
                PatchOperation(
                    operation="replace_text",
                    path="target.txt",
                    find_text="beta",
                    text="gamma",
                )
            ],
            apply_on_success=True,
        )

        self.assertEqual(report.status, "rejected")
        self.assertFalse(report.applied)
        self.assertEqual(target.read_text(encoding="utf-8"), "alpha\nbeta\n")
        self.assertIsNotNone(report.candidate_evaluation)
        self.assertEqual(report.candidate_evaluation["score"], 0.5)
        self.assertEqual(report.validations[-1].kind, "evaluation")
        latest_patch_run = self.store.latest_patch_run()
        self.assertIsNotNone(latest_patch_run)
        self.assertEqual(latest_patch_run["status"], "rejected")
        self.assertEqual(latest_patch_run["candidate_score"], 0.5)

    def test_patch_runner_rejects_invalid_python_symbol_patch_before_validation(self) -> None:
        self._record_green_baseline()
        workspace = self._make_workspace()
        target = workspace / "module_bad.py"
        target.write_text(
            "def greet() -> str:\n"
            "    return 'hello'\n",
            encoding="utf-8",
        )

        runner = WorkspacePatchRunner(
            self.store,
            workspace_root=workspace,
        )
        report = runner.run(
            "reject invalid symbol patch",
            operations=[
                PatchOperation(
                    operation="replace_python_function",
                    path="module_bad.py",
                    symbol_name="greet",
                    text="def greet(\n    return 'broken'\n",
                )
            ],
            apply_on_success=True,
        )

        self.assertEqual(report.status, "rejected")
        self.assertIn("updated_python_parse_error", report.rejection_reason)
        self.assertEqual(report.validations, [])
        self.assertEqual(
            target.read_text(encoding="utf-8"),
            "def greet() -> str:\n    return 'hello'\n",
        )

    def test_patch_runner_default_validation_commands_target_explicit_test_modules(self) -> None:
        runner = WorkspacePatchRunner(
            self.store,
            workspace_root=Path.cwd(),
        )

        commands = runner._default_validation_commands(Path.cwd())

        self.assertEqual(len(commands), 1)
        self.assertIn("python3 -m unittest -v", commands[0])
        self.assertIn("tests.test_memory", commands[0])
        self.assertNotIn("discover -s tests -v", commands[0])

    def test_improvement_engine_runs_patch_candidate_and_completes_task(self) -> None:
        self._record_green_baseline()
        workspace = self._make_workspace()
        target = workspace / "target.txt"
        target.write_text("alpha\nbeta\n", encoding="utf-8")
        self.store.record_task(
            "Apply code-aware patching prototype",
            status="open",
            area="self_improvement",
        )

        def fake_runner(argv, cwd, capture_output, text, timeout, shell):
            self.assertFalse(shell)
            if argv[:3] == ["python", "-m", "memory_agent.cli"]:
                payload = {
                    "passed": True,
                    "score": 1.0,
                    "scenario_results": [
                        {
                            "name": "candidate",
                            "description": "Candidate eval passed.",
                            "passed": True,
                            "score": 1.0,
                            "checks": [
                                {
                                    "name": "candidate-check",
                                    "passed": True,
                                    "details": "All checks passed.",
                                }
                            ],
                        }
                    ],
                }
                return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")
            return subprocess.CompletedProcess(argv, 0, "ok", "")

        engine = MemoryImprovementEngine(self.store, evaluator=object())
        report = engine.run_patch_candidate(
            "code-aware patching prototype",
            operations=[
                PatchOperation(
                    operation="replace_text",
                    path="target.txt",
                    find_text="beta",
                    text="delta",
                )
            ],
            apply_on_success=True,
            task_title="Apply code-aware patching prototype",
            patch_runner=WorkspacePatchRunner(
                self.store,
                workspace_root=workspace,
                shell_runner=fake_runner,
            ),
        )

        self.assertEqual(report.status, "applied")
        self.assertIsNotNone(report.task_update)
        self.assertEqual(report.task_update.metadata.get("status"), "done")
        self.assertIsNone(
            self.store.find_active_task(
                "Apply code-aware patching prototype",
                area="self_improvement",
            )
        )
        self.assertEqual(target.read_text(encoding="utf-8"), "alpha\ndelta\n")

    def test_agent_respond_uses_model_with_memory_and_plan_context(self) -> None:
        self.store.record_task(
            "Build task graph maintenance",
            status="blocked",
            area="execution",
            blocked_by=["Finish entity resolution"],
            due_date="2026-04-02",
        )

        captured_messages = []

        class FakeModelAdapter:
            @property
            def enabled(self):
                return True

            def status(self):
                return {
                    "enabled": True,
                    "backend": "fake",
                    "model": "fake-main",
                }

            def chat(self, messages):
                captured_messages.extend(messages)
                return ModelResponse(
                    content="We should clear the blocker on Build task graph maintenance first.",
                    model="fake-main",
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            soul_path = Path(tmpdir) / "SOUL.md"
            soul_path.write_text(
                "# Test Soul\n\n- Teach gently.\n- Protect operator trust.\n",
                encoding="utf-8",
            )
            agent = MemoryFirstAgent(
                self.store,
                model_adapter=FakeModelAdapter(),
                workspace_root=Path(tmpdir),
            )
            report = agent.respond("What should I do next?")

            self.assertIsNone(report.error)
            self.assertEqual(report.assistant_message, "We should clear the blocker on Build task graph maintenance first.")
            self.assertIsNotNone(report.assistant_event_id)
            self.assertEqual(len(captured_messages), 2)
            self.assertEqual(captured_messages[0].role, "system")
            self.assertIn("Build task graph maintenance", captured_messages[0].content)
            self.assertIn("Planner state:", captured_messages[0].content)
            self.assertIn("Ernie SOUL:", captured_messages[0].content)
            self.assertIn("Teach gently.", captured_messages[0].content)
        events = self.store.recent_events(limit=2)
        self.assertEqual(events[-1].role, "assistant")
        self.assertIn("clear the blocker", events[-1].content.lower())

    def test_agent_decide_executes_valid_structured_plan_option(self) -> None:
        self.store.record_task(
            "Ship overdue ready memory loop",
            status="open",
            area="execution",
            due_date="2000-01-01",
        )
        captured_messages = []

        class FakeModelAdapter:
            @property
            def enabled(self):
                return True

            def status(self):
                return {
                    "enabled": True,
                    "backend": "fake",
                    "model": "fake-main",
                }

            def chat(self, messages):
                captured_messages.extend(messages)
                return ModelResponse(
                    content=(
                        '{"assistant_message":"I am starting the overdue ready task now.",'
                        '"action":{"type":"execute_plan_action","option_id":"A1",'
                        '"rationale":"The top recommendation clearly fits the request."}}'
                    ),
                    model="fake-main",
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            soul_path = Path(tmpdir) / "SOUL.md"
            soul_path.write_text(
                "# Test Soul\n\n- Prefer bounded progress.\n- Say what is unknown.\n",
                encoding="utf-8",
            )
            agent = MemoryFirstAgent(
                self.store,
                model_adapter=FakeModelAdapter(),
                workspace_root=Path(tmpdir),
            )
            report = agent.decide("What should I do next?")

            self.assertIsNone(report.error)
            self.assertIsNotNone(report.model_action)
            self.assertEqual(report.model_action.action_type, "execute_plan_action")
            self.assertIsNotNone(report.model_action.chosen_option)
            self.assertEqual(report.model_action.chosen_option.option_id, "A1")
            self.assertIsNotNone(report.execution_result)
            self.assertEqual(report.execution_result.executed_kind, "work_task")
            self.assertEqual(report.execution_result.status, "success")
            self.assertEqual(report.execution_result.task_update.metadata.get("status"), "in_progress")
            self.assertEqual(report.assistant_message, "I am starting the overdue ready task now.")
            self.assertIsNotNone(report.after_plan)
            self.assertEqual(len(captured_messages), 2)
            self.assertIn("Ernie SOUL:", captured_messages[0].content)
            self.assertIn("Prefer bounded progress.", captured_messages[0].content)
        events = self.store.recent_events(limit=2)
        self.assertEqual(events[-1].role, "assistant")
        self.assertIn("starting the overdue ready task", events[-1].content.lower())

    def test_agent_decide_falls_back_to_reply_when_option_id_is_invalid(self) -> None:
        self.store.record_task(
            "Ship overdue ready memory loop",
            status="open",
            area="execution",
            due_date="2000-01-01",
        )

        class FakeModelAdapter:
            @property
            def enabled(self):
                return True

            def status(self):
                return {
                    "enabled": True,
                    "backend": "fake",
                    "model": "fake-main",
                }

            def chat(self, messages):
                return ModelResponse(
                    content=(
                        '{"assistant_message":"The action proposal was invalid, so here is the'
                        ' status update instead.","action":{"type":"execute_plan_action",'
                        '"option_id":"A99"}}'
                    ),
                    model="fake-main",
                )

        agent = MemoryFirstAgent(self.store, model_adapter=FakeModelAdapter())
        report = agent.decide("What should I do next?")

        self.assertIsNone(report.error)
        self.assertIsNotNone(report.model_action)
        self.assertEqual(report.model_action.action_type, "reply_only")
        self.assertTrue(report.model_action.fallback_to_reply)
        self.assertEqual(report.model_action.validation_error, "unknown_option_id=A99")
        self.assertIsNone(report.execution_result)
        self.assertEqual(
            report.assistant_message,
            "The action proposal was invalid, so here is the status update instead.",
        )
        events = self.store.recent_events(limit=2)
        self.assertEqual(events[-1].role, "assistant")
        self.assertIn("status update instead", events[-1].content.lower())

    def test_agent_explain_plan_uses_model_with_planner_context(self) -> None:
        self.store.record_task(
            "Explain current recommendation",
            status="open",
            area="execution",
            due_date="2000-01-01",
        )
        captured_messages = []

        class FakeModelAdapter:
            @property
            def enabled(self):
                return True

            def status(self):
                return {
                    "enabled": True,
                    "backend": "fake",
                    "model": "fake-main",
                }

            def chat(self, messages):
                captured_messages.extend(messages)
                return ModelResponse(
                    content="This is next because it is the most bounded overdue step and there is still a safe alternative if you want a different path.",
                    model="fake-main",
                )

        agent = MemoryFirstAgent(self.store, model_adapter=FakeModelAdapter())
        report = agent.explain_plan("What should I do next?")

        self.assertTrue(report.used_model)
        self.assertIn("bounded overdue step", report.text)
        self.assertEqual(len(captured_messages), 2)
        self.assertIn("Explain the current planner recommendation", captured_messages[0].content)
        self.assertIn("Planner state:", captured_messages[0].content)

    def test_agent_narrate_execution_uses_model_with_execution_result(self) -> None:
        self.store.record_task(
            "Narrate the executed step",
            status="open",
            area="execution",
            due_date="2000-01-01",
        )
        captured_messages = []

        class FakeModelAdapter:
            @property
            def enabled(self):
                return True

            def status(self):
                return {
                    "enabled": True,
                    "backend": "fake",
                    "model": "fake-main",
                }

            def chat(self, messages):
                captured_messages.extend(messages)
                return ModelResponse(
                    content="The step moved the task into active work and the next bounded recommendation is to inspect the refreshed queue.",
                    model="fake-main",
                )

        agent = MemoryFirstAgent(self.store, model_adapter=FakeModelAdapter())
        cycle = agent.execute_next("Narrate the executed step", action_limit=3)
        report = agent.narrate_execution(
            query="Narrate the executed step",
            before_plan=cycle.before_plan,
            result=cycle.result,
            after_plan=cycle.after_plan,
        )

        self.assertTrue(report.used_model)
        self.assertIn("next bounded recommendation", report.text)
        self.assertEqual(len(captured_messages), 2)
        self.assertIn("Summarize what changed after a bounded execution step", captured_messages[0].content)
        self.assertIn("Execution result:", captured_messages[0].content)

    def test_agent_prompt_workshop_uses_model_without_storing_events(self) -> None:
        baseline_events = self.store.stats()["events"]
        captured_messages = []

        class FakeModelAdapter:
            @property
            def enabled(self):
                return True

            def status(self):
                return {
                    "enabled": True,
                    "backend": "fake",
                    "model": "fake-main",
                }

            def chat(self, messages):
                captured_messages.extend(messages)
                return ModelResponse(
                    content="Rewrite the prompt so it names the goal, the constraints, and the desired output in one bounded request.",
                    model="fake-main",
                )

        agent = MemoryFirstAgent(self.store, model_adapter=FakeModelAdapter())
        report = agent.workshop_prompt("help me make this better", mode="clarify")

        self.assertTrue(report.used_model)
        self.assertIn("names the goal", report.text)
        self.assertEqual(self.store.stats()["events"], baseline_events)
        self.assertEqual(len(captured_messages), 2)
        self.assertIn("quarantined text", captured_messages[0].content)

    def test_ollama_chat_adapter_parses_response_payload(self) -> None:
        payloads = []

        def fake_fetch(payload):
            payloads.append(payload)
            return {
                "model": "qwen3:14b",
                "message": {
                    "role": "assistant",
                    "content": "Use SQLite as the source of truth.",
                },
                "done_reason": "stop",
                "prompt_eval_count": 123,
                "eval_count": 45,
            }

        adapter = OllamaChatAdapter(
            model="qwen3:14b",
            fetch_response=fake_fetch,
        )
        response = adapter.chat(
            [
                type("Msg", (), {"role": "system", "content": "Be helpful."})(),
                type("Msg", (), {"role": "user", "content": "What storage should we use?"})(),
            ]
        )

        self.assertEqual(response.content, "Use SQLite as the source of truth.")
        self.assertEqual(response.model, "qwen3:14b")
        self.assertEqual(response.done_reason, "stop")
        self.assertEqual(response.prompt_eval_count, 123)
        self.assertEqual(response.eval_count, 45)
        self.assertEqual(payloads[0]["model"], "qwen3:14b")
        self.assertFalse(payloads[0]["stream"])

    def test_reflect_recent_creates_source_linked_summary(self) -> None:
        cost = self.store.remember(
            MemoryDraft(
                kind="preference",
                subject="optimization",
                content="Optimize the agent for low ongoing cost.",
                tags=["cost", "optimization"],
                importance=0.95,
                confidence=0.9,
            )
        )
        speed = self.store.remember(
            MemoryDraft(
                kind="preference",
                subject="optimization",
                content="Optimize the agent for efficient, low-latency operation.",
                tags=["performance", "optimization"],
                importance=0.92,
                confidence=0.9,
            )
        )
        reflections = self.store.reflect_recent(limit=10, max_reflections=3)
        optimization_reflection = next(
            reflection for reflection in reflections if reflection.subject == "optimization"
        )
        self.assertEqual(optimization_reflection.layer, "reflection")
        sources = self.store.get_memory_sources(optimization_reflection.id)
        source_memory_ids = {
            source.source_id for source in sources if source.source_type == "memory"
        }
        self.assertTrue({cost.id, speed.id}.issubset(source_memory_ids))

    def test_revise_memory_supersedes_old_version(self) -> None:
        original = self.store.remember(
            MemoryDraft(
                kind="constraint",
                subject="runtime",
                content="The agent should run locally on the user's main PC.",
                tags=["runtime", "local"],
                importance=0.95,
                confidence=0.95,
            )
        )
        revised = self.store.revise_memory(
            original.id,
            "The agent should run locally on the user's main PC and laptop.",
        )
        original_after = self.store.get_memory(original.id)
        self.assertIsNotNone(original_after.archived_at)
        edges = self.store.get_memory_edges(revised.id, direction="outgoing")
        self.assertTrue(
            any(edge.edge_type == "supersedes" and edge.to_memory_id == original.id for edge in edges)
        )
        results = self.store.search("laptop", limit=3)
        self.assertEqual(results[0].memory.id, revised.id)

    def test_contradiction_detection_links_conflicting_memories(self) -> None:
        local = self.store.remember(
            MemoryDraft(
                kind="constraint",
                subject="runtime",
                content="The agent should run locally on the user's main PC.",
                tags=["runtime", "local"],
                importance=0.95,
                confidence=0.95,
            )
        )
        cloud = self.store.remember(
            MemoryDraft(
                kind="constraint",
                subject="runtime",
                content="The agent should run in the cloud instead of locally.",
                tags=["runtime", "cloud"],
                importance=0.9,
                confidence=0.9,
            )
        )
        edges = self.store.get_memory_edges(cloud.id, direction="outgoing")
        self.assertTrue(
            any(edge.edge_type == "contradicts" and edge.to_memory_id == local.id for edge in edges)
        )
        context = self.store.build_context("cloud runtime", memory_limit=3, recent_event_count=1)
        self.assertTrue(any(bundle.contradictions for bundle in context.bundles))

    def test_maintenance_runs_due_tasks_and_tracks_state(self) -> None:
        self.store.remember(
            MemoryDraft(
                kind="preference",
                subject="optimization",
                content="Optimize the agent for low ongoing cost.",
                tags=["cost", "optimization"],
                importance=0.95,
                confidence=0.9,
            )
        )
        self.store.remember(
            MemoryDraft(
                kind="preference",
                subject="optimization",
                content="Optimize the agent for efficient, low-latency operation.",
                tags=["performance", "optimization"],
                importance=0.92,
                confidence=0.9,
            )
        )
        report = self.store.run_maintenance()
        self.assertIn("contradiction_scan", report["executed"])
        self.assertIn("reflection", report["executed"])
        self.assertIn("profile", report["executed"])
        stats = self.store.stats()
        self.assertIn("maintenance", stats)
        self.assertFalse(stats["maintenance"]["reflection"]["due"])

    def test_sync_service_tasks_creates_dedupes_and_resolves_recommended_actions(self) -> None:
        initial = self.store.sync_service_tasks(
            {
                "onboarding": {
                    "actions": [
                        {
                            "action": "install_remote_service",
                            "label": "Install remote service",
                            "description": "Install or repair the managed remote cockpit service.",
                            "enabled": True,
                            "requires_confirmation": True,
                            "confirmation_message": "Install or repair remote access for this machine?",
                            "success_message": "Managed remote service installed or repaired.",
                        }
                    ]
                }
            }
        )

        self.assertEqual(initial["recommended_actions"], ["install_remote_service"])
        self.assertEqual(len(initial["created"]), 1)
        synced_task = self.store.find_active_task(
            "Cockpit setup: Install remote service",
            area="execution",
        )
        self.assertIsNotNone(synced_task)
        self.assertEqual(synced_task.metadata.get("service_action"), "install_remote_service")
        self.assertEqual(synced_task.metadata.get("service_label"), "Install remote service")
        self.assertTrue(synced_task.metadata.get("service_requires_confirmation"))
        self.assertIn(
            "Install or repair remote access for this machine?",
            str(synced_task.metadata.get("service_confirmation_message") or ""),
        )
        self.assertIn("service-sync", synced_task.tags)

        repeated = self.store.sync_service_tasks(
            {
                "onboarding": {
                    "actions": [
                        {
                            "action": "install_remote_service",
                            "label": "Install remote service",
                            "description": "Install or repair the managed remote cockpit service.",
                            "enabled": True,
                            "requires_confirmation": True,
                            "confirmation_message": "Install or repair remote access for this machine?",
                            "success_message": "Managed remote service installed or repaired.",
                        }
                    ]
                }
            }
        )

        self.assertFalse(repeated["created"])
        self.assertEqual(len(repeated["unchanged"]), 1)
        reviewable_titles = [
            str(task.metadata.get("title"))
            for task in self.store.get_open_tasks(limit=10)
        ]
        self.assertEqual(reviewable_titles.count("Cockpit setup: Install remote service"), 1)

        resolved = self.store.sync_service_tasks({"onboarding": {"actions": []}})

        self.assertEqual(len(resolved["resolved"]), 1)
        latest = self.store._find_active_task_any_area("Cockpit setup: Install remote service")
        self.assertIsNotNone(latest)
        self.assertEqual(latest.metadata.get("status"), "done")

    def test_optional_semantic_reranker_can_reorder_candidates(self) -> None:
        self.store.remember(
            MemoryDraft(
                kind="constraint",
                subject="runtime",
                content="Local desktop agent for offline work.",
                tags=["runtime", "local"],
                importance=0.85,
                confidence=0.9,
            )
        )
        self.store.remember(
            MemoryDraft(
                kind="constraint",
                subject="runtime",
                content="Cloud server agent for distributed work.",
                tags=["runtime", "cloud"],
                importance=0.9,
                confidence=0.9,
            )
        )

        def fake_fetch(texts: list[str]) -> list[list[float]]:
            embeddings: list[list[float]] = []
            for text in texts:
                lowered = text.lower()
                if lowered == "agent":
                    embeddings.append([1.0, 0.0])
                elif "desktop" in lowered or "offline" in lowered:
                    embeddings.append([1.0, 0.0])
                elif "cloud" in lowered or "server" in lowered:
                    embeddings.append([0.0, 1.0])
                else:
                    embeddings.append([0.5, 0.5])
            return embeddings

        self.store.reranker = OptionalSemanticReranker(
            self.store.connection,
            model="fake-embed",
            fetch_embeddings=fake_fetch,
        )
        results = self.store.search("agent", limit=2)
        self.assertEqual(results[0].memory.content, "Local desktop agent for offline work.")
        self.assertTrue(any(reason.startswith("semantic=") for reason in results[0].reasons))
        status = self.store.reranker.status()
        self.assertTrue(status["enabled"])
        self.assertEqual(status["cached_vectors"], 3)

    def test_domain_aware_profile_summarizes_decisions_tasks_and_tools(self) -> None:
        self.store.record_decision(
            "architecture",
            "Use SQLite as the source of truth for memory storage",
            rationale="It is cheap and local-first",
        )
        self.store.record_task(
            "Add contradiction handling",
            status="open",
            area="architecture",
        )
        self.store.record_tool_outcome(
            "tests",
            "All verification checks passed",
            subject="architecture",
        )
        profiles = self.store.synthesize_profiles(limit=20, max_profiles=5)
        architecture_profile = next(
            profile for profile in profiles if profile.subject == "architecture"
        )
        self.assertIn("decisions:", architecture_profile.content.lower())
        self.assertIn("tasks:", architecture_profile.content.lower())
        self.assertIn("tool outcomes:", architecture_profile.content.lower())

    def test_builtin_evaluation_suite_passes(self) -> None:
        report = MemoryEvaluator(Path.cwd()).run_builtin_suite()
        self.assertTrue(report.passed)
        self.assertGreaterEqual(report.score, 0.99)

    def test_cockpit_service_snapshot_and_context_surface_core_state(self) -> None:
        self.store.record_task(
            "Ship cockpit API",
            status="open",
            area="execution",
            due_date="2026-04-10",
        )
        service = CockpitService(self.store, workspace_root=Path.cwd())

        snapshot = service.snapshot(query="cockpit", limit=5)
        context = service.context(query="cockpit", limit=5)

        self.assertIn("stats", snapshot)
        self.assertIn("plan", snapshot)
        self.assertIn("ready_tasks", snapshot)
        self.assertIn("open_tasks", snapshot)
        self.assertIn("recent_nudges", snapshot)
        ready_titles = [
            str(task["metadata"].get("title") or task["content"])
            for task in snapshot["ready_tasks"]
        ]
        self.assertIn("Ship cockpit API", ready_titles)
        self.assertEqual(context["query"], "cockpit")
        self.assertIn("ready_tasks", context)

    def test_cockpit_service_execute_plan_action_includes_model_explanation(self) -> None:
        self.store.record_task(
            "Ship local cockpit",
            status="open",
            area="execution",
            due_date="2000-01-01",
        )
        service = CockpitService(self.store, workspace_root=Path.cwd())

        class FakeModelAdapter:
            @property
            def enabled(self):
                return True

            def status(self):
                return {
                    "enabled": True,
                    "backend": "fake",
                    "model": "fake-main",
                }

            def chat(self, messages):
                return ModelResponse(
                    content="The step ran, updated the task state, and left a refreshed bounded recommendation behind it.",
                    model="fake-main",
                )

        service.agent.model_adapter = FakeModelAdapter()
        payload = service.execute_plan_action(
            query="ship local cockpit",
            kind="work_task",
            title="Ship local cockpit",
            limit=3,
        )

        self.assertIn("model_explanation", payload)
        self.assertTrue(payload["model_explanation"]["used_model"])
        self.assertIn("refreshed bounded recommendation", payload["model_explanation"]["text"])

    def test_cockpit_service_prompt_workshop_is_quarantined(self) -> None:
        baseline_events = self.store.stats()["events"]
        service = CockpitService(self.store, workspace_root=Path.cwd())

        class FakeModelAdapter:
            @property
            def enabled(self):
                return True

            def status(self):
                return {
                    "enabled": True,
                    "backend": "fake",
                    "model": "fake-main",
                }

            def chat(self, messages):
                return ModelResponse(
                    content="Draft workshop result: tighten the prompt and separate execution from exploration.",
                    model="fake-main",
                )

        service.agent.model_adapter = FakeModelAdapter()
        payload = service.prompt_workshop(
            draft="help me tighten this prompt",
            mode="improve",
        )

        self.assertTrue(payload["used_model"])
        self.assertIn("Draft workshop result", payload["text"])
        self.assertEqual(self.store.stats()["events"], baseline_events)

    def test_cockpit_service_prompt_promotion_records_live_input(self) -> None:
        baseline_events = self.store.stats()["events"]
        service = CockpitService(self.store, workspace_root=Path.cwd())

        payload = service.promote_prompt_text(
            text="Use a calm tone and keep the answer bounded.",
            source="result",
        )

        self.assertEqual(payload["prompt_source"], "result")
        self.assertEqual(self.store.stats()["events"], baseline_events + 1)
        self.assertIsNotNone(payload["plan"])
        latest_activity = self.store.get_recent_tool_outcomes(limit=1)[0]
        self.assertEqual(latest_activity.metadata.get("tool_name"), "prompt-promote")
        self.assertEqual(latest_activity.metadata.get("prompt_source"), "result")

    def test_cockpit_service_send_prompt_to_pilot_queues_guided_review(self) -> None:
        workspace = self._make_workspace()
        target = workspace / "pilot_workshop.txt"
        target.write_text("alpha\nbeta\n", encoding="utf-8")
        self.store.record_task(
            "Pilot workshop write",
            status="open",
            area="execution",
            file_operation="replace_text",
            file_path=str(target.relative_to(workspace)),
            find_text="beta",
            file_text="gamma",
            complete_on_success=True,
        )
        baseline_events = self.store.stats()["events"]
        service = CockpitService(self.store, workspace_root=workspace)

        payload = service.send_prompt_to_pilot(
            text="pilot workshop write",
            source="result",
            action_limit=3,
            use_model=False,
        )

        self.assertEqual(payload["prompt_source"], "result")
        self.assertEqual(payload["request_origin"], "prompt_workshop")
        self.assertEqual(payload["approval"]["status"], "needs_approval")
        self.assertIsNotNone(payload["pending_id"])
        queue = service.pending_pilot_queue()
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["request_origin"], "prompt_workshop")
        self.assertEqual(queue[0]["prompt_source"], "result")
        self.assertEqual(self.store.stats()["events"], baseline_events + 1)
        latest_activity = self.store.get_recent_tool_outcomes(limit=1)[0]
        self.assertEqual(latest_activity.metadata.get("tool_name"), "prompt-pilot-bridge")
        self.assertEqual(latest_activity.metadata.get("prompt_source"), "result")

    def test_cockpit_service_approve_workshop_prompt_pilot_records_execution_trail(self) -> None:
        workspace = self._make_workspace()
        target = workspace / "pilot_workshop_approve.txt"
        target.write_text("alpha\nbeta\n", encoding="utf-8")
        self.store.record_task(
            "Pilot workshop approve write",
            status="open",
            area="execution",
            file_operation="replace_text",
            file_path=str(target.relative_to(workspace)),
            find_text="beta",
            file_text="gamma",
            complete_on_success=True,
        )
        service = CockpitService(self.store, workspace_root=workspace)

        preview = service.send_prompt_to_pilot(
            text="pilot workshop approve write",
            source="draft",
            action_limit=3,
            use_model=False,
        )
        approved = service.approve_pilot_turn(pending_id=preview["pending_id"])

        self.assertEqual(approved["request_origin"], "prompt_workshop")
        self.assertEqual(approved["prompt_source"], "draft")
        self.assertIsNotNone(approved["execution_result"])
        self.assertEqual(approved["execution_result"]["status"], "success")
        latest_activity = self.store.get_recent_tool_outcomes(limit=1)[0]
        self.assertEqual(latest_activity.metadata.get("tool_name"), "prompt-pilot-execution")
        self.assertEqual(latest_activity.metadata.get("prompt_source"), "draft")
        self.assertEqual(latest_activity.metadata.get("request_origin"), "prompt_workshop")
        self.assertEqual(
            latest_activity.metadata.get("execution_status"),
            approved["execution_result"]["status"],
        )
        self.assertEqual(latest_activity.metadata.get("patch_status"), "applied")
        self.assertEqual(str(latest_activity.metadata.get("rejection_reason") or ""), "")
        self.assertEqual(
            latest_activity.metadata.get("selected_action_title"),
            approved["selected_action"]["title"],
        )
        self.assertIn("Prompt Workshop draft", latest_activity.metadata.get("outcome") or "")
        self.assertEqual(target.read_text(encoding="utf-8"), "alpha\ngamma\n")

    def test_cockpit_html_includes_operator_tutorial_copy(self) -> None:
        self.assertIn("Prepare = preflight", COCKPIT_HTML)
        self.assertIn("Trusted writes only cover repeated low-risk single-file edits", COCKPIT_HTML)
        self.assertIn("Prepare = safer preflight. Delegate = tracked handoff.", COCKPIT_HTML)
        self.assertIn("First Run Tutorial", COCKPIT_HTML)
        self.assertIn("Use bounded steps for your first hands-on test", COCKPIT_HTML)
        self.assertIn("Safe stopping rule", COCKPIT_HTML)
        self.assertIn("Recent demo walkthrough", COCKPIT_HTML)
        self.assertIn("How Ernie Operates", COCKPIT_HTML)
        self.assertIn("Review soul", COCKPIT_HTML)
        self.assertIn("Governed identity loop", COCKPIT_HTML)
        self.assertIn("Dismiss for now", COCKPIT_HTML)
        self.assertIn("Soul audit trail", COCKPIT_HTML)
        self.assertIn("Why now:", COCKPIT_HTML)
        self.assertIn("Target section:", COCKPIT_HTML)
        self.assertIn("Use result as draft", COCKPIT_HTML)
        self.assertIn("Undo local edit", COCKPIT_HTML)
        self.assertIn("Replace selection", COCKPIT_HTML)
        self.assertIn("Append result", COCKPIT_HTML)
        self.assertIn("Promote result to live note", COCKPIT_HTML)
        self.assertIn("Send result to guided loop", COCKPIT_HTML)
        self.assertIn("Send draft to guided loop", COCKPIT_HTML)
        self.assertIn("Prompt Workshop result", COCKPIT_HTML)
        self.assertIn("Manual pilot request", COCKPIT_HTML)
        self.assertIn("Request origin", COCKPIT_HTML)
        self.assertIn("prompt workshop chain", COCKPIT_HTML)
        self.assertIn("entered the guided loop", COCKPIT_HTML)
        self.assertIn("A pilot review packet was queued for approval.", COCKPIT_HTML)
        self.assertIn("rejected by validation before apply", COCKPIT_HTML)
        self.assertIn("Workshop mode", COCKPIT_HTML)
        self.assertIn("Promotion boundary", COCKPIT_HTML)
        self.assertIn("Draft editing helpers", COCKPIT_HTML)
        self.assertIn("Local history buffer", COCKPIT_HTML)
        self.assertIn("Local prompt history", COCKPIT_HTML)
        self.assertIn("Restore this draft", COCKPIT_HTML)
        self.assertIn("Draft before workshop", COCKPIT_HTML)
        self.assertIn("Workshop result after rewrite", COCKPIT_HTML)

    def test_cockpit_service_observe_and_execute_next(self) -> None:
        self.store.record_task(
            "Ship local cockpit",
            status="open",
            area="execution",
        )
        service = CockpitService(self.store, workspace_root=Path.cwd())

        observed = service.observe(
            role="user",
            text="We need a cockpit UI for the local agent.",
        )
        executed = service.execute_next(query="ship local cockpit", limit=3)

        self.assertGreater(observed["event_id"], 0)
        self.assertTrue(observed["stored_memories"])
        self.assertEqual(executed["result"]["status"], "success")
        self.assertEqual(executed["result"]["executed_kind"], "work_task")
        active = self.store.find_active_task("Ship local cockpit", area="execution")
        self.assertIsNotNone(active)
        self.assertEqual(active.metadata.get("status"), "in_progress")

    def test_cockpit_service_execute_plan_action_runs_selected_recommendation(self) -> None:
        self.store.record_task(
            "Ship guided cockpit loop",
            status="open",
            area="execution",
        )
        service = CockpitService(self.store, workspace_root=Path.cwd())

        snapshot = service.snapshot(query="ship guided cockpit loop", limit=3)
        recommendation = snapshot["plan"]["recommendation"]
        executed = service.execute_plan_action(
            query="ship guided cockpit loop",
            kind=recommendation["kind"],
            title=recommendation["title"],
            task_id=recommendation.get("task_id"),
            limit=3,
        )

        self.assertEqual(executed["selection_source"], "explicit_plan_action")
        self.assertEqual(executed["selected_action"]["title"], "Ship guided cockpit loop")
        self.assertEqual(executed["result"]["status"], "success")
        self.assertEqual(executed["result"]["executed_kind"], "work_task")

    def test_cockpit_service_execute_plan_action_can_run_alternative(self) -> None:
        self.store.record_task(
            "Ship guided cockpit loop",
            status="open",
            area="execution",
        )
        self.store.record_task(
            "Document cockpit tutorial basics",
            status="open",
            area="execution",
        )
        service = CockpitService(self.store, workspace_root=Path.cwd())

        snapshot = service.snapshot(query="what should I do next", limit=5)
        alternatives = snapshot["plan"]["alternatives"]

        self.assertTrue(alternatives)
        selected = alternatives[0]
        executed = service.execute_plan_action(
            query="what should I do next",
            kind=selected["kind"],
            title=selected["title"],
            task_id=selected.get("task_id"),
            limit=5,
        )

        self.assertEqual(executed["selection_source"], "explicit_plan_action")
        self.assertEqual(executed["selected_action"]["title"], selected["title"])
        self.assertEqual(executed["result"]["status"], "success")
        self.assertEqual(executed["result"]["executed_kind"], selected["kind"])

    def test_cockpit_health_payload_reports_service_state(self) -> None:
        service = CockpitService(self.store, workspace_root=Path.cwd())
        snapshot = service.snapshot(query="health", limit=3)
        self.assertIn("stats", snapshot)
        self.assertIn("plan", snapshot)

    def test_cockpit_service_task_detail_and_patch_runs(self) -> None:
        task = self.store.record_task(
            "Review cockpit task detail",
            status="open",
            area="execution",
            blocked_by=["Upstream review"],
            due_date="2026-04-11",
        )
        self.store.record_patch_run(
            run_name="cockpit patch demo",
            suite_name="builtin",
            task_title="Review cockpit task detail",
            status="applied",
            workspace_path=str(Path.cwd()),
            changed_files=["memory_agent/cockpit.py"],
            summary={"git": {"rollback_ready": False, "status": "direct_apply"}},
        )

        service = CockpitService(self.store, workspace_root=Path.cwd())
        detail = service.task_detail(title="Review cockpit task detail", area="execution")
        runs = service.patch_runs(limit=3)

        self.assertEqual(detail["task"]["id"], task.id)
        self.assertEqual(detail["task"]["metadata"]["due_date"], "2026-04-11")
        self.assertTrue(
            any(
                item["kind"] in {"work_task", "resolve_blocker"}
                for item in detail["task_plan_actions"]
            )
        )
        self.assertEqual(runs[0]["run_name"], "cockpit patch demo")

    def test_cockpit_service_task_detail_surfaces_prepare_action_for_confirmation_heavy_task(self) -> None:
        self.store.record_task(
            "Restart remote cockpit service",
            status="open",
            area="execution",
            service_action="restart_remote_service",
            service_label="Restart remote service",
            service_requires_confirmation=True,
            service_confirmation_message="Restart remote access for this machine?",
            complete_on_success=True,
        )
        service = CockpitService(self.store, workspace_root=Path.cwd())

        detail = service.task_detail(
            title="Restart remote cockpit service",
            area="execution",
        )

        self.assertTrue(
            any(item["kind"] == "prepare_task" for item in detail["task_plan_actions"])
        )

    def test_cockpit_service_preview_and_reject_pilot_turn(self) -> None:
        workspace = self._make_workspace()
        target = workspace / "pilot_eval.txt"
        target.write_text("alpha\nbeta\n", encoding="utf-8")
        self.store.record_task(
            "Pilot risky write",
            status="open",
            area="execution",
            file_operation="replace_text",
            file_path=str(target.relative_to(workspace)),
            find_text="beta",
            file_text="gamma",
            complete_on_success=True,
        )
        service = CockpitService(self.store, workspace_root=workspace)

        preview = service.preview_pilot_turn(
            text="pilot risky write",
            action_limit=3,
            use_model=False,
        )

        self.assertIsNotNone(preview["pending_id"])
        self.assertEqual(len(service.pending_pilot_queue()), 1)
        rejected = service.reject_pilot_turn(
            pending_id=preview["pending_id"],
            reason="reject in test",
        )
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(service.pending_pilot_queue(), [])

    def test_cockpit_service_session_and_task_actions(self) -> None:
        self.store.record_task(
            "Session task",
            status="open",
            area="execution",
        )
        service = CockpitService(self.store, workspace_root=Path.cwd())
        service._configured_auth_token = "secret"

        session = service.create_session(token="secret")
        self.assertIn("session_id", session)
        self.assertIn(session["session_id"], service.active_sessions)

        resumed = service.task_action(
            action="resume",
            title="Session task",
            area="execution",
        )
        self.assertEqual(resumed["metadata"]["status"], "open")

        completed = service.task_action(
            action="complete",
            title="Session task",
            area="execution",
        )
        self.assertEqual(completed["completed"]["metadata"]["status"], "done")

    def test_cockpit_service_seed_demo_workflow_creates_visible_tasks(self) -> None:
        service = CockpitService(self.store, workspace_root=Path.cwd())

        seeded = service.seed_demo_workflow()

        self.assertEqual(
            seeded["created_titles"],
            ["Try cockpit guided loop", "Inspect local cockpit service status"],
        )
        self.assertIn("Try cockpit guided loop", seeded["open_tasks"])
        self.assertIn("Inspect local cockpit service status", seeded["open_tasks"])
        inspection_task = self.store.find_active_task(
            "Inspect local cockpit service status",
            area="execution",
        )
        self.assertIsNotNone(inspection_task)
        self.assertEqual(
            inspection_task.metadata.get("service_inspection"),
            "restart_local_service",
        )
        latest_demo_activity = self.store.get_recent_tool_outcomes(limit=1)[0]
        self.assertEqual(latest_demo_activity.metadata.get("tool_name"), "demo-workflow")
        self.assertEqual(latest_demo_activity.metadata.get("demo_event"), "seed")

    def test_cockpit_service_seed_demo_workflow_is_idempotent(self) -> None:
        service = CockpitService(self.store, workspace_root=Path.cwd())

        first = service.seed_demo_workflow()
        second = service.seed_demo_workflow()

        self.assertEqual(
            first["created_titles"],
            ["Try cockpit guided loop", "Inspect local cockpit service status"],
        )
        self.assertEqual(second["created_titles"], [])
        self.assertEqual(
            second["unchanged_titles"],
            ["Try cockpit guided loop", "Inspect local cockpit service status"],
        )

    def test_cockpit_service_reset_demo_workflow_restores_starting_state(self) -> None:
        service = CockpitService(self.store, workspace_root=Path.cwd())
        service.seed_demo_workflow()
        self.store.record_task(
            "Try cockpit guided loop",
            status="in_progress",
            area="execution",
        )
        self.store.record_task(
            "Inspect local cockpit service status",
            status="done",
            area="execution",
        )

        reset = service.reset_demo_workflow()

        self.assertEqual(
            reset["reset_titles"],
            ["Try cockpit guided loop", "Inspect local cockpit service status"],
        )
        guided = self.store.find_active_task("Try cockpit guided loop", area="execution")
        inspection = self.store.find_active_task(
            "Inspect local cockpit service status",
            area="execution",
        )
        self.assertIsNotNone(guided)
        self.assertIsNotNone(inspection)
        self.assertEqual(guided.metadata.get("status"), "open")
        self.assertEqual(inspection.metadata.get("status"), "open")
        self.assertEqual(
            inspection.metadata.get("service_inspection"),
            "restart_local_service",
        )
        latest_demo_activity = self.store.get_recent_tool_outcomes(limit=1)[0]
        self.assertEqual(latest_demo_activity.metadata.get("tool_name"), "demo-workflow")
        self.assertEqual(latest_demo_activity.metadata.get("demo_event"), "reset")

    def test_cockpit_service_execute_plan_action_records_demo_run_activity(self) -> None:
        service = CockpitService(self.store, workspace_root=Path.cwd())
        service.seed_demo_workflow()

        payload = service.execute_plan_action(
            query="try cockpit guided loop",
            kind="work_task",
            title="Try cockpit guided loop",
            limit=3,
        )

        self.assertEqual(payload["selected_action"]["title"], "Try cockpit guided loop")
        latest_demo_activity = self.store.get_recent_tool_outcomes(limit=1)[0]
        self.assertEqual(latest_demo_activity.metadata.get("tool_name"), "demo-workflow")
        self.assertEqual(latest_demo_activity.metadata.get("demo_event"), "run")
        self.assertEqual(
            latest_demo_activity.metadata.get("plan_action_kind"),
            "work_task",
        )
        self.assertEqual(
            latest_demo_activity.metadata.get("execution_status"),
            "success",
        )

    def test_cockpit_service_soul_review_and_apply_proposal(self) -> None:
        workspace = self._make_workspace()
        service = CockpitService(self.store, workspace_root=workspace)
        self.store.observe(
            role="user",
            content="I am inexperienced and I will need tutorials on things.",
        )
        self.store.observe(
            role="user",
            content="Keep going while I sleep and stop at real approval boundaries.",
        )

        review = service.soul_review()

        proposal_ids = [item["proposal_id"] for item in review["proposals"]]
        self.assertIn("operator_tutorial_duty", proposal_ids)
        applied = service.apply_soul_amendment(proposal_id="operator_tutorial_duty")

        soul_path = workspace / "SOUL.md"
        self.assertTrue(soul_path.exists())
        soul_text = soul_path.read_text(encoding="utf-8")
        self.assertIn(
            "When introducing a new workflow, pair it with a short hands-on tutorial",
            soul_text,
        )
        remaining_ids = [item["proposal_id"] for item in applied["review"]["proposals"]]
        self.assertNotIn("operator_tutorial_duty", remaining_ids)
        self.assertIn("autonomous_chunking_rule", remaining_ids)

    def test_cockpit_service_soul_dismiss_suppresses_proposal_and_records_audit(self) -> None:
        workspace = self._make_workspace()
        service = CockpitService(self.store, workspace_root=workspace)
        self.store.observe(
            role="user",
            content="I am inexperienced and I will need tutorials on things.",
        )

        review = service.soul_review()
        proposal_ids = [item["proposal_id"] for item in review["proposals"]]
        self.assertIn("operator_tutorial_duty", proposal_ids)

        dismissed = service.dismiss_soul_amendment(proposal_id="operator_tutorial_duty")

        remaining_ids = [item["proposal_id"] for item in dismissed["review"]["proposals"]]
        self.assertNotIn("operator_tutorial_duty", remaining_ids)
        audit = service.settings()["soul_audit"]
        self.assertTrue(audit)
        self.assertEqual(audit[0]["tool_name"], "soul-dismiss")
        self.assertEqual(audit[0]["proposal_id"], "operator_tutorial_duty")

    def test_cockpit_service_soul_dismissal_reappears_when_evidence_changes(self) -> None:
        workspace = self._make_workspace()
        service = CockpitService(self.store, workspace_root=workspace)
        self.store.observe(
            role="user",
            content="I am inexperienced and I will need tutorials on things.",
        )

        initial_review = service.soul_review()
        initial_proposal = next(
            item for item in initial_review["proposals"]
            if item["proposal_id"] == "operator_tutorial_duty"
        )
        service.dismiss_soul_amendment(proposal_id="operator_tutorial_duty")

        suppressed_review = service.soul_review()
        suppressed_ids = [item["proposal_id"] for item in suppressed_review["proposals"]]
        self.assertNotIn("operator_tutorial_duty", suppressed_ids)

        self.store.observe(
            role="user",
            content="Please add beginner tutorials and a hands-on learning walkthrough for new workflows.",
        )

        renewed_review = service.soul_review()
        renewed_proposal = next(
            item for item in renewed_review["proposals"]
            if item["proposal_id"] == "operator_tutorial_duty"
        )
        self.assertNotEqual(
            initial_proposal["evidence_signature"],
            renewed_proposal["evidence_signature"],
        )
        self.assertTrue(renewed_proposal["resurfaced_after_dismissal"])
        self.assertIn("tutorials", renewed_proposal["explanation"].lower())

    def test_cockpit_service_settings_and_remote_token_rotation(self) -> None:
        config_dir = self.temp_root / f"remote_cfg_{uuid.uuid4().hex}"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "remote.env").write_text(
            'PORT="8766"\nTOKEN="secret-token"\nDISPLAY_HOST="100.1.2.3"\n',
            encoding="utf-8",
        )
        service = CockpitService(self.store, workspace_root=Path.cwd())
        service.service_manager = CockpitServiceManager(config_dir=config_dir)

        with patch("memory_agent.service_manager.subprocess.run") as run_mock:
            run_mock.side_effect = [
                subprocess.CompletedProcess(
                    ["systemctl", "--user", "is-active", "ernie-cockpit.service"],
                    0,
                    stdout="active\n",
                    stderr="",
                ),
                subprocess.CompletedProcess(
                    ["systemctl", "--user", "is-active", "ernie-cockpit-remote.service"],
                    0,
                    stdout="active\n",
                    stderr="",
                ),
                subprocess.CompletedProcess(
                    ["systemctl", "--user", "restart", "ernie-cockpit-remote.service"],
                    0,
                    stdout="",
                    stderr="",
                ),
                subprocess.CompletedProcess(
                    ["systemctl", "--user", "is-active", "ernie-cockpit-remote.service"],
                    0,
                    stdout="active\n",
                    stderr="",
                ),
            ]
            settings = service.settings()
            rotated = service.rotate_remote_access_token()

        self.assertTrue(settings["local_service"]["active"])
        self.assertEqual(settings["remote_service"]["url"], "http://100.1.2.3:8766/")
        self.assertTrue(settings["actions"])
        self.assertFalse(
            next(
                item["enabled"]
                for item in settings["actions"]
                if item["action"] == "install_desktop_launcher"
            )
        )
        self.assertTrue(
            next(
                item["requires_confirmation"]
                for item in settings["actions"]
                if item["action"] == "install_remote_service"
            )
        )
        self.assertTrue(rotated["rotated"])
        self.assertNotEqual(rotated["token"], "secret-token")

    def test_cockpit_service_settings_include_and_update_pilot_policy(self) -> None:
        workspace = self._make_workspace()
        config_dir = self.temp_root / f"pilot_cfg_{uuid.uuid4().hex}"
        config_dir.mkdir(parents=True, exist_ok=True)
        service = CockpitService(self.store, workspace_root=workspace)
        service.service_manager = CockpitServiceManager(config_dir=config_dir)

        with patch("memory_agent.service_manager.subprocess.run") as run_mock:
            run_mock.side_effect = [
                subprocess.CompletedProcess(
                    ["systemctl", "--user", "is-active", "ernie-cockpit.service"],
                    3,
                    stdout="inactive\n",
                    stderr="",
                ),
                subprocess.CompletedProcess(
                    ["systemctl", "--user", "is-active", "ernie-cockpit-remote.service"],
                    3,
                    stdout="inactive\n",
                    stderr="",
                ),
            ]
            settings = service.settings()

        self.assertIn("pilot_policy", settings)
        self.assertIn("soul", settings)
        self.assertEqual(settings["soul"]["title"], "Ernie")
        self.assertIn("careful local operator", settings["soul"]["summary"])
        self.assertTrue(settings["pilot_policy"]["trusted_writes_enabled"])
        self.assertEqual(
            settings["pilot_policy"]["trusted_auto_approve_required_successes"],
            2,
        )

        disabled = service.update_pilot_policy(
            trusted_writes_enabled=False,
            trusted_write_required_successes=3,
        )
        self.assertFalse(disabled["pilot_policy"]["trusted_writes_enabled"])
        self.assertEqual(
            disabled["pilot_policy"]["trusted_auto_approve_file_operations"],
            [],
        )
        self.assertEqual(
            disabled["pilot_policy"]["trusted_auto_approve_required_successes"],
            3,
        )

        enabled = service.update_pilot_policy(
            trusted_writes_enabled=True,
            trusted_write_required_successes=4,
            trusted_write_operations=["append_text"],
            service_sync_suppression_window_seconds={
                "local_service": 300,
                "remote_service": 2400,
            },
        )
        self.assertTrue(enabled["pilot_policy"]["trusted_writes_enabled"])
        self.assertEqual(
            enabled["pilot_policy"]["trusted_auto_approve_file_operations"],
            ["append_text"],
        )
        self.assertEqual(
            enabled["pilot_policy"]["trusted_auto_approve_required_successes"],
            4,
        )
        self.assertEqual(
            enabled["pilot_policy"]["service_sync_suppression_window_seconds"],
            {
                "desktop_launcher": 1200,
                "local_service": 300,
                "remote_service": 2400,
            },
        )

        reloaded = LinuxPilotPolicy.load(workspace_root=workspace)
        self.assertEqual(reloaded.trusted_auto_approve_file_operations, {"append_text"})
        self.assertEqual(reloaded.trusted_auto_approve_required_successes, 4)
        self.assertEqual(
            reloaded.service_sync_suppression_window_seconds,
            {
                "desktop_launcher": 1200,
                "local_service": 300,
                "remote_service": 2400,
            },
        )

    def test_cockpit_service_settings_include_trusted_write_recommendations(self) -> None:
        workspace = self._make_workspace()
        config_dir = self.temp_root / f"pilot_cfg_reco_{uuid.uuid4().hex}"
        config_dir.mkdir(parents=True, exist_ok=True)
        rel_path = ".test_tmp/recommended_write.txt"
        for index in range(2):
            self.store.record_tool_outcome(
                "pilot-review",
                f"Pilot review trust recommendation {index}",
                status="blocked",
                subject="self_improvement",
                tags=["pilot", "review", "self-improvement"],
                metadata={
                    "goal_text": f"pilot trusted write seed {index}",
                    "stop_reason": "needs_approval",
                    "executed_steps": 0,
                    "approval_requests": 1,
                    "approvals_granted": 0,
                    "opportunity_count": 1,
                    "opportunity_categories": ["approval_friction", "pilot_trusted_write_candidate"],
                    "recurring_patterns": [
                        {
                            "kind": "trusted_write_candidate",
                            "key": f"write_text:{rel_path}",
                            "label": f"write_text on {rel_path}",
                            "count": index + 1,
                            "file_operation": "write_text",
                            "file_path": rel_path,
                        }
                    ],
                },
            )

        service = CockpitService(self.store, workspace_root=workspace)
        service.service_manager = CockpitServiceManager(config_dir=config_dir)

        with patch("memory_agent.service_manager.subprocess.run") as run_mock:
            run_mock.side_effect = [
                subprocess.CompletedProcess(
                    ["systemctl", "--user", "is-active", "ernie-cockpit.service"],
                    3,
                    stdout="inactive\n",
                    stderr="",
                ),
                subprocess.CompletedProcess(
                    ["systemctl", "--user", "is-active", "ernie-cockpit-remote.service"],
                    3,
                    stdout="inactive\n",
                    stderr="",
                ),
            ]
            settings = service.settings()

        recommendations = settings["pilot_policy"]["trusted_write_recommendations"]
        self.assertTrue(recommendations)
        self.assertEqual(recommendations[0]["file_operation"], "write_text")
        self.assertEqual(recommendations[0]["file_path"], rel_path)
        self.assertGreaterEqual(recommendations[0]["count"], 2)

    def test_service_manager_reports_unconfigured_remote_service(self) -> None:
        config_dir = self.temp_root / f"remote_cfg_missing_{uuid.uuid4().hex}"
        manager = CockpitServiceManager(config_dir=config_dir)
        with patch("memory_agent.service_manager.subprocess.run") as run_mock:
            run_mock.side_effect = [
                subprocess.CompletedProcess(
                    ["systemctl", "--user", "is-active", "ernie-cockpit.service"],
                    3,
                    stdout="inactive\n",
                    stderr="",
                ),
                subprocess.CompletedProcess(
                    ["systemctl", "--user", "is-active", "ernie-cockpit-remote.service"],
                    3,
                    stdout="inactive\n",
                    stderr="",
                ),
            ]
            payload = manager.settings()
        self.assertFalse(payload["remote_service"]["configured"])
        self.assertEqual(payload["remote_service"]["token"], "")
        self.assertTrue(payload["onboarding"]["recommended_steps"])
        onboarding_actions = {item["action"] for item in payload["onboarding"]["actions"]}
        self.assertIn("install_remote_service", onboarding_actions)

    def test_service_manager_performs_guided_setup_action(self) -> None:
        config_dir = self.temp_root / f"remote_cfg_action_{uuid.uuid4().hex}"
        config_dir.mkdir(parents=True, exist_ok=True)
        manager = CockpitServiceManager(config_dir=config_dir)
        with patch("memory_agent.service_manager.subprocess.run") as run_mock:
            run_mock.side_effect = [
                subprocess.CompletedProcess(
                    ["systemctl", "--user", "restart", "ernie-cockpit.service"],
                    0,
                    stdout="",
                    stderr="",
                ),
                subprocess.CompletedProcess(
                    ["systemctl", "--user", "is-active", "ernie-cockpit.service"],
                    0,
                    stdout="active\n",
                    stderr="",
                ),
                subprocess.CompletedProcess(
                    ["systemctl", "--user", "is-active", "ernie-cockpit.service"],
                    0,
                    stdout="active\n",
                    stderr="",
                ),
                subprocess.CompletedProcess(
                    ["systemctl", "--user", "is-active", "ernie-cockpit-remote.service"],
                    3,
                    stdout="inactive\n",
                    stderr="",
                ),
            ]
            payload = manager.perform_action("restart_local_service")
        self.assertEqual(payload["action"], "restart_local_service")
        self.assertEqual(payload["meta"]["label"], "Restart local service")
        self.assertTrue(payload["result"]["ok"])
        self.assertIn("Local cockpit service restarted.", payload["message"])
        self.assertTrue(payload["settings"]["local_service"]["active"])

    def test_cockpit_service_settings_syncs_recommended_service_tasks(self) -> None:
        service = CockpitService(self.store, workspace_root=Path.cwd())

        class FakeServiceManager:
            def settings(self) -> dict[str, object]:
                return {
                    "onboarding": {
                        "actions": [
                            {
                                "action": "install_desktop_launcher",
                                "label": "Install desktop launcher",
                                "description": "Install the Ernie app-menu entry and launcher.",
                                "enabled": True,
                                "success_message": "Desktop launcher installed.",
                            }
                        ]
                    },
                    "actions": [],
                }

        service.service_manager = FakeServiceManager()
        settings = service.settings()

        self.assertIn("service_sync", settings)
        self.assertEqual(
            settings["service_sync"]["recommended_actions"],
            ["install_desktop_launcher"],
        )
        synced_task = self.store.find_active_task(
            "Cockpit setup: Install desktop launcher",
            area="execution",
        )
        self.assertIsNotNone(synced_task)
        self.assertEqual(
            synced_task.metadata.get("service_action"),
            "install_desktop_launcher",
        )

    def test_cockpit_service_settings_surface_recent_verification_suppression(self) -> None:
        service = CockpitService(self.store, workspace_root=Path.cwd())

        class FakeServiceManager:
            def settings(self) -> dict[str, object]:
                return {
                    "onboarding": {
                        "actions": [
                            {
                                "action": "install_remote_service",
                                "label": "Install remote service",
                                "description": "Install or repair remote access.",
                                "enabled": True,
                            }
                        ]
                    },
                    "actions": [],
                }

        self.store.record_tool_outcome(
            "service_manager",
            "Healthy remote verification just completed.",
            status="success",
            subject="execution",
            tags=["executor", "service_inspection", "success"],
            metadata={
                "service_inspection": "restart_remote_service",
                "service_inspection_healthy": True,
                "resolved_service_sync_titles": ["Cockpit setup: Install remote service"],
            },
        )

        service.service_manager = FakeServiceManager()
        settings = service.settings()

        self.assertIn("service_sync_status", settings)
        self.assertFalse(settings["service_sync_status"]["due"])
        self.assertEqual(
            settings["service_sync_status"]["suppressed_recent_verification_actions"],
            ["install_remote_service"],
        )
        self.assertEqual(
            settings["service_sync_status"]["suppressed_recent_verification_titles"],
            ["Cockpit setup: Install remote service"],
        )
        self.assertIsNotNone(
            settings["service_sync_status"]["suppressed_recent_verification_updated_at"]
        )
        self.assertGreaterEqual(
            settings["service_sync_status"]["suppressed_recent_verification_age_seconds"],
            0,
        )
        self.assertGreater(
            settings["service_sync_status"]["suppressed_recent_verification_expires_in_seconds"],
            0,
        )

    def test_cockpit_service_settings_respect_custom_service_sync_suppression_windows(self) -> None:
        workspace = self._make_workspace()
        service = CockpitService(self.store, workspace_root=workspace)

        class FakeServiceManager:
            def settings(self) -> dict[str, object]:
                return {
                    "onboarding": {
                        "actions": [
                            {
                                "action": "install_remote_service",
                                "label": "Install remote service",
                                "description": "Install or repair remote access.",
                                "enabled": True,
                            }
                        ]
                    },
                    "actions": [],
                }

        verification = self.store.record_tool_outcome(
            "service_manager",
            "Healthy remote verification just completed.",
            status="success",
            subject="execution",
            tags=["executor", "service_inspection", "success"],
            metadata={
                "service_inspection": "restart_remote_service",
                "service_inspection_healthy": True,
                "resolved_service_sync_titles": ["Cockpit setup: Install remote service"],
            },
        )
        backdated = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
        self.store.connection.execute(
            "update memories set updated_at = ? where id = ?",
            (backdated, verification.id),
        )
        self.store.connection.commit()

        service.service_manager = FakeServiceManager()
        default_status = service._planner().service_sync_status(
            service.service_manager.settings()
        )
        self.assertFalse(default_status["due"])

        updated = service.update_pilot_policy(
            service_sync_suppression_window_seconds={"remote_service": 300}
        )
        self.assertEqual(
            updated["pilot_policy"]["service_sync_suppression_window_seconds"]["remote_service"],
            300,
        )

        tightened_status = service._planner().service_sync_status(
            service.service_manager.settings()
        )
        self.assertTrue(tightened_status["due"])
        self.assertEqual(
            tightened_status["recommended_actions"],
            ["install_remote_service"],
        )
        self.assertEqual(
            tightened_status["suppressed_recent_verification_actions"],
            [],
        )

    def test_build_parser_includes_serve_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["serve", "--host", "127.0.0.1", "--port", "9001"])
        self.assertEqual(args.command, "serve")
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 9001)

    def test_build_parser_accepts_serve_token(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["serve", "--host", "0.0.0.0", "--port", "9001", "--token", "secret"])
        self.assertEqual(args.command, "serve")
        self.assertEqual(args.token, "secret")

    def test_build_parser_accepts_remote_serve_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["serve", "--remote", "--display-host", "100.1.2.3"])
        self.assertEqual(args.command, "serve")
        self.assertTrue(args.remote)
        self.assertEqual(args.display_host, "100.1.2.3")

    def test_build_parser_accepts_task_retry_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "task",
                "Retry task",
                "--command",
                "python -m memory_agent.cli --help",
                "--retry-limit",
                "2",
                "--retry-cooldown-minutes",
                "15",
            ]
        )
        self.assertEqual(args.command, "task")
        self.assertEqual(args.retry_limit, 2)
        self.assertEqual(args.retry_cooldown_minutes, 15)

    def test_build_parser_accepts_task_service_action(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "task",
                "Restart cockpit",
                "--service-action",
                "restart_local_service",
            ]
        )
        self.assertEqual(args.command, "task")
        self.assertEqual(args.service_action, "restart_local_service")

    def test_resolve_serve_config_generates_remote_token_and_access_url(self) -> None:
        args = Namespace(
            host="127.0.0.1",
            port=8765,
            remote=True,
            token=None,
            display_host="100.1.2.3",
        )
        with patch.object(cli_module.secrets, "token_hex", return_value="feedfacefeedface"):
            config = _resolve_serve_config(args)
        self.assertEqual(config["host"], "0.0.0.0")
        self.assertEqual(config["token"], "feedfacefeedface")
        self.assertEqual(
            config["access_url"],
            "http://100.1.2.3:8765/?token=feedfacefeedface",
        )

    def test_resolve_serve_config_preserves_local_access_url_without_token(self) -> None:
        args = Namespace(
            host="127.0.0.1",
            port=9001,
            remote=False,
            token=None,
            display_host=None,
        )
        config = _resolve_serve_config(args)
        self.assertEqual(config["host"], "127.0.0.1")
        self.assertIsNone(config["token"])
        self.assertEqual(config["access_url"], "http://127.0.0.1:9001/")

    def test_synthesize_profiles_creates_stable_long_term_memory(self) -> None:
        self.store.remember(
            MemoryDraft(
                kind="preference",
                subject="optimization",
                content="Optimize the agent for low ongoing cost.",
                tags=["cost", "optimization"],
                importance=0.95,
                confidence=0.9,
            )
        )
        self.store.remember(
            MemoryDraft(
                kind="preference",
                subject="optimization",
                content="Optimize the agent for efficient, low-latency operation.",
                tags=["performance", "optimization"],
                importance=0.92,
                confidence=0.9,
            )
        )
        self.store.reflect_recent(limit=10, max_reflections=3)
        profiles = self.store.synthesize_profiles(limit=20, max_profiles=3)
        optimization_profile = next(
            profile for profile in profiles if profile.subject == "optimization"
        )
        self.assertEqual(optimization_profile.layer, "profile")
        edges = self.store.get_memory_edges(optimization_profile.id, direction="outgoing")
        self.assertTrue(any(edge.edge_type == "abstracts" for edge in edges))
        results = self.store.search("optimization cost latency", limit=3, layers=("profile",))
        self.assertEqual(results[0].memory.id, optimization_profile.id)


if __name__ == "__main__":
    unittest.main()
