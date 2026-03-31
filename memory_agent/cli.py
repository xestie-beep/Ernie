from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path
from typing import Callable

from .agent import MemoryFirstAgent
from .config import DEFAULT_DB_PATH
from .cockpit import serve_cockpit
from .evaluation import MemoryEvaluator
from .executor import MemoryExecutor
from .improvement import MemoryImprovementEngine, PilotHistoryReporter, PilotRunReviewer
from .linux_runtime import LinuxPilotPolicy, LinuxPilotRuntime
from .memory import MemoryStore
from .migration import ProjectHandoffManager
from .models import MemoryDraft
from .patch_runner import PatchOperation, WorkspacePatchRunner
from .planner import MemoryPlanner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Memory-first local agent scaffold")
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to the SQLite memory database.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    observe_parser = subparsers.add_parser("observe", help="Log a turn and extract memory.")
    observe_parser.add_argument(
        "--role",
        choices=["user", "assistant", "system", "tool"],
        default="user",
        help="Role for the observed message.",
    )
    observe_parser.add_argument("text", help="Text to ingest as a user message.")

    remember_parser = subparsers.add_parser("remember", help="Insert a memory manually.")
    remember_parser.add_argument("kind")
    remember_parser.add_argument("subject")
    remember_parser.add_argument("content")
    remember_parser.add_argument("--importance", type=float, default=0.7)
    remember_parser.add_argument("--confidence", type=float, default=0.8)
    remember_parser.add_argument("--tags", nargs="*", default=[])

    search_parser = subparsers.add_parser("search", help="Search long-term memory.")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=5)

    entities_parser = subparsers.add_parser("entities", help="Resolve query entities.")
    entities_parser.add_argument("query")

    context_parser = subparsers.add_parser("context", help="Render the current context window.")
    context_parser.add_argument("query")
    context_parser.add_argument("--limit", type=int, default=5)

    reply_parser = subparsers.add_parser("reply", help="Generate a model-backed reply using memory context.")
    reply_parser.add_argument("text")
    reply_parser.add_argument("--json", action="store_true")

    decide_parser = subparsers.add_parser(
        "decide",
        help="Let the main model choose a structured action from planner-approved options.",
    )
    decide_parser.add_argument("text")
    decide_parser.add_argument(
        "--preview",
        action="store_true",
        help="Validate the model's action proposal without executing it.",
    )
    decide_parser.add_argument("--json", action="store_true")

    plan_parser = subparsers.add_parser("plan", help="Recommend the next best action from memory.")
    plan_parser.add_argument("query", nargs="?", default="next best action")
    plan_parser.add_argument("--limit", type=int, default=5)
    plan_parser.add_argument("--json", action="store_true")

    execute_parser = subparsers.add_parser("execute", help="Execute the next planned action and replan.")
    execute_parser.add_argument("query", nargs="?", default="next best action")
    execute_parser.add_argument("--limit", type=int, default=5)
    execute_parser.add_argument("--json", action="store_true")

    pilot_parser = subparsers.add_parser(
        "pilot",
        help="Run one supervised Linux-pilot turn with approval gating and trace logging.",
    )
    pilot_parser.add_argument("text")
    pilot_parser.add_argument("--policy-file", type=Path)
    pilot_parser.add_argument("--approve", action="store_true")
    pilot_parser.add_argument("--no-model", action="store_true")
    pilot_parser.add_argument("--limit", type=int, default=5)
    pilot_parser.add_argument("--json", action="store_true")

    pilot_policy_parser = subparsers.add_parser(
        "pilot-policy",
        help="Show or write the current Linux pilot policy template.",
    )
    pilot_policy_parser.add_argument("--policy-file", type=Path)
    pilot_policy_parser.add_argument("--write-template", type=Path)
    pilot_policy_parser.add_argument("--json", action="store_true")

    pilot_report_parser = subparsers.add_parser(
        "pilot-report",
        help="Show cross-run trends from recent pilot reviews.",
    )
    pilot_report_parser.add_argument("--limit", type=int, default=12)
    pilot_report_parser.add_argument("--json", action="store_true")

    pilot_run_parser = subparsers.add_parser(
        "pilot-run",
        help="Run a supervised multi-step Linux pilot session with budgets and stop conditions.",
    )
    pilot_run_parser.add_argument("text")
    pilot_run_parser.add_argument("--policy-file", type=Path)
    pilot_run_parser.add_argument("--approve", action="store_true")
    pilot_run_parser.add_argument("--no-model", action="store_true")
    pilot_run_parser.add_argument("--limit", type=int, default=5)
    pilot_run_parser.add_argument("--steps", type=int, default=5)
    pilot_run_parser.add_argument(
        "--promote-limit",
        type=int,
        default=0,
        help="Optionally promote top pilot debrief opportunities into self-improvement tasks.",
    )
    pilot_run_parser.add_argument("--json", action="store_true")

    pilot_chat_parser = subparsers.add_parser(
        "pilot-chat",
        help="Run an interactive supervised Linux-pilot session with approval prompts.",
    )
    pilot_chat_parser.add_argument("--policy-file", type=Path)
    pilot_chat_parser.add_argument("--no-model", action="store_true")
    pilot_chat_parser.add_argument("--limit", type=int, default=5)
    pilot_chat_parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Approve gated actions automatically for the whole session.",
    )

    ready_parser = subparsers.add_parser("ready", help="List tasks that are ready to work on now.")
    ready_parser.add_argument("--limit", type=int, default=5)

    overdue_parser = subparsers.add_parser("overdue", help="List overdue active tasks.")
    overdue_parser.add_argument("--limit", type=int, default=5)

    review_parser = subparsers.add_parser("review-tasks", help="Generate due task nudges.")
    review_parser.add_argument("--limit", type=int, default=10)

    complete_parser = subparsers.add_parser("complete-task", help="Complete a task and roll recurrence forward.")
    complete_parser.add_argument("title")
    complete_parser.add_argument("--area", default="execution")
    complete_parser.add_argument("--completed-at")

    snooze_parser = subparsers.add_parser("snooze-task", help="Snooze a task until a future date.")
    snooze_parser.add_argument("title")
    snooze_parser.add_argument("--until", required=True)
    snooze_parser.add_argument("--area", default="execution")

    unblock_parser = subparsers.add_parser("unblock-task", help="Clear blockers and reopen a task.")
    unblock_parser.add_argument("title")
    unblock_parser.add_argument("--area", default="execution")

    resume_parser = subparsers.add_parser("resume-task", help="Clear a snooze and return a task to active rotation.")
    resume_parser.add_argument("title")
    resume_parser.add_argument("--area", default="execution")

    reflect_parser = subparsers.add_parser("reflect", help="Create source-linked reflections.")
    reflect_parser.add_argument("--limit", type=int, default=20)
    reflect_parser.add_argument("--max-reflections", type=int, default=5)
    reflect_parser.add_argument("--max-profiles", type=int, default=4)

    maintain_parser = subparsers.add_parser("maintain", help="Run due maintenance tasks.")
    maintain_parser.add_argument("--force", action="store_true")
    maintain_parser.add_argument("--limit", type=int, default=20)
    maintain_parser.add_argument("--max-reflections", type=int, default=5)
    maintain_parser.add_argument("--max-profiles", type=int, default=4)

    revise_parser = subparsers.add_parser("revise", help="Create a new version of a memory.")
    revise_parser.add_argument("memory_id", type=int)
    revise_parser.add_argument("content")
    revise_parser.add_argument("--importance", type=float)
    revise_parser.add_argument("--confidence", type=float)
    revise_parser.add_argument("--tags", nargs="*", default=[])

    history_parser = subparsers.add_parser("history", help="Inspect memory provenance.")
    history_parser.add_argument("memory_id", type=int)

    task_parser = subparsers.add_parser("task", help="Record a structured task memory.")
    task_parser.add_argument("title")
    task_parser.add_argument("--status", default="open")
    task_parser.add_argument("--area", default="execution")
    task_parser.add_argument("--owner", default="agent")
    task_parser.add_argument("--details")
    task_parser.add_argument("--depends-on", nargs="*", default=None)
    task_parser.add_argument("--blocked-by", nargs="*", default=None)
    task_parser.add_argument("--due-date")
    task_parser.add_argument("--recurrence-days", type=int)
    task_parser.add_argument("--snoozed-until")
    task_parser.add_argument("--command", dest="task_command")
    task_parser.add_argument("--cwd")
    task_parser.add_argument(
        "--service-action",
        choices=[
            "install_desktop_launcher",
            "install_local_service",
            "install_remote_service",
            "restart_local_service",
            "restart_remote_service",
        ],
    )
    task_parser.add_argument(
        "--file-op",
        choices=[
            "read_text",
            "write_text",
            "append_text",
            "replace_text",
            "replace_python_function",
            "replace_python_class",
            "insert_python_before_symbol",
            "insert_python_after_symbol",
            "delete_python_symbol",
            "rename_python_identifier",
            "rename_python_method",
            "add_python_import",
            "remove_python_import",
            "add_python_function_parameter",
            "add_python_method_parameter",
            "rename_python_export_across_imports",
            "move_python_export_to_module",
        ],
    )
    task_parser.add_argument("--file-path")
    task_parser.add_argument("--file-text")
    task_parser.add_argument("--find-text")
    task_parser.add_argument("--symbol-name")
    task_parser.add_argument("--replace-all", action="store_true", default=None)
    task_parser.add_argument("--complete-on-success", action="store_true", default=None)
    task_parser.add_argument("--retry-limit", type=int)
    task_parser.add_argument("--retry-cooldown-minutes", type=int)
    task_parser.add_argument("--tags", nargs="*", default=[])

    decision_parser = subparsers.add_parser("decision", help="Record a structured decision.")
    decision_parser.add_argument("topic")
    decision_parser.add_argument("decision")
    decision_parser.add_argument("--rationale")
    decision_parser.add_argument("--tags", nargs="*", default=[])

    tool_parser = subparsers.add_parser("tool-outcome", help="Record a tool execution outcome.")
    tool_parser.add_argument("tool_name")
    tool_parser.add_argument("outcome")
    tool_parser.add_argument("--status", default="success")
    tool_parser.add_argument("--subject", default="tooling")
    tool_parser.add_argument("--tags", nargs="*", default=[])

    evaluate_parser = subparsers.add_parser("evaluate", help="Run the built-in memory benchmark suite.")
    evaluate_parser.add_argument("--json", action="store_true", help="Print the full report as JSON.")

    improve_parser = subparsers.add_parser(
        "improve",
        help="Run an eval-gated self-improvement review and promote top follow-up tasks.",
    )
    improve_parser.add_argument("--json", action="store_true")
    improve_parser.add_argument("--promote-limit", type=int, default=3)
    improve_parser.add_argument(
        "--preview",
        action="store_true",
        help="Review opportunities without promoting improvement tasks.",
    )
    improve_parser.add_argument(
        "--no-strategic-backlog",
        action="store_true",
        help="Skip strategic backlog reminders such as code-aware patching.",
    )
    improve_parser.add_argument("--suite-name", default="builtin")

    patch_parser = subparsers.add_parser(
        "patch-run",
        help="Try bounded code/file changes in a temp workspace, validate them, and optionally apply them back.",
    )
    patch_parser.add_argument("run_name")
    patch_parser.add_argument("--task-title")
    patch_parser.add_argument("--task-area", default="self_improvement")
    patch_parser.add_argument("--suite-name", default="builtin")
    patch_parser.add_argument("--spec-file", type=Path)
    patch_parser.add_argument(
        "--file-op",
        choices=[
            "write_text",
            "append_text",
            "replace_text",
            "replace_python_function",
            "replace_python_class",
            "insert_python_before_symbol",
            "insert_python_after_symbol",
            "delete_python_symbol",
            "rename_python_identifier",
            "rename_python_method",
            "add_python_import",
            "remove_python_import",
            "add_python_function_parameter",
            "add_python_method_parameter",
            "rename_python_export_across_imports",
            "move_python_export_to_module",
        ],
    )
    patch_parser.add_argument("--file-path")
    patch_parser.add_argument("--file-text")
    patch_parser.add_argument("--find-text")
    patch_parser.add_argument("--symbol-name")
    patch_parser.add_argument("--replace-all", action="store_true", default=None)
    patch_parser.add_argument(
        "--validate",
        action="append",
        default=None,
        help="Additional validation command to run in the temp workspace. Repeat as needed.",
    )
    patch_parser.add_argument(
        "--git-mode",
        choices=["auto", "branch", "off"],
        default="auto",
        help="How to apply successful candidates back: auto git branch, required git branch, or direct apply only.",
    )
    patch_parser.add_argument("--apply-on-success", action="store_true")
    patch_parser.add_argument("--json", action="store_true")

    rollback_parser = subparsers.add_parser(
        "patch-rollback",
        help="Roll back the latest git-backed disposable patch branch, or a specific patch run id.",
    )
    rollback_parser.add_argument("--run-id", type=int)
    rollback_parser.add_argument("--force", action="store_true")
    rollback_parser.add_argument("--json", action="store_true")

    handoff_pack_parser = subparsers.add_parser(
        "handoff-pack",
        help="Create a Linux handoff bundle with the important local agent state.",
    )
    handoff_pack_parser.add_argument("--output-dir", type=Path)
    handoff_pack_parser.add_argument(
        "--skip-traces",
        action="store_true",
        help="Exclude pilot trace files from the bundle.",
    )
    handoff_pack_parser.add_argument("--json", action="store_true")

    handoff_restore_parser = subparsers.add_parser(
        "handoff-restore",
        help="Restore a Linux handoff bundle into the current project root.",
    )
    handoff_restore_parser.add_argument("bundle_path", type=Path)
    handoff_restore_parser.add_argument("--target-root", type=Path)
    handoff_restore_parser.add_argument("--force", action="store_true")
    handoff_restore_parser.add_argument("--json", action="store_true")

    serve_parser = subparsers.add_parser(
        "serve",
        help="Run the local cockpit service and browser UI on this machine.",
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument(
        "--remote",
        action="store_true",
        help="Bind on 0.0.0.0 and require token auth for remote cockpit access.",
    )
    serve_parser.add_argument(
        "--token",
        help="Optional shared token required for remote cockpit API access.",
    )
    serve_parser.add_argument(
        "--display-host",
        help="Optional browser-facing host or IP to print in the startup URL.",
    )

    subparsers.add_parser("model-status", help="Print main-model backend status.")
    subparsers.add_parser("stats", help="Print basic database stats.")
    subparsers.add_parser("chat", help="Run a simple interactive shell.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "handoff-restore":
        manager = ProjectHandoffManager(workspace_root=Path.cwd())
        report = manager.restore_bundle(
            args.bundle_path,
            target_root=args.target_root,
            force=args.force,
        )
        if args.json:
            print(json.dumps(_handoff_restore_to_json(report), indent=2))
        else:
            print(report.render())
        return 0
    store = MemoryStore(args.db)
    try:
        if args.command == "observe":
            agent = MemoryFirstAgent(store)
            report = agent.observe_message(role=args.role, text=args.text)
            print(report.render())
            return 0

        if args.command == "remember":
            record = store.remember(
                MemoryDraft(
                    kind=args.kind,
                    subject=args.subject,
                    content=args.content,
                    tags=args.tags,
                    importance=args.importance,
                    confidence=args.confidence,
                )
            )
            print(json.dumps(_memory_to_json(record), indent=2))
            return 0

        if args.command == "search":
            results = store.search(args.query, limit=args.limit)
            payload = [
                {
                    "id": item.memory.id,
                    "kind": item.memory.kind,
                    "layer": item.memory.layer,
                    "content": item.memory.content,
                    "entities": [
                        _entity_to_json(link.entity)
                        for link in store.get_memory_entities(item.memory.id)
                    ],
                    "score": round(item.score, 4),
                    "reasons": item.reasons,
                }
                for item in results
            ]
            print(json.dumps(payload, indent=2))
            return 0

        if args.command == "entities":
            payload = [_entity_to_json(entity) for entity in store.resolve_entities(args.query)]
            print(json.dumps(payload, indent=2))
            return 0

        if args.command == "context":
            context = store.build_context(args.query, memory_limit=args.limit)
            print(context.render())
            return 0

        if args.command == "reply":
            agent = MemoryFirstAgent(store)
            report = agent.respond(args.text)
            if args.json:
                print(json.dumps(_reply_to_json(report), indent=2))
            else:
                print(report.render())
            return 0 if report.error is None else 1

        if args.command == "decide":
            agent = MemoryFirstAgent(store)
            report = agent.decide(
                args.text,
                execute_actions=not args.preview,
            )
            if args.json:
                print(json.dumps(_reply_to_json(report), indent=2))
            else:
                print(report.render())
            return 0 if _reply_succeeded(report) else 1

        if args.command == "plan":
            planner = MemoryPlanner(store)
            snapshot = planner.build_plan(args.query, action_limit=args.limit)
            if args.json:
                print(json.dumps(_plan_to_json(snapshot), indent=2))
            else:
                print(snapshot.render())
            return 0

        if args.command == "execute":
            executor = MemoryExecutor(store)
            cycle = executor.execute_next(args.query, action_limit=args.limit)
            if args.json:
                print(json.dumps(_execution_cycle_to_json(cycle), indent=2))
            else:
                print(cycle.render())
            return 0 if cycle.result.status != "error" else 1

        if args.command == "pilot":
            policy = LinuxPilotPolicy.load(args.policy_file, workspace_root=Path.cwd())
            runtime = LinuxPilotRuntime(store, policy=policy)
            report = runtime.run_turn(
                args.text,
                approve=args.approve,
                use_model=False if args.no_model else None,
                action_limit=args.limit,
            )
            if args.json:
                print(json.dumps(_pilot_turn_to_json(report), indent=2))
            else:
                print(report.render())
            if report.execution_result is None:
                return 0
            return 0 if report.execution_result.status != "error" else 1

        if args.command == "pilot-policy":
            policy = LinuxPilotPolicy.load(args.policy_file, workspace_root=Path.cwd())
            if args.write_template is not None:
                target = (
                    args.write_template
                    if args.write_template.is_absolute()
                    else (Path.cwd() / args.write_template)
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(policy.render_template(), encoding="utf-8")
                if args.json:
                    print(
                        json.dumps(
                            {
                                "written_to": str(target),
                                "policy": policy.status(),
                            },
                            indent=2,
                        )
                    )
                else:
                    print(f"Wrote pilot policy template to {target}")
                return 0
            if args.json:
                print(json.dumps(policy.status(), indent=2))
            else:
                print(policy.render_template())
            return 0

        if args.command == "pilot-report":
            reporter = PilotHistoryReporter(store)
            report = reporter.build(limit=args.limit)
            if args.json:
                print(json.dumps(_pilot_history_to_json(report), indent=2))
            else:
                print(report.render())
            return 0

        if args.command == "pilot-run":
            policy = LinuxPilotPolicy.load(args.policy_file, workspace_root=Path.cwd())
            runtime = LinuxPilotRuntime(store, policy=policy)
            report = runtime.run_session(
                args.text,
                max_steps=args.steps,
                auto_approve=args.approve,
                use_model=False if args.no_model else None,
                action_limit=args.limit,
            )
            reviewer = PilotRunReviewer(store)
            report.review = reviewer.review(
                report,
                promote_limit=args.promote_limit,
            )
            if args.json:
                print(json.dumps(_pilot_run_to_json(report), indent=2))
            else:
                print(report.render())
            last_step = report.steps[-1] if report.steps else None
            if last_step is not None and last_step.execution_result is not None:
                return 0 if last_step.execution_result.status != "error" else 1
            return 0

        if args.command == "pilot-chat":
            return _run_pilot_chat(
                store,
                policy_file=args.policy_file,
                use_model=False if args.no_model else None,
                action_limit=args.limit,
                auto_approve=args.auto_approve,
            )

        if args.command == "ready":
            payload = [_memory_to_json(task) for task in store.get_ready_tasks(limit=args.limit)]
            print(json.dumps(payload, indent=2))
            return 0

        if args.command == "overdue":
            payload = [_memory_to_json(task) for task in store.get_overdue_tasks(limit=args.limit)]
            print(json.dumps(payload, indent=2))
            return 0

        if args.command == "review-tasks":
            payload = [_memory_to_json(nudge) for nudge in store.review_tasks(limit=args.limit)]
            print(json.dumps(payload, indent=2))
            return 0

        if args.command == "complete-task":
            payload = {
                key: (_memory_to_json(value) if value is not None else None)
                for key, value in store.complete_task(
                    args.title,
                    area=args.area,
                    completed_at=args.completed_at,
                ).items()
            }
            print(json.dumps(payload, indent=2))
            return 0

        if args.command == "snooze-task":
            payload = _memory_to_json(
                store.snooze_task(
                    args.title,
                    until=args.until,
                    area=args.area,
                )
            )
            print(json.dumps(payload, indent=2))
            return 0

        if args.command == "unblock-task":
            payload = _memory_to_json(
                store.unblock_task(
                    args.title,
                    area=args.area,
                )
            )
            print(json.dumps(payload, indent=2))
            return 0

        if args.command == "resume-task":
            payload = _memory_to_json(
                store.resume_task(
                    args.title,
                    area=args.area,
                )
            )
            print(json.dumps(payload, indent=2))
            return 0

        if args.command == "reflect":
            consolidated = store.consolidate_recent(
                reflection_limit=args.limit,
                max_reflections=args.max_reflections,
                max_profiles=args.max_profiles,
            )
            payload = {
                layer: [
                    {
                        "id": memory.id,
                        "subject": memory.subject,
                        "layer": memory.layer,
                        "content": memory.content,
                        "sources": [
                            {
                                "type": source.source_type,
                                "id": source.source_id,
                                "relation": source.relation_type,
                            }
                            for source in store.get_memory_sources(memory.id)
                        ],
                    }
                    for memory in memories
                ]
                for layer, memories in consolidated.items()
            }
            print(json.dumps(payload, indent=2))
            return 0

        if args.command == "maintain":
            report = store.run_maintenance(
                force=args.force,
                reflection_limit=args.limit,
                max_reflections=args.max_reflections,
                max_profiles=args.max_profiles,
            )
            print(json.dumps(report, indent=2))
            return 0

        if args.command == "revise":
            revised = store.revise_memory(
                args.memory_id,
                args.content,
                importance=args.importance,
                confidence=args.confidence,
                tags=args.tags,
            )
            print(json.dumps(_memory_to_json(revised), indent=2))
            return 0

        if args.command == "history":
            memory = store.get_memory(args.memory_id)
            memory_entities = store.get_memory_entities(args.memory_id)
            primary_task_entity = next(
                (link.entity for link in memory_entities if link.entity.entity_type == "task"),
                None,
            )
            payload = {
                "memory": _memory_to_json(memory),
                "sources": [
                    {
                        "type": source.source_type,
                        "id": source.source_id,
                        "relation": source.relation_type,
                    }
                    for source in store.get_memory_sources(args.memory_id)
                ],
                "edges": [
                    {
                        "from": edge.from_memory_id,
                        "to": edge.to_memory_id,
                        "type": edge.edge_type,
                    }
                    for edge in store.get_memory_edges(args.memory_id)
                ],
                "entities": [
                    {
                        "confidence": link.confidence,
                        "evidence_text": link.evidence_text,
                        "entity": _entity_to_json(link.entity),
                    }
                    for link in memory_entities
                ],
                "entity_edges": (
                    [
                        {
                            "from": edge.from_entity_id,
                            "to": edge.to_entity_id,
                            "type": edge.edge_type,
                            "to_entity": _entity_to_json(store.get_entity(edge.to_entity_id)),
                        }
                        for edge in store.get_entity_edges(primary_task_entity.id, direction="outgoing")
                    ]
                    if primary_task_entity is not None
                    else []
                ),
            }
            print(json.dumps(payload, indent=2))
            return 0

        if args.command == "task":
            record = store.record_task(
                args.title,
                status=args.status,
                area=args.area,
                owner=args.owner,
                details=args.details,
                depends_on=args.depends_on,
                blocked_by=args.blocked_by,
                due_date=args.due_date,
                recurrence_days=args.recurrence_days,
                snoozed_until=args.snoozed_until,
                command=args.task_command,
                cwd=args.cwd,
                service_action=args.service_action,
                file_operation=args.file_op,
                file_path=args.file_path,
                file_text=args.file_text,
                find_text=args.find_text,
                symbol_name=args.symbol_name,
                replace_all=args.replace_all,
                complete_on_success=args.complete_on_success,
                retry_limit=args.retry_limit,
                retry_cooldown_minutes=args.retry_cooldown_minutes,
                tags=args.tags,
            )
            print(json.dumps(_memory_to_json(record), indent=2))
            return 0

        if args.command == "decision":
            record = store.record_decision(
                args.topic,
                args.decision,
                rationale=args.rationale,
                tags=args.tags,
            )
            print(json.dumps(_memory_to_json(record), indent=2))
            return 0

        if args.command == "tool-outcome":
            record = store.record_tool_outcome(
                args.tool_name,
                args.outcome,
                status=args.status,
                subject=args.subject,
                tags=args.tags,
            )
            print(json.dumps(_memory_to_json(record), indent=2))
            return 0

        if args.command == "evaluate":
            evaluator = MemoryEvaluator(Path.cwd())
            report = evaluator.run_builtin_suite()
            store.record_evaluation_run("builtin", report)
            if args.json:
                print(json.dumps(report.to_dict(), indent=2))
            else:
                print(report.render())
            return 0 if report.passed else 1

        if args.command == "improve":
            evaluator = MemoryEvaluator(Path.cwd())
            engine = MemoryImprovementEngine(
                store,
                evaluator,
                suite_name=args.suite_name,
            )
            review = engine.review(
                promote_limit=0 if args.preview else args.promote_limit,
                include_strategic_backlog=not args.no_strategic_backlog,
            )
            if args.json:
                print(json.dumps(_improvement_review_to_json(review), indent=2))
            else:
                print(review.render())
            return 0 if review.passed else 1

        if args.command == "patch-run":
            try:
                operations, validation_commands, apply_on_success, task_title = (
                    _resolve_patch_run_args(args)
                )
            except ValueError as exc:
                parser.error(str(exc))
            runner = WorkspacePatchRunner(
                store,
                workspace_root=Path.cwd(),
                suite_name=args.suite_name,
            )
            report = runner.run(
                args.run_name,
                operations=operations,
                validation_commands=validation_commands,
                apply_on_success=apply_on_success,
                task_title=task_title,
                task_area=args.task_area,
                suite_name=args.suite_name,
                git_mode=args.git_mode,
            )
            if args.json:
                print(json.dumps(_patch_run_to_json(report), indent=2))
            else:
                print(report.render())
            return 0 if report.status in {"accepted", "applied"} else 1

        if args.command == "patch-rollback":
            runner = WorkspacePatchRunner(
                store,
                workspace_root=Path.cwd(),
            )
            report = runner.rollback(args.run_id, force=args.force)
            if args.json:
                print(json.dumps(_patch_rollback_to_json(report), indent=2))
            else:
                print(report.render())
            return 0 if report.status == "rolled_back" else 1

        if args.command == "handoff-pack":
            manager = ProjectHandoffManager(store, workspace_root=Path.cwd())
            report = manager.create_bundle(
                output_dir=args.output_dir,
                include_traces=not args.skip_traces,
            )
            if args.json:
                print(json.dumps(_handoff_bundle_to_json(report), indent=2))
            else:
                print(report.render())
            return 0

        if args.command == "serve":
            config = _resolve_serve_config(args)
            serve_cockpit(
                store,
                host=str(config["host"]),
                port=int(config["port"]),
                workspace_root=Path.cwd(),
                auth_token=str(config["token"] or "") or None,
                access_url=str(config["access_url"] or "") or None,
            )
            return 0

        if args.command == "model-status":
            agent = MemoryFirstAgent(store)
            print(json.dumps(agent.model_status(), indent=2))
            return 0

        if args.command == "stats":
            print(json.dumps(store.stats(), indent=2))
            return 0

        if args.command == "chat":
            return _run_chat(store)

        parser.error(f"Unsupported command: {args.command}")
        return 2
    finally:
        store.close()


def _run_chat(store: MemoryStore) -> int:
    agent = MemoryFirstAgent(store)
    print("Memory-first agent shell")
    print("Type ':quit' to exit.")
    while True:
        text = input("you> ").strip()
        if not text:
            continue
        if text in {":quit", ":q", "exit"}:
            print("bye")
            return 0
        status = agent.model_status()
        if bool(status.get("enabled")):
            reply = agent.respond(text)
            print("")
            print(reply.render())
            print("")
            continue
        report = agent.observe_user_message(text)
        print("")
        print(report.render())
        print("")
        print(
            "model-backend> not configured. The memory context above is what we would send "
            "into the main model next."
        )
        print("")


def _resolve_serve_config(args: argparse.Namespace) -> dict[str, object]:
    host = str(getattr(args, "host", "127.0.0.1") or "127.0.0.1").strip() or "127.0.0.1"
    port = int(getattr(args, "port", 8765) or 8765)
    remote = bool(getattr(args, "remote", False))
    token = str(getattr(args, "token", "") or "").strip() or None
    display_host = str(getattr(args, "display_host", "") or "").strip() or None

    if remote:
        if host == "127.0.0.1":
            host = "0.0.0.0"
        if not token:
            token = secrets.token_hex(16)

    access_url: str | None = None
    if display_host:
        access_url = f"http://{display_host}:{port}/"
        if token:
            access_url += f"?token={token}"
    elif not remote and host not in {"0.0.0.0", "::"}:
        access_url = f"http://{host}:{port}/"

    return {
        "host": host,
        "port": port,
        "remote": remote,
        "token": token,
        "display_host": display_host,
        "access_url": access_url,
    }


def _run_pilot_chat(
    store: MemoryStore,
    *,
    policy_file: Path | None = None,
    use_model: bool | None = None,
    action_limit: int = 5,
    auto_approve: bool = False,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    policy = LinuxPilotPolicy.load(policy_file, workspace_root=Path.cwd())
    runtime = LinuxPilotRuntime(store, policy=policy)
    session_auto_approve = auto_approve
    last_report = None

    output_fn("Supervised Linux pilot session")
    output_fn("Type ':help' for commands and ':quit' to exit.")
    output_fn("")

    while True:
        try:
            raw_text = input_fn("pilot> ")
        except EOFError:
            output_fn("bye")
            return 0
        text = raw_text.strip()
        if not text:
            continue
        if text in {":quit", ":q", "exit"}:
            output_fn("bye")
            return 0
        if text == ":help":
            output_fn("Commands:")
            output_fn(":help - show this help")
            output_fn(":policy - show the active pilot policy")
            output_fn(":model - show the configured model backend status")
            output_fn(":status - show session settings")
            output_fn(":last - show the most recent pilot turn again")
            output_fn(":approve on - auto-approve gated actions for this session")
            output_fn(":approve off - require prompts for gated actions")
            output_fn(":quit - exit the pilot session")
            output_fn("")
            continue
        if text == ":policy":
            output_fn(json.dumps(policy.status(), indent=2))
            output_fn("")
            continue
        if text == ":model":
            output_fn(json.dumps(runtime.agent.model_status(), indent=2))
            output_fn("")
            continue
        if text == ":status":
            output_fn(
                json.dumps(
                    {
                        "auto_approve": session_auto_approve,
                        "use_model": policy.prefer_model if use_model is None else use_model,
                        "action_limit": action_limit,
                        "last_trace_path": (
                            last_report.trace_path if last_report is not None else None
                        ),
                    },
                    indent=2,
                )
            )
            output_fn("")
            continue
        if text == ":last":
            if last_report is None:
                output_fn("No pilot turn has run yet.")
            else:
                output_fn(last_report.render())
            output_fn("")
            continue
        if text == ":approve on":
            session_auto_approve = True
            output_fn("Session auto-approve is now on.")
            output_fn("")
            continue
        if text == ":approve off":
            session_auto_approve = False
            output_fn("Session auto-approve is now off.")
            output_fn("")
            continue

        report = runtime.run_turn(
            text,
            approve=session_auto_approve,
            use_model=use_model,
            action_limit=action_limit,
        )
        last_report = report
        output_fn(report.render())
        output_fn("")

        if (
            report.selected_action is not None
            and report.execution_result is None
            and report.approval is not None
            and report.approval.status == "needs_approval"
        ):
            try:
                approval_text = input_fn("approve> Run this action now? [y/N] ")
            except EOFError:
                output_fn("Approval input closed. No action executed.")
                output_fn("bye")
                return 0
            if approval_text.strip().lower() in {"y", "yes"}:
                report = runtime.approve_turn(report)
                last_report = report
                output_fn("Approved and executed queued action.")
                output_fn(report.render())
                output_fn("")
            else:
                output_fn("Approval declined. No action executed.")
                output_fn("")


def _memory_to_json(record) -> dict[str, object]:
    return {
        "id": record.id,
        "kind": record.kind,
        "subject": record.subject,
        "content": record.content,
        "layer": record.layer,
        "tags": record.tags,
        "importance": record.importance,
        "confidence": record.confidence,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "archived_at": record.archived_at,
        "metadata": record.metadata,
    }


def _entity_to_json(entity) -> dict[str, object]:
    return {
        "id": entity.id,
        "canonical_name": entity.canonical_name,
        "display_name": entity.display_name,
        "entity_type": entity.entity_type,
        "aliases": entity.aliases,
    }


def _plan_action_to_json(action) -> dict[str, object]:
    return {
        "kind": action.kind,
        "title": action.title,
        "summary": action.summary,
        "score": round(action.score, 4),
        "reasons": action.reasons,
        "task_id": action.task_id,
        "evidence_memory_ids": action.evidence_memory_ids,
        "metadata": action.metadata,
    }


def _plan_to_json(snapshot) -> dict[str, object]:
    return {
        "query": snapshot.query,
        "recommendation": (
            _plan_action_to_json(snapshot.recommendation)
            if snapshot.recommendation is not None
            else None
        ),
        "alternatives": [_plan_action_to_json(action) for action in snapshot.alternatives],
        "recent_nudges": [_memory_to_json(nudge) for nudge in snapshot.recent_nudges],
        "maintenance": snapshot.maintenance,
        "pilot_history": snapshot.pilot_history,
    }


def _execution_result_to_json(result) -> dict[str, object]:
    return {
        "requested_action": (
            _plan_action_to_json(result.requested_action)
            if result.requested_action is not None
            else None
        ),
        "executed_kind": result.executed_kind,
        "status": result.status,
        "summary": result.summary,
        "reasons": result.reasons,
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
        "shell_result": result.shell_result.to_dict() if result.shell_result is not None else None,
        "file_result": result.file_result.to_dict() if result.file_result is not None else None,
        "patch_run": _patch_run_to_json(result.patch_run) if result.patch_run is not None else None,
        "prompt": result.prompt,
        "metadata": result.metadata,
    }


def _execution_cycle_to_json(cycle) -> dict[str, object]:
    return {
        "query": cycle.query,
        "before_plan": _plan_to_json(cycle.before_plan),
        "result": _execution_result_to_json(cycle.result),
        "after_plan": _plan_to_json(cycle.after_plan),
    }


def _model_response_to_json(response) -> dict[str, object]:
    return {
        "content": response.content,
        "model": response.model,
        "role": response.role,
        "thinking": response.thinking,
        "done_reason": response.done_reason,
        "prompt_eval_count": response.prompt_eval_count,
        "eval_count": response.eval_count,
    }


def _model_action_to_json(action) -> dict[str, object]:
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


def _reply_to_json(report) -> dict[str, object]:
    return {
        "user_event_id": report.user_event_id,
        "user_memories": [_memory_to_json(memory) for memory in report.user_memories],
        "context": report.context.render() if report.context is not None else None,
        "plan": _plan_to_json(report.plan) if report.plan is not None else None,
        "assistant_event_id": report.assistant_event_id,
        "assistant_message": report.assistant_message,
        "assistant_memories": [_memory_to_json(memory) for memory in report.assistant_memories],
        "model_response": (
            _model_response_to_json(report.model_response)
            if report.model_response is not None
            else None
        ),
        "model_status": report.model_status,
        "model_action": (
            _model_action_to_json(report.model_action)
            if report.model_action is not None
            else None
        ),
        "execution_result": (
            _execution_result_to_json(report.execution_result)
            if report.execution_result is not None
            else None
        ),
        "after_plan": _plan_to_json(report.after_plan) if report.after_plan is not None else None,
        "error": report.error,
    }


def _reply_succeeded(report) -> bool:
    if report.error is not None:
        return False
    if report.execution_result is None:
        return True
    return report.execution_result.status != "error"


def _pilot_turn_to_json(report) -> dict[str, object]:
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


def _pilot_run_to_json(report) -> dict[str, object]:
    return {
        "goal_text": report.goal_text,
        "root_user_event_id": report.root_user_event_id,
        "max_steps": report.max_steps,
        "action_limit_used": report.action_limit_used,
        "auto_approve": report.auto_approve,
        "use_model": report.use_model,
        "stop_reason": report.stop_reason,
        "stop_summary": report.stop_summary,
        "executed_steps": report.executed_steps,
        "approval_requests": report.approval_requests,
        "approvals_granted": report.approvals_granted,
        "trace_paths": report.trace_paths,
        "steps": [_pilot_turn_to_json(step) for step in report.steps],
        "review": _pilot_review_to_json(report.review) if report.review is not None else None,
    }


def _pilot_review_to_json(review) -> dict[str, object]:
    return {
        "goal_text": review.goal_text,
        "stop_reason": review.stop_reason,
        "stop_summary": review.stop_summary,
        "executed_steps": review.executed_steps,
        "approval_requests": review.approval_requests,
        "approvals_granted": review.approvals_granted,
        "recurring_patterns": review.recurring_patterns,
        "opportunities": [
            _improvement_opportunity_to_json(item) for item in review.opportunities
        ],
        "promoted_tasks": [_memory_to_json(task) for task in review.promoted_tasks],
        "review_outcome": (
            _memory_to_json(review.review_outcome)
            if review.review_outcome is not None
            else None
        ),
    }


def _pilot_history_to_json(report) -> dict[str, object]:
    return {
        "total_reviews": report.total_reviews,
        "window_size": report.window_size,
        "recent_runs": [
            {
                "review_outcome_id": item.review_outcome_id,
                "created_at": item.created_at,
                "goal_text": item.goal_text,
                "stop_reason": item.stop_reason,
                "executed_steps": item.executed_steps,
                "approval_requests": item.approval_requests,
                "approvals_granted": item.approvals_granted,
                "opportunity_count": item.opportunity_count,
                "opportunity_categories": item.opportunity_categories,
                "recurring_patterns": item.recurring_patterns,
            }
            for item in report.recent_runs
        ],
        "stop_reasons": report.stop_reasons,
        "opportunity_categories": report.opportunity_categories,
        "recurring_patterns": report.recurring_patterns,
    }


def _improvement_opportunity_to_json(opportunity) -> dict[str, object]:
    return {
        "title": opportunity.title,
        "summary": opportunity.summary,
        "score": round(opportunity.score, 4),
        "category": opportunity.category,
        "details": opportunity.details,
        "source": opportunity.source,
        "metadata": opportunity.metadata,
    }


def _improvement_review_to_json(review) -> dict[str, object]:
    return {
        "suite_name": review.suite_name,
        "passed": review.passed,
        "current_evaluation": review.current_evaluation,
        "previous_evaluation": review.previous_evaluation,
        "best_evaluation": review.best_evaluation,
        "opportunities": [
            _improvement_opportunity_to_json(item) for item in review.opportunities
        ],
        "promoted_tasks": [_memory_to_json(task) for task in review.promoted_tasks],
        "review_outcome": (
            _memory_to_json(review.review_outcome)
            if review.review_outcome is not None
            else None
        ),
    }


def _resolve_patch_run_args(
    args,
) -> tuple[list[PatchOperation], list[str] | None, bool, str | None]:
    spec: dict[str, object] = {}
    if args.spec_file is not None:
        spec = json.loads(args.spec_file.read_text(encoding="utf-8-sig"))
        if not isinstance(spec, dict):
            raise ValueError("Patch spec file must contain a JSON object.")

    raw_operations = spec.get("operations")
    operations: list[PatchOperation] = []
    if raw_operations is not None:
        if not isinstance(raw_operations, list):
            raise ValueError("Patch spec 'operations' must be a JSON array.")
        operations = [_patch_operation_from_mapping(item) for item in raw_operations]
    elif args.file_op and args.file_path:
        operations = [
            PatchOperation(
                operation=args.file_op,
                path=args.file_path,
                text=args.file_text,
                find_text=args.find_text,
                symbol_name=args.symbol_name,
                replace_all=bool(args.replace_all),
            )
        ]
    else:
        raise ValueError(
            "Provide either --spec-file or an inline --file-op with --file-path."
        )

    validation_commands: list[str] | None
    if args.validate is not None:
        validation_commands = [str(item).strip() for item in args.validate if str(item).strip()]
    else:
        raw_commands = spec.get("validation_commands")
        if raw_commands is None:
            validation_commands = None
        else:
            if not isinstance(raw_commands, list):
                raise ValueError("Patch spec 'validation_commands' must be a JSON array.")
            validation_commands = [
                str(item).strip() for item in raw_commands if str(item).strip()
            ]

    spec_apply = bool(spec.get("apply_on_success", False))
    task_title = str(args.task_title or spec.get("task_title") or "").strip() or None
    for operation in operations:
        _validate_patch_operation(operation)
    return operations, validation_commands, bool(args.apply_on_success or spec_apply), task_title


def _patch_operation_from_mapping(payload: object) -> PatchOperation:
    if not isinstance(payload, dict):
        raise ValueError("Each patch operation must be a JSON object.")
    operation = str(payload.get("operation") or "").strip()
    path = str(payload.get("path") or "").strip()
    if not operation or not path:
        raise ValueError("Each patch operation requires both 'operation' and 'path'.")
    operation_spec = PatchOperation(
        operation=operation,
        path=path,
        text=payload.get("text"),
        find_text=payload.get("find_text"),
        symbol_name=str(payload.get("symbol_name") or "").strip() or None,
        replace_all=bool(payload.get("replace_all", False)),
        cwd=str(payload.get("cwd") or "").strip() or None,
    )
    _validate_patch_operation(operation_spec)
    return operation_spec


def _validate_patch_operation(operation: PatchOperation) -> None:
    requires_symbol = {
        "replace_python_function",
        "replace_python_class",
        "insert_python_before_symbol",
        "insert_python_after_symbol",
        "delete_python_symbol",
        "rename_python_identifier",
        "rename_python_method",
        "add_python_function_parameter",
        "add_python_method_parameter",
        "rename_python_export_across_imports",
        "move_python_export_to_module",
    }
    requires_text = {
        "write_text",
        "append_text",
        "replace_text",
        "replace_python_function",
        "replace_python_class",
        "insert_python_before_symbol",
        "insert_python_after_symbol",
        "rename_python_identifier",
        "rename_python_method",
        "add_python_import",
        "remove_python_import",
        "add_python_function_parameter",
        "add_python_method_parameter",
        "rename_python_export_across_imports",
        "move_python_export_to_module",
    }
    if operation.operation in requires_symbol and not str(operation.symbol_name or "").strip():
        raise ValueError(
            f"Patch operation '{operation.operation}' requires a non-empty symbol_name."
        )
    if operation.operation in requires_text and operation.text is None:
        raise ValueError(f"Patch operation '{operation.operation}' requires text.")


def _patch_validation_to_json(validation) -> dict[str, object]:
    return {
        "kind": validation.kind,
        "name": validation.name,
        "status": validation.status,
        "passed": validation.passed,
        "details": validation.details,
        "command_text": validation.command_text,
        "result": validation.result,
    }


def _patch_run_to_json(report) -> dict[str, object]:
    return {
        "run_id": report.run_id,
        "run_name": report.run_name,
        "suite_name": report.suite_name,
        "status": report.status,
        "task_title": report.task_title,
        "apply_on_success": report.apply_on_success,
        "applied": report.applied,
        "temp_workspace": report.temp_workspace,
        "changed_files": report.changed_files,
        "diff_preview": report.diff_preview,
        "operations": [operation.to_dict() for operation in report.operations],
        "operation_results": [result.to_dict() for result in report.operation_results],
        "validations": [
            _patch_validation_to_json(validation) for validation in report.validations
        ],
        "baseline_evaluation": report.baseline_evaluation,
        "candidate_evaluation": report.candidate_evaluation,
        "rejection_reason": report.rejection_reason,
        "preview_only": report.preview_only,
        "git_apply": report.git_apply.to_dict() if report.git_apply is not None else None,
        "tool_outcome": (
            _memory_to_json(report.tool_outcome)
            if report.tool_outcome is not None
            else None
        ),
        "task_update": (
            _memory_to_json(report.task_update)
            if report.task_update is not None
            else None
        ),
    }


def _patch_rollback_to_json(report) -> dict[str, object]:
    return {
        "status": report.status,
        "patch_run_id": report.patch_run_id,
        "repo_root": report.repo_root,
        "branch_name": report.branch_name,
        "original_branch": report.original_branch,
        "commit": report.commit,
        "switched_to": report.switched_to,
        "deleted_branch": report.deleted_branch,
        "reason": report.reason,
        "tool_outcome": (
            _memory_to_json(report.tool_outcome)
            if report.tool_outcome is not None
            else None
        ),
    }


def _handoff_bundle_to_json(report) -> dict[str, object]:
    return {
        "bundle_path": report.bundle_path,
        "output_dir": report.output_dir,
        "manifest_path": report.manifest_path,
        "summary_path": report.summary_path,
        "included_files": list(report.included_files),
        "created_at": report.created_at,
    }


def _handoff_restore_to_json(report) -> dict[str, object]:
    return {
        "bundle_path": report.bundle_path,
        "target_root": report.target_root,
        "restored_files": list(report.restored_files),
        "backup_dir": report.backup_dir,
        "created_at": report.created_at,
    }


if __name__ == "__main__":
    raise SystemExit(main())
