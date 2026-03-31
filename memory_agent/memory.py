from __future__ import annotations

import json
import math
import re
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import (
    CONTRADICTION_SCAN_EVENT_THRESHOLD,
    DEFAULT_MEMORY_LIMIT,
    DEFAULT_RECENT_EVENT_COUNT,
    DEFAULT_DB_PATH,
    SEMANTIC_RERANK_CANDIDATE_LIMIT,
    SEMANTIC_RERANK_WEIGHT,
    TASK_BLOCKED_STALE_DAYS,
    TASK_ESCALATION_NUDGE_COUNT,
    TASK_NUDGE_COOLDOWN_HOURS,
    TASK_REVIEW_EVENT_THRESHOLD,
    TASK_STALE_DAYS,
    PROFILE_REFLECTION_THRESHOLD,
    RECENCY_HALF_LIFE_DAYS,
    REFLECTION_EVENT_THRESHOLD,
    REFLECTION_MEMORY_THRESHOLD,
)
from .entity_resolution import HeuristicEntityResolver
from .extractors import HeuristicMemoryExtractor
from .models import (
    ContextWindow,
    EntityEdge,
    EntityRecord,
    Event,
    MemoryBundle,
    MemoryDraft,
    MemoryEdge,
    MemoryEntityLink,
    MemoryRecord,
    MemorySource,
    SearchResult,
    utc_now_iso,
)
from .reflection import HeuristicReflector
from .reranker import OptionalSemanticReranker


class MemoryStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("pragma foreign_keys = on")
        self.connection.execute("pragma journal_mode = wal")
        self.connection.execute("pragma synchronous = normal")
        self.extractor = HeuristicMemoryExtractor()
        self.reflector = HeuristicReflector()
        self.entity_resolver = HeuristicEntityResolver()
        self._init_schema()
        self.reranker = OptionalSemanticReranker(self.connection)

    def close(self) -> None:
        self.connection.close()

    def _init_schema(self) -> None:
        self.connection.executescript(
            """
            create table if not exists events (
                id integer primary key autoincrement,
                role text not null,
                content text not null,
                created_at text not null,
                metadata_json text not null default '{}'
            );

            create table if not exists memories (
                id integer primary key autoincrement,
                kind text not null,
                subject text not null,
                content text not null,
                normalized_content text not null,
                layer text not null default 'atomic',
                tags_json text not null default '[]',
                importance real not null default 0.5,
                confidence real not null default 0.7,
                source_event_id integer references events(id),
                created_at text not null,
                updated_at text not null,
                last_accessed_at text,
                access_count integer not null default 0,
                archived_at text,
                metadata_json text not null default '{}',
                unique(kind, subject, normalized_content)
            );

            create virtual table if not exists memories_fts using fts5(
                subject,
                content,
                tags
            );

            create table if not exists memory_sources (
                id integer primary key autoincrement,
                memory_id integer not null references memories(id) on delete cascade,
                source_type text not null,
                source_id integer not null,
                relation_type text not null,
                created_at text not null,
                unique(memory_id, source_type, source_id, relation_type)
            );

            create table if not exists memory_edges (
                id integer primary key autoincrement,
                from_memory_id integer not null references memories(id) on delete cascade,
                to_memory_id integer not null references memories(id) on delete cascade,
                edge_type text not null,
                created_at text not null,
                unique(from_memory_id, to_memory_id, edge_type)
            );

            create table if not exists maintenance_state (
                task_name text primary key,
                last_run_at text,
                last_event_id integer not null default 0,
                details_json text not null default '{}'
            );

            create table if not exists embedding_cache (
                model text not null,
                content_hash text not null,
                content text not null,
                embedding_json text not null,
                updated_at text not null,
                primary key(model, content_hash)
            );

            create table if not exists evaluation_runs (
                id integer primary key autoincrement,
                suite_name text not null,
                score real not null,
                passed integer not null default 0,
                scenarios_passed integer not null default 0,
                scenarios_total integer not null default 0,
                checks_passed integer not null default 0,
                checks_total integer not null default 0,
                summary_json text not null,
                created_at text not null
            );

            create table if not exists patch_runs (
                id integer primary key autoincrement,
                run_name text not null,
                suite_name text not null,
                task_title text,
                status text not null,
                baseline_score real,
                candidate_score real,
                apply_on_success integer not null default 0,
                applied integer not null default 0,
                workspace_path text not null,
                changed_files_json text not null default '[]',
                operation_results_json text not null default '[]',
                validation_results_json text not null default '[]',
                summary_json text not null default '{}',
                created_at text not null
            );

            create table if not exists entities (
                id integer primary key autoincrement,
                canonical_name text not null unique,
                display_name text not null,
                entity_type text not null,
                metadata_json text not null default '{}',
                created_at text not null,
                updated_at text not null
            );

            create table if not exists entity_aliases (
                id integer primary key autoincrement,
                entity_id integer not null references entities(id) on delete cascade,
                alias text not null,
                normalized_alias text not null,
                created_at text not null,
                unique(entity_id, normalized_alias)
            );

            create table if not exists memory_entities (
                id integer primary key autoincrement,
                memory_id integer not null references memories(id) on delete cascade,
                entity_id integer not null references entities(id) on delete cascade,
                confidence real not null default 0.7,
                evidence_text text not null default '',
                created_at text not null,
                unique(memory_id, entity_id)
            );

            create table if not exists entity_edges (
                id integer primary key autoincrement,
                from_entity_id integer not null references entities(id) on delete cascade,
                to_entity_id integer not null references entities(id) on delete cascade,
                edge_type text not null,
                created_at text not null,
                unique(from_entity_id, to_entity_id, edge_type)
            );

            create index if not exists idx_events_created_at on events(created_at desc);
            create index if not exists idx_memories_active on memories(archived_at, updated_at desc);
            create index if not exists idx_memories_kind on memories(kind, updated_at desc);
            create index if not exists idx_memory_sources_memory on memory_sources(memory_id, relation_type);
            create index if not exists idx_memory_edges_from on memory_edges(from_memory_id, edge_type);
            create index if not exists idx_memory_edges_to on memory_edges(to_memory_id, edge_type);
            create index if not exists idx_maintenance_state_task on maintenance_state(task_name);
            create index if not exists idx_embedding_cache_model on embedding_cache(model, updated_at desc);
            create index if not exists idx_evaluation_runs_suite on evaluation_runs(suite_name, created_at desc);
            create index if not exists idx_patch_runs_suite on patch_runs(suite_name, created_at desc);
            create index if not exists idx_entities_canonical_name on entities(canonical_name);
            create index if not exists idx_entity_aliases_normalized on entity_aliases(normalized_alias);
            create index if not exists idx_memory_entities_memory on memory_entities(memory_id, entity_id);
            create index if not exists idx_memory_entities_entity on memory_entities(entity_id, memory_id);
            create index if not exists idx_entity_edges_from on entity_edges(from_entity_id, edge_type);
            create index if not exists idx_entity_edges_to on entity_edges(to_entity_id, edge_type);
            """
        )
        self._ensure_memory_columns()
        self.connection.execute(
            "create index if not exists idx_memories_layer on memories(layer, archived_at, updated_at desc)"
        )
        self._seed_entity_catalog()
        self._backfill_source_links()
        self._backfill_memory_entities()
        self.connection.commit()

    def _ensure_memory_columns(self) -> None:
        columns = {
            row["name"]: row
            for row in self.connection.execute("pragma table_info(memories)").fetchall()
        }
        if "layer" not in columns:
            self.connection.execute(
                "alter table memories add column layer text not null default 'atomic'"
            )

    def _backfill_source_links(self) -> None:
        rows = self.connection.execute(
            """
            select id, source_event_id
            from memories
            where source_event_id is not null
            """
        ).fetchall()
        for row in rows:
            self._add_memory_source(
                memory_id=int(row["id"]),
                source_type="event",
                source_id=int(row["source_event_id"]),
                relation_type="derived_from",
            )

    def _seed_entity_catalog(self) -> None:
        for draft in self.entity_resolver.catalog():
            self._upsert_entity(
                canonical_name=draft.canonical_name,
                display_name=draft.display_name,
                entity_type=draft.entity_type,
                aliases=draft.aliases,
                metadata=draft.metadata,
            )

    def _backfill_memory_entities(self) -> None:
        rows = self.connection.execute(
            """
            select *
            from memories
            where not exists(
                select 1
                from memory_entities me
                where me.memory_id = memories.id
            )
            """
        ).fetchall()
        for row in rows:
            self._refresh_memory_entities(self._row_to_memory(row))

    def log_event(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> int:
        now = utc_now_iso()
        cursor = self.connection.execute(
            """
            insert into events(role, content, created_at, metadata_json)
            values (?, ?, ?, ?)
            """,
            (role, content, now, json.dumps(metadata or {}, sort_keys=True)),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def observe(
        self, role: str, content: str, metadata: dict[str, Any] | None = None
    ) -> tuple[int, list[MemoryRecord]]:
        event_id = self.log_event(role=role, content=content, metadata=metadata)
        stored: list[MemoryRecord] = []
        for draft in self.extractor.extract(role=role, text=content):
            stored.append(self.remember(draft, source_event_id=event_id))
        self.run_maintenance_if_due()
        return event_id, stored

    def remember(
        self,
        draft: MemoryDraft,
        source_event_id: int | None = None,
        source_event_ids: list[int] | None = None,
        source_memory_ids: list[int] | None = None,
        supersedes_memory_id: int | None = None,
        archive_superseded: bool = True,
    ) -> MemoryRecord:
        normalized = self._normalize_text(draft.content)
        now = utc_now_iso()
        event_ids = set(source_event_ids or [])
        if source_event_id is not None:
            event_ids.add(source_event_id)
        memory_source_ids = list(dict.fromkeys(source_memory_ids or []))
        existing = self.connection.execute(
            """
            select *
            from memories
            where kind = ? and subject = ? and normalized_content = ? and layer = ?
            limit 1
            """,
            (draft.kind, draft.subject, normalized, draft.layer),
        ).fetchone()

        if existing is None:
            cursor = self.connection.execute(
                """
                insert into memories(
                    kind,
                    subject,
                    content,
                    normalized_content,
                    layer,
                    tags_json,
                    importance,
                    confidence,
                    source_event_id,
                    created_at,
                    updated_at,
                    metadata_json
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    draft.kind,
                    draft.subject,
                    draft.content,
                    normalized,
                    draft.layer,
                    json.dumps(sorted(set(draft.tags)), sort_keys=True),
                    draft.importance,
                    draft.confidence,
                    source_event_id,
                    now,
                    now,
                    json.dumps(draft.metadata, sort_keys=True),
                ),
            )
            memory_id = int(cursor.lastrowid)
        else:
            existing_tags = set(json.loads(existing["tags_json"]))
            merged_tags = sorted(existing_tags.union(draft.tags))
            existing_metadata = json.loads(existing["metadata_json"])
            merged_metadata = {**existing_metadata, **draft.metadata}
            self.connection.execute(
                """
                update memories
                set content = ?,
                    layer = ?,
                    tags_json = ?,
                    importance = ?,
                    confidence = ?,
                    source_event_id = coalesce(source_event_id, ?),
                    updated_at = ?,
                    archived_at = null,
                    metadata_json = ?
                where id = ?
                """,
                (
                    draft.content,
                    draft.layer,
                    json.dumps(merged_tags, sort_keys=True),
                    max(existing["importance"], draft.importance),
                    max(existing["confidence"], draft.confidence),
                    source_event_id,
                    now,
                    json.dumps(merged_metadata, sort_keys=True),
                    existing["id"],
                ),
            )
            memory_id = int(existing["id"])

        self._upsert_fts(memory_id)
        for event_id in sorted(event_ids):
            self._add_memory_source(
                memory_id=memory_id,
                source_type="event",
                source_id=event_id,
                relation_type="derived_from",
            )
        for source_memory_id in memory_source_ids:
            self._add_memory_source(
                memory_id=memory_id,
                source_type="memory",
                source_id=source_memory_id,
                relation_type="derived_from",
            )
        if supersedes_memory_id is not None and supersedes_memory_id != memory_id:
            self._add_memory_edge(memory_id, supersedes_memory_id, "supersedes")
            self._remove_memory_edge(memory_id, supersedes_memory_id, "contradicts")
            self._remove_memory_edge(supersedes_memory_id, memory_id, "contradicts")
            self._add_memory_source(
                memory_id=memory_id,
                source_type="memory",
                source_id=supersedes_memory_id,
                relation_type="derived_from",
            )
            if archive_superseded:
                self._archive_memory(supersedes_memory_id)
        stored_memory = self.get_memory(memory_id)
        self._refresh_memory_entities(stored_memory)
        if stored_memory.layer == "atomic":
            self._link_contradictions_for_memory(stored_memory)
        self.connection.commit()
        return self.get_memory(memory_id)

    def get_memory(self, memory_id: int) -> MemoryRecord:
        row = self.connection.execute(
            "select * from memories where id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown memory id: {memory_id}")
        return self._row_to_memory(row)

    def find_active_task(
        self,
        title: str,
        *,
        area: str | None = None,
        decorate: bool = False,
    ) -> MemoryRecord | None:
        task = (
            self._find_active_task(title, area)
            if area is not None
            else self._find_active_task_any_area(title)
        )
        if task is None or not decorate:
            return task
        return self._decorate_task_for_execution(task)

    def record_task(
        self,
        title: str,
        *,
        status: str = "open",
        area: str = "execution",
        owner: str = "agent",
        details: str | None = None,
        depends_on: list[str] | None = None,
        blocked_by: list[str] | None = None,
        due_date: str | None = None,
        recurrence_days: int | None = None,
        snoozed_until: str | None = None,
        command: str | None = None,
        cwd: str | None = None,
        service_action: str | None = None,
        service_inspection: str | None = None,
        service_label: str | None = None,
        service_requires_confirmation: bool | None = None,
        service_confirmation_message: str | None = None,
        service_success_message: str | None = None,
        file_operation: str | None = None,
        file_path: str | None = None,
        file_text: str | None = None,
        find_text: str | None = None,
        symbol_name: str | None = None,
        replace_all: bool | None = None,
        complete_on_success: bool | None = None,
        retry_limit: int | None = None,
        retry_count: int | None = None,
        retry_cooldown_minutes: int | None = None,
        last_retry_at: str | None = None,
        last_failure_at: str | None = None,
        tags: list[str] | None = None,
        importance: float = 0.84,
        confidence: float = 0.88,
    ) -> MemoryRecord:
        clean_title = title.strip().rstrip(".")
        existing_task = self._find_active_task(clean_title, area)
        prior_metadata = existing_task.metadata if existing_task is not None else {}
        resolved_owner = owner or str(prior_metadata.get("owner") or "agent")
        resolved_details = details if details is not None else prior_metadata.get("details")
        resolved_due_date = due_date if due_date is not None else prior_metadata.get("due_date")
        resolved_recurrence_days = (
            recurrence_days
            if recurrence_days is not None
            else self._optional_int(prior_metadata.get("recurrence_days"))
        )
        resolved_snoozed_until = (
            snoozed_until
            if snoozed_until is not None
            else prior_metadata.get("snoozed_until")
        )
        resolved_command = command if command is not None else prior_metadata.get("command")
        resolved_cwd = cwd if cwd is not None else prior_metadata.get("cwd")
        resolved_service_action = (
            service_action if service_action is not None else prior_metadata.get("service_action")
        )
        resolved_service_inspection = (
            service_inspection
            if service_inspection is not None
            else prior_metadata.get("service_inspection")
        )
        resolved_service_label = (
            service_label if service_label is not None else prior_metadata.get("service_label")
        )
        resolved_service_requires_confirmation = (
            service_requires_confirmation
            if service_requires_confirmation is not None
            else bool(prior_metadata.get("service_requires_confirmation", False))
        )
        resolved_service_confirmation_message = (
            service_confirmation_message
            if service_confirmation_message is not None
            else prior_metadata.get("service_confirmation_message")
        )
        resolved_service_success_message = (
            service_success_message
            if service_success_message is not None
            else prior_metadata.get("service_success_message")
        )
        resolved_file_operation = (
            file_operation if file_operation is not None else prior_metadata.get("file_operation")
        )
        resolved_file_path = file_path if file_path is not None else prior_metadata.get("file_path")
        resolved_file_text = file_text if file_text is not None else prior_metadata.get("file_text")
        resolved_find_text = find_text if find_text is not None else prior_metadata.get("find_text")
        resolved_symbol_name = (
            symbol_name if symbol_name is not None else prior_metadata.get("symbol_name")
        )
        resolved_replace_all = (
            replace_all
            if replace_all is not None
            else bool(prior_metadata.get("replace_all", False))
        )
        resolved_complete_on_success = (
            complete_on_success
            if complete_on_success is not None
            else bool(prior_metadata.get("complete_on_success", False))
        )
        resolved_retry_limit = (
            retry_limit
            if retry_limit is not None
            else self._optional_int(prior_metadata.get("retry_limit"))
        )
        resolved_retry_count = (
            retry_count
            if retry_count is not None
            else self._optional_int(prior_metadata.get("retry_count"))
        )
        resolved_retry_cooldown_minutes = (
            retry_cooldown_minutes
            if retry_cooldown_minutes is not None
            else self._optional_int(prior_metadata.get("retry_cooldown_minutes"))
        )
        resolved_last_retry_at = (
            last_retry_at if last_retry_at is not None else prior_metadata.get("last_retry_at")
        )
        resolved_last_failure_at = (
            last_failure_at
            if last_failure_at is not None
            else prior_metadata.get("last_failure_at")
        )
        resolved_cycle_key = str(prior_metadata.get("cycle_key") or uuid.uuid4().hex)
        dependency_titles = self._normalize_task_titles(
            depends_on if depends_on is not None else list(prior_metadata.get("depends_on", []))
        )
        blocker_titles = self._normalize_task_titles(
            blocked_by if blocked_by is not None else list(prior_metadata.get("blocked_by", []))
        )
        content = f"Task [{status}]: {clean_title}."
        if resolved_due_date:
            content += f" Due: {str(resolved_due_date).strip()}."
        if dependency_titles:
            content += f" Depends on: {', '.join(dependency_titles)}."
        if blocker_titles:
            content += f" Blocked by: {', '.join(blocker_titles)}."
        if resolved_recurrence_days:
            content += f" Recurs every {resolved_recurrence_days} days."
        if resolved_snoozed_until:
            content += f" Snoozed until: {str(resolved_snoozed_until).strip()}."
        if resolved_retry_limit:
            content += (
                f" Retry policy: {max(int(resolved_retry_count or 0), 0)}/"
                f"{max(int(resolved_retry_limit), 0)} attempts used."
            )
            if resolved_retry_cooldown_minutes:
                content += (
                    f" Cooldown: {max(int(resolved_retry_cooldown_minutes), 0)} minutes."
                )
            if resolved_last_failure_at:
                content += f" Last failure: {str(resolved_last_failure_at).strip()}."
            if resolved_last_retry_at:
                content += f" Last retry: {str(resolved_last_retry_at).strip()}."
        if resolved_command:
            content += f" Command: {str(resolved_command).strip()}."
        if resolved_service_action:
            content += f" Service action: {str(resolved_service_action).strip()}."
        if resolved_service_inspection:
            content += f" Service inspection: {str(resolved_service_inspection).strip()}."
        if resolved_service_label:
            content += f" Service label: {str(resolved_service_label).strip()}."
        if resolved_service_requires_confirmation:
            content += " Service confirmation: required."
        if resolved_file_operation and resolved_file_path:
            content += (
                f" File operation: {str(resolved_file_operation).strip()} "
                f"on {str(resolved_file_path).strip()}."
            )
        if resolved_symbol_name:
            content += f" Target symbol: {str(resolved_symbol_name).strip()}."
        if resolved_details:
            content += f" Details: {str(resolved_details).strip().rstrip('.')}."
        draft = MemoryDraft(
            kind="task",
            subject=area,
            content=content,
            tags=sorted(set((tags or []) + ["task", area, status])),
            importance=importance,
            confidence=confidence,
            metadata={
                "title": clean_title,
                "status": status,
                "owner": resolved_owner,
                "details": resolved_details,
                "depends_on": dependency_titles,
                "blocked_by": blocker_titles,
                "due_date": str(resolved_due_date).strip() if resolved_due_date else None,
                "recurrence_days": resolved_recurrence_days,
                "snoozed_until": (
                    str(resolved_snoozed_until).strip() if resolved_snoozed_until else None
                ),
                "command": str(resolved_command).strip() if resolved_command else None,
                "cwd": str(resolved_cwd).strip() if resolved_cwd else None,
                "service_action": (
                    str(resolved_service_action).strip() if resolved_service_action else None
                ),
                "service_inspection": (
                    str(resolved_service_inspection).strip()
                    if resolved_service_inspection
                    else None
                ),
                "service_label": (
                    str(resolved_service_label).strip() if resolved_service_label else None
                ),
                "service_requires_confirmation": bool(
                    resolved_service_requires_confirmation
                ),
                "service_confirmation_message": (
                    str(resolved_service_confirmation_message).strip()
                    if resolved_service_confirmation_message
                    else None
                ),
                "service_success_message": (
                    str(resolved_service_success_message).strip()
                    if resolved_service_success_message
                    else None
                ),
                "file_operation": (
                    str(resolved_file_operation).strip() if resolved_file_operation else None
                ),
                "file_path": str(resolved_file_path).strip() if resolved_file_path else None,
                "file_text": resolved_file_text,
                "find_text": resolved_find_text,
                "symbol_name": (
                    str(resolved_symbol_name).strip() if resolved_symbol_name else None
                ),
                "replace_all": bool(resolved_replace_all),
                "complete_on_success": bool(resolved_complete_on_success),
                "retry_limit": resolved_retry_limit,
                "retry_count": resolved_retry_count or 0,
                "retry_cooldown_minutes": resolved_retry_cooldown_minutes,
                "last_retry_at": (
                    str(resolved_last_retry_at).strip() if resolved_last_retry_at else None
                ),
                "last_failure_at": (
                    str(resolved_last_failure_at).strip() if resolved_last_failure_at else None
                ),
                "cycle_key": resolved_cycle_key,
            },
        )
        if existing_task is None:
            task = self.remember(draft)
        elif self._normalize_text(existing_task.content) == self._normalize_text(content):
            task = self.remember(draft)
        else:
            task = self.remember(
                draft,
                source_memory_ids=[existing_task.id],
                supersedes_memory_id=existing_task.id,
            )
        self._sync_task_entity_edges(
            task,
            depends_on=dependency_titles,
            blocked_by=blocker_titles,
        )
        self.connection.commit()
        return self.get_memory(task.id)

    def complete_task(
        self,
        title: str,
        *,
        area: str = "execution",
        completed_at: str | None = None,
    ) -> dict[str, MemoryRecord | None]:
        current = self._find_active_task(title, area)
        if current is None:
            raise KeyError(f"Unknown active task: {title} [{area}]")
        completed = self.record_task(
            title,
            status="done",
            area=area,
            owner=str(current.metadata.get("owner") or "agent"),
            details=current.metadata.get("details"),
            depends_on=list(current.metadata.get("depends_on", [])),
            blocked_by=[],
            due_date=current.metadata.get("due_date"),
            recurrence_days=self._optional_int(current.metadata.get("recurrence_days")),
            snoozed_until=None,
            command=current.metadata.get("command"),
            cwd=current.metadata.get("cwd"),
            service_action=current.metadata.get("service_action"),
            service_inspection=current.metadata.get("service_inspection"),
            service_label=current.metadata.get("service_label"),
            service_requires_confirmation=bool(
                current.metadata.get("service_requires_confirmation", False)
            ),
            service_confirmation_message=current.metadata.get("service_confirmation_message"),
            service_success_message=current.metadata.get("service_success_message"),
            file_operation=current.metadata.get("file_operation"),
            file_path=current.metadata.get("file_path"),
            file_text=current.metadata.get("file_text"),
            find_text=current.metadata.get("find_text"),
            symbol_name=current.metadata.get("symbol_name"),
            replace_all=bool(current.metadata.get("replace_all", False)),
            complete_on_success=bool(current.metadata.get("complete_on_success", False)),
            retry_limit=self._optional_int(current.metadata.get("retry_limit")),
            retry_count=self._optional_int(current.metadata.get("retry_count")),
            retry_cooldown_minutes=self._optional_int(current.metadata.get("retry_cooldown_minutes")),
            last_retry_at=current.metadata.get("last_retry_at"),
            last_failure_at=current.metadata.get("last_failure_at"),
            tags=[tag for tag in current.tags if tag not in {"open", "in_progress", "blocked"}],
            importance=current.importance,
            confidence=current.confidence,
        )
        next_occurrence: MemoryRecord | None = None
        recurrence_days = self._optional_int(completed.metadata.get("recurrence_days"))
        if recurrence_days and recurrence_days > 0:
            next_due_date = self._shift_temporal_value(
                str(completed.metadata.get("due_date") or "").strip() or completed_at or utc_now_iso(),
                recurrence_days,
            )
            next_occurrence = self.record_task(
                title,
                status="open",
                area=area,
                owner=str(completed.metadata.get("owner") or "agent"),
                details=completed.metadata.get("details"),
                depends_on=list(completed.metadata.get("depends_on", [])),
                blocked_by=[],
                due_date=next_due_date,
                recurrence_days=recurrence_days,
                snoozed_until=None,
                command=completed.metadata.get("command"),
                cwd=completed.metadata.get("cwd"),
                service_action=completed.metadata.get("service_action"),
                service_inspection=completed.metadata.get("service_inspection"),
                service_label=completed.metadata.get("service_label"),
                service_requires_confirmation=bool(
                    completed.metadata.get("service_requires_confirmation", False)
                ),
                service_confirmation_message=completed.metadata.get("service_confirmation_message"),
                service_success_message=completed.metadata.get("service_success_message"),
                file_operation=completed.metadata.get("file_operation"),
                file_path=completed.metadata.get("file_path"),
                file_text=completed.metadata.get("file_text"),
                find_text=completed.metadata.get("find_text"),
                symbol_name=completed.metadata.get("symbol_name"),
                replace_all=bool(completed.metadata.get("replace_all", False)),
                complete_on_success=bool(completed.metadata.get("complete_on_success", False)),
                retry_limit=self._optional_int(completed.metadata.get("retry_limit")),
                retry_count=0,
                retry_cooldown_minutes=self._optional_int(completed.metadata.get("retry_cooldown_minutes")),
                last_retry_at=None,
                last_failure_at=None,
                tags=[tag for tag in completed.tags if tag not in {"done"}],
                importance=completed.importance,
                confidence=completed.confidence,
            )
            self._add_memory_edge(next_occurrence.id, completed.id, "recurs_from")
            self._add_memory_edge(completed.id, next_occurrence.id, "recurs_to")
            self.connection.commit()
        return {
            "completed": self.get_memory(completed.id),
            "next_occurrence": self.get_memory(next_occurrence.id) if next_occurrence else None,
        }

    def snooze_task(
        self,
        title: str,
        *,
        until: str,
        area: str = "execution",
    ) -> MemoryRecord:
        current = self._find_active_task(title, area)
        if current is None:
            raise KeyError(f"Unknown active task: {title} [{area}]")
        return self.record_task(
            title,
            status=str(current.metadata.get("status", "open")),
            area=area,
            owner=str(current.metadata.get("owner") or "agent"),
            details=current.metadata.get("details"),
            depends_on=list(current.metadata.get("depends_on", [])),
            blocked_by=list(current.metadata.get("blocked_by", [])),
            due_date=current.metadata.get("due_date"),
            recurrence_days=self._optional_int(current.metadata.get("recurrence_days")),
            snoozed_until=until,
            command=current.metadata.get("command"),
            cwd=current.metadata.get("cwd"),
            service_action=current.metadata.get("service_action"),
            service_inspection=current.metadata.get("service_inspection"),
            service_label=current.metadata.get("service_label"),
            service_requires_confirmation=bool(
                current.metadata.get("service_requires_confirmation", False)
            ),
            service_confirmation_message=current.metadata.get("service_confirmation_message"),
            service_success_message=current.metadata.get("service_success_message"),
            file_operation=current.metadata.get("file_operation"),
            file_path=current.metadata.get("file_path"),
            file_text=current.metadata.get("file_text"),
            find_text=current.metadata.get("find_text"),
            symbol_name=current.metadata.get("symbol_name"),
            replace_all=bool(current.metadata.get("replace_all", False)),
            complete_on_success=bool(current.metadata.get("complete_on_success", False)),
            retry_limit=self._optional_int(current.metadata.get("retry_limit")),
            retry_count=self._optional_int(current.metadata.get("retry_count")),
            retry_cooldown_minutes=self._optional_int(current.metadata.get("retry_cooldown_minutes")),
            last_retry_at=current.metadata.get("last_retry_at"),
            last_failure_at=current.metadata.get("last_failure_at"),
            tags=list(current.tags),
            importance=current.importance,
            confidence=current.confidence,
        )

    def unblock_task(
        self,
        title: str,
        *,
        area: str = "execution",
    ) -> MemoryRecord:
        current = self._find_active_task(title, area)
        if current is None:
            raise KeyError(f"Unknown active task: {title} [{area}]")
        return self.record_task(
            title,
            status="open",
            area=area,
            owner=str(current.metadata.get("owner") or "agent"),
            details=current.metadata.get("details"),
            depends_on=list(current.metadata.get("depends_on", [])),
            blocked_by=[],
            due_date=current.metadata.get("due_date"),
            recurrence_days=self._optional_int(current.metadata.get("recurrence_days")),
            snoozed_until=current.metadata.get("snoozed_until"),
            command=current.metadata.get("command"),
            cwd=current.metadata.get("cwd"),
            service_action=current.metadata.get("service_action"),
            service_inspection=current.metadata.get("service_inspection"),
            service_label=current.metadata.get("service_label"),
            service_requires_confirmation=bool(
                current.metadata.get("service_requires_confirmation", False)
            ),
            service_confirmation_message=current.metadata.get("service_confirmation_message"),
            service_success_message=current.metadata.get("service_success_message"),
            file_operation=current.metadata.get("file_operation"),
            file_path=current.metadata.get("file_path"),
            file_text=current.metadata.get("file_text"),
            find_text=current.metadata.get("find_text"),
            symbol_name=current.metadata.get("symbol_name"),
            replace_all=bool(current.metadata.get("replace_all", False)),
            complete_on_success=bool(current.metadata.get("complete_on_success", False)),
            retry_limit=self._optional_int(current.metadata.get("retry_limit")),
            retry_count=self._optional_int(current.metadata.get("retry_count")),
            retry_cooldown_minutes=self._optional_int(current.metadata.get("retry_cooldown_minutes")),
            last_retry_at=current.metadata.get("last_retry_at"),
            last_failure_at=current.metadata.get("last_failure_at"),
            tags=[tag for tag in current.tags if tag != "blocked"],
            importance=current.importance,
            confidence=current.confidence,
        )

    def resume_task(
        self,
        title: str,
        *,
        area: str = "execution",
    ) -> MemoryRecord:
        current = self._find_active_task(title, area)
        if current is None:
            raise KeyError(f"Unknown active task: {title} [{area}]")
        return self.record_task(
            title,
            status=str(current.metadata.get("status", "open")),
            area=area,
            owner=str(current.metadata.get("owner") or "agent"),
            details=current.metadata.get("details"),
            depends_on=list(current.metadata.get("depends_on", [])),
            blocked_by=list(current.metadata.get("blocked_by", [])),
            due_date=current.metadata.get("due_date"),
            recurrence_days=self._optional_int(current.metadata.get("recurrence_days")),
            snoozed_until="",
            command=current.metadata.get("command"),
            cwd=current.metadata.get("cwd"),
            service_action=current.metadata.get("service_action"),
            service_inspection=current.metadata.get("service_inspection"),
            service_label=current.metadata.get("service_label"),
            service_requires_confirmation=bool(
                current.metadata.get("service_requires_confirmation", False)
            ),
            service_confirmation_message=current.metadata.get("service_confirmation_message"),
            service_success_message=current.metadata.get("service_success_message"),
            file_operation=current.metadata.get("file_operation"),
            file_path=current.metadata.get("file_path"),
            file_text=current.metadata.get("file_text"),
            find_text=current.metadata.get("find_text"),
            symbol_name=current.metadata.get("symbol_name"),
            replace_all=bool(current.metadata.get("replace_all", False)),
            complete_on_success=bool(current.metadata.get("complete_on_success", False)),
            retry_limit=self._optional_int(current.metadata.get("retry_limit")),
            retry_count=self._optional_int(current.metadata.get("retry_count")),
            retry_cooldown_minutes=self._optional_int(current.metadata.get("retry_cooldown_minutes")),
            last_retry_at=current.metadata.get("last_retry_at"),
            last_failure_at=current.metadata.get("last_failure_at"),
            tags=list(current.tags),
            importance=current.importance,
            confidence=current.confidence,
        )

    def record_decision(
        self,
        topic: str,
        decision: str,
        *,
        rationale: str | None = None,
        tags: list[str] | None = None,
        importance: float = 0.9,
        confidence: float = 0.9,
    ) -> MemoryRecord:
        clean_decision = decision.strip().rstrip(".")
        content = f"Decision: {clean_decision}."
        if rationale:
            content += f" Rationale: {rationale.strip().rstrip('.')}."
        return self.remember(
            MemoryDraft(
                kind="decision",
                subject=topic,
                content=content,
                tags=sorted(set((tags or []) + ["decision", topic])),
                importance=importance,
                confidence=confidence,
                metadata={
                    "decision": clean_decision,
                    "rationale": rationale,
                },
            )
        )

    def record_tool_outcome(
        self,
        tool_name: str,
        outcome: str,
        *,
        status: str = "success",
        subject: str = "tooling",
        tags: list[str] | None = None,
        importance: float = 0.76,
        confidence: float = 0.88,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        clean_outcome = outcome.strip().rstrip(".")
        return self.remember(
            MemoryDraft(
                kind="tool_outcome",
                subject=subject,
                content=f"Tool outcome [{status}] {tool_name}: {clean_outcome}.",
                tags=sorted(set((tags or []) + ["tooling", tool_name, status])),
                importance=importance,
                confidence=confidence,
                metadata={
                    "tool_name": tool_name,
                    "status": status,
                    "outcome": clean_outcome,
                    **(metadata or {}),
                },
            )
        )

    def _record_task_nudge(
        self,
        *,
        task: MemoryRecord,
        nudge_type: str,
        content: str,
        importance: float,
        confidence: float,
        metadata: dict[str, Any],
    ) -> MemoryRecord:
        existing = self._latest_task_nudge(task, nudge_type)
        cycle_key = str(task.metadata.get("cycle_key") or "")
        draft = MemoryDraft(
            kind="nudge",
            subject=task.subject,
            content=content,
            tags=sorted(
                {
                    "nudge",
                    "task-review",
                    nudge_type,
                    str(task.metadata.get("status", "open")),
                    task.subject,
                }
            ),
            importance=importance,
            confidence=confidence,
            metadata={**metadata, "task_cycle_key": cycle_key},
        )
        nudge = self.remember(
            draft,
            source_memory_ids=[task.id],
            supersedes_memory_id=existing.id if existing is not None else None,
            archive_superseded=True,
        )
        self._add_memory_edge(nudge.id, task.id, "nudges")
        return self.get_memory(nudge.id)

    def revise_memory(
        self,
        memory_id: int,
        content: str,
        *,
        importance: float | None = None,
        confidence: float | None = None,
        tags: list[str] | None = None,
    ) -> MemoryRecord:
        current = self.get_memory(memory_id)
        if self._normalize_text(current.content) == self._normalize_text(content):
            return current
        next_draft = MemoryDraft(
            kind=current.kind,
            subject=current.subject,
            content=content,
            tags=sorted(set(current.tags).union(tags or [])),
            importance=current.importance if importance is None else importance,
            confidence=current.confidence if confidence is None else confidence,
            metadata={**current.metadata, "revised_from_memory_id": current.id},
            layer=current.layer,
        )
        return self.remember(
            next_draft,
            source_memory_ids=[current.id],
            supersedes_memory_id=current.id,
        )

    def reflect_recent(
        self,
        *,
        limit: int = 20,
        max_reflections: int = 5,
    ) -> list[MemoryRecord]:
        rows = self.connection.execute(
            """
            select *
            from memories
            where archived_at is null and layer = 'atomic'
            order by updated_at desc, id desc
            limit ?
            """,
            (limit,),
        ).fetchall()
        candidates = [self._row_to_memory(row) for row in rows]
        groups = self.reflector.build_reflection_groups(candidates)
        groups.sort(key=lambda item: item[0].importance, reverse=True)

        reflections: list[MemoryRecord] = []
        for draft, source_memory_ids in groups[:max_reflections]:
            latest = self._get_latest_active_memory(
                kind="reflection",
                subject=draft.subject,
                layer="reflection",
            )
            supersedes_memory_id = None
            if latest is not None and self._normalize_text(latest.content) != self._normalize_text(
                draft.content
            ):
                supersedes_memory_id = latest.id
            reflection = self.remember(
                draft,
                source_memory_ids=source_memory_ids,
                supersedes_memory_id=supersedes_memory_id,
            )
            for source_memory_id in source_memory_ids:
                self._add_memory_edge(reflection.id, source_memory_id, "compacts")
            self.connection.commit()
            reflections.append(self.get_memory(reflection.id))
        return reflections

    def synthesize_profiles(
        self,
        *,
        limit: int = 30,
        max_profiles: int = 4,
    ) -> list[MemoryRecord]:
        rows = self.connection.execute(
            """
            select *
            from memories
            where archived_at is null and layer in ('atomic', 'reflection')
            order by updated_at desc, id desc
            limit ?
            """,
            (limit,),
        ).fetchall()
        candidates = [self._row_to_memory(row) for row in rows]
        groups = self.reflector.build_profile_groups(candidates)
        groups.sort(key=lambda item: item[0].importance, reverse=True)

        profiles: list[MemoryRecord] = []
        for draft, source_memory_ids in groups[:max_profiles]:
            latest = self._get_latest_active_memory(
                kind="profile",
                subject=draft.subject,
                layer="profile",
            )
            supersedes_memory_id = None
            if latest is not None and self._normalize_text(latest.content) != self._normalize_text(
                draft.content
            ):
                supersedes_memory_id = latest.id
            profile = self.remember(
                draft,
                source_memory_ids=source_memory_ids,
                supersedes_memory_id=supersedes_memory_id,
            )
            for source_memory_id in source_memory_ids:
                self._add_memory_edge(profile.id, source_memory_id, "abstracts")
            self.connection.commit()
            profiles.append(self.get_memory(profile.id))
        return profiles

    def consolidate_recent(
        self,
        *,
        reflection_limit: int = 20,
        max_reflections: int = 5,
        profile_limit: int = 30,
        max_profiles: int = 4,
    ) -> dict[str, list[MemoryRecord]]:
        reflections = self.reflect_recent(limit=reflection_limit, max_reflections=max_reflections)
        profiles = self.synthesize_profiles(limit=profile_limit, max_profiles=max_profiles)
        return {
            "reflections": reflections,
            "profiles": profiles,
        }

    def maintenance_status(self) -> dict[str, dict[str, Any]]:
        current_event_id = self._current_event_id()
        updated_atomic = self._count_updated_memories_since("atomic", "contradiction_scan")
        updated_atomic_for_reflection = self._count_updated_memories_since("atomic", "reflection")
        updated_reflections = self._count_updated_memories_since("reflection", "profile")
        active_atomic = self._count_active_memories("atomic")
        active_reflections = self._count_active_memories("reflection")
        active_tasks = self._count_reviewable_tasks()
        updated_tasks = self._count_updated_tasks_since("task_review")
        pending_task_nudges = self._count_pending_task_nudges()
        contradiction_state = self._get_maintenance_state("contradiction_scan")
        reflection_state = self._get_maintenance_state("reflection")
        profile_state = self._get_maintenance_state("profile")
        task_review_state = self._get_maintenance_state("task_review")
        return {
            "contradiction_scan": {
                "due": updated_atomic >= CONTRADICTION_SCAN_EVENT_THRESHOLD and active_atomic >= 2,
                "updated_atomic": updated_atomic,
                "active_atomic": active_atomic,
                "last_run_at": contradiction_state["last_run_at"],
                "last_event_id": contradiction_state["last_event_id"],
                "current_event_id": current_event_id,
            },
            "reflection": {
                "due": (
                    updated_atomic_for_reflection >= REFLECTION_EVENT_THRESHOLD
                    and active_atomic >= REFLECTION_MEMORY_THRESHOLD
                ),
                "updated_atomic": updated_atomic_for_reflection,
                "active_atomic": active_atomic,
                "last_run_at": reflection_state["last_run_at"],
                "last_event_id": reflection_state["last_event_id"],
                "current_event_id": current_event_id,
            },
            "profile": {
                "due": (
                    updated_reflections >= PROFILE_REFLECTION_THRESHOLD
                    and active_reflections >= PROFILE_REFLECTION_THRESHOLD
                ),
                "updated_reflections": updated_reflections,
                "active_reflections": active_reflections,
                "last_run_at": profile_state["last_run_at"],
                "last_event_id": profile_state["last_event_id"],
                "current_event_id": current_event_id,
            },
            "task_review": {
                "due": (
                    active_tasks >= 1
                    and (
                        task_review_state["last_run_at"] is None
                        or updated_tasks >= TASK_REVIEW_EVENT_THRESHOLD
                        or pending_task_nudges >= 1
                    )
                ),
                "updated_tasks": updated_tasks,
                "active_tasks": active_tasks,
                "pending_nudges": pending_task_nudges,
                "last_run_at": task_review_state["last_run_at"],
                "last_event_id": task_review_state["last_event_id"],
                "current_event_id": current_event_id,
            },
        }

    def run_maintenance(
        self,
        *,
        force: bool = False,
        reflection_limit: int = 20,
        max_reflections: int = 5,
        profile_limit: int = 30,
        max_profiles: int = 4,
    ) -> dict[str, Any]:
        status_before = self.maintenance_status()
        executed: dict[str, Any] = {}

        if force or status_before["contradiction_scan"]["due"]:
            contradictions = self.scan_for_contradictions()
            executed["contradiction_scan"] = {
                "contradiction_pairs": contradictions,
                "contradictions_found": len(contradictions),
            }
            self._mark_maintenance_run(
                "contradiction_scan",
                {
                    "contradictions_found": len(contradictions),
                },
            )

        if force or status_before["reflection"]["due"]:
            reflections = self.reflect_recent(
                limit=reflection_limit,
                max_reflections=max_reflections,
            )
            executed["reflection"] = {
                "created": [reflection.id for reflection in reflections],
                "count": len(reflections),
            }
            self._mark_maintenance_run(
                "reflection",
                {
                    "created_ids": [reflection.id for reflection in reflections],
                },
            )

        profile_status = self.maintenance_status()["profile"]
        if force or profile_status["due"] or "reflection" in executed:
            profiles = self.synthesize_profiles(
                limit=profile_limit,
                max_profiles=max_profiles,
            )
            executed["profile"] = {
                "created": [profile.id for profile in profiles],
                "count": len(profiles),
            }
            self._mark_maintenance_run(
                "profile",
                {
                    "created_ids": [profile.id for profile in profiles],
                },
            )

        task_review_status = self.maintenance_status()["task_review"]
        if force or task_review_status["due"]:
            nudges = self.review_tasks()
            executed["task_review"] = {
                "created": [nudge.id for nudge in nudges],
                "count": len(nudges),
            }
            self._mark_maintenance_run(
                "task_review",
                {
                    "created_ids": [nudge.id for nudge in nudges],
                    "count": len(nudges),
                },
            )

        return {
            "force": force,
            "before": status_before,
            "executed": executed,
            "after": self.maintenance_status(),
        }

    def service_sync_status(
        self,
        settings: dict[str, Any],
        *,
        area: str = "execution",
    ) -> dict[str, Any]:
        onboarding = settings.get("onboarding") if isinstance(settings, dict) else {}
        action_items = onboarding.get("actions") if isinstance(onboarding, dict) else []
        if not isinstance(action_items, list):
            action_items = []

        recommended_actions: list[str] = []
        missing_titles: list[str] = []
        stale_titles: list[str] = []

        for raw_item in action_items:
            if not isinstance(raw_item, dict):
                continue
            action_name = str(raw_item.get("action") or "").strip()
            if not action_name or not bool(raw_item.get("enabled", True)):
                continue
            recommended_actions.append(action_name)
            title = self._service_sync_task_title(raw_item)
            existing = self.find_active_task(title, area=area, decorate=True)
            if existing is None:
                missing_titles.append(title)
                continue
            existing_action = str(existing.metadata.get("service_action") or "").strip()
            if existing_action != action_name:
                stale_titles.append(title)

        return {
            "due": bool(missing_titles or stale_titles),
            "recommended_actions": sorted(dict.fromkeys(recommended_actions)),
            "missing_titles": missing_titles,
            "stale_titles": stale_titles,
            "recommended_count": len(set(recommended_actions)),
        }

    def sync_service_tasks(
        self,
        settings: dict[str, Any],
        *,
        area: str = "execution",
    ) -> dict[str, Any]:
        onboarding = settings.get("onboarding") if isinstance(settings, dict) else {}
        action_items = onboarding.get("actions") if isinstance(onboarding, dict) else []
        if not isinstance(action_items, list):
            action_items = []

        created: list[dict[str, Any]] = []
        updated: list[dict[str, Any]] = []
        unchanged: list[dict[str, Any]] = []
        resolved: list[dict[str, Any]] = []
        recommended_actions: set[str] = set()

        for raw_item in action_items:
            if not isinstance(raw_item, dict):
                continue
            action_name = str(raw_item.get("action") or "").strip()
            if not action_name or not bool(raw_item.get("enabled", True)):
                continue
            recommended_actions.add(action_name)
            title = self._service_sync_task_title(raw_item)
            details = self._service_sync_task_details(raw_item)
            existing = self.find_active_task(title, area=area, decorate=True)
            status = (
                str(existing.metadata.get("status", "open"))
                if existing is not None
                else "open"
            )
            task = self.record_task(
                title,
                status=status if status in {"open", "in_progress", "blocked"} else "open",
                area=area,
                owner="agent",
                details=details,
                service_action=action_name,
                service_label=raw_item.get("label"),
                service_requires_confirmation=bool(
                    raw_item.get("requires_confirmation", False)
                ),
                service_confirmation_message=raw_item.get("confirmation_message"),
                service_success_message=raw_item.get("success_message"),
                complete_on_success=True,
                tags=["cockpit", "service-action", "service-sync"],
                importance=self._service_sync_importance(action_name),
                confidence=0.92,
            )
            task_summary = {
                "task_id": task.id,
                "title": title,
                "service_action": action_name,
            }
            if existing is None:
                created.append(task_summary)
            elif existing.id == task.id:
                unchanged.append(task_summary)
            else:
                updated.append(task_summary)

        for task in self._reviewable_task_views():
            action_name = str(task.metadata.get("service_action") or "").strip()
            if not action_name:
                continue
            if "service-sync" not in set(task.tags):
                continue
            if action_name in recommended_actions:
                continue
            title = str(task.metadata.get("title") or task.content)
            completion = self.complete_task(title, area=task.subject)
            completed = completion["completed"]
            resolved.append(
                {
                    "task_id": completed.id if completed is not None else task.id,
                    "title": title,
                    "service_action": action_name,
                }
            )

        return {
            "recommended_actions": sorted(recommended_actions),
            "created": created,
            "updated": updated,
            "unchanged": unchanged,
            "resolved": resolved,
            "count": len(created) + len(updated) + len(unchanged),
        }

    def run_maintenance_if_due(self) -> dict[str, Any] | None:
        status = self.maintenance_status()
        if not any(task["due"] for task in status.values()):
            return None
        return self.run_maintenance()

    def get_open_tasks(
        self,
        *,
        limit: int = 5,
        statuses: tuple[str, ...] = ("blocked", "in_progress", "open"),
    ) -> list[MemoryRecord]:
        tasks = [
            self._decorate_task_for_execution(memory)
            for memory in self._active_task_memories()
            if str(memory.metadata.get("status", "open")) in statuses
        ]
        tasks.sort(key=self._task_queue_sort_key, reverse=True)
        return tasks[:limit]

    def get_ready_tasks(
        self,
        *,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        tasks = [
            self._decorate_task_for_execution(memory)
            for memory in self._active_task_memories()
        ]
        ready = [task for task in tasks if bool(task.metadata.get("ready_now"))]
        ready.sort(key=self._task_queue_sort_key, reverse=True)
        return ready[:limit]

    def get_overdue_tasks(
        self,
        *,
        limit: int = 5,
        statuses: tuple[str, ...] = ("blocked", "in_progress", "open"),
    ) -> list[MemoryRecord]:
        tasks = [
            self._decorate_task_for_execution(memory)
            for memory in self._active_task_memories()
            if str(memory.metadata.get("status", "open")) in statuses
        ]
        overdue = [task for task in tasks if bool(task.metadata.get("overdue"))]
        overdue.sort(key=self._task_queue_sort_key, reverse=True)
        return overdue[:limit]

    def get_recent_nudges(self, *, limit: int = 5) -> list[MemoryRecord]:
        rows = self.connection.execute(
            """
            select *
            from memories
            where archived_at is null and kind = 'nudge' and layer = 'atomic'
            order by updated_at desc, id desc
            limit ?
            """,
            (limit,),
        ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def review_tasks(self, *, limit: int = 10) -> list[MemoryRecord]:
        candidates = self._task_review_candidates()
        candidates.sort(
            key=lambda item: (
                item["priority"],
                self._task_queue_sort_key(item["task"]),
            ),
            reverse=True,
        )

        nudges: list[MemoryRecord] = []
        for candidate in candidates[:limit]:
            task = candidate["task"]
            nudge = self._record_task_nudge(
                task=task,
                nudge_type=candidate["nudge_type"],
                content=candidate["content"],
                importance=float(candidate["importance"]),
                confidence=float(candidate["confidence"]),
                metadata=dict(candidate["metadata"]),
            )
            nudges.append(nudge)
        self.connection.commit()
        return nudges

    def scan_for_contradictions(self, *, limit: int = 50) -> list[tuple[int, int]]:
        rows = self.connection.execute(
            """
            select *
            from memories
            where archived_at is null and layer = 'atomic'
            order by updated_at desc, id desc
            limit ?
            """,
            (limit,),
        ).fetchall()
        contradictions: list[tuple[int, int]] = []
        for row in rows:
            memory = self._row_to_memory(row)
            contradictions.extend(self._link_contradictions_for_memory(memory))
        unique_pairs = list(dict.fromkeys(contradictions))
        self.connection.commit()
        return unique_pairs

    def resolve_entities(self, query: str) -> list[EntityRecord]:
        normalized_query = self._normalize_text(query)
        if not normalized_query:
            return []
        rows = self.connection.execute(
            """
            select
                e.id,
                e.canonical_name,
                e.display_name,
                e.entity_type,
                e.metadata_json,
                e.created_at,
                e.updated_at,
                ea.alias
            from entities e
            join entity_aliases ea on ea.entity_id = e.id
            order by e.display_name asc, ea.alias asc
            """
        ).fetchall()
        matched: dict[int, dict[str, Any]] = {}
        for row in rows:
            alias = row["alias"]
            if not self._contains_alias(normalized_query, alias):
                continue
            existing = matched.setdefault(
                int(row["id"]),
                {
                    "id": int(row["id"]),
                    "canonical_name": row["canonical_name"],
                    "display_name": row["display_name"],
                    "entity_type": row["entity_type"],
                    "metadata": json.loads(row["metadata_json"]),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "aliases": [],
                },
            )
            existing["aliases"].append(alias)
        entities = [
            EntityRecord(
                id=data["id"],
                canonical_name=data["canonical_name"],
                display_name=data["display_name"],
                entity_type=data["entity_type"],
                aliases=sorted(dict.fromkeys(data["aliases"])),
                metadata=data["metadata"],
                created_at=data["created_at"],
                updated_at=data["updated_at"],
            )
            for data in matched.values()
        ]
        entities.sort(
            key=lambda item: (-len(max(item.aliases, key=len, default="")), item.display_name)
        )
        return entities

    def get_entity(self, entity_id: int) -> EntityRecord:
        row = self.connection.execute(
            """
            select id, canonical_name, display_name, entity_type, metadata_json, created_at, updated_at
            from entities
            where id = ?
            """,
            (entity_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown entity id: {entity_id}")
        return self._row_to_entity(row)

    def get_memory_entities(self, memory_id: int) -> list[MemoryEntityLink]:
        rows = self.connection.execute(
            """
            select
                me.memory_id,
                me.confidence,
                me.evidence_text,
                me.created_at,
                e.id as entity_id,
                e.canonical_name,
                e.display_name,
                e.entity_type,
                e.metadata_json,
                e.created_at as entity_created_at,
                e.updated_at as entity_updated_at
            from memory_entities me
            join entities e on e.id = me.entity_id
            where me.memory_id = ?
            order by me.confidence desc, e.display_name asc
            """,
            (memory_id,),
        ).fetchall()
        return [self._row_to_memory_entity_link(row) for row in rows]

    def get_entity_edges(
        self,
        entity_id: int,
        *,
        direction: str = "both",
    ) -> list[EntityEdge]:
        clauses: list[str] = []
        parameters: list[int] = []
        if direction in {"both", "outgoing"}:
            clauses.append("from_entity_id = ?")
            parameters.append(entity_id)
        if direction in {"both", "incoming"}:
            clauses.append("to_entity_id = ?")
            parameters.append(entity_id)
        if not clauses:
            raise ValueError(f"Unsupported direction: {direction}")
        rows = self.connection.execute(
            f"""
            select from_entity_id, to_entity_id, edge_type, created_at
            from entity_edges
            where {' or '.join(clauses)}
            order by created_at asc
            """,
            tuple(parameters),
        ).fetchall()
        return [self._row_to_entity_edge(row) for row in rows]

    def get_memory_sources(self, memory_id: int) -> list[MemorySource]:
        rows = self.connection.execute(
            """
            select memory_id, source_type, source_id, relation_type, created_at
            from memory_sources
            where memory_id = ?
            order by source_type asc, source_id asc
            """,
            (memory_id,),
        ).fetchall()
        return [self._row_to_memory_source(row) for row in rows]

    def get_memory_edges(
        self,
        memory_id: int,
        *,
        direction: str = "both",
    ) -> list[MemoryEdge]:
        clauses: list[str] = []
        parameters: list[int] = []
        if direction in {"both", "outgoing"}:
            clauses.append("from_memory_id = ?")
            parameters.append(memory_id)
        if direction in {"both", "incoming"}:
            clauses.append("to_memory_id = ?")
            parameters.append(memory_id)
        if not clauses:
            raise ValueError(f"Unsupported direction: {direction}")
        rows = self.connection.execute(
            f"""
            select from_memory_id, to_memory_id, edge_type, created_at
            from memory_edges
            where {' or '.join(clauses)}
            order by created_at asc
            """,
            tuple(parameters),
        ).fetchall()
        return [self._row_to_memory_edge(row) for row in rows]

    def recent_events(self, limit: int = DEFAULT_RECENT_EVENT_COUNT) -> list[Event]:
        rows = self.connection.execute(
            """
            select *
            from events
            order by id desc
            limit ?
            """,
            (limit,),
        ).fetchall()
        return [self._row_to_event(row) for row in reversed(rows)]

    def get_recent_tool_outcomes(
        self,
        *,
        limit: int = 10,
        statuses: tuple[str, ...] | None = None,
        subject: str | None = None,
        tool_name: str | None = None,
    ) -> list[MemoryRecord]:
        rows = self.connection.execute(
            """
            select *
            from memories
            where archived_at is null and kind = 'tool_outcome'
            order by updated_at desc, id desc
            limit ?
            """,
            (max(limit * 5, limit),),
        ).fetchall()
        results: list[MemoryRecord] = []
        status_filter = {item.strip() for item in statuses or () if item and item.strip()}
        subject_filter = str(subject or "").strip()
        tool_name_filter = str(tool_name or "").strip()
        for row in rows:
            memory = self._row_to_memory(row)
            tool_status = str(memory.metadata.get("status") or "").strip()
            if status_filter and tool_status not in status_filter:
                continue
            if subject_filter and memory.subject != subject_filter:
                continue
            if tool_name_filter and str(memory.metadata.get("tool_name") or "").strip() != tool_name_filter:
                continue
            results.append(memory)
            if len(results) >= limit:
                break
        return results

    def record_evaluation_run(self, suite_name: str, report: Any) -> dict[str, Any]:
        payload = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        scenario_results = payload.get("scenario_results", [])
        scenarios_total = len(scenario_results)
        scenarios_passed = sum(1 for scenario in scenario_results if bool(scenario.get("passed")))
        checks_total = sum(len(scenario.get("checks", [])) for scenario in scenario_results)
        checks_passed = sum(
            1
            for scenario in scenario_results
            for check in scenario.get("checks", [])
            if bool(check.get("passed"))
        )
        created_at = utc_now_iso()
        cursor = self.connection.execute(
            """
            insert into evaluation_runs(
                suite_name,
                score,
                passed,
                scenarios_passed,
                scenarios_total,
                checks_passed,
                checks_total,
                summary_json,
                created_at
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                suite_name,
                float(payload.get("score", 0.0) or 0.0),
                1 if bool(payload.get("passed")) else 0,
                scenarios_passed,
                scenarios_total,
                checks_passed,
                checks_total,
                json.dumps(payload, sort_keys=True),
                created_at,
            ),
        )
        run_id = int(cursor.lastrowid)
        self.connection.commit()
        self.record_tool_outcome(
            "evaluation",
            (
                f"Suite '{suite_name}' scored {float(payload.get('score', 0.0) or 0.0):.1%} "
                f"with {checks_passed}/{checks_total} checks passing"
            ),
            status="success" if bool(payload.get("passed")) else "error",
            subject="self_improvement",
            tags=["evaluation", suite_name],
            metadata={
                "evaluation_run_id": run_id,
                "suite_name": suite_name,
                "score": float(payload.get("score", 0.0) or 0.0),
                "scenarios_passed": scenarios_passed,
                "scenarios_total": scenarios_total,
                "checks_passed": checks_passed,
                "checks_total": checks_total,
            },
        )
        recorded = self.get_evaluation_run(run_id)
        return recorded

    def get_evaluation_run(self, run_id: int) -> dict[str, Any]:
        row = self.connection.execute(
            """
            select *
            from evaluation_runs
            where id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown evaluation run id: {run_id}")
        return self._row_to_evaluation_run(row)

    def latest_evaluation_run(
        self,
        *,
        suite_name: str | None = None,
        offset: int = 0,
    ) -> dict[str, Any] | None:
        if suite_name is None:
            row = self.connection.execute(
                """
                select *
                from evaluation_runs
                order by created_at desc, id desc
                limit 1 offset ?
                """,
                (offset,),
            ).fetchone()
        else:
            row = self.connection.execute(
                """
                select *
                from evaluation_runs
                where suite_name = ?
                order by created_at desc, id desc
                limit 1 offset ?
                """,
                (suite_name, offset),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_evaluation_run(row)

    def best_evaluation_run(self, *, suite_name: str | None = None) -> dict[str, Any] | None:
        if suite_name is None:
            row = self.connection.execute(
                """
                select *
                from evaluation_runs
                order by score desc, created_at asc, id asc
                limit 1
                """
            ).fetchone()
        else:
            row = self.connection.execute(
                """
                select *
                from evaluation_runs
                where suite_name = ?
                order by score desc, created_at asc, id asc
                limit 1
                """,
                (suite_name,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_evaluation_run(row)

    def record_patch_run(
        self,
        *,
        run_name: str,
        suite_name: str,
        task_title: str | None,
        status: str,
        baseline_evaluation: dict[str, Any] | None = None,
        candidate_evaluation: dict[str, Any] | None = None,
        apply_on_success: bool = False,
        applied: bool = False,
        workspace_path: str,
        changed_files: list[str] | None = None,
        operation_results: list[dict[str, Any]] | None = None,
        validation_results: list[dict[str, Any]] | None = None,
        summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        created_at = utc_now_iso()
        payload = {
            "baseline_evaluation": baseline_evaluation,
            "candidate_evaluation": candidate_evaluation,
            **(summary or {}),
        }
        cursor = self.connection.execute(
            """
            insert into patch_runs(
                run_name,
                suite_name,
                task_title,
                status,
                baseline_score,
                candidate_score,
                apply_on_success,
                applied,
                workspace_path,
                changed_files_json,
                operation_results_json,
                validation_results_json,
                summary_json,
                created_at
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_name,
                suite_name,
                task_title,
                status,
                (
                    float(baseline_evaluation.get("score", 0.0) or 0.0)
                    if baseline_evaluation is not None
                    else None
                ),
                (
                    float(candidate_evaluation.get("score", 0.0) or 0.0)
                    if candidate_evaluation is not None
                    else None
                ),
                1 if apply_on_success else 0,
                1 if applied else 0,
                workspace_path,
                json.dumps(changed_files or [], sort_keys=True),
                json.dumps(operation_results or [], sort_keys=True),
                json.dumps(validation_results or [], sort_keys=True),
                json.dumps(payload, sort_keys=True),
                created_at,
            ),
        )
        patch_run_id = int(cursor.lastrowid)
        self.connection.commit()
        return self.get_patch_run(patch_run_id)

    def get_patch_run(self, patch_run_id: int) -> dict[str, Any]:
        row = self.connection.execute(
            """
            select *
            from patch_runs
            where id = ?
            """,
            (patch_run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown patch run id: {patch_run_id}")
        return self._row_to_patch_run(row)

    def latest_patch_run(
        self,
        *,
        suite_name: str | None = None,
        offset: int = 0,
    ) -> dict[str, Any] | None:
        if suite_name is None:
            row = self.connection.execute(
                """
                select *
                from patch_runs
                order by created_at desc, id desc
                limit 1 offset ?
                """,
                (offset,),
            ).fetchone()
        else:
            row = self.connection.execute(
                """
                select *
                from patch_runs
                where suite_name = ?
                order by created_at desc, id desc
                limit 1 offset ?
                """,
                (suite_name, offset),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_patch_run(row)

    def _query_fts_rows(
        self,
        fts_query: str,
        *,
        layers: tuple[str, ...] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        layer_clause, layer_params = self._layer_filter_clause("m", layers)
        rows = self.connection.execute(
            """
            select
                m.*,
                bm25(memories_fts, 2.0, 1.0, 0.4) as raw_text_score,
                exists(
                    select 1
                    from memory_edges e
                    join memories r on r.id = e.from_memory_id
                    where e.edge_type = 'compacts'
                      and e.to_memory_id = m.id
                      and r.archived_at is null
                ) as compacted_signal,
                (
                    select count(*)
                    from memory_sources ms
                    where ms.memory_id = m.id and ms.source_type = 'memory'
                ) as source_memory_count,
                (
                    select count(*)
                    from memory_edges e
                    join memories other on other.id = e.to_memory_id
                    where e.from_memory_id = m.id
                      and e.edge_type = 'contradicts'
                      and other.archived_at is null
                ) as contradiction_count
            from memories_fts
            join memories m on m.id = memories_fts.rowid
            where memories_fts match ? and m.archived_at is null
            """
            + layer_clause
            + """
            limit ?
            """,
            (fts_query, *layer_params, max(limit * 5, 10)),
        ).fetchall()
        return [dict(row) for row in rows]

    def _query_fallback_rows(
        self,
        *,
        layers: tuple[str, ...] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        layer_clause, layer_params = self._layer_filter_clause("memories", layers)
        rows = self.connection.execute(
            """
            select
                *,
                10.0 as raw_text_score,
                exists(
                    select 1
                    from memory_edges e
                    join memories r on r.id = e.from_memory_id
                    where e.edge_type = 'compacts'
                      and e.to_memory_id = memories.id
                      and r.archived_at is null
                ) as compacted_signal,
                (
                    select count(*)
                    from memory_sources ms
                    where ms.memory_id = memories.id and ms.source_type = 'memory'
                ) as source_memory_count,
                (
                    select count(*)
                    from memory_edges e
                    join memories other on other.id = e.to_memory_id
                    where e.from_memory_id = memories.id
                      and e.edge_type = 'contradicts'
                      and other.archived_at is null
                ) as contradiction_count
            from memories
            where archived_at is null
            """
            + layer_clause
            + """
            order by importance desc, updated_at desc
            limit ?
            """,
            (*layer_params, max(limit * 5, 10)),
        ).fetchall()
        return [dict(row) for row in rows]

    def _query_entity_rows(
        self,
        entity_ids: list[int],
        *,
        layers: tuple[str, ...] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        if not entity_ids:
            return []
        placeholders = ", ".join("?" for _ in entity_ids)
        layer_clause, layer_params = self._layer_filter_clause("m", layers)
        rows = self.connection.execute(
            """
            select
                m.*,
                6.0 as raw_text_score,
                exists(
                    select 1
                    from memory_edges e
                    join memories r on r.id = e.from_memory_id
                    where e.edge_type = 'compacts'
                      and e.to_memory_id = m.id
                      and r.archived_at is null
                ) as compacted_signal,
                (
                    select count(*)
                    from memory_sources ms
                    where ms.memory_id = m.id and ms.source_type = 'memory'
                ) as source_memory_count,
                (
                    select count(*)
                    from memory_edges e
                    join memories other on other.id = e.to_memory_id
                    where e.from_memory_id = m.id
                      and e.edge_type = 'contradicts'
                      and other.archived_at is null
                ) as contradiction_count
            from memories m
            join memory_entities me on me.memory_id = m.id
            where m.archived_at is null
              and me.entity_id in ("""
            + placeholders
            + ")"
            + layer_clause
            + """
            group by m.id
            order by max(me.confidence) desc, m.importance desc, m.updated_at desc
            limit ?
            """,
            (*entity_ids, *layer_params, max(limit * 5, 10)),
        ).fetchall()
        return [dict(row) for row in rows]

    def _merge_candidate_rows(
        self,
        primary: list[dict[str, Any]],
        secondary: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[int, dict[str, Any]] = {int(row["id"]): dict(row) for row in primary}
        for row in secondary:
            row_id = int(row["id"])
            existing = merged.get(row_id)
            if existing is None:
                merged[row_id] = dict(row)
                continue
            existing["raw_text_score"] = min(
                float(existing["raw_text_score"]),
                float(row["raw_text_score"]),
            )
            existing["compacted_signal"] = max(
                int(existing["compacted_signal"]),
                int(row["compacted_signal"]),
            )
            existing["source_memory_count"] = max(
                int(existing["source_memory_count"]),
                int(row["source_memory_count"]),
            )
            existing["contradiction_count"] = max(
                int(existing["contradiction_count"]),
                int(row["contradiction_count"]),
            )
        return list(merged.values())

    def _entity_names_by_memory_id(self, memory_ids: list[int]) -> dict[int, set[str]]:
        if not memory_ids:
            return {}
        placeholders = ", ".join("?" for _ in memory_ids)
        rows = self.connection.execute(
            """
            select me.memory_id, e.canonical_name
            from memory_entities me
            join entities e on e.id = me.entity_id
            where me.memory_id in ("""
            + placeholders
            + ")",
            tuple(memory_ids),
        ).fetchall()
        names: dict[int, set[str]] = {memory_id: set() for memory_id in memory_ids}
        for row in rows:
            names[int(row["memory_id"])].add(row["canonical_name"])
        return names

    def search(
        self,
        query: str,
        limit: int = DEFAULT_MEMORY_LIMIT,
        *,
        layers: tuple[str, ...] | None = None,
    ) -> list[SearchResult]:
        fts_query = self._build_fts_query(query)
        query_tokens = self._tokenize(query)
        query_entities = self.resolve_entities(query)
        query_entity_names = {entity.canonical_name for entity in query_entities}
        query_entity_ids = [entity.id for entity in query_entities]

        rows = (
            self._query_fts_rows(fts_query, layers=layers, limit=limit)
            if fts_query
            else self._query_fallback_rows(layers=layers, limit=limit)
        )
        if query_entity_ids:
            rows = self._merge_candidate_rows(
                rows,
                self._query_entity_rows(query_entity_ids, layers=layers, limit=limit),
            )

        memory_entity_names = self._entity_names_by_memory_id([int(row["id"]) for row in rows])

        ranked: list[SearchResult] = []
        for row in rows:
            memory = self._row_to_memory(row)
            score, reasons = self._score_memory(
                memory,
                raw_text_score=row["raw_text_score"],
                compacted_signal=bool(row["compacted_signal"]),
                source_memory_count=int(row["source_memory_count"]),
                contradiction_count=int(row["contradiction_count"]),
                query_tokens=query_tokens,
                query_entity_names=query_entity_names,
                memory_entity_names=memory_entity_names.get(memory.id, set()),
            )
            if query_entity_names:
                overlap = sorted(query_entity_names & memory_entity_names.get(memory.id, set()))
                if overlap:
                    reasons.append("entities=" + ",".join(overlap))
            ranked.append(SearchResult(memory=memory, score=score, reasons=reasons))

        semantic_candidates = ranked[:SEMANTIC_RERANK_CANDIDATE_LIMIT]
        semantic_scores = self.reranker.rerank(query, semantic_candidates)
        for item in semantic_candidates:
            semantic = semantic_scores.get(item.memory.id)
            if semantic is None:
                continue
            item.score += SEMANTIC_RERANK_WEIGHT * semantic.bonus
            item.reasons.append(f"semantic={semantic.similarity:.2f}")

        ranked.sort(key=lambda item: item.score, reverse=True)
        top_results = ranked[:limit]
        if top_results:
            self._touch_memories([item.memory.id for item in top_results])
        return top_results

    def build_context(
        self,
        query: str,
        memory_limit: int = DEFAULT_MEMORY_LIMIT,
        recent_event_count: int = DEFAULT_RECENT_EVENT_COUNT,
    ) -> ContextWindow:
        profiles = self.search(query=query, limit=max(2, memory_limit // 2), layers=("profile",))
        if not profiles:
            profiles = self.search("", limit=max(2, memory_limit // 2), layers=("profile",))
        memories = self.search(
            query=query,
            limit=memory_limit,
            layers=("reflection", "atomic"),
        )
        return ContextWindow(
            query=query,
            memories=memories,
            profiles=profiles,
            bundles=self._build_memory_bundles(memories),
            ready_tasks=self.get_ready_tasks(limit=max(3, memory_limit)),
            overdue_tasks=self.get_overdue_tasks(limit=max(3, memory_limit)),
            open_tasks=self.get_open_tasks(limit=max(3, memory_limit)),
            recent_events=self.recent_events(limit=recent_event_count),
        )

    def stats(self) -> dict[str, Any]:
        events_count = self.connection.execute("select count(*) from events").fetchone()[0]
        memories_count = self.connection.execute(
            "select count(*) from memories where archived_at is null"
        ).fetchone()[0]
        by_kind_rows = self.connection.execute(
            """
            select kind, count(*) as count
            from memories
            where archived_at is null
            group by kind
            order by count desc, kind asc
            """
        ).fetchall()
        by_layer_rows = self.connection.execute(
            """
            select layer, count(*) as count
            from memories
            where archived_at is null
            group by layer
            order by count desc, layer asc
            """
        ).fetchall()
        contradiction_edges = self.connection.execute(
            """
            select count(*)
            from memory_edges
            where edge_type = 'contradicts'
            """
        ).fetchone()[0]
        entity_count = self.connection.execute("select count(*) from entities").fetchone()[0]
        entity_links = self.connection.execute("select count(*) from memory_entities").fetchone()[0]
        task_views = [
            self._decorate_task_for_execution(task)
            for task in self._active_task_memories()
            if str(task.metadata.get("status", "open")) in {"blocked", "in_progress", "open"}
        ]
        latest_evaluation = self.latest_evaluation_run()
        latest_patch_run = self.latest_patch_run()
        return {
            "db_path": str(self.db_path),
            "events": int(events_count),
            "active_memories": int(memories_count),
            "by_kind": {row["kind"]: int(row["count"]) for row in by_kind_rows},
            "by_layer": {row["layer"]: int(row["count"]) for row in by_layer_rows},
            "contradiction_edges": int(contradiction_edges),
            "entities": int(entity_count),
            "memory_entity_links": int(entity_links),
            "ready_tasks": sum(1 for task in task_views if bool(task.metadata.get("ready_now"))),
            "overdue_tasks": sum(1 for task in task_views if bool(task.metadata.get("overdue"))),
            "maintenance": self.maintenance_status(),
            "semantic_reranker": self.reranker.status(),
            "latest_evaluation": latest_evaluation,
            "latest_patch_run": latest_patch_run,
        }

    def _current_event_id(self) -> int:
        value = self.connection.execute("select coalesce(max(id), 0) from events").fetchone()[0]
        return int(value)

    def _count_active_memories(self, layer: str) -> int:
        value = self.connection.execute(
            """
            select count(*)
            from memories
            where archived_at is null and layer = ?
            """,
            (layer,),
        ).fetchone()[0]
        return int(value)

    def _get_maintenance_state(self, task_name: str) -> dict[str, Any]:
        row = self.connection.execute(
            """
            select task_name, last_run_at, last_event_id, details_json
            from maintenance_state
            where task_name = ?
            """,
            (task_name,),
        ).fetchone()
        if row is None:
            return {
                "task_name": task_name,
                "last_run_at": None,
                "last_event_id": 0,
                "details": {},
            }
        return {
            "task_name": row["task_name"],
            "last_run_at": row["last_run_at"],
            "last_event_id": int(row["last_event_id"]),
            "details": json.loads(row["details_json"]),
        }

    def _mark_maintenance_run(self, task_name: str, details: dict[str, Any] | None = None) -> None:
        self.connection.execute(
            """
            insert into maintenance_state(task_name, last_run_at, last_event_id, details_json)
            values (?, ?, ?, ?)
            on conflict(task_name) do update set
                last_run_at = excluded.last_run_at,
                last_event_id = excluded.last_event_id,
                details_json = excluded.details_json
            """,
            (
                task_name,
                utc_now_iso(),
                self._current_event_id(),
                json.dumps(details or {}, sort_keys=True),
            ),
        )
        self.connection.commit()

    def _count_updated_memories_since(self, layer: str, task_name: str) -> int:
        state = self._get_maintenance_state(task_name)
        if state["last_run_at"] is None:
            value = self.connection.execute(
                """
                select count(*)
                from memories
                where archived_at is null and layer = ?
                """,
                (layer,),
            ).fetchone()[0]
            return int(value)
        value = self.connection.execute(
            """
            select count(*)
            from memories
            where archived_at is null and layer = ? and updated_at > ?
            """,
            (layer, state["last_run_at"]),
        ).fetchone()[0]
        return int(value)

    def _find_active_task(self, title: str, area: str) -> MemoryRecord | None:
        rows = [task for task in self._active_task_memories() if task.subject == area]
        normalized_title = self._normalize_text(title)
        for task in rows:
            task_title = str(task.metadata.get("title", ""))
            if str(task.metadata.get("status", "open")) == "done":
                continue
            if self._normalize_text(task_title) == normalized_title:
                return task
        return None

    def _find_active_task_any_area(self, title: str) -> MemoryRecord | None:
        normalized_title = self._normalize_text(title)
        for task in self._active_task_memories():
            task_title = str(task.metadata.get("title", ""))
            if self._normalize_text(task_title) == normalized_title:
                return task
        return None

    def _active_task_memories(self) -> list[MemoryRecord]:
        rows = self.connection.execute(
            """
            select *
            from memories
            where archived_at is null and kind = 'task' and layer = 'atomic'
            order by updated_at desc, id desc
            """
        ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def _reviewable_task_views(self) -> list[MemoryRecord]:
        return [
            self._decorate_task_for_execution(task)
            for task in self._active_task_memories()
            if str(task.metadata.get("status", "open")) in {"blocked", "in_progress", "open"}
        ]

    def _service_sync_task_title(self, action_item: dict[str, Any]) -> str:
        label = str(
            action_item.get("label") or action_item.get("action") or "Service action"
        ).strip()
        return f"Cockpit setup: {label}"

    def _service_sync_task_details(self, action_item: dict[str, Any]) -> str:
        description = str(action_item.get("description") or "").strip()
        confirmation = str(action_item.get("confirmation_message") or "").strip()
        success = str(action_item.get("success_message") or "").strip()
        parts = [part for part in [description, confirmation, success] if part]
        if not parts:
            action_name = str(action_item.get("action") or "service action").strip()
            return f"Sync and complete the recommended cockpit service action '{action_name}'."
        return " ".join(parts)

    def _service_sync_importance(self, action_name: str) -> float:
        normalized = str(action_name or "").strip()
        if normalized == "install_remote_service":
            return 0.94
        if normalized == "install_local_service":
            return 0.92
        if normalized == "restart_remote_service":
            return 0.9
        if normalized == "restart_local_service":
            return 0.86
        if normalized == "install_desktop_launcher":
            return 0.82
        return 0.84

    def _count_reviewable_tasks(self) -> int:
        return len(self._reviewable_task_views())

    def _count_updated_tasks_since(self, task_name: str) -> int:
        state = self._get_maintenance_state(task_name)
        tasks = self._reviewable_task_views()
        if state["last_run_at"] is None:
            return len(tasks)
        return sum(1 for task in tasks if task.updated_at > str(state["last_run_at"]))

    def _normalize_task_titles(self, titles: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for title in titles:
            clean_title = title.strip().rstrip(".")
            key = self._normalize_text(clean_title)
            if not key or key in seen:
                continue
            seen.add(key)
            normalized.append(clean_title)
        return normalized

    def _task_status_priority(self, status: str) -> int:
        return {
            "in_progress": 3,
            "open": 2,
            "blocked": 1,
            "done": 0,
        }.get(status, 0)

    def _task_due_priority(self, task: MemoryRecord) -> float:
        due_date = str(task.metadata.get("due_date") or "").strip()
        if not due_date:
            return 0.0
        due = self._parse_temporal_value(due_date)
        if due is None:
            return 0.05
        now = datetime.now(timezone.utc)
        days_until_due = (due - now).total_seconds() / 86400.0
        if days_until_due <= 0:
            return 1.0
        return max(0.1, 1.0 / (1.0 + days_until_due))

    def _is_task_overdue(self, task: MemoryRecord) -> bool:
        due_date = str(task.metadata.get("due_date") or "").strip()
        if not due_date:
            return False
        due = self._parse_temporal_value(due_date)
        if due is None:
            return False
        return due <= datetime.now(timezone.utc)

    def _is_task_snoozed(self, task: MemoryRecord) -> bool:
        snoozed_until = str(task.metadata.get("snoozed_until") or "").strip()
        if not snoozed_until:
            return False
        until = self._parse_temporal_value(snoozed_until)
        if until is None:
            return False
        return until > datetime.now(timezone.utc)

    def _task_title_is_done(self, title: str) -> bool:
        task = self._find_active_task_any_area(title)
        if task is None:
            return False
        return str(task.metadata.get("status", "open")) == "done"

    def _task_unresolved_dependencies(self, task: MemoryRecord) -> list[str]:
        return [
            title
            for title in self._normalize_task_titles(list(task.metadata.get("depends_on", [])))
            if not self._task_title_is_done(title)
        ]

    def _task_active_blockers(self, task: MemoryRecord) -> list[str]:
        return [
            title
            for title in self._normalize_task_titles(list(task.metadata.get("blocked_by", [])))
            if not self._task_title_is_done(title)
        ]

    def _decorate_task_for_execution(self, task: MemoryRecord) -> MemoryRecord:
        unresolved_dependencies = self._task_unresolved_dependencies(task)
        active_blockers = self._task_active_blockers(task)
        status = str(task.metadata.get("status", "open"))
        snoozed_now = self._is_task_snoozed(task)
        blocked_now = status == "blocked" or bool(unresolved_dependencies) or bool(active_blockers)
        ready_now = status in {"open", "in_progress"} and not blocked_now and not snoozed_now
        overdue = self._is_task_overdue(task) and not snoozed_now
        escalation_level = self._task_nudge_count(task)
        execution_bucket = self._task_execution_bucket(
            ready_now=ready_now,
            blocked_now=blocked_now,
            overdue=overdue,
            snoozed_now=snoozed_now,
        )
        metadata = {
            **task.metadata,
            "ready_now": ready_now,
            "blocked_now": blocked_now,
            "overdue": overdue,
            "snoozed_now": snoozed_now,
            "unresolved_dependencies": unresolved_dependencies,
            "active_blockers": active_blockers,
            "escalation_level": escalation_level,
            "execution_bucket": execution_bucket,
        }
        return MemoryRecord(
            id=task.id,
            kind=task.kind,
            subject=task.subject,
            content=task.content,
            tags=list(task.tags),
            importance=task.importance,
            confidence=task.confidence,
            metadata=metadata,
            layer=task.layer,
            created_at=task.created_at,
            updated_at=task.updated_at,
            last_accessed_at=task.last_accessed_at,
            access_count=task.access_count,
            archived_at=task.archived_at,
            source_event_id=task.source_event_id,
        )

    def _task_execution_bucket(
        self,
        *,
        ready_now: bool,
        blocked_now: bool,
        overdue: bool,
        snoozed_now: bool,
    ) -> int:
        if snoozed_now:
            return 0
        if ready_now and overdue:
            return 4
        if ready_now:
            return 3
        if overdue:
            return 2
        if blocked_now:
            return 1
        return 0

    def _task_queue_sort_key(self, task: MemoryRecord) -> tuple[int, int, float, str, int]:
        return (
            int(task.metadata.get("execution_bucket", 0)),
            self._task_status_priority(str(task.metadata.get("status", "open"))),
            self._task_due_priority(task),
            task.updated_at,
            task.id,
        )

    def _task_nudge_count(self, task: MemoryRecord) -> int:
        task_entity = self._primary_task_entity(task.id)
        task_entity_name = task_entity.canonical_name if task_entity is not None else ""
        title = str(task.metadata.get("title") or "").strip()
        cycle_key = str(task.metadata.get("cycle_key") or "")
        rows = self.connection.execute(
            """
            select metadata_json
            from memories
            where archived_at is null and kind = 'nudge' and layer = 'atomic'
            """
        ).fetchall()
        count = 0
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            if self._nudge_matches_task(
                metadata,
                task_entity_name=task_entity_name,
                title=title,
                cycle_key=cycle_key,
            ):
                count += 1
        return count

    def _task_age_days(self, task: MemoryRecord) -> float:
        updated = datetime.fromisoformat(task.updated_at)
        now = datetime.now(timezone.utc)
        return max((now - updated).total_seconds() / 86400.0, 0.0)

    def _task_review_candidates(self) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for task in self._reviewable_task_views():
            candidate = self._task_review_candidate(task)
            if candidate is None:
                continue
            if not self._task_nudge_due(
                task=task,
                nudge_type=str(candidate["nudge_type"]),
                cooldown_hours=TASK_NUDGE_COOLDOWN_HOURS,
                any_type=str(candidate["nudge_type"]) == "escalation",
            ):
                continue
            candidates.append(candidate)
        return candidates

    def _count_pending_task_nudges(self) -> int:
        return len(self._task_review_candidates())

    def _task_review_candidate(self, task: MemoryRecord) -> dict[str, Any] | None:
        title = str(task.metadata.get("title") or task.content)
        status = str(task.metadata.get("status", "open"))
        due_date = str(task.metadata.get("due_date") or "").strip()
        overdue = bool(task.metadata.get("overdue"))
        ready_now = bool(task.metadata.get("ready_now"))
        blocked_now = bool(task.metadata.get("blocked_now"))
        snoozed_now = bool(task.metadata.get("snoozed_now"))
        age_days = self._task_age_days(task)
        unresolved_dependencies = [
            str(item)
            for item in task.metadata.get("unresolved_dependencies", [])
            if str(item).strip()
        ]
        active_blockers = [
            str(item)
            for item in task.metadata.get("active_blockers", [])
            if str(item).strip()
        ]
        task_entity = self._primary_task_entity(task.id)
        task_entity_name = task_entity.canonical_name if task_entity is not None else ""
        prior_nudges = self._task_nudge_count(task)

        if snoozed_now:
            return None

        if prior_nudges >= TASK_ESCALATION_NUDGE_COUNT and (overdue or blocked_now):
            escalation_reason = "overdue" if overdue else "blocked"
            blocker_summary = ", ".join((active_blockers or unresolved_dependencies)[:2])
            content = f"Escalation: Task '{title}' is still {escalation_reason} after earlier nudges."
            if blocker_summary and blocked_now:
                content = content[:-1] + f" Current blocker: {blocker_summary}."
            return {
                "task": task,
                "nudge_type": "escalation",
                "content": content,
                "importance": 0.95,
                "confidence": 0.92,
                "priority": 5,
                "metadata": {
                    "nudge_type": "escalation",
                    "task_title": title,
                    "task_entity_name": task_entity_name,
                    "task_status": status,
                    "due_date": due_date,
                    "age_days": round(age_days, 2),
                    "prior_nudges": prior_nudges,
                    "blocked_by": active_blockers or unresolved_dependencies,
                },
            }

        if overdue and blocked_now:
            blocker_summary = ", ".join((active_blockers or unresolved_dependencies)[:2])
            content = (
                f"Nudge: Task '{title}' is overdue and still blocked"
                + (f" by {blocker_summary}." if blocker_summary else ".")
            )
            return {
                "task": task,
                "nudge_type": "overdue_blocked",
                "content": content,
                "importance": 0.91,
                "confidence": 0.9,
                "priority": 4,
                "metadata": {
                    "nudge_type": "overdue_blocked",
                    "task_title": title,
                    "task_entity_name": task_entity_name,
                    "task_status": status,
                    "due_date": due_date,
                    "age_days": round(age_days, 2),
                    "prior_nudges": prior_nudges,
                    "blocked_by": active_blockers or unresolved_dependencies,
                },
            }

        if overdue and ready_now:
            content = f"Nudge: Task '{title}' is overdue and ready to work on now."
            return {
                "task": task,
                "nudge_type": "overdue_ready",
                "content": content,
                "importance": 0.9,
                "confidence": 0.9,
                "priority": 3,
                "metadata": {
                    "nudge_type": "overdue_ready",
                    "task_title": title,
                    "task_entity_name": task_entity_name,
                    "task_status": status,
                    "due_date": due_date,
                    "age_days": round(age_days, 2),
                    "prior_nudges": prior_nudges,
                },
            }

        if blocked_now and age_days >= TASK_BLOCKED_STALE_DAYS:
            blocker_summary = ", ".join((active_blockers or unresolved_dependencies)[:2])
            content = (
                f"Nudge: Task '{title}' has been blocked for {max(int(age_days), 1)} days"
                + (f" by {blocker_summary}." if blocker_summary else ".")
            )
            return {
                "task": task,
                "nudge_type": "stale_blocked",
                "content": content,
                "importance": 0.82,
                "confidence": 0.86,
                "priority": 2,
                "metadata": {
                    "nudge_type": "stale_blocked",
                    "task_title": title,
                    "task_entity_name": task_entity_name,
                    "task_status": status,
                    "due_date": due_date,
                    "age_days": round(age_days, 2),
                    "prior_nudges": prior_nudges,
                    "blocked_by": active_blockers or unresolved_dependencies,
                },
            }

        if ready_now and age_days >= TASK_STALE_DAYS:
            content = (
                f"Nudge: Task '{title}' has been untouched for {max(int(age_days), 1)} days"
                " and is ready to work on now."
            )
            return {
                "task": task,
                "nudge_type": "stale_ready",
                "content": content,
                "importance": 0.78,
                "confidence": 0.84,
                "priority": 1,
                "metadata": {
                    "nudge_type": "stale_ready",
                    "task_title": title,
                    "task_entity_name": task_entity_name,
                    "task_status": status,
                    "due_date": due_date,
                    "age_days": round(age_days, 2),
                    "prior_nudges": prior_nudges,
                },
            }

        return None

    def _task_nudge_due(
        self,
        *,
        task: MemoryRecord,
        nudge_type: str,
        cooldown_hours: float,
        any_type: bool = False,
    ) -> bool:
        latest = (
            self._latest_task_nudge_any(task)
            if any_type
            else self._latest_task_nudge(task, nudge_type)
        )
        if latest is None:
            return True
        updated = datetime.fromisoformat(latest.updated_at)
        now = datetime.now(timezone.utc)
        return (now - updated).total_seconds() >= cooldown_hours * 3600.0

    def _latest_task_nudge(self, task: MemoryRecord, nudge_type: str) -> MemoryRecord | None:
        task_entity = self._primary_task_entity(task.id)
        task_entity_name = task_entity.canonical_name if task_entity is not None else ""
        title = str(task.metadata.get("title") or "").strip()
        cycle_key = str(task.metadata.get("cycle_key") or "")
        rows = self.connection.execute(
            """
            select *
            from memories
            where archived_at is null and kind = 'nudge' and layer = 'atomic'
            order by updated_at desc, id desc
            """
        ).fetchall()
        for row in rows:
            memory = self._row_to_memory(row)
            metadata = memory.metadata
            if str(metadata.get("nudge_type", "")) != nudge_type:
                continue
            if self._nudge_matches_task(
                metadata,
                task_entity_name=task_entity_name,
                title=title,
                cycle_key=cycle_key,
            ):
                return memory
        return None

    def _latest_task_nudge_any(self, task: MemoryRecord) -> MemoryRecord | None:
        task_entity = self._primary_task_entity(task.id)
        task_entity_name = task_entity.canonical_name if task_entity is not None else ""
        title = str(task.metadata.get("title") or "").strip()
        cycle_key = str(task.metadata.get("cycle_key") or "")
        rows = self.connection.execute(
            """
            select *
            from memories
            where archived_at is null and kind = 'nudge' and layer = 'atomic'
            order by updated_at desc, id desc
            """
        ).fetchall()
        for row in rows:
            memory = self._row_to_memory(row)
            if self._nudge_matches_task(
                memory.metadata,
                task_entity_name=task_entity_name,
                title=title,
                cycle_key=cycle_key,
            ):
                return memory
        return None

    def _nudge_matches_task(
        self,
        metadata: dict[str, Any],
        *,
        task_entity_name: str,
        title: str,
        cycle_key: str,
    ) -> bool:
        if cycle_key and str(metadata.get("task_cycle_key", "")) == cycle_key:
            return True
        if task_entity_name and str(metadata.get("task_entity_name", "")) == task_entity_name:
            return True
        return str(metadata.get("task_title", "")).strip() == title

    def _primary_task_entity(self, memory_id: int) -> EntityRecord | None:
        for link in self.get_memory_entities(memory_id):
            if link.entity.entity_type == "task":
                return link.entity
        return None

    def _build_memory_bundles(
        self,
        memories: list[SearchResult],
        *,
        evidence_limit: int = 3,
        event_limit: int = 2,
    ) -> list[MemoryBundle]:
        bundles: list[MemoryBundle] = []
        for item in memories:
            evidence = self._get_supporting_memories(item.memory, limit=evidence_limit)
            contradictions = self._get_contradicting_memories(item.memory, limit=evidence_limit)
            events = self._get_supporting_events(item.memory, limit=event_limit)
            bundles.append(
                MemoryBundle(
                    anchor=item,
                    evidence=evidence,
                    contradictions=contradictions,
                    supporting_events=events,
                )
            )
        return bundles

    def _get_supporting_memories(self, memory: MemoryRecord, *, limit: int) -> list[MemoryRecord]:
        source_ids = [
            source.source_id
            for source in self.get_memory_sources(memory.id)
            if source.source_type == "memory"
        ]
        if memory.layer == "atomic":
            incoming = [
                edge.from_memory_id
                for edge in self.get_memory_edges(memory.id, direction="incoming")
                if edge.edge_type in {"compacts", "abstracts"}
            ]
            source_ids = incoming + source_ids
        unique_ids = list(dict.fromkeys(source_ids))
        evidence: list[MemoryRecord] = []
        for source_id in unique_ids:
            candidate = self.get_memory(source_id)
            if candidate.archived_at is None:
                evidence.append(candidate)
            if len(evidence) >= limit:
                break
        return evidence

    def _get_contradicting_memories(self, memory: MemoryRecord, *, limit: int) -> list[MemoryRecord]:
        contradiction_ids = [
            edge.to_memory_id
            for edge in self.get_memory_edges(memory.id, direction="outgoing")
            if edge.edge_type == "contradicts"
        ]
        contradictions: list[MemoryRecord] = []
        for memory_id in contradiction_ids:
            candidate = self.get_memory(memory_id)
            if candidate.archived_at is None:
                contradictions.append(candidate)
            if len(contradictions) >= limit:
                break
        return contradictions

    def _get_supporting_events(self, memory: MemoryRecord, *, limit: int) -> list[Event]:
        event_ids = [
            source.source_id
            for source in self.get_memory_sources(memory.id)
            if source.source_type == "event"
        ]
        if not event_ids and memory.layer in {"reflection", "profile"}:
            for evidence in self._get_supporting_memories(memory, limit=limit * 2):
                if evidence.source_event_id is not None:
                    event_ids.append(int(evidence.source_event_id))
        unique_ids = list(dict.fromkeys(event_ids))
        return self._get_events_by_ids(unique_ids[:limit])

    def _get_events_by_ids(self, event_ids: list[int]) -> list[Event]:
        events: list[Event] = []
        for event_id in event_ids:
            row = self.connection.execute(
                "select * from events where id = ?",
                (event_id,),
            ).fetchone()
            if row is not None:
                events.append(self._row_to_event(row))
        return events

    def _layer_filter_clause(
        self,
        alias: str,
        layers: tuple[str, ...] | None,
    ) -> tuple[str, list[str]]:
        if not layers:
            return "", []
        placeholders = ", ".join("?" for _ in layers)
        return f" and {alias}.layer in ({placeholders})", list(layers)

    def _touch_memories(self, memory_ids: list[int]) -> None:
        now = utc_now_iso()
        for memory_id in memory_ids:
            self.connection.execute(
                """
                update memories
                set access_count = access_count + 1,
                    last_accessed_at = ?
                where id = ?
                """,
                (now, memory_id),
            )
        self.connection.commit()

    def _refresh_memory_entities(self, memory: MemoryRecord) -> None:
        self.connection.execute("delete from memory_entities where memory_id = ?", (memory.id,))
        drafts = self.entity_resolver.resolve_memory(
            kind=memory.kind,
            subject=memory.subject,
            content=memory.content,
            tags=memory.tags,
            metadata=memory.metadata,
        )
        for draft in drafts:
            entity_id = self._upsert_entity(
                canonical_name=draft.canonical_name,
                display_name=draft.display_name,
                entity_type=draft.entity_type,
                aliases=draft.aliases,
                metadata=draft.metadata,
            )
            self.connection.execute(
                """
                insert into memory_entities(
                    memory_id,
                    entity_id,
                    confidence,
                    evidence_text,
                    created_at
                )
                values (?, ?, ?, ?, ?)
                on conflict(memory_id, entity_id) do update set
                    confidence = excluded.confidence,
                    evidence_text = excluded.evidence_text
                """,
                (
                    memory.id,
                    entity_id,
                    draft.confidence,
                    draft.evidence_text,
                    utc_now_iso(),
                ),
            )

    def _sync_task_entity_edges(
        self,
        task: MemoryRecord,
        *,
        depends_on: list[str],
        blocked_by: list[str],
    ) -> None:
        task_entity = self._primary_task_entity(task.id)
        if task_entity is None:
            return
        self.connection.execute(
            """
            delete from entity_edges
            where from_entity_id = ? and edge_type in ('depends_on', 'blocked_by')
            """,
            (task_entity.id,),
        )
        for title in depends_on:
            dependency_entity = self._ensure_task_entity(title)
            self._add_entity_edge(task_entity.id, dependency_entity.id, "depends_on")
        for title in blocked_by:
            blocker_entity = self._ensure_task_entity(title)
            self._add_entity_edge(task_entity.id, blocker_entity.id, "blocked_by")

    def _upsert_entity(
        self,
        *,
        canonical_name: str,
        display_name: str,
        entity_type: str,
        aliases: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        metadata = metadata or {}
        now = utc_now_iso()
        row = self.connection.execute(
            """
            select id, metadata_json
            from entities
            where canonical_name = ?
            """,
            (canonical_name,),
        ).fetchone()
        existing_metadata = json.loads(row["metadata_json"]) if row is not None else {}
        merged_metadata = {**existing_metadata, **metadata}
        self.connection.execute(
            """
            insert into entities(
                canonical_name,
                display_name,
                entity_type,
                metadata_json,
                created_at,
                updated_at
            )
            values (?, ?, ?, ?, ?, ?)
            on conflict(canonical_name) do update set
                display_name = excluded.display_name,
                entity_type = excluded.entity_type,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                canonical_name,
                display_name,
                entity_type,
                json.dumps(merged_metadata, sort_keys=True),
                now,
                now,
            ),
        )
        entity_id = int(
            self.connection.execute(
                "select id from entities where canonical_name = ?",
                (canonical_name,),
            ).fetchone()[0]
        )
        for alias in aliases:
            normalized_alias = self._normalize_alias_text(alias)
            if not normalized_alias:
                continue
            self.connection.execute(
                """
                insert or ignore into entity_aliases(
                    entity_id,
                    alias,
                    normalized_alias,
                    created_at
                )
                values (?, ?, ?, ?)
                """,
                (entity_id, alias.strip(), normalized_alias, now),
            )
        return entity_id

    def _ensure_task_entity(self, title: str) -> EntityRecord:
        clean_title = title.strip().rstrip(".")
        canonical_name = f"task:{self._normalize_alias_text(clean_title).replace(' ', '_')}"
        entity_id = self._upsert_entity(
            canonical_name=canonical_name,
            display_name=clean_title,
            entity_type="task",
            aliases=[clean_title, canonical_name.replace("task:", "").replace("_", " ")],
            metadata={},
        )
        return self.get_entity(entity_id)

    def _add_memory_source(
        self,
        *,
        memory_id: int,
        source_type: str,
        source_id: int,
        relation_type: str,
    ) -> None:
        self.connection.execute(
            """
            insert or ignore into memory_sources(
                memory_id,
                source_type,
                source_id,
                relation_type,
                created_at
            )
            values (?, ?, ?, ?, ?)
            """,
            (memory_id, source_type, source_id, relation_type, utc_now_iso()),
        )

    def _add_memory_edge(self, from_memory_id: int, to_memory_id: int, edge_type: str) -> None:
        self.connection.execute(
            """
            insert or ignore into memory_edges(
                from_memory_id,
                to_memory_id,
                edge_type,
                created_at
            )
            values (?, ?, ?, ?)
            """,
            (from_memory_id, to_memory_id, edge_type, utc_now_iso()),
        )

    def _add_entity_edge(self, from_entity_id: int, to_entity_id: int, edge_type: str) -> None:
        self.connection.execute(
            """
            insert or ignore into entity_edges(
                from_entity_id,
                to_entity_id,
                edge_type,
                created_at
            )
            values (?, ?, ?, ?)
            """,
            (from_entity_id, to_entity_id, edge_type, utc_now_iso()),
        )

    def _remove_memory_edge(self, from_memory_id: int, to_memory_id: int, edge_type: str) -> None:
        self.connection.execute(
            """
            delete from memory_edges
            where from_memory_id = ? and to_memory_id = ? and edge_type = ?
            """,
            (from_memory_id, to_memory_id, edge_type),
        )

    def _archive_memory(self, memory_id: int) -> None:
        self.connection.execute(
            """
            update memories
            set archived_at = coalesce(archived_at, ?)
            where id = ?
            """,
            (utc_now_iso(), memory_id),
        )

    def _get_latest_active_memory(
        self,
        *,
        kind: str,
        subject: str,
        layer: str,
    ) -> MemoryRecord | None:
        row = self.connection.execute(
            """
            select *
            from memories
            where kind = ? and subject = ? and layer = ? and archived_at is null
            order by updated_at desc, id desc
            limit 1
            """,
            (kind, subject, layer),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    def _link_contradictions_for_memory(self, memory: MemoryRecord) -> list[tuple[int, int]]:
        if memory.archived_at is not None or memory.layer != "atomic":
            return []
        rows = self.connection.execute(
            """
            select *
            from memories
            where archived_at is null
              and layer = 'atomic'
              and subject = ?
              and kind = ?
              and id != ?
            order by updated_at desc, id desc
            """,
            (memory.subject, memory.kind, memory.id),
        ).fetchall()
        contradictions: list[tuple[int, int]] = []
        for row in rows:
            other = self._row_to_memory(row)
            if self._memories_contradict(memory, other):
                self._add_memory_edge(memory.id, other.id, "contradicts")
                self._add_memory_edge(other.id, memory.id, "contradicts")
                contradictions.append((memory.id, other.id))
            else:
                self._remove_memory_edge(memory.id, other.id, "contradicts")
                self._remove_memory_edge(other.id, memory.id, "contradicts")
        return contradictions

    def _memories_contradict(self, left: MemoryRecord, right: MemoryRecord) -> bool:
        left_tokens = self._tokenize(left.content)
        right_tokens = self._tokenize(right.content)
        if not left_tokens or not right_tokens:
            return False
        if left_tokens == right_tokens:
            return False
        if self._has_negation(left.content) != self._has_negation(right.content):
            shared = (left_tokens & right_tokens) - self._stop_words()
            if shared:
                return True
        contradiction_pairs = [
            ("local", "cloud"),
            ("local", "remote"),
            ("locally", "cloud"),
            ("locally", "remote"),
            ("offline", "online"),
            ("enable", "disable"),
            ("enabled", "disabled"),
            ("allow", "block"),
            ("private", "public"),
            ("desktop", "mobile"),
            ("always", "never"),
            ("cheap", "expensive"),
        ]
        return any(
            (a in left_tokens and b in right_tokens) or (b in left_tokens and a in right_tokens)
            for a, b in contradiction_pairs
        )

    def _upsert_fts(self, memory_id: int) -> None:
        row = self.connection.execute(
            """
            select id, subject, content, tags_json
            from memories
            where id = ?
            """,
            (memory_id,),
        ).fetchone()
        if row is None:
            return
        self.connection.execute("delete from memories_fts where rowid = ?", (memory_id,))
        self.connection.execute(
            """
            insert into memories_fts(rowid, subject, content, tags)
            values (?, ?, ?, ?)
            """,
            (
                memory_id,
                row["subject"],
                row["content"],
                " ".join(json.loads(row["tags_json"])),
            ),
        )

    def _tokenize(self, value: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", self._normalize_text(value).replace("-", " ")))

    def _has_negation(self, value: str) -> bool:
        normalized = self._normalize_text(value)
        return any(
            token in normalized
            for token in (" not ", " never ", " no ", " without ", "cannot", "can't", "don't")
        ) or normalized.startswith("no ")

    def _stop_words(self) -> set[str]:
        return {
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "of",
            "for",
            "on",
            "in",
            "my",
            "your",
            "our",
            "is",
            "are",
            "should",
            "must",
            "agent",
        }

    def _score_memory(
        self,
        memory: MemoryRecord,
        *,
        raw_text_score: float,
        compacted_signal: bool,
        source_memory_count: int,
        contradiction_count: int,
        query_tokens: set[str],
        query_entity_names: set[str],
        memory_entity_names: set[str],
    ) -> tuple[float, list[str]]:
        text_signal = 1.0 / (1.0 + max(float(raw_text_score), 0.0))
        memory_tokens = self._tokenize(
            " ".join([memory.subject, memory.content, " ".join(memory.tags)])
        )
        query_overlap = (
            len(query_tokens & memory_tokens) / len(query_tokens) if query_tokens else 0.0
        )
        entity_overlap = (
            len(query_entity_names & memory_entity_names) / len(query_entity_names)
            if query_entity_names
            else 0.0
        )
        recency = self._recency_score(memory.updated_at)
        access_signal = min(memory.access_count / 10.0, 1.0)
        coverage_signal = min(source_memory_count / 5.0, 1.0)
        layer_signal = {
            "atomic": 0.55,
            "reflection": 0.72,
            "profile": 0.68,
        }.get(memory.layer, 0.5)
        compaction_penalty = 0.08 if compacted_signal and memory.layer == "atomic" else 0.0
        contradiction_penalty = min(contradiction_count * 0.07, 0.21)
        score = (
            0.32 * text_signal
            + 0.10 * query_overlap
            + 0.12 * entity_overlap
            + 0.24 * memory.importance
            + 0.14 * memory.confidence
            + 0.10 * recency
            + 0.05 * access_signal
            + 0.05 * layer_signal
            + 0.04 * coverage_signal
            - compaction_penalty
            - contradiction_penalty
        )
        reasons = [
            f"text={text_signal:.2f}",
            f"overlap={query_overlap:.2f}",
            f"entity_overlap={entity_overlap:.2f}",
            f"importance={memory.importance:.2f}",
            f"confidence={memory.confidence:.2f}",
            f"recency={recency:.2f}",
            f"layer={memory.layer}",
        ]
        if coverage_signal:
            reasons.append(f"sources={source_memory_count}")
        if compaction_penalty:
            reasons.append("compacted_source_penalty")
        if contradiction_count:
            reasons.append(f"contradictions={contradiction_count}")
        return score, reasons

    def _recency_score(self, updated_at: str) -> float:
        updated = datetime.fromisoformat(updated_at)
        now = datetime.now(timezone.utc)
        age_days = max((now - updated).total_seconds() / 86400.0, 0.0)
        return math.exp(-age_days / RECENCY_HALF_LIFE_DAYS)

    def _build_fts_query(self, query: str) -> str | None:
        terms = [term for term in self._normalize_text(query).split(" ") if len(term) > 1]
        if not terms:
            return None
        return " OR ".join(f'"{term}"' for term in terms[:8])

    def _parse_temporal_value(self, value: str) -> datetime | None:
        cleaned = value.strip()
        if not cleaned:
            return None
        try:
            parsed = datetime.fromisoformat(cleaned)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    def _shift_temporal_value(self, value: str, days: int) -> str:
        parsed = self._parse_temporal_value(value) or datetime.now(timezone.utc)
        shifted = parsed + timedelta(days=days)
        return shifted.date().isoformat() if "T" not in value else shifted.isoformat()

    def _optional_int(self, value: Any) -> int | None:
        if value in {None, ""}:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _contains_alias(self, text: str, alias: str) -> bool:
        normalized_text = self._normalize_alias_text(text)
        normalized_alias = self._normalize_alias_text(alias)
        if not normalized_text or not normalized_alias:
            return False
        return f" {normalized_alias} " in f" {normalized_text} "

    def _normalize_alias_text(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    def _normalize_text(self, value: str) -> str:
        return " ".join(value.lower().split())

    def _row_to_event(self, row: sqlite3.Row) -> Event:
        return Event(
            id=int(row["id"]),
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
            metadata=json.loads(row["metadata_json"]),
        )

    def _row_to_memory(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=int(row["id"]),
            kind=row["kind"],
            subject=row["subject"],
            content=row["content"],
            tags=json.loads(row["tags_json"]),
            importance=float(row["importance"]),
            confidence=float(row["confidence"]),
            layer=row["layer"],
            source_event_id=row["source_event_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_accessed_at=row["last_accessed_at"],
            access_count=int(row["access_count"]),
            archived_at=row["archived_at"],
            metadata=json.loads(row["metadata_json"]),
        )

    def _row_to_memory_source(self, row: sqlite3.Row) -> MemorySource:
        return MemorySource(
            memory_id=int(row["memory_id"]),
            source_type=row["source_type"],
            source_id=int(row["source_id"]),
            relation_type=row["relation_type"],
            created_at=row["created_at"],
        )

    def _row_to_memory_edge(self, row: sqlite3.Row) -> MemoryEdge:
        return MemoryEdge(
            from_memory_id=int(row["from_memory_id"]),
            to_memory_id=int(row["to_memory_id"]),
            edge_type=row["edge_type"],
            created_at=row["created_at"],
        )

    def _get_entity_aliases(self, entity_id: int) -> list[str]:
        rows = self.connection.execute(
            """
            select alias
            from entity_aliases
            where entity_id = ?
            order by alias asc
            """,
            (entity_id,),
        ).fetchall()
        return [row["alias"] for row in rows]

    def _row_to_entity(self, row: sqlite3.Row | dict[str, Any]) -> EntityRecord:
        entity_id = int(row["entity_id"] if "entity_id" in row.keys() else row["id"])
        return EntityRecord(
            id=entity_id,
            canonical_name=row["canonical_name"],
            display_name=row["display_name"],
            entity_type=row["entity_type"],
            aliases=self._get_entity_aliases(entity_id),
            metadata=json.loads(row["metadata_json"]),
            created_at=row["entity_created_at"] if "entity_created_at" in row.keys() else row["created_at"],
            updated_at=row["entity_updated_at"] if "entity_updated_at" in row.keys() else row["updated_at"],
        )

    def _row_to_memory_entity_link(self, row: sqlite3.Row) -> MemoryEntityLink:
        return MemoryEntityLink(
            memory_id=int(row["memory_id"]),
            entity=self._row_to_entity(row),
            confidence=float(row["confidence"]),
            evidence_text=row["evidence_text"],
            created_at=row["created_at"],
        )

    def _row_to_entity_edge(self, row: sqlite3.Row) -> EntityEdge:
        return EntityEdge(
            from_entity_id=int(row["from_entity_id"]),
            to_entity_id=int(row["to_entity_id"]),
            edge_type=row["edge_type"],
            created_at=row["created_at"],
        )

    def _row_to_evaluation_run(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = json.loads(row["summary_json"])
        return {
            "id": int(row["id"]),
            "suite_name": row["suite_name"],
            "score": float(row["score"]),
            "passed": bool(row["passed"]),
            "scenarios_passed": int(row["scenarios_passed"]),
            "scenarios_total": int(row["scenarios_total"]),
            "checks_passed": int(row["checks_passed"]),
            "checks_total": int(row["checks_total"]),
            "created_at": row["created_at"],
            "summary": payload,
        }

    def _row_to_patch_run(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = json.loads(row["summary_json"])
        return {
            "id": int(row["id"]),
            "run_name": row["run_name"],
            "suite_name": row["suite_name"],
            "task_title": row["task_title"],
            "status": row["status"],
            "baseline_score": (
                float(row["baseline_score"]) if row["baseline_score"] is not None else None
            ),
            "candidate_score": (
                float(row["candidate_score"]) if row["candidate_score"] is not None else None
            ),
            "apply_on_success": bool(row["apply_on_success"]),
            "applied": bool(row["applied"]),
            "workspace_path": row["workspace_path"],
            "changed_files": json.loads(row["changed_files_json"]),
            "operation_results": json.loads(row["operation_results_json"]),
            "validation_results": json.loads(row["validation_results_json"]),
            "created_at": row["created_at"],
            "summary": payload,
        }
