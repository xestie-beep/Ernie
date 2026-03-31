from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


@dataclass(slots=True)
class Event:
    id: int
    role: str
    content: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryDraft:
    kind: str
    subject: str
    content: str
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5
    confidence: float = 0.7
    metadata: dict[str, Any] = field(default_factory=dict)
    layer: str = "atomic"


@dataclass(slots=True)
class MemoryRecord(MemoryDraft):
    id: int = 0
    created_at: str = ""
    updated_at: str = ""
    last_accessed_at: str | None = None
    access_count: int = 0
    archived_at: str | None = None
    source_event_id: int | None = None


@dataclass(slots=True)
class SearchResult:
    memory: MemoryRecord
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MemoryBundle:
    anchor: SearchResult
    evidence: list[MemoryRecord] = field(default_factory=list)
    contradictions: list[MemoryRecord] = field(default_factory=list)
    supporting_events: list[Event] = field(default_factory=list)


@dataclass(slots=True)
class MemorySource:
    memory_id: int
    source_type: str
    source_id: int
    relation_type: str
    created_at: str


@dataclass(slots=True)
class MemoryEdge:
    from_memory_id: int
    to_memory_id: int
    edge_type: str
    created_at: str


@dataclass(slots=True)
class EntityRecord:
    id: int
    canonical_name: str
    display_name: str
    entity_type: str
    aliases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


@dataclass(slots=True)
class MemoryEntityLink:
    memory_id: int
    entity: EntityRecord
    confidence: float
    evidence_text: str
    created_at: str


@dataclass(slots=True)
class EntityEdge:
    from_entity_id: int
    to_entity_id: int
    edge_type: str
    created_at: str


@dataclass(slots=True)
class ContextWindow:
    query: str
    memories: list[SearchResult] = field(default_factory=list)
    profiles: list[SearchResult] = field(default_factory=list)
    bundles: list[MemoryBundle] = field(default_factory=list)
    ready_tasks: list[MemoryRecord] = field(default_factory=list)
    overdue_tasks: list[MemoryRecord] = field(default_factory=list)
    open_tasks: list[MemoryRecord] = field(default_factory=list)
    recent_events: list[Event] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"Query: {self.query}"]
        lines.extend(["", "Stable profiles:"])
        if not self.profiles:
            lines.append("- none")
        else:
            for item in self.profiles:
                lines.append(
                    f"- [{item.memory.layer}] {item.memory.content} "
                    f"(score={item.score:.3f})"
                )
        lines.extend(["", "Ready now:"])
        if not self.ready_tasks:
            lines.append("- none")
        else:
            for task in self.ready_tasks:
                lines.append(self._task_line(task))
        lines.extend(["", "Overdue:"])
        if not self.overdue_tasks:
            lines.append("- none")
        else:
            for task in self.overdue_tasks:
                lines.append(self._task_line(task))
        lines.extend(["", "Open loops:"])
        if not self.open_tasks:
            lines.append("- none")
        else:
            for task in self.open_tasks:
                lines.append(self._task_line(task))
        lines.extend(["", "Relevant memory:"])
        if self.bundles:
            for bundle in self.bundles:
                lines.append(
                    f"- [{bundle.anchor.memory.layer}/{bundle.anchor.memory.kind}] "
                    f"{bundle.anchor.memory.content} "
                    f"(score={bundle.anchor.score:.3f})"
                )
                for evidence in bundle.evidence:
                    lines.append(f"  evidence: [{evidence.kind}] {evidence.content}")
                for contradiction in bundle.contradictions:
                    lines.append(
                        f"  contradiction: [{contradiction.kind}] {contradiction.content}"
                    )
                for event in bundle.supporting_events:
                    lines.append(f"  event: {event.role}: {event.content}")
        elif not self.memories:
            lines.append("- none")
        else:
            for item in self.memories:
                lines.append(
                    f"- [{item.memory.kind}] {item.memory.content} "
                    f"(score={item.score:.3f})"
                )
        lines.extend(["", "Recent events:"])
        if not self.recent_events:
            lines.append("- none")
        else:
            for event in self.recent_events:
                lines.append(f"- {event.role}: {event.content}")
        return "\n".join(lines)

    def _task_line(self, task: MemoryRecord) -> str:
        status = str(task.metadata.get("status", "open"))
        title = str(task.metadata.get("title") or task.content)
        extras: list[str] = []
        due_date = str(task.metadata.get("due_date") or "").strip()
        recurrence_days = task.metadata.get("recurrence_days")
        snoozed_until = str(task.metadata.get("snoozed_until") or "").strip()
        command = str(task.metadata.get("command") or "").strip()
        cwd = str(task.metadata.get("cwd") or "").strip()
        service_action = str(task.metadata.get("service_action") or "").strip()
        service_inspection = str(task.metadata.get("service_inspection") or "").strip()
        service_label = str(task.metadata.get("service_label") or "").strip()
        file_operation = str(task.metadata.get("file_operation") or "").strip()
        file_path = str(task.metadata.get("file_path") or "").strip()
        symbol_name = str(task.metadata.get("symbol_name") or "").strip()
        depends_on = [
            str(item)
            for item in task.metadata.get("depends_on", [])
            if str(item).strip()
        ]
        blocked_by = [
            str(item)
            for item in task.metadata.get("blocked_by", [])
            if str(item).strip()
        ]
        if due_date:
            extras.append(f"due {due_date}")
        if bool(task.metadata.get("overdue")):
            extras.append("overdue")
        if bool(task.metadata.get("ready_now")):
            extras.append("ready")
        if bool(task.metadata.get("snoozed_now")):
            extras.append("snoozed")
        if recurrence_days:
            extras.append(f"every {recurrence_days}d")
        if snoozed_until:
            extras.append(f"snoozed until {snoozed_until}")
        if command:
            extras.append(f"cmd {command}")
        if cwd:
            extras.append(f"cwd {cwd}")
        if service_action:
            extras.append(f"service {service_label or service_action}")
        if service_inspection:
            extras.append(f"inspect {service_inspection}")
        if bool(task.metadata.get("service_requires_confirmation", False)):
            extras.append("confirm")
        if file_operation and file_path:
            extras.append(f"file {file_operation} {file_path}")
        if symbol_name:
            extras.append(f"symbol {symbol_name}")
        escalation_level = int(task.metadata.get("escalation_level", 0) or 0)
        if escalation_level:
            extras.append(f"escalation {escalation_level}")
        if depends_on:
            extras.append("depends on " + ", ".join(depends_on))
        if blocked_by:
            extras.append("blocked by " + ", ".join(blocked_by))
        suffix = f" ({'; '.join(extras)})" if extras else ""
        return f"- [{status}] {title}{suffix}"
