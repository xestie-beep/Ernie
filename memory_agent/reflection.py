from __future__ import annotations

from collections import defaultdict

from .models import MemoryDraft, MemoryRecord


class HeuristicReflector:
    """Builds topic reflections without losing links to source memories."""

    def build_reflection_groups(
        self,
        memories: list[MemoryRecord],
        max_group_size: int = 6,
    ) -> list[tuple[MemoryDraft, list[int]]]:
        by_subject: dict[str, list[MemoryRecord]] = defaultdict(list)
        for memory in memories:
            if memory.layer != "atomic" or memory.archived_at is not None or memory.kind == "nudge":
                continue
            by_subject[memory.subject].append(memory)

        groups: list[tuple[MemoryDraft, list[int]]] = []
        for subject, group in by_subject.items():
            if len(group) < 2:
                continue
            ordered = sorted(
                group,
                key=lambda item: (item.importance, item.updated_at, item.id),
                reverse=True,
            )[:max_group_size]
            groups.append((self._build_summary(subject, ordered), [item.id for item in ordered]))
        return groups

    def build_profile_groups(
        self,
        memories: list[MemoryRecord],
        max_group_size: int = 5,
    ) -> list[tuple[MemoryDraft, list[int]]]:
        by_subject: dict[str, list[MemoryRecord]] = defaultdict(list)
        for memory in memories:
            if memory.archived_at is not None:
                continue
            if memory.layer == "profile":
                continue
            if memory.kind == "nudge":
                continue
            if (
                memory.layer == "reflection"
                or memory.importance >= 0.8
                or memory.kind in {"task", "decision", "tool_outcome"}
            ):
                by_subject[memory.subject].append(memory)

        groups: list[tuple[MemoryDraft, list[int]]] = []
        for subject, group in by_subject.items():
            ordered = sorted(
                group,
                key=lambda item: (
                    self._layer_priority(item.layer),
                    item.importance,
                    item.confidence,
                    item.updated_at,
                    item.id,
                ),
                reverse=True,
            )[:max_group_size]
            reflection_count = sum(1 for item in ordered if item.layer == "reflection")
            if reflection_count == 0 and len(ordered) < 2:
                continue
            groups.append((self._build_profile(subject, ordered), [item.id for item in ordered]))
        return groups

    def _build_summary(self, subject: str, memories: list[MemoryRecord]) -> MemoryDraft:
        title = subject.replace("_", " ").strip() or "general"
        tags = sorted({tag for memory in memories for tag in memory.tags}.union({"reflection"}))
        kinds = sorted({memory.kind for memory in memories})
        statements = [self._memory_statement(memory) for memory in memories]
        content = f"{title.title()} summary: " + " ".join(statements)
        importance = min(max(memory.importance for memory in memories) + 0.03, 0.99)
        confidence = max(sum(memory.confidence for memory in memories) / len(memories) - 0.02, 0.6)
        return MemoryDraft(
            kind="reflection",
            subject=subject,
            content=content,
            tags=tags,
            importance=importance,
            confidence=confidence,
            metadata={
                "source_count": len(memories),
                "source_kinds": kinds,
            },
            layer="reflection",
        )

    def _build_profile(self, subject: str, memories: list[MemoryRecord]) -> MemoryDraft:
        title = subject.replace("_", " ").strip() or "general"
        tags = sorted(
            {tag for memory in memories for tag in memory.tags}.union({"profile", "stable"})
        )
        content = f"{title.title()} profile: " + self._build_profile_body(memories)
        importance = min(max(memory.importance for memory in memories) + 0.02, 0.99)
        confidence = min(
            max(sum(memory.confidence for memory in memories) / len(memories), 0.72),
            0.98,
        )
        return MemoryDraft(
            kind="profile",
            subject=subject,
            content=content,
            tags=tags,
            importance=importance,
            confidence=confidence,
            metadata={
                "source_count": len(memories),
                "source_layers": sorted({memory.layer for memory in memories}),
            },
            layer="profile",
        )

    def _build_profile_body(self, memories: list[MemoryRecord]) -> str:
        atomic = [memory for memory in memories if memory.layer == "atomic"]
        sections: list[str] = []

        for kind, label in (
            ("goal", "goals"),
            ("priority", "priorities"),
            ("constraint", "constraints"),
            ("preference", "preferences"),
            ("decision", "decisions"),
        ):
            kind_memories = [memory for memory in atomic if memory.kind == kind]
            if not kind_memories:
                continue
            statements = self._unique_statements(
                [self._short_statement(memory) for memory in kind_memories]
            )
            if statements:
                sections.append(f"{label}: {'; '.join(statements[:3])}.")

        task_memories = [memory for memory in atomic if memory.kind == "task"]
        if task_memories:
            status_sections: list[str] = []
            for status in ("open", "in_progress", "blocked", "done"):
                titles = self._unique_statements(
                    [
                        self._task_label(memory)
                        for memory in task_memories
                        if memory.metadata.get("status", "open") == status
                    ]
                )
                if titles:
                    status_sections.append(f"{status.replace('_', ' ')} - {', '.join(titles[:3])}")
            if status_sections:
                sections.append(f"tasks: {'; '.join(status_sections)}.")

        tool_memories = [memory for memory in atomic if memory.kind == "tool_outcome"]
        if tool_memories:
            tool_statements = self._unique_statements(
                [self._tool_statement(memory) for memory in tool_memories]
            )
            if tool_statements:
                sections.append(f"tool outcomes: {'; '.join(tool_statements[:3])}.")

        if sections:
            return " ".join(sections)

        summary_sources = [memory for memory in memories if memory.layer == "reflection"] or memories
        statements = [self._profile_statement(memory) for memory in summary_sources]
        return " ".join(statements)

    def _memory_statement(self, memory: MemoryRecord) -> str:
        content = memory.content.strip()
        if content.endswith("."):
            content = content[:-1]
        return f"{memory.kind}: {content}."

    def _profile_statement(self, memory: MemoryRecord) -> str:
        content = memory.content.strip()
        if memory.layer == "reflection":
            prefix = f"{memory.subject.replace('_', ' ').title()} summary:"
            if content.startswith(prefix):
                content = content[len(prefix) :].strip()
            if content.endswith("."):
                return content
            return f"{content}."
        return self._memory_statement(memory)

    def _short_statement(self, memory: MemoryRecord) -> str:
        content = memory.content.strip()
        for prefix in ("Decision:", "Task [open]:", "Task [in_progress]:", "Task [blocked]:", "Task [done]:"):
            if content.startswith(prefix):
                content = content[len(prefix) :].strip()
        if " Details:" in content:
            content = content.split(" Details:", 1)[0].strip()
        if content.endswith("."):
            content = content[:-1]
        return content

    def _task_label(self, memory: MemoryRecord) -> str:
        title = str(memory.metadata.get("title") or self._short_statement(memory))
        extras: list[str] = []
        due_date = str(memory.metadata.get("due_date") or "").strip()
        recurrence_days = memory.metadata.get("recurrence_days")
        snoozed_until = str(memory.metadata.get("snoozed_until") or "").strip()
        depends_on = [str(item) for item in memory.metadata.get("depends_on", []) if str(item).strip()]
        blocked_by = [str(item) for item in memory.metadata.get("blocked_by", []) if str(item).strip()]
        if due_date:
            extras.append(f"due {due_date}")
        if recurrence_days:
            extras.append(f"every {recurrence_days}d")
        if snoozed_until:
            extras.append(f"snoozed until {snoozed_until}")
        if depends_on:
            extras.append(f"depends on {', '.join(depends_on[:2])}")
        if blocked_by:
            extras.append(f"blocked by {', '.join(blocked_by[:2])}")
        if not extras:
            return title
        return f"{title} ({'; '.join(extras)})"

    def _tool_statement(self, memory: MemoryRecord) -> str:
        tool_name = str(memory.metadata.get("tool_name") or "tool")
        status = str(memory.metadata.get("status") or "unknown")
        outcome = str(memory.metadata.get("outcome") or self._short_statement(memory))
        return f"{tool_name} [{status}] {outcome}"

    def _unique_statements(self, statements: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for statement in statements:
            normalized = statement.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(statement.strip())
        return unique

    def _layer_priority(self, layer: str) -> int:
        return {
            "profile": 3,
            "reflection": 2,
            "atomic": 1,
        }.get(layer, 0)
