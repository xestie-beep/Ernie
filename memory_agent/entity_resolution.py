from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True, slots=True)
class EntitySpec:
    canonical_name: str
    display_name: str
    entity_type: str
    aliases: tuple[str, ...]


@dataclass(slots=True)
class EntityDraft:
    canonical_name: str
    display_name: str
    entity_type: str
    aliases: list[str] = field(default_factory=list)
    confidence: float = 0.72
    metadata: dict[str, str] = field(default_factory=dict)
    evidence_text: str = ""


class HeuristicEntityResolver:
    """Resolves recurring concepts into stable canonical entities."""

    _ENTITY_SPECS: tuple[EntitySpec, ...] = (
        EntitySpec(
            canonical_name="memory_system",
            display_name="Memory system",
            entity_type="capability",
            aliases=(
                "memory system",
                "memory stack",
                "memory layer",
                "memory architecture",
                "retrieval stack",
                "retrieval layer",
                "memory retrieval",
                "long term memory",
                "long-term memory",
            ),
        ),
        EntitySpec(
            canonical_name="sqlite",
            display_name="SQLite",
            entity_type="technology",
            aliases=(
                "sqlite",
                "sqlite3",
                "database choice",
                "storage backend",
                "storage engine",
                "local database",
            ),
        ),
        EntitySpec(
            canonical_name="fts5",
            display_name="FTS5",
            entity_type="technology",
            aliases=(
                "fts5",
                "full text search",
                "full-text search",
                "lexical retrieval",
                "lexical search",
            ),
        ),
        EntitySpec(
            canonical_name="semantic_reranking",
            display_name="Semantic reranking",
            entity_type="capability",
            aliases=(
                "semantic reranking",
                "semantic reranker",
                "embedding reranking",
                "embedding reranker",
                "semantic search",
                "local embeddings",
            ),
        ),
        EntitySpec(
            canonical_name="ollama",
            display_name="Ollama",
            entity_type="technology",
            aliases=(
                "ollama",
                "nomic-embed-text",
                "local embed model",
                "local embedding model",
            ),
        ),
        EntitySpec(
            canonical_name="local_runtime",
            display_name="Local runtime",
            entity_type="constraint",
            aliases=(
                "run locally",
                "local runtime",
                "desktop runtime",
                "main pc",
                "main computer",
                "on device",
                "on-device",
                "offline runtime",
            ),
        ),
        EntitySpec(
            canonical_name="cost_optimization",
            display_name="Cost optimization",
            entity_type="priority",
            aliases=(
                "low ongoing cost",
                "low cost",
                "cheap to run",
                "cost effective",
                "cost-effective",
                "cost efficient",
                "cost-efficient",
                "cost optimization",
            ),
        ),
        EntitySpec(
            canonical_name="contradiction_handling",
            display_name="Contradiction handling",
            entity_type="capability",
            aliases=(
                "contradiction handling",
                "contradiction detection",
                "contradiction scan",
                "conflicting memories",
                "memory contradictions",
            ),
        ),
        EntitySpec(
            canonical_name="task_management",
            display_name="Task management",
            entity_type="workflow",
            aliases=(
                "open loops",
                "task lifecycle",
                "task management",
                "task resurfacing",
                "execution tasks",
            ),
        ),
        EntitySpec(
            canonical_name="profile_synthesis",
            display_name="Profile synthesis",
            entity_type="capability",
            aliases=(
                "stable profiles",
                "profile synthesis",
                "long term profile",
                "long-term profile",
            ),
        ),
        EntitySpec(
            canonical_name="reflection_compaction",
            display_name="Reflection compaction",
            entity_type="capability",
            aliases=(
                "reflection",
                "memory compaction",
                "reflection layer",
                "source linked summary",
                "source-linked summary",
            ),
        ),
    )

    _SUBJECT_TYPES: dict[str, str] = {
        "architecture": "topic",
        "execution": "workflow",
        "implementation": "workflow",
        "optimization": "priority",
        "project": "topic",
        "runtime": "topic",
        "storage": "topic",
        "tooling": "topic",
        "verification": "topic",
    }

    def catalog(self) -> list[EntityDraft]:
        drafts: list[EntityDraft] = []
        for spec in self._ENTITY_SPECS:
            drafts.append(
                EntityDraft(
                    canonical_name=spec.canonical_name,
                    display_name=spec.display_name,
                    entity_type=spec.entity_type,
                    aliases=self._dedupe_aliases((spec.display_name, spec.canonical_name, *spec.aliases)),
                    confidence=0.9,
                    metadata={"seeded": "true"},
                )
            )
        return drafts

    def resolve_memory(
        self,
        *,
        kind: str,
        subject: str,
        content: str,
        tags: Iterable[str],
        metadata: dict[str, object] | None = None,
    ) -> list[EntityDraft]:
        metadata = metadata or {}
        combined_chunks = [subject, content, " ".join(tags)]
        for key in ("title", "decision", "tool_name", "outcome"):
            value = metadata.get(key)
            if isinstance(value, str):
                combined_chunks.append(value)
        normalized = self._normalize_text(" ".join(chunk for chunk in combined_chunks if chunk))
        drafts: dict[str, EntityDraft] = {}

        subject_draft = self._subject_entity(subject)
        if subject_draft is not None:
            drafts[subject_draft.canonical_name] = subject_draft

        for spec in self._ENTITY_SPECS:
            matches = [alias for alias in spec.aliases if self._contains_alias(normalized, alias)]
            if not matches:
                continue
            existing = drafts.get(spec.canonical_name)
            aliases = self._dedupe_aliases((spec.display_name, spec.canonical_name, *spec.aliases))
            evidence = max(matches, key=len)
            if existing is None:
                drafts[spec.canonical_name] = EntityDraft(
                    canonical_name=spec.canonical_name,
                    display_name=spec.display_name,
                    entity_type=spec.entity_type,
                    aliases=aliases,
                    confidence=0.82,
                    evidence_text=evidence,
                )
                continue
            existing.aliases = self._dedupe_aliases((*existing.aliases, *aliases))
            existing.confidence = max(existing.confidence, 0.82)
            if not existing.evidence_text:
                existing.evidence_text = evidence

        if kind == "task":
            task_entity = self._workflow_entity_from_title(metadata.get("title"))
            if task_entity is not None:
                drafts[task_entity.canonical_name] = task_entity

        return list(drafts.values())

    def _subject_entity(self, subject: str) -> EntityDraft | None:
        normalized_subject = self._slug(subject)
        if not normalized_subject:
            return None
        display_name = subject.replace("_", " ").strip().title()
        entity_type = self._SUBJECT_TYPES.get(normalized_subject, "topic")
        aliases = self._dedupe_aliases((display_name, subject, subject.replace("_", " ")))
        return EntityDraft(
            canonical_name=normalized_subject,
            display_name=display_name,
            entity_type=entity_type,
            aliases=aliases,
            confidence=0.68,
            evidence_text=subject.replace("_", " "),
        )

    def _workflow_entity_from_title(self, title: object) -> EntityDraft | None:
        if not isinstance(title, str):
            return None
        title = title.strip().rstrip(".")
        if not title:
            return None
        slug = self._slug(title)
        if not slug:
            return None
        return EntityDraft(
            canonical_name=f"task:{slug}",
            display_name=title,
            entity_type="task",
            aliases=self._dedupe_aliases((title, slug.replace("_", " "))),
            confidence=0.74,
            evidence_text=title,
        )

    def _contains_alias(self, normalized_text: str, alias: str) -> bool:
        normalized_alias = self._normalize_text(alias)
        if not normalized_alias:
            return False
        haystack = f" {normalized_text} "
        needle = f" {normalized_alias} "
        return needle in haystack

    def _dedupe_aliases(self, aliases: Iterable[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for alias in aliases:
            cleaned = alias.strip()
            normalized = self._normalize_text(cleaned)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(cleaned)
        return unique

    def _slug(self, value: str) -> str:
        return self._normalize_text(value).replace(" ", "_")

    def _normalize_text(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
