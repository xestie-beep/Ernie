from __future__ import annotations

import re

from .models import MemoryDraft


_SENTENCE_SPLIT_RE = re.compile(r"[.!?\n]+")
_LIST_SPLIT_RE = re.compile(r",| and |/|;")


class HeuristicMemoryExtractor:
    """Extracts durable memories without requiring a model call."""

    def extract(self, role: str, text: str) -> list[MemoryDraft]:
        if role == "assistant":
            return self._extract_assistant_memories(text)
        if role != "user":
            return []

        normalized_text = " ".join(text.split())
        lowered = normalized_text.lower()
        drafts: list[MemoryDraft] = []

        if "from scratch" in lowered:
            drafts.append(
                MemoryDraft(
                    kind="goal",
                    subject="project",
                    content="Build the agent from scratch.",
                    tags=["project", "scope"],
                    importance=0.9,
                    confidence=0.95,
                )
            )

        if (
            "run locally" in lowered
            or "local pc" in lowered
            or "main pc" in lowered
            or ("local" in lowered and "pc" in lowered)
        ):
            drafts.append(
                MemoryDraft(
                    kind="constraint",
                    subject="runtime",
                    content="The agent should run locally on the user's main PC.",
                    tags=["runtime", "local"],
                    importance=0.95,
                    confidence=0.95,
                )
            )

        if "memory system" in lowered:
            drafts.append(
                MemoryDraft(
                    kind="priority",
                    subject="architecture",
                    content="Implement the memory system first.",
                    tags=["memory", "architecture"],
                    importance=0.98,
                    confidence=0.95,
                )
            )

        if (
            "cost effective" in lowered
            or "cost-effective" in lowered
            or "cheap" in lowered
            or "low ongoing cost" in lowered
            or "keep costs low" in lowered
            or (
                "cost" in lowered
                and any(
                    marker in lowered
                    for marker in ("optimize", "priority", "priorities", "efficient", "low")
                )
            )
        ):
            drafts.append(
                MemoryDraft(
                    kind="preference",
                    subject="optimization",
                    content="Optimize the agent for low ongoing cost.",
                    tags=["cost", "optimization"],
                    importance=0.95,
                    confidence=0.9,
                )
            )

        if "efficient" in lowered or "fast" in lowered or "low latency" in lowered:
            drafts.append(
                MemoryDraft(
                    kind="preference",
                    subject="optimization",
                    content="Optimize the agent for efficient, low-latency operation.",
                    tags=["performance", "optimization"],
                    importance=0.92,
                    confidence=0.9,
                )
            )

        drafts.extend(self._extract_priority_list(normalized_text))
        drafts.extend(self._extract_sentence_level_memories(normalized_text))
        drafts.extend(self._extract_task_memories(normalized_text))
        drafts.extend(self._extract_decision_memories(normalized_text))

        unique: dict[tuple[str, str, str], MemoryDraft] = {}
        for draft in drafts:
            key = (draft.kind, draft.subject, draft.content.lower())
            if key not in unique:
                unique[key] = draft
        return list(unique.values())

    def _extract_priority_list(self, text: str) -> list[MemoryDraft]:
        match = re.search(r"priorit(?:y|ies)(?:.*?)(?:is|are|:)\s*(.+)", text, flags=re.IGNORECASE)
        if not match:
            return []
        value = match.group(1).strip().rstrip(".")
        items = [
            item.strip(" .")
            for item in _LIST_SPLIT_RE.split(value)
            if item.strip(" .")
        ]
        drafts: list[MemoryDraft] = []
        for item in items:
            drafts.append(
                MemoryDraft(
                    kind="priority",
                    subject="project",
                    content=f"Priority: {item}.",
                    tags=["priority"],
                    importance=0.88,
                    confidence=0.75,
                )
            )
        return drafts

    def _extract_sentence_level_memories(self, text: str) -> list[MemoryDraft]:
        drafts: list[MemoryDraft] = []
        for sentence in _SENTENCE_SPLIT_RE.split(text):
            cleaned = sentence.strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()

            if re.search(r"\b(we are building|we're building|let's make|build)\b", lowered):
                drafts.append(
                    MemoryDraft(
                        kind="goal",
                        subject="project",
                        content=cleaned[0].upper() + cleaned[1:],
                        tags=["project"],
                        importance=0.82,
                        confidence=0.7,
                    )
                )
                continue

            if re.search(r"\b(i want|we need|goal is|need to)\b", lowered):
                drafts.append(
                    MemoryDraft(
                        kind="goal",
                        subject="project",
                        content=cleaned[0].upper() + cleaned[1:],
                        tags=["goal"],
                        importance=0.78,
                        confidence=0.68,
                    )
                )
                continue

            if re.search(r"\b(i prefer|prefer|must|should|always|never|don't|do not)\b", lowered):
                drafts.append(
                    MemoryDraft(
                        kind="constraint",
                        subject="behavior",
                        content=cleaned[0].upper() + cleaned[1:],
                        tags=["constraint"],
                        importance=0.8,
                        confidence=0.68,
                    )
                )
        return drafts

    def _extract_task_memories(self, text: str) -> list[MemoryDraft]:
        drafts: list[MemoryDraft] = []
        for sentence in _SENTENCE_SPLIT_RE.split(text):
            cleaned = sentence.strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()

            explicit_match = re.search(r"\b(?:next step|todo|task)\s*:\s*(.+)", cleaned, re.IGNORECASE)
            if explicit_match:
                drafts.append(self._task_draft(explicit_match.group(1).strip()))
                continue

            if re.search(
                r"\b(we need to|need to|please add|please implement|please wire|please build|next we should)\b",
                lowered,
            ):
                drafts.append(self._task_draft(cleaned))
        return drafts

    def _extract_decision_memories(self, text: str) -> list[MemoryDraft]:
        drafts: list[MemoryDraft] = []
        for sentence in _SENTENCE_SPLIT_RE.split(text):
            cleaned = sentence.strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()

            explicit_match = re.search(
                r"\b(?:decision|we decided to|we're going with|we are going with|we will use|we'll use)\b[: ]*(.+)",
                cleaned,
                re.IGNORECASE,
            )
            if explicit_match:
                decision_text = explicit_match.group(1).strip()
                subject = self._infer_subject(decision_text)
                drafts.append(
                    MemoryDraft(
                        kind="decision",
                        subject=subject,
                        content=f"Decision: {decision_text.rstrip('.')}.",
                        tags=["decision", subject],
                        importance=0.9,
                        confidence=0.85,
                        metadata={"decision": decision_text.rstrip(".")},
                    )
                )
        return drafts

    def _extract_assistant_memories(self, text: str) -> list[MemoryDraft]:
        drafts: list[MemoryDraft] = []
        normalized_text = " ".join(text.split())
        lowered = normalized_text.lower()

        if "test" in lowered and any(marker in lowered for marker in ("passed", "ok", "green")):
            drafts.append(
                MemoryDraft(
                    kind="tool_outcome",
                    subject="verification",
                    content="Tool outcome [success] tests: verification passed.",
                    tags=["tooling", "tests", "success"],
                    importance=0.82,
                    confidence=0.9,
                    metadata={
                        "tool_name": "tests",
                        "status": "success",
                        "outcome": "verification passed",
                    },
                )
            )

        if any(marker in lowered for marker in ("implemented", "added", "wired", "integrated")):
            drafts.append(
                MemoryDraft(
                    kind="tool_outcome",
                    subject="implementation",
                    content=f"Tool outcome [success] implementation: {normalized_text.rstrip('.')}.",
                    tags=["tooling", "implementation", "success"],
                    importance=0.76,
                    confidence=0.78,
                    metadata={
                        "tool_name": "implementation",
                        "status": "success",
                        "outcome": normalized_text.rstrip("."),
                    },
                )
            )
        return drafts

    def _task_draft(self, text: str) -> MemoryDraft:
        title = text.strip().rstrip(".")
        return MemoryDraft(
            kind="task",
            subject="execution",
            content=f"Task [open]: {title}.",
            tags=["task", "execution", "open"],
            importance=0.84,
            confidence=0.8,
            metadata={"title": title, "status": "open"},
        )

    def _infer_subject(self, text: str) -> str:
        lowered = text.lower()
        if any(token in lowered for token in ("sqlite", "database", "storage")):
            return "storage"
        if any(token in lowered for token in ("runtime", "local", "cloud", "desktop")):
            return "runtime"
        if any(token in lowered for token in ("memory", "architecture", "retrieval")):
            return "architecture"
        if any(token in lowered for token in ("cost", "latency", "performance", "efficient")):
            return "optimization"
        return "project"
