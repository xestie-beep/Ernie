from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


SOUL_FILENAME = "SOUL.md"
DEFAULT_SOUL_TEXT = """# Ernie

Ernie is a careful local operator for a memory-first agent.

## Core posture

- Be calm, direct, and practical.
- Prefer bounded progress over impressive but risky leaps.
- Explain why a step is being taken before pushing deeper.
- Treat operator trust as hard to earn and easy to lose.

## Operator stance

- Assume the operator may be inexperienced unless the context proves otherwise.
- Teach in short concrete steps.
- When uncertainty is real, say so plainly and narrow the next move.
- Prefer reversible or inspectable actions before high-impact actions.
"""


@dataclass(slots=True)
class SoulDocument:
    path: Path
    content: str
    exists: bool

    def render_system_prompt(self) -> str:
        return "Ernie SOUL:\n" + self.content.strip()

    def ui_summary(self) -> dict[str, object]:
        lines = [line.strip() for line in self.content.splitlines() if line.strip()]
        title = "Ernie"
        summary = ""
        principles: list[str] = []
        current_section = ""
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip() or title
                continue
            if line.startswith("## "):
                current_section = line[3:].strip().lower()
                continue
            if not summary and not line.startswith("-"):
                summary = line
                continue
            if line.startswith("-") and current_section in {
                "values",
                "communication style",
                "planning posture",
                "operator care",
            }:
                principles.append(line[1:].strip())
        return {
            "title": title,
            "summary": summary,
            "principles": principles[:4],
            "path": str(path := self.path),
            "exists": self.exists,
            "source": path.name,
        }


@dataclass(slots=True)
class SoulProposal:
    proposal_id: str
    section: str
    title: str
    rationale: str
    amendment: str
    evidence: list[str]
    evidence_signature: str
    explanation: str
    resurfaced_after_dismissal: bool


@dataclass(slots=True)
class SoulReview:
    summary: dict[str, object]
    proposals: list[SoulProposal]
    dismissed_evidence_signatures: list[str]


def load_soul_document(workspace_root: Path | None = None) -> SoulDocument:
    root = workspace_root or Path.cwd()
    path = root / SOUL_FILENAME
    if path.exists():
        content = path.read_text(encoding="utf-8").strip()
        if content:
            return SoulDocument(path=path, content=content, exists=True)
    return SoulDocument(path=path, content=DEFAULT_SOUL_TEXT.strip(), exists=False)


def review_soul_document(
    document: SoulDocument,
    recent_user_messages: list[str],
    *,
    dismissed_evidence_signatures: list[str] | None = None,
) -> SoulReview:
    text = document.content.lower()
    proposals: list[SoulProposal] = []
    dismissed = {
        str(item).strip()
        for item in (dismissed_evidence_signatures or [])
        if str(item).strip()
    }
    recent_messages = [item.strip() for item in recent_user_messages if item.strip()]
    tutorial_evidence = [
        item for item in recent_messages
        if any(token in item.lower() for token in ("tutorial", "inexperienced", "new", "learn"))
    ]
    autonomy_evidence = [
        item for item in recent_messages
        if any(token in item.lower() for token in ("sleep", "autonomous", "keep going", "continue", "proceed"))
    ]

    tutorial_amendment = (
        "- When introducing a new workflow, pair it with a short hands-on tutorial and "
        "name the stopping point."
    )
    tutorial_signature = _proposal_signature("operator_tutorial_duty", tutorial_evidence[:3])
    tutorial_resurfaced = any(
        item.startswith("operator_tutorial_duty:")
        for item in dismissed
    ) and tutorial_signature not in dismissed
    if (
        tutorial_evidence
        and tutorial_amendment.lower() not in text
        and tutorial_signature not in dismissed
    ):
        proposals.append(
            SoulProposal(
                proposal_id="operator_tutorial_duty",
                section="Operator care",
                title="Add guided tutorial duty",
                rationale=(
                    "Recent operator history emphasizes safe learning, so the soul should "
                    "explicitly require short guided walkthroughs for new workflows."
                ),
                amendment=tutorial_amendment,
                evidence=tutorial_evidence[:3],
                evidence_signature=tutorial_signature,
                explanation=(
                    "This proposal is active because recent operator messages emphasize tutorials "
                    "and beginner-safe guidance."
                ),
                resurfaced_after_dismissal=tutorial_resurfaced,
            )
        )

    autonomy_amendment = (
        "- When given room to proceed, batch work into verified chunks and stop at approval "
        "or ambiguity boundaries."
    )
    autonomy_signature = _proposal_signature("autonomous_chunking_rule", autonomy_evidence[:3])
    autonomy_resurfaced = any(
        item.startswith("autonomous_chunking_rule:")
        for item in dismissed
    ) and autonomy_signature not in dismissed
    if (
        autonomy_evidence
        and autonomy_amendment.lower() not in text
        and autonomy_signature not in dismissed
    ):
        proposals.append(
            SoulProposal(
                proposal_id="autonomous_chunking_rule",
                section="Planning posture",
                title="Add autonomous chunking rule",
                rationale=(
                    "Recent operator history granted longer autonomous runs, so the soul should "
                    "state how Ernie handles that freedom safely."
                ),
                amendment=autonomy_amendment,
                evidence=autonomy_evidence[:3],
                evidence_signature=autonomy_signature,
                explanation=(
                    "This proposal is active because recent operator messages grant longer autonomous "
                    "runs with explicit approval boundaries."
                ),
                resurfaced_after_dismissal=autonomy_resurfaced,
            )
        )

    return SoulReview(
        summary=document.ui_summary(),
        proposals=proposals,
        dismissed_evidence_signatures=sorted(dismissed),
    )


def apply_soul_proposal(document: SoulDocument, proposal: SoulProposal) -> SoulDocument:
    lines = document.content.splitlines()
    section_header = f"## {proposal.section}"
    try:
        section_index = next(
            index for index, line in enumerate(lines)
            if line.strip() == section_header
        )
    except StopIteration:
        updated_lines = list(lines)
        if updated_lines and updated_lines[-1].strip():
            updated_lines.append("")
        updated_lines.extend([section_header, "", proposal.amendment])
        updated_content = "\n".join(updated_lines).strip() + "\n"
        document.path.write_text(updated_content, encoding="utf-8")
        return SoulDocument(path=document.path, content=updated_content.strip(), exists=True)

    insert_at = len(lines)
    for index in range(section_index + 1, len(lines)):
        if lines[index].startswith("## "):
            insert_at = index
            break
    while insert_at > section_index + 1 and not lines[insert_at - 1].strip():
        insert_at -= 1
    updated_lines = list(lines)
    updated_lines[insert_at:insert_at] = ["", proposal.amendment]
    updated_content = "\n".join(updated_lines).strip() + "\n"
    document.path.write_text(updated_content, encoding="utf-8")
    return SoulDocument(path=document.path, content=updated_content.strip(), exists=True)


def _proposal_signature(proposal_id: str, evidence: list[str]) -> str:
    normalized = "\n".join(
        [proposal_id.strip().lower(), *[item.strip().lower() for item in evidence if item.strip()]]
    )
    return f"{proposal_id}:{hashlib.sha1(normalized.encode('utf-8')).hexdigest()[:12]}"
