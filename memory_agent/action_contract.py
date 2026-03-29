from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .planner import PlannerAction, PlannerSnapshot

ACTION_TYPE_REPLY_ONLY = "reply_only"
ACTION_TYPE_ASK_USER = "ask_user"
ACTION_TYPE_EXECUTE_PLAN_ACTION = "execute_plan_action"
ACTION_TYPE_NOOP = "noop"

ALLOWED_ACTION_TYPES = {
    ACTION_TYPE_REPLY_ONLY,
    ACTION_TYPE_ASK_USER,
    ACTION_TYPE_EXECUTE_PLAN_ACTION,
    ACTION_TYPE_NOOP,
}


@dataclass(slots=True)
class ActionOption:
    option_id: str
    source: str
    action: PlannerAction


@dataclass(slots=True)
class ModelActionProposal:
    action_type: str
    assistant_message: str | None = None
    option_id: str | None = None
    rationale: str | None = None
    raw_text: str = ""
    raw_payload: dict[str, Any] | None = None
    parse_error: str | None = None


@dataclass(slots=True)
class ValidatedModelAction:
    action_type: str
    assistant_message: str | None = None
    chosen_option: ActionOption | None = None
    rationale: str | None = None
    raw_text: str = ""
    parse_error: str | None = None
    validation_error: str | None = None
    fallback_to_reply: bool = False

    @property
    def chosen_action(self) -> PlannerAction | None:
        if self.chosen_option is None:
            return None
        return self.chosen_option.action


def build_action_options(
    plan: PlannerSnapshot,
    *,
    limit: int = 5,
) -> list[ActionOption]:
    options: list[ActionOption] = []
    if plan.recommendation is not None:
        options.append(ActionOption(option_id="A1", source="recommendation", action=plan.recommendation))
    for index, action in enumerate(plan.alternatives[: max(limit - len(options), 0)], start=len(options) + 1):
        options.append(ActionOption(option_id=f"A{index}", source="alternative", action=action))
    return options


def render_action_contract(options: list[ActionOption]) -> str:
    lines = [
        "Structured action contract:",
        "Respond with JSON only. Do not wrap it in markdown fences.",
        (
            '{"assistant_message":"...","action":{"type":"reply_only|ask_user|'
            'execute_plan_action|noop","option_id":"A1","rationale":"..."}}'
        ),
        "Rules:",
        "- Use execute_plan_action only when one planner-approved option clearly fits the user's request.",
        "- Never invent tasks, commands, action kinds, or option IDs.",
        "- Use ask_user when a short clarifying question is required before acting.",
        "- Use reply_only for explanation, advice, or status updates without executor action.",
        "- option_id is required only for execute_plan_action.",
        "Planner-approved action options:",
    ]
    if not options:
        lines.append("- none")
    else:
        for option in options:
            lines.append(
                f"- {option.option_id} [{option.action.kind}] {option.action.title}: "
                f"{option.action.summary} ({option.source})"
            )
    return "\n".join(lines)


def parse_model_action_response(raw_text: str) -> ModelActionProposal:
    payload, error = _extract_json_payload(raw_text)
    stripped = raw_text.strip()
    if payload is None:
        return ModelActionProposal(
            action_type=ACTION_TYPE_REPLY_ONLY,
            assistant_message=stripped or None,
            raw_text=raw_text,
            parse_error=error or "structured_json_not_found",
        )

    action_payload = payload.get("action", {})
    if not isinstance(action_payload, dict):
        return ModelActionProposal(
            action_type=ACTION_TYPE_REPLY_ONLY,
            assistant_message=_coerce_message(payload),
            raw_text=raw_text,
            raw_payload=payload,
            parse_error="action_payload_must_be_object",
        )

    action_type = str(
        action_payload.get("type")
        or payload.get("action_type")
        or ACTION_TYPE_REPLY_ONLY
    ).strip() or ACTION_TYPE_REPLY_ONLY
    option_id = str(
        action_payload.get("option_id")
        or payload.get("option_id")
        or ""
    ).strip() or None
    rationale = str(
        action_payload.get("rationale")
        or payload.get("rationale")
        or ""
    ).strip() or None
    return ModelActionProposal(
        action_type=action_type,
        assistant_message=_coerce_message(payload),
        option_id=option_id,
        rationale=rationale,
        raw_text=raw_text,
        raw_payload=payload,
    )


def validate_model_action(
    proposal: ModelActionProposal,
    options: list[ActionOption],
) -> ValidatedModelAction:
    assistant_message = (proposal.assistant_message or "").strip() or None
    if proposal.action_type not in ALLOWED_ACTION_TYPES:
        return ValidatedModelAction(
            action_type=ACTION_TYPE_REPLY_ONLY,
            assistant_message=assistant_message or proposal.raw_text.strip() or None,
            rationale=proposal.rationale,
            raw_text=proposal.raw_text,
            parse_error=proposal.parse_error,
            validation_error=f"unsupported_action_type={proposal.action_type}",
            fallback_to_reply=True,
        )

    if proposal.action_type == ACTION_TYPE_EXECUTE_PLAN_ACTION:
        if proposal.option_id is None:
            return ValidatedModelAction(
                action_type=ACTION_TYPE_REPLY_ONLY,
                assistant_message=assistant_message or proposal.raw_text.strip() or None,
                rationale=proposal.rationale,
                raw_text=proposal.raw_text,
                parse_error=proposal.parse_error,
                validation_error="missing_option_id",
                fallback_to_reply=True,
            )
        option_map = {option.option_id: option for option in options}
        chosen_option = option_map.get(proposal.option_id)
        if chosen_option is None:
            return ValidatedModelAction(
                action_type=ACTION_TYPE_REPLY_ONLY,
                assistant_message=assistant_message or proposal.raw_text.strip() or None,
                rationale=proposal.rationale,
                raw_text=proposal.raw_text,
                parse_error=proposal.parse_error,
                validation_error=f"unknown_option_id={proposal.option_id}",
                fallback_to_reply=True,
            )
        return ValidatedModelAction(
            action_type=ACTION_TYPE_EXECUTE_PLAN_ACTION,
            assistant_message=assistant_message,
            chosen_option=chosen_option,
            rationale=proposal.rationale,
            raw_text=proposal.raw_text,
            parse_error=proposal.parse_error,
        )

    if proposal.action_type == ACTION_TYPE_ASK_USER:
        return ValidatedModelAction(
            action_type=ACTION_TYPE_ASK_USER,
            assistant_message=assistant_message or "I need one detail before I can safely continue.",
            rationale=proposal.rationale,
            raw_text=proposal.raw_text,
            parse_error=proposal.parse_error,
        )

    return ValidatedModelAction(
        action_type=proposal.action_type,
        assistant_message=assistant_message,
        rationale=proposal.rationale,
        raw_text=proposal.raw_text,
        parse_error=proposal.parse_error,
        fallback_to_reply=proposal.parse_error is not None,
    )


def _coerce_message(payload: dict[str, Any]) -> str | None:
    for key in ("assistant_message", "reply", "message"):
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _extract_json_payload(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    last_error: str | None = None
    for candidate in _candidate_json_blobs(raw_text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = f"invalid_json:{exc.msg}"
            continue
        if isinstance(parsed, dict):
            return parsed, None
        last_error = "json_payload_must_be_object"
    return None, last_error


def _candidate_json_blobs(raw_text: str) -> list[str]:
    stripped = raw_text.strip()
    candidates: list[str] = []
    if stripped:
        candidates.append(stripped)

    for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.IGNORECASE | re.DOTALL):
        block = match.group(1).strip()
        if block:
            candidates.append(block)

    first_brace = raw_text.find("{")
    last_brace = raw_text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(raw_text[first_brace : last_brace + 1].strip())

    unique_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return unique_candidates
