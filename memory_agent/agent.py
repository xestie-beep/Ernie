from __future__ import annotations

from dataclasses import dataclass, field

from .action_contract import (
    ACTION_TYPE_ASK_USER,
    ACTION_TYPE_EXECUTE_PLAN_ACTION,
    ActionOption,
    ValidatedModelAction,
    build_action_options,
    parse_model_action_response,
    render_action_contract,
    validate_model_action,
)
from .executor import ExecutionCycle, ExecutorResult, MemoryExecutor
from .improvement import ImprovementReviewReport, MemoryImprovementEngine
from .memory import MemoryStore
from .model_adapter import BaseModelAdapter, ModelMessage, ModelResponse, build_default_model_adapter
from .models import ContextWindow, MemoryRecord, SearchResult
from .planner import MemoryPlanner, PlannerAction, PlannerSnapshot


@dataclass(slots=True)
class TurnReport:
    event_id: int
    stored_memories: list[MemoryRecord] = field(default_factory=list)
    retrieved_memories: list[SearchResult] = field(default_factory=list)
    context: ContextWindow | None = None
    plan: PlannerSnapshot | None = None

    def render(self) -> str:
        lines = [f"Event #{self.event_id}"]
        lines.append("")
        lines.append("Stored memories:")
        if not self.stored_memories:
            lines.append("- none")
        else:
            for memory in self.stored_memories:
                lines.append(f"- [{memory.kind}] {memory.content}")
        if self.context is not None:
            lines.append("")
            lines.append(self.context.render())
        if self.plan is not None:
            lines.append("")
            lines.append(self.plan.render())
        return "\n".join(lines)


@dataclass(slots=True)
class ReplyReport:
    user_event_id: int
    user_memories: list[MemoryRecord] = field(default_factory=list)
    context: ContextWindow | None = None
    plan: PlannerSnapshot | None = None
    assistant_event_id: int | None = None
    assistant_message: str | None = None
    assistant_memories: list[MemoryRecord] = field(default_factory=list)
    model_response: ModelResponse | None = None
    model_status: dict[str, object] = field(default_factory=dict)
    model_action: ValidatedModelAction | None = None
    execution_result: ExecutorResult | None = None
    after_plan: PlannerSnapshot | None = None
    error: str | None = None

    def render(self) -> str:
        lines = [f"User event #{self.user_event_id}"]
        if self.context is not None:
            lines.extend(["", self.context.render()])
        if self.plan is not None:
            lines.extend(["", self.plan.render()])
        if self.model_action is not None:
            lines.extend(["", self._render_model_action()])
        if self.assistant_message is not None:
            lines.extend(["", "Assistant response:", self.assistant_message])
        if self.execution_result is not None:
            lines.extend(["", self.execution_result.render()])
        if self.after_plan is not None:
            lines.extend(["", "Updated plan:", self.after_plan.render()])
        if self.error is not None:
            lines.extend(["", f"Model error: {self.error}"])
        return "\n".join(lines)

    def _render_model_action(self) -> str:
        assert self.model_action is not None
        action = self.model_action
        parts = [f"Model action: {action.action_type}"]
        if action.chosen_option is not None:
            parts.append(
                f"{action.chosen_option.option_id} -> "
                f"[{action.chosen_option.action.kind}] {action.chosen_option.action.title}"
            )
        if action.validation_error is not None:
            parts.append(f"validation={action.validation_error}")
        if action.parse_error is not None:
            parts.append(f"parse={action.parse_error}")
        return " | ".join(parts)


class MemoryFirstAgent:
    """A local agent scaffold that makes memory available before model wiring."""

    def __init__(
        self,
        memory_store: MemoryStore,
        *,
        model_adapter: BaseModelAdapter | None = None,
    ):
        self.memory_store = memory_store
        self.planner = MemoryPlanner(memory_store)
        self.executor = MemoryExecutor(memory_store)
        self.model_adapter = model_adapter or build_default_model_adapter()

    def observe_message(self, role: str, text: str) -> TurnReport:
        event_id, stored_memories = self.memory_store.observe(role=role, content=text)
        context = self.memory_store.build_context(query=text)
        plan = self.planner.build_plan(query=text, context=context)
        return TurnReport(
            event_id=event_id,
            stored_memories=stored_memories,
            retrieved_memories=context.memories,
            context=context,
            plan=plan,
        )

    def observe_user_message(self, text: str) -> TurnReport:
        return self.observe_message(role="user", text=text)

    def observe_assistant_message(self, text: str) -> TurnReport:
        return self.observe_message(role="assistant", text=text)

    def plan(self, query: str = "next best action") -> PlannerSnapshot:
        return self.planner.build_plan(query=query)

    def execute_next(
        self,
        query: str = "next best action",
        *,
        action_limit: int = 5,
    ) -> ExecutionCycle:
        return self.executor.execute_next(query=query, action_limit=action_limit)

    def model_status(self) -> dict[str, object]:
        return self.model_adapter.status()

    def review_self_improvement(
        self,
        evaluator,
        *,
        suite_name: str = "builtin",
        promote_limit: int = 3,
        include_strategic_backlog: bool = True,
    ) -> ImprovementReviewReport:
        engine = MemoryImprovementEngine(
            self.memory_store,
            evaluator,
            suite_name=suite_name,
        )
        return engine.review(
            promote_limit=promote_limit,
            include_strategic_backlog=include_strategic_backlog,
        )

    def respond(self, text: str) -> ReplyReport:
        event_id, stored_memories, context, plan, status = self._prepare_turn(
            text,
            action_limit=5,
        )
        if not bool(status.get("enabled")):
            return ReplyReport(
                user_event_id=event_id,
                user_memories=stored_memories,
                context=context,
                plan=plan,
                model_status=status,
                error="No main chat model is configured.",
            )

        messages = self._build_model_messages(text=text, context=context, plan=plan)
        try:
            response = self.model_adapter.chat(messages)
        except Exception as exc:
            return ReplyReport(
                user_event_id=event_id,
                user_memories=stored_memories,
                context=context,
                plan=plan,
                model_status=status,
                error=str(exc),
            )

        assistant_event_id, assistant_memories = self._store_assistant_message(
            response.content,
            metadata={
                "response_mode": "reply",
                "model_backend": status.get("backend"),
                "model_name": status.get("model"),
            },
        )
        return ReplyReport(
            user_event_id=event_id,
            user_memories=stored_memories,
            context=context,
            plan=plan,
            assistant_event_id=assistant_event_id,
            assistant_message=response.content,
            assistant_memories=assistant_memories,
            model_response=response,
            model_status=status,
        )

    def decide(
        self,
        text: str,
        *,
        execute_actions: bool = True,
        action_limit: int = 5,
    ) -> ReplyReport:
        event_id, stored_memories, context, plan, status = self._prepare_turn(
            text,
            action_limit=action_limit,
        )
        if not bool(status.get("enabled")):
            return ReplyReport(
                user_event_id=event_id,
                user_memories=stored_memories,
                context=context,
                plan=plan,
                model_status=status,
                error="No main chat model is configured.",
            )

        options = build_action_options(plan, limit=action_limit)
        messages = self._build_decision_messages(
            text=text,
            context=context,
            plan=plan,
            options=options,
        )
        try:
            response = self.model_adapter.chat(messages)
        except Exception as exc:
            return ReplyReport(
                user_event_id=event_id,
                user_memories=stored_memories,
                context=context,
                plan=plan,
                model_status=status,
                error=str(exc),
            )

        proposal = parse_model_action_response(response.content)
        validated_action = validate_model_action(proposal, options)

        assistant_message = validated_action.assistant_message
        execution_result = None
        after_plan = None
        if execute_actions:
            if (
                validated_action.action_type == ACTION_TYPE_EXECUTE_PLAN_ACTION
                and validated_action.chosen_action is not None
            ):
                execution_result = self.executor.execute_action(validated_action.chosen_action)
                after_plan = self.planner.build_plan(query=text, action_limit=action_limit)
            elif validated_action.action_type == ACTION_TYPE_ASK_USER:
                execution_result = self.executor.execute_action(
                    PlannerAction(
                        kind="ask_user",
                        title="Ask the user for clarification",
                        summary=assistant_message or "I need one detail before I can safely continue.",
                        score=0.0,
                        reasons=["model_requested_clarification"],
                        metadata={"area": "execution"},
                    )
                )
                after_plan = self.planner.build_plan(query=text, action_limit=action_limit)

        if (assistant_message is None or not assistant_message.strip()) and execution_result is not None:
            assistant_message = execution_result.prompt or execution_result.summary

        assistant_event_id = None
        assistant_memories: list[MemoryRecord] = []
        if assistant_message is not None and assistant_message.strip():
            assistant_event_id, assistant_memories = self._store_assistant_message(
                assistant_message,
                metadata={
                    "response_mode": "structured_decision",
                    "model_backend": status.get("backend"),
                    "model_name": status.get("model"),
                    "model_action_type": validated_action.action_type,
                    "model_action_option_id": (
                        validated_action.chosen_option.option_id
                        if validated_action.chosen_option is not None
                        else None
                    ),
                    "execution_kind": (
                        execution_result.executed_kind if execution_result is not None else None
                    ),
                    "execution_status": (
                        execution_result.status if execution_result is not None else None
                    ),
                },
            )

        return ReplyReport(
            user_event_id=event_id,
            user_memories=stored_memories,
            context=context,
            plan=plan,
            assistant_event_id=assistant_event_id,
            assistant_message=assistant_message,
            assistant_memories=assistant_memories,
            model_response=response,
            model_status=status,
            model_action=validated_action,
            execution_result=execution_result,
            after_plan=after_plan,
        )

    def _build_model_messages(
        self,
        *,
        text: str,
        context: ContextWindow,
        plan: PlannerSnapshot,
    ) -> list[ModelMessage]:
        system_prompt = "\n\n".join(
            [
                (
                    "You are the main reasoning model for a memory-first local agent. "
                    "Use the provided memory and planner state as source of truth."
                ),
                (
                    "Guardrails:\n"
                    "- Do not claim you executed commands or changed task state unless it appears in the provided context.\n"
                    "- If key information is missing or a blocker is external, ask a short direct question.\n"
                    "- Prefer concise, practical answers.\n"
                    "- Respect the recommended next action when it clearly fits the user's request."
                ),
                "Memory context:\n" + context.render(),
                "Planner state:\n" + plan.render(),
            ]
        )
        return [
            ModelMessage(role="system", content=system_prompt),
            ModelMessage(role="user", content=text),
        ]

    def _build_decision_messages(
        self,
        *,
        text: str,
        context: ContextWindow,
        plan: PlannerSnapshot,
        options: list[ActionOption],
    ) -> list[ModelMessage]:
        system_prompt = "\n\n".join(
            [
                (
                    "You are the main reasoning model for a memory-first local agent. "
                    "Use the provided memory and planner state as source of truth."
                ),
                (
                    "Guardrails:\n"
                    "- You may only choose from the planner-approved action options.\n"
                    "- Do not invent tasks, commands, or option IDs.\n"
                    "- If the user only needs explanation, use reply_only.\n"
                    "- If a missing detail blocks safe execution, use ask_user with one short question."
                ),
                render_action_contract(options),
                "Memory context:\n" + context.render(),
                "Planner state:\n" + plan.render(),
            ]
        )
        return [
            ModelMessage(role="system", content=system_prompt),
            ModelMessage(role="user", content=text),
        ]

    def _prepare_turn(
        self,
        text: str,
        *,
        action_limit: int,
    ) -> tuple[int, list[MemoryRecord], ContextWindow, PlannerSnapshot, dict[str, object]]:
        event_id, stored_memories = self.memory_store.observe(role="user", content=text)
        context = self.memory_store.build_context(query=text)
        plan = self.planner.build_plan(query=text, context=context, action_limit=action_limit)
        status = self.model_adapter.status()
        return event_id, stored_memories, context, plan, status

    def _store_assistant_message(
        self,
        text: str,
        *,
        metadata: dict[str, object] | None = None,
    ) -> tuple[int, list[MemoryRecord]]:
        return self.memory_store.observe(
            role="assistant",
            content=text,
            metadata=metadata,
        )
