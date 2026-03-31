# Ernie Plan

This document tracks two related plans:

1. how Ernie becomes more visibly LLM-capable without losing control
2. what the next best system steps are for the agent as a whole

The guiding constraint is simple: increase model leverage in explanation and reasoning before increasing model freedom in execution.

## Current baseline

Ernie already has model-backed paths for:

- normal reply generation through `MemoryFirstAgent.respond()`
- structured choice among planner-approved options through `MemoryFirstAgent.decide()`
- supervised pilot runs that can use the model to choose an approved action
- prompt conditioning from `SOUL.md`

Ernie does not yet feel fully LLM-native because the visible system is still mostly deterministic:

- planner scoring
- executor routing
- approval gates
- cockpit state rendering
- memory extraction and reflections

That is intentional. The control plane came first so the model has rails.

## LLM rollout plan

### Phase 1: model-backed explanations on top of deterministic execution

Goal: make Ernie feel smarter and more conversational without changing the safety boundary.

Implement:

- model-backed action explanations in cockpit
- model-backed tutorial summaries for first-run and demo flows
- model-backed soul proposal rationales and amendment summaries
- model-backed post-action narration: what changed, what did not, and what the next bounded step is

Guardrails:

- model explains, but does not directly execute
- planner recommendation and executor result remain source of truth
- approval prompts remain policy-driven

Success condition:

- the user experiences more intelligence in the interface without losing auditability

### Phase 2: model-assisted interpretation inside bounded planning

Goal: let the model help interpret context more often while still choosing only from constrained actions.

Implement:

- model-assisted prioritization narration for planner alternatives
- model-assisted task summarization for crowded queues
- model-backed translation from vague user goals into bounded planner queries
- optional model-backed extraction and reflection while keeping heuristics as the floor

Guardrails:

- execution must still route through approved planner/executor paths
- no free-form commands or writes from model output alone
- fallback remains deterministic

Success condition:

- Ernie feels more adaptive when the user asks messy or ambiguous things

### Phase 3: governed autonomous reasoning loops

Goal: allow longer model-guided runs without making behavior opaque.

Implement:

- model-backed chunk planning for multi-step supervised runs
- explicit stop conditions and debriefs for long runs
- richer self-review and self-improvement proposals
- stronger model use in preparing, delegating, and verifying work

Guardrails:

- approval and tool boundaries remain external to the model
- every long run leaves an audit trail
- model-selected actions still have to map to bounded executor operations

Success condition:

- Ernie can carry longer sessions without feeling brittle or losing legibility

## Whole-system next best step

The next best step for the system as a whole is:

**add model-backed explanation surfaces in cockpit and pilot while keeping execution deterministic**

Why this comes next:

- the control plane is now strong enough to support it
- it improves user-visible intelligence fast
- it does not weaken the safety model
- it makes future personality and tutorial work more valuable

Concrete slice:

1. add a model-backed `why this action` explainer for the current recommendation
2. add a model-backed `what just changed` summary after executor results
3. add a model-backed tutorial/narration pass for first-run and demo flows
4. keep planner score, executor result, and approval state as the hard facts beneath the explanation

## After that

Once the explanation layer is stable, the next likely steps are:

1. model-backed queue summarization and alternative comparison
2. model-assisted extraction/reflection with heuristics retained as fallback
3. only later, a separate governed personal-agency lane for non-work self-directed behavior

## Personal agency note

Personal independence should not be mixed into the operator lane.

When it is added, it should live behind:

- a distinct mode or lane
- explicit policy
- visible logs
- bounded budgets
- clear separation from operator work

That keeps Ernie understandable while still leaving room for emergence later.
