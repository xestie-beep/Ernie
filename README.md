# Memory-First Agent

This project is a local-first agent scaffold built around the cheapest durable memory stack that is still worth scaling:

- `SQLite` for persistence
- `FTS5` for fast lexical retrieval
- heuristic memory extraction for zero-model baseline behavior
- source-linked reflection and compaction
- stable profile synthesis
- evidence-backed context retrieval
- contradiction tracking
- entity resolution with canonical IDs and alias-aware recall
- dependency-aware open loops with blockers, due dates, and entity-linked task relationships
- ready-now and overdue task views for execution prioritization
- automatic task review that generates cooldown-based nudges for stale or overdue work
- recurring task cadence, snooze/resume flows, unblock flows, and escalation after repeated nudges
- planner-layer next-action recommendations grounded in ready tasks, blocked work, nudges, and maintenance state
- a bounded executor layer for safe internal actions and plan-execute-observe-replan control loops
- a Linux-first guarded shell adapter for bounded local command execution
- a workspace-bound file adapter for bounded text reads and narrow edits
- a pluggable single-main-model adapter with an Ollama chat backend
- a structured action contract that lets the main model choose only planner-approved actions
- a supervised Linux pilot runtime with approval gating, TOML policy, and per-turn trace logs
- diff-first pilot write execution, so risky file changes generate a review packet before apply
- disposable git-branch apply and rollback for approved patch previews when the workspace is a clean git repo
- pilot-history-aware preparation actions that split approval-prone work into safer follow-up steps before supervised runs stall again
- an eval-gated self-improvement loop with baseline history and promoted follow-up tasks
- a patch-runner that trials candidate file changes in a temp workspace before applying them back
- Python symbol-aware patching for safe function/class replacement, insertion, deletion, import edits, file-scoped rename, bounded signature refactors, cross-file export renames, and conservative module moves
- due-based maintenance scheduling
- optional local semantic reranking via Ollama
- first-class task, decision, and tool-outcome memories
- built-in evaluation harness for recall and fidelity regression checks
- a clean handoff point for a future local or hosted model backend

## Why this memory design first

If we optimize for cost, efficiency, and iteration speed, a vector database is usually the wrong first move. The stronger starting point is a tiered memory system:

1. raw events for perfect recall
2. distilled long-term memories for important facts and constraints
3. non-destructive reflection that compacts many memories into traceable summaries
4. stable profiles that preserve enduring goals, constraints, and preferences
5. retrieval that mixes text relevance, importance, confidence, recency, and provenance

That gives us:

- no recurring infrastructure cost
- very low local latency
- explainable retrieval
- a future path to optional local embeddings without rewriting storage

## Project layout

- `memory_agent/memory.py`: durable memory store and retrieval engine
- `memory_agent/extractors.py`: lightweight fact and preference extraction
- `memory_agent/reflection.py`: heuristic reflection and compaction
- `memory_agent/evaluation.py`: benchmark scenarios and scoring
- `memory_agent/agent.py`: local agent wrapper around the memory store
- `memory_agent/planner.py`: next-action planner over execution and maintenance state
- `memory_agent/executor.py`: bounded executor for task starts, blocker routing, maintenance, and ask-user fallbacks
- `memory_agent/shell_adapter.py`: guarded local command execution for task-attached shell work
- `memory_agent/file_adapter.py`: guarded workspace file reads and narrow text edits for task-attached file work
- `memory_agent/model_adapter.py`: main-model interface and Ollama chat backend
- `memory_agent/action_contract.py`: planner-approved model action schema, parsing, and validation
- `memory_agent/improvement.py`: eval-gated self-improvement review and backlog promotion
- `memory_agent/migration.py`: Linux handoff bundle export and restore helpers
- `memory_agent/patch_runner.py`: temp-workspace patch validation and apply-back flow
- `memory_agent/cli.py`: interactive and command-line entry points
- `scripts/bootstrap_linux.sh`: Linux virtualenv/bootstrap helper
- `tests/test_memory.py`: regression tests for the memory behavior

## Quick start

```powershell
python -m memory_agent.cli observe "We are building the agent from scratch and it must run locally."
python -m memory_agent.cli search "local agent"
python -m memory_agent.cli model-status
python -m memory_agent.cli reply "What should I do next?"
python -m memory_agent.cli decide "What should I do next?"
python -m memory_agent.cli improve
python -m memory_agent.cli handoff-pack
python -m memory_agent.cli patch-run "try README patch" --file-op replace_text --file-path README.md --find-text "foo" --file-text "bar"
python -m memory_agent.cli patch-run "upgrade greet helper" --file-op replace_python_function --file-path memory_agent/file_adapter.py --symbol-name _trim_preview --file-text "def _trim_preview(self, text: str) -> str:\n    stripped = text.strip()\n    if len(stripped) <= self.preview_char_limit:\n        return stripped\n    return stripped[: self.preview_char_limit - 3] + '...'\n"
python -m memory_agent.cli patch-run "insert helper" --file-op insert_python_after_symbol --file-path app/helpers.py --symbol-name greet --file-text "def helper() -> str:\n    return 'helper'\n"
python -m memory_agent.cli patch-run "rename greet" --file-op rename_python_identifier --file-path app/helpers.py --symbol-name greet --file-text salute
python -m memory_agent.cli plan "what should I do next"
python -m memory_agent.cli task "Check CLI help" --command "python3 -m memory_agent.cli --help" --complete-on-success
python -m memory_agent.cli task "Patch notes file" --file-op replace_text --file-path README.md --find-text "foo" --file-text "bar" --complete-on-success
python -m memory_agent.cli execute "what should I do next"
python -m memory_agent.cli reflect
python -m memory_agent.cli history 1
python -m memory_agent.cli chat
```

The default database lives at `.agent/agent_memory.sqlite3` in the workspace.

## Linux handoff

If you want to move the working agent state to a Linux box without juggling multiple files by hand, use the handoff bundle flow:

```powershell
python -m memory_agent.cli handoff-pack
python -m memory_agent.cli handoff-pack --json
```

That creates a bundle under `.agent/handoffs/...` containing:

- a clean backup copy of `.agent/agent_memory.sqlite3`
- `.agent/pilot_policy.toml` if it exists
- `.agent/pilot_traces/...` by default
- a machine-readable manifest
- a short handoff summary

On Linux, after cloning the repo:

```bash
./scripts/bootstrap_linux.sh
python3 -m memory_agent.cli handoff-restore /path/to/linux_handoff.zip
python3 -m memory_agent.cli pilot-chat --no-model
```

If you want to restore into a different checkout root:

```bash
python3 -m memory_agent.cli handoff-restore /path/to/linux_handoff.zip --target-root /path/to/repo
```

The repo also includes a `.gitignore` that keeps live local state like `.agent/agent_memory.sqlite3`, pilot traces, and handoff bundles out of Git by default.

## Reflection and versioning

The memory layer now uses a lossless compaction model:

- raw events stay append-only
- atomic memories keep extracted facts, preferences, goals, and constraints
- reflections summarize related atomic memories by subject
- profiles summarize the stable long-term shape of each subject
- profiles synthesize domain-aware sections for decisions, tasks, and tool outcomes
- superseded memories are archived instead of deleted
- every reflection keeps provenance links back to its source memories
- context retrieval includes evidence memories and supporting events
- contradictions are preserved as explicit links instead of silently overwriting memory
- recurring concepts are linked to canonical entities so alias-style queries can still find the right memory
- tasks can carry due dates, blockers, and dependencies without losing that state when the task status changes
- ready tasks and overdue tasks are surfaced separately so blocked work does not crowd out actionable work
- maintenance can emit source-linked nudge memories when a task is overdue, blocked too long, or untouched too long
- recurring tasks can roll forward automatically, snoozed tasks stay out of the ready queue, and repeated nudges can escalate
- the planner can choose whether the next best move is working a ready task, clearing a blocker, or running maintenance
- the executor can safely start a task, reroute work to a prerequisite, run maintenance, or fall back to asking the user
- command-bearing tasks can execute through a guarded Linux-first shell adapter with allowlisted prefixes and workspace-bound `cwd`
- file-bearing tasks can execute through a guarded workspace file adapter for `read_text`, `write_text`, `append_text`, and exact `replace_text`
- a single main model can answer with memory and planner context without gaining direct execution authority
- the main model can optionally choose one planner-approved action through a narrow JSON contract instead of inventing new actions
- improvement reviews persist evaluation baselines, compare against prior runs, and promote top self-improvement tasks
- patch runs can copy the workspace into `.agent/patch_runs`, apply bounded file edits there, run validations, compare against the eval baseline, and only then apply the changed files back
- when the workspace is a clean git repo, successful apply-back can land on a disposable `codex/...` branch with a recorded rollback hint instead of touching the current branch directly
- Python files can be patched by exact symbol name so the agent can replace, insert around, delete, or rename a symbol and reject the patch if the updated file no longer parses
- maintenance decides when contradiction scans, reflections, and profiles are due
- semantic reranking is optional and blends into retrieval instead of replacing the graph

Useful commands:

```powershell
python -m memory_agent.cli observe --role assistant "Implemented local semantic reranking and tests passed."
python -m memory_agent.cli entities "database choice"
python -m memory_agent.cli model-status
python -m memory_agent.cli reply "What should I do next?"
python -m memory_agent.cli decide "What should I do next?"
python -m memory_agent.cli decide "What should I do next?" --preview
python -m memory_agent.cli improve
python -m memory_agent.cli improve --preview
python -m memory_agent.cli patch-run "try README patch" --file-op replace_text --file-path README.md --find-text "foo" --file-text "bar"
python -m memory_agent.cli patch-run "ship README patch" --file-op replace_text --file-path README.md --find-text "foo" --file-text "bar" --apply-on-success
python -m memory_agent.cli patch-run "upgrade trim preview" --file-op replace_python_function --file-path memory_agent/file_adapter.py --symbol-name _trim_preview --file-text "def _trim_preview(self, text: str) -> str:\n    stripped = text.strip()\n    if len(stripped) <= self.preview_char_limit:\n        return stripped\n    return stripped[: self.preview_char_limit - 3] + '...'\n"
python -m memory_agent.cli patch-run "insert helper" --file-op insert_python_after_symbol --file-path app/helpers.py --symbol-name greet --file-text "def helper() -> str:\n    return 'helper'\n"
python -m memory_agent.cli patch-run "rename greet" --file-op rename_python_identifier --file-path app/helpers.py --symbol-name greet --file-text salute
python -m memory_agent.cli patch-run "delete helper" --file-op delete_python_symbol --file-path app/helpers.py --symbol-name helper
python -m memory_agent.cli task "Finish entity resolution" --status in_progress --due-date 2026-04-01
python -m memory_agent.cli task "Build task graph maintenance" --status blocked --depends-on "Finish entity resolution" --blocked-by "Finish entity resolution" --due-date 2026-04-02
python -m memory_agent.cli task "Weekly memory audit" --due-date 2026-04-01 --recurrence-days 7
python -m memory_agent.cli task "Check CLI help" --command "python3 -m memory_agent.cli --help" --complete-on-success
python -m memory_agent.cli task "Patch notes file" --file-op replace_text --file-path README.md --find-text "foo" --file-text "bar" --complete-on-success
python -m memory_agent.cli plan "what should I do next"
python -m memory_agent.cli execute "what should I do next"
python -m memory_agent.cli ready
python -m memory_agent.cli overdue
python -m memory_agent.cli snooze-task "Weekly memory audit" --until 2026-04-05
python -m memory_agent.cli resume-task "Weekly memory audit"
python -m memory_agent.cli unblock-task "Build task graph maintenance"
python -m memory_agent.cli complete-task "Weekly memory audit"
python -m memory_agent.cli review-tasks
python -m memory_agent.cli decision architecture "Use SQLite as the source of truth"
python -m memory_agent.cli tool-outcome tests "All verification checks passed" --subject architecture
python -m memory_agent.cli evaluate
python -m memory_agent.cli reflect --limit 20 --max-reflections 5
python -m memory_agent.cli maintain
python -m memory_agent.cli revise 1 "The agent should run locally on the user's main PC and laptop."
python -m memory_agent.cli history 8
python -m memory_agent.cli context "local cost memory"
```

## Evaluation harness

The project now includes a built-in benchmark suite that measures whether the memory system still does the important things well:

- recalls core constraints and preferences
- preserves latest truth after revisions
- resolves recurring concepts through entity aliases as well as direct wording
- surfaces contradictions explicitly
- builds operational profiles from decisions, tasks, and tool outcomes
- preserves dependency-aware open-loop state for tasks
- prioritizes ready and overdue work into explicit execution views
- automatically reviews stale execution state and emits nudges for overdue or stuck work
- supports recurring task cadence, snoozing, resume/unblock flows, and escalation after repeated reminders
- recommends the next best action from execution state and memory maintenance signals
- executes bounded internal actions and records tool outcomes back into memory
- runs allowlisted local commands such as `python3 -m ...`, `pytest`, `git status`, and `ruff check` through the shell adapter
- runs workspace-bound file tasks such as exact text replacement without allowing arbitrary paths outside the workspace
- can drive a single main model with curated memory context and planner state
- can let that main model select a validated planner-approved action and route it through the executor
- extracts useful state from assistant/tool progress updates
- records evaluation runs and can promote self-improvement work from regressions, failures, and strategic backlog items
- trials candidate patches in a temp workspace and only applies them back when validations and evals stay green
- supports code-aware Python symbol edits so candidate patches can target one function/class or a file-scoped identifier instead of rewriting whole files

Run it any time with:

```powershell
python -m memory_agent.cli evaluate
python -m memory_agent.cli evaluate --json
```

Every `evaluate` run is also written into the local SQLite store as an evaluation baseline so later self-improvement reviews can compare against previous and best-known scores.

## Self-improvement loop

The system now has a first-class self-improvement review path:

```powershell
python -m memory_agent.cli improve
python -m memory_agent.cli improve --preview
python -m memory_agent.cli stats
```

What it does:

- runs the built-in evaluation suite
- stores the result as a baseline in SQLite
- compares the current score with previous and best runs
- inspects recent blocked/error tool outcomes
- promotes the top opportunities into `self_improvement` tasks
- keeps strategic backlog items like `Add code-aware patching primitives` from getting forgotten

This is intentionally eval-gated. The agent does not get to declare that it improved itself just because it changed something; the benchmark history is the receipt.

Patch candidates can now move through the same safety gate:

```powershell
python -m memory_agent.cli patch-run "try code-aware patching" --file-op replace_text --file-path README.md --find-text "foo" --file-text "bar"
python -m memory_agent.cli patch-run "apply code-aware patching" --file-op replace_text --file-path README.md --find-text "foo" --file-text "bar" --task-title "Add code-aware patching primitives" --apply-on-success
python -m memory_agent.cli patch-run "apply code-aware patching on branch" --file-op replace_text --file-path README.md --find-text "foo" --file-text "bar" --apply-on-success --git-mode branch
python -m memory_agent.cli patch-rollback --run-id 12
```

What the patch runner does:

- copies the workspace into `.agent/patch_runs/.../workspace`
- applies bounded file edits there first
- runs validation commands in the temp workspace
- runs the built-in eval suite in the temp workspace and compares it to the stored baseline
- records the patch run result in SQLite
- only copies changed files back to the main workspace when the candidate passes and you opt into `--apply-on-success`
- rolls copied files back if the promotion step fails partway through
- when `--git-mode auto` or `--git-mode branch` is active and the workspace is a clean git repo, creates a disposable `codex/patch-run/...` branch, commits the approved change there, and records rollback metadata
- falls back to direct apply in `auto` mode when git is unavailable, the repo is dirty, or the workspace is detached from a named branch

For Python code, you can now target a specific symbol directly:

```powershell
python -m memory_agent.cli patch-run "upgrade greet helper" --file-op replace_python_function --file-path app/helpers.py --symbol-name greet --file-text "def greet(name: str) -> str:\n    return f'hello there {name}'\n"
python -m memory_agent.cli patch-run "upgrade greeter class" --file-op replace_python_class --file-path app/helpers.py --symbol-name Greeter --file-text "class Greeter:\n    def speak(self) -> str:\n        return 'hello'\n"
python -m memory_agent.cli patch-run "insert helper" --file-op insert_python_after_symbol --file-path app/helpers.py --symbol-name greet --file-text "def helper() -> str:\n    return 'helper'\n"
python -m memory_agent.cli patch-run "rename greet" --file-op rename_python_identifier --file-path app/helpers.py --symbol-name greet --file-text salute
python -m memory_agent.cli patch-run "delete helper" --file-op delete_python_symbol --file-path app/helpers.py --symbol-name helper
python -m memory_agent.cli patch-run "rename method" --file-op rename_python_method --file-path app/helpers.py --symbol-name Greeter.speak --file-text salute
python -m memory_agent.cli patch-run "add pathlib import" --file-op add_python_import --file-path app/helpers.py --file-text "from pathlib import Path"
python -m memory_agent.cli patch-run "remove pathlib import" --file-op remove_python_import --file-path app/helpers.py --file-text "from pathlib import Path"
python -m memory_agent.cli patch-run "add excited param" --file-op add_python_function_parameter --file-path app/helpers.py --symbol-name greet --file-text "{\"parameter_name\":\"excited\",\"call_argument\":\"True\"}"
python -m memory_agent.cli patch-run "add method param" --file-op add_python_method_parameter --file-path app/helpers.py --symbol-name Greeter.speak --file-text "{\"parameter_name\":\"excited\",\"call_argument\":\"True\"}"
python -m memory_agent.cli patch-run "rename exported greet" --file-op rename_python_export_across_imports --file-path app/helpers.py --symbol-name greet --file-text salute
python -m memory_agent.cli patch-run "move greet to core module" --file-op move_python_export_to_module --file-path app/helpers.py --symbol-name greet --file-text "{\"destination_path\":\"app/core.py\"}"
```

Those operations:

- locate the exact Python function or class by symbol name
- replace only that symbol span, including decorators
- can insert new code immediately before or after a target symbol
- can delete a target function or class cleanly
- can rename a file-scoped Python identifier without touching comments or string literals
- can rename a class method and update file-scoped attribute call sites such as `obj.method()` and `Class.method`
- can add an import into the top-level import section while respecting module docstrings and `from __future__` imports
- can remove an exact top-level import statement cleanly
- can add a parameter to a target function or method and rewrite simple same-file call sites using a JSON spec
- blocks signature refactors when the file needs multiline call-site rewrites without a default fallback
- can rename a top-level exported function or class and update matching import sites plus simple module attribute uses across the workspace
- blocks cross-file export renames when a consumer file shadows the imported symbol and the rewrite would be ambiguous
- can move a top-level exported function or class into another module, copy over required top-level import statements, update straightforward `from ... import ...` consumers, and keep the source module alive through a compatibility re-export
- blocks module moves when consumer files would need import splitting or when the moved export depends on other source-local bindings
- reject the patch if the source file cannot be parsed before the edit
- reject the patch if the updated file no longer parses after the edit

## Supervised Linux pilot mode

The pilot runtime is the first real supervised control loop for running the agent on a Linux box. It does:

- observe -> retrieve -> plan -> optionally ask the main model to choose from approved options
- classify the selected action against a pilot policy
- auto-run only the safe buckets
- stop and ask for approval before file writes, refactors, or non-whitelisted shell commands
- write a JSON trace for every turn under `.agent/pilot_traces`

One-shot usage:

```bash
python3 -m memory_agent.cli pilot "check the workspace status"
python3 -m memory_agent.cli pilot "update config.py with the new setting" --approve
python3 -m memory_agent.cli pilot "what should I do next?" --json
```

Interactive supervised session:

```bash
python3 -m memory_agent.cli pilot-chat
python3 -m memory_agent.cli pilot-chat --no-model
python3 -m memory_agent.cli pilot-chat --policy-file .agent/pilot_policy.toml
```

Supervised multi-step run:

```bash
python3 -m memory_agent.cli pilot-run "stabilize the workspace and clear ready tasks"
python3 -m memory_agent.cli pilot-run "work through the current queue" --steps 3 --approve
python3 -m memory_agent.cli pilot-run "review the latest pilot friction" --promote-limit 2
python3 -m memory_agent.cli pilot-run "what should I do next?" --json
```

Pilot history report:

```bash
python3 -m memory_agent.cli pilot-report
python3 -m memory_agent.cli pilot-report --limit 20 --json
```

Policy template:

```bash
python3 -m memory_agent.cli pilot-policy
python3 -m memory_agent.cli pilot-policy --write-template .agent/pilot_policy.toml
python3 -m memory_agent.cli pilot-policy --policy-file .agent/pilot_policy.toml --json
```

Default pilot behavior:

- auto-approves `ask_user`, `noop`, and `run_maintenance`
- auto-approves non-tool task state changes
- auto-approves `read_text`
- auto-approves read/test shell prefixes such as `git status`, `git diff`, `pytest`, and `ruff check`
- risky file writes and refactors now produce a temp-workspace review packet first, including changed files and a diff preview, and only apply after approval
- approval-backed file writes/refactors run through the patch runner before apply, so validations and eval checks can fail the change before it reaches the main workspace
- pilot policy now includes `git_write_mode` with `auto`, `branch`, or `off` so supervised writes can prefer disposable-branch apply on real repos
- still requires approval for shell commands outside the allowlist
- `pilot-chat` supports `:help`, `:policy`, `:model`, `:status`, `:last`, and `:approve on|off`
- `pilot-run` stops on approval gates, blocked/user-input situations, errors, repeated actions, or the configured step budget and prints a review summary at the end
- `pilot-run` automatically generates a post-run debrief from approvals, stop reasons, and blocked/error steps; use `--promote-limit` to turn the top findings into self-improvement tasks
- pilot reviews now mine recent `pilot-review` outcomes too, so repeated approval friction or stop reasons show up as recurring cross-run patterns instead of isolated incidents
- `pilot-report` lets you inspect those cross-run trends directly without waiting for the next supervised run
- the planner now reads recent pilot-review history too, so repeated approval friction nudges it toward safer ready work when there is a reasonable alternative
- when approval friction keeps recurring around the same kind of risky work, the planner can now propose a preparation step that blocks the risky task behind a smaller safer task first

## Optional local semantic reranking

If you have Ollama running locally, you can add embedding-based reranking on top of the existing memory graph:

```powershell
$env:MEMORY_AGENT_EMBED_MODEL = "nomic-embed-text"
$env:MEMORY_AGENT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
python -m memory_agent.cli search "cloud runtime"
```

This is optional. If the model is not configured or Ollama is unavailable, the system falls back to the normal graph-and-FTS retrieval path.

## Optional main model

If you want one main reasoning model, you can point the agent at a local Ollama chat model:

```bash
export MEMORY_AGENT_CHAT_MODEL="qwen3:14b"
export MEMORY_AGENT_OLLAMA_BASE_URL="http://127.0.0.1:11434"
python3 -m memory_agent.cli model-status
python3 -m memory_agent.cli reply "What should I do next?"
```

The model sees:

- the current user message
- stable profiles
- relevant memory and evidence
- current planner state

It does not get direct authority to invent or run commands. It can only:

- reply in text
- ask a short clarification question
- choose one planner-approved action option through the structured contract

Execution still stays behind the planner/executor guardrails, including shell allowlists and workspace-bound file policies.

If you want the model to choose from planner-approved actions:

```bash
export MEMORY_AGENT_CHAT_MODEL="qwen3:14b"
export MEMORY_AGENT_OLLAMA_BASE_URL="http://127.0.0.1:11434"
python3 -m memory_agent.cli decide "What should I do next?"
python3 -m memory_agent.cli decide "What should I do next?" --preview
```

## Near-term roadmap

- add model-backed extraction and reflection while keeping the heuristic path as the floor
- add richer code-aware editing primitives beyond function/class replacement, such as insertion and rename-aware updates
- add richer task orchestration such as delegation, batching, retry policies, and external tool adapters
