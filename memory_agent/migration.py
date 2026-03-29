from __future__ import annotations

import json
import shutil
import sqlite3
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import DEFAULT_DB_PATH, DEFAULT_HANDOFFS_DIR, DEFAULT_PILOT_POLICY_PATH, DEFAULT_PILOT_TRACE_DIR
from .models import utc_now_iso


@dataclass(slots=True)
class HandoffBundleReport:
    bundle_path: str
    output_dir: str
    manifest_path: str
    summary_path: str
    included_files: list[str] = field(default_factory=list)
    created_at: str = ""

    def render(self) -> str:
        lines = ["Linux handoff bundle created", f"Bundle: {self.bundle_path}"]
        lines.append(f"Manifest: {self.manifest_path}")
        lines.append(f"Summary: {self.summary_path}")
        if self.included_files:
            lines.append("Included:")
            lines.extend(f"- {item}" for item in self.included_files)
        return "\n".join(lines)


@dataclass(slots=True)
class HandoffRestoreReport:
    bundle_path: str
    target_root: str
    restored_files: list[str] = field(default_factory=list)
    backup_dir: str | None = None
    created_at: str = ""

    def render(self) -> str:
        lines = ["Linux handoff restored", f"Bundle: {self.bundle_path}", f"Target: {self.target_root}"]
        if self.backup_dir:
            lines.append(f"Backups: {self.backup_dir}")
        if self.restored_files:
            lines.append("Restored:")
            lines.extend(f"- {item}" for item in self.restored_files)
        return "\n".join(lines)


class ProjectHandoffManager:
    def __init__(
        self,
        memory_store=None,
        *,
        workspace_root: Path | None = None,
        handoffs_root: Path | None = None,
    ):
        self.memory_store = memory_store
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        configured_root = handoffs_root or DEFAULT_HANDOFFS_DIR
        self.handoffs_root = (
            configured_root
            if Path(configured_root).is_absolute()
            else (self.workspace_root / configured_root)
        ).resolve()
        self.handoffs_root.mkdir(parents=True, exist_ok=True)

    def create_bundle(
        self,
        *,
        output_dir: Path | None = None,
        include_traces: bool = True,
    ) -> HandoffBundleReport:
        if self.memory_store is None:
            raise RuntimeError("create_bundle requires an attached memory store.")
        created_at = utc_now_iso()
        stamp = self._timestamp_token(created_at)
        destination = (
            output_dir.resolve()
            if output_dir is not None and output_dir.is_absolute()
            else (self.workspace_root / output_dir).resolve()
            if output_dir is not None
            else (self.handoffs_root / f"linux_handoff_{stamp}").resolve()
        )
        destination.mkdir(parents=True, exist_ok=True)

        manifest = self._build_manifest(include_traces=include_traces, created_at=created_at)
        manifest_path = destination / "handoff_manifest.json"
        summary_path = destination / "HANDOFF.md"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        summary_path.write_text(self._render_summary(manifest), encoding="utf-8")

        bundle_path = destination / f"linux_handoff_{stamp}.zip"
        included_files = self._write_bundle_archive(
            bundle_path,
            manifest,
            summary_path=summary_path,
            include_traces=include_traces,
            staging_root=destination,
        )
        return HandoffBundleReport(
            bundle_path=str(bundle_path),
            output_dir=str(destination),
            manifest_path=str(manifest_path),
            summary_path=str(summary_path),
            included_files=included_files,
            created_at=created_at,
        )

    def restore_bundle(
        self,
        bundle_path: Path,
        *,
        target_root: Path | None = None,
        force: bool = False,
    ) -> HandoffRestoreReport:
        resolved_bundle = bundle_path.resolve()
        if not resolved_bundle.exists():
            raise FileNotFoundError(f"Missing handoff bundle: {resolved_bundle}")
        destination_root = (target_root or self.workspace_root).resolve()
        if not force and not (destination_root / "pyproject.toml").exists():
            raise FileNotFoundError(
                f"Target root does not look like the project root: {destination_root}"
            )

        created_at = utc_now_iso()
        backup_dir = destination_root / ".agent" / "restore_backups" / self._timestamp_token(created_at)
        restored_files: list[str] = []
        backup_used = False

        with zipfile.ZipFile(resolved_bundle) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                member_path = Path(member.filename)
                if member_path.parts and member_path.parts[0] != ".agent":
                    continue
                target_path = (destination_root / member.filename).resolve()
                try:
                    target_path.relative_to(destination_root)
                except ValueError as exc:
                    raise ValueError(f"Refusing to restore outside target root: {member.filename}") from exc
                if target_path.exists():
                    backup_used = True
                    backup_target = backup_dir / member.filename
                    backup_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target_path, backup_target)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target_path.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
                restored_files.append(member.filename.replace("\\", "/"))

        return HandoffRestoreReport(
            bundle_path=str(resolved_bundle),
            target_root=str(destination_root),
            restored_files=restored_files,
            backup_dir=str(backup_dir) if backup_used else None,
            created_at=created_at,
        )

    def _build_manifest(
        self,
        *,
        include_traces: bool,
        created_at: str,
    ) -> dict[str, Any]:
        context = self.memory_store.build_context("next best action", memory_limit=5)
        latest_patch = self.memory_store.latest_patch_run()
        recent_outcomes = self.memory_store.get_recent_tool_outcomes(limit=5)
        return {
            "created_at": created_at,
            "workspace_root": str(self.workspace_root),
            "python_requirement": ">=3.11",
            "state_files": {
                "db": str(DEFAULT_DB_PATH),
                "pilot_policy": str(DEFAULT_PILOT_POLICY_PATH),
                "pilot_traces": str(DEFAULT_PILOT_TRACE_DIR) if include_traces else None,
            },
            "stats": self.memory_store.stats(),
            "ready_tasks": [
                str(item.metadata.get("title") or item.content)
                for item in context.ready_tasks[:5]
            ],
            "overdue_tasks": [
                str(item.metadata.get("title") or item.content)
                for item in context.overdue_tasks[:5]
            ],
            "open_tasks": [
                str(item.metadata.get("title") or item.content)
                for item in context.open_tasks[:8]
            ],
            "latest_patch_run": latest_patch,
            "recent_tool_outcomes": [
                {
                    "id": item.id,
                    "tool_name": item.metadata.get("tool_name"),
                    "content": item.content,
                    "status": item.metadata.get("status"),
                    "created_at": item.created_at,
                }
                for item in recent_outcomes
            ],
        }

    def _render_summary(self, manifest: dict[str, Any]) -> str:
        ready_tasks = manifest.get("ready_tasks", [])
        overdue_tasks = manifest.get("overdue_tasks", [])
        open_tasks = manifest.get("open_tasks", [])
        latest_patch = manifest.get("latest_patch_run") or {}
        lines = [
            "# Linux Handoff Summary",
            "",
            f"- Created at: {manifest.get('created_at')}",
            f"- Workspace root at export: {manifest.get('workspace_root')}",
            f"- Python requirement: {manifest.get('python_requirement')}",
            "",
            "## Current focus",
        ]
        if ready_tasks:
            lines.extend(f"- Ready: {item}" for item in ready_tasks)
        else:
            lines.append("- Ready: none")
        if overdue_tasks:
            lines.extend(f"- Overdue: {item}" for item in overdue_tasks)
        else:
            lines.append("- Overdue: none")
        if open_tasks:
            lines.extend(f"- Open loop: {item}" for item in open_tasks[:5])
        else:
            lines.append("- Open loop: none")
        lines.extend(["", "## Recent patch state"])
        if latest_patch:
            lines.append(
                f"- Latest patch run: {latest_patch.get('run_name')} [{latest_patch.get('status')}]"
            )
            git_payload = dict((latest_patch.get("summary") or {}).get("git") or {})
            if git_payload.get("branch_name"):
                lines.append(f"- Disposable branch: {git_payload.get('branch_name')}")
            if git_payload.get("rollback_hint"):
                lines.append(f"- Rollback hint: {git_payload.get('rollback_hint')}")
        else:
            lines.append("- No patch runs recorded yet.")
        lines.extend(
            [
                "",
                "## Linux bring-up",
                "1. Clone the repo on Linux.",
                "2. Run scripts/bootstrap_linux.sh in the repo root.",
                "3. Restore this bundle with `python3 -m memory_agent.cli handoff-restore <bundle.zip>`.",
                "4. Start with `python3 -m memory_agent.cli pilot-chat --no-model` or `pilot-run` for supervised testing.",
            ]
        )
        return "\n".join(lines) + "\n"

    def _write_bundle_archive(
        self,
        bundle_path: Path,
        manifest: dict[str, Any],
        *,
        summary_path: Path,
        include_traces: bool,
        staging_root: Path,
    ) -> list[str]:
        included_files: list[str] = []
        db_path = (self.workspace_root / DEFAULT_DB_PATH).resolve()
        policy_path = (self.workspace_root / DEFAULT_PILOT_POLICY_PATH).resolve()
        traces_path = (self.workspace_root / DEFAULT_PILOT_TRACE_DIR).resolve()
        staged_db = staging_root / "_handoff_db.sqlite3"
        try:
            if db_path.exists():
                self._backup_database(staged_db)
                included_files.append(str(DEFAULT_DB_PATH).replace("\\", "/"))
            with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                if db_path.exists():
                    archive.write(staged_db, Path(DEFAULT_DB_PATH).as_posix())
                if policy_path.exists():
                    archive.write(policy_path, Path(DEFAULT_PILOT_POLICY_PATH).as_posix())
                    included_files.append(str(DEFAULT_PILOT_POLICY_PATH).replace("\\", "/"))
                if include_traces and traces_path.exists():
                    for file_path in sorted(traces_path.rglob("*")):
                        if not file_path.is_file():
                            continue
                        archive.write(
                            file_path,
                            file_path.relative_to(self.workspace_root).as_posix(),
                        )
                        included_files.append(
                            file_path.relative_to(self.workspace_root).as_posix()
                        )
                archive.writestr(
                    "_handoff/handoff_manifest.json",
                    json.dumps(manifest, indent=2, sort_keys=True),
                )
                archive.write(summary_path, "_handoff/HANDOFF.md")
        finally:
            if staged_db.exists():
                staged_db.unlink()

        return included_files

    def _backup_database(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        backup_connection = sqlite3.connect(destination)
        try:
            self.memory_store.connection.backup(backup_connection)
        finally:
            backup_connection.close()

    def _timestamp_token(self, timestamp: str) -> str:
        return (
            timestamp.replace(":", "")
            .replace("-", "")
            .replace("+", "_")
            .replace(".", "_")
        )
