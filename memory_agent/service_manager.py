from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path
from typing import Any


class CockpitServiceManager:
    def __init__(
        self,
        *,
        config_dir: Path | None = None,
        local_unit: str = "ernie-cockpit.service",
        remote_unit: str = "ernie-cockpit-remote.service",
    ) -> None:
        self.config_dir = config_dir or (Path.home() / ".config" / "ernie-cockpit")
        self.remote_env_path = self.config_dir / "remote.env"
        self.local_unit = local_unit
        self.remote_unit = remote_unit
        self.local_launcher_path = Path.home() / ".local" / "bin" / "ernie-cockpit"
        self.desktop_entry_path = (
            Path.home() / ".local" / "share" / "applications" / "ernie-cockpit.desktop"
        )
        self.icon_path = (
            Path.home()
            / ".local"
            / "share"
            / "icons"
            / "hicolor"
            / "scalable"
            / "apps"
            / "ernie-cockpit.svg"
        )
        self.repo_root = Path(__file__).resolve().parent.parent

    def settings(self) -> dict[str, Any]:
        remote = self._load_remote_config()
        local_service = self._unit_status(self.local_unit)
        remote_service = {
            **self._unit_status(self.remote_unit),
            **remote,
            "config_path": str(self.remote_env_path),
        }
        desktop = {
            "launcher_installed": self.local_launcher_path.exists(),
            "launcher_path": str(self.local_launcher_path),
            "desktop_entry_installed": self.desktop_entry_path.exists(),
            "desktop_entry_path": str(self.desktop_entry_path),
            "icon_installed": self.icon_path.exists(),
            "icon_path": str(self.icon_path),
        }
        action_catalog = self._action_catalog(local_service, remote_service, desktop)
        onboarding = self._onboarding_payload(
            local_service,
            remote_service,
            desktop,
            action_catalog=action_catalog,
        )
        return {
            "local_service": local_service,
            "remote_service": remote_service,
            "desktop": desktop,
            "onboarding": onboarding,
            "actions": action_catalog,
        }

    def rotate_remote_token(self) -> dict[str, Any]:
        remote = self._load_remote_config()
        if not remote.get("configured"):
            raise FileNotFoundError(str(self.remote_env_path))
        new_token = str(uuid.uuid4())
        payload = {
            "PORT": str(remote.get("port") or "8766"),
            "TOKEN": new_token,
            "DISPLAY_HOST": str(remote.get("display_host") or ""),
        }
        self.config_dir.mkdir(parents=True, exist_ok=True)
        lines = [f'{key}="{value}"' for key, value in payload.items()]
        self.remote_env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._restart_unit(self.remote_unit)
        refreshed = self._load_remote_config()
        return {
            **self._unit_status(self.remote_unit),
            **refreshed,
            "rotated": True,
        }

    def perform_action(self, action: str) -> dict[str, Any]:
        normalized = action.strip().lower()
        meta = self._action_details(normalized)
        if normalized == "install_desktop_launcher":
            result = self._run_script("scripts/install_desktop_launcher.sh")
        elif normalized == "install_local_service":
            result = self._run_script("scripts/install_user_service.sh")
        elif normalized == "install_remote_service":
            result = self._run_script("scripts/install_remote_service.sh")
        elif normalized == "restart_local_service":
            result = self._restart_unit_result(self.local_unit)
        elif normalized == "restart_remote_service":
            result = self._restart_unit_result(self.remote_unit)
        else:
            raise ValueError(f"unsupported_service_action:{action}")
        refreshed = self.settings()
        inspection_payload = self._inspection_payload(
            normalized,
            settings=refreshed,
            meta=meta,
        )
        return {
            "action": normalized,
            "meta": meta,
            "result": result,
            "message": str(meta.get("success_message") or "Action completed."),
            "settings": refreshed,
            "inspection": inspection_payload["inspection"],
            "verification_target": inspection_payload["verification_target"],
        }

    def inspect_action(self, action: str) -> dict[str, Any]:
        normalized = action.strip().lower()
        meta = self._action_details(normalized)
        settings = self.settings()
        inspection_payload = self._inspection_payload(
            normalized,
            settings=settings,
            meta=meta,
        )
        return {
            "action": normalized,
            "meta": meta,
            "inspection": inspection_payload["inspection"],
            "verification_target": inspection_payload["verification_target"],
            "settings": settings,
        }

    def _inspection_payload(
        self,
        action: str,
        *,
        settings: dict[str, Any],
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = action.strip().lower()
        if normalized in {"install_local_service", "restart_local_service"}:
            inspection = settings["local_service"]
            verification_target = (
                settings["onboarding"].get("local_url") or "http://127.0.0.1:8765/"
            )
        elif normalized in {"install_remote_service", "restart_remote_service"}:
            inspection = settings["remote_service"]
            verification_target = (
                settings["onboarding"].get("remote_url")
                or inspection.get("url")
                or inspection.get("config_path")
                or "remote service status"
            )
        elif normalized == "install_desktop_launcher":
            inspection = settings["desktop"]
            verification_target = (
                settings["onboarding"].get("local_url")
                or inspection.get("desktop_entry_path")
            )
        else:
            raise ValueError(f"unsupported_service_action:{action}")
        return {
            "meta": meta,
            "inspection": inspection,
            "verification_target": verification_target,
        }

    def _load_remote_config(self) -> dict[str, Any]:
        if not self.remote_env_path.exists():
            return {
                "configured": False,
                "port": None,
                "token": "",
                "display_host": "",
                "url": "",
            }
        values: dict[str, str] = {}
        for raw_line in self.remote_env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
        port = values.get("PORT", "").strip() or None
        token = values.get("TOKEN", "").strip()
        display_host = values.get("DISPLAY_HOST", "").strip()
        url = f"http://{display_host}:{port}/" if display_host and port else ""
        return {
            "configured": True,
            "port": port,
            "token": token,
            "display_host": display_host,
            "url": url,
        }

    def _unit_status(self, unit_name: str) -> dict[str, Any]:
        completed = subprocess.run(
            ["systemctl", "--user", "is-active", unit_name],
            capture_output=True,
            text=True,
            check=False,
            env=os.environ.copy(),
        )
        active = completed.returncode == 0 and completed.stdout.strip() == "active"
        return {
            "unit_name": unit_name,
            "active": active,
            "status": completed.stdout.strip() or completed.stderr.strip() or "unknown",
        }

    def _onboarding_payload(
        self,
        local_service: dict[str, Any],
        remote_service: dict[str, Any],
        desktop: dict[str, Any],
        *,
        action_catalog: list[dict[str, Any]],
    ) -> dict[str, Any]:
        steps: list[str] = []
        suggested_actions: list[str] = []
        if not bool(local_service.get("active")):
            steps.append("Start or repair the local cockpit service for this machine.")
            suggested_actions.append("install_local_service")
        if not bool(desktop.get("desktop_entry_installed")):
            steps.append("Install the desktop launcher so Ernie appears in the app menu.")
            suggested_actions.append("install_desktop_launcher")
        if not bool(remote_service.get("configured")):
            steps.append("Install the managed remote service if you want Tailscale or other remote access.")
            suggested_actions.append("install_remote_service")
        elif not bool(remote_service.get("active")):
            steps.append("Restart the managed remote service so remote access is available again.")
            suggested_actions.append("restart_remote_service")
        if not steps:
            steps.append("Use the cockpit normally. This machine already has the main local and remote access paths in place.")
            if not bool(local_service.get("active")):
                suggested_actions.append("restart_local_service")
        selected_actions = [item for item in action_catalog if item.get("action") in suggested_actions]
        return {
            "local_url": "http://127.0.0.1:8765/",
            "remote_url": str(remote_service.get("url") or ""),
            "desktop_ready": bool(desktop.get("desktop_entry_installed")),
            "local_service_ready": bool(local_service.get("active")),
            "remote_service_ready": bool(remote_service.get("active")),
            "recommended_steps": steps,
            "actions": selected_actions,
        }

    def _restart_unit(self, unit_name: str) -> None:
        completed = subprocess.run(
            ["systemctl", "--user", "restart", unit_name],
            capture_output=True,
            text=True,
            check=False,
            env=os.environ.copy(),
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "restart_failed"
            raise RuntimeError(detail)

    def _restart_unit_result(self, unit_name: str) -> dict[str, Any]:
        completed = subprocess.run(
            ["systemctl", "--user", "restart", unit_name],
            capture_output=True,
            text=True,
            check=False,
            env=os.environ.copy(),
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "restart_failed"
            raise RuntimeError(detail)
        status = self._unit_status(unit_name)
        return {
            "ok": True,
            "kind": "systemctl",
            "unit_name": unit_name,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "status": status,
        }

    def _run_script(self, relative_path: str) -> dict[str, Any]:
        script_path = self.repo_root / relative_path
        completed = subprocess.run(
            [str(script_path)],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(self.repo_root),
            env=os.environ.copy(),
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "script_failed"
            raise RuntimeError(detail)
        return {
            "ok": True,
            "kind": "script",
            "script_path": str(script_path),
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }

    def _action_catalog(
        self,
        local_service: dict[str, Any],
        remote_service: dict[str, Any],
        desktop: dict[str, Any],
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []

        def add(
            action: str,
            label: str,
            description: str,
            *,
            enabled: bool = True,
            requires_confirmation: bool = False,
            confirmation_message: str = "",
            success_message: str = "",
        ) -> None:
            actions.append(
                {
                    "action": action,
                    "label": label,
                    "description": description,
                    "enabled": enabled,
                    "requires_confirmation": requires_confirmation,
                    "confirmation_message": confirmation_message,
                    "success_message": success_message,
                }
            )

        add(
            "install_desktop_launcher",
            "Install desktop launcher",
            "Install the Ernie app-menu entry, launcher command, and icon.",
            enabled=not bool(desktop.get("desktop_entry_installed")),
            success_message="Desktop launcher installed. Ernie should now appear in the app menu.",
        )
        add(
            "install_local_service",
            "Install local service",
            "Install or repair the local cockpit background service on this machine.",
            enabled=not bool(local_service.get("active")),
            success_message="Local cockpit service installed or repaired.",
        )
        add(
            "install_remote_service",
            "Install remote service",
            "Install or repair the managed remote cockpit service for Tailscale or other remote access.",
            enabled=not bool(remote_service.get("configured")) or not bool(remote_service.get("active")),
            requires_confirmation=True,
            confirmation_message="Install or repair remote access for this machine? This keeps the cockpit reachable from your remote network until you stop the remote service or rotate the token.",
            success_message="Managed remote service installed or repaired. Check the remote URL and token in Settings.",
        )
        add(
            "restart_local_service",
            "Restart local service",
            "Restart the local cockpit service without reinstalling it.",
            enabled=bool(local_service.get("active")) or self._unit_exists(self.local_unit),
            success_message="Local cockpit service restarted.",
        )
        add(
            "restart_remote_service",
            "Restart remote service",
            "Restart the managed remote service after configuration changes.",
            enabled=bool(remote_service.get("configured")),
            requires_confirmation=True,
            confirmation_message="Restart remote access for this machine? Existing remote browser sessions may need to reconnect afterward.",
            success_message="Managed remote service restarted.",
        )
        return actions

    def _unit_exists(self, unit_name: str) -> bool:
        return (Path.home() / ".config" / "systemd" / "user" / unit_name).exists()

    def _action_details(self, action: str) -> dict[str, Any]:
        fallback = {
            "action": action,
            "label": action,
            "description": "",
            "enabled": True,
            "requires_confirmation": False,
            "confirmation_message": "",
            "success_message": "Action completed.",
        }
        for item in self._action_catalog(
            {"active": False},
            {"configured": False, "active": False},
            {"desktop_entry_installed": False},
        ):
            if item["action"] == action:
                return item
        return fallback
