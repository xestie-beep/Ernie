# GitHub And Linux Setup

This repo is designed to keep live agent state out of git.

## What goes to GitHub

- project code
- tests
- scripts
- docs

## What stays local

- `.agent/agent_memory.sqlite3`
- `.agent/pilot_traces/`
- any other live `.agent/` runtime data

Use the handoff bundle to move live state safely:

```powershell
python -m memory_agent.cli handoff-pack
```

## Safe path from Windows to Linux

1. Create a private empty GitHub repo in the browser.
2. Push this repo to GitHub.
3. Clone it on the Linux machine.
4. Run:

```bash
./scripts/bootstrap_linux.sh
python3 -m memory_agent.cli handoff-restore /path/to/linux_handoff.zip
./scripts/run_cockpit.sh start local
./scripts/install_desktop_launcher.sh
./scripts/install_user_service.sh
./scripts/install_remote_service.sh
./scripts/manage_remote_access.sh show
```

## Notes

- The repo can be public later if you want, but private is the safer starting point.
- The handoff bundle is the important piece for continuity.
- If something asks for the memory database directly, do not manually drag random files around. Use the bundle.
