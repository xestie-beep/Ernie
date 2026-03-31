from __future__ import annotations

import json
import secrets
import uuid
from dataclasses import asdict, is_dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .agent import MemoryFirstAgent
from .executor import ExecutionCycle, MemoryExecutor
from .improvement import PilotHistoryReporter
from .linux_runtime import LinuxPilotPolicy, LinuxPilotRuntime, PilotTurnReport
from .memory import MemoryStore
from .patch_runner import WorkspacePatchRunner
from .planner import MemoryPlanner, PlannerSnapshot
from .service_manager import CockpitServiceManager
from .soul import SoulProposal, apply_soul_proposal, load_soul_document, review_soul_document


COCKPIT_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ernie Cockpit</title>
  <style>
    :root {
      --bg: #070811;
      --bg-2: #101126;
      --panel: rgba(10, 14, 28, 0.78);
      --panel-2: rgba(17, 22, 44, 0.88);
      --ink: #f8f5ff;
      --muted: #9ea4c7;
      --accent: #35f6d5;
      --accent-2: #ff4fa3;
      --accent-3: #7a6dff;
      --warning: #ffb347;
      --line: rgba(123, 137, 255, 0.24);
      --shadow: rgba(0, 0, 0, 0.45);
      --glow: rgba(53, 246, 213, 0.3);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Trebuchet MS", "Avenir Next", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(53, 246, 213, 0.14), transparent 24%),
        radial-gradient(circle at top right, rgba(255, 79, 163, 0.18), transparent 26%),
        radial-gradient(circle at bottom center, rgba(122, 109, 255, 0.18), transparent 34%),
        linear-gradient(180deg, var(--bg) 0%, var(--bg-2) 100%);
      min-height: 100vh;
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
      background-size: 28px 28px;
      opacity: 0.25;
    }
    .shell {
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px;
    }
    .hidden {
      display: none !important;
    }
    .hero {
      display: grid;
      gap: 16px;
      grid-template-columns: 1.5fr 1fr;
      align-items: end;
      margin-bottom: 24px;
    }
    .title {
      margin: 0;
      font-size: clamp(2.4rem, 5vw, 4.6rem);
      line-height: 0.92;
      letter-spacing: -0.04em;
      text-transform: uppercase;
      text-shadow:
        0 0 24px rgba(53, 246, 213, 0.24),
        0 0 44px rgba(255, 79, 163, 0.18);
    }
    .subtitle {
      margin: 12px 0 0;
      max-width: 42rem;
      color: var(--muted);
      font-size: 1.05rem;
      line-height: 1.5;
    }
    .status-banner {
      background: linear-gradient(135deg, rgba(16, 24, 52, 0.96), rgba(19, 11, 39, 0.94));
      color: white;
      border-radius: 18px;
      padding: 18px 20px;
      box-shadow: 0 18px 48px var(--shadow);
      border: 1px solid rgba(53, 246, 213, 0.22);
    }
    .status-banner h2 {
      margin: 0 0 8px;
      font-size: 1rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--accent);
    }
    .status-banner p {
      margin: 0;
      line-height: 1.45;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 18px;
    }
    .panel {
      background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 18px;
      box-shadow: 0 16px 40px var(--shadow);
      backdrop-filter: blur(6px);
    }
    .panel h3 {
      margin: 0 0 10px;
      font-size: 0.95rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--accent);
    }
    .panel pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 0.92rem;
      line-height: 1.5;
      color: #d9defb;
    }
    .metrics {
      grid-column: span 4;
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .metric {
      padding: 14px;
      border-radius: 18px;
      background: rgba(12, 17, 34, 0.84);
      border: 1px solid rgba(122, 109, 255, 0.24);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
    }
    .metric strong {
      display: block;
      font-size: 1.7rem;
      margin-top: 6px;
      color: var(--ink);
    }
    .metric span {
      color: var(--accent);
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .wide { grid-column: span 8; }
    .half { grid-column: span 6; }
    .full { grid-column: 1 / -1; }
    .cards {
      display: grid;
      gap: 12px;
    }
    .card {
      border-radius: 18px;
      padding: 14px;
      background: rgba(8, 12, 25, 0.72);
      border: 1px solid rgba(255,255,255,0.06);
    }
    .card.active {
      border-color: rgba(53, 246, 213, 0.45);
      box-shadow: 0 0 0 1px rgba(53, 246, 213, 0.12), 0 0 28px rgba(53, 246, 213, 0.08);
    }
    .card .eyebrow {
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.09em;
      font-size: 0.76rem;
      margin-bottom: 6px;
    }
    .card .headline {
      font-size: 1.08rem;
      font-weight: 700;
      margin-bottom: 6px;
    }
    .card .body {
      color: #d4d8f3;
      line-height: 1.5;
      font-size: 0.95rem;
    }
    .controls {
      display: grid;
      gap: 12px;
    }
    label {
      display: block;
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 6px;
    }
    input, textarea, select, button {
      width: 100%;
      border-radius: 14px;
      border: 1px solid rgba(122, 109, 255, 0.28);
      padding: 12px 14px;
      font: inherit;
      color: var(--ink);
      background: rgba(10, 16, 30, 0.9);
    }
    textarea {
      min-height: 108px;
      resize: vertical;
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .actions.tight button {
      min-width: 0;
      padding: 10px 12px;
    }
    button {
      width: auto;
      min-width: 180px;
      cursor: pointer;
      background: linear-gradient(135deg, var(--accent), #0ea5e9);
      color: #051014;
      border: 1px solid rgba(255,255,255,0.06);
      font-weight: 600;
      box-shadow: 0 0 0 1px rgba(53, 246, 213, 0.08), 0 0 26px rgba(53, 246, 213, 0.16);
    }
    button.secondary {
      background: linear-gradient(135deg, var(--accent-2), #ff8e5b);
      color: white;
    }
    button.ghost {
      background: rgba(10, 16, 30, 0.86);
      color: var(--ink);
      border: 1px solid rgba(255,255,255,0.08);
      box-shadow: none;
    }
    ul {
      margin: 0;
      padding-left: 18px;
      line-height: 1.5;
    }
    li + li { margin-top: 8px; }
    .muted { color: var(--muted); }
    .status-line {
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      font-size: 0.95rem;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(53, 246, 213, 0.12);
      color: var(--accent);
      border: 1px solid rgba(53, 246, 213, 0.2);
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .pill.warn {
      background: rgba(255, 179, 71, 0.12);
      color: var(--warning);
      border-color: rgba(255, 179, 71, 0.24);
    }
    .pill.danger {
      background: rgba(255, 79, 163, 0.12);
      color: var(--accent-2);
      border-color: rgba(255, 79, 163, 0.24);
    }
    .pill.dim {
      background: rgba(158, 164, 199, 0.12);
      color: var(--muted);
      border-color: rgba(158, 164, 199, 0.18);
    }
    .warning {
      color: var(--warning);
    }
    .notice {
      border-radius: 18px;
      padding: 14px 16px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(8, 12, 25, 0.72);
    }
    .notice.success {
      border-color: rgba(53, 246, 213, 0.34);
      box-shadow: 0 0 0 1px rgba(53, 246, 213, 0.08), 0 0 24px rgba(53, 246, 213, 0.08);
    }
    .notice.error {
      border-color: rgba(255, 79, 163, 0.34);
      box-shadow: 0 0 0 1px rgba(255, 79, 163, 0.08), 0 0 24px rgba(255, 79, 163, 0.08);
    }
    .notice.info {
      border-color: rgba(122, 109, 255, 0.28);
    }
    .inline-auth {
      display: grid;
      gap: 10px;
      grid-template-columns: minmax(0, 1fr) auto;
      margin-top: 12px;
    }
    .tiny {
      font-size: 0.82rem;
      line-height: 1.45;
    }
    .mono {
      font-family: "SFMono-Regular", Consolas, monospace;
      white-space: pre-wrap;
      word-break: break-word;
      color: #d9defb;
      font-size: 0.9rem;
      line-height: 1.45;
    }
    @media (max-width: 980px) {
      .hero { grid-template-columns: 1fr; }
      .metrics, .wide, .half { grid-column: 1 / -1; }
      .inline-auth { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div>
        <h1 class="title">Ernie<br>cockpit</h1>
        <p class="subtitle">A local control room for memory, planning, and safe execution. This is a first usable interface over the existing agent core, not a separate agent.</p>
      </div>
      <div class="status-banner">
        <h2>Service</h2>
        <p id="service-summary">Loading current state...</p>
      </div>
    </section>

    <section id="login-panel" class="panel full hidden" style="margin-bottom:18px;">
      <h3>Remote login</h3>
      <div class="cards">
        <div class="card">
          <div class="headline">This cockpit is protected.</div>
          <div id="login-message" class="body">Paste the shared access token to open a browser session on this machine.</div>
        </div>
      </div>
      <div class="inline-auth">
        <input id="login-token" placeholder="Paste cockpit access token">
        <button id="login-button">Open cockpit session</button>
      </div>
      <div id="login-note" class="tiny muted" style="margin-top:10px;">No remote session is active yet.</div>
    </section>

    <section id="app-grid" class="grid">
      <div class="panel full">
        <h3>Action status</h3>
        <div id="action-banner" class="notice info">
          <div class="headline">Waiting for the next move.</div>
          <div class="body">Run an action and Ernie will explain what changed here.</div>
        </div>
      </div>

      <div class="panel full">
        <h3>How Ernie Operates</h3>
        <div id="soul-panel" class="cards"></div>
      </div>

      <div class="panel full">
        <h3>First Run Tutorial</h3>
        <div id="tutorial-panel" class="cards"></div>
      </div>

      <div class="panel full">
        <h3>Machine status</h3>
        <div id="onboarding-panel" class="cards"></div>
      </div>

      <div class="panel full">
        <h3>Work queue</h3>
        <div id="work-queue" class="cards"></div>
      </div>

      <div class="panel full">
        <h3>Focused task stack</h3>
        <div id="focused-stack" class="cards"></div>
      </div>

      <div class="panel metrics" id="metrics"></div>

      <div class="panel wide">
        <h3>Next action</h3>
        <div id="plan-card" class="cards"></div>
      </div>

      <div class="panel half">
        <h3>Ready tasks</h3>
        <div id="ready-tasks" class="cards"></div>
      </div>

      <div class="panel half">
        <h3>Overdue tasks</h3>
        <div id="overdue-tasks" class="cards"></div>
      </div>

      <div class="panel half">
        <h3>Context</h3>
        <div class="controls">
          <div>
            <label for="query">Focus query</label>
            <input id="query" value="what should I do next">
          </div>
          <div class="actions">
            <button id="refresh">Refresh view</button>
            <button id="execute" class="secondary">Execute next action</button>
            <button id="review" class="ghost">Review action first</button>
          </div>
        </div>
        <div id="context-output" class="cards"></div>
      </div>

      <div class="panel half">
        <h3>Capture note</h3>
        <div class="controls">
          <div>
            <label for="role">Role</label>
            <select id="role">
              <option value="user">user</option>
              <option value="assistant">assistant</option>
              <option value="tool">tool</option>
              <option value="system">system</option>
            </select>
          </div>
          <div>
            <label for="note">Text</label>
            <textarea id="note" placeholder="Capture a fact, decision, task, or blocker."></textarea>
          </div>
          <div class="actions">
            <button id="observe">Store note</button>
          </div>
        </div>
        <pre id="observe-output" class="muted">No note captured yet.</pre>
      </div>

      <div class="panel half">
        <h3>Prompt Workshop</h3>
        <div class="controls">
          <div>
            <label for="prompt-draft">Draft prompt</label>
            <textarea id="prompt-draft" placeholder="Write a draft prompt here. This workshop text stays quarantined until you explicitly promote it."></textarea>
          </div>
          <div class="actions">
            <button class="ghost" id="prompt-improve">Improve</button>
            <button class="ghost" id="prompt-clarify">Clarify</button>
            <button class="ghost" id="prompt-shorten">Shorten</button>
            <button class="ghost" id="prompt-safety-check">Safety check</button>
            <button class="secondary" id="prompt-undo-history">Undo local edit</button>
            <button class="secondary" id="prompt-replace-selection">Replace selection</button>
            <button class="secondary" id="prompt-append-result">Append result</button>
            <button class="secondary" id="prompt-adopt-result">Use result as draft</button>
            <button id="prompt-pilot-result">Send result to guided loop</button>
            <button id="prompt-pilot">Send draft to guided loop</button>
            <button id="prompt-promote-result">Promote result to live note</button>
            <button id="prompt-promote">Promote draft to live note</button>
          </div>
        </div>
        <div id="prompt-workshop-summary" class="cards"></div>
        <div id="prompt-workshop-compare" class="cards"></div>
        <div id="prompt-workshop-history" class="cards"></div>
        <pre id="prompt-workshop-output" class="muted">No draft workshop run yet.</pre>
      </div>

      <div class="panel half">
        <h3>Recent nudges</h3>
        <div id="recent-nudges" class="cards"></div>
      </div>

      <div class="panel half">
        <h3>Open tasks</h3>
        <div id="open-tasks" class="cards"></div>
      </div>

      <div class="panel half">
        <h3>Task detail</h3>
        <div id="task-detail" class="cards"></div>
      </div>

      <div class="panel full">
        <h3>Pilot approval queue</h3>
        <div class="muted tiny" style="margin-bottom:12px;">Manual pilot requests and Prompt Workshop results both land here after preview. Request origin stays visible in the queue and detail view.</div>
        <div class="controls">
          <div>
            <label for="pilot-text">Pilot request</label>
            <textarea id="pilot-text" placeholder="Describe the supervised action you want the pilot to evaluate."></textarea>
          </div>
          <div class="actions">
            <button id="pilot-preview">Preview pilot action</button>
          </div>
        </div>
        <div id="pilot-pending" class="cards" style="margin-top:14px;"></div>
      </div>

      <div class="panel full">
        <h3>Pilot review detail</h3>
        <div id="pilot-detail" class="cards"></div>
      </div>

      <div class="panel full">
        <h3>Patch runs and rollback</h3>
        <div id="patch-runs" class="cards"></div>
      </div>

      <div class="panel full">
        <h3>Patch review detail</h3>
        <div id="patch-detail" class="cards"></div>
      </div>

      <div class="panel full">
        <h3>Operator timeline</h3>
        <div id="recent-activity" class="cards"></div>
      </div>

      <div class="panel full">
        <h3>Timeline detail</h3>
        <div id="activity-detail" class="cards"></div>
      </div>

      <div class="panel full">
        <h3>Execution result</h3>
        <pre id="execution-output" class="muted">No execution run yet.</pre>
      </div>

      <div class="panel full">
        <h3>Service health</h3>
        <div class="status-line">
          <span class="pill" id="health-pill">Checking</span>
          <span id="health-summary">Checking cockpit health...</span>
        </div>
        <div class="inline-auth">
          <input id="auth-token" placeholder="Paste access token for this cockpit">
          <button id="save-session" class="ghost">Save remote session</button>
        </div>
        <div id="auth-note" class="tiny muted" style="margin-top:10px;">No remote session stored yet.</div>
      </div>

      <div class="panel full">
        <h3>Settings</h3>
        <div id="settings-panel" class="cards"></div>
      </div>
    </section>
  </div>

  <script>
    let authToken = new URLSearchParams(window.location.search).get("token") || "";
    let sessionId = window.localStorage.getItem("ernieCockpitSession") || "";
    let currentPilotItems = [];
    let currentPatchRuns = [];
    let currentActivityItems = [];
    let currentSoulReview = null;
    let currentPlanExplanation = null;
    let currentPromptWorkshop = null;
    let currentPromptHistory = [];
    let selectedPilotPendingId = "";
    let selectedPatchRunId = 0;
    let selectedActivityId = 0;
    let pinnedWorkQueueKey = window.localStorage.getItem("ernieCockpitPinnedWorkQueue") || "";
    let dismissedWorkQueueKeys = JSON.parse(window.localStorage.getItem("ernieCockpitDismissedWorkQueue") || "[]");
    let focusedTaskTitle = window.localStorage.getItem("ernieCockpitFocusedTask") || "";

    function setActionBanner(kind, title, body) {
      const node = document.getElementById("action-banner");
      node.className = "notice " + (kind || "info");
      node.innerHTML = `
        <div class="headline">${title || "Status update"}</div>
        <div class="body">${body || ""}</div>
      `;
    }

    async function requestJson(url, options) {
      const init = options ? {...options} : {};
      const headers = new Headers(init.headers || {});
      if (sessionId) {
        headers.set("X-Ernie-Session", sessionId);
      }
      if (authToken) {
        headers.set("X-Ernie-Token", authToken);
      }
      init.headers = headers;
      const response = await fetch(url, init);
      const payload = await response.json();
      if (response.status === 401) {
        sessionId = "";
        window.localStorage.removeItem("ernieCockpitSession");
        if (!authToken) {
          setLoginRequired(true, "Your browser session expired. Paste the cockpit token to reconnect.");
        }
      }
      if (!response.ok) {
        setActionBanner("error", "Action failed.", payload.error || ("Request failed: " + response.status));
        throw new Error(payload.error || ("Request failed: " + response.status));
      }
      return payload;
    }

    function currentAuthToken() {
      return authToken || window.localStorage.getItem("ernieCockpitToken") || "";
    }

    function setAuthNote(text) {
      document.getElementById("auth-note").textContent = text;
    }

    function setLoginNote(text) {
      document.getElementById("login-note").textContent = text;
    }

    function setLoginRequired(required, message) {
      const loginPanel = document.getElementById("login-panel");
      const appGrid = document.getElementById("app-grid");
      if (required) {
        loginPanel.classList.remove("hidden");
        appGrid.classList.add("hidden");
        if (message) {
          document.getElementById("login-message").textContent = message;
        }
      } else {
        loginPanel.classList.add("hidden");
        appGrid.classList.remove("hidden");
      }
    }

    async function saveSession(token) {
      const normalized = (token || "").trim();
      if (!normalized) {
        throw new Error("Missing access token.");
      }
      authToken = normalized;
      window.localStorage.setItem("ernieCockpitToken", normalized);
      const payload = await requestJson("/api/session", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({token: normalized})
      });
      sessionId = payload.session_id || "";
      if (!sessionId) {
        throw new Error("Session creation failed.");
      }
      window.localStorage.setItem("ernieCockpitSession", sessionId);
      document.getElementById("auth-token").value = "";
      document.getElementById("login-token").value = "";
      const next = new URL(window.location.href);
      if (next.searchParams.get("token")) {
        next.searchParams.delete("token");
        window.history.replaceState({}, "", next.toString());
      }
      setAuthNote("Remote session saved in this browser. URL token cleared.");
      setLoginNote("Remote session saved in this browser.");
      setLoginRequired(false);
      setActionBanner("success", "Remote session saved.", "This browser can now reopen the cockpit without keeping the token in the URL.");
      return payload;
    }

    function metricCard(label, value) {
      return `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;
    }

    function escapeHtml(value) {
      return String(value || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    function card(title, body) {
      return `
        <div class="card">
          <div class="headline">${title}</div>
          <div class="body">${body}</div>
        </div>
      `;
    }

    function pillClassForState(state) {
      const normalized = String(state || "").toLowerCase();
      if (["error", "failed", "rejected", "blocked", "overdue"].includes(normalized)) return "danger";
      if (["pending", "needs approval", "inactive", "warning"].includes(normalized)) return "warn";
      if (["done", "active", "applied", "approved", "ready", "open", "success"].includes(normalized)) return "";
      return "dim";
    }

    function renderStatePill(label) {
      return `<span class="pill ${pillClassForState(label)}">${label || "unknown"}</span>`;
    }

    function summarizeTaskCard(task) {
      const meta = task.metadata || {};
      const details = [];
      if (meta.blocked_by && meta.blocked_by.length) {
        details.push(`Blocked by ${meta.blocked_by.join(", ")}.`);
      } else if (meta.depends_on && meta.depends_on.length) {
        details.push(`Waiting on ${meta.depends_on.join(", ")}.`);
      } else if (meta.status === "done") {
        details.push("Finished.");
      } else if (meta.status === "open") {
        details.push("Available for work.");
      }
      if (meta.due_date) details.push(`Due ${meta.due_date}.`);
      if (!details.length) details.push(meta.details || task.content || "No extra task detail recorded.");
      return details.join(" ");
    }

    function summarizePilotCard(item) {
      const action = item.selected_action || {};
      const approval = item.approval || {};
      const source = item.request_origin === "prompt_workshop"
        ? `Prompt Workshop ${item.prompt_source === "result" ? "result" : "draft"}`
        : "";
      if (approval.category === "trusted_file_operation") {
        const base = summarizeTrustedApproval(approval) || approval.reason || action.summary || "Trusted pilot action.";
        return source ? `${source}. ${base}` : base;
      }
      const base = approval.prompt || approval.reason || action.summary || "Review this proposed pilot action before it runs.";
      return source ? `${source}. ${base}` : base;
    }

    function summarizeTrustedApproval(approval) {
      const metadata = approval?.metadata || {};
      const operation = metadata.trusted_operation || metadata.file_path ? metadata.trusted_operation || "file edit" : "";
      const path = metadata.trusted_path || metadata.file_path || "";
      const matched = metadata.matched_successes;
      const required = metadata.required_successes;
      const pieces = [];
      if (operation || path) {
        pieces.push(`Trusted ${operation || "file edit"}${path ? ` on ${path}` : ""}.`);
      }
      if (matched && required) {
        pieces.push(`Matched ${matched} successful supervised previews with a threshold of ${required}.`);
      }
      return pieces.join(" ");
    }

    function workshopModeLabel(mode) {
      const normalized = String(mode || "").trim().toLowerCase();
      if (normalized === "clarify") return "Clarify";
      if (normalized === "shorten") return "Shorten";
      if (normalized === "safety_check") return "Safety check";
      return "Improve";
    }

    function rememberPromptHistory(label) {
      const node = promptDraftNode();
      currentPromptHistory.unshift({
        label: label || "Draft state",
        draft: node ? node.value : "",
        workshop: currentPromptWorkshop
          ? {
              mode: currentPromptWorkshop.mode,
              draft: currentPromptWorkshop.draft,
              text: currentPromptWorkshop.text
            }
          : null
      });
      currentPromptHistory = currentPromptHistory.slice(0, 6);
    }

    function renderPromptHistory() {
      const node = document.getElementById("prompt-workshop-history");
      if (!currentPromptHistory.length) {
        node.innerHTML = card("Local prompt history", "No local draft history yet. Workshop runs and draft edits will build a short undo buffer in this browser only.");
        return;
      }
      const items = currentPromptHistory
        .map((entry, index) => {
          const draftPreview = escapeHtml(String(entry.draft || "").slice(0, 140) || "Empty draft.");
          return `
            <div class="card">
              <div class="eyebrow">History ${index + 1}</div>
              <div class="headline">${escapeHtml(entry.label || "Draft state")}</div>
              <div class="body">
                <div class="muted">${draftPreview}${String(entry.draft || "").length > 140 ? "..." : ""}</div>
                <div style="margin-top:10px;"><button class="ghost prompt-history-restore" data-history-index="${index}">Restore this draft</button></div>
              </div>
            </div>
          `;
        })
        .join("");
      node.innerHTML = card("Local prompt history", "This undo buffer lives only in the browser and does not enter live memory.") + items;
      for (const button of node.querySelectorAll(".prompt-history-restore")) {
        button.addEventListener("click", () => restorePromptHistory(Number(button.dataset.historyIndex || "0")));
      }
    }

    function renderPromptWorkshop() {
      const summaryNode = document.getElementById("prompt-workshop-summary");
      const compareNode = document.getElementById("prompt-workshop-compare");
      if (!currentPromptWorkshop) {
        summaryNode.innerHTML = card("Workshop mode", "No workshop run yet. Run one of the draft helpers to compare your current prompt against a quarantined result.");
        compareNode.innerHTML = "";
        renderPromptHistory();
        return;
      }
      const modeLabel = workshopModeLabel(currentPromptWorkshop.mode);
      const modelLine = currentPromptWorkshop.used_model
        ? "Model-assisted draft help."
        : "Fallback draft help.";
      summaryNode.innerHTML =
        card("Workshop mode", `${modeLabel}. ${modelLine} The result is still quarantined until you promote it.`) +
        card(
          "Promotion boundary",
          "Use result as draft keeps the text quarantined. Promote result to live note is the point where it becomes live input."
        ) +
        card(
          "Draft editing helpers",
          "Replace selection swaps only the highlighted part of your draft. Append result keeps your original draft and adds the workshop result below it."
        ) +
        card(
          "Local history buffer",
          "Undo local edit restores recent draft states inside this browser only. It does not touch memory, plans, or live input."
        );
      const draftText = currentPromptWorkshop.draft || "No draft captured.";
      const resultText = currentPromptWorkshop.text || "No workshop result returned.";
      const sameText = draftText.trim() === resultText.trim();
      compareNode.innerHTML =
        card("Draft before workshop", `<pre>${escapeHtml(draftText)}</pre>`) +
        card(
          sameText ? "Workshop result" : "Workshop result after rewrite",
          `<pre>${escapeHtml(resultText)}</pre>${sameText ? '<div class="muted">The workshop preserved the original wording.</div>' : '<div class="muted">Review the rewritten result before adopting or promoting it.</div>'}`
        );
      renderPromptHistory();
    }

    function promptDraftNode() {
      return document.getElementById("prompt-draft");
    }

    function replacePromptDraftSelection() {
      if (!currentPromptWorkshop?.text) return;
      const node = promptDraftNode();
      const start = Number(node.selectionStart || 0);
      const end = Number(node.selectionEnd || 0);
      if (start === end) return;
      rememberPromptHistory("Before replace selection");
      node.setRangeText(currentPromptWorkshop.text, start, end, "end");
      node.focus();
      setActionBanner("info", "Draft selection replaced.", "Only the highlighted part of the draft was replaced. The result is still quarantined until you promote it.");
      renderPromptWorkshop();
    }

    function appendPromptWorkshopResult() {
      if (!currentPromptWorkshop?.text) return;
      const node = promptDraftNode();
      rememberPromptHistory("Before append result");
      const current = node.value.trimEnd();
      node.value = current ? `${current}\n\n${currentPromptWorkshop.text}` : currentPromptWorkshop.text;
      node.focus();
      setActionBanner("info", "Workshop result appended.", "The result was appended below the current draft and remains quarantined until you promote it.");
      renderPromptWorkshop();
    }

    function restorePromptHistory(index) {
      const entry = currentPromptHistory[index];
      if (!entry) return;
      const node = promptDraftNode();
      node.value = entry.draft || "";
      currentPromptWorkshop = entry.workshop
        ? {
            mode: entry.workshop.mode,
            draft: entry.workshop.draft,
            text: entry.workshop.text
          }
        : null;
      currentPromptHistory.splice(index, 1);
      node.focus();
      setActionBanner("info", "Local draft restored.", "A recent quarantined draft state was restored from the browser history buffer.");
      renderPromptWorkshop();
    }

    function undoPromptHistory() {
      restorePromptHistory(0);
    }

    function explainPlannerAction(kind) {
      const normalized = String(kind || "").trim().toLowerCase();
      if (normalized === "prepare_task") {
        return "Prepare splits a risky or confirmation-heavy task into a safer preflight step before the main action runs.";
      }
      if (normalized === "delegate_task") {
        return "Delegate creates a tracked child task so the original work can route through a smaller handoff.";
      }
      if (normalized === "work_task") {
        return "Run starts the task's current bounded execution path, such as a command, service action, inspection, or state transition.";
      }
      if (normalized === "resolve_blocker") {
        return "Resolve blocker tries to clear the active dependency preventing this task from moving.";
      }
      if (normalized === "run_maintenance") {
        return "Maintenance refreshes Ernie's own task and service state, then resurfaces anything still due.";
      }
      if (normalized === "batch_ready_tasks") {
        return "Batch runs several already-safe ready tasks in sequence.";
      }
      return "This planner step is a bounded next move surfaced from the current task and system state.";
    }

    function summarizePatchCard(run) {
      const candidate = (run.summary || {}).candidate_evaluation || {};
      const git = (run.summary || {}).git || {};
      if (candidate.status === "passed") return "Validation passed. This candidate was safe enough to apply.";
      if (candidate.status === "failed") {
        const failures = candidate.validation_failures || candidate.failures || [];
        return failures.length ? `Validation failed: ${failures.join(" | ")}` : "Validation failed.";
      }
      if (git.rollback_ready) return "Applied with rollback metadata available.";
      return run.status ? `Run status: ${run.status}.` : "Patch run recorded without extra summary.";
    }

    function normalizeText(value) {
      return String(value || "").trim().toLowerCase();
    }

    function matchesFocusedTask(...values) {
      const target = normalizeText(focusedTaskTitle);
      if (!target) return false;
      return values.some((value) => normalizeText(value).includes(target));
    }

    function prioritizeFocused(items, extractor) {
      if (!focusedTaskTitle) return items;
      return [...items].sort((a, b) => {
        const aMatch = extractor(a) ? 1 : 0;
        const bMatch = extractor(b) ? 1 : 0;
        return bMatch - aMatch;
      });
    }

    function taskList(node, tasks) {
      node.innerHTML = "";
      if (!tasks.length) {
        node.innerHTML = "<div class='card'><div class='body muted'>None right now.</div></div>";
        return;
      }
      for (const task of tasks) {
        const title = task.metadata.title || task.content;
        const status = task.metadata.status || task.subject;
        const extras = [];
        if (task.metadata.due_date) extras.push("due " + task.metadata.due_date);
        if (task.metadata.blocked_by && task.metadata.blocked_by.length) {
          extras.push("blocked by " + task.metadata.blocked_by.join(", "));
        }
        if (task.metadata.depends_on && task.metadata.depends_on.length) {
          extras.push("depends on " + task.metadata.depends_on.join(", "));
        }
        const div = document.createElement("div");
        div.className = "card";
        div.innerHTML = `
          <div class="eyebrow">${status}</div>
          <div class="headline">${title}</div>
          <div class="body">${extras.length ? extras.join(" · ") : "Ready to act."}</div>
        `;
        node.appendChild(div);
      }
    }

    function renderEmptyDetail(nodeId, title, body) {
      document.getElementById(nodeId).innerHTML = `
        <div class="card">
          <div class="eyebrow">${title}</div>
          <div class="body muted">${body}</div>
        </div>
      `;
    }

    function renderSettings(settings) {
      const node = document.getElementById("settings-panel");
      const localService = settings.local_service || {};
      const remoteService = settings.remote_service || {};
      const desktop = settings.desktop || {};
      const serviceSyncStatus = settings.service_sync_status || {};
      const pilotPolicy = settings.pilot_policy || {};
      const actions = settings.actions || [];
      const remoteUrl = remoteService.url || (remoteService.display_host && remoteService.port
        ? `http://${remoteService.display_host}:${remoteService.port}/`
        : "No browser URL configured.");
      const trustedOps = pilotPolicy.trusted_auto_approve_file_operations || [];
      const trustedThreshold = pilotPolicy.trusted_auto_approve_required_successes || 2;
      const trustedEnabled = Boolean(pilotPolicy.trusted_writes_enabled);
      const policyCard = `
        <div class="card">
          <div class="eyebrow">Pilot trusted writes</div>
          <div class="headline">${trustedEnabled ? "Enabled" : "Disabled"}</div>
          <div class="body">Policy file: ${pilotPolicy.policy_path || pilotPolicy.loaded_from || "workspace default"}</div>
          <div class="body" style="margin-top:8px;">Operations: ${trustedOps.length ? trustedOps.join(", ") : "none"}</div>
          <div class="body" style="margin-top:8px;">Success threshold: ${trustedThreshold}</div>
          <div class="body muted" style="margin-top:10px;">Trusted writes only cover repeated low-risk single-file edits that already succeeded under supervision. They are a shortcut for familiar safe changes, not open-ended write permission.</div>
          <div class="controls" style="margin-top:12px;">
            <div>
              <label for="trusted-write-enabled">Trusted low-risk writes</label>
              <select id="trusted-write-enabled">
                <option value="true"${trustedEnabled ? " selected" : ""}>enabled</option>
                <option value="false"${trustedEnabled ? "" : " selected"}>disabled</option>
              </select>
            </div>
            <div>
              <label for="trusted-write-threshold">Required successful previews</label>
              <input id="trusted-write-threshold" type="number" min="1" max="10" value="${trustedThreshold}">
            </div>
            <div>
              <label for="trusted-write-operations">Trusted operations</label>
              <input id="trusted-write-operations" value="${trustedOps.join(", ")}" placeholder="write_text, replace_text, append_text">
            </div>
            <div class="actions">
              <button id="save-pilot-policy" class="ghost">Save pilot write policy</button>
            </div>
          </div>
        </div>
      `;
      const trustedSignals = pilotPolicy.trusted_write_recommendations || [];
      const suppressedSyncActions = serviceSyncStatus.suppressed_recent_verification_actions || [];
      const suppressedSyncTitles = serviceSyncStatus.suppressed_recent_verification_titles || [];
      const suppressedSyncScopes = serviceSyncStatus.suppressed_recent_verification_scopes || [];
      const suppressionCard = suppressedSyncActions.length ? `
        <div class="card">
          <div class="eyebrow">Recent healthy verification</div>
          <div class="headline">Service maintenance temporarily suppressed</div>
          <div class="body">Ernie just verified a healthy service path in this scope, so the matching setup recommendation is temporarily downgraded instead of resurfacing immediately.</div>
          <div class="body" style="margin-top:8px;">Scopes: ${suppressedSyncScopes.join(", ")}</div>
          <div class="body" style="margin-top:8px;">Suppressed actions: ${suppressedSyncActions.join(", ")}</div>
          <div class="body" style="margin-top:8px;">Suppressed tasks: ${suppressedSyncTitles.length ? suppressedSyncTitles.join(", ") : "No queued setup task titles were left to suppress."}</div>
        </div>
      ` : "";
      const trustedSignalCard = `
        <div class="card">
          <div class="eyebrow">Pilot history trust signals</div>
          <div class="headline">${trustedSignals.length ? `${trustedSignals.length} repeated write pattern(s)` : "No repeated trust signals yet"}</div>
          <div class="body">These come from recent pilot reviews that repeatedly stopped on the same low-risk single-file write.</div>
          <div style="margin-top:12px; display:grid; gap:12px;">
            ${trustedSignals.length ? trustedSignals.map((item) => `
              <div class="card" style="padding:12px; border-radius:16px;">
                <div class="headline" style="font-size:1rem;">${item.label}</div>
                <div class="body" style="margin-top:6px;">Seen ${item.count} times in recent pilot reviews.</div>
                <div class="body" style="margin-top:6px;">Operation: ${item.file_operation || "unknown"} · Path: ${item.file_path || "unknown"}</div>
              </div>
            `).join("") : `<div class="card" style="padding:12px; border-radius:16px;"><div class="body muted">No repeated low-risk write approval pattern has crossed the reporting threshold yet.</div></div>`}
          </div>
        </div>
      `;
      const actionButtons = actions.length ? `
        <div class="card">
          <div class="eyebrow">Guided setup actions</div>
          <div class="body">Run common setup and repair tasks from the cockpit.</div>
          <div style="margin-top:12px; display:grid; gap:12px;">
            ${actions.map((item) => `
              <div class="card" style="padding:12px; border-radius:16px;">
                <div class="headline" style="font-size:1rem;">${item.label}</div>
                <div class="body" style="margin-top:6px;">${item.description || ""}</div>
                <div class="actions" style="margin-top:10px;">
                  <button class="ghost" data-service-action="${item.action}"${item.enabled ? "" : " disabled"}>${item.enabled ? item.label : "Already ready"}</button>
                </div>
              </div>
            `).join("")}
          </div>
        </div>
      ` : "";
      node.innerHTML = `
        <div class="card">
          <div class="eyebrow">Local service</div>
          <div class="headline">${localService.unit_name || "ernie-cockpit.service"}</div>
          <div class="body">Status: ${localService.status || "unknown"}</div>
          <div class="body" style="margin-top:8px;">URL: http://127.0.0.1:8765/</div>
        </div>
        <div class="card">
          <div class="eyebrow">Managed remote service</div>
          <div class="headline">${remoteService.unit_name || "ernie-cockpit-remote.service"}</div>
          <div class="body">Status: ${remoteService.status || "unknown"}</div>
          <div class="body" style="margin-top:8px;">URL: ${remoteUrl}</div>
          <div class="body" style="margin-top:8px;">Token: ${remoteService.token || "No remote token configured."}</div>
          <div class="actions" style="margin-top:12px;">
            <button id="rotate-remote-token" class="ghost"${remoteService.configured ? "" : " disabled"}>Rotate remote token</button>
          </div>
        </div>
        <div class="card">
          <div class="eyebrow">Desktop integration</div>
          <div class="headline">App launcher</div>
          <div class="body">Desktop entry: ${desktop.desktop_entry_installed ? "installed" : "missing"}</div>
          <div class="body" style="margin-top:8px;">Launcher: ${desktop.launcher_installed ? "installed" : "missing"}</div>
          <div class="body" style="margin-top:8px;">Icon: ${desktop.icon_installed ? "installed" : "missing"}</div>
        </div>
        ${policyCard}
        ${suppressionCard}
        ${trustedSignalCard}
        ${actionButtons}
      `;
      const rotate = document.getElementById("rotate-remote-token");
      if (rotate && !rotate.disabled) {
        rotate.addEventListener("click", async () => {
          if (!window.confirm("Rotate the managed remote token and restart the remote service?")) return;
          const payload = await requestJson("/api/settings/rotate-remote-token", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({})
          });
          document.getElementById("execution-output").textContent = JSON.stringify(payload, null, 2);
          await loadDashboard();
        });
      }
      node.querySelectorAll("[data-service-action]").forEach((button) => {
        if (button.disabled) return;
        button.addEventListener("click", async () => {
          const action = button.getAttribute("data-service-action");
          await runSetupAction(action);
        });
      });
      const savePilotPolicy = document.getElementById("save-pilot-policy");
      if (savePilotPolicy) {
        savePilotPolicy.addEventListener("click", async () => {
          const enabled = document.getElementById("trusted-write-enabled").value === "true";
          const threshold = Number(document.getElementById("trusted-write-threshold").value || trustedThreshold);
          const operations = document.getElementById("trusted-write-operations").value
            .split(",")
            .map((item) => item.trim())
            .filter(Boolean);
          const payload = await requestJson("/api/settings/pilot-policy", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
              trusted_writes_enabled: enabled,
              trusted_write_required_successes: threshold,
              trusted_write_operations: operations
            })
          });
          const updatedPolicy = payload.pilot_policy || {};
          const summary = [
            `Trusted writes ${updatedPolicy.trusted_writes_enabled ? "enabled" : "disabled"}.`,
            `Threshold ${updatedPolicy.trusted_auto_approve_required_successes || threshold}.`,
            (updatedPolicy.trusted_auto_approve_file_operations || []).length
              ? `Operations: ${(updatedPolicy.trusted_auto_approve_file_operations || []).join(", ")}.`
              : "No trusted write operations are active."
          ].join(" ");
          setActionBanner("success", "Pilot policy updated.", summary);
          document.getElementById("execution-output").textContent = JSON.stringify(payload, null, 2);
          await loadDashboard();
        });
      }
    }

    function renderOnboarding(settings) {
      const node = document.getElementById("onboarding-panel");
      const onboarding = settings.onboarding || {};
      const serviceSyncStatus = settings.service_sync_status || {};
      const steps = onboarding.recommended_steps || [];
      const actions = onboarding.actions || [];
      const suppressedSyncActions = serviceSyncStatus.suppressed_recent_verification_actions || [];
      const suppressedSyncTitles = serviceSyncStatus.suppressed_recent_verification_titles || [];
      const suppressedSyncScopes = serviceSyncStatus.suppressed_recent_verification_scopes || [];
      const suppressedAgeSeconds = Number(serviceSyncStatus.suppressed_recent_verification_age_seconds || 0);
      const suppressedExpiresInSeconds = Number(serviceSyncStatus.suppressed_recent_verification_expires_in_seconds || 0);
      const freshnessText = suppressedSyncActions.length
        ? `Healthy verification recorded about ${Math.max(0, Math.floor(suppressedAgeSeconds / 60))} minute(s) ago.`
        : "";
      const expiryText = suppressedSyncActions.length
        ? `Suppression window expires in about ${Math.max(0, Math.ceil(suppressedExpiresInSeconds / 60))} minute(s).`
        : "";
      const chips = [];
      chips.push(`Local service ${onboarding.local_service_ready ? "ready" : "needs attention"}`);
      chips.push(`Desktop ${onboarding.desktop_ready ? "installed" : "not installed"}`);
      chips.push(`Remote ${onboarding.remote_service_ready ? "ready" : "not ready"}`);
      node.innerHTML = `
        <div class="card">
          <div class="eyebrow">This machine</div>
          <div class="headline">Ernie setup status</div>
          <div class="body">${chips.join(" · ")}</div>
          <div class="body" style="margin-top:10px;">Local URL: ${onboarding.local_url || "unknown"}</div>
          <div class="body" style="margin-top:8px;">Remote URL: ${onboarding.remote_url || "not configured"}</div>
        </div>
        <div class="card">
          <div class="eyebrow">Recommended next step</div>
          <div class="body">${steps.length ? steps.join(" ") : "No setup guidance available."}</div>
          ${suppressedSyncActions.length ? `
            <div class="notice info" style="margin-top:12px;">
              <div class="headline" style="font-size:1rem;">Recent healthy verification is calming one setup recommendation.</div>
              <div class="body">Suppressed actions: ${suppressedSyncActions.join(", ")}</div>
              <div class="body" style="margin-top:8px;">Scopes: ${suppressedSyncScopes.join(", ")}</div>
              <div class="body" style="margin-top:8px;">${freshnessText} ${expiryText}</div>
              <div class="body" style="margin-top:8px;">Suppressed tasks: ${suppressedSyncTitles.length ? suppressedSyncTitles.join(", ") : "No queued setup task titles were left to suppress."}</div>
            </div>
          ` : ""}
          <div style="margin-top:12px; display:grid; gap:10px;">
            ${actions.map((item) => `
              <div class="card" style="padding:12px; border-radius:16px;">
                <div class="headline" style="font-size:1rem;">${item.label}</div>
                <div class="body" style="margin-top:6px;">${item.description || ""}</div>
                <div class="actions" style="margin-top:10px;">
                  <button class="ghost" data-service-action="${item.action}"${item.enabled ? "" : " disabled"}>${item.label}</button>
                </div>
              </div>
            `).join("")}
          </div>
        </div>
      `;
      node.querySelectorAll("[data-service-action]").forEach((button) => {
        if (button.disabled) return;
        button.addEventListener("click", async () => {
          const action = button.getAttribute("data-service-action");
          await runSetupAction(action);
        });
      });
    }

    function renderSoul(settings) {
      const node = document.getElementById("soul-panel");
      const soul = settings.soul || {};
      const principles = soul.principles || [];
      const proposals = currentSoulReview?.proposals || [];
      const auditTrail = settings.soul_audit || [];
      node.innerHTML = `
        <div class="card">
          <div class="eyebrow">${soul.source || "SOUL.md"}</div>
          <div class="headline">${soul.title || "Ernie"}</div>
          <div class="body">${soul.summary || "Ernie should stay calm, bounded, and legible while operating the local machine."}</div>
          <div class="body" style="margin-top:10px;">${principles.length ? principles.map((item) => `- ${item}`).join("<br>") : "No soul principles are loaded yet."}</div>
          <div class="actions" style="margin-top:12px;">
            <button id="review-soul" class="ghost">Review soul</button>
          </div>
        </div>
        <div class="card">
          <div class="eyebrow">Governed identity loop</div>
          <div class="headline">${proposals.length ? "Review proposed amendments" : "No soul amendments proposed yet"}</div>
          <div class="body">${proposals.length ? "Ernie can suggest amendments from operator history, but nothing applies until you click an approval button." : "Run a soul review to inspect evidence-backed amendments before making any identity change."}</div>
          <div style="margin-top:12px; display:grid; gap:10px;">
            ${proposals.map((proposal) => `
              <div class="card" style="padding:12px; border-radius:16px;">
                <div class="headline" style="font-size:1rem;">${proposal.title}</div>
                <div class="body" style="margin-top:6px;">${proposal.rationale}</div>
                <div class="body muted" style="margin-top:8px;">Why now: ${proposal.explanation || "No explanation recorded."}</div>
                <div class="body muted" style="margin-top:6px;">Target section: ${proposal.section || "unknown"} · Signature: ${proposal.evidence_signature || "n/a"}</div>
                <div class="body muted" style="margin-top:6px;">${proposal.resurfaced_after_dismissal ? "This proposal resurfaced because the evidence changed after an earlier dismissal." : "This proposal has not resurfaced after dismissal."}</div>
                <div class="body" style="margin-top:8px;">${proposal.amendment}</div>
                <div class="body muted" style="margin-top:8px;">${(proposal.evidence || []).length ? "Evidence from recent operator messages:" : "No evidence recorded."}</div>
                <div class="body muted" style="margin-top:6px;">${(proposal.evidence || []).length ? proposal.evidence.map((item) => `- ${item}`).join("<br>") : ""}</div>
                <div class="actions" style="margin-top:10px;">
                  <button class="ghost" data-soul-apply="${proposal.proposal_id}">Apply this amendment</button>
                  <button class="ghost" data-soul-dismiss="${proposal.proposal_id}">Dismiss for now</button>
                </div>
              </div>
            `).join("")}
          </div>
        </div>
        <div class="card">
          <div class="eyebrow">Soul audit trail</div>
          <div class="headline">${auditTrail.length ? "Recent identity reviews and changes" : "No soul audit entries yet"}</div>
          <div class="body">${auditTrail.length ? auditTrail.map((item) => `${item.tool_name} [${item.status}]: ${item.outcome}<br><span class="muted">proposal=${item.proposal_id || "n/a"} · signature=${item.evidence_signature || "n/a"}</span>`).join("<br>") : "Reviewed, applied, and dismissed soul actions will appear here."}</div>
        </div>
      `;
      const reviewSoul = document.getElementById("review-soul");
      if (reviewSoul) {
        reviewSoul.addEventListener("click", async () => {
          const payload = await requestJson("/api/soul/review", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({})
          });
          currentSoulReview = payload;
          setActionBanner("success", "Soul review ready.", payload.proposals?.length ? `Ernie proposed ${payload.proposals.length} amendment(s) for review.` : "No new soul amendments were proposed from recent operator history.");
          renderSoul(settings);
          document.getElementById("execution-output").textContent = JSON.stringify(payload, null, 2);
        });
      }
      node.querySelectorAll("[data-soul-apply]").forEach((button) => {
        button.addEventListener("click", async () => {
          const proposalId = button.getAttribute("data-soul-apply");
          const payload = await requestJson("/api/soul/apply", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({proposal_id: proposalId})
          });
          currentSoulReview = payload.review || null;
          setActionBanner("success", "Soul amendment applied.", payload.summary || "The approved soul amendment was written to SOUL.md.");
          document.getElementById("execution-output").textContent = JSON.stringify(payload, null, 2);
          await loadDashboard();
        });
      });
      node.querySelectorAll("[data-soul-dismiss]").forEach((button) => {
        button.addEventListener("click", async () => {
          const proposalId = button.getAttribute("data-soul-dismiss");
          const payload = await requestJson("/api/soul/dismiss", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({proposal_id: proposalId})
          });
          currentSoulReview = payload.review || null;
          setActionBanner("info", "Soul amendment dismissed.", payload.summary || "The proposal was dismissed and will stop resurfacing until the evidence changes.");
          document.getElementById("execution-output").textContent = JSON.stringify(payload, null, 2);
          await loadDashboard();
        });
      });
    }

    function renderTutorial(snapshot, settings) {
      const node = document.getElementById("tutorial-panel");
      const recommendation = snapshot.plan?.recommendation || null;
      const pendingApprovals = snapshot.pending_pilot_turns || [];
      const recentActivity = snapshot.recent_activity || [];
      const recentDemoEvents = recentActivity.filter((item) => (item.metadata || {}).tool_name === "demo-workflow").slice(0, 3);
      const onboardingActions = settings.onboarding?.actions || [];
      const dueSetupActions = onboardingActions.filter((item) => item.enabled);
      const steps = [];

      steps.push({
        state: dueSetupActions.length ? "recommended" : "ready",
        title: dueSetupActions.length ? "Start with machine setup" : "Machine setup looks usable",
        body: dueSetupActions.length
          ? `Use Machine status to run ${dueSetupActions[0].label}. Clear setup work before testing deeper agent flows.`
          : "No urgent setup action is blocking your first bounded test run. You can move straight to the planner loop."
      });
      steps.push({
        state: recommendation ? "ready" : "waiting",
        title: "Review the planner recommendation",
        body: recommendation
          ? `Open Next action and read why Ernie chose "${recommendation.title}". If it looks bounded and sensible, run one step only.`
          : "No recommendation is surfaced yet. Capture a task or refresh the view first."
      });
      steps.push({
        state: pendingApprovals.length ? "approval" : "ready",
        title: pendingApprovals.length ? "Stop and review approval requests" : "If approval appears, stop and inspect it",
        body: pendingApprovals.length
          ? `Pilot approval queue currently has ${pendingApprovals.length} item(s). Read the prompt before approving anything.`
          : "If Ernie asks for approval, use the Pilot approval queue and read the rationale before continuing."
      });
      steps.push({
        state: recentActivity.length ? "ready" : "waiting",
        title: "Check what changed after each step",
        body: recentActivity.length
          ? "Use Action status and Operator timeline to confirm what just ran and what Ernie recommends next."
          : "After you run a step, read Action status and Operator timeline before taking another action."
      });
      steps.push({
        state: "info",
        title: "Safe stopping rule",
        body: "For an early test session, stop after one or two bounded actions. If the next move looks unclear, refresh and review instead of pushing deeper."
      });

      node.innerHTML = `
        <div class="card">
          <div class="eyebrow">Guided first session</div>
          <div class="headline">Use bounded steps for your first hands-on test</div>
          <div class="body">Refresh state, inspect one recommendation, run one bounded step, confirm the outcome, then pause or review again.</div>
          <div class="actions" style="margin-top:12px;">
            <button id="seed-demo-workflow" class="ghost">Seed demo workflow</button>
            <button id="reset-demo-workflow" class="ghost">Reset demo workflow</button>
          </div>
        </div>
        <div class="card">
          <div class="eyebrow">Recent demo walkthrough</div>
          <div class="headline">${recentDemoEvents.length ? "Most recent demo events" : "No demo events recorded yet"}</div>
          <div class="body">${recentDemoEvents.length ? recentDemoEvents.map((item) => summarizeActivity(item)).join(" ") : "Seed the demo workflow, run one bounded demo step, and the walkthrough history will appear here."}</div>
        </div>
        ${steps.map((step, index) => `
          <div class="card">
            <div class="status-line"><span class="eyebrow">step ${index + 1}</span>${renderStatePill(step.state)}</div>
            <div class="headline">${step.title}</div>
            <div class="body">${step.body}</div>
          </div>
        `).join("")}
      `;
      const seedDemo = document.getElementById("seed-demo-workflow");
      if (seedDemo) {
        seedDemo.addEventListener("click", async () => {
          const payload = await requestJson("/api/demo/seed", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({})
          });
          const created = payload.created_titles || [];
          const unchanged = payload.unchanged_titles || [];
          const summary = [
            created.length ? `Created: ${created.join(", ")}.` : "No new demo tasks were needed.",
            unchanged.length ? `Already present: ${unchanged.join(", ")}.` : "",
            "Use Next action or task detail to try one bounded step."
          ].filter(Boolean).join(" ");
          setActionBanner("success", "Demo workflow ready.", summary);
          document.getElementById("execution-output").textContent = JSON.stringify(payload, null, 2);
          await loadDashboard();
        });
      }
      const resetDemo = document.getElementById("reset-demo-workflow");
      if (resetDemo) {
        resetDemo.addEventListener("click", async () => {
          const payload = await requestJson("/api/demo/reset", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({})
          });
          const resetTitles = payload.reset_titles || [];
          const summary = [
            resetTitles.length ? `Reset: ${resetTitles.join(", ")}.` : "No demo tasks were reset.",
            "This restores the demo tasks only. It does not undo unrelated edits or service changes.",
          ].join(" ");
          setActionBanner("success", "Demo workflow reset.", summary);
          document.getElementById("execution-output").textContent = JSON.stringify(payload, null, 2);
          await loadDashboard();
        });
      }
    }

    async function runSetupAction(action) {
      if (!action) return;
      const currentSettings = await requestJson("/api/settings");
      const actionList = [
        ...(currentSettings.onboarding?.actions || []),
        ...(currentSettings.actions || [])
      ];
      const meta = actionList.find((item) => item.action === action) || {};
      if (meta.requires_confirmation && !window.confirm(meta.confirmation_message || "Run this setup action?")) {
        return;
      }
      const payload = await requestJson("/api/settings/action", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({action})
      });
      const summary = [
        payload.message || "Action completed.",
        payload.result?.stdout || "",
        payload.result?.stderr || ""
      ].filter(Boolean).join("\n\n");
      setActionBanner("success", "Setup action finished.", summary || "The requested setup step completed.");
      document.getElementById("execution-output").textContent = summary || JSON.stringify(payload, null, 2);
      await loadDashboard();
    }

    async function executePlanAction(action, query) {
      if (!action || !action.kind || !action.title) return;
      const payload = await requestJson("/api/execute-plan-action", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          query: query || document.getElementById("query").value.trim() || "what should I do next",
          kind: action.kind,
          title: action.title,
          task_id: action.task_id || null,
          limit: 5
        })
      });
      currentPlanExplanation = payload.model_explanation || null;
      setActionBanner("success", "Planner action executed.", summarizeExecution(payload));
      document.getElementById("execution-output").textContent = JSON.stringify(payload, null, 2);
      await loadDashboard();
    }

    function taskActionSummary(action, title, payload) {
      if (action === "complete") return `Marked "${title}" complete and refreshed the task list.`;
      if (action === "unblock") return `Removed blockers for "${title}" and refreshed readiness.`;
      if (action === "resume") return `Returned "${title}" to active work.`;
      if (action === "snooze") return `Snoozed "${title}" for one day.`;
      return payload?.message || `Updated task "${title}".`;
    }

    function summarizeExecution(payload) {
      const result = payload?.result || {};
      const afterPlan = payload?.after_plan || {};
      const recommendation = afterPlan?.recommendation || {};
      const pieces = [];
      if (result.status) pieces.push(`Result: ${result.status}.`);
      if (result.message) pieces.push(result.message);
      if (payload?.model_explanation?.text) pieces.push(payload.model_explanation.text);
      if (recommendation.title) {
        pieces.push(`Next recommendation: ${recommendation.title}.`);
      }
      return pieces.join(" ") || "Executed the current planner recommendation and refreshed state.";
    }

    function summarizePilotApproval(payload) {
      const outcome = payload?.execution_result || {};
      const action = payload?.selected_action || {};
      const approval = payload?.approval || {};
      const pieces = [];
      if (action.title) pieces.push(`Approved "${action.title}".`);
      if (outcome.status) pieces.push(`Execution status: ${outcome.status}.`);
      if (approval.category === "trusted_file_operation") {
        pieces.push(summarizeTrustedApproval(approval));
      }
      if (payload?.assistant_message) pieces.push(payload.assistant_message);
      return pieces.join(" ") || "Approved the pilot action and refreshed the queue.";
    }

    function summarizePilotPreview(payload) {
      const approval = payload?.approval || {};
      const action = payload?.selected_action || {};
      if (approval.status === "needs_approval") {
        return approval.prompt || approval.reason || `Review "${action.title || "the proposed action"}" before it runs.`;
      }
      if (approval.category === "trusted_file_operation") {
        return summarizeTrustedApproval(approval) || approval.reason || "This pilot action was auto-approved by trusted preview history.";
      }
      if (approval.reason) {
        return approval.reason;
      }
      return `Pilot preview recorded for "${action.title || "the proposed action"}".`;
    }

    function summarizePatchRollback(payload, runId) {
      const status = payload?.status || payload?.git?.status || "completed";
      return `Rollback for patch run ${runId} finished with status: ${status}.`;
    }

    function summarizeObserve(role, text) {
      const trimmed = (text || "").trim();
      const preview = trimmed.length > 120 ? trimmed.slice(0, 117) + "..." : trimmed;
      return `Stored a ${role} note${preview ? `: "${preview}"` : ""}`;
    }

    async function loadTaskDetail(title, area) {
      const payload = await requestJson(
        "/api/task-detail?title=" + encodeURIComponent(title) + (area ? "&area=" + encodeURIComponent(area) : "")
      );
      const node = document.getElementById("task-detail");
      const task = payload.task || {};
      const meta = task.metadata || {};
      const blocks = [];
      if (meta.blocked_by && meta.blocked_by.length) blocks.push("blocked by " + meta.blocked_by.join(", "));
      if (meta.depends_on && meta.depends_on.length) blocks.push("depends on " + meta.depends_on.join(", "));
      if (meta.due_date) blocks.push("due " + meta.due_date);
      if (meta.command) blocks.push("command " + meta.command);
      if (meta.file_operation && meta.file_path) blocks.push("file " + meta.file_operation + " " + meta.file_path);
      const entityList = (payload.entities || []).slice(0, 4).map((item) => {
        const entity = item.entity || {};
        return entity.name || entity.canonical_name || entity.entity_type || "entity";
      });
      const taskPlanActions = payload.task_plan_actions || [];
      node.innerHTML = `
        <div class="card">
          <div class="status-line"><span class="eyebrow">${task.subject || "task"}</span>${renderStatePill(meta.status || "open")}</div>
          <div class="headline">${meta.title || task.content || title}</div>
          <div class="body">${meta.details || task.content || ""}</div>
          <div class="actions tight" style="margin-top:12px;">
            <button class="ghost" data-task-action="focus">Focus</button>
            <button class="ghost" data-task-action="complete">Complete</button>
            <button class="ghost" data-task-action="unblock">Unblock</button>
            <button class="ghost" data-task-action="resume">Resume</button>
            <button class="ghost" data-task-action="snooze">Snooze 1 day</button>
          </div>
        </div>
        <div class="card">
          <div class="eyebrow">Suggested next moves</div>
          <div class="body">${taskPlanActions.length ? "These task-specific actions come from the current planner view for this task title." : "No task-scoped planner actions are surfaced for this task yet."}</div>
          <div class="body muted" style="margin-top:8px;">Prepare = safer preflight. Delegate = tracked handoff. Run = execute the currently bounded step for this task.</div>
          ${taskPlanActions.length ? `
            <div style="display:grid; gap:10px; margin-top:12px;">
              ${taskPlanActions.map((item) => `
                <div class="card" style="padding:12px; border-radius:16px;">
                  <div class="status-line"><span class="eyebrow">${item.kind}</span>${renderStatePill("ready")}</div>
                  <div class="headline">${item.title}</div>
                  <div class="body">${item.summary}</div>
                  <div class="body muted" style="margin-top:8px;">${explainPlannerAction(item.kind)}</div>
                  <div class="actions" style="margin-top:10px;">
                    <button class="ghost" data-task-plan-run="${item.kind}::${item.title}">Run this step</button>
                  </div>
                </div>
              `).join("")}
            </div>
          ` : ""}
        </div>
        <div class="card">
          <div class="eyebrow">Execution shape</div>
          <div class="body">${blocks.length ? blocks.join(" · ") : "No blockers, dependencies, or execution metadata recorded."}</div>
        </div>
        <div class="card">
          <div class="eyebrow">Related entities</div>
          <div class="body">${entityList.length ? entityList.join(" · ") : "No linked entities surfaced for this task yet."}</div>
        </div>
        <div class="card">
          <div class="eyebrow">Recent nudges</div>
          <div class="body">${(payload.recent_nudges || []).length ? payload.recent_nudges.map(item => item.content).join(" | ") : "No recent nudges for this task."}</div>
        </div>
      `;
      node.querySelectorAll("[data-task-action]").forEach((button) => {
        button.addEventListener("click", async () => {
          const action = button.getAttribute("data-task-action");
          if (action === "focus") {
            focusedTaskTitle = meta.title || task.content || title;
            persistFocusedTask();
            setActionBanner("info", "Focused task selected.", `Planner context is now centered on "${focusedTaskTitle}".`);
            await loadDashboard();
            return;
          }
          await runTaskAction(action, meta.title || task.content || title, task.subject || area || "execution");
        });
      });
      node.querySelectorAll("[data-task-plan-run]").forEach((button, index) => {
        button.addEventListener("click", async () => {
          const action = taskPlanActions[index];
          if (!action) return;
          await executePlanAction(action, meta.title || task.content || title);
        });
      });
    }

    function renderContext(context) {
      const node = document.getElementById("context-output");
      node.innerHTML = "";
      const topProfiles = (context.profiles || []).slice(0, 2);
      const topMemories = (context.memories || []).slice(0, 3);
      const recentEvents = (context.recent_events || []).slice(0, 3);

      if (!topProfiles.length && !topMemories.length && !recentEvents.length) {
        node.innerHTML = "<div class='card'><div class='body muted'>No context surfaced yet.</div></div>";
        return;
      }

      topProfiles.forEach((item) => {
        const profile = item.memory || {};
        const div = document.createElement("div");
        div.className = "card";
        div.innerHTML = `
          <div class="eyebrow">Profile · ${profile.subject || "memory"}</div>
          <div class="body">${profile.content || ""}</div>
        `;
        node.appendChild(div);
      });

      topMemories.forEach((item) => {
        const memory = item.memory || {};
        const div = document.createElement("div");
        div.className = "card";
        div.innerHTML = `
          <div class="eyebrow">${memory.kind || "memory"} · ${memory.subject || ""}</div>
          <div class="body">${memory.content || ""}</div>
        `;
        node.appendChild(div);
      });

      recentEvents.forEach((event) => {
        const div = document.createElement("div");
        div.className = "card";
        div.innerHTML = `
          <div class="eyebrow">Recent ${event.role || "event"}</div>
          <div class="body">${event.content || ""}</div>
        `;
        node.appendChild(div);
      });
    }

    function renderNudges(nudges) {
      const node = document.getElementById("recent-nudges");
      node.innerHTML = "";
      const ordered = prioritizeFocused(nudges, (nudge) => matchesFocusedTask(
        nudge?.metadata?.task_title,
        nudge?.subject,
        nudge?.content
      ));
      if (!ordered.length) {
        node.innerHTML = "<div class='card'><div class='body muted'>No recent nudges.</div></div>";
        return;
      }
      for (const nudge of ordered) {
        const title = nudge.metadata.task_title || nudge.subject || "nudge";
        const type = nudge.metadata.nudge_type || "reminder";
        const div = document.createElement("div");
        div.className = "card";
        div.innerHTML = `
          <div class="eyebrow">${type}</div>
          <div class="headline">${title}</div>
          <div class="body">${nudge.content}</div>
        `;
        node.appendChild(div);
      }
    }

    function renderOpenTasks(tasks) {
      const node = document.getElementById("open-tasks");
      node.innerHTML = "";
      const ordered = prioritizeFocused(tasks, (task) => matchesFocusedTask(
        task?.metadata?.title,
        task?.content
      ));
      if (!ordered.length) {
        node.innerHTML = "<div class='card'><div class='body muted'>No open tasks.</div></div>";
        return;
      }
      ordered.forEach((task, index) => {
        const title = task.metadata.title || task.content;
        const div = document.createElement("div");
        div.className = "card";
        div.innerHTML = `
          <div class="status-line"><span class="eyebrow">${task.subject || "task"}</span>${renderStatePill(task.metadata.status || "open")}</div>
          <div class="headline">${title}</div>
          <div class="body">${summarizeTaskCard(task)}</div>
        `;
        div.style.cursor = "pointer";
        div.addEventListener("click", () => loadTaskDetail(title, task.subject || ""));
        node.appendChild(div);
        if (index === 0) {
          loadTaskDetail(title, task.subject || "").catch(() => {});
        }
      });
    }

    function renderPilotPending(items) {
      currentPilotItems = prioritizeFocused(items, (item) => matchesFocusedTask(
        item?.selected_action?.title,
        item?.selected_action?.summary,
        item?.approval?.prompt,
        item?.approval?.reason
      ));
      const node = document.getElementById("pilot-pending");
      node.innerHTML = "";
      if (!currentPilotItems.length) {
        node.innerHTML = "<div class='card'><div class='body muted'>No pending pilot approvals.</div></div>";
        renderEmptyDetail("pilot-detail", "Pilot detail", "Select or preview a pending pilot action to inspect its exact review packet.");
        selectedPilotPendingId = "";
        return;
      }
      currentPilotItems.forEach((item) => {
        const action = item.selected_action || {};
        const approval = item.approval || {};
        const sourceLabel = item.request_origin === "prompt_workshop"
          ? `Prompt Workshop ${item.prompt_source === "result" ? "result" : "draft"}`
          : "Manual pilot request";
        const div = document.createElement("div");
        div.className = "card";
        if (item.pending_id === selectedPilotPendingId) {
          div.classList.add("active");
        }
        div.innerHTML = `
          <div class="status-line"><span class="eyebrow">${sourceLabel}</span>${renderStatePill(approval.status || "pending")}</div>
          <div class="headline">${action.title || "Pending action"}</div>
          <div class="body">${summarizePilotCard(item)}</div>
          <div class="actions" style="margin-top:12px;">
            <button class="approve-btn">Approve</button>
            <button class="ghost reject-btn">Reject</button>
          </div>
        `;
        div.style.cursor = "pointer";
        div.addEventListener("click", () => selectPilotPending(item.pending_id));
        div.querySelector(".approve-btn").addEventListener("click", async (event) => {
          event.stopPropagation();
          const payload = await requestJson("/api/pilot/approve", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({pending_id: item.pending_id})
          });
          setActionBanner("success", "Pilot action approved.", summarizePilotApproval(payload));
          document.getElementById("execution-output").textContent = JSON.stringify(payload, null, 2);
          await loadDashboard();
        });
        div.querySelector(".reject-btn").addEventListener("click", async (event) => {
          event.stopPropagation();
          await requestJson("/api/pilot/reject", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({pending_id: item.pending_id, reason: "rejected from cockpit"})
          });
          setActionBanner("info", "Pilot action rejected.", "The pending pilot action was removed from the approval queue.");
          await loadDashboard();
        });
        node.appendChild(div);
      });
      if (!selectedPilotPendingId || !currentPilotItems.some((item) => item.pending_id === selectedPilotPendingId)) {
        selectedPilotPendingId = currentPilotItems[0].pending_id || "";
      }
      selectPilotPending(selectedPilotPendingId, false);
    }

    function renderPatchRuns(runs) {
      currentPatchRuns = prioritizeFocused(runs, (run) => matchesFocusedTask(
        run?.task_title,
        run?.run_name,
        run?.summary?.task_title
      ));
      const node = document.getElementById("patch-runs");
      node.innerHTML = "";
      if (!currentPatchRuns.length) {
        node.innerHTML = "<div class='card'><div class='body muted'>No patch runs recorded yet.</div></div>";
        renderEmptyDetail("patch-detail", "Patch detail", "Select a patch run to inspect diff preview, validations, and rollback hints.");
        selectedPatchRunId = 0;
        return;
      }
      currentPatchRuns.forEach((run) => {
        const git = (run.summary || {}).git || {};
        const candidate = (run.summary || {}).candidate_evaluation || {};
        const validations = candidate.validation_failures || candidate.failures || [];
        const delta = [];
        if (candidate.baseline_score !== undefined) delta.push("baseline " + candidate.baseline_score);
        if (candidate.candidate_score !== undefined) delta.push("candidate " + candidate.candidate_score);
        const div = document.createElement("div");
        div.className = "card";
        if (run.id === selectedPatchRunId) {
          div.classList.add("active");
        }
        div.innerHTML = `
          <div class="status-line"><span class="eyebrow">${run.suite_name}</span>${renderStatePill(run.status || "recorded")}</div>
          <div class="headline">${run.run_name}</div>
          <div class="body">${summarizePatchCard(run)}</div>
          <div class="body" style="margin-top:10px;">${candidate.status ? "Validation " + candidate.status : "No candidate validation summary recorded."}${delta.length ? " · " + delta.join(" · ") : ""}</div>
          <div class="body" style="margin-top:8px;">${validations.length ? validations.join(" | ") : "No validation failures recorded for this run."}</div>
          <div class="status-line" style="margin-top:10px;">
            <span class="pill">run ${run.id}</span>
            <span>${git.status ? "git " + git.status : "no git apply metadata"}</span>
          </div>
          <div class="actions" style="margin-top:12px;">
            <button class="ghost rollback-btn"${git.rollback_ready ? "" : " disabled"}>Rollback branch</button>
          </div>
        `;
        div.style.cursor = "pointer";
        div.addEventListener("click", () => selectPatchRun(run.id));
        const rollback = div.querySelector(".rollback-btn");
        rollback.addEventListener("click", async (event) => {
          event.stopPropagation();
          if (!window.confirm(`Rollback patch run ${run.id}?`)) return;
          const payload = await requestJson("/api/patch-rollback", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({run_id: run.id})
          });
          setActionBanner("success", "Patch rollback finished.", summarizePatchRollback(payload, run.id));
          document.getElementById("execution-output").textContent = JSON.stringify(payload, null, 2);
          await loadDashboard();
        });
        node.appendChild(div);
      });
      if (!selectedPatchRunId || !currentPatchRuns.some((run) => run.id === selectedPatchRunId)) {
        selectedPatchRunId = Number(currentPatchRuns[0].id || 0);
      }
      selectPatchRun(selectedPatchRunId, false);
    }

    function summarizeActivity(item) {
      const meta = item.metadata || {};
      const tool = meta.tool_name || "tool";
      const status = meta.status || "recorded";
      if (tool === "patch-run") {
        const changed = (meta.changed_files || []).length;
        return `Patch run ${status}. ${changed ? changed + " files changed." : "No file list was recorded."}`;
      }
      if (tool === "patch-rollback") {
        return `Rollback ${status}. ${item.content || ""}`.trim();
      }
      if (tool === "pilot-review") {
        return `Pilot review captured. ${item.content || ""}`.trim();
      }
      if (tool === "executor") {
        return `Executor ${status}. ${item.content || ""}`.trim();
      }
      return `${tool} ${status}. ${item.content || ""}`.trim();
    }

    function activityTopic(item) {
      const meta = item.metadata || {};
      if (meta.run_id) return `Patch run ${meta.run_id}`;
      if (meta.task_title) return `Task: ${meta.task_title}`;
      if (meta.pending_id) return `Pilot review ${meta.pending_id}`;
      if (item.subject) return item.subject;
      return meta.tool_name || "activity";
    }

    function activityNarrative(item) {
      const meta = item.metadata || {};
      const pieces = [];
      if (meta.tool_name) pieces.push(`Tool: ${meta.tool_name}.`);
      if (meta.status) pieces.push(`Status: ${meta.status}.`);
      if (meta.task_title) pieces.push(`Related task: ${meta.task_title}.`);
      if (meta.outcome) pieces.push(meta.outcome);
      else if (item.content) pieces.push(item.content);
      return pieces.join(" ") || "No additional detail recorded.";
    }

    function isPromptPilotActivity(item) {
      const tool = item?.metadata?.tool_name || "";
      return tool === "prompt-pilot-bridge" || tool === "prompt-pilot-execution";
    }

    function summarizePromptPilotChain(group) {
      const bridge = group.items.find((entry) => entry?.metadata?.tool_name === "prompt-pilot-bridge") || null;
      const execution = group.items.find((entry) => entry?.metadata?.tool_name === "prompt-pilot-execution") || null;
      const promptSource = execution?.metadata?.prompt_source || bridge?.metadata?.prompt_source || "draft";
      const selectedTitle = execution?.metadata?.selected_action_title || bridge?.metadata?.selected_action_title || "the selected action";
      const executionStatus = execution?.metadata?.execution_status || execution?.metadata?.status || "";
      const rejectionReason = execution?.metadata?.rejection_reason || "";
      const pieces = [
        `Prompt Workshop ${promptSource} entered the guided loop for ${selectedTitle}.`
      ];
      if (bridge) {
        pieces.push("A pilot review packet was queued for approval.");
      }
      if (execution) {
        if (rejectionReason.startsWith("validation_failed:")) {
          pieces.push("After approval, the previewed patch was rejected by validation before apply.");
        } else if (rejectionReason) {
          pieces.push(`After approval, execution stopped before apply: ${rejectionReason}.`);
        } else {
          pieces.push(`After approval, execution finished with status ${executionStatus || "recorded"}.`);
        }
      } else {
        pieces.push("Execution has not been recorded yet.");
      }
      return pieces.join(" ");
    }

    function activityGroupSummary(group) {
      if (group.items.some(isPromptPilotActivity)) {
        return summarizePromptPilotChain(group);
      }
      return summarizeActivity(group.items[0]);
    }

    function activityGroupNarrative(group) {
      if (group.items.some(isPromptPilotActivity)) {
        return summarizePromptPilotChain(group);
      }
      return group.items.map((entry) => summarizeActivity(entry)).join(" ");
    }

    function groupActivity(items) {
      const groups = [];
      items.forEach((item) => {
        const topic = activityTopic(item);
        const last = groups[groups.length - 1];
        if (last && last.topic === topic) {
          last.items.push(item);
        } else {
          groups.push({topic, items: [item]});
        }
      });
      return groups;
    }

    function renderActivity(items) {
      currentActivityItems = prioritizeFocused(items, (item) => matchesFocusedTask(
        item?.metadata?.task_title,
        item?.subject,
        item?.content,
        activityTopic(item)
      ));
      const node = document.getElementById("recent-activity");
      node.innerHTML = "";
      if (!currentActivityItems.length) {
        node.innerHTML = "<div class='card'><div class='body muted'>No recent tool activity.</div></div>";
        renderEmptyDetail("activity-detail", "Timeline detail", "Select a timeline item to inspect the recorded outcome and metadata.");
        selectedActivityId = 0;
        return;
      }
      groupActivity(currentActivityItems).forEach((group) => {
        const item = group.items[0];
        const meta = item.metadata || {};
        const div = document.createElement("div");
        div.className = "card";
        const groupIds = group.items.map((entry) => Number(entry.id));
        if (groupIds.includes(selectedActivityId)) {
          div.classList.add("active");
        }
        div.innerHTML = `
          <div class="status-line"><span class="eyebrow">${group.items.some(isPromptPilotActivity) ? "prompt workshop chain" : meta.tool_name || "tool"}</span>${renderStatePill(meta.status || "status")}</div>
          <div class="headline">${group.topic}</div>
          <div class="body">${activityGroupSummary(group)}</div>
          <div class="body" style="margin-top:8px;">${group.items.length > 1 ? `${group.items.length} related events are grouped here.` : "Single recorded event."}</div>
        `;
        div.style.cursor = "pointer";
        div.addEventListener("click", () => selectActivity(item.id));
        node.appendChild(div);
      });
      if (!selectedActivityId || !currentActivityItems.some((item) => item.id === selectedActivityId)) {
        selectedActivityId = Number(currentActivityItems[0].id || 0);
      }
      selectActivity(selectedActivityId, false);
    }

    function selectPilotPending(pendingId, rerender = true) {
      selectedPilotPendingId = pendingId || "";
      const item = currentPilotItems.find((entry) => entry.pending_id === selectedPilotPendingId);
      const node = document.getElementById("pilot-detail");
      if (!item) {
        renderEmptyDetail("pilot-detail", "Pilot detail", "Select or preview a pending pilot action to inspect its exact review packet.");
        if (rerender) renderPilotPending(currentPilotItems);
        return;
      }
      const action = item.selected_action || {};
      const approval = item.approval || {};
      const preview = approval.preview_patch || {};
      const changedFiles = (preview.changed_files || []).join(", ");
      const diffPreview = preview.diff_preview || "No diff preview recorded.";
      const trustedSummary = approval.category === "trusted_file_operation"
        ? summarizeTrustedApproval(approval)
        : "";
      const sourceLabel = item.request_origin === "prompt_workshop"
        ? `Prompt Workshop ${item.prompt_source === "result" ? "result" : "draft"}`
        : "Manual pilot request";
      node.innerHTML = `
        <div class="card">
          <div class="status-line"><span class="eyebrow">${sourceLabel}</span>${renderStatePill(approval.status || "needs approval")}</div>
          <div class="headline">${action.title || "Pending action"}</div>
          <div class="body">${approval.prompt || approval.reason || action.summary || ""}</div>
          <div class="actions" style="margin-top:12px;">
            <button id="detail-approve">Approve this action</button>
            <button id="detail-reject" class="ghost">Reject this action</button>
          </div>
        </div>
        <div class="card">
          <div class="eyebrow">Request origin</div>
          <div class="body">${sourceLabel}.</div>
          <div class="body" style="margin-top:8px;">${item.request_origin === "prompt_workshop" ? "This preview came from the Prompt Workshop bridge after explicit confirmation." : "This preview came directly from the pilot request box."}</div>
        </div>
        <div class="card">
          <div class="eyebrow">Selected action</div>
          <div class="body">${action.kind || "unknown"} · ${action.summary || "No summary recorded."}</div>
          <div class="body" style="margin-top:8px;">${action.details || item.assistant_message || "No extra detail recorded."}</div>
        </div>
        <div class="card">
          <div class="eyebrow">Approval rationale</div>
          <div class="body">${trustedSummary || approval.reason || "No approval rationale recorded."}</div>
          <div class="mono" style="margin-top:10px;">${JSON.stringify(approval.metadata || {}, null, 2)}</div>
        </div>
        <div class="card">
          <div class="eyebrow">Previewed change set</div>
          <div class="body">${changedFiles || "No changed files previewed."}</div>
          <div class="mono" style="margin-top:10px;">${diffPreview}</div>
        </div>
      `;
      node.querySelector("#detail-approve").addEventListener("click", async () => {
        const payload = await requestJson("/api/pilot/approve", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({pending_id: item.pending_id})
        });
        setActionBanner("success", "Pilot action approved.", summarizePilotApproval(payload));
        document.getElementById("execution-output").textContent = JSON.stringify(payload, null, 2);
        await loadDashboard();
      });
      node.querySelector("#detail-reject").addEventListener("click", async () => {
        await requestJson("/api/pilot/reject", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({pending_id: item.pending_id, reason: "rejected from cockpit detail"})
        });
        setActionBanner("info", "Pilot action rejected.", "The pending pilot action was removed from the approval queue.");
        await loadDashboard();
      });
      if (rerender) renderPilotPending(currentPilotItems);
    }

    function selectPatchRun(runId, rerender = true) {
      selectedPatchRunId = Number(runId || 0);
      const run = currentPatchRuns.find((entry) => Number(entry.id) === selectedPatchRunId);
      const node = document.getElementById("patch-detail");
      if (!run) {
        renderEmptyDetail("patch-detail", "Patch detail", "Select a patch run to inspect diff preview, validations, and rollback hints.");
        if (rerender) renderPatchRuns(currentPatchRuns);
        return;
      }
      const summary = run.summary || {};
      const git = summary.git || {};
      const candidate = summary.candidate_evaluation || {};
      const validations = candidate.validation_failures || candidate.failures || [];
      const diffPreview = summary.preview_diff || summary.diff_preview || "No diff preview recorded.";
      node.innerHTML = `
        <div class="card">
          <div class="status-line"><span class="eyebrow">${run.suite_name}</span>${renderStatePill(run.status || "recorded")}</div>
          <div class="headline">${run.run_name}</div>
          <div class="body">${run.task_title || "No linked task title."}</div>
          <div class="status-line" style="margin-top:12px;">
            <span class="pill">run ${run.id}</span>
            <span>${git.status ? "git " + git.status : "no git metadata"}</span>
            <span>${git.rollback_hint || ""}</span>
          </div>
          <div class="actions" style="margin-top:12px;">
            <button id="detail-rollback" class="ghost"${git.rollback_ready ? "" : " disabled"}>Rollback this run</button>
          </div>
        </div>
        <div class="card">
          <div class="eyebrow">Changed files</div>
          <div class="body">${(run.changed_files || []).length ? run.changed_files.join(" · ") : "No changed files recorded."}</div>
        </div>
        <div class="card">
          <div class="eyebrow">Validation</div>
          <div class="body">${candidate.status || "No candidate validation summary recorded."}</div>
          <div class="body" style="margin-top:8px;">${validations.length ? validations.join(" | ") : "No validation failures recorded."}</div>
        </div>
        <div class="card">
          <div class="eyebrow">Diff preview</div>
          <div class="mono">${diffPreview}</div>
        </div>
      `;
      const rollbackButton = node.querySelector("#detail-rollback");
      if (rollbackButton && !rollbackButton.disabled) {
        rollbackButton.addEventListener("click", async () => {
          if (!window.confirm(`Rollback patch run ${run.id}?`)) return;
          const payload = await requestJson("/api/patch-rollback", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({run_id: run.id})
          });
          setActionBanner("success", "Patch rollback finished.", summarizePatchRollback(payload, run.id));
          document.getElementById("execution-output").textContent = JSON.stringify(payload, null, 2);
          await loadDashboard();
        });
      }
      if (rerender) renderPatchRuns(currentPatchRuns);
    }

    function selectActivity(activityId, rerender = true) {
      selectedActivityId = Number(activityId || 0);
      const item = currentActivityItems.find((entry) => Number(entry.id) === selectedActivityId);
      const node = document.getElementById("activity-detail");
      if (!item) {
        renderEmptyDetail("activity-detail", "Timeline detail", "Select a timeline item to inspect the recorded outcome and metadata.");
        if (rerender) renderActivity(currentActivityItems);
        return;
      }
      const meta = item.metadata || {};
      const related = currentActivityItems.filter((entry) => activityTopic(entry) === activityTopic(item));
      const relatedGroup = {topic: activityTopic(item), items: related};
      node.innerHTML = `
        <div class="card">
          <div class="status-line"><span class="eyebrow">${related.some(isPromptPilotActivity) ? "prompt workshop chain" : meta.tool_name || "tool"}</span>${renderStatePill(meta.status || "status")}</div>
          <div class="headline">${activityTopic(item)}</div>
          <div class="body">${related.some(isPromptPilotActivity) ? activityGroupSummary(relatedGroup) : activityNarrative(item)}</div>
        </div>
        <div class="card">
          <div class="eyebrow">Cause and effect</div>
          <div class="body">${related.length > 1 ? `This topic has ${related.length} related events.` : "This topic has one recorded event."}</div>
          <div class="body" style="margin-top:8px;">${activityGroupNarrative(relatedGroup)}</div>
        </div>
        <div class="card">
          <div class="eyebrow">Metadata</div>
          <div class="mono">${JSON.stringify(meta, null, 2)}</div>
        </div>
      `;
      if (rerender) renderActivity(currentActivityItems);
    }

    function renderPlan(snapshot) {
      const node = document.getElementById("plan-card");
      node.innerHTML = "";
      const rec = snapshot.plan.recommendation;
      const alternatives = (snapshot.plan.alternatives || []).slice(0, 3);
      const planQuery = snapshot.plan.query || document.getElementById("query").value.trim() || "what should I do next";
      if (!rec) {
        node.innerHTML = "<div class='card'><div class='headline'>No recommendation</div><div class='body'>Capture a fresh task or blocker first.</div></div>";
        return;
      }
      const reasons = (rec.reasons || []).map(reason => `<li>${reason}</li>`).join("");
      node.innerHTML = `
        <div class="card">
          <div class="eyebrow">Guided loop · recommended now</div>
          <div class="headline">[${rec.kind}] ${rec.title}</div>
          <div class="body">${rec.summary}</div>
          <div class="status-line" style="margin-top:12px;">
            <span class="pill">score ${Number(rec.score).toFixed(3)}</span>
            <span>${reasons ? "Review the reasons before executing." : "No extra planner reasons were recorded."}</span>
          </div>
          ${reasons ? `<ul style="margin-top:12px;">${reasons}</ul>` : ""}
          <div class="actions" style="margin-top:12px;">
            <button id="plan-run-recommendation">Run this step</button>
            <button id="plan-explain" class="ghost">Explain with model</button>
            <button id="plan-refresh" class="ghost">Refresh around this query</button>
          </div>
          <div class="body muted" style="margin-top:10px;">Quick loop: review the recommendation, run one bounded step, then inspect the updated recommendation and timeline below.</div>
        </div>
        <div class="card">
          <div class="eyebrow">Model-backed explanation</div>
          <div class="headline">${currentPlanExplanation?.text ? "Why this step is next" : "No explanation loaded yet"}</div>
          <div class="body">${currentPlanExplanation?.text || "Use Explain with model to get a bounded explanation of the current recommendation. Planner state remains the hard source of truth."}</div>
        </div>
        ${alternatives.length ? `
          <div class="card">
            <div class="eyebrow">Other workable options</div>
            <div style="display:grid; gap:12px; margin-top:8px;">
              ${alternatives.map((item) => `
                <div class="card" style="padding:12px; border-radius:16px;">
                  <div class="status-line"><span class="eyebrow">${item.kind}</span>${renderStatePill("ready")}</div>
                  <div class="headline">${item.title}</div>
                  <div class="body">${item.summary}</div>
                  <div class="actions" style="margin-top:10px;">
                    <button class="ghost" data-plan-run="${item.kind}::${item.title}">Run instead</button>
                  </div>
                </div>
              `).join("")}
            </div>
          </div>
        ` : ""}
        <div class="card">
          <div class="eyebrow">How to use this</div>
          <div class="body">Use <strong>Run this step</strong> for the planner's current top choice. If you want a different safe option, run one of the alternatives instead. After each step, check the work queue and operator timeline to see what changed.</div>
          <div class="body muted" style="margin-top:8px;">Prepare = preflight, Delegate = handoff, Run = execute the current bounded step, Verify = check that the last service change actually worked.</div>
        </div>
      `;
      const runRecommendation = document.getElementById("plan-run-recommendation");
      if (runRecommendation) {
        runRecommendation.addEventListener("click", async () => {
          await executePlanAction(rec, planQuery);
        });
      }
      const refresh = document.getElementById("plan-refresh");
      const explain = document.getElementById("plan-explain");
      if (refresh) {
        refresh.addEventListener("click", async () => {
          await loadDashboard();
        });
      }
      if (explain) {
        explain.addEventListener("click", async () => {
          const payload = await requestJson("/api/explain-plan", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
              query: planQuery,
              limit: 5
            })
          });
          currentPlanExplanation = payload;
          setActionBanner("info", "Planner explanation ready.", payload.text || "The current recommendation was explained.");
          document.getElementById("execution-output").textContent = JSON.stringify(payload, null, 2);
          renderPlan(snapshot);
        });
      }
      node.querySelectorAll("[data-plan-run]").forEach((button, index) => {
        button.addEventListener("click", async () => {
          const action = alternatives[index];
          if (!action) return;
          await executePlanAction(action, planQuery);
        });
      });
    }

    function persistWorkQueuePrefs() {
      window.localStorage.setItem("ernieCockpitPinnedWorkQueue", pinnedWorkQueueKey || "");
      window.localStorage.setItem("ernieCockpitDismissedWorkQueue", JSON.stringify(dismissedWorkQueueKeys || []));
    }

    function persistFocusedTask() {
      if (focusedTaskTitle) {
        window.localStorage.setItem("ernieCockpitFocusedTask", focusedTaskTitle);
      } else {
        window.localStorage.removeItem("ernieCockpitFocusedTask");
      }
    }

    function workQueueItemKey(kind, title) {
      return `${kind}::${title || "untitled"}`;
    }

    function renderWorkQueue(snapshot) {
      const node = document.getElementById("work-queue");
      const recommendation = snapshot.plan?.recommendation || null;
      const nextTask = (snapshot.open_tasks || [])[0] || null;
      const nextApproval = (snapshot.pending_pilot_turns || [])[0] || null;
      const latestActivity = (snapshot.recent_activity || [])[0] || null;
      const items = [];

      if (recommendation) {
        items.push({
          key: workQueueItemKey("recommendation", recommendation.title || "No title"),
          title: recommendation.title || "No title",
          kind: "recommendation",
          state: recommendation.kind || "action",
          body: recommendation.summary || "No planner summary recorded."
        });
      }

      if (nextTask) {
        const title = nextTask.metadata?.title || nextTask.content || "task";
        items.push({
          key: workQueueItemKey("task", title),
          title,
          kind: "task",
          state: nextTask.metadata?.status || "open",
          body: summarizeTaskCard(nextTask)
        });
      }

      if (nextApproval) {
        const action = nextApproval.selected_action || {};
        const approval = nextApproval.approval || {};
        items.push({
          key: workQueueItemKey("approval", action.title || "Pending pilot action"),
          title: action.title || "Pending pilot action",
          kind: "approval",
          state: approval.status || "pending",
          body: summarizePilotCard(nextApproval)
        });
      }

      if (latestActivity) {
        items.push({
          key: workQueueItemKey("activity", activityTopic(latestActivity)),
          title: activityTopic(latestActivity),
          kind: "activity",
          state: latestActivity.metadata?.status || "recorded",
          body: summarizeActivity(latestActivity)
        });
      }

      const visibleItems = items.filter((item) => !dismissedWorkQueueKeys.includes(item.key));
      if (pinnedWorkQueueKey) {
        visibleItems.sort((a, b) => {
          if (a.key === pinnedWorkQueueKey) return -1;
          if (b.key === pinnedWorkQueueKey) return 1;
          return 0;
        });
      }

      const cards = [];
      if (focusedTaskTitle) {
        cards.push(`
          <div class="card active">
            <div class="status-line"><span class="eyebrow">focused task mode</span>${renderStatePill("active")}</div>
            <div class="headline">${focusedTaskTitle}</div>
            <div class="body">Planner context and recommendation ranking are currently centered on this task title.</div>
            <div class="actions" style="margin-top:12px;">
              <button class="ghost" id="clear-focused-task">Clear focus</button>
            </div>
          </div>
        `);
      }

      cards.push(...visibleItems.map((item) => `
        <div class="card${item.key === pinnedWorkQueueKey ? " active" : ""}">
          <div class="status-line"><span class="eyebrow">${item.kind}</span>${renderStatePill(item.state)}</div>
          <div class="headline">${item.title}</div>
          <div class="body">${item.body}</div>
          <div class="actions" style="margin-top:12px;">
            ${item.kind === "task" ? `<button class="ghost" data-workqueue-focus="${item.title}">Focus this task</button>` : ""}
            <button class="ghost" data-workqueue-pin="${item.key}">${item.key === pinnedWorkQueueKey ? "Pinned" : "Pin here"}</button>
            <button class="ghost" data-workqueue-dismiss="${item.key}">Dismiss</button>
          </div>
        </div>
      `));

      if (!cards.length) {
        node.innerHTML = `
          <div class='card'>
            <div class='body muted'>No active recommendation, task, approval, or recent outcome is available yet.</div>
            <div class='actions' style='margin-top:12px;'>
              <button class='ghost' id='reset-work-queue'>Restore dismissed items</button>
            </div>
          </div>
        `;
        const reset = document.getElementById("reset-work-queue");
        if (reset) {
          reset.addEventListener("click", () => {
            dismissedWorkQueueKeys = [];
            pinnedWorkQueueKey = "";
            persistWorkQueuePrefs();
            renderWorkQueue(snapshot);
          });
        }
        return;
      }

      node.innerHTML = cards.join("");
      node.querySelectorAll("[data-workqueue-pin]").forEach((button) => {
        button.addEventListener("click", () => {
          pinnedWorkQueueKey = button.getAttribute("data-workqueue-pin") || "";
          persistWorkQueuePrefs();
          renderWorkQueue(snapshot);
        });
      });
      node.querySelectorAll("[data-workqueue-focus]").forEach((button) => {
        button.addEventListener("click", async () => {
          focusedTaskTitle = button.getAttribute("data-workqueue-focus") || "";
          persistFocusedTask();
          setActionBanner("info", "Focused task selected.", `Planner context is now centered on "${focusedTaskTitle}".`);
          await loadDashboard();
        });
      });
      node.querySelectorAll("[data-workqueue-dismiss]").forEach((button) => {
        button.addEventListener("click", () => {
          const key = button.getAttribute("data-workqueue-dismiss") || "";
          if (!key) return;
          if (!dismissedWorkQueueKeys.includes(key)) {
            dismissedWorkQueueKeys.push(key);
          }
          if (pinnedWorkQueueKey === key) {
            pinnedWorkQueueKey = "";
          }
          persistWorkQueuePrefs();
          renderWorkQueue(snapshot);
        });
      });
      const clearFocused = document.getElementById("clear-focused-task");
      if (clearFocused) {
        clearFocused.addEventListener("click", async () => {
          focusedTaskTitle = "";
          persistFocusedTask();
          setActionBanner("info", "Focused task cleared.", "Planner context returned to the free-form query field.");
          await loadDashboard();
        });
      }
    }

    function renderFocusedStack(snapshot) {
      const node = document.getElementById("focused-stack");
      if (!focusedTaskTitle) {
        node.innerHTML = "<div class='card'><div class='body muted'>No focused task is active. Use the task detail pane or work queue to focus one task.</div></div>";
        return;
      }

      const openTasks = snapshot.open_tasks || [];
      const approvals = snapshot.pending_pilot_turns || [];
      const patchRuns = snapshot.patch_runs || [];
      const recentActivity = snapshot.recent_activity || [];
      const recommendation = snapshot.plan?.recommendation || null;

      const task = openTasks.find((item) => matchesFocusedTask(item?.metadata?.title, item?.content)) || null;
      const approval = approvals.find((item) => matchesFocusedTask(
        item?.selected_action?.title,
        item?.selected_action?.summary,
        item?.approval?.prompt,
        item?.approval?.reason
      )) || null;
      const patch = patchRuns.find((item) => matchesFocusedTask(
        item?.task_title,
        item?.run_name,
        item?.summary?.task_title
      )) || null;
      const activity = recentActivity.find((item) => matchesFocusedTask(
        item?.metadata?.task_title,
        item?.subject,
        item?.content,
        activityTopic(item)
      )) || null;

      node.innerHTML = `
        <div class="card active">
          <div class="status-line"><span class="eyebrow">Focus anchor</span>${renderStatePill("active")}</div>
          <div class="headline">${focusedTaskTitle}</div>
          <div class="body">This stack is now ordered around the focused task.</div>
          <div class="actions" style="margin-top:12px;">
            <button class="ghost" id="focused-clear">Clear focus</button>
          </div>
        </div>
        <div class="card">
          <div class="eyebrow">Planner recommendation</div>
          <div class="headline">${recommendation?.title || "No recommendation available"}</div>
          <div class="body">${recommendation?.summary || "The planner has not surfaced a recommendation for this focus yet."}</div>
          <div class="body muted" style="margin-top:8px;">${recommendation ? explainPlannerAction(recommendation.kind) : "No focused planner action is available yet."}</div>
        </div>
        <div class="card">
          <div class="eyebrow">Focused task</div>
          <div class="headline">${task?.metadata?.title || task?.content || focusedTaskTitle}</div>
          <div class="body">${task ? summarizeTaskCard(task) : "The focused task is not in the current open-task list."}</div>
          <div class="actions" style="margin-top:12px;">
            <button class="ghost" id="focused-open-task"${task ? "" : " disabled"}>Open task detail</button>
            <button id="focused-run-recommendation"${recommendation && task && matchesFocusedTask(recommendation?.title, recommendation?.metadata?.target_task_title) ? "" : " disabled"}>Run focused recommendation</button>
          </div>
        </div>
        <div class="card">
          <div class="eyebrow">Related approval</div>
          <div class="headline">${approval?.selected_action?.title || "No related approval waiting"}</div>
          <div class="body">${approval ? summarizePilotCard(approval) : "No pending pilot approval currently matches this task focus."}</div>
          <div class="actions" style="margin-top:12px;">
            <button class="ghost" id="focused-open-approval"${approval ? "" : " disabled"}>Open approval review</button>
          </div>
        </div>
        <div class="card">
          <div class="eyebrow">Related patch run</div>
          <div class="headline">${patch?.run_name || "No related patch run"}</div>
          <div class="body">${patch ? summarizePatchCard(patch) : "No patch run currently matches this task focus."}</div>
          <div class="actions" style="margin-top:12px;">
            <button class="ghost" id="focused-open-patch"${patch ? "" : " disabled"}>Open patch review</button>
          </div>
        </div>
        <div class="card">
          <div class="eyebrow">Latest related outcome</div>
          <div class="headline">${activity ? activityTopic(activity) : "No related recent outcome"}</div>
          <div class="body">${activity ? summarizeActivity(activity) : "No recent timeline entry currently matches this task focus."}</div>
          <div class="actions" style="margin-top:12px;">
            <button class="ghost" id="focused-open-activity"${activity ? "" : " disabled"}>Open timeline detail</button>
          </div>
        </div>
      `;

      const clear = document.getElementById("focused-clear");
      if (clear) {
        clear.addEventListener("click", async () => {
          focusedTaskTitle = "";
          persistFocusedTask();
          setActionBanner("info", "Focused task cleared.", "Planner context returned to the free-form query field.");
          await loadDashboard();
        });
      }
      const openTask = document.getElementById("focused-open-task");
      if (openTask && task) {
        openTask.addEventListener("click", async () => {
          await loadTaskDetail(task.metadata?.title || task.content || focusedTaskTitle, task.subject || "");
        });
      }
      const runFocusedRecommendation = document.getElementById("focused-run-recommendation");
      if (runFocusedRecommendation && recommendation && task && matchesFocusedTask(recommendation?.title, recommendation?.metadata?.target_task_title)) {
        runFocusedRecommendation.addEventListener("click", async () => {
          await executePlanAction(recommendation, focusedTaskTitle);
        });
      }
      const openApproval = document.getElementById("focused-open-approval");
      if (openApproval && approval) {
        openApproval.addEventListener("click", () => {
          selectPilotPending(approval.pending_id);
        });
      }
      const openPatch = document.getElementById("focused-open-patch");
      if (openPatch && patch) {
        openPatch.addEventListener("click", () => {
          selectPatchRun(patch.id);
        });
      }
      const openActivity = document.getElementById("focused-open-activity");
      if (openActivity && activity) {
        openActivity.addEventListener("click", () => {
          selectActivity(activity.id);
        });
      }
    }

    async function loadHealth() {
      const payload = await requestJson("/health");
      const pill = document.getElementById("health-pill");
      const summary = document.getElementById("health-summary");
      pill.textContent = payload.status;
      summary.textContent =
        `${payload.service} on ${payload.host}:${payload.port}. Auth ${payload.auth_required ? "enabled" : "disabled"}.`;
      if (payload.session_auth_available && sessionId) {
        setAuthNote("Remote session is active in this browser.");
      } else if (payload.session_auth_available) {
        setAuthNote("Token auth is enabled. Save a browser session to stop relying on the URL token.");
      } else {
        setAuthNote("This cockpit does not require remote auth.");
      }
    }

    async function loadDashboard() {
      const queryInput = document.getElementById("query");
      const query = focusedTaskTitle || queryInput.value.trim() || "what should I do next";
      if (focusedTaskTitle) {
        queryInput.value = focusedTaskTitle;
      }
      const [snapshot, context, settings] = await Promise.all([
        requestJson("/api/snapshot?query=" + encodeURIComponent(query)),
        requestJson("/api/context?query=" + encodeURIComponent(query)),
        requestJson("/api/settings")
      ]);
      const stats = snapshot.stats;
      document.getElementById("service-summary").textContent =
        `${stats.events} events, ${stats.active_memories} active memories, ${stats.ready_tasks} ready tasks.`;
      document.getElementById("metrics").innerHTML =
        metricCard("events", stats.events) +
        metricCard("memories", stats.active_memories) +
        metricCard("ready", stats.ready_tasks) +
        metricCard("overdue", stats.overdue_tasks);
      renderSoul(settings);
      renderTutorial(snapshot, settings);
      renderWorkQueue(snapshot);
      renderFocusedStack(snapshot);
      renderPlan(snapshot);
      renderContext(context);
      taskList(document.getElementById("ready-tasks"), snapshot.ready_tasks);
      taskList(document.getElementById("overdue-tasks"), snapshot.overdue_tasks);
      renderOpenTasks(snapshot.open_tasks || []);
      renderNudges(snapshot.recent_nudges || []);
      renderPilotPending(snapshot.pending_pilot_turns || []);
      renderPatchRuns(snapshot.patch_runs || []);
      renderActivity(snapshot.recent_activity || []);
      renderOnboarding(settings);
      renderSettings(settings);
      renderPromptWorkshop();
    }

    async function runTaskAction(action, title, area) {
      let until = null;
      if (action === "snooze") {
        const next = new Date(Date.now() + 24 * 60 * 60 * 1000);
        until = next.toISOString().slice(0, 10);
      }
      const payload = await requestJson("/api/task-action", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({action, title, area, until})
      });
      setActionBanner("success", "Task updated.", taskActionSummary(action, title, payload));
      document.getElementById("execution-output").textContent = JSON.stringify(payload, null, 2);
      await loadDashboard();
      await loadTaskDetail(title, area);
    }

    async function observeNote() {
      const text = document.getElementById("note").value.trim();
      if (!text) return;
      const role = document.getElementById("role").value;
      const payload = await requestJson("/api/observe", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({role, text})
      });
      const summary = summarizeObserve(role, text);
      setActionBanner("success", "Note captured.", summary + " The planner and memory view were refreshed.");
      document.getElementById("observe-output").textContent = summary;
      document.getElementById("note").value = "";
      await loadDashboard();
    }

    async function runPromptWorkshop(mode) {
      const draft = document.getElementById("prompt-draft").value.trim();
      if (!draft) return;
      rememberPromptHistory(`Before ${workshopModeLabel(mode)} workshop`);
      const payload = await requestJson("/api/prompt-workshop", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({draft, mode})
      });
      currentPromptWorkshop = payload;
      setActionBanner("info", "Prompt workshop ready.", "This result is quarantined draft help only. It does not enter memory or planning until you explicitly promote it.");
      renderPromptWorkshop();
      document.getElementById("prompt-workshop-output").textContent = payload.text || JSON.stringify(payload, null, 2);
      document.getElementById("execution-output").textContent = JSON.stringify(payload, null, 2);
    }

    async function adoptPromptWorkshopResult() {
      if (!currentPromptWorkshop?.text) return;
      rememberPromptHistory("Before use result as draft");
      document.getElementById("prompt-draft").value = currentPromptWorkshop.text;
      setActionBanner("info", "Workshop result copied into draft.", "The rewritten text is still quarantined. Review it, edit it, or promote it explicitly when you are ready.");
      renderPromptWorkshop();
    }

    async function promotePromptText(source) {
      const text = source === "result"
        ? String(currentPromptWorkshop?.text || "").trim()
        : document.getElementById("prompt-draft").value.trim();
      if (!text) return;
      const payload = await requestJson("/api/prompt-promote", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({text, source})
      });
      const sourceLabel = source === "result" ? "workshop result" : "draft prompt";
      setActionBanner("success", "Prompt promoted to live note.", `The ${sourceLabel} is now stored as live input and can influence memory, planning, and later actions.`);
      document.getElementById("observe-output").textContent = summarizeObserve("user", text);
      document.getElementById("prompt-workshop-output").textContent =
        source === "result"
          ? "Workshop result promoted to live note."
          : "Draft prompt promoted to live note.";
      rememberPromptHistory(`Before promote ${source}`);
      document.getElementById("prompt-draft").value = "";
      currentPromptWorkshop = null;
      renderPromptWorkshop();
      document.getElementById("execution-output").textContent = JSON.stringify(payload, null, 2);
      await loadDashboard();
    }

    async function sendPromptToPilot(source) {
      const text = source === "result"
        ? String(currentPromptWorkshop?.text || "").trim()
        : document.getElementById("prompt-draft").value.trim();
      if (!text) return;
      const sourceLabel = source === "result" ? "workshop result" : "draft prompt";
      const confirmed = window.confirm(
        `Send this ${sourceLabel} into the guided loop for pilot review?\n\n` +
        "This promotes the text out of quarantine and queues the normal pilot preview path."
      );
      if (!confirmed) {
        return;
      }
      const payload = await requestJson("/api/prompt-pilot", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({text, source, limit: 5, use_model: false})
      });
      const approval = payload?.approval || {};
      const pendingId = payload?.pending_id || "";
      const detail = approval.status === "needs_approval"
        ? `It is now waiting in the pilot approval queue${pendingId ? ` as ${pendingId}` : ""}.`
        : approval.status === "auto_approved"
          ? "The pilot policy auto-approved the selected action."
          : "The pilot preview packet was refreshed.";
      setActionBanner("success", "Prompt sent to guided loop.", `${sourceLabel} moved out of quarantine. ${detail}`);
      document.getElementById("observe-output").textContent = summarizeObserve("user", text);
      document.getElementById("prompt-workshop-output").textContent =
        source === "result"
          ? "Workshop result sent to guided loop for pilot review."
          : "Draft prompt sent to guided loop for pilot review.";
      rememberPromptHistory(`Before send to guided loop ${source}`);
      document.getElementById("prompt-draft").value = "";
      currentPromptWorkshop = null;
      renderPromptWorkshop();
      document.getElementById("execution-output").textContent = JSON.stringify(payload, null, 2);
      await loadDashboard();
    }

    async function executeNext() {
      const query = document.getElementById("query").value.trim() || "what should I do next";
      const snapshot = await requestJson("/api/snapshot?query=" + encodeURIComponent(query));
      const rec = snapshot.plan.recommendation;
      const message = rec
        ? `Execute this action?\n\n[${rec.kind}] ${rec.title}\n\n${rec.summary}`
        : "No recommendation is available. Execute planner anyway?";
      if (!window.confirm(message)) {
        return;
      }
      const payload = await requestJson("/api/execute-next", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({query})
      });
      setActionBanner("success", "Planner action executed.", summarizeExecution(payload));
      document.getElementById("execution-output").textContent = JSON.stringify(payload, null, 2);
      await loadDashboard();
    }

    async function previewPilotTurn() {
      const text = document.getElementById("pilot-text").value.trim();
      if (!text) return;
      const payload = await requestJson("/api/pilot/preview", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({text, limit: 5, use_model: false})
      });
      const approval = payload?.approval || {};
      const title = approval.status === "needs_approval"
        ? "Pilot preview ready."
        : approval.status === "auto_approved"
          ? "Pilot action auto-approved."
          : "Pilot preview updated.";
      setActionBanner("info", title, summarizePilotPreview(payload));
      document.getElementById("execution-output").textContent = JSON.stringify(payload, null, 2);
      document.getElementById("pilot-text").value = "";
      await loadDashboard();
    }

    document.getElementById("refresh").addEventListener("click", loadDashboard);
    document.getElementById("observe").addEventListener("click", observeNote);
    document.getElementById("prompt-improve").addEventListener("click", async () => runPromptWorkshop("improve"));
    document.getElementById("prompt-clarify").addEventListener("click", async () => runPromptWorkshop("clarify"));
    document.getElementById("prompt-shorten").addEventListener("click", async () => runPromptWorkshop("shorten"));
    document.getElementById("prompt-safety-check").addEventListener("click", async () => runPromptWorkshop("safety_check"));
    document.getElementById("prompt-undo-history").addEventListener("click", undoPromptHistory);
    document.getElementById("prompt-replace-selection").addEventListener("click", replacePromptDraftSelection);
    document.getElementById("prompt-append-result").addEventListener("click", appendPromptWorkshopResult);
    document.getElementById("prompt-adopt-result").addEventListener("click", adoptPromptWorkshopResult);
    document.getElementById("prompt-pilot-result").addEventListener("click", async () => sendPromptToPilot("result"));
    document.getElementById("prompt-pilot").addEventListener("click", async () => sendPromptToPilot("draft"));
    document.getElementById("prompt-promote-result").addEventListener("click", async () => promotePromptText("result"));
    document.getElementById("prompt-promote").addEventListener("click", async () => promotePromptText("draft"));
    document.getElementById("execute").addEventListener("click", executeNext);
    document.getElementById("review").addEventListener("click", loadDashboard);
    document.getElementById("pilot-preview").addEventListener("click", previewPilotTurn);
    document.getElementById("save-session").addEventListener("click", async () => {
      const typed = document.getElementById("auth-token").value.trim();
      try {
        await saveSession(typed || currentAuthToken());
        await Promise.all([loadDashboard(), loadHealth()]);
      } catch (error) {
        setAuthNote(error.message);
      }
    });
    document.getElementById("login-button").addEventListener("click", async () => {
      const typed = document.getElementById("login-token").value.trim();
      try {
        await saveSession(typed || currentAuthToken());
        await Promise.all([loadDashboard(), loadHealth()]);
      } catch (error) {
        setLoginNote(error.message);
      }
    });
    (async () => {
      try {
        const health = await requestJson("/health");
        if (health.auth_required && !authToken && !sessionId && !currentAuthToken()) {
          setLoginRequired(true, "This cockpit requires a token before the dashboard can load.");
          await loadHealth();
          return;
        }
        if (authToken) {
          await saveSession(authToken);
        } else if (!sessionId && currentAuthToken()) {
          await saveSession(currentAuthToken());
        }
        await Promise.all([loadDashboard(), loadHealth()]);
      } catch (error) {
        setLoginRequired(true, "Enter the cockpit token to continue.");
        document.getElementById("service-summary").textContent = error.message;
        document.getElementById("health-summary").textContent = error.message;
        document.getElementById("health-pill").textContent = "error";
      }
    })().catch((error) => {
      document.getElementById("service-summary").textContent = error.message;
      document.getElementById("health-summary").textContent = error.message;
      document.getElementById("health-pill").textContent = "error";
    });
  </script>
</body>
</html>
"""


class CockpitService:
    def __init__(self, store: MemoryStore, *, workspace_root: Path | None = None):
        self.store = store
        self.workspace_root = workspace_root or Path.cwd()
        self.agent = MemoryFirstAgent(store, workspace_root=self.workspace_root)
        self.planner = MemoryPlanner(store)
        self.executor = MemoryExecutor(store)
        self.pilot_policy = LinuxPilotPolicy.load(workspace_root=self.workspace_root)
        self.pilot_runtime = LinuxPilotRuntime(store, policy=self.pilot_policy)
        self.patch_runner = WorkspacePatchRunner(
            store,
            workspace_root=self.workspace_root,
            git_mode=self.pilot_policy.git_write_mode,
        )
        self.pending_pilot_turns: dict[str, PilotTurnReport] = {}
        self.pending_pilot_contexts: dict[str, dict[str, Any]] = {}
        self.active_sessions: set[str] = set()
        self.service_manager = CockpitServiceManager()
        self.soul = load_soul_document(self.workspace_root)

    def snapshot(self, *, query: str = "what should I do next", limit: int = 5) -> dict[str, Any]:
        plan = self._planner().build_plan(query, action_limit=limit)
        return {
            "stats": self.store.stats(),
            "plan": self._plan_to_json(plan),
            "ready_tasks": [self._memory_to_json(task) for task in self.store.get_ready_tasks(limit=limit)],
            "overdue_tasks": [self._memory_to_json(task) for task in self.store.get_overdue_tasks(limit=limit)],
            "open_tasks": [self._memory_to_json(task) for task in self.store.get_open_tasks(limit=limit)],
            "recent_nudges": [self._memory_to_json(item) for item in self.store.get_recent_nudges(limit=limit)],
            "recent_activity": [self._memory_to_json(item) for item in self.store.get_recent_tool_outcomes(limit=limit)],
            "pending_pilot_turns": self.pending_pilot_queue(),
            "patch_runs": self.patch_runs(limit=limit),
        }

    def context(self, *, query: str, limit: int = 5) -> dict[str, Any]:
        context = self.store.build_context(query, memory_limit=limit)
        return self._jsonify(context)

    def explain_plan(self, *, query: str = "what should I do next", limit: int = 5) -> dict[str, Any]:
        report = self.agent.explain_plan(query, action_limit=limit)
        return self._jsonify(report)

    def observe(self, *, role: str, text: str) -> dict[str, Any]:
        report = self.agent.observe_message(role=role, text=text)
        return {
            "event_id": report.event_id,
            "stored_memories": [self._memory_to_json(item) for item in report.stored_memories],
            "plan": self._plan_to_json(report.plan) if report.plan is not None else None,
        }

    def prompt_workshop(self, *, draft: str, mode: str = "improve") -> dict[str, Any]:
        report = self.agent.workshop_prompt(draft, mode=mode)
        return self._jsonify(report)

    def promote_prompt_text(self, *, text: str, source: str = "draft") -> dict[str, Any]:
        report = self.agent.observe_message(role="user", text=text)
        self.store.record_tool_outcome(
            "prompt-promote",
            f"Promoted {source} prompt text into live input.",
            status="success",
            subject="prompt-workshop",
            tags=["prompt-workshop", "prompt-promote"],
            metadata={
                "prompt_source": source,
                "text_preview": text[:160],
            },
        )
        return {
            "prompt_source": source,
            "event_id": report.event_id,
            "stored_memories": [self._memory_to_json(item) for item in report.stored_memories],
            "plan": self._plan_to_json(report.plan) if report.plan is not None else None,
        }

    def send_prompt_to_pilot(
        self,
        *,
        text: str,
        source: str = "draft",
        action_limit: int = 5,
        use_model: bool = False,
    ) -> dict[str, Any]:
        payload = self.preview_pilot_turn(
            text=text,
            action_limit=action_limit,
            use_model=use_model,
        )
        pending_id = str(payload.get("pending_id") or "").strip()
        if pending_id:
            context = {
                "request_origin": "prompt_workshop",
                "prompt_source": source,
            }
            self.pending_pilot_contexts[pending_id] = context
            payload.update(context)
        approval = payload.get("approval") or {}
        selected_action = payload.get("selected_action") or {}
        self.store.record_tool_outcome(
            "prompt-pilot-bridge",
            f"Sent {source} prompt text into the guided loop.",
            status="success",
            subject="prompt-workshop",
            tags=["prompt-workshop", "prompt-pilot"],
            metadata={
                "prompt_source": source,
                "pending_id": payload.get("pending_id"),
                "approval_status": approval.get("status"),
                "selected_action_title": selected_action.get("title"),
                "text_preview": text[:160],
            },
        )
        payload["prompt_source"] = source
        return payload

    def execute_next(self, *, query: str = "what should I do next", limit: int = 5) -> dict[str, Any]:
        plan_query = query.strip() or "what should I do next"
        planner = self._planner()
        before_plan = planner.build_plan(plan_query, action_limit=limit)
        cycle = ExecutionCycle(
            query=plan_query,
            before_plan=before_plan,
            result=self.executor.execute_action(before_plan.recommendation),
            after_plan=planner.build_plan(plan_query, action_limit=limit),
        )
        payload = self._execution_cycle_to_json(cycle)
        payload["model_explanation"] = self._jsonify(
            self.agent.narrate_execution(
                query=plan_query,
                before_plan=cycle.before_plan,
                result=cycle.result,
                after_plan=cycle.after_plan,
            )
        )
        return payload

    def execute_plan_action(
        self,
        *,
        query: str = "what should I do next",
        kind: str,
        title: str,
        task_id: int | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        plan_query = query.strip() or "what should I do next"
        planner = self._planner()
        before_plan = planner.build_plan(plan_query, action_limit=limit)
        selected_action = self._select_plan_action(
            before_plan,
            kind=kind,
            title=title,
            task_id=task_id,
        )
        if selected_action is None:
            raise KeyError(f"Unknown plan action: {kind}:{title}")
        cycle = ExecutionCycle(
            query=plan_query,
            before_plan=before_plan,
            result=self.executor.execute_action(selected_action),
            after_plan=planner.build_plan(plan_query, action_limit=limit),
        )
        if self._is_demo_action(selected_action):
            self._record_demo_event(
                event="run",
                outcome=(
                    f"Ran demo action [{selected_action.kind}] {selected_action.title} "
                    f"with {cycle.result.status} status."
                ),
                titles=[selected_action.title],
                metadata={
                    "plan_action_kind": selected_action.kind,
                    "execution_status": cycle.result.status,
                    "execution_summary": cycle.result.summary,
                },
            )
        payload = self._execution_cycle_to_json(cycle)
        payload["model_explanation"] = self._jsonify(
            self.agent.narrate_execution(
                query=plan_query,
                before_plan=cycle.before_plan,
                result=cycle.result,
                after_plan=cycle.after_plan,
            )
        )
        payload["selected_action"] = self._jsonify(selected_action)
        payload["selection_source"] = "explicit_plan_action"
        return payload

    def recent_nudges(self, *, limit: int = 5) -> list[dict[str, Any]]:
        return [self._memory_to_json(item) for item in self.store.get_recent_nudges(limit=limit)]

    def pending_pilot_queue(self) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for pending_id, report in self.pending_pilot_turns.items():
            item = self._pilot_turn_to_json(report)
            item["pending_id"] = pending_id
            item.update(self.pending_pilot_contexts.get(pending_id, {}))
            payload.append(item)
        payload.sort(key=lambda item: int(item.get("user_event_id", 0) or 0), reverse=True)
        return payload

    def preview_pilot_turn(
        self,
        *,
        text: str,
        action_limit: int = 5,
        use_model: bool = False,
    ) -> dict[str, Any]:
        report = self.pilot_runtime.run_turn(
            text,
            approve=False,
            use_model=use_model,
            action_limit=action_limit,
        )
        pending_id: str | None = None
        if (
            report.approval is not None
            and report.approval.status == "needs_approval"
            and report.execution_result is None
            and report.selected_action is not None
        ):
            pending_id = uuid.uuid4().hex[:10]
            self.pending_pilot_turns[pending_id] = report
        payload = self._pilot_turn_to_json(report)
        payload["pending_id"] = pending_id
        return payload

    def approve_pilot_turn(self, *, pending_id: str) -> dict[str, Any]:
        report = self.pending_pilot_turns.get(pending_id)
        if report is None:
            raise KeyError(f"Unknown pending pilot turn: {pending_id}")
        approved = self.pilot_runtime.approve_turn(report)
        self.pending_pilot_turns.pop(pending_id, None)
        context = self.pending_pilot_contexts.pop(pending_id, {})
        payload = self._pilot_turn_to_json(approved)
        payload["pending_id"] = pending_id
        payload.update(context)
        if context.get("request_origin") == "prompt_workshop":
            selected_action = payload.get("selected_action") or {}
            execution_result = payload.get("execution_result") or {}
            patch_run = execution_result.get("patch_run") or {}
            rejection_reason = (
                patch_run.get("rejection_reason")
                or (execution_result.get("approval") or {}).get("metadata", {}).get("rejection_reason")
                or ""
            )
            prompt_source = str(context.get("prompt_source") or "draft")
            self.store.record_tool_outcome(
                "prompt-pilot-execution",
                (
                    f"Executed guided-loop action from Prompt Workshop {prompt_source}: "
                    f"{selected_action.get('title') or 'untitled action'}."
                ),
                status=str(execution_result.get("status") or "success"),
                subject="prompt-workshop",
                tags=["prompt-workshop", "prompt-pilot", "pilot-execution"],
                metadata={
                    "prompt_source": prompt_source,
                    "request_origin": "prompt_workshop",
                    "pending_id": pending_id,
                    "selected_action_title": selected_action.get("title"),
                    "selected_action_kind": selected_action.get("kind"),
                    "execution_status": execution_result.get("status"),
                    "patch_status": patch_run.get("status"),
                    "rejection_reason": rejection_reason,
                    "task_title": (
                        (execution_result.get("related_task") or {}).get("metadata", {}).get("title")
                    ),
                },
            )
        return payload

    def reject_pilot_turn(self, *, pending_id: str, reason: str | None = None) -> dict[str, Any]:
        report = self.pending_pilot_turns.pop(pending_id, None)
        context = self.pending_pilot_contexts.pop(pending_id, {})
        if report is None:
            raise KeyError(f"Unknown pending pilot turn: {pending_id}")
        payload = {
            "pending_id": pending_id,
            "status": "rejected",
            "reason": str(reason or "rejected_by_user").strip() or "rejected_by_user",
            "selected_action": (
                self._jsonify(report.selected_action)
                if report.selected_action is not None
                else None
            ),
        }
        payload.update(context)
        return payload

    def patch_runs(self, *, limit: int = 5) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        for offset in range(max(1, limit)):
            item = self.store.latest_patch_run(offset=offset)
            if item is None:
                break
            runs.append(self._jsonify(item))
        return runs

    def rollback_patch_run(self, *, run_id: int, force: bool = False) -> dict[str, Any]:
        report = self.patch_runner.rollback(run_id, force=force)
        return self._jsonify(report)

    def task_action(
        self,
        *,
        action: str,
        title: str,
        area: str = "execution",
        until: str | None = None,
    ) -> dict[str, Any]:
        if action == "complete":
            return self._jsonify(self.store.complete_task(title, area=area))
        if action == "snooze":
            if not until:
                raise ValueError("missing_until")
            return self._jsonify(self.store.snooze_task(title, until=until, area=area))
        if action == "resume":
            return self._memory_to_json(self.store.resume_task(title, area=area))
        if action == "unblock":
            return self._memory_to_json(self.store.unblock_task(title, area=area))
        raise ValueError(f"unsupported_task_action:{action}")

    def seed_demo_workflow(self) -> dict[str, Any]:
        demo_specs = self._demo_workflow_specs()
        created_titles: list[str] = []
        unchanged_titles: list[str] = []
        for spec in demo_specs:
            title = str(spec["title"])
            existing = self.store.find_active_task(title, area=str(spec["area"]))
            if existing is not None:
                unchanged_titles.append(title)
                continue
            created = self.store.record_task(
                title,
                status=str(spec.get("status", "open")),
                area=str(spec.get("area", "execution")),
                details=str(spec.get("details") or ""),
                service_inspection=str(spec.get("service_inspection") or ""),
                service_label=str(spec.get("service_label") or ""),
                complete_on_success=bool(spec.get("complete_on_success", False)),
                tags=list(spec.get("tags") or []),
            )
            created_titles.append(str(created.metadata.get("title") or created.content))
        self._record_demo_event(
            event="seed",
            outcome=(
                f"Seeded demo workflow with {len(created_titles)} new task(s) and "
                f"{len(unchanged_titles)} existing task(s)."
            ),
            titles=created_titles or unchanged_titles,
            metadata={
                "created_titles": created_titles,
                "unchanged_titles": unchanged_titles,
            },
        )
        return {
            "created_titles": created_titles,
            "unchanged_titles": unchanged_titles,
            "open_tasks": [
                str(task.metadata.get("title") or task.content)
                for task in self.store.get_open_tasks(limit=8)
            ],
        }

    def reset_demo_workflow(self) -> dict[str, Any]:
        reset_titles: list[str] = []
        for spec in self._demo_workflow_specs():
            restored = self.store.record_task(
                str(spec["title"]),
                status=str(spec.get("status", "open")),
                area=str(spec.get("area", "execution")),
                details=str(spec.get("details") or ""),
                service_inspection=str(spec.get("service_inspection") or ""),
                service_label=str(spec.get("service_label") or ""),
                complete_on_success=bool(spec.get("complete_on_success", False)),
                tags=list(spec.get("tags") or []),
            )
            reset_titles.append(str(restored.metadata.get("title") or restored.content))
        self._record_demo_event(
            event="reset",
            outcome=f"Reset demo workflow to its starting state for {len(reset_titles)} task(s).",
            titles=reset_titles,
        )
        return {
            "reset_titles": reset_titles,
            "open_tasks": [
                str(task.metadata.get("title") or task.content)
                for task in self.store.get_open_tasks(limit=8)
            ],
        }

    def task_detail(self, *, title: str, area: str | None = None) -> dict[str, Any]:
        task = self.store.find_active_task(title, area=area, decorate=True)
        if task is None:
            raise KeyError(f"Unknown active task: {title}")
        memory_entities = self.store.get_memory_entities(task.id)
        primary_task_entity = next(
            (link.entity for link in memory_entities if link.entity.entity_type == "task"),
            None,
        )
        return {
            "task": self._memory_to_json(task),
            "sources": self._jsonify(self.store.get_memory_sources(task.id)),
            "edges": self._jsonify(self.store.get_memory_edges(task.id)),
            "entities": self._jsonify(memory_entities),
            "entity_edges": (
                self._jsonify(self.store.get_entity_edges(primary_task_entity.id, direction="outgoing"))
                if primary_task_entity is not None
                else []
            ),
            "recent_nudges": [
                self._memory_to_json(item)
                for item in self.store.get_recent_nudges(limit=8)
                if str(item.metadata.get("task_title") or "").strip().lower() == title.strip().lower()
            ],
            "task_plan_actions": [
                self._jsonify(item)
                for item in self._task_plan_actions(task, limit=6)
            ],
        }

    def recent_activity(self, *, limit: int = 8) -> list[dict[str, Any]]:
        return [
            self._memory_to_json(item)
            for item in self.store.get_recent_tool_outcomes(limit=limit)
        ]

    def settings(self) -> dict[str, Any]:
        payload = self.service_manager.settings()
        payload["service_sync"] = self.store.sync_service_tasks(payload)
        payload["service_sync_status"] = self._planner().service_sync_status(payload)
        payload["pilot_policy"] = self._pilot_policy_settings()
        payload["soul"] = self._soul_settings()
        payload["soul_audit"] = self._soul_audit(limit=8)
        return self._jsonify(payload)

    def rotate_remote_access_token(self) -> dict[str, Any]:
        return self._jsonify(self.service_manager.rotate_remote_token())

    def service_action(self, action: str) -> dict[str, Any]:
        payload = self.service_manager.perform_action(action)
        settings = payload.get("settings")
        if isinstance(settings, dict):
            payload["service_sync"] = self.store.sync_service_tasks(settings)
            payload["service_sync_status"] = self._planner().service_sync_status(settings)
        return self._jsonify(payload)

    def soul_review(self, *, limit: int = 8) -> dict[str, Any]:
        review = review_soul_document(
            self.soul,
            self._recent_user_messages(limit=limit),
            dismissed_evidence_signatures=self._dismissed_soul_proposal_signatures(limit=32),
        )
        self.store.record_tool_outcome(
            "soul-review",
            f"Generated {len(review.proposals)} soul amendment proposal(s).",
            status="success",
            subject="identity",
            tags=["soul", "review"],
            metadata={
                "proposal_ids": [proposal.proposal_id for proposal in review.proposals],
                "proposal_count": len(review.proposals),
            },
        )
        return self._jsonify(review)

    def apply_soul_amendment(self, *, proposal_id: str, limit: int = 8) -> dict[str, Any]:
        review = review_soul_document(
            self.soul,
            self._recent_user_messages(limit=limit),
            dismissed_evidence_signatures=self._dismissed_soul_proposal_signatures(limit=32),
        )
        proposal = next(
            (item for item in review.proposals if item.proposal_id == proposal_id),
            None,
        )
        if proposal is None:
            raise KeyError(f"Unknown soul proposal: {proposal_id}")
        self.soul = apply_soul_proposal(self.soul, proposal)
        self.agent.soul = self.soul
        updated_review = review_soul_document(
            self.soul,
            self._recent_user_messages(limit=limit),
            dismissed_evidence_signatures=self._dismissed_soul_proposal_signatures(limit=32),
        )
        self.store.record_tool_outcome(
            "soul-apply",
            f"Applied soul amendment '{proposal.title}'.",
            status="success",
            subject="identity",
            tags=["soul", "apply"],
            metadata={
                "proposal_id": proposal.proposal_id,
                "section": proposal.section,
                "amendment": proposal.amendment,
                "evidence_signature": proposal.evidence_signature,
            },
        )
        return {
            "applied": self._jsonify(proposal),
            "review": self._jsonify(updated_review),
            "summary": f"Applied '{proposal.title}' to {self.soul.path.name}.",
        }

    def dismiss_soul_amendment(self, *, proposal_id: str, limit: int = 8) -> dict[str, Any]:
        review = review_soul_document(
            self.soul,
            self._recent_user_messages(limit=limit),
            dismissed_evidence_signatures=self._dismissed_soul_proposal_signatures(limit=32),
        )
        proposal = next(
            (item for item in review.proposals if item.proposal_id == proposal_id),
            None,
        )
        if proposal is None:
            raise KeyError(f"Unknown soul proposal: {proposal_id}")
        self.store.record_tool_outcome(
            "soul-dismiss",
            f"Dismissed soul amendment '{proposal.title}'.",
            status="dismissed",
            subject="identity",
            tags=["soul", "dismiss"],
            metadata={
                "proposal_id": proposal.proposal_id,
                "section": proposal.section,
                "amendment": proposal.amendment,
                "evidence_signature": proposal.evidence_signature,
            },
        )
        updated_review = review_soul_document(
            self.soul,
            self._recent_user_messages(limit=limit),
            dismissed_evidence_signatures=self._dismissed_soul_proposal_signatures(limit=32),
        )
        return {
            "dismissed": self._jsonify(proposal),
            "review": self._jsonify(updated_review),
            "summary": f"Dismissed '{proposal.title}' for now. It will stay suppressed until the soul or evidence changes.",
        }

    def update_pilot_policy(
        self,
        *,
        trusted_writes_enabled: bool | None = None,
        trusted_write_operations: list[str] | None = None,
        trusted_write_required_successes: int | None = None,
        service_sync_suppression_window_seconds: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        policy = LinuxPilotPolicy.load(workspace_root=self.workspace_root)
        valid_operations = {"write_text", "replace_text", "append_text"}
        current_operations = set(policy.trusted_auto_approve_file_operations)
        operations = current_operations
        if trusted_write_operations is not None:
            operations = {
                str(item).strip()
                for item in trusted_write_operations
                if str(item).strip()
            }
            unknown = sorted(item for item in operations if item not in valid_operations)
            if unknown:
                raise ValueError(f"unsupported_trusted_write_operations:{','.join(unknown)}")
            operations = {item for item in operations if item in valid_operations}
        if trusted_writes_enabled is False:
            operations = set()
        if trusted_writes_enabled is True and not operations:
            operations = set(
                LinuxPilotPolicy.default(
                    workspace_root=self.workspace_root
                ).trusted_auto_approve_file_operations
            )
        required_successes = policy.trusted_auto_approve_required_successes
        if trusted_write_required_successes is not None:
            if trusted_write_required_successes < 1 or trusted_write_required_successes > 10:
                raise ValueError("trusted_write_required_successes_out_of_range")
            required_successes = int(trusted_write_required_successes)
        suppression_windows = dict(policy.service_sync_suppression_window_seconds)
        if service_sync_suppression_window_seconds is not None:
            valid_scopes = {"local_service", "remote_service", "desktop_launcher"}
            normalized_windows: dict[str, int] = {}
            for scope, raw_value in service_sync_suppression_window_seconds.items():
                normalized_scope = str(scope).strip().lower()
                if normalized_scope not in valid_scopes:
                    raise ValueError(f"unsupported_service_sync_scope:{normalized_scope}")
                try:
                    seconds = int(raw_value)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"service_sync_suppression_window_seconds_must_be_int:{normalized_scope}"
                    ) from None
                if seconds < 0 or seconds > 86400:
                    raise ValueError(
                        f"service_sync_suppression_window_seconds_out_of_range:{normalized_scope}"
                    )
                normalized_windows[normalized_scope] = seconds
            suppression_windows.update(normalized_windows)
        policy.trusted_auto_approve_file_operations = set(sorted(operations))
        policy.trusted_auto_approve_required_successes = required_successes
        policy.service_sync_suppression_window_seconds = suppression_windows
        written_to = policy.write()
        self._reload_pilot_runtime()
        return {
            "written_to": str(written_to),
            "pilot_policy": self._pilot_policy_settings(),
        }

    def create_session(self, *, token: str) -> dict[str, Any]:
        expected = token.strip()
        configured = self._configured_token()
        if not configured or not secrets.compare_digest(expected, configured):
            raise ValueError("invalid_token")
        session_id = uuid.uuid4().hex
        self.active_sessions.add(session_id)
        return {"session_id": session_id}

    def _configured_token(self) -> str:
        return str(getattr(self, "_configured_auth_token", "") or "")

    def _select_plan_action(
        self,
        snapshot: PlannerSnapshot,
        *,
        kind: str,
        title: str,
        task_id: int | None = None,
    ) -> PlannerAction | None:
        normalized_kind = str(kind or "").strip().lower()
        normalized_title = str(title or "").strip().lower()
        for action in [snapshot.recommendation, *snapshot.alternatives]:
            if action is None:
                continue
            if str(action.kind or "").strip().lower() != normalized_kind:
                continue
            if str(action.title or "").strip().lower() != normalized_title:
                continue
            if task_id is not None and int(action.task_id or 0) != int(task_id):
                continue
            return action
        return None

    def _task_plan_actions(
        self,
        task: MemoryRecord,
        *,
        limit: int = 6,
    ) -> list[PlannerAction]:
        task_title = str(task.metadata.get("title") or task.content).strip()
        if not task_title:
            return []
        snapshot = self._planner().build_plan(task_title, action_limit=max(limit, 6))
        matches: list[PlannerAction] = []
        seen: set[tuple[str, str, int | None]] = set()
        for action in [snapshot.recommendation, *snapshot.alternatives]:
            if action is None or not self._action_matches_task(action, task):
                continue
            key = (action.kind, action.title, action.task_id)
            if key in seen:
                continue
            seen.add(key)
            matches.append(action)
        return matches[:limit]

    def _action_matches_task(self, action: PlannerAction, task: MemoryRecord) -> bool:
        task_title = str(task.metadata.get("title") or task.content).strip().lower()
        target_task_title = str(action.metadata.get("target_task_title") or "").strip().lower()
        if action.task_id is not None and int(action.task_id) == int(task.id):
            return True
        if target_task_title and target_task_title == task_title:
            return True
        if str(action.title or "").strip().lower() == task_title:
            return True
        return False

    def _demo_workflow_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "title": "Try cockpit guided loop",
                "status": "open",
                "area": "execution",
                "details": (
                    "Use Next action or task detail to run one bounded step from the cockpit. "
                    "This is a safe starter task for your first hands-on walkthrough."
                ),
                "tags": ["demo"],
            },
            {
                "title": "Inspect local cockpit service status",
                "status": "open",
                "area": "execution",
                "details": (
                    "This demo task uses a read-only service inspection so you can test a visible "
                    "execution path without changing system state."
                ),
                "service_inspection": "restart_local_service",
                "service_label": "Inspect local service status",
                "complete_on_success": True,
                "tags": ["service-verification", "demo"],
            },
        ]

    def _demo_workflow_titles(self) -> set[str]:
        return {
            str(spec.get("title") or "").strip().lower()
            for spec in self._demo_workflow_specs()
            if str(spec.get("title") or "").strip()
        }

    def _is_demo_action(self, action: PlannerAction) -> bool:
        demo_titles = self._demo_workflow_titles()
        candidate_titles = {
            str(action.title or "").strip().lower(),
            str(action.metadata.get("target_task_title") or "").strip().lower(),
            str(action.metadata.get("task_title") or "").strip().lower(),
        }
        candidate_titles.discard("")
        return bool(candidate_titles & demo_titles)

    def _record_demo_event(
        self,
        *,
        event: str,
        outcome: str,
        titles: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        demo_titles = [str(title).strip() for title in (titles or []) if str(title).strip()]
        self.store.record_tool_outcome(
            "demo-workflow",
            outcome,
            status="success",
            subject="demo walkthrough",
            tags=["demo", "walkthrough"],
            metadata={
                "demo_event": str(event).strip() or "event",
                "demo_titles": demo_titles,
                **(metadata or {}),
            },
        )

    def _reload_pilot_runtime(self) -> None:
        self.pilot_policy = LinuxPilotPolicy.load(workspace_root=self.workspace_root)
        self.pilot_runtime = LinuxPilotRuntime(self.store, policy=self.pilot_policy)
        self.patch_runner = WorkspacePatchRunner(
            self.store,
            workspace_root=self.workspace_root,
            git_mode=self.pilot_policy.git_write_mode,
        )

    def _planner(self) -> MemoryPlanner:
        return MemoryPlanner(
            self.store,
            service_manager=self.service_manager,
            service_sync_suppression_window_seconds=(
                self.pilot_policy.service_sync_suppression_window_seconds
            ),
        )

    def _pilot_policy_settings(self) -> dict[str, Any]:
        policy_status = self.pilot_policy.status()
        operations = list(policy_status.get("trusted_auto_approve_file_operations") or [])
        history_report = PilotHistoryReporter(self.store).build(limit=12)
        trusted_write_recommendations = [
            {
                "key": str(item.get("key") or ""),
                "label": str(item.get("label") or ""),
                "count": int(item.get("count") or 0),
                "file_operation": str(item.get("file_operation") or ""),
                "file_path": str(item.get("file_path") or ""),
            }
            for item in history_report.recurring_patterns
            if str(item.get("kind") or "") == "trusted_write_candidate"
        ]
        return {
            **policy_status,
            "policy_path": str(self.pilot_policy.resolved_policy_path()),
            "trusted_writes_enabled": bool(operations)
            and int(policy_status.get("trusted_auto_approve_required_successes") or 0) > 0,
            "trusted_write_supported_operations": [
                "append_text",
                "replace_text",
                "write_text",
            ],
            "service_sync_supported_scopes": [
                "desktop_launcher",
                "local_service",
                "remote_service",
            ],
            "trusted_write_recommendations": trusted_write_recommendations,
        }

    def _soul_settings(self) -> dict[str, Any]:
        return self.soul.ui_summary()

    def _recent_user_messages(self, *, limit: int) -> list[str]:
        return [
            event.content
            for event in self.store.recent_events(limit=max(limit * 2, 8))
            if str(event.role).strip().lower() == "user"
        ][-limit:]

    def _dismissed_soul_proposal_signatures(self, *, limit: int) -> list[str]:
        dismissed: list[str] = []
        for item in self.store.get_recent_tool_outcomes(limit=max(limit * 3, 12)):
            tool_name = str(item.metadata.get("tool_name") or "").strip()
            signature = str(item.metadata.get("evidence_signature") or "").strip()
            if tool_name == "soul-dismiss" and signature:
                dismissed.append(signature)
            if tool_name == "soul-apply" and signature:
                dismissed = [entry for entry in dismissed if entry != signature]
        return list(dict.fromkeys(dismissed))

    def _soul_audit(self, *, limit: int) -> list[dict[str, Any]]:
        audit: list[dict[str, Any]] = []
        for item in self.store.get_recent_tool_outcomes(limit=max(limit * 3, 12)):
            tool_name = str(item.metadata.get("tool_name") or "").strip()
            if tool_name not in {"soul-review", "soul-apply", "soul-dismiss"}:
                continue
            audit.append(
                {
                    "id": item.id,
                    "tool_name": tool_name,
                    "status": str(item.metadata.get("status") or ""),
                    "outcome": str(item.metadata.get("outcome") or item.content),
                    "proposal_id": str(item.metadata.get("proposal_id") or ""),
                    "evidence_signature": str(item.metadata.get("evidence_signature") or ""),
                    "created_at": item.created_at,
                }
            )
            if len(audit) >= limit:
                break
        return audit

    def _execution_cycle_to_json(self, cycle: ExecutionCycle) -> dict[str, Any]:
        return {
            "query": cycle.query,
            "before_plan": self._plan_to_json(cycle.before_plan),
            "result": self._jsonify(cycle.result),
            "after_plan": self._plan_to_json(cycle.after_plan),
        }

    def _pilot_turn_to_json(self, report: PilotTurnReport) -> dict[str, Any]:
        return {
            "user_event_id": report.user_event_id,
            "user_text": report.user_text,
            "action_limit_used": report.action_limit_used,
            "context": report.context_render,
            "plan": self._plan_to_json(report.plan),
            "policy_status": report.policy_status,
            "model_status": report.model_status,
            "selected_action": self._jsonify(report.selected_action) if report.selected_action is not None else None,
            "selected_action_source": report.selected_action_source,
            "approval": self._jsonify(report.approval) if report.approval is not None else None,
            "execution_result": self._jsonify(report.execution_result) if report.execution_result is not None else None,
            "after_plan": self._plan_to_json(report.after_plan) if report.after_plan is not None else None,
            "assistant_event_id": report.assistant_event_id,
            "assistant_message": report.assistant_message,
            "trace_path": report.trace_path,
            "error": report.error,
        }

    def _plan_to_json(self, snapshot: PlannerSnapshot) -> dict[str, Any]:
        return self._jsonify(snapshot)

    def _memory_to_json(self, memory: Any) -> dict[str, Any]:
        return self._jsonify(memory)

    def _jsonify(self, value: Any) -> Any:
        if is_dataclass(value):
            return {key: self._jsonify(item) for key, item in asdict(value).items()}
        if isinstance(value, dict):
            return {str(key): self._jsonify(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonify(item) for item in value]
        return value


class CockpitHTTPServer(HTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        store: MemoryStore,
        workspace_root: Path | None = None,
        auth_token: str | None = None,
    ):
        super().__init__(server_address, handler_class)
        self.cockpit_service = CockpitService(store, workspace_root=workspace_root)
        self.auth_token = str(auth_token or "").strip()
        self.cockpit_service._configured_auth_token = self.auth_token


class CockpitRequestHandler(BaseHTTPRequestHandler):
    server: CockpitHTTPServer

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/health", "/api/health"}:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(HTTPStatus.NOT_FOUND)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        if parsed.path in {"/health", "/api/health"}:
            self._send_json(self._health_payload())
            return
        if parsed.path == "/":
            self._send_html(COCKPIT_HTML)
            return
        if not self._authorized(params):
            self._send_error_json(HTTPStatus.UNAUTHORIZED, "unauthorized")
            return
        if parsed.path == "/api/snapshot":
            query = self._first(params, "query", "what should I do next")
            limit = self._int_value(params, "limit", 5)
            self._send_json(self.server.cockpit_service.snapshot(query=query, limit=limit))
            return
        if parsed.path == "/api/context":
            query = self._first(params, "query", "what should I do next")
            limit = self._int_value(params, "limit", 5)
            self._send_json(self.server.cockpit_service.context(query=query, limit=limit))
            return
        if parsed.path == "/api/explain-plan":
            self._send_error_json(HTTPStatus.METHOD_NOT_ALLOWED, "post_only")
            return
        if parsed.path == "/api/prompt-workshop":
            self._send_error_json(HTTPStatus.METHOD_NOT_ALLOWED, "post_only")
            return
        if parsed.path == "/api/prompt-promote":
            self._send_error_json(HTTPStatus.METHOD_NOT_ALLOWED, "post_only")
            return
        if parsed.path == "/api/prompt-pilot":
            self._send_error_json(HTTPStatus.METHOD_NOT_ALLOWED, "post_only")
            return
        if parsed.path == "/api/recent-nudges":
            limit = self._int_value(params, "limit", 5)
            self._send_json(self.server.cockpit_service.recent_nudges(limit=limit))
            return
        if parsed.path == "/api/recent-activity":
            limit = self._int_value(params, "limit", 8)
            self._send_json(self.server.cockpit_service.recent_activity(limit=limit))
            return
        if parsed.path == "/api/pilot/pending":
            self._send_json(self.server.cockpit_service.pending_pilot_queue())
            return
        if parsed.path == "/api/patch-runs":
            limit = self._int_value(params, "limit", 5)
            self._send_json(self.server.cockpit_service.patch_runs(limit=limit))
            return
        if parsed.path == "/api/settings":
            self._send_json(self.server.cockpit_service.settings())
            return
        if parsed.path == "/api/task-detail":
            title = self._first(params, "title", "").strip()
            area = self._first(params, "area", "").strip() or None
            if not title:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "missing_title")
                return
            try:
                self._send_json(self.server.cockpit_service.task_detail(title=title, area=area))
            except KeyError:
                self._send_error_json(HTTPStatus.NOT_FOUND, "task_not_found")
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, "not_found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        if not self._authorized(params):
            self._send_error_json(HTTPStatus.UNAUTHORIZED, "unauthorized")
            return
        payload = self._read_json_body()
        if parsed.path == "/api/observe":
            text = str(payload.get("text", "")).strip()
            role = str(payload.get("role", "user")).strip() or "user"
            if not text:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "missing_text")
                return
            self._send_json(self.server.cockpit_service.observe(role=role, text=text))
            return
        if parsed.path == "/api/prompt-workshop":
            draft = str(payload.get("draft", "")).strip()
            mode = str(payload.get("mode", "improve")).strip() or "improve"
            self._send_json(self.server.cockpit_service.prompt_workshop(draft=draft, mode=mode))
            return
        if parsed.path == "/api/prompt-promote":
            text = str(payload.get("text", "")).strip()
            source = str(payload.get("source", "draft")).strip() or "draft"
            if not text:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "missing_text")
                return
            self._send_json(self.server.cockpit_service.promote_prompt_text(text=text, source=source))
            return
        if parsed.path == "/api/prompt-pilot":
            text = str(payload.get("text", "")).strip()
            source = str(payload.get("source", "draft")).strip() or "draft"
            if not text:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "missing_text")
                return
            action_limit = int(payload.get("limit", 5) or 5)
            use_model = bool(payload.get("use_model", False))
            self._send_json(
                self.server.cockpit_service.send_prompt_to_pilot(
                    text=text,
                    source=source,
                    action_limit=action_limit,
                    use_model=use_model,
                )
            )
            return
        if parsed.path == "/api/session":
            token = str(payload.get("token", "")).strip()
            if not token:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "missing_token")
                return
            try:
                self._send_json(self.server.cockpit_service.create_session(token=token))
            except ValueError:
                self._send_error_json(HTTPStatus.UNAUTHORIZED, "invalid_token")
            return
        if parsed.path == "/api/execute-next":
            query = str(payload.get("query", "what should I do next")).strip() or "what should I do next"
            limit = int(payload.get("limit", 5) or 5)
            self._send_json(self.server.cockpit_service.execute_next(query=query, limit=limit))
            return
        if parsed.path == "/api/explain-plan":
            query = str(payload.get("query", "what should I do next")).strip() or "what should I do next"
            limit = int(payload.get("limit", 5) or 5)
            self._send_json(self.server.cockpit_service.explain_plan(query=query, limit=limit))
            return
        if parsed.path == "/api/execute-plan-action":
            query = str(payload.get("query", "what should I do next")).strip() or "what should I do next"
            kind = str(payload.get("kind", "")).strip()
            title = str(payload.get("title", "")).strip()
            if not kind or not title:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "missing_plan_action")
                return
            raw_task_id = payload.get("task_id")
            task_id = None
            if raw_task_id not in {None, ""}:
                try:
                    task_id = int(raw_task_id)
                except (TypeError, ValueError):
                    self._send_error_json(HTTPStatus.BAD_REQUEST, "task_id_must_be_int")
                    return
            limit = int(payload.get("limit", 5) or 5)
            try:
                self._send_json(
                    self.server.cockpit_service.execute_plan_action(
                        query=query,
                        kind=kind,
                        title=title,
                        task_id=task_id,
                        limit=limit,
                    )
                )
            except KeyError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        if parsed.path == "/api/pilot/preview":
            text = str(payload.get("text", "")).strip()
            if not text:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "missing_text")
                return
            action_limit = int(payload.get("limit", 5) or 5)
            use_model = bool(payload.get("use_model", False))
            self._send_json(
                self.server.cockpit_service.preview_pilot_turn(
                    text=text,
                    action_limit=action_limit,
                    use_model=use_model,
                )
            )
            return
        if parsed.path == "/api/pilot/approve":
            pending_id = str(payload.get("pending_id", "")).strip()
            if not pending_id:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "missing_pending_id")
                return
            try:
                self._send_json(self.server.cockpit_service.approve_pilot_turn(pending_id=pending_id))
            except KeyError:
                self._send_error_json(HTTPStatus.NOT_FOUND, "pending_turn_not_found")
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        if parsed.path == "/api/pilot/reject":
            pending_id = str(payload.get("pending_id", "")).strip()
            reason = str(payload.get("reason", "")).strip() or None
            if not pending_id:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "missing_pending_id")
                return
            try:
                self._send_json(
                    self.server.cockpit_service.reject_pilot_turn(
                        pending_id=pending_id,
                        reason=reason,
                    )
                )
            except KeyError:
                self._send_error_json(HTTPStatus.NOT_FOUND, "pending_turn_not_found")
            return
        if parsed.path == "/api/patch-rollback":
            run_id = int(payload.get("run_id", 0) or 0)
            if run_id <= 0:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "missing_run_id")
                return
            force = bool(payload.get("force", False))
            self._send_json(
                self.server.cockpit_service.rollback_patch_run(
                    run_id=run_id,
                    force=force,
                )
            )
            return
        if parsed.path == "/api/task-action":
            action = str(payload.get("action", "")).strip()
            title = str(payload.get("title", "")).strip()
            area = str(payload.get("area", "execution")).strip() or "execution"
            until = str(payload.get("until", "")).strip() or None
            if not action or not title:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "missing_task_action_fields")
                return
            try:
                self._send_json(
                    self.server.cockpit_service.task_action(
                        action=action,
                        title=title,
                        area=area,
                        until=until,
                    )
                )
            except KeyError:
                self._send_error_json(HTTPStatus.NOT_FOUND, "task_not_found")
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        if parsed.path == "/api/demo/seed":
            self._send_json(self.server.cockpit_service.seed_demo_workflow())
            return
        if parsed.path == "/api/demo/reset":
            self._send_json(self.server.cockpit_service.reset_demo_workflow())
            return
        if parsed.path == "/api/soul/review":
            self._send_json(self.server.cockpit_service.soul_review())
            return
        if parsed.path == "/api/soul/apply":
            proposal_id = str(payload.get("proposal_id", "")).strip()
            if not proposal_id:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "missing_proposal_id")
                return
            try:
                self._send_json(
                    self.server.cockpit_service.apply_soul_amendment(
                        proposal_id=proposal_id,
                    )
                )
            except KeyError:
                self._send_error_json(HTTPStatus.NOT_FOUND, "soul_proposal_not_found")
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        if parsed.path == "/api/soul/dismiss":
            proposal_id = str(payload.get("proposal_id", "")).strip()
            if not proposal_id:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "missing_proposal_id")
                return
            try:
                self._send_json(
                    self.server.cockpit_service.dismiss_soul_amendment(
                        proposal_id=proposal_id,
                    )
                )
            except KeyError:
                self._send_error_json(HTTPStatus.NOT_FOUND, "soul_proposal_not_found")
            return
        if parsed.path == "/api/settings/rotate-remote-token":
            try:
                self._send_json(self.server.cockpit_service.rotate_remote_access_token())
            except FileNotFoundError:
                self._send_error_json(HTTPStatus.NOT_FOUND, "remote_access_not_configured")
            except RuntimeError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        if parsed.path == "/api/settings/action":
            action = str(payload.get("action", "")).strip()
            if not action:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "missing_action")
                return
            try:
                self._send_json(self.server.cockpit_service.service_action(action))
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            except RuntimeError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        if parsed.path == "/api/settings/pilot-policy":
            trusted_writes_enabled = payload.get("trusted_writes_enabled")
            if trusted_writes_enabled is not None:
                trusted_writes_enabled = bool(trusted_writes_enabled)
            raw_operations = payload.get("trusted_write_operations")
            trusted_write_operations = None
            if raw_operations is not None:
                if not isinstance(raw_operations, list):
                    self._send_error_json(
                        HTTPStatus.BAD_REQUEST,
                        "trusted_write_operations_must_be_list",
                    )
                    return
                trusted_write_operations = [str(item) for item in raw_operations]
            raw_threshold = payload.get("trusted_write_required_successes")
            trusted_write_required_successes = None
            if raw_threshold is not None:
                try:
                    trusted_write_required_successes = int(raw_threshold)
                except (TypeError, ValueError):
                    self._send_error_json(
                        HTTPStatus.BAD_REQUEST,
                        "trusted_write_required_successes_must_be_int",
                    )
                    return
            raw_suppression_windows = payload.get("service_sync_suppression_window_seconds")
            service_sync_suppression_window_seconds = None
            if raw_suppression_windows is not None:
                if not isinstance(raw_suppression_windows, dict):
                    self._send_error_json(
                        HTTPStatus.BAD_REQUEST,
                        "service_sync_suppression_window_seconds_must_be_object",
                    )
                    return
                service_sync_suppression_window_seconds = {}
                for scope, raw_value in raw_suppression_windows.items():
                    try:
                        service_sync_suppression_window_seconds[str(scope)] = int(raw_value)
                    except (TypeError, ValueError):
                        self._send_error_json(
                            HTTPStatus.BAD_REQUEST,
                            f"service_sync_suppression_window_seconds_must_be_int:{scope}",
                        )
                        return
            try:
                self._send_json(
                    self.server.cockpit_service.update_pilot_policy(
                        trusted_writes_enabled=trusted_writes_enabled,
                        trusted_write_operations=trusted_write_operations,
                        trusted_write_required_successes=trusted_write_required_successes,
                        service_sync_suppression_window_seconds=(
                            service_sync_suppression_window_seconds
                        ),
                    )
                )
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, "not_found")

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length > 0 else b"{}"
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _send_html(self, html: str) -> None:
        encoded = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_error_json(self, status: HTTPStatus, error: str) -> None:
        self._send_json({"error": error}, status=status)

    def _health_payload(self) -> dict[str, Any]:
        host, port = self.server.server_address
        return {
            "status": "ok",
            "service": "ernie-cockpit",
            "host": host,
            "port": port,
            "auth_required": bool(self.server.auth_token),
            "session_auth_available": bool(self.server.auth_token),
        }

    def _first(self, params: dict[str, list[str]], key: str, default: str) -> str:
        values = params.get(key, [])
        return values[0] if values else default

    def _authorized(self, params: dict[str, list[str]]) -> bool:
        if not self.server.auth_token:
            return True
        header_token = str(self.headers.get("X-Ernie-Token", "")).strip()
        query_token = self._first(params, "token", "").strip()
        session_id = str(self.headers.get("X-Ernie-Session", "")).strip()
        if session_id and session_id in self.server.cockpit_service.active_sessions:
            return True
        supplied = header_token or query_token
        return bool(supplied) and secrets.compare_digest(supplied, self.server.auth_token)

    def _int_value(self, params: dict[str, list[str]], key: str, default: int) -> int:
        raw = self._first(params, key, str(default))
        try:
            return max(1, int(raw))
        except ValueError:
            return default


def serve_cockpit(
    store: MemoryStore,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    workspace_root: Path | None = None,
    auth_token: str | None = None,
    access_url: str | None = None,
) -> None:
    server = CockpitHTTPServer(
        (host, port),
        CockpitRequestHandler,
        store=store,
        workspace_root=workspace_root,
        auth_token=auth_token,
    )
    try:
        print(f"Ernie cockpit listening on http://{host}:{port}")
        if access_url:
            print(f"Open cockpit: {access_url}")
        if auth_token:
            print("Remote API auth is enabled.")
            print(f"Access token: {auth_token}")
            if not access_url:
                print("Open the cockpit in a browser, then paste the access token into the login prompt.")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down cockpit.")
    finally:
        server.server_close()
