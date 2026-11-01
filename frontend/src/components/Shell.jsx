import React, { useEffect, useState } from "react";
import {
  LayoutGrid, Box, Activity, Cpu, Code2, BookOpen, Settings as Cog, Github, Plus, Search,
  PanelLeftClose, PanelLeftOpen, ChevronsUpDown, LogOut, Sun, Moon, User, Users, Menu as Burger,
  Play, Pause, Check,
} from "lucide-react";
import { Mark } from "./Logo.jsx";
import Menu from "./Menu.jsx";
import { BRAND, GITHUB_URL } from "../lib/brand.js";
import { getTheme, toggleTheme } from "../lib/theme.js";
import { listWorkspaces, switchWorkspace } from "../lib/account.js";
import { navigate } from "../lib/router.js";

const NAV = [
  { path: "/", label: "Overview", icon: LayoutGrid },
  { path: "/models", label: "Models", icon: Box },
  { path: "/studio", label: "Studio", icon: Activity },
  { path: "/compute", label: "Compute", icon: Cpu },
  { path: "/api", label: "API", icon: Code2 },
];
const RAIL_KEY = "fiddlehead.sidebar";

export const initials = (n = "") =>
  n.trim().split(/\s+/).slice(0, 2).map((w) => w[0]).join("").toUpperCase() || "?";
const fmt = (n) => (n == null ? "—" : Number(n).toLocaleString());

// A tiny live loss trace for the sidebar's run card.
function Spark({ values }) {
  if (!values || values.length < 2) return <svg className="spark" viewBox="0 0 100 24" aria-hidden="true" />;
  const lo = Math.min(...values), hi = Math.max(...values), span = hi - lo || 1;
  const pts = values.map((v, i) =>
    `${((i / (values.length - 1)) * 100).toFixed(1)},${(2 + (1 - (v - lo) / span) * 20).toFixed(1)}`).join(" ");
  return (
    <svg className="spark" viewBox="0 0 100 24" preserveAspectRatio="none" aria-hidden="true">
      <polyline points={pts} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

export default function Shell({ route, user, engine, run, onNewModel, onSearch, onSignOut, onEngineHelp, onRunToggle, children }) {
  const [railPref, setRailPref] = useState(() => { try { return localStorage.getItem(RAIL_KEY) === "rail"; } catch { return false; } });
  const [drawer, setDrawer] = useState(false);
  const [theme, setThemeState] = useState(getTheme());
  const [spaces, setSpaces] = useState([]);
  const top = "/" + (route.split("/")[1] || "");
  const inStudio = top === "/studio";
  const rail = railPref || inStudio;       // the studio wants every pixel for its instruments

  useEffect(() => { setDrawer(false); }, [route]);
  useEffect(() => { listWorkspaces().then(setSpaces).catch(() => {}); }, []);

  function toggleRail() {
    const next = !railPref;
    setRailPref(next);
    try { localStorage.setItem(RAIL_KEY, next ? "rail" : "full"); } catch {}
  }

  const org = user?.org || {};
  const engineLabel = engine.online ? "Engine online" : engine.online === false ? "Engine offline" : "Checking engine…";
  const caps = engine.caps;

  const navLink = ({ path, label, icon: Icon }) => {
    const active = path === top;
    const live = path === "/studio" && run?.training;
    return (
      <a key={path} href={`#${path}`} className="sb-item" aria-current={active ? "page" : undefined} title={rail ? label : undefined}>
        <Icon size={17} strokeWidth={1.75} aria-hidden="true" />
        <span className="sb-label">{label}</span>
        {live && <span className="sb-live" aria-label="training now"><span className="live-dot" />{!rail && "Live"}</span>}
      </a>
    );
  };

  return (
    <div className={`shell${rail ? " rail" : ""}${drawer ? " drawer-open" : ""}`}>
      <a className="skip" href="#main" onClick={(e) => { e.preventDefault(); document.getElementById("main")?.focus(); }}>Skip to content</a>

      <aside className="sb" aria-label="Sidebar">
        <div className="sb-top">
          <a className="sb-brand" href="#/" aria-label={`${BRAND} overview`}>
            <Mark size={24} />
            <span className="sb-word">{BRAND.toLowerCase()}</span>
          </a>
          {!inStudio && (
            <button type="button" className="btn btn-ghost btn-icon btn-sm sb-collapse" onClick={toggleRail}
                    aria-label={rail ? "Expand sidebar" : "Collapse sidebar"} title={rail ? "Expand sidebar" : "Collapse sidebar"}>
              {rail ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
            </button>
          )}
        </div>

        <Menu label="Workspace" align="start" className="sb-ws-wrap" trigger={({ props }) => (
          <button type="button" className="sb-ws" {...props} title={rail ? org.name : undefined}>
            <span className="sb-ws-av">{initials(org.name || user?.name)}</span>
            <span className="sb-ws-n">{org.name || "Workspace"}</span>
            <ChevronsUpDown size={14} className="sb-ws-c" aria-hidden="true" />
          </button>
        )} items={[
          { heading: "Workspaces" },
          ...(spaces.length ? spaces : [{ id: org.id, name: org.name, current: true }]).map((w) => ({
            label: w.name, icon: w.current ? Check : null,
            onSelect: () => { if (!w.current) switchWorkspace(w.id).then(() => window.location.reload()); },
          })),
          "-",
          { label: "Workspace settings", icon: Cog, onSelect: () => navigate("/settings/workspace") },
          { label: "Invite teammates", icon: Users, onSelect: () => navigate("/settings/members") },
        ]} />

        <button type="button" className="btn btn-primary sb-new" onClick={onNewModel} title={rail ? "New model" : undefined}>
          <Plus size={16} aria-hidden="true" /><span className="sb-label">New model</span>
        </button>
        <button type="button" className="sb-search" onClick={onSearch} title={rail ? "Search (⌘K)" : undefined}>
          <Search size={15} aria-hidden="true" /><span className="sb-label">Search</span><kbd className="kbd sb-label">⌘K</kbd>
        </button>

        <nav className="sb-nav" aria-label="Main">{NAV.map(navLink)}</nav>

        {run && (
          <div className={`sb-run${run.training ? " on" : ""}`}>
            <a className="sb-run-main" href="#/studio" title={rail ? `${run.name} · step ${fmt(run.step)}` : undefined}>
              <span className="sb-run-top">
                <span className={`status-dot ${run.training ? "ok pulse" : "idle"}`} />
                <span className="sb-label">{run.training ? "Training" : run.label}</span>
                <span className="sb-run-step sb-label">step {fmt(run.step)}</span>
              </span>
              <span className="sb-run-name sb-label">{run.name}</span>
              <span className="sb-label"><Spark values={run.losses} /></span>
            </a>
            {run.canToggle && (
              <button type="button" className="btn btn-icon btn-sm sb-run-btn" onClick={onRunToggle}
                      aria-label={run.training ? "Pause training" : "Resume training"} title={run.training ? "Pause" : "Resume"}>
                {run.training ? <Pause size={14} /> : <Play size={14} />}
              </button>
            )}
          </div>
        )}

        <div className="sb-fill" />

        <nav className="sb-nav" aria-label="Resources">
          {navLink({ path: "/docs", label: "Docs", icon: BookOpen })}
          {GITHUB_URL && (
            <a className="sb-item" href={GITHUB_URL} target="_blank" rel="noreferrer" title={rail ? "GitHub" : undefined}>
              <Github size={17} strokeWidth={1.75} aria-hidden="true" /><span className="sb-label">GitHub</span>
            </a>
          )}
          {navLink({ path: "/settings", label: "Settings", icon: Cog })}
        </nav>

        <button type="button" className={`sb-engine ${engine.online ? "ok" : engine.online === false ? "off" : ""}`}
                onClick={onEngineHelp} title={rail ? engineLabel : undefined}>
          <span className={`status-dot ${engine.online ? "ok" : engine.online === false ? "off" : "idle"}`} />
          <span className="sb-engine-text sb-label">
            <span className="sb-engine-t">{engineLabel}</span>
            <span className="sb-engine-s">
              {engine.online ? (caps ? `${caps.cpu_cores} cores · ${caps.accelerator}` : "ws://localhost:8765")
                : engine.online === false ? "Click for how to start it" : " "}
            </span>
          </span>
        </button>

        <Menu label="Account" align="start" side="top" className="sb-user-wrap" trigger={({ props }) => (
          <button type="button" className="sb-user" {...props} title={rail ? user?.name : undefined}>
            <span className="avatar">{initials(user?.name)}</span>
            <span className="sb-user-id sb-label">
              <span className="sb-user-n">{user?.name}</span>
              <span className="sb-user-e">{user?.email}</span>
            </span>
            <ChevronsUpDown size={14} className="sb-label" aria-hidden="true" />
          </button>
        )} items={[
          { label: "Account settings", icon: User, onSelect: () => navigate("/settings/profile") },
          { label: theme === "dark" ? "Light theme" : "Dark theme", icon: theme === "dark" ? Sun : Moon,
            onSelect: () => setThemeState(toggleTheme()) },
          "-",
          { label: "Sign out", icon: LogOut, onSelect: onSignOut },
        ]} />
      </aside>

      <div className="sb-scrim" onClick={() => setDrawer(false)} aria-hidden="true" />

      <div className="main-col">
        <header className="mbar">
          <button type="button" className="btn btn-ghost btn-icon" onClick={() => setDrawer(true)} aria-label="Open menu">
            <Burger size={18} />
          </button>
          <a className="mbar-brand" href="#/"><Mark size={22} /><span>{BRAND.toLowerCase()}</span></a>
          <button type="button" className="btn btn-ghost btn-icon" onClick={onSearch} aria-label="Search"><Search size={17} /></button>
          <button type="button" className="btn btn-primary btn-icon" onClick={onNewModel} aria-label="New model"><Plus size={17} /></button>
        </header>
        <main className={`main${inStudio ? " main-studio" : ""}`} id="main" tabIndex={-1}>{children}</main>
      </div>
    </div>
  );
}
