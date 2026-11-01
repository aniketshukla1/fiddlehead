import React, { useEffect, useState } from "react";
import { KeyRound, Plus, Trash2, BookOpen, Radio, ShieldCheck, Box } from "lucide-react";
import { listKeys, createKey, revokeKey, listModels, getUsage } from "../lib/account.js";
import { apiBase } from "../lib/api.js";
import CodeBlock, { CopyField } from "./CodeBlock.jsx";

const fmtDate = (d) => (d ? new Date(d).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" }) : "—");

function snippet(lang, model, key) {
  const url = `${apiBase}/v1/${model}/generate`;
  if (lang === "python") return `import requests

resp = requests.post(
    "${url}",
    headers={"Authorization": "Bearer ${key}"},
    json={"prompt": "Once upon a time", "max_new_tokens": 80},
)
resp.raise_for_status()
print(resp.json()["text"])`;
  if (lang === "javascript") return `const res = await fetch("${url}", {
  method: "POST",
  headers: {
    "Authorization": "Bearer ${key}",
    "Content-Type": "application/json",
  },
  body: JSON.stringify({ prompt: "Once upon a time", max_new_tokens: 80 }),
});
const { text } = await res.json();`;
  return `curl ${url} \\
  -H "Authorization: Bearer ${key}" \\
  -H "Content-Type: application/json" \\
  -d '{"prompt": "Once upon a time", "max_new_tokens": 80}'`;
}

export default function ApiPage({ onNotice }) {
  const [keys, setKeys] = useState(null);
  const [models, setModels] = useState([]);
  const [usage, setUsage] = useState(null);
  const [fresh, setFresh] = useState(null);     // the full key, shown once
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [lang, setLang] = useState("curl");
  const [model, setModel] = useState("");

  useEffect(() => {
    listKeys().then(setKeys).catch(() => setKeys([]));
    listModels().then((ms) => { setModels(ms); const l = ms.find((m) => m.deployed); if (l) setModel(l.name); }).catch(() => {});
    getUsage().then(setUsage).catch(() => {});
  }, []);

  const live = models.filter((m) => m.deployed);
  const target = model || live[0]?.name || "my-model";

  async function add(e) {
    e.preventDefault();
    setBusy(true);
    try {
      const k = await createKey(name.trim() || "default");
      setFresh(k.key);
      setKeys((ks) => [k, ...(ks || [])]);
      setName("");
    } catch (err) { onNotice?.(err.message); } finally { setBusy(false); }
  }
  async function kill(k) {
    if (!window.confirm(`Revoke “${k.name}”? Anything using it stops working immediately.`)) return;
    try { await revokeKey(k.id); setKeys((ks) => ks.filter((x) => x.id !== k.id)); onNotice?.(`Revoked ${k.name}`); }
    catch (err) { onNotice?.(err.message); }
  }

  return (
    <div className="page">
      <header className="ph">
        <div>
          <h1>API</h1>
          <p>Serve your models over HTTP. Create a key, deploy a model, then send it a prompt from anywhere.</p>
        </div>
        <div className="ph-actions">
          <a className="btn" href="#/docs"><BookOpen size={15} />API reference</a>
        </div>
      </header>

      <div className="api-grid">
        <section className="ui-card" aria-labelledby="qs-t">
          <div className="ui-card-h">
            <div>
              <h2 id="qs-t" className="ui-card-t">Quickstart</h2>
              <p className="ui-card-d">Generate text with a deployed model.</p>
            </div>
            <div className="seg-ctl" role="tablist" aria-label="Language">
              {[["curl", "cURL"], ["python", "Python"], ["javascript", "JavaScript"]].map(([id, label]) => (
                <button key={id} type="button" role="tab" aria-selected={lang === id} className={lang === id ? "on" : ""} onClick={() => setLang(id)}>{label}</button>
              ))}
            </div>
          </div>
          <div className="ui-card-b stack">
            <div className="field-row">
              <label className="field">
                <span className="field-l">Model</span>
                <select className="select" value={target} onChange={(e) => setModel(e.target.value)} disabled={!live.length}>
                  {live.length ? live.map((m) => <option key={m.id} value={m.name}>{m.name}</option>) : <option>my-model</option>}
                </select>
              </label>
              <div className="field">
                <span className="field-l">Base URL</span>
                <CopyField value={apiBase} />
              </div>
            </div>
            {!live.length && (
              <div className="note">
                <Box size={15} aria-hidden="true" />
                <span>No model is deployed yet. Switch one on in <a className="btn-link" href="#/models">Models</a>, then come back — this example fills in its name.</span>
              </div>
            )}
            <CodeBlock code={snippet(lang, target, fresh || "sk-YOUR_KEY")} lang={lang === "curl" ? "shell" : lang} />
          </div>
        </section>

        <div className="stack">
          <section className="ui-card" aria-labelledby="ep-t">
            <div className="ui-card-h"><h2 id="ep-t" className="ui-card-t">Live endpoints</h2><span className="badge num">{live.length}</span></div>
            {live.length ? (
              <div className="list">
                {live.map((m) => (
                  <div key={m.id} className="list-row">
                    <span className="status-dot ok" />
                    <div className="grow">
                      <div className="cell-t">{m.name}</div>
                      <div className="cell-s mono">POST /v1/{m.name}/generate</div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="ov-mini-empty"><Radio size={16} aria-hidden="true" /><span>Nothing is being served.</span></div>
            )}
          </section>
          <section className="ui-card tile">
            <div className="tile-l"><ShieldCheck size={14} aria-hidden="true" />API calls</div>
            <div className="tile-v">{usage ? Math.round(usage.infer_units || 0).toLocaleString() : "—"}</div>
            <div className="tile-s">metered per request, all keys</div>
          </section>
        </div>
      </div>

      <section className="ui-card" style={{ marginTop: 16 }} aria-labelledby="keys-t">
        <div className="ui-card-h">
          <div>
            <h2 id="keys-t" className="ui-card-t">API keys</h2>
            <p className="ui-card-d">Keys belong to this workspace. Each is shown once — store it somewhere safe.</p>
          </div>
        </div>
        <div className="ui-card-b stack">
          <form className="row" onSubmit={add}>
            <input className="input grow" style={{ maxWidth: 360 }} placeholder="Key name, e.g. production" value={name}
                   onChange={(e) => setName(e.target.value)} aria-label="New key name" />
            <button type="submit" className="btn btn-primary" disabled={busy}><Plus size={15} />{busy ? "Creating…" : "Create key"}</button>
          </form>
          {fresh && (
            <div className="note ok api-fresh">
              <KeyRound size={16} aria-hidden="true" />
              <div className="grow stack" style={{ gap: 8 }}>
                <span><b>Copy your new key now.</b> It won't be shown again.</span>
                <CopyField value={fresh} />
              </div>
            </div>
          )}
        </div>
        {keys === null ? null : keys.length === 0 ? (
          <div className="ov-mini-empty" style={{ borderTop: "1px solid var(--line)" }}><KeyRound size={16} aria-hidden="true" /><span>No keys yet.</span></div>
        ) : (
          <div className="tbl-wrap">
            <table className="tbl keys-tbl">
              <thead><tr><th>Name</th><th>Key</th><th>Created</th><th>Last used</th><th className="ta-r"><span className="sr-only">Actions</span></th></tr></thead>
              <tbody>
                {keys.map((k) => (
                  <tr key={k.id}>
                    <td><span className="cell-t">{k.name}</span></td>
                    <td><code className="mono subtle" style={{ fontSize: 12.5 }}>{k.prefix}…</code></td>
                    <td className="muted">{fmtDate(k.created_at)}</td>
                    <td className="muted">{k.last_used_at ? fmtDate(k.last_used_at) : "Never"}</td>
                    <td><div className="cell-actions">
                      <button type="button" className="btn btn-ghost btn-sm btn-danger" onClick={() => kill(k)}><Trash2 size={13} />Revoke</button>
                    </div></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
