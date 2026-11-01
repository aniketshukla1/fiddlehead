import React, { useState } from "react";
import { ArrowRight } from "lucide-react";
import ThemeToggle from "./ThemeToggle.jsx";
import Logo from "./Logo.jsx";
import CodeBlock from "./CodeBlock.jsx";
import { apiBase } from "../lib/api.js";

// Developer reference for the hosted-inference API. The product's whole promise
// is "deploy as an API", so this documents the real endpoint (see routers/infer.py)
// with copy-pasteable examples that point at this deployment's base URL.

const BASE = apiBase;

const SECTIONS = [
  { id: "quickstart", label: "Quickstart" },
  { id: "auth", label: "Authentication" },
  { id: "generate", label: "Generate" },
  { id: "params", label: "Parameters" },
  { id: "errors", label: "Errors" },
  { id: "limits", label: "Rate limits" },
];

const PARAMS = [
  ["prompt", "string", '""', "The text your model continues."],
  ["max_new_tokens", "integer", "80", "How many tokens to generate (1–1024)."],
  ["temperature", "float", "0.8", "Higher = more random. 0 is greedy/deterministic."],
  ["top_k", "integer", "40", "Sample only from the k most-likely tokens. 0 disables."],
  ["top_p", "float", "1.0", "Nucleus sampling: keep the smallest set of tokens with cumulative probability ≥ p."],
];

const ERRORS = [
  ["401", "Missing or invalid API key.", "Send Authorization: Bearer sk-… with a key from the API page."],
  ["404", "No model by that name in your workspace.", "Check the model name; it must belong to the key's workspace."],
  ["409", "The model exists but isn't deployed.", "Switch its endpoint on in Models."],
  ["429", "Too many requests.", "Back off and retry after the Retry-After header."],
  ["501", "Hosted inference for this modality isn't available yet.", "Speech/vision models can be trained & exported today; hosted endpoints are text-first."],
];

const curl = `curl ${BASE}/v1/my-model/generate \\
  -H "Authorization: Bearer sk-YOUR_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "Once upon a time",
    "max_new_tokens": 80,
    "temperature": 0.8
  }'`;

const python = `import requests

resp = requests.post(
    "${BASE}/v1/my-model/generate",
    headers={"Authorization": "Bearer sk-YOUR_KEY"},
    json={"prompt": "Once upon a time", "max_new_tokens": 80},
)
resp.raise_for_status()
print(resp.json()["text"])`;

const node = `const res = await fetch("${BASE}/v1/my-model/generate", {
  method: "POST",
  headers: {
    "Authorization": "Bearer sk-YOUR_KEY",
    "Content-Type": "application/json",
  },
  body: JSON.stringify({ prompt: "Once upon a time", max_new_tokens: 80 }),
});
const { text } = await res.json();
console.log(text);`;

const response = `{
  "model": "my-model",
  "prompt": "Once upon a time",
  "text": "Once upon a time, in a quiet…",
  "latency_ms": 128
}`;

export default function Docs({ user, embedded, onStart }) {
  const [tab, setTab] = useState("curl");
  const [sec, setSec] = useState("quickstart");
  const examples = { curl, python, node };
  const langLabel = { curl: "cURL", python: "Python", node: "JavaScript" };

  const jump = (id) => {
    setSec(id);
    document.getElementById(`doc-${id}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const content = (
    <div className="docs-v2">
      <nav className="docs-nav" aria-label="On this page">
        <div className="set-subnav-h">API reference</div>
        {SECTIONS.map((s) => (
          <button key={s.id} type="button" className={`set-subnav-i${sec === s.id ? " on" : ""}`} aria-current={sec === s.id ? "true" : undefined} onClick={() => jump(s.id)}>
            {s.label}
          </button>
        ))}
        <div className="docs-base">
          <span className="field-l">Base URL</span>
          <code>{BASE}</code>
        </div>
      </nav>

      <article className="docs-main">
        <header className="ph" style={{ marginBottom: 8 }}>
          <div>
            <div className="ph-eyebrow">Developer API</div>
            <h1>Call your models over HTTP</h1>
            <p>Every deployed model gets a JSON endpoint behind your API key — the same shape for every text model you train. Swap the model name and go.</p>
          </div>
        </header>

        <section id="doc-quickstart" className="doc-sec">
          <h2>Quickstart</h2>
          <ol className="doc-steps-v2">
            <li><b>Deploy a model.</b> Train one, save it, then switch its endpoint on in <em>Models</em>.</li>
            <li><b>Create an API key</b> on the <em>API</em> page. It's shown once — store it safely.</li>
            <li><b>Call it.</b> POST a prompt to the endpoint below.</li>
          </ol>
          <div className="seg-ctl" role="tablist" aria-label="Language" style={{ marginBottom: 10 }}>
            {Object.keys(examples).map((k) => (
              <button key={k} type="button" role="tab" aria-selected={tab === k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{langLabel[k]}</button>
            ))}
          </div>
          <CodeBlock code={examples[tab]} lang={langLabel[tab].toLowerCase()} />
          {!user && <button type="button" className="btn btn-primary btn-lg doc-cta" onClick={onStart}>Create an account to get a key <ArrowRight size={15} /></button>}
        </section>

        <section id="doc-auth" className="doc-sec">
          <h2>Authentication</h2>
          <p>Authenticate every request with a workspace API key in the <code>Authorization</code> header:</p>
          <CodeBlock code={`Authorization: Bearer sk-YOUR_KEY`} />
          <p>Keys are created and revoked on the <b>API</b> page, scoped to one workspace, and shown only once.
            Revoking a key stops anything using it immediately. Every call is counted in your workspace's usage.</p>
        </section>

        <section id="doc-generate" className="doc-sec">
          <h2>Generate</h2>
          <div className="doc-endpoint-v2">
            <span className="badge badge-ok badge-mono">POST</span>
            <code>{`${BASE}/v1/{model}/generate`}</code>
          </div>
          <p>Continues <code>prompt</code> with your deployed text model and returns the completion and latency.
            <code>{`{model}`}</code> is the model's name in your workspace.</p>
          <h3>Response</h3>
          <CodeBlock code={response} lang="json" />
        </section>

        <section id="doc-params" className="doc-sec">
          <h2>Parameters</h2>
          <div className="ui-card tbl-wrap">
            <table className="tbl doc-tbl">
              <thead><tr><th>Field</th><th>Type</th><th>Default</th><th>Description</th></tr></thead>
              <tbody>
                {PARAMS.map(([f, t, d, desc]) => (
                  <tr key={f}><td><code>{f}</code></td><td className="subtle">{t}</td><td><code>{d}</code></td><td>{desc}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section id="doc-errors" className="doc-sec">
          <h2>Errors</h2>
          <p>Errors return a JSON body <code>{`{ "detail": "…" }`}</code> with a standard status code.</p>
          <div className="ui-card tbl-wrap">
            <table className="tbl doc-tbl">
              <thead><tr><th>Status</th><th>Meaning</th><th>Fix</th></tr></thead>
              <tbody>
                {ERRORS.map(([c, m, fix]) => (
                  <tr key={c}><td><span className="badge badge-mono">{c}</span></td><td>{m}</td><td className="muted">{fix}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section id="doc-limits" className="doc-sec">
          <h2>Rate limits</h2>
          <p>Requests are rate-limited per workspace. Past a limit you get a <code>429</code> with a
            <code>Retry-After</code> header in seconds — wait that long, then retry with exponential backoff.</p>
        </section>
      </article>
    </div>
  );

  if (embedded) return <div className="page">{content}</div>;
  return (
    <div className="public-page">
      <nav className="pub-nav">
        <div className="pub-nav-in">
          <Logo href="#/" />
          <div className="row">
            <ThemeToggle />
            <a className="btn btn-ghost" href="#/signin">Sign in</a>
            <a className="btn btn-primary" href="#/signup">Get started</a>
          </div>
        </div>
      </nav>
      <div className="page">{content}</div>
    </div>
  );
}
