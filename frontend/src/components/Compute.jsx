import React, { useEffect, useRef, useState } from "react";
import { Rocket, Check, X, Network, Square } from "lucide-react";
import { listGpus, planCompute, submitJob, listJobs, cancelJob } from "../lib/account.js";
import CodeBlock from "./CodeBlock.jsx";
import { gpuTime } from "./Overview.jsx";

// Distributed-training control plane: a parallelism planner (how a model splits
// across a GPU pool + what it costs) and a live jobs board. Submitting a job
// actually runs the real distributed trainer and meters GPU-seconds.

const STRAT_LABEL = {
  single: "Single GPU", ddp: "Data-parallel (DDP)", fsdp: "Sharded (FSDP)", "tp+pp": "Tensor + pipeline parallel",
};
const money = (n) => (n == null ? "—" : `$${Number(n).toLocaleString(undefined, { maximumFractionDigits: 2 })}`);

function humanTime(s) {
  if (s == null) return "—";
  if (s < 90) return `${Math.round(s)}s`;
  if (s < 5400) return `${(s / 60).toFixed(1)} min`;
  if (s < 172800) return `${(s / 3600).toFixed(1)} hr`;
  return `${(s / 86400).toFixed(1)} days`;
}

export default function Compute({ onNotice }) {
  const [gpus, setGpus] = useState([]);
  const [params, setParams] = useState(1300);
  const [gpu, setGpu] = useState("a100-80");
  const [perNode, setPerNode] = useState(8);
  const [nodes, setNodes] = useState(1);
  const [tokens, setTokens] = useState(5);           // billions
  const [plan, setPlan] = useState(null);
  const [jobs, setJobs] = useState(null);
  const [launching, setLaunching] = useState(false);
  const [err, setErr] = useState(null);
  const poll = useRef(null);

  useEffect(() => { listGpus().then(setGpus).catch(() => {}); }, []);

  // re-plan whenever the knobs change (debounced)
  useEffect(() => {
    const id = setTimeout(() => {
      planCompute({ params_m: Number(params) || 1, gpu, gpus_per_node: Number(perNode) || 1,
                    nodes: Number(nodes) || 1, tokens: (Number(tokens) || 0) * 1e9 })
        .then((p) => { setPlan(p); setErr(null); }).catch((e) => setErr(e.message));
    }, 180);
    return () => clearTimeout(id);
  }, [params, gpu, perNode, nodes, tokens]);

  const loadJobs = () => { listJobs().then(setJobs).catch(() => setJobs((j) => j || [])); };
  useEffect(() => {
    loadJobs();
    poll.current = setInterval(loadJobs, 1500);   // live job status
    return () => clearInterval(poll.current);
  }, []);

  async function launchDemo() {
    setLaunching(true); setErr(null);
    try {
      const spec = {
        size: "tiny", chunking: { block_size: 32 },
        data: { text: "the quick brown fox jumps over the lazy dog. pack my box with five dozen liquor jugs. " },
        train: { lr: 3e-3, max_steps: 60, batch_size: 8, seed: 7 },
      };
      await submitJob({ model_name: `dist-run-${Date.now().toString().slice(-4)}`, target: "cloud",
                        gpu, gpus_per_node: Number(perNode) || 1, nodes: Number(nodes) || 1,
                        params_m: Number(params) || 1, spec });
      onNotice?.("Distributed run launched — it's on the board below.");
      loadJobs();
    } catch (e) { setErr(e.message); } finally { setLaunching(false); }
  }

  const world = (Number(perNode) || 1) * (Number(nodes) || 1);

  return (
    <div className="page">
      <header className="ph">
        <div>
          <h1>Compute</h1>
          <p>Plan a run across a GPU pool. Pick a model size and hardware — we choose the parallelism strategy,
            estimate memory, cost and time, and write the launch command.</p>
        </div>
      </header>

      <div className="cmp-grid-v2">
        <section className="ui-card" aria-labelledby="cfg-t">
          <div className="ui-card-h"><h2 id="cfg-t" className="ui-card-t">Configuration</h2></div>
          <div className="ui-card-b stack">
            <label className="field"><span className="field-l">Model size</span>
              <div className="input-suffix"><input className="input mono" type="number" min={1} value={params} onChange={(e) => setParams(e.target.value)} /><span>M params</span></div>
            </label>
            <label className="field"><span className="field-l">GPU</span>
              <select className="select" value={gpu} onChange={(e) => setGpu(e.target.value)}>
                {gpus.map((g) => <option key={g.id} value={g.id}>{g.id} · {g.mem_gb} GB{g.cost_hr ? ` · $${g.cost_hr}/hr` : ""}</option>)}
              </select>
            </label>
            <div className="field-row">
              <label className="field"><span className="field-l">GPUs per node</span>
                <input className="input mono" type="number" min={1} max={8} value={perNode} onChange={(e) => setPerNode(e.target.value)} />
              </label>
              <label className="field"><span className="field-l">Nodes</span>
                <input className="input mono" type="number" min={1} max={64} value={nodes} onChange={(e) => setNodes(e.target.value)} />
              </label>
            </div>
            <label className="field"><span className="field-l">Token budget</span>
              <div className="input-suffix"><input className="input mono" type="number" min={0} step={1} value={tokens} onChange={(e) => setTokens(e.target.value)} /><span>billion</span></div>
            </label>
            <div className="cmp-world"><Network size={15} aria-hidden="true" />World size <b className="num">{world}</b> GPU{world === 1 ? "" : "s"}</div>
          </div>
        </section>

        <section className="ui-card" aria-labelledby="plan-t" aria-busy={!plan}>
          <div className="ui-card-h">
            <h2 id="plan-t" className="ui-card-t">Recommended plan</h2>
            {plan && <span className={`badge ${plan.fits ? "badge-ok" : "badge-danger"}`}>{plan.fits ? <><Check size={12} />Fits this pool</> : <><X size={12} />Too large — add GPUs</>}</span>}
          </div>
          {!plan ? <div className="ui-card-b"><div className="skel" style={{ width: "50%" }} /></div> : (
            <div className="ui-card-b stack">
              <div className="cmp-strat-v2">
                <span className="badge badge-mono">{plan.strategy}</span>
                <span className="cmp-strat-n">{STRAT_LABEL[plan.strategy] || plan.strategy}</span>
              </div>
              <p className="cmp-note-v2">{plan.note}</p>
              <dl className="cmp-metrics-v2">
                <div><dt>Training memory</dt><dd className="num">{plan.est_train_mem_gb} GB</dd></div>
                <div><dt>Per GPU</dt><dd className="num">{plan.gpu_mem_gb} GB <span className="subtle">({plan.usable_gb} usable)</span></dd></div>
                <div><dt>Pool rate</dt><dd className="num">{money(plan.cost_per_hr)}/hr</dd></div>
                <div><dt>Throughput</dt><dd className="num">{(plan.throughput_tok_s / 1000).toFixed(0)}k tok/s</dd></div>
                {plan.est_cost != null && <div><dt>Estimated cost</dt><dd className="num">{money(plan.est_cost)}</dd></div>}
                {plan.est_seconds != null && <div><dt>Wall time</dt><dd className="num">{humanTime(plan.est_seconds)}</dd></div>}
              </dl>
              <div className="field">
                <span className="field-l">Launch command</span>
                <CodeBlock code={plan.launcher} lang="shell" />
              </div>
              {err && <div className="note err"><X size={15} /><span>{err}</span></div>}
              <div className="row">
                <button type="button" className="btn btn-primary" disabled={launching} onClick={launchDemo}>
                  <Rocket size={15} />{launching ? "Launching…" : "Launch a test run"}
                </button>
                <span className="field-h">Trains a tiny model across local ranks with real DDP and meters GPU time.</span>
              </div>
            </div>
          )}
        </section>
      </div>

      <section className="ui-card" style={{ marginTop: 16 }} aria-labelledby="jobs-t2">
        <div className="ui-card-h"><h2 id="jobs-t2" className="ui-card-t">Training jobs</h2>{jobs?.length ? <span className="badge num">{jobs.length}</span> : null}</div>
        {jobs === null ? <div className="list-row"><div className="skel" style={{ width: "40%" }} /></div>
          : jobs.length === 0 ? (
            <div className="empty">
              <div className="empty-i"><Network size={20} /></div>
              <h3>No jobs yet</h3>
              <p>Launch a test run above to see a real data-parallel job train, sync its weights across ranks, and report GPU time.</p>
            </div>
          ) : (
            <div className="list">
              {jobs.map((j) => {
                const p = j.progress || {};
                const pct = p.max_steps ? Math.round(((p.step || 0) / p.max_steps) * 100) : (j.status === "done" ? 100 : 0);
                const active = j.status === "running" || j.status === "queued";
                return (
                  <div className="list-row cmp-job-v2" key={j.id}>
                    <div className="grow">
                      <div className="row">
                        <span className="cell-t">{j.model_name}</span>
                        <span className={`badge ${j.status === "done" ? "badge-ok" : j.status === "failed" ? "badge-danger" : active ? "badge-ok" : ""}`}>
                          {active && <span className="live-dot" />}{j.status}
                        </span>
                        {j.weights_synced && <span className="badge" title="every rank ended with identical weights"><Check size={11} />synced</span>}
                      </div>
                      <div className="cell-s">
                        {STRAT_LABEL[j.strategy] || j.strategy} · {j.world_size} GPU{j.world_size === 1 ? "" : "s"} · {j.gpu}
                        {j.gpu_seconds ? ` · ${gpuTime(j.gpu_seconds)} metered` : ""}
                      </div>
                      <div className={`progress ${j.status === "failed" ? "danger" : ""}`} style={{ marginTop: 8 }}><span style={{ width: `${pct}%` }} /></div>
                    </div>
                    <div className="cmp-job-side-v2">
                      {p.loss != null && <span className="mono subtle num">loss {p.loss}</span>}
                      {active && (
                        <button type="button" className="btn btn-sm btn-danger" onClick={() => cancelJob(j.id).then(loadJobs)}><Square size={12} />Cancel</button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
      </section>
    </div>
  );
}
