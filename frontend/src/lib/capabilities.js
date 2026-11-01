// Ask the local engine what this machine can realistically train. The engine
// answers a `capabilities` frame (device, cores, RAM, and per-size / per-base
// guidance). If it isn't running yet, fall back to what the browser can see.

export function probeLocalCapabilities(url = "ws://localhost:8765", timeout = 1600) {
  return new Promise((resolve) => {
    let ws;
    let done = false;
    const finish = (v) => {
      if (done) return;
      done = true;
      try { ws && ws.close(); } catch {}
      resolve(v);
    };
    try {
      ws = new WebSocket(url);
      ws.onopen = () => ws.send(JSON.stringify({ type: "capabilities" }));
      ws.onmessage = (e) => {
        try {
          const f = JSON.parse(e.data);
          if (f.type === "capabilities") {
            // "Apple GPU (MPS) present — CPU is faster at these sizes" → a short name + the advice
            const [accel, ...note] = String(f.accelerator || "CPU").split(" — ");
            finish({ ...f, source: "engine", accelerator: accel, accelerator_note: note.join(" — ") });
          }
        } catch {}
      };
      ws.onerror = () => finish(null);
    } catch {
      finish(null);
    }
    setTimeout(() => finish(null), timeout);
  });
}

// gather the configured data and get stats + a sample back, without training
export function probePrepare(url, spec, timeout = 60000) {
  return new Promise((resolve) => {
    let ws;
    let done = false;
    const finish = (v) => {
      if (done) return;
      done = true;
      try { ws && ws.close(); } catch {}
      resolve(v);
    };
    try {
      ws = new WebSocket(url);
      ws.onopen = () => ws.send(JSON.stringify({ type: "prepare", spec }));
      ws.onmessage = (e) => {
        try {
          const f = JSON.parse(e.data);
          if (f.type === "prepared") finish(f);
          else if (f.type === "error") finish({ error: f.message });
        } catch {}
      };
      ws.onerror = () => finish(null);
    } catch {
      finish(null);
    }
    setTimeout(() => finish(null), timeout);
  });
}

function modelRequest(url, msg, timeout = 5000) {
  return new Promise((resolve) => {
    let ws;
    let done = false;
    const finish = (v) => { if (done) return; done = true; try { ws && ws.close(); } catch {} resolve(v); };
    try {
      ws = new WebSocket(url);
      ws.onopen = () => ws.send(JSON.stringify(msg));
      ws.onmessage = (e) => { try { const f = JSON.parse(e.data); if (f.type === "models") finish(f.models || []); } catch {} };
      ws.onerror = () => finish(null);
    } catch { finish(null); }
    setTimeout(() => finish(null), timeout);
  });
}

// list / delete saved models on the engine (returns the fresh list, or null)
export const probeModels = (url) => modelRequest(url, { type: "list_models" });
export const deleteModel = (url, name) => modelRequest(url, { type: "delete_model", name });

// saved-dataset library: list / save / delete (each resolves to the fresh list)
function datasetRequest(url, msg, timeout = 8000) {
  return new Promise((resolve) => {
    let ws;
    let done = false;
    const finish = (v) => { if (done) return; done = true; try { ws && ws.close(); } catch {} resolve(v); };
    try {
      ws = new WebSocket(url);
      ws.onopen = () => ws.send(JSON.stringify(msg));
      ws.onmessage = (e) => {
        try {
          const f = JSON.parse(e.data);
          if (f.type === "datasets") finish(f.datasets || []);
          else if (f.type === "error") finish({ error: f.message });
        } catch {}
      };
      ws.onerror = () => finish(null);
    } catch { finish(null); }
    setTimeout(() => finish(null), timeout);
  });
}
export const probeDatasets = (url) => datasetRequest(url, { type: "list_datasets" });
export const deleteDataset = (url, name) => datasetRequest(url, { type: "delete_dataset", name });
export const saveDataset = (url, spec) => datasetRequest(url, { type: "save_dataset", spec }, 60000);

function downloadB64(b64, filename) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const url = URL.createObjectURL(new Blob([bytes], { type: "application/zip" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename || "model.zip";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ask the engine to zip a saved model, then download it in the browser
export function exportModel(url, name, timeout = 60000) {
  return new Promise((resolve) => {
    let ws;
    let done = false;
    const finish = (v) => { if (done) return; done = true; try { ws && ws.close(); } catch {} resolve(v); };
    try {
      ws = new WebSocket(url);
      ws.onopen = () => ws.send(JSON.stringify({ type: "export", name }));
      ws.onmessage = (e) => {
        try {
          const f = JSON.parse(e.data);
          if (f.type === "exported") { downloadB64(f.b64, f.filename); finish(true); }
          else if (f.type === "error") finish({ error: f.message });
        } catch {}
      };
      ws.onerror = () => finish(null);
    } catch { finish(null); }
    setTimeout(() => finish(null), timeout);
  });
}

export function browserFallback() {
  return {
    source: "browser",
    cpu_cores: navigator.hardwareConcurrency || null,
    ram_gb: navigator.deviceMemory || null,
  };
}

// a connected cloud engine is assumed to have a GPU, so every size is fine there
export function cloudGuidance() {
  const sizes = {};
  ["tiny", "small", "medium", "large", "xl"].forEach((s) => (sizes[s] = { tier: "local" }));
  return { source: "cloud", accelerator: "cloud GPU", cloud: true, sizes, bases: {} };
}
