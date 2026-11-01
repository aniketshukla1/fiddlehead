import React, { useEffect, useRef, useState } from "react";
import {
  Check, ChevronLeft, ChevronRight, Play, Type, Wand2, Repeat, ArrowDownUp, Mic, Image as ImageIcon,
  Volume2, MessageSquare, PenLine, Code2, Feather, Network, ClipboardPaste, CloudUpload, FolderOpen,
  Link, Globe, Database, Bookmark, Eye, X, Laptop, Server, MonitorPlay, Rocket, AlertTriangle, Zap, Trash2,
} from "lucide-react";
import { probeLocalCapabilities, browserFallback, cloudGuidance, probePrepare, probeDatasets, saveDataset, deleteDataset } from "../lib/capabilities.js";

const SIZE_PARAMS = { tiny: 0.15, small: 0.8, medium: 4.8, large: 25, xl: 85 };
const TIER_LABEL = {
  local: "runs on your machine",
  heavy: "slow on CPU · a GPU is faster",
  cloud: "needs a GPU · use a remote engine",
};

const TEMPLATES = [
  { id: "text", icon: Type, name: "Autocomplete on your text",
    blurb: "Train a model from scratch to continue your text.",
    usable: "Great for understanding; useful on focused data." },
  { id: "finetune", icon: Wand2, name: "Fine-tune an open-source model",
    blurb: "Adapt a real pretrained model (GPT-2, SmolLM, Qwen…) to your data with LoRA.",
    usable: "The path to a genuinely usable model — best on GPU/cloud for bigger bases." },
  { id: "copy", icon: Repeat, name: "Copy patterns",
    blurb: "Learns to repeat sequences from context — forms a visible induction circuit.",
    usable: "Best for seeing in-context learning emerge." },
  { id: "sort", icon: ArrowDownUp, name: "Sort sequences",
    blurb: "Learns to output a sorted version of its input.",
    usable: "A crisp 'did it actually learn?' demo." },
  { id: "asr", icon: Mic, name: "Fine-tune speech-to-text",
    blurb: "Adapt OpenAI Whisper to transcribe your audio — upload clips with their transcripts.",
    usable: "Real ASR for your domain/accent — whisper-tiny/base run on CPU, bigger want a GPU." },
  { id: "vlm", icon: ImageIcon, name: "Fine-tune image→text",
    blurb: "Adapt a vision-language model to caption your images — upload pictures with captions.",
    usable: "Real image captioning on your domain — ViT-GPT2 runs on CPU with LoRA." },
  { id: "tts", icon: Volume2, name: "Fine-tune text-to-speech",
    blurb: "Adapt SpeechT5 to speak in a voice — upload short audio clips with what's said in each.",
    usable: "Real TTS in a target voice/style. Fine-tune on CPU; synthesis is on-demand." },
];

const ASR_MODELS = ["openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small"];
const VLM_MODELS = ["nlpconnect/vit-gpt2-image-captioning"];
const TTS_MODELS = ["microsoft/speecht5_tts"];

// Guided quick-start recipes — pick one and the wizard presets template + model
// + tokenizer so a newcomer never has to know the knobs. Power users can still
// choose a raw template below.
const RECIPES = [
  { id: "assistant", icon: MessageSquare, name: "Chat assistant", tag: "fine-tune",
    template: "finetune", baseModel: "HuggingFaceTB/SmolLM-135M", method: "lora",
    desc: "Adapt a small pretrained model to answer in your domain and voice." },
  { id: "writing", icon: PenLine, name: "Autocomplete my writing", tag: "from scratch",
    template: "text", size: "small", tokKind: "bpe",
    desc: "Train on your writing so it continues text the way you would." },
  { id: "code", icon: Code2, name: "Code completion", tag: "fine-tune",
    template: "finetune", baseModel: "distilgpt2", method: "lora",
    desc: "Fine-tune GPT-2 on a codebase to autocomplete in that style." },
  { id: "mimic", icon: Feather, name: "Style mimic", tag: "from scratch",
    template: "text", size: "small", tokKind: "char",
    desc: "Learn one author or voice from a focused corpus and imitate it." },
  { id: "pattern", icon: Repeat, name: "Learn a pattern", tag: "demo",
    template: "copy", size: "small",
    desc: "Watch an induction circuit form as the model learns to copy from context." },
  { id: "moe", icon: Network, name: "Mixture of Experts", tag: "from scratch",
    template: "text", size: "small", tokKind: "bpe", moe: { experts: 4, top_k: 2 },
    desc: "A sparse model: a router sends each token to a few of many expert FFNs. Watch them specialize live." },
  { id: "speech", icon: Mic, name: "Speech-to-text", tag: "fine-tune",
    template: "asr", asrBase: "openai/whisper-tiny",
    desc: "Fine-tune Whisper to transcribe speech — upload audio clips with their transcripts." },
  { id: "vision", icon: ImageIcon, name: "Image captioning", tag: "fine-tune",
    template: "vlm", vlmBase: "nlpconnect/vit-gpt2-image-captioning",
    desc: "Fine-tune a vision-language model to caption images — upload pictures with captions." },
  { id: "voice", icon: Volume2, name: "Text-to-speech", tag: "fine-tune",
    template: "tts", ttsBase: "microsoft/speecht5_tts",
    desc: "Fine-tune SpeechT5 to speak — upload short clips with what's said in each, then type to hear it." },
];

const BASE_MODELS = [
  "distilgpt2", "gpt2", "HuggingFaceTB/SmolLM-135M", "EleutherAI/pythia-160m",
  "Qwen/Qwen2.5-0.5B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
];

const SIZES = [
  { id: "tiny", name: "Tiny" }, { id: "small", name: "Small" }, { id: "medium", name: "Medium" },
  { id: "large", name: "Large" }, { id: "xl", name: "XL" },
];

const COMPUTE = [
  { id: "local", icon: Laptop, name: "This machine", blurb: "Train on the engine running next to this app." },
  { id: "cloud", icon: Server, name: "Remote engine", blurb: "Connect to an engine on a GPU server by URL." },
  { id: "demo", icon: MonitorPlay, name: "Demo — no training", blurb: "Explore the studio on a recorded synthetic run." },
];

const SOURCES = [
  { id: "paste", icon: ClipboardPaste, name: "Paste" }, { id: "upload", icon: CloudUpload, name: "Upload" },
  { id: "path", icon: FolderOpen, name: "File or folder" }, { id: "url", icon: Link, name: "URL" },
  { id: "scrape", icon: Globe, name: "Crawl the web" }, { id: "cloud", icon: Database, name: "Cloud storage" },
];

const STEPS = ["Template", "Data", "Model", "Compute", "Review"];
const LOCAL_URL = "ws://localhost:8765";
const TEXT_EXTS = /\.(txt|md|markdown|rst|csv|json|jsonl|py|js|ts|log)$/i;

function readFile(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    if (TEXT_EXTS.test(file.name)) {
      reader.onload = () => resolve({ name: file.name, size: file.size, text: reader.result });
      reader.readAsText(file);
    } else {
      reader.onload = () => {
        const bytes = new Uint8Array(reader.result);
        let bin = "";
        for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
        resolve({ name: file.name, size: file.size, b64: btoa(bin) });
      };
      reader.readAsArrayBuffer(file);
    }
  });
}

const SAMPLE_TEXT =
  "to be, or not to be, that is the question:\nwhether 'tis nobler in the mind to suffer\n" +
  "the slings and arrows of outrageous fortune,\nor to take arms against a sea of troubles\n";

// A fuller public-domain corpus for the one-click sample run (Hamlet's soliloquy).
const SAMPLE_CORPUS = [
  "to be, or not to be, that is the question:",
  "whether 'tis nobler in the mind to suffer",
  "the slings and arrows of outrageous fortune,",
  "or to take arms against a sea of troubles",
  "and by opposing end them. to die—to sleep,",
  "no more; and by a sleep to say we end",
  "the heart-ache and the thousand natural shocks",
  "that flesh is heir to: 'tis a consummation",
  "devoutly to be wish'd. to die, to sleep;",
  "to sleep, perchance to dream—ay, there's the rub:",
  "for in that sleep of death what dreams may come,",
  "when we have shuffled off this mortal coil,",
  "must give us pause—there's the respect",
  "that makes calamity of so long life.",
].join("\n") + "\n";

// the one-click sample: a tiny character model on Hamlet, trained locally, in seconds
export const SAMPLE_SPEC = {
  name: "sample-shakespeare", template: "text", size: "tiny", autostart: true,
  data: { source: "paste", text: SAMPLE_CORPUS },
  tokenizer: { kind: "char", vocab_size: 256 },
  chunking: { block_size: 64, strategy: "window" },
};

// which template a saved model (retrain-from) came from
const templateOf = (m) => (["finetune", "asr", "vlm", "tts"].includes(m.kind) ? m.kind : m.template || "text");

// A mid-setup draft persists across refreshes so you resume where you left off
// (never secrets or uploaded file blobs).
const DRAFT_KEY = "fiddlehead.wizard";
const DRAFT_TEXT_CAP = 50000; // don't persist huge pastes into localStorage
function loadWizardDraft() {
  try { return JSON.parse(localStorage.getItem(DRAFT_KEY) || "null") || {}; } catch { return {}; }
}

// Client-side mirror of the engine's scheduled_lr, used to draw the schedule
// preview so the shape you configure is exactly what training will run.
function SchedulePreview({ train, baseLr }) {
  const lr = train.lr !== "" && !isNaN(Number(train.lr)) ? Number(train.lr) : baseLr;
  const minLr = Math.max(0, Number(train.minLr) || 0);
  const warm = Math.max(0, Number(train.warmup) || 0);
  const total = Math.max(0, Number(train.maxSteps) || 0);
  const horizon = total > 0 ? total : Math.max(warm * 2, 400);
  const N = 72;
  const raw = [];
  for (let i = 0; i <= N; i++) {
    const step = (i / N) * horizon;
    let v;
    if (warm > 0 && step < warm) v = lr * (step + 1) / warm;
    else if (train.schedule === "constant" || total <= 0) v = lr;
    else {
      const prog = Math.min(1, Math.max(0, (step - warm) / Math.max(1, total - warm)));
      v = train.schedule === "cosine"
        ? minLr + 0.5 * (lr - minLr) * (1 + Math.cos(Math.PI * prog))
        : lr - (lr - minLr) * prog;
    }
    raw.push(v);
  }
  const maxV = Math.max(lr, ...raw) || 1;
  const W = 300, H = 60, pad = 4;
  const pts = raw.map((v, i) => {
    const x = pad + (i / N) * (W - 2 * pad);
    const y = pad + (1 - v / maxV) * (H - 2 * pad);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  const warmX = warm > 0 ? pad + Math.min(1, warm / horizon) * (W - 2 * pad) : null;
  const fmt = (v) => Number(v).toExponential(1).replace("e-", "e−");
  return (
    <div className="sched-preview">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden>
        <polyline points={`${pad},${H - pad} ${pts.split(" ")[0]}`} className="sched-base" />
        {warmX != null && <line x1={warmX} y1={pad} x2={warmX} y2={H - pad} className="sched-warm" />}
        <polyline points={pts} className="sched-line" />
      </svg>
      <div className="sched-cap">
        <span>peak <b>{fmt(lr)}</b></span>
        <span>{warm > 0 ? `warmup ${warm}` : "no warmup"}</span>
        <span>{train.schedule === "constant" || total <= 0 ? "constant" : `${train.schedule} → ${fmt(minLr)}`}</span>
      </div>
    </div>
  );
}

const STEP_COPY = [
  ["What do you want to build?", "Pick a recipe and we choose sensible settings, or start from a blank template."],
  ["Add your data", "The more consistent your data, the more useful the model."],
  ["Choose the model", "Size, base model and — if you want them — the full production training knobs."],
  ["Where should it train?", "On this machine, on a GPU server, or as a recorded demo."],
  ["Review and train", "Name it and start. You'll watch it learn in the studio."],
];

export default function Setup({ onLaunch, prefill, engine, onEngineHelp }) {
  // restore a saved draft on first mount — unless we're prefilling from a retrain
  const [draft] = useState(() => (prefill ? {} : loadWizardDraft()));
  const [step, setStep] = useState(prefill ? 1 : (Number.isInteger(draft.step) ? draft.step : 0));
  const [compute, setCompute] = useState(draft.compute || "local");
  const [cloudUrl, setCloudUrl] = useState(draft.cloudUrl || "wss://");
  const [template, setTemplate] = useState(prefill ? templateOf(prefill) : (draft.template || "text"));
  const [recipe, setRecipe] = useState(prefill ? null : (draft.recipe ?? null));
  const [size, setSize] = useState(prefill?.size || draft.size || "small");
  const [name, setName] = useState(prefill?.name ? `${prefill.name}-v2` : (draft.name || "my-model"));

  const [datasets, setDatasets] = useState([]);
  const [datasetName, setDatasetName] = useState(draft.datasetName || "");
  const [savingDataset, setSavingDataset] = useState(false);

  const [source, setSource] = useState(draft.source || "paste");
  const [text, setText] = useState(draft.text != null ? draft.text : SAMPLE_TEXT);
  const [path, setPath] = useState(draft.path || "");
  const [url, setUrl] = useState(draft.url || "");
  const [scrapeUrls, setScrapeUrls] = useState(draft.scrapeUrls || "");
  const [depth, setDepth] = useState(draft.depth ?? 0);
  const [cloudUri, setCloudUri] = useState(draft.cloudUri || "");
  const [accessKey, setAccessKey] = useState("");
  const [secretKey, setSecretKey] = useState("");

  const pBase = prefill?.base_model;
  const [baseModel, setBaseModel] = useState((prefill?.kind === "finetune" && pBase) || draft.baseModel || "distilgpt2");
  const [method, setMethod] = useState(prefill?.method || draft.method || "lora");

  const [caps, setCaps] = useState(null);
  const [preview, setPreview] = useState(null);
  const [previewing, setPreviewing] = useState(false);
  const [uploadFiles, setUploadFiles] = useState([]);
  const [reading, setReading] = useState(false);
  const [tokKind, setTokKind] = useState(draft.tokKind || "char");
  const [vocabSize, setVocabSize] = useState(draft.vocabSize ?? 512);
  const [blockSize, setBlockSize] = useState(draft.blockSize ?? 128);
  const [strategy, setStrategy] = useState(draft.strategy || "window");
  const [earlyStop, setEarlyStop] = useState(draft.earlyStop ?? false);
  const [patience, setPatience] = useState(draft.patience ?? 5);
  const [moeExperts, setMoeExperts] = useState(draft.moeExperts ?? 0);
  const [moeTopK, setMoeTopK] = useState(draft.moeTopK ?? 2);
  const [asrBase, setAsrBase] = useState((prefill?.kind === "asr" && pBase) || draft.asrBase || "openai/whisper-tiny");
  const [audioPairs, setAudioPairs] = useState([]);   // {name, b64, text} — not persisted (blobs)
  const [readingAudio, setReadingAudio] = useState(false);
  const [vlmBase, setVlmBase] = useState((prefill?.kind === "vlm" && pBase) || draft.vlmBase || "nlpconnect/vit-gpt2-image-captioning");
  const [imagePairs, setImagePairs] = useState([]);   // {name, b64, text} — not persisted (blobs)
  const [readingImages, setReadingImages] = useState(false);
  const [ttsBase, setTtsBase] = useState((prefill?.kind === "tts" && pBase) || draft.ttsBase || "microsoft/speecht5_tts");

  // production training hyperparameters (all optional; blank lr = template default)
  const [train, setTrain] = useState(() => ({
    lr: "", schedule: "cosine", warmup: 0, minLr: 0, weightDecay: 0.01, gradClip: 1.0,
    batchSize: 32, gradAccum: 1, maxSteps: 0, labelSmoothing: 0, dropout: 0, precision: "fp32", seed: 1337,
    checkpointEvery: 0,
    ...(draft.train || {}),
  }));
  const setT = (k, v) => setTrain((t) => ({ ...t, [k]: v }));
  const [lora, setLora] = useState(() => draft.lora || { r: 8, alpha: 16, dropout: 0.05 });
  const setL = (k, v) => setLora((l) => ({ ...l, [k]: v }));

  // keyboard nav: Enter advances a valid step, ⌘/Ctrl+Enter launches. A ref keeps
  // the single listener reading fresh state without re-binding every render.
  const kbd = useRef({});
  useEffect(() => {
    function onKey(e) {
      if (e.key !== "Enter" || e.shiftKey) return;
      const tag = e.target?.tagName;
      if (tag === "TEXTAREA" || tag === "BUTTON" || tag === "SUMMARY" || tag === "A") return; // native Enter wins
      if (document.querySelector(".mc-overlay, .cmdk-overlay, .coach-overlay, .dlg-overlay")) return; // a modal owns the keys
      const k = kbd.current;
      if (e.metaKey || e.ctrlKey) { if (k.ready) { e.preventDefault(); k.launch(); } return; }
      if (k.step < 4) { if (k.canNext) { e.preventDefault(); k.next(); } }
      else if (k.ready) { e.preventDefault(); k.launch(); }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // persist the in-progress draft so a refresh resumes mid-setup (no secrets/blobs)
  useEffect(() => {
    const d = { step, compute, cloudUrl, template, recipe, size, name, source,
      text: text.length <= DRAFT_TEXT_CAP ? text : "", path, url, scrapeUrls, depth,
      cloudUri, baseModel, method, tokKind, vocabSize, blockSize, strategy, earlyStop, patience, datasetName, moeExperts, moeTopK, asrBase, vlmBase, ttsBase, train, lora };
    try { localStorage.setItem(DRAFT_KEY, JSON.stringify(d)); } catch {}
  }, [step, compute, cloudUrl, template, recipe, size, name, source, text, path, url,
      scrapeUrls, depth, cloudUri, baseModel, method, tokKind, vocabSize, blockSize, strategy, earlyStop, patience, datasetName, moeExperts, moeTopK, asrBase, vlmBase, ttsBase, train, lora]);
  const clearDraft = () => { try { localStorage.removeItem(DRAFT_KEY); } catch {} };

  // scroll the page back to the top on step change (the footer stays put)
  const top = useRef(null);
  useEffect(() => { top.current?.closest(".main")?.scrollTo({ top: 0 }); }, [step]);

  async function addFiles(fileList) {
    if (!fileList?.length) return;
    setReading(true);
    const read = await Promise.all([...fileList].map(readFile));
    setUploadFiles((prev) => [...prev, ...read]);
    setReading(false);
  }

  function trySample() {
    clearDraft();
    onLaunch({ mode: "local", url: LOCAL_URL }, SAMPLE_SPEC);
  }

  // apply a guided recipe: preset template + model + tokenizer in one click
  function applyRecipe(r) {
    setRecipe(r.id);
    setTemplate(r.template);
    if (r.size) setSize(r.size);
    if (r.baseModel) setBaseModel(r.baseModel);
    if (r.method) setMethod(r.method);
    if (r.tokKind) setTokKind(r.tokKind);
    if (r.asrBase) setAsrBase(r.asrBase);
    if (r.vlmBase) setVlmBase(r.vlmBase);
    if (r.ttsBase) setTtsBase(r.ttsBase);
    setMoeExperts(r.moe?.experts ?? 0);
    setMoeTopK(r.moe?.top_k ?? 2);
  }

  async function readBinary(fileList, withUrl) {
    return Promise.all([...fileList].map((f) => new Promise((res) => {
      const r = new FileReader();
      r.onload = () => {
        const bytes = new Uint8Array(r.result);
        let bin = "";
        for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
        res({ name: f.name, b64: btoa(bin), text: "", ...(withUrl ? { url: URL.createObjectURL(f) } : {}) });
      };
      r.readAsArrayBuffer(f);
    })));
  }
  async function addImages(fileList) {
    if (!fileList?.length) return;
    setReadingImages(true);
    const read = await readBinary(fileList, true);
    setImagePairs((p) => [...p, ...read]);
    setReadingImages(false);
  }
  async function addAudio(fileList) {
    if (!fileList?.length) return;
    setReadingAudio(true);
    const read = await readBinary(fileList, false);
    setAudioPairs((p) => [...p, ...read]);
    setReadingAudio(false);
  }

  const isText = template === "text";
  const isFinetune = template === "finetune";
  const isAsr = template === "asr";
  const isVlm = template === "vlm";
  const isTts = template === "tts";
  const usesAudioPairs = isAsr || isTts;   // both use (audio, text) clips
  const needsData = isText || isFinetune || isAsr || isVlm || isTts;

  const computeUrl = () => (compute === "cloud" ? cloudUrl : LOCAL_URL);

  useEffect(() => {
    let alive = true;
    if (compute === "local") {
      setCaps(null);
      probeLocalCapabilities().then((c) => { if (alive) setCaps(c || browserFallback()); });
      probeDatasets(computeUrl()).then((ds) => { if (alive && Array.isArray(ds)) setDatasets(ds); });
    } else if (compute === "cloud") {
      setCaps(cloudGuidance());
      if (cloudUrl.length > 6) probeDatasets(cloudUrl).then((ds) => { if (alive && Array.isArray(ds)) setDatasets(ds); });
    } else { setCaps(null); setDatasets([]); }
    return () => { alive = false; };
  }, [compute, engine?.online]); // eslint-disable-line

  const sizeGuide = caps?.sizes || {};
  const selectedTier = isFinetune
    ? (caps?.cloud ? "local" : caps?.bases?.[baseModel]?.tier)
    : sizeGuide[size]?.tier;

  function dataSpec() {
    if (source === "path") return { source: "path", path };
    if (source === "url") return { source: "url", url };
    if (source === "scrape") return { source: "scrape", urls: scrapeUrls.split(/\s+/).filter(Boolean), depth: Number(depth), max_pages: 12 };
    if (source === "cloud") return { source: "cloud", uri: cloudUri, credentials: { access_key: accessKey, secret_key: secretKey } };
    if (source === "upload") return { source: "upload", files: uploadFiles.map((f) => ({ name: f.name, ...(f.text != null ? { text: f.text } : { b64: f.b64 }) })) };
    if (source === "dataset") return { source: "dataset", name: datasetName };
    return { source: "paste", text };
  }

  // save the currently configured data as a reusable, named dataset
  async function saveCurrentDataset() {
    const nm = window.prompt("Save this data as a reusable dataset.\nName:", name ? `${name}-data` : "my-dataset");
    if (!nm) return;
    setSavingDataset(true);
    const res = await saveDataset(computeUrl(), { name: nm, data: dataSpec() });
    setSavingDataset(false);
    if (Array.isArray(res)) setDatasets(res);
    else if (res?.error) setPreview({ error: res.error });
  }

  async function doPreview() {
    if (compute === "demo") return;
    setPreviewing(true); setPreview(null);
    const res = await probePrepare(computeUrl(), { data: dataSpec(), tokenizer: { kind: tokKind, vocab_size: Number(vocabSize) } });
    setPreviewing(false);
    setPreview(res || { error: "Couldn't reach the engine — is it running?" });
  }

  const dataReady = !needsData || (usesAudioPairs
    ? (audioPairs.length > 0 && audioPairs.every((p) => p.text.trim().length > 0))
    : isVlm
    ? (imagePairs.length > 0 && imagePairs.every((p) => p.text.trim().length > 0))
    : {
        paste: text.trim().length > 0, upload: uploadFiles.length > 0, path: path.trim().length > 0,
        url: url.trim().length > 0, scrape: scrapeUrls.trim().length > 0, cloud: cloudUri.trim().length > 0,
        dataset: !!datasetName,
      }[source]);
  const modelReady = (!isFinetune || baseModel.trim().length > 1) && (!isAsr || asrBase.trim().length > 1) && (!isVlm || vlmBase.trim().length > 1) && (!isTts || ttsBase.trim().length > 1);
  const ready = (compute !== "cloud" || cloudUrl.length > 6) && modelReady && name.trim().length > 0;
  const canNext = [true, dataReady, modelReady, (compute !== "cloud" || cloudUrl.length > 6), ready][step];

  // Map the advanced-panel state to the backend `train` block. Blank/NaN fields
  // fall back to the engine defaults, so an untouched panel changes nothing.
  function trainBlock() {
    const n = (v, d) => (v === "" || v == null || isNaN(Number(v)) ? d : Number(v));
    const b = {
      schedule: train.schedule, warmup_steps: n(train.warmup, 0), min_lr: n(train.minLr, 0),
      weight_decay: n(train.weightDecay, 0.01), grad_clip: n(train.gradClip, 1.0),
      batch_size: n(train.batchSize, 32), grad_accum: n(train.gradAccum, 1),
      max_steps: n(train.maxSteps, 0), label_smoothing: n(train.labelSmoothing, 0),
      dropout: n(train.dropout, 0), precision: train.precision, seed: n(train.seed, 1337),
      checkpoint_every: n(train.checkpointEvery, 0),
    };
    if (train.lr !== "" && train.lr != null && !isNaN(Number(train.lr))) b.lr = Number(train.lr);
    return b;
  }
  const loraBlock = () => ({ r: Number(lora.r), alpha: Number(lora.alpha), dropout: Number(lora.dropout) });

  function launch() {
    const target = compute === "cloud" ? { mode: "cloud", url: cloudUrl }
      : compute === "demo" ? { mode: "demo" } : { mode: "local", url: LOCAL_URL };
    let spec = null;
    const nm = name.trim();
    if (compute !== "demo") {
      const early_stop = { enabled: earlyStop, patience: Number(patience) };
      const tb = trainBlock();
      if (isAsr) spec = { name: nm, template: "asr", base_model: asrBase.trim(), method: "lora", autostart: true, lr: 1e-4, train: tb, lora: loraBlock(), data: { pairs: audioPairs.map((p) => ({ audio: { b64: p.b64 }, text: p.text })) } };
      else if (isTts) spec = { name: nm, template: "tts", base_model: ttsBase.trim(), method: "lora", autostart: true, lr: 1e-4, train: tb, lora: loraBlock(), data: { pairs: audioPairs.map((p) => ({ audio: { b64: p.b64 }, text: p.text })) } };
      else if (isVlm) spec = { name: nm, template: "vlm", base_model: vlmBase.trim(), method: "lora", autostart: true, lr: 2e-4, train: tb, lora: loraBlock(), data: { pairs: imagePairs.map((p) => ({ image: { b64: p.b64 }, text: p.text })) } };
      else if (isFinetune) spec = { name: nm, template: "finetune", base_model: baseModel.trim(), method, autostart: true, data: dataSpec(), chunking: { block_size: Number(blockSize) }, lr: 2e-4, train: tb, ...(method === "lora" ? { lora: loraBlock() } : {}), early_stop };
      else if (isText) spec = { name: nm, template, size, autostart: true, data: dataSpec(), tokenizer: { kind: tokKind, vocab_size: Number(vocabSize) }, chunking: { block_size: Number(blockSize), strategy }, train: tb, early_stop, ...(moeExperts > 0 ? { moe: { experts: Number(moeExperts), top_k: Number(moeTopK) } } : {}) };
      else spec = { name: nm, template, size, autostart: true };
    }
    clearDraft();
    onLaunch(target, spec);
  }

  kbd.current = { step, canNext, ready, launch, next: () => setStep((s) => Math.min(4, s + 1)) };

  const recipeObj = RECIPES.find((r) => r.id === recipe);
  const tplObj = TEMPLATES.find((t) => t.id === template);
  const engineDown = compute === "local" && engine?.online === false;
  const modelLabel = isAsr ? `${asrBase} · LoRA` : isTts ? `${ttsBase} · LoRA` : isVlm ? `${vlmBase} · LoRA`
    : isFinetune ? `${baseModel} · ${method.toUpperCase()}` : `${SIZES.find((x) => x.id === size)?.name} · ~${sizeGuide[size]?.params_m ?? SIZE_PARAMS[size]}M params`;
  const dataLabel = usesAudioPairs ? `${audioPairs.length} audio clip${audioPairs.length === 1 ? "" : "s"}`
    : isVlm ? `${imagePairs.length} image${imagePairs.length === 1 ? "" : "s"}`
    : needsData ? (source === "dataset" ? `dataset “${datasetName}”` : SOURCES.find((s) => s.id === source)?.name) : "generated for you";

  return (
    <div className="page wiz" ref={top}>
      <header className="ph wiz-ph">
        <div>
          <div className="ph-eyebrow">{prefill ? `Retrain · ${prefill.name}` : "New model"}</div>
          <h1>{STEP_COPY[step][0]}</h1>
          <p>{STEP_COPY[step][1]}</p>
        </div>
      </header>

      <ol className="stepper" aria-label="Setup steps">
        {STEPS.map((s, i) => (
          <li key={s} className={`stepper-i${i === step ? " on" : ""}${i < step ? " done" : ""}`}>
            <button type="button" onClick={() => setStep(i)} aria-current={i === step ? "step" : undefined}>
              <span className="stepper-n">{i < step ? <Check size={12} strokeWidth={3} /> : i + 1}</span>
              <span className="stepper-l">{s}</span>
            </button>
          </li>
        ))}
      </ol>

      <div className="wiz-body">
        {/* ---- Step 0: Template ---- */}
        {step === 0 && (
          <>
            <div className="qs">
              <div className="qs-art" aria-hidden="true">
                <svg viewBox="0 0 120 56" preserveAspectRatio="none">
                  <path className="qs-grid" d="M0 14H120M0 28H120M0 42H120M30 0V56M60 0V56M90 0V56" />
                  <path className="qs-curve" pathLength="1" d="M2 8C14 10 18 30 30 33S48 38 60 41 84 45 96 46 112 47 118 47" />
                </svg>
              </div>
              <div className="grow">
                <div className="qs-t">New here? Train a sample in 30 seconds</div>
                <div className="qs-d">A tiny model learns Hamlet's soliloquy on your machine. Watch the loss fall and its attention heads form, live.</div>
              </div>
              <div className="qs-cta">
                <button type="button" className="btn btn-primary" onClick={trySample} disabled={engine?.online === false}>
                  <Play size={14} />Train the sample
                </button>
                {engine?.online === false && <button type="button" className="btn-link qs-eng" onClick={onEngineHelp}>Start the engine first</button>}
              </div>
            </div>

            <h2 className="wiz-sub">Start from a recipe</h2>
            <div className="recipe-grid">
              {RECIPES.map((r) => {
                const Icon = r.icon;
                return (
                  <button key={r.id} type="button" className={`recipe${recipe === r.id ? " sel" : ""}`} aria-pressed={recipe === r.id} onClick={() => applyRecipe(r)}>
                    <span className="recipe-top">
                      <span className="recipe-i"><Icon size={17} strokeWidth={1.75} aria-hidden="true" /></span>
                      <span className="badge">{r.tag}</span>
                    </span>
                    <span className="recipe-t">{r.name}</span>
                    <span className="recipe-d">{r.desc}</span>
                    {recipe === r.id && <span className="recipe-check" aria-hidden="true"><Check size={12} strokeWidth={3} /></span>}
                  </button>
                );
              })}
            </div>

            <details className="wiz-more" open={!recipe && template !== "text" ? true : undefined}>
              <summary><ChevronRight size={14} className="wiz-more-c" aria-hidden="true" />Or start from a blank template</summary>
              <div className="tpl-grid">
                {TEMPLATES.map((tp) => {
                  const Icon = tp.icon;
                  const sel = template === tp.id && !recipe;
                  return (
                    <button key={tp.id} type="button" className={`tpl${sel ? " sel" : ""}`} aria-pressed={sel} onClick={() => { setTemplate(tp.id); setRecipe(null); }}>
                      <span className="recipe-i"><Icon size={16} strokeWidth={1.75} aria-hidden="true" /></span>
                      <span className="grow">
                        <span className="recipe-t">{tp.name}</span>
                        <span className="recipe-d">{tp.blurb}</span>
                        <span className="tpl-note">{tp.usable}</span>
                      </span>
                    </button>
                  );
                })}
              </div>
            </details>
            <p className="wiz-hint">Retraining or reopening a model you saved? Find it in <a className="btn-link" href="#/models">Models</a>.</p>
          </>
        )}

        {/* ---- Step 1: Data ---- */}
        {step === 1 && (
          <div className="stack">
            {needsData ? (usesAudioPairs || isVlm) ? (
              <>
                <label className="drop" onDragOver={(e) => e.preventDefault()}
                       onDrop={(e) => { e.preventDefault(); (isVlm ? addImages : addAudio)(e.dataTransfer.files); }}>
                  <input type="file" multiple hidden accept={isVlm ? "image/*" : ".wav,.flac,.mp3,.ogg,.m4a,.aac"}
                         onChange={(e) => { (isVlm ? addImages : addAudio)(e.target.files); e.target.value = ""; }} />
                  <span className="drop-i"><CloudUpload size={20} aria-hidden="true" /></span>
                  <span className="drop-t">{(isVlm ? readingImages : readingAudio) ? "Reading files…" : isVlm ? "Drop images here, or click to choose" : "Drop audio clips here, or click to choose"}</span>
                  <span className="drop-s">{isVlm ? ".jpg · .png — then caption each one" : ".wav · .flac · .mp3 — then type what's said in each"}</span>
                </label>
                {(isVlm ? imagePairs : audioPairs).length > 0 && (
                  <div className="ui-card pairs">
                    {(isVlm ? imagePairs : audioPairs).map((p, i) => (
                      <div className="pair" key={i}>
                        {isVlm && p.url ? <img className="pair-thumb" src={p.url} alt="" />
                          : <span className="pair-i">{isVlm ? <ImageIcon size={15} /> : <Volume2 size={15} />}</span>}
                        <div className="grow">
                          <div className="cell-s">{p.name}</div>
                          <input className="input" value={p.text}
                                 placeholder={isVlm ? "Describe what's in this image…" : "Type exactly what is said…"}
                                 aria-label={`${isVlm ? "Caption" : "Transcript"} for ${p.name}`}
                                 onChange={(e) => (isVlm ? setImagePairs : setAudioPairs)((a) => a.map((x, j) => (j === i ? { ...x, text: e.target.value } : x)))} />
                        </div>
                        <button type="button" className="btn btn-ghost btn-icon btn-sm" aria-label={`Remove ${p.name}`}
                                onClick={() => (isVlm ? setImagePairs : setAudioPairs)((a) => a.filter((_, j) => j !== i))}><X size={14} /></button>
                      </div>
                    ))}
                  </div>
                )}
                <div className="field-h">
                  {(isVlm ? imagePairs : audioPairs).length} {isVlm ? "image" : "clip"}{(isVlm ? imagePairs : audioPairs).length === 1 ? "" : "s"} · files are read in your browser and sent to the engine.
                  {(isVlm ? imagePairs : audioPairs).length > 0 && !(isVlm ? imagePairs : audioPairs).every((p) => p.text.trim()) && <span className="warn-text"> Add the missing {isVlm ? "captions" : "transcripts"}.</span>}
                </div>
              </>
            ) : (
              <>
                <div className="seg-ctl src-tabs" role="tablist" aria-label="Data source">
                  {SOURCES.map((s) => {
                    const Icon = s.icon;
                    return (
                      <button key={s.id} type="button" role="tab" aria-selected={source === s.id} className={source === s.id ? "on" : ""} onClick={() => setSource(s.id)}>
                        <Icon size={14} aria-hidden="true" />{s.name}
                      </button>
                    );
                  })}
                  {datasets.length > 0 && (
                    <button type="button" role="tab" aria-selected={source === "dataset"} className={source === "dataset" ? "on" : ""} onClick={() => setSource("dataset")}>
                      <Bookmark size={14} aria-hidden="true" />Saved datasets
                    </button>
                  )}
                </div>

                {source === "paste" && (
                  <div className="field">
                    <textarea className="textarea data-text" value={text} onChange={(e) => setText(e.target.value)} spellCheck={false}
                              aria-label="Training text" placeholder="Paste text — the more consistent the style, the more useful the model…" />
                    <div className="field-h">{text.length.toLocaleString()} characters · {new Set(text).size} unique symbols{text.length < 400 && <span className="warn-text"> · add more for a useful model</span>}</div>
                  </div>
                )}
                {source === "upload" && (
                  <>
                    <label className="drop" onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); addFiles(e.dataTransfer.files); }}>
                      <input type="file" multiple hidden accept=".txt,.md,.markdown,.rst,.csv,.json,.jsonl,.py,.pdf" onChange={(e) => { addFiles(e.target.files); e.target.value = ""; }} />
                      <span className="drop-i"><CloudUpload size={20} aria-hidden="true" /></span>
                      <span className="drop-t">{reading ? "Reading files…" : "Drop files here, or click to choose"}</span>
                      <span className="drop-s">.txt · .md · .csv · .jsonl · .pdf — read in your browser</span>
                    </label>
                    {uploadFiles.length > 0 && (
                      <div className="ui-card pairs">
                        {uploadFiles.map((f, i) => (
                          <div className="pair" key={i}>
                            <span className="pair-i"><Type size={15} /></span>
                            <span className="grow cell-t">{f.name}</span>
                            <span className="subtle num" style={{ fontSize: 12 }}>{(f.size / 1024).toFixed(0)} KB</span>
                            <button type="button" className="btn btn-ghost btn-icon btn-sm" aria-label={`Remove ${f.name}`} onClick={() => setUploadFiles((u) => u.filter((_, j) => j !== i))}><X size={14} /></button>
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                )}
                {source === "path" && (
                  <label className="field"><span className="field-l">Path on the engine's machine</span>
                    <input className="input mono" value={path} onChange={(e) => setPath(e.target.value)} placeholder="/path/to/data.txt  or  /path/to/folder" />
                    <span className="field-h">A .txt / .md / .csv / .jsonl / .pdf file, or a folder (read recursively).</span>
                  </label>
                )}
                {source === "url" && (
                  <label className="field"><span className="field-l">Page or file URL</span>
                    <input className="input" value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://example.com/page  or  a raw .txt URL" />
                    <span className="field-h">Fetches the page and extracts readable text.</span>
                  </label>
                )}
                {source === "scrape" && (
                  <label className="field"><span className="field-l">Start pages, one per line</span>
                    <textarea className="textarea mono" value={scrapeUrls} onChange={(e) => setScrapeUrls(e.target.value)} spellCheck={false}
                              placeholder={"https://en.wikipedia.org/wiki/Transformer_(deep_learning)"} />
                    <span className="field-h row">Main content is extracted automatically. Follow links
                      <select className="select select-sm" value={depth} onChange={(e) => setDepth(e.target.value)} aria-label="Crawl depth">
                        <option value={0}>just these pages</option><option value={1}>1 level deep</option><option value={2}>2 levels deep</option>
                      </select>(same domain, up to 12 pages).</span>
                  </label>
                )}
                {source === "cloud" && (
                  <div className="stack" style={{ gap: 12 }}>
                    <label className="field"><span className="field-l">Object or bucket</span>
                      <input className="input mono" value={cloudUri} onChange={(e) => setCloudUri(e.target.value)} placeholder="s3://bucket/key · gs://bucket/prefix/ · a Drive or Dropbox link" />
                      <span className="field-h">Public links and any https:// object work as-is. Private S3 / GCS need credentials.</span>
                    </label>
                    {cloudUri.startsWith("s3://") && (
                      <div className="field-row">
                        <label className="field"><span className="field-l">Access key ID</span><input className="input mono" value={accessKey} onChange={(e) => setAccessKey(e.target.value)} placeholder="AKIA… (optional)" autoComplete="off" /></label>
                        <label className="field"><span className="field-l">Secret key</span><input className="input mono" type="password" value={secretKey} onChange={(e) => setSecretKey(e.target.value)} placeholder="optional" autoComplete="off" /></label>
                      </div>
                    )}
                  </div>
                )}
                {source === "dataset" && (
                  <div className="ui-card pairs" role="radiogroup" aria-label="Saved datasets">
                    {datasets.map((ds) => (
                      <label className={`pair ds${datasetName === ds.name ? " sel" : ""}`} key={ds.name}>
                        <input type="radio" name="ds" checked={datasetName === ds.name} onChange={() => setDatasetName(ds.name)} />
                        <span className="grow">
                          <span className="cell-t">{ds.name}</span>
                          <span className="cell-s"> · {(ds.documents ?? 0).toLocaleString()} docs · {(ds.chars ?? 0).toLocaleString()} chars</span>
                        </span>
                        <button type="button" className="btn btn-ghost btn-icon btn-sm" aria-label={`Delete dataset ${ds.name}`}
                                onClick={(e) => { e.preventDefault(); if (datasetName === ds.name) setDatasetName(""); deleteDataset(computeUrl(), ds.name).then((l) => Array.isArray(l) && setDatasets(l)); }}>
                          <Trash2 size={14} />
                        </button>
                      </label>
                    ))}
                  </div>
                )}

                {compute !== "demo" && (
                  <div className="prep">
                    <div className="row">
                      <button type="button" className="btn" onClick={doPreview} disabled={previewing || !dataReady}><Eye size={14} />{previewing ? "Reading your data…" : "Preview data"}</button>
                      {source !== "dataset" && dataReady && (
                        <button type="button" className="btn btn-ghost" onClick={saveCurrentDataset} disabled={savingDataset}><Bookmark size={14} />{savingDataset ? "Saving…" : "Save as dataset"}</button>
                      )}
                    </div>
                    {preview && (preview.error ? (
                      <div className="note err"><AlertTriangle size={15} /><span>{preview.error}</span></div>
                    ) : (
                      <div className="ui-card prep-card">
                        <div className="prep-stats num">
                          <span><b>{(preview.documents ?? 0).toLocaleString()}</b> docs</span>
                          <span><b>{(preview.chars ?? 0).toLocaleString()}</b> chars</span>
                          <span>~<b>{(preview.est_tokens ?? 0).toLocaleString()}</b> {preview.tokenizer} tokens</span>
                          {preview.sources > 1 && <span><b>{preview.sources}</b> sources</span>}
                        </div>
                        {preview.errors?.length ? <div className="warn-text">{preview.errors.length} source(s) failed to read</div> : null}
                        {(preview.documents ?? 0) === 0 && <div className="warn-text">No text found — check the path or URL.</div>}
                        {preview.preview && <pre className="prep-sample">{preview.preview}</pre>}
                      </div>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <div className="note"><Zap size={15} /><span>This template generates its own synthetic data — nothing to add here.</span></div>
            )}
          </div>
        )}

        {/* ---- Step 2: Model ---- */}
        {step === 2 && (
          <div className="stack">
            {isAsr || isVlm || isTts || isFinetune ? (
              <div className="field-row">
                {isAsr && (
                  <label className="field"><span className="field-l">Whisper model</span>
                    <input className="input mono" list="asr-models" value={asrBase} onChange={(e) => setAsrBase(e.target.value)} placeholder="openai/whisper-tiny" />
                    <datalist id="asr-models">{ASR_MODELS.map((m) => <option key={m} value={m} />)}</datalist>
                    <span className="field-h">whisper-tiny and base run on CPU with LoRA; small and up want a GPU. Downloads on first use.</span>
                  </label>
                )}
                {isVlm && (
                  <label className="field"><span className="field-l">Vision model</span>
                    <input className="input mono" list="vlm-models" value={vlmBase} onChange={(e) => setVlmBase(e.target.value)} placeholder="nlpconnect/vit-gpt2-image-captioning" />
                    <datalist id="vlm-models">{VLM_MODELS.map((m) => <option key={m} value={m} />)}</datalist>
                    <span className="field-h">A ViT encoder with a GPT-2 decoder, adapted with LoRA. Runs on CPU; downloads on first use.</span>
                  </label>
                )}
                {isTts && (
                  <label className="field"><span className="field-l">Voice model</span>
                    <input className="input mono" list="tts-models" value={ttsBase} onChange={(e) => setTtsBase(e.target.value)} placeholder="microsoft/speecht5_tts" />
                    <datalist id="tts-models">{TTS_MODELS.map((m) => <option key={m} value={m} />)}</datalist>
                    <span className="field-h">SpeechT5 (text → mel) with a HiFi-GAN vocoder, adapted with LoRA. Fine-tunes on CPU.</span>
                  </label>
                )}
                {isFinetune && (
                  <>
                    <label className="field"><span className="field-l">Base model</span>
                      <input className="input mono" list="base-models" value={baseModel} onChange={(e) => setBaseModel(e.target.value)} placeholder="distilgpt2" />
                      <datalist id="base-models">{BASE_MODELS.map((m) => <option key={m} value={m} />)}</datalist>
                      <span className="field-h">
                        {caps ? `${baseModel}: ${caps.cloud ? "runs on the remote GPU" : (caps.bases?.[baseModel] ? TIER_LABEL[caps.bases[baseModel].tier] : "depends on your machine")}`
                              : "distilgpt2 · SmolLM-135M · pythia-160m are CPU-friendly; bigger bases want a GPU."}
                      </span>
                    </label>
                    <label className="field"><span className="field-l">Method</span>
                      <select className="select" value={method} onChange={(e) => setMethod(e.target.value)}>
                        <option value="lora">LoRA — fast, recommended</option><option value="full">Full fine-tune — small models only</option>
                      </select>
                      <span className="field-h">LoRA trains a small adapter (~0.5M params on distilgpt2) and keeps the base frozen.</span>
                    </label>
                  </>
                )}
              </div>
            ) : (
              <div className="size-grid" role="radiogroup" aria-label="Model size">
                {SIZES.map((sz) => {
                  const tier = sizeGuide[sz.id]?.tier;
                  return (
                    <button key={sz.id} type="button" role="radio" aria-checked={size === sz.id} className={`size${size === sz.id ? " sel" : ""}`} onClick={() => setSize(sz.id)}>
                      <span className="size-bars" aria-hidden="true">{[0, 1, 2, 3, 4].map((b) => <i key={b} className={b <= SIZES.indexOf(sz) ? "on" : ""} />)}</span>
                      <span className="recipe-t">{sz.name}</span>
                      <span className="size-p num">~{sizeGuide[sz.id]?.params_m ?? SIZE_PARAMS[sz.id]}M params</span>
                      {tier && <span className={`tier tier-${tier}`}>{TIER_LABEL[tier]}</span>}
                    </button>
                  );
                })}
              </div>
            )}
            {needsData && !isAsr && !isVlm && !isTts && (
              <details className="adv">
                <summary><ChevronRight size={14} className="wiz-more-c" aria-hidden="true" />Advanced — production training config<span className="subtle"> · optional, defaults are tuned</span></summary>
                <div className="adv-body">
                  <div className="adv-group">Optimization</div>
                  <div className="adv-grid">
                    <label>Learning rate<input className="input mono" type="number" step="0.0001" placeholder={isFinetune ? "2e-4" : "1e-3"} value={train.lr} onChange={(e) => setT("lr", e.target.value)} /></label>
                    <label>LR schedule<select className="select" value={train.schedule} onChange={(e) => setT("schedule", e.target.value)}><option value="cosine">Cosine decay</option><option value="linear">Linear decay</option><option value="constant">Constant</option></select></label>
                    <label>Warmup steps<input className="input mono" type="number" min={0} step={10} value={train.warmup} onChange={(e) => setT("warmup", e.target.value)} /></label>
                    <label>Min LR<input className="input mono" type="number" step="0.00001" value={train.minLr} onChange={(e) => setT("minLr", e.target.value)} /></label>
                    <label>Weight decay<input className="input mono" type="number" min={0} step="0.001" value={train.weightDecay} onChange={(e) => setT("weightDecay", e.target.value)} /></label>
                    <label>Gradient clip<input className="input mono" type="number" min={0} step="0.1" value={train.gradClip} onChange={(e) => setT("gradClip", e.target.value)} /></label>
                  </div>
                  <SchedulePreview train={train} baseLr={isFinetune ? 2e-4 : 1e-3} />

                  <div className="adv-group">Batch &amp; duration</div>
                  <div className="adv-grid">
                    <label>Batch size<input className="input mono" type="number" min={1} value={train.batchSize} onChange={(e) => setT("batchSize", e.target.value)} /></label>
                    <label>Grad accumulation<input className="input mono" type="number" min={1} value={train.gradAccum} onChange={(e) => setT("gradAccum", e.target.value)} /></label>
                    <label>Max steps<input className="input mono" type="number" min={0} step={100} value={train.maxSteps} onChange={(e) => setT("maxSteps", e.target.value)} /></label>
                    <label>Context length<input className="input mono" type="number" min={16} max={2048} step={16} value={blockSize} onChange={(e) => setBlockSize(e.target.value)} /></label>
                    <label>Checkpoint every<input className="input mono" type="number" min={0} step={10} value={train.checkpointEvery} onChange={(e) => setT("checkpointEvery", e.target.value)} /></label>
                  </div>
                  <div className="adv-hint">Effective batch <b>{(Number(train.batchSize) || 1) * (Number(train.gradAccum) || 1)}</b> sequences per step · {Number(train.maxSteps) > 0 ? `stops at ${train.maxSteps} steps` : "trains until you pause"}{Number(train.checkpointEvery) > 0 ? ` · checkpoint every ${train.checkpointEvery} steps, rewindable` : ""}</div>

                  <div className="adv-group">Regularization</div>
                  <div className="adv-grid">
                    {!isFinetune && (<label>Dropout<input className="input mono" type="number" min={0} max={0.9} step="0.05" value={train.dropout} onChange={(e) => setT("dropout", e.target.value)} /></label>)}
                    <label>Label smoothing<input className="input mono" type="number" min={0} max={0.5} step="0.01" value={train.labelSmoothing} onChange={(e) => setT("labelSmoothing", e.target.value)} /></label>
                    <label>Early stopping<select className="select" value={earlyStop ? "on" : "off"} onChange={(e) => setEarlyStop(e.target.value === "on")}><option value="off">Off — run to the end</option><option value="on">On — stop when it plateaus</option></select></label>
                    {earlyStop && (<label>Patience (evals)<input className="input mono" type="number" min={2} max={20} value={patience} onChange={(e) => setPatience(e.target.value)} /></label>)}
                  </div>

                  <div className="adv-group">Performance &amp; reproducibility</div>
                  <div className="adv-grid">
                    <label>Precision<select className="select" value={train.precision} onChange={(e) => setT("precision", e.target.value)}><option value="fp32">fp32 — full precision</option><option value="bf16">bf16 — faster, stable</option><option value="fp16">fp16 — GPU, scaled</option></select></label>
                    <label>Seed<input className="input mono" type="number" value={train.seed} onChange={(e) => setT("seed", e.target.value)} /></label>
                  </div>

                  {isFinetune && method === "lora" && (<>
                    <div className="adv-group">LoRA adapter</div>
                    <div className="adv-grid">
                      <label>Rank (r)<input className="input mono" type="number" min={1} max={256} value={lora.r} onChange={(e) => setL("r", e.target.value)} /></label>
                      <label>Alpha<input className="input mono" type="number" min={1} max={512} value={lora.alpha} onChange={(e) => setL("alpha", e.target.value)} /></label>
                      <label>LoRA dropout<input className="input mono" type="number" min={0} max={0.9} step="0.01" value={lora.dropout} onChange={(e) => setL("dropout", e.target.value)} /></label>
                    </div>
                  </>)}

                  {!isFinetune && (<>
                    <div className="adv-group">Architecture &amp; data</div>
                    <div className="adv-grid">
                      <label>Tokenizer<select className="select" value={tokKind} onChange={(e) => setTokKind(e.target.value)}><option value="char">Characters — simplest</option><option value="bpe">Subword BPE — better on real text</option></select></label>
                      {tokKind === "bpe" && (<label>Vocab size<input className="input mono" type="number" min={256} max={4096} step={64} value={vocabSize} onChange={(e) => setVocabSize(e.target.value)} /></label>)}
                      <label>Chunking<select className="select" value={strategy} onChange={(e) => setStrategy(e.target.value)}><option value="window">Sliding window</option><option value="by_document">By document</option></select></label>
                      <label>Mixture of Experts<select className="select" value={moeExperts} onChange={(e) => setMoeExperts(Number(e.target.value))}><option value={0}>Off — dense FFN</option><option value={4}>4 experts (sparse)</option><option value={8}>8 experts (sparse)</option></select></label>
                      {moeExperts > 0 && (<label>Active per token<select className="select" value={moeTopK} onChange={(e) => setMoeTopK(Number(e.target.value))}><option value={1}>top-1</option><option value={2}>top-2</option></select></label>)}
                    </div>
                  </>)}
                </div>
              </details>
            )}
          </div>
        )}

        {/* ---- Step 3: Compute ---- */}
        {step === 3 && (
          <div className="stack">
            <div className="choice-grid" role="radiogroup" aria-label="Where to train">
              {COMPUTE.map((c) => {
                const Icon = c.icon;
                return (
                  <button key={c.id} type="button" role="radio" aria-checked={compute === c.id} className={`choice${compute === c.id ? " sel" : ""}`} onClick={() => setCompute(c.id)}>
                    <span className="recipe-i"><Icon size={17} strokeWidth={1.75} aria-hidden="true" /></span>
                    <span className="recipe-t">{c.name}</span>
                    <span className="recipe-d">{c.blurb}</span>
                    {c.id === "local" && engine && (
                      <span className={`choice-status ${engine.online ? "ok" : engine.online === false ? "off" : ""}`}>
                        <span className={`status-dot ${engine.online ? "ok" : engine.online === false ? "off" : "idle"}`} />
                        {engine.online ? "Engine running" : engine.online === false ? "Engine not running" : "Checking…"}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
            {compute === "local" && engineDown && (
              <div className="note warn">
                <AlertTriangle size={15} aria-hidden="true" />
                <span className="grow">The local engine isn't running yet. Start it and this page picks it up — or train anyway and the studio waits for it.</span>
                <button type="button" className="btn btn-sm" onClick={onEngineHelp}>How to start</button>
              </div>
            )}
            {compute === "cloud" && (
              <label className="field"><span className="field-l">Engine URL</span>
                <input className="input mono" value={cloudUrl} onChange={(e) => setCloudUrl(e.target.value)} placeholder="wss://your-training-engine.example.com" />
                <span className="field-h">Run <code>python engine.py</code> on the GPU box and expose its WebSocket. The studio works the same either way.</span>
              </label>
            )}
            {caps && (compute === "local" || compute === "cloud") && !engineDown && (
              <div className="cap-line">
                <span className="status-dot ok" />
                {caps.cloud ? "Remote engine — a GPU handles every size."
                  : caps.source === "browser" ? `~${caps.cpu_cores || "?"}-core CPU detected by the browser · start the engine to detect your GPU`
                  : `This machine: ${caps.cpu_cores} cores · ${caps.ram_gb} GB RAM · ${caps.accelerator}${caps.accelerator_note ? ` — ${caps.accelerator_note}` : ""}`}
              </div>
            )}
            {compute === "local" && selectedTier && selectedTier !== "local" && (
              <div className="note warn"><Zap size={15} aria-hidden="true" /><span>{selectedTier === "cloud"
                ? "That's a lot for this machine — a GPU engine will be far faster. It will still train here, just slowly."
                : "This will train slowly on CPU — a GPU engine would speed it up."}</span></div>
            )}
          </div>
        )}

        {/* ---- Step 4: Review ---- */}
        {step === 4 && (
          <div className="review-v2">
            <div className="ui-card">
              <dl className="review-list">
                <div><dt>Starting point</dt><dd>{recipeObj ? `${recipeObj.name} recipe` : tplObj?.name}</dd><button type="button" className="btn-link" onClick={() => setStep(0)}>Change</button></div>
                <div><dt>Data</dt><dd>{dataLabel}</dd><button type="button" className="btn-link" onClick={() => setStep(1)}>Change</button></div>
                <div><dt>Model</dt><dd>{modelLabel}</dd><button type="button" className="btn-link" onClick={() => setStep(2)}>Change</button></div>
                {!isFinetune && moeExperts > 0 && <div><dt>Experts</dt><dd>{moeExperts} experts · top-{moeTopK}</dd><span /></div>}
                {needsData && !isFinetune && !usesAudioPairs && !isVlm && <div><dt>Tokenizer</dt><dd>{tokKind === "bpe" ? `Subword BPE · ${vocabSize}` : "Characters"}</dd><span /></div>}
                <div><dt>Trains on</dt><dd>{COMPUTE.find((c) => c.id === compute)?.name}{compute === "cloud" ? ` · ${cloudUrl}` : ""}</dd><button type="button" className="btn-link" onClick={() => setStep(3)}>Change</button></div>
              </dl>
            </div>
            <label className="field">
              <span className="field-l">Model name</span>
              <input className="input review-name" value={name} onChange={(e) => setName(e.target.value)} aria-label="Model name" />
              <span className="field-h">Used for the saved checkpoint and its endpoint: <code>/v1/{name.trim() || "my-model"}/generate</code></span>
            </label>
          </div>
        )}
      </div>

      <div className="wiz-foot">
        <button type="button" className="btn btn-ghost" onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step === 0}><ChevronLeft size={15} />Back</button>
        <span className="wiz-prog num" aria-live="polite">Step {step + 1} of {STEPS.length}</span>
        {step < 4 ? (
          <button type="button" className="btn btn-primary" title="Enter" onClick={() => setStep((s) => Math.min(4, s + 1))} disabled={!canNext}>Continue<ChevronRight size={15} /></button>
        ) : (
          <button type="button" className="btn btn-primary" title="⌘/Ctrl + Enter" disabled={!ready} onClick={launch}>
            <Rocket size={15} />{compute === "demo" ? "Open demo" : "Create & train"}
          </button>
        )}
      </div>
    </div>
  );
}
