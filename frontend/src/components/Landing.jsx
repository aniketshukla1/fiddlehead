import React, { useEffect, useState } from "react";
import {
  ArrowRight, Github, BookOpen, Terminal, Check, ChevronDown, Eye, Wand2, AudioLines,
  SlidersHorizontal, Network, Radio,
} from "lucide-react";
import Logo, { Mark, SPIRAL } from "./Logo.jsx";
import ThemeToggle from "./ThemeToggle.jsx";
import HeroScope from "./HeroScope.jsx";
import CodeBlock from "./CodeBlock.jsx";
import { useReveal } from "../lib/motion.js";
import { BRAND, GITHUB_URL } from "../lib/brand.js";

// The public front door, written for an open-source launch: what it is, what it
// looks like running, how to self-host it. Plans appear only on a deployment
// that sells them (the API reports billing as configured).

const MODELS = ["GPT-2", "DistilGPT-2", "SmolLM", "Qwen2.5", "Pythia", "TinyLlama", "Whisper", "SpeechT5", "ViT-GPT2"];

const QUICKSTART = `# 1 · the training engine
cd backend
pip install -r requirements.txt
python engine.py

# 2 · the API: accounts, models, keys  (second terminal)
cd backend && uvicorn api.app:app --port 8000

# 3 · the app  (third terminal)
cd frontend && npm install && npm run dev`;

const API_SNIPPET = `curl localhost:8000/v1/my-model/generate \\
  -H "Authorization: Bearer sk-…" \\
  -d '{"prompt": "Once upon a time"}'`;

const PLANS = [
  { id: "free", name: "Free", price: "$0", per: "forever", blurb: "Everything you need for your first model.",
    feats: ["5 compute credits", "1 workspace", "Train, fine-tune & deploy", "Community support"] },
  { id: "pro", name: "Pro", price: "$29", per: "/mo", blurb: "For builders shipping real models.", featured: true,
    feats: ["500 credits / month", "Cloud GPU training", "Hosted, metered endpoints", "Email support"] },
  { id: "team", name: "Team", price: "$99", per: "/mo", blurb: "For teams building together.",
    feats: ["2,000 credits / month", "Unlimited seats & roles", "Audit log & two-factor auth", "Priority support"] },
];

const faq = (billing) => [
  { q: "Do I need a GPU?", a: "No. Small models train on a laptop CPU in minutes, and Fiddlehead uses Apple Silicon or CUDA automatically when it finds them. For bigger base models, run the engine on a GPU server and connect to it by URL." },
  { q: "What can I build with it?", a: "Focused models: autocomplete in your own writing style, code completion for your codebase, a domain assistant fine-tuned from SmolLM or Qwen, speech recognition tuned to your accent, a custom voice, an image captioner. Small and specific — not a replacement for frontier chat models." },
  { q: "Where does my data go?", a: "Only to the training engine you connect to — the one on your machine, or a server you run. It's used to train your model and nothing else." },
  { q: "How is this different from a notebook?", a: "A notebook gives you a loss curve when you remember to plot it. Fiddlehead streams the whole run — loss, gradients, attention circuits, samples — handles tokenizing, splitting and checkpoints for you, and turns the result into an API with one switch." },
  { q: "Can I take my model elsewhere?", a: "Yes. Download any model as a .zip — weights, tokenizer and metadata — and serve it anywhere with the included serve.py, or call it through the built-in API." },
  billing
    ? { q: "Is there a free option?", a: "Yes. The open-source version is free to run yourself. Hosted plans add managed GPUs and endpoints." }
    : { q: "Is it really free?", a: "Yes. It's open source: run it yourself, change it, ship it." },
];

// Large, faint fern spirals that slowly unfurl behind the hero.
function Fronds() {
  const fronds = [
    { left: "-6%", top: "4%", size: 520, rot: -18, delay: 0 },
    { left: "74%", top: "-6%", size: 440, rot: 24, delay: 2.4 },
    { left: "82%", top: "46%", size: 300, rot: -40, delay: 4.2 },
  ];
  return (
    <div className="fronds" aria-hidden="true">
      {fronds.map((f, i) => (
        <svg key={i} viewBox="0 0 24 24" style={{ left: f.left, top: f.top, width: f.size, height: f.size, rotate: `${f.rot}deg`, animationDelay: `${f.delay}s` }}>
          <path d={SPIRAL} pathLength="1" style={{ animationDelay: `${f.delay}s` }} />
        </svg>
      ))}
    </div>
  );
}

function Section({ id, eyebrow, title, lede, children, className = "" }) {
  const [ref, shown] = useReveal();
  return (
    <section id={id} ref={ref} className={`lp2-sec ${className}`}>
      {title && (
        <div className={`lp2-head reveal ${shown ? "in" : ""}`}>
          {eyebrow && <div className="lp2-eyebrow">{eyebrow}</div>}
          <h2>{title}</h2>
          {lede && <p>{lede}</p>}
        </div>
      )}
      <div className={`stagger ${shown ? "in" : ""}`}>{children}</div>
    </section>
  );
}

export default function Landing({ features = {} }) {
  const [scrolled, setScrolled] = useState(false);
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    window.addEventListener("scroll", onScroll, { passive: true });
    const r = requestAnimationFrame(() => setMounted(true));
    return () => { window.removeEventListener("scroll", onScroll); cancelAnimationFrame(r); };
  }, []);
  const scrollTo = (id) => document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  const on = mounted ? "in" : "";

  return (
    <div className="lp2">
      <nav className={`lp2-nav${scrolled ? " scrolled" : ""}`} aria-label="Main">
        <div className="lp2-nav-in">
          <Logo href="#/" />
          <div className="lp2-links">
            <button type="button" onClick={() => scrollTo("features")}>Features</button>
            <button type="button" onClick={() => scrollTo("how")}>How it works</button>
            <button type="button" onClick={() => scrollTo("self-host")}>Self-host</button>
            <a href="#/docs">Docs</a>
          </div>
          <div className="lp2-nav-r">
            {GITHUB_URL && <a className="btn btn-ghost btn-icon" href={GITHUB_URL} aria-label="GitHub"><Github size={17} /></a>}
            <ThemeToggle />
            <a className="btn btn-ghost lp2-signin" href="#/signin">Sign in</a>
            <a className="btn btn-primary" href="#/signup">Get started</a>
          </div>
        </div>
      </nav>

      <header className="lp2-hero">
        <Fronds />
        <div className="lp2-hero-in">
          <button type="button" className={`lp2-pill reveal d1 ${on}`} onClick={() => scrollTo("self-host")}>
            <span className="lp2-pill-tag">Open source</span>Runs on your own machine<ArrowRight size={13} aria-hidden="true" />
          </button>
          <h1 className={`reveal d2 ${on}`}>Grow your own AI models.<br /><span className="lp2-accent">Watch them unfurl.</span></h1>
          <p className={`lp2-lede reveal d3 ${on}`}>
            {BRAND} is an open-source studio for training, fine-tuning and serving small language, speech and
            vision models — on your laptop or your own GPUs — with a live view inside the network as it learns.
          </p>
          <div className={`lp2-cta reveal d4 ${on}`}>
            <a className="btn btn-primary btn-xl" href="#/signup">Start building<ArrowRight size={16} aria-hidden="true" /></a>
            {GITHUB_URL
              ? <a className="btn btn-xl" href={GITHUB_URL}><Github size={16} aria-hidden="true" />Star on GitHub</a>
              : <button type="button" className="btn btn-xl" onClick={() => scrollTo("self-host")}><Terminal size={16} aria-hidden="true" />Run it locally</button>}
          </div>
          <ul className={`lp2-proof reveal d5 ${on}`}>
            <li><Check size={14} aria-hidden="true" />No GPU required</li>
            <li><Check size={14} aria-hidden="true" />Your data stays on your hardware</li>
            <li><Check size={14} aria-hidden="true" />Text, speech and vision</li>
          </ul>
        </div>
        <div className={`lp2-scope reveal d5 ${on}`}>
          <HeroScope />
        </div>
      </header>

      <section className="lp2-models" aria-label="Supported models">
        <span>Fine-tune the open models you already know</span>
        <div className="lp2-chips">{MODELS.map((m) => <span key={m} className="lp2-chip">{m}</span>)}</div>
      </section>

      <Section id="features" eyebrow="Features" title="A training run you can actually see"
               lede="Most tools give you a progress bar and a loss number. Fiddlehead shows you what the model is learning, while it learns.">
        <div className="bento">
          <article className="bento-card span2 bento-see">
            <div className="bento-text">
              <span className="bento-i"><Eye size={18} aria-hidden="true" /></span>
              <h3>See inside the network</h3>
              <p>Every attention head is drawn live and labeled by the behavior it's picking up — previous-token,
                positional, copying. Induction heads light up the moment in-context learning appears. Click a head to
                switch it off and watch the quality move.</p>
            </div>
            <div className="bento-viz"><HeroScope compact /></div>
          </article>
          <article className="bento-card">
            <span className="bento-i"><Wand2 size={18} aria-hidden="true" /></span>
            <h3>Fine-tune real models</h3>
            <p>LoRA or full fine-tuning on GPT-2, SmolLM, Qwen2.5, Pythia and TinyLlama — or a transformer from scratch, sized in plain words.</p>
          </article>
          <article className="bento-card">
            <span className="bento-i"><AudioLines size={18} aria-hidden="true" /></span>
            <h3>Text, speech and vision</h3>
            <p>Adapt Whisper to your accent, give SpeechT5 a voice, or teach a captioner your images. Same studio, same workflow.</p>
          </article>
          <article className="bento-card">
            <span className="bento-i"><SlidersHorizontal size={18} aria-hidden="true" /></span>
            <h3>Production training knobs</h3>
            <p>Warmup and cosine schedules, gradient accumulation, bf16 and fp16, clipping, label smoothing, early stopping — and checkpoints you can rewind to.</p>
          </article>
          <article className="bento-card">
            <span className="bento-i"><Network size={18} aria-hidden="true" /></span>
            <h3>Scale past one GPU</h3>
            <p>A planner picks data-parallel, sharded or tensor-plus-pipeline for your model and estimates memory, time and cost. Real PyTorch DDP underneath.</p>
          </article>
          <article className="bento-card bento-api">
            <div className="bento-text">
              <span className="bento-i"><Radio size={18} aria-hidden="true" /></span>
              <h3>Serve it as an API</h3>
              <p>Flip a switch to put a saved text model behind an HTTP endpoint, create a key, and call it from anywhere —
                or download the weights and run them yourself.</p>
            </div>
            <CodeBlock code={API_SNIPPET} lang="shell" />
          </article>
        </div>
      </Section>

      <Section id="how" eyebrow="How it works" title="From raw text to an endpoint in three steps">
        <ol className="how">
          <li>
            <span className="how-n">1</span>
            <h3>Bring your data</h3>
            <p>Paste it, drop in files and PDFs, point at a folder or URL, crawl a site, or read from S3 and GCS. It's cleaned, de-duplicated and split for you.</p>
            <div className="how-tags"><span>paste</span><span>.pdf</span><span>https://</span><span>s3://</span></div>
          </li>
          <li>
            <span className="how-n">2</span>
            <h3>Train and watch</h3>
            <p>Pick a recipe and a size in plain words. Loss, gradients, attention circuits and live samples stream in — pause, rewind, or keep the best checkpoint.</p>
            <div className="how-tags"><span>loss ↓</span><span>heads</span><span>samples</span></div>
          </li>
          <li>
            <span className="how-n">3</span>
            <h3>Deploy and call</h3>
            <p>Save the model, switch on its endpoint and create a key. Chat with it in the studio, or call it over HTTP from your own code.</p>
            <div className="how-tags"><span>POST /v1/…/generate</span></div>
          </li>
        </ol>
      </Section>

      <Section id="self-host" className="lp2-qs">
        <div className="lp2-qs-grid">
          <div className="lp2-qs-text">
            <div className="lp2-eyebrow">Self-host</div>
            <h2>Running in a couple of minutes</h2>
            <p>{BRAND} is three small processes you run yourself: the training engine (PyTorch), an API for accounts,
              models and keys (FastAPI and SQLite), and this app (React). No cloud account needed.</p>
            <ul className="lp2-list">
              <li><Check size={14} aria-hidden="true" />Uses Apple Silicon or CUDA automatically, the CPU otherwise</li>
              <li><Check size={14} aria-hidden="true" />Point the studio at a GPU server with a remote engine URL</li>
              <li><Check size={14} aria-hidden="true" />Prefer a desktop app? <code>npm run tauri dev</code></li>
            </ul>
            <div className="row">
              {GITHUB_URL && <a className="btn btn-primary" href={GITHUB_URL}><Github size={15} />Get the code</a>}
              <a className="btn" href="#/docs"><BookOpen size={15} />Read the docs</a>
            </div>
          </div>
          <CodeBlock code={QUICKSTART} lang="shell" className="lp2-term" />
        </div>
      </Section>

      {features.billing && (
        <Section id="pricing" eyebrow="Pricing" title="Start free. Pay only when you scale."
                 lede="The open-source studio is free to run yourself. Hosted plans add managed GPUs and endpoints.">
          <div className="plans">
            {PLANS.map((p) => (
              <div className={`ui-card plan${p.featured ? " featured" : ""}`} key={p.id}>
                {p.featured && <span className="badge badge-ok plan-tag">Most popular</span>}
                <div className="plan-n">{p.name}</div>
                <div className="plan-p"><b>{p.price}</b><span>{p.per}</span></div>
                <p className="plan-b">{p.blurb}</p>
                <ul>{p.feats.map((f) => <li key={f}><Check size={14} aria-hidden="true" />{f}</li>)}</ul>
                <a className={`btn btn-block ${p.featured ? "btn-primary" : ""}`} href="#/signup">Start {p.name}</a>
              </div>
            ))}
          </div>
        </Section>
      )}

      <Section id="faq" eyebrow="FAQ" title="Questions, answered" className="lp2-faq">
        <div className="faq">
          {faq(features.billing).map((f) => (
            <details className="faq-i" key={f.q}>
              <summary>{f.q}<ChevronDown size={16} className="faq-c" aria-hidden="true" /></summary>
              <p>{f.a}</p>
            </details>
          ))}
        </div>
      </Section>

      <section className="lp2-final">
        <div className="unfurl lp2-final-mark"><Mark size={48} /></div>
        <h2>Grow your first model in the next five minutes.</h2>
        <p>Train the 30-second sample, then make it yours.</p>
        <a className="btn btn-primary btn-xl" href="#/signup">Start building<ArrowRight size={16} aria-hidden="true" /></a>
      </section>

      <footer className="lp2-foot">
        <div className="lp2-foot-in">
          <div className="lp2-foot-brand">
            <Logo href="#/" />
            <p>Train, watch and serve your own models.</p>
          </div>
          <nav className="lp2-foot-links" aria-label="Footer">
            <a href="#/docs">Docs</a>
            {GITHUB_URL && <a href={GITHUB_URL}>GitHub</a>}
            <a href="#/signin">Sign in</a>
            <a href="#/signup">Get started</a>
          </nav>
        </div>
        <div className="lp2-foot-b">
          <span>© {new Date().getFullYear()} {BRAND} contributors</span>
          <span>Open source · made for people who like to see how things work</span>
        </div>
      </footer>
    </div>
  );
}
