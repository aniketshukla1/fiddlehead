import React, { useEffect, useRef, useState } from "react";

// A real chat window for talking to your model in natural language, with live
// sampling controls (temperature / top-p / top-k / max tokens) and a system
// prompt. Multi-turn: the whole conversation is sent each time so the model has
// context. On a small from-scratch model this "continues" more than converses;
// on a fine-tuned instruct model it genuinely chats.

function Param({ label, value, min, max, step, onChange, hint }) {
  return (
    <label className="cp-param">
      <span className="cp-row"><span>{label}</span><b>{value}</b></span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(e.target.value)} />
      {hint && <span className="cp-hint">{hint}</span>}
    </label>
  );
}

export default function Chat({ meta, sample, error, send }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [system, setSystem] = useState("");
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [topK, setTopK] = useState(40);
  const [maxTokens, setMaxTokens] = useState(96);
  const [pending, setPending] = useState(false);
  const pendingId = useRef(null);
  const streaming = useRef(false);
  const timer = useRef(null);
  const scrollRef = useRef(null);

  // the reply streams in as "sample" frames tagged with our request id: append
  // the assistant bubble once, then keep updating it until the final frame.
  useEffect(() => {
    if (!(sample && sample.chat && sample.id === pendingId.current)) return;
    setMessages((m) => {
      if (!streaming.current) {
        streaming.current = true;
        return [...m, { role: "assistant", content: sample.text || "" }];
      }
      const copy = m.slice();
      copy[copy.length - 1] = { role: "assistant", content: sample.text || "" };
      return copy;
    });
    if (sample.done !== false) {
      setPending(false);
      clearTimeout(timer.current);
      pendingId.current = null;
      streaming.current = false;
    }
  }, [sample]);

  // never leave the Send button stuck if generation errors
  useEffect(() => {
    if (error && pending) { setPending(false); clearTimeout(timer.current); pendingId.current = null; streaming.current = false; }
  }, [error]); // eslint-disable-line

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages, pending]);

  function sendMsg() {
    const text = input.trim();
    if (!text || pending) return;
    const next = [...messages, { role: "user", content: text }];
    setMessages(next);
    setInput("");
    const id = `chat-${Date.now()}`;
    pendingId.current = id;
    streaming.current = false;
    setPending(true);
    send?.({
      type: "generate", id, messages: next, system: system || undefined,
      temperature: Number(temperature), top_k: Number(topK), top_p: Number(topP),
      max_new_tokens: Number(maxTokens),
    });
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setPending(false), 30000); // safety net
  }

  return (
    <div className="chat">
      <div className="chat-main">
        <div className="chat-scroll" ref={scrollRef}>
          {messages.length === 0 && (
            <div className="chat-empty">
              Ask your model anything.<br />
              <span>A from-scratch model will <em>continue</em> your text more than converse — fine-tune an open-source model for real answers, then tune the sampling on the right.</span>
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`bubble ${m.role}`}>
              <span className="bubble-role">{m.role === "user" ? "you" : (meta?.base_model || "model")}</span>
              {m.content}
            </div>
          ))}
          {pending && <div className="bubble assistant pending"><span className="bubble-role">{meta?.base_model || "model"}</span><span className="typing">▋</span></div>}
        </div>
        <div className="chat-input">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMsg(); } }}
            placeholder="Message your model…  (Enter to send · Shift+Enter for newline)"
            spellCheck={false}
          />
          <button onClick={sendMsg} disabled={pending}>Send</button>
        </div>
      </div>

      <aside className="chat-params">
        <div className="cp-title">Sampling</div>
        <Param label="Temperature" value={temperature} min={0} max={1.5} step={0.05} onChange={setTemperature} hint="higher = more random & creative" />
        <Param label="Top-p" value={topP} min={0} max={1} step={0.05} onChange={setTopP} hint="nucleus: smallest set covering this probability" />
        <Param label="Top-k" value={topK} min={0} max={100} step={1} onChange={setTopK} hint="0 = disabled" />
        <Param label="Max tokens" value={maxTokens} min={8} max={256} step={8} onChange={setMaxTokens} hint="length cap per reply" />
        <div className="cp-title">System prompt</div>
        <textarea className="cp-system" value={system} onChange={(e) => setSystem(e.target.value)} placeholder="(optional) You are a helpful assistant." spellCheck={false} />
        <button className="cp-clear" onClick={() => setMessages([])}>Clear conversation</button>
      </aside>
    </div>
  );
}
