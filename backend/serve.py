"""
serve.py — serve a saved Fiddlehead model as a tiny HTTP API.

    python serve.py <model-name>            # a name under saved/, or a path
    python serve.py my-model --port 8770

Then call it from anywhere:

    curl -s localhost:8770/generate \
      -H 'content-type: application/json' \
      -d '{"prompt": "Neural networks are", "max_new_tokens": 40}'
    -> {"text": "...", "prompt": "Neural networks are"}

GET / returns the model's identity. This is how a trained model "travels" — save
it in the Studio, then serve it as an endpoint your own code can hit.
"""

from __future__ import annotations
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from artifacts import META, SAVE_DIR, meta_path
from model import GPT, GPTConfig
from tokenizer import load_tokenizer



def load_model(name: str, device: str = "cpu"):
    """Return (generate(prompt, params) -> str, meta) for a saved model — either
    a from-scratch GPT or a fine-tuned HF model."""
    d = name if os.path.isdir(name) else os.path.join(SAVE_DIR, name)
    meta_file = meta_path(d)
    if not os.path.isfile(meta_file):
        raise SystemExit(f"no saved model at '{d}' (need {META})")
    meta = json.load(open(meta_file))

    if meta.get("kind") == "finetune":
        from finetune import FineTuner
        ft = FineTuner.load(d, device)
        return (lambda prompt, params: ft.generate_text(prompt, params)), meta

    ckpt = torch.load(os.path.join(d, "model.pt"), weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tok = load_tokenizer(ckpt["tokenizer"])

    def generate(prompt, params):
        ids = torch.tensor([tok.encode(prompt) or [0]], dtype=torch.long, device=device)
        out = model.generate(
            ids, int(params.get("max_new_tokens", 64)),
            temperature=float(params.get("temperature", 0.8)),
            top_k=int(params.get("top_k", 40) or 0) or None,
            top_p=float(params.get("top_p", 1.0)))
        return tok.decode(out[0])

    return generate, meta


def make_handler(generate, meta):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.end_headers()

        def do_GET(self):
            self._send(200, {"ok": True, "model": meta.get("name"), "kind": meta.get("kind"),
                             "base_model": meta.get("base_model"), "template": meta.get("template")})

        def do_POST(self):
            if self.path.rstrip("/") != "/generate":
                self._send(404, {"error": "POST /generate"})
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                params = json.loads(self.rfile.read(n) or "{}")
            except Exception:
                self._send(400, {"error": "invalid JSON body"})
                return
            try:
                text = generate(str(params.get("prompt", "")), params)
            except Exception as e:
                self._send(500, {"error": str(e)})
                return
            self._send(200, {"text": text, "prompt": params.get("prompt", "")})

        def log_message(self, *a):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description="Serve a saved Fiddlehead model over HTTP")
    ap.add_argument("model", help="a name under saved/, or a path to a saved model dir")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    generate, meta = load_model(args.model)
    print(f"[serve] {meta.get('name')} ({meta.get('kind')}) → http://{args.host}:{args.port}")
    print(f"[serve] curl -s {args.host}:{args.port}/generate -d '{{\"prompt\":\"Hello\",\"max_new_tokens\":40}}'")
    ThreadingHTTPServer((args.host, args.port), make_handler(generate, meta)).serve_forever()


if __name__ == "__main__":
    main()
