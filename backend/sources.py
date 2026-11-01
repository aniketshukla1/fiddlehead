"""
sources.py — gather raw text from wherever the user's data lives.

Sources:
  • paste  — text pasted into the UI
  • path   — a local file or a directory (walks .txt/.md/.csv/.jsonl/.json/.py/…)
  • url    — fetch a single http(s) URL (raw text or an HTML page)
  • scrape — crawl one or more web pages, extract readable text, follow links

Everything comes back as a list of Documents (each a chunk of text + provenance)
so downstream steps can split train/val by document and avoid leakage. Pure
stdlib: urllib for fetching, an HTMLParser-based extractor for pages — no heavy
scraping dependencies to install.
"""

from __future__ import annotations
import csv
import io
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

USER_AGENT = "Fiddlehead/0.2 (+data-ingest)"
MAX_BYTES = 8_000_000          # per-fetch safety cap
PDF_MAX_CHARS = 1_400_000      # per-PDF extraction cap (keeps "preparing" bounded)
TEXT_EXTS = {".txt", ".md", ".markdown", ".rst", ".py", ".js", ".ts", ".json", ".jsonl", ".csv", ".log", ".pdf"}


@dataclass
class Document:
    text: str
    source: str
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# HTML -> readable text (stdlib)
# --------------------------------------------------------------------------- #
class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "template", "svg", "head"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
              "section", "article", "header", "footer", "blockquote", "pre"}

    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        if tag == "a":
            for k, v in attrs:
                if k == "href" and v:
                    self.links.append(v)
        if tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        if tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0 and data.strip():
            self.parts.append(data)

    def text(self) -> str:
        return _normalize("".join(self.parts))


def _normalize(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n[ \t]*(\n[ \t]*)+", "\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def html_to_text(html: str) -> tuple[str, list[str]]:
    p = _TextExtractor()
    try:
        p.feed(html)
    except Exception:
        pass
    return p.text(), p.links


# --------------------------------------------------------------------------- #
# readability: find the main content, drop nav / menus / sidebars / footers
# --------------------------------------------------------------------------- #
_STRIP = {"script", "style", "noscript", "template", "svg", "head", "nav", "aside",
          "footer", "form", "header", "button", "figure", "iframe", "object", "select", "label"}
_BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
          "section", "article", "blockquote", "pre", "ul", "ol", "table"}
_POS = re.compile(r"article|body|content|entry|main|page|post|blog|story|markdown|prose|text", re.I)
_NEG = re.compile(r"comb|comment|contact|foot|masthead|media|meta|nav|menu|sidebar|sponsor|"
                  r"advert|\bads?\b|banner|social|share|promo|related|breadcrumb|pager|popup|"
                  r"cookie|subscribe|newsletter|hidden|modal|widget|toolbar|header", re.I)
_VOID = {"br", "img", "hr", "input", "meta", "link", "source", "area", "base", "col", "embed", "param", "track", "wbr"}


class _Node:
    __slots__ = ("tag", "attrs", "parent", "seq", "score")

    def __init__(self, tag, attrs=None, parent=None):
        self.tag = tag
        self.attrs = attrs or {}
        self.parent = parent
        self.seq = []          # ordered mix of str (text) and child _Node
        self.score = 0.0

    def children(self):
        return [x for x in self.seq if isinstance(x, _Node)]


class _DOMBuilder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _Node("[root]")
        self.stack = [self.root]
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "a" and a.get("href"):
            self.links.append(a["href"])
        if tag in _VOID:
            return
        node = _Node(tag, a, self.stack[-1])
        self.stack[-1].seq.append(node)
        self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        a = dict(attrs)
        if tag == "a" and a.get("href"):
            self.links.append(a["href"])

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):   # tolerant: pop to match
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        if data.strip():
            self.stack[-1].seq.append(data)


def _text_of(node: _Node) -> str:
    if node.tag in _STRIP:
        return ""
    parts = []
    for item in node.seq:
        if isinstance(item, str):
            parts.append(item)
        else:
            s = _text_of(item)
            if s:
                parts.append("\n" + s + "\n" if item.tag in _BLOCK else s)
    return "".join(parts)


def _link_density(node: _Node) -> float:
    total = len(_text_of(node)) or 1
    link = 0
    stack = [node]
    while stack:
        n = stack.pop()
        if n.tag in _STRIP:
            continue
        if n.tag == "a":
            link += len(_text_of(n))
        stack.extend(n.children())
    return min(1.0, link / total)


def _candidate(node: _Node, cache: dict) -> _Node:
    if id(node) in cache:
        return cache[id(node)]
    base = {"article": 12, "main": 12, "section": 8, "div": 5, "td": 3, "blockquote": 3, "pre": 3}.get(node.tag, 0)
    ci = f"{node.attrs.get('class', '')} {node.attrs.get('id', '')}"
    if _POS.search(ci):
        base += 25
    if _NEG.search(ci):
        base -= 25
    node.score = base
    cache[id(node)] = node
    return node


def extract_main(html: str) -> tuple[str, list[str]]:
    """Readability-style main-content extraction. Scores block containers by the
    density of real paragraph text they hold (minus link density and nav-ish
    class/id names) and returns the winning container's text. Falls back to the
    whole-page extractor when scoring finds too little."""
    b = _DOMBuilder()
    try:
        b.feed(html)
    except Exception:
        return html_to_text(html)

    cache: dict = {}
    stack = [b.root]
    while stack:
        n = stack.pop()
        if n.tag in ("p", "pre", "blockquote", "li", "td"):
            t = _text_of(n).strip()
            if len(t) >= 25:
                sc = 1 + t.count(",") + min(len(t) // 100, 3)
                if n.parent:
                    _candidate(n.parent, cache).score += sc
                    if n.parent.parent:
                        _candidate(n.parent.parent, cache).score += sc * 0.5
        stack.extend(n.children())

    best, best_final = None, -1.0
    for node in cache.values():
        txt = _text_of(node)
        if len(txt.strip()) < 50:
            continue
        final = node.score * (1 - _link_density(node))
        if final > best_final:
            best_final, best = final, node

    main = _normalize(_text_of(best)) if best is not None else ""
    if len(main) < 200:  # too little confidence — prefer the fuller extraction
        full, _ = html_to_text(html)
        if len(full) > len(main):
            main = full
    return main, b.links


# --------------------------------------------------------------------------- #
# fetching
# --------------------------------------------------------------------------- #
def _fetch(url: str, timeout: float = 15.0) -> tuple[str, str]:
    """Returns (body_text, content_type). Raises on non-http(s) or errors."""
    if urlparse(url).scheme not in ("http", "https"):
        raise ValueError("only http(s) URLs are allowed")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        ctype = resp.headers.get("Content-Type", "")
        raw = resp.read(MAX_BYTES)
    charset = "utf-8"
    m = re.search(r"charset=([\w-]+)", ctype)
    if m:
        charset = m.group(1)
    return raw.decode(charset, errors="replace"), ctype


# --------------------------------------------------------------------------- #
# local files
# --------------------------------------------------------------------------- #
def _pdf_pages(path: str | None = None, stream: bytes | None = None):
    """Yield each page's raw text, from a file path or raw bytes (uploads).
    Prefers PyMuPDF (much better word spacing on the multi-column / justified
    layouts common in books — pypdf glues words like 'MedicalAIdemands') and
    falls back to pypdf."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        fitz = None
    if fitz is not None:
        doc = fitz.open(stream=stream, filetype="pdf") if stream is not None else fitz.open(path)
        try:
            for page in doc:
                yield page.get_text()
        finally:
            doc.close()
        return
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("PDF support needs pymupdf or pypdf — pip install pymupdf")
    import io
    reader = PdfReader(io.BytesIO(stream)) if stream is not None else PdfReader(path)
    for page in reader.pages:
        try:
            yield page.extract_text() or ""
        except Exception:
            yield ""


def _reflow_lines(t: str) -> str:
    """Merge lines wrapped mid-sentence (a line that doesn't end in punctuation
    followed by one that starts lowercase) back into flowing prose."""
    out: list[str] = []
    for line in t.split("\n"):
        s = line.strip()
        if not s:
            if out and out[-1] != "":
                out.append("")
            continue
        if out and out[-1] and not re.search(r"[.!?:;)]$", out[-1]) and s[0].islower():
            out[-1] += " " + s
        else:
            out.append(s)
    return "\n".join(out)


def _clean_pdf_text(t: str) -> str:
    t = re.sub(r"(?:\.\s*){4,}", " ", t)                  # TOC dot leaders
    t = re.sub(r"([A-Za-z])-\s*\n\s*([a-z])", r"\1\2", t)  # de-hyphenate line splits
    return _normalize(_reflow_lines(t))


def _pdf_docs(pages, name: str) -> list[Document]:
    """One cleaned Document per page (so the train/val split stays leak-free),
    stopping at the extraction cap."""
    docs, total = [], 0
    for i, raw in enumerate(pages):
        if total >= PDF_MAX_CHARS:
            break
        text = _clean_pdf_text(raw or "")
        if len(text) > 20:
            docs.append(Document(text, f"{name} p.{i + 1}", {"page": i + 1}))
            total += len(text)
    return docs


def _read_pdf(path: str) -> list[Document]:
    return _pdf_docs(_pdf_pages(path=path), os.path.basename(path))


def _read_file(path: str) -> list[Document]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _read_pdf(path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read(MAX_BYTES)
    if ext == ".jsonl":
        docs = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                docs.append(_json_text(obj))
            except json.JSONDecodeError:
                docs.append(line)
        return [Document(t, path, {"format": "jsonl"}) for t in docs if t]
    if ext == ".json":
        try:
            obj = json.loads(raw)
            texts = [_json_text(o) for o in obj] if isinstance(obj, list) else [_json_text(obj)]
            return [Document(t, path, {"format": "json"}) for t in texts if t]
        except json.JSONDecodeError:
            return [Document(raw, path, {"format": "text"})]
    if ext == ".csv":
        rows = list(csv.reader(io.StringIO(raw)))
        # take the longest text-ish column per row
        docs = []
        for r in rows[1:] if rows else []:
            cell = max(r, key=len) if r else ""
            if cell:
                docs.append(Document(cell, path, {"format": "csv"}))
        return docs
    return [Document(raw, path, {"format": "text"})]


def _json_text(obj) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for key in ("text", "content", "body", "value", "message"):
            if isinstance(obj.get(key), str):
                return obj[key]
        return " ".join(str(v) for v in obj.values() if isinstance(v, str))
    return str(obj)


def _read_path(path: str) -> list[Document]:
    path = os.path.expanduser(path)
    if os.path.isfile(path):
        return _read_file(path)
    if os.path.isdir(path):
        docs = []
        for root, _, files in os.walk(path):
            for name in sorted(files):
                if os.path.splitext(name)[1].lower() in TEXT_EXTS:
                    try:
                        docs.extend(_read_file(os.path.join(root, name)))
                    except Exception:
                        pass
                if len(docs) > 5000:
                    return docs
        return docs
    raise FileNotFoundError(f"no such path: {path}")


# --------------------------------------------------------------------------- #
# scraping / crawl
# --------------------------------------------------------------------------- #
def _scrape(urls: list[str], depth: int = 0, max_pages: int = 12,
            same_domain: bool = True, delay: float = 0.4) -> list[Document]:
    seen: set[str] = set()
    queue = [(u, 0) for u in urls]
    docs: list[Document] = []
    domains = {urlparse(u).netloc for u in urls}
    while queue and len(docs) < max_pages:
        url, d = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            body, ctype = _fetch(url)
        except Exception as e:
            docs.append(Document("", url, {"error": str(e)}))
            continue
        if "html" in ctype or body.lstrip()[:1] == "<":
            text, links = extract_main(body)
        else:
            text, links = _normalize(body), []
        if text:
            docs.append(Document(text, url, {"chars": len(text)}))
        if d < depth:
            for href in links:
                nxt = urljoin(url, href).split("#")[0]
                if urlparse(nxt).scheme not in ("http", "https"):
                    continue
                if same_domain and urlparse(nxt).netloc not in domains:
                    continue
                if nxt not in seen:
                    queue.append((nxt, d + 1))
        time.sleep(delay)  # be polite
    return docs


# --------------------------------------------------------------------------- #
# cloud storage connectors (S3 / GCS via optional SDKs; Drive/Dropbox links)
# --------------------------------------------------------------------------- #
def _normalize_share_url(uri: str) -> str:
    """Turn a Google Drive / Dropbox *share* link into a direct-download URL so
    it can be fetched over plain HTTPS (no OAuth for public files)."""
    if "drive.google.com" in uri:
        m = re.search(r"/file/d/([\w-]+)", uri) or re.search(r"[?&]id=([\w-]+)", uri)
        if m:
            return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    if "dropbox.com" in uri:
        return uri.replace("www.dropbox.com", "dl.dropboxusercontent.com").replace("?dl=0", "").replace("?dl=1", "")
    return uri


def _read_bytes_as_docs(raw: bytes, uri: str) -> list[Document]:
    text = raw.decode("utf-8", errors="replace")
    if uri.endswith(".jsonl"):
        docs = []
        for line in text.splitlines():
            if line.strip():
                try:
                    docs.append(Document(_json_text(json.loads(line)), uri, {"format": "jsonl"}))
                except json.JSONDecodeError:
                    docs.append(Document(line, uri, {"format": "text"}))
        return [d for d in docs if d.text]
    return [Document(_normalize(text), uri, {"chars": len(text)})]


def _fetch_s3(uri: str, creds: dict) -> list[Document]:
    try:
        import boto3
    except ImportError:
        raise RuntimeError("S3 needs boto3 — install it (pip install boto3) or use a public https:// URL")
    bucket, _, key = uri[len("s3://"):].partition("/")
    session = boto3.session.Session(
        aws_access_key_id=creds.get("access_key") or None,
        aws_secret_access_key=creds.get("secret_key") or None,
        aws_session_token=creds.get("token") or None,
        region_name=creds.get("region") or None)
    s3 = session.client("s3")
    docs = []
    if not key or key.endswith("/"):
        for obj in s3.list_objects_v2(Bucket=bucket, Prefix=key).get("Contents", [])[:200]:
            body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read(MAX_BYTES)
            docs.extend(_read_bytes_as_docs(body, f"s3://{bucket}/{obj['Key']}"))
    else:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read(MAX_BYTES)
        docs.extend(_read_bytes_as_docs(body, uri))
    return docs


def _fetch_gcs(uri: str, creds: dict) -> list[Document]:
    try:
        from google.cloud import storage
    except ImportError:
        raise RuntimeError("GCS needs google-cloud-storage — install it or use a public https:// URL")
    bucket_name, _, prefix = uri[len("gs://"):].partition("/")
    client = (storage.Client.from_service_account_json(creds["service_account_json"])
              if creds.get("service_account_json") else storage.Client())
    bucket = client.bucket(bucket_name)
    docs = []
    if not prefix or prefix.endswith("/"):
        for blob in list(client.list_blobs(bucket_name, prefix=prefix))[:200]:
            docs.extend(_read_bytes_as_docs(blob.download_as_bytes()[:MAX_BYTES], f"gs://{bucket_name}/{blob.name}"))
    else:
        docs.extend(_read_bytes_as_docs(bucket.blob(prefix).download_as_bytes()[:MAX_BYTES], uri))
    return docs


def fetch_cloud(uri: str, creds: dict | None = None) -> list[Document]:
    """Read text from cloud storage. s3:// and gs:// use their SDKs (with the
    creds you pass, or ambient/instance credentials); Drive/Dropbox share links
    and any https:// object are fetched directly."""
    uri = (uri or "").strip()
    creds = creds or {}
    if uri.startswith("s3://"):
        return _fetch_s3(uri, creds)
    if uri.startswith("gs://"):
        return _fetch_gcs(uri, creds)
    url = _normalize_share_url(uri)
    body, ctype = _fetch(url)
    text = extract_main(body)[0] if ("html" in ctype or body.lstrip()[:1] == "<") else _normalize(body)
    return [Document(text, uri, {"chars": len(text)})] if text else []


# --------------------------------------------------------------------------- #
# public entry
# --------------------------------------------------------------------------- #
def gather(spec: dict) -> list[Document]:
    """spec = {source, ...} -> list[Document]. Never raises for empty results;
    surfaces per-source errors in Document.meta so the UI can report them."""
    src = (spec or {}).get("source", "paste")
    if src == "paste":
        text = (spec.get("text") or "").strip()
        return [Document(text, "paste", {"chars": len(text)})] if text else []
    if src == "path":
        return _read_path(spec["path"])
    if src == "url":
        body, ctype = _fetch(spec["url"])
        text = extract_main(body)[0] if ("html" in ctype or body.lstrip()[:1] == "<") else _normalize(body)
        return [Document(text, spec["url"], {"chars": len(text)})]
    if src == "scrape":
        urls = spec.get("urls") or ([spec["url"]] if spec.get("url") else [])
        return _scrape(urls, depth=int(spec.get("depth", 0)),
                       max_pages=int(spec.get("max_pages", 12)),
                       same_domain=bool(spec.get("same_domain", True)))
    if src == "cloud":
        return fetch_cloud(spec.get("uri", ""), spec.get("credentials"))
    if src == "upload":
        return _read_uploads(spec.get("files", []))
    if src == "dataset":
        return _read_dataset(spec.get("name"))
    raise ValueError(f"unknown source: {src}")


# --------------------------------------------------------------------------- #
# Dataset library — save a prepared corpus once, reuse it across models
# --------------------------------------------------------------------------- #
DATASETS_DIR = os.environ.get("FIDDLEHEAD_DATASETS_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")


def _dataset_slug(name: str) -> str:
    return re.sub(r"[^\w.-]", "_", str(name or "")).strip("_") or "dataset"


def _read_dataset(name: str) -> list[Document]:
    slug = _dataset_slug(name)
    p = os.path.join(DATASETS_DIR, slug, "docs.jsonl")
    if not os.path.isfile(p):
        return [Document("", slug, {"error": f"no saved dataset '{slug}'"})]
    docs = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                o = json.loads(line)
                docs.append(Document(o.get("text", ""), o.get("source", f"dataset:{slug}"), {}))
            except Exception:
                pass
    return docs


def save_dataset(name: str, docs: list[Document]) -> dict:
    """Persist the gathered documents so they can be reused without re-fetching."""
    slug = _dataset_slug(name)
    d = os.path.join(DATASETS_DIR, slug)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "docs.jsonl"), "w", encoding="utf-8") as f:
        for doc in docs:
            if doc.text:
                f.write(json.dumps({"text": doc.text, "source": doc.source}) + "\n")
    info = corpus_stats([dd for dd in docs if dd.text])
    meta = {"name": slug, "created": int(time.time()),
            "documents": info.get("documents"), "chars": info.get("chars"),
            "preview": info.get("preview")}
    with open(os.path.join(d, "dataset.json"), "w") as f:
        json.dump(meta, f)
    return meta


def list_datasets() -> list[dict]:
    out = []
    if os.path.isdir(DATASETS_DIR):
        for name in sorted(os.listdir(DATASETS_DIR)):
            mp = os.path.join(DATASETS_DIR, name, "dataset.json")
            if os.path.isfile(mp):
                try:
                    out.append(json.load(open(mp)))
                except Exception:
                    pass
    return out


def delete_dataset(name: str) -> list[dict]:
    d = os.path.join(DATASETS_DIR, _dataset_slug(name))
    if os.path.isdir(d):
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    return list_datasets()


def _read_uploads(files: list[dict]) -> list[Document]:
    """Files dropped in the browser, sent inline: each is {name, text} for text
    or {name, b64} for binary (PDF). PDFs are extracted from bytes."""
    import base64
    docs = []
    for f in files:
        name = f.get("name", "upload")
        try:
            if f.get("text") is not None:
                docs.extend(_read_bytes_as_docs(f["text"].encode("utf-8", "ignore"), name))
            elif f.get("b64"):
                raw = base64.b64decode(f["b64"])
                if name.lower().endswith(".pdf"):
                    docs.extend(_pdf_docs(_pdf_pages(stream=raw), name))
                else:
                    docs.extend(_read_bytes_as_docs(raw, name))
        except Exception as e:
            docs.append(Document("", name, {"error": str(e)}))
    return docs


def corpus_stats(docs: list[Document]) -> dict:
    ok = [d for d in docs if d.text]
    chars = sum(len(d.text) for d in ok)
    errors = [d.meta.get("error") for d in docs if d.meta.get("error")]
    return {
        "documents": len(ok),
        "chars": chars,
        "sources": len({d.source for d in ok}),
        "errors": errors[:5],
        "preview": (ok[0].text[:280] if ok else ""),
    }
