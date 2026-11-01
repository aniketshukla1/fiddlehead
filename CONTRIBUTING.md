# Contributing to Fiddlehead

Thanks for helping. Fiddlehead is small enough to understand in an afternoon, and every part of it — the
training engine, the API, the studio — is open to improvement.

## Set up

You need Python 3.10+ and Node 18+. A virtual environment keeps things tidy:

```bash
python3 -m venv .venv && source .venv/bin/activate
make setup          # Python + Node dependencies (includes pytest)
make dev            # engine + API + web app → http://localhost:5173
```

`make help` lists every task. Create a local account in the app, then click **Train the sample** to check that
the engine, API and studio all talk to each other.

## Before you open a pull request

```bash
make check          # builds the web app, byte-compiles the backend, runs the tests
```

CI runs the same checks on every pull request.

- **Keep pull requests focused.** One change per PR is much easier to review than a bundle.
- **Show UI changes.** Add a screenshot or a short clip of anything visual, in both the dark and light themes.
- **Match the code around you.** Same naming, same comment density, same idioms. The design tokens live in
  `frontend/src/ui.css`; reuse them instead of new colours or spacing.
- **Tests for logic.** A branch, a parser, anything touching money or security should leave a small test behind
  (see `backend/tests/`).

## Where things live

| Area | Path |
|------|------|
| Training engine (PyTorch, WebSocket protocol) | `backend/engine.py`, `backend/server.py`, `backend/model.py` |
| Fine-tuning, speech, vision | `backend/finetune.py`, `backend/asr.py`, `backend/tts.py`, `backend/vlm.py` |
| Data gathering and tokenizers | `backend/sources.py`, `backend/tokenizer.py`, `backend/dataset.py` |
| Control-plane API (accounts, models, keys, inference) | `backend/api/` |
| Web app | `frontend/src/` — pages in `components/`, design system in `ui.css` |
| Studio instrument panels | `frontend/src/components/{LossScope,CircuitMap,ArchGraph,…}.jsx`, `styles.css` |
| Desktop shell | `frontend/src-tauri/` |

## Reporting bugs and ideas

Open an issue with the template that fits. For security problems, please don't open a public issue — see
[SECURITY.md](SECURITY.md).

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE), and to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).
