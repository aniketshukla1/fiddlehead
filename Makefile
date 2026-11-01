# Fiddlehead — common tasks. `make help` lists them.
PYTHON ?= python3

.PHONY: help setup dev check desktop

help:  ## list the tasks
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[32m%-9s\033[0m %s\n", $$1, $$2}'

setup:  ## install the Python and Node dependencies
	$(PYTHON) -m pip install -r backend/requirements-dev.txt
	cd frontend && npm install

dev:  ## run the engine, API and web app together (Ctrl-C stops all three)
	PYTHON=$(PYTHON) ./scripts/dev.sh

check:  ## build the web app, byte-compile the backend, run the tests
	cd frontend && npm run build
	$(PYTHON) -m compileall -q backend
	cd backend && $(PYTHON) artifacts.py && $(PYTHON) -m pytest -q tests

desktop:  ## run the native desktop app (needs a Rust toolchain)
	cd frontend && npm run tauri dev
