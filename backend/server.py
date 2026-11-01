"""
server.py — WebSocket broadcast hub for training telemetry.

Any training loop calls `TelemetryServer.publish(frame)` per step; every
connected frontend (web or Tauri) receives it in real time. Runs the
websocket server on a background thread so it doesn't block a synchronous
PyTorch training loop.
"""

from __future__ import annotations
import asyncio
import http
import json
import logging
import queue
import threading
import websockets

# A browser tab, curl or a health check hitting the port is not an error worth a traceback.
_ws_log = logging.getLogger("fiddlehead.ws")
_ws_log.addFilter(lambda rec: "opening handshake failed" not in rec.getMessage())


def _plain_http(connection, request):
    """A plain GET (no WebSocket upgrade) gets a one-line answer, so the port doubles as a health check."""
    if "upgrade" not in request.headers.get("Connection", "").lower():
        return connection.respond(http.HTTPStatus.OK, "Fiddlehead engine is running. The app connects here over WebSocket.\n")


class TelemetryServer:
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self._clients: set = set()
        self._queue: "queue.Queue[dict]" = queue.Queue()
        self._inbound: "queue.Queue[dict]" = queue.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main())

    async def _main(self):
        # allow large inbound messages (browser file uploads arrive inline as
        # base64) — the default limit is 1 MB, which a PDF would blow past.
        async with websockets.serve(self._handle_client, self.host, self.port, max_size=64 * 1024 * 1024,
                                    process_request=_plain_http, logger=_ws_log):
            print(f"[telemetry] ws://{self.host}:{self.port}")
            await self._drain_queue()

    async def _handle_client(self, websocket):
        self._clients.add(websocket)
        try:
            async for raw in websocket:
                # frontend sends control messages (e.g. head ablation); queue
                # them for the training loop to drain via poll_control().
                try:
                    self._inbound.put(json.loads(raw))
                except (ValueError, TypeError):
                    pass
        finally:
            self._clients.discard(websocket)

    async def _drain_queue(self):
        while True:
            try:
                frame = self._queue.get_nowait()
                if self._clients:
                    payload = json.dumps(frame)
                    await asyncio.gather(
                        *(c.send(payload) for c in list(self._clients)), return_exceptions=True
                    )
            except queue.Empty:
                pass
            await asyncio.sleep(0.02)

    def publish(self, frame: dict):
        """Call from your (synchronous) training loop — thread-safe."""
        self._queue.put(frame)

    def poll_control(self) -> list[dict]:
        """Drain any control messages sent by connected frontends. Call once
        per training step; returns [] when there's nothing pending."""
        msgs = []
        while True:
            try:
                msgs.append(self._inbound.get_nowait())
            except queue.Empty:
                break
        return msgs
