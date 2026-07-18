from __future__ import annotations

import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict
from urllib.parse import parse_qs, urlparse
from uuid import uuid4


PROTOCOL_VERSION = "2024-11-05"
TOOL_NAME = "score_schedule_risk"
TOOL_SCHEMA = {
    "type": "object",
    "required": ["request_id", "projects"],
    "properties": {
        "request_id": {"type": "string", "minLength": 1},
        "projects": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["customer_name", "project_id"],
                "properties": {
                    "customer_name": {"type": "string"},
                    "project_id": {"type": ["string", "number"]},
                },
            },
        },
        "options": {"type": "object"},
        "database_auth": {"type": "object"},
    },
}


def handle_rpc(message: Dict, score: Callable[[Dict], Dict]):
    method = message.get("method")
    request_id = message.get("id")
    if method == "initialize":
        result = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "schedule-risk-agent", "version": "0.1.0"},
        }
    elif method == "tools/list":
        result = {"tools": [{
            "name": TOOL_NAME,
            "description": "Predict three-bin project schedule-delay risk.",
            "inputSchema": TOOL_SCHEMA,
        }]}
    elif method == "tools/call":
        params = message.get("params") or {}
        if params.get("name") != TOOL_NAME:
            return {
                "jsonrpc": "2.0", "id": request_id,
                "error": {"code": -32602, "message": "Unknown tool"},
            }
        try:
            payload = params.get("arguments") or {}
            result_payload = score(payload)
            result = {
                "content": [{"type": "text", "text": json.dumps(result_payload)}],
                "structuredContent": result_payload,
                "isError": False,
            }
        except Exception as exc:
            result = {
                "content": [{"type": "text", "text": json.dumps({
                    "status": "failed",
                    "error": {"code": getattr(exc, "code", "INTERNAL_ERROR"), "message": str(exc)},
                })}],
                "isError": True,
            }
    elif method and method.startswith("notifications/"):
        return None
    else:
        return {
            "jsonrpc": "2.0", "id": request_id,
            "error": {"code": -32601, "message": "Method not found"},
        }
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def make_handler(score: Callable[[Dict], Dict], ready=lambda: True):
    sessions = {}
    sessions_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "ScheduleRiskMCP/0.1"

        def log_message(self, format_string, *args):
            print(json.dumps({
                "event": "http_access",
                "client": self.client_address[0],
                "message": format_string % args,
            }), flush=True)

        def _json(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/health/live":
                self._json(200, {"status": "ok"})
                return
            if parsed.path == "/health/ready":
                is_ready = ready()
                self._json(200 if is_ready else 503, {"status": "ok" if is_ready else "not_ready"})
                return
            if parsed.path != "/sse":
                self._json(404, {"error": "not_found"})
                return
            session_id = uuid4().hex
            messages = queue.Queue()
            with sessions_lock:
                sessions[session_id] = messages
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                endpoint = "/messages?session_id=" + session_id
                self.wfile.write(("event: endpoint\ndata: " + endpoint + "\n\n").encode("utf-8"))
                self.wfile.flush()
                while True:
                    try:
                        payload = messages.get(timeout=15)
                        event = "event: message\ndata: " + json.dumps(payload) + "\n\n"
                    except queue.Empty:
                        event = ": keepalive\n\n"
                    self.wfile.write(event.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with sessions_lock:
                    sessions.pop(session_id, None)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path not in {"/messages", "/mcp"}:
                self._json(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                message = json.loads(self.rfile.read(length))
                response = handle_rpc(message, score)
            except Exception as exc:
                self._json(400, {
                    "jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": str(exc)},
                })
                return
            session_id = parse_qs(parsed.query).get("session_id", [None])[0]
            if session_id:
                with sessions_lock:
                    messages = sessions.get(session_id)
                if messages is None:
                    self._json(404, {"error": "unknown_session"})
                    return
                if response is not None:
                    messages.put(response)
                self.send_response(202)
                self.end_headers()
            elif response is None:
                self.send_response(202)
                self.end_headers()
            else:
                self._json(200, response)

    return Handler


def serve(score: Callable[[Dict], Dict], ready=lambda: True):
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8011"))
    server = ThreadingHTTPServer((host, port), make_handler(score, ready))
    print(json.dumps({"event": "server_started", "host": host, "port": port}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

