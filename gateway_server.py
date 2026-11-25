"""
gateway_server.py — Cross-Platform Zero-Dependency Universal Dual-Mode HTTP/SSE Server.

Exposes a single local gateway compatible with OpenAI Chat Completions,
OpenAI Responses API, Anthropic Messages API, and CORS-enabled browser/IDE clients.

Usage:
  python gateway_server.py
  python gateway_server.py --port 8787 --muscle-provider ollama --muscle-model llama3.2
  python gateway_server.py --brain-provider ollama --brain-model llama3.2
  python gateway_server.py --print-env
"""

from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

from gateway import (
    GatewayDecision,
    GatewayEngine,
    PROTOCOL_ANTHROPIC,
    PROTOCOL_OPENAI_CHAT,
    PROTOCOL_OPENAI_RESPONSES,
    format_json_response,
    format_sse_stream,
    normalize_request,
    resolve_upstream_endpoint,
)
from provider_registry import build_default_registry


ROUTE_TO_PROTOCOL = {
    "/v1/chat/completions": PROTOCOL_OPENAI_CHAT,
    "/chat/completions": PROTOCOL_OPENAI_CHAT,
    "/v1/responses": PROTOCOL_OPENAI_RESPONSES,
    "/responses": PROTOCOL_OPENAI_RESPONSES,
    "/v1/messages": PROTOCOL_ANTHROPIC,
    "/messages": PROTOCOL_ANTHROPIC,
}


def _build_error_payload(protocol: str, message: str, err_type: str = "gateway_error") -> Dict[str, Any]:
    if protocol == PROTOCOL_ANTHROPIC:
        return {"type": "error", "error": {"type": err_type, "message": message}}
    return {"error": {"message": message, "type": err_type}}


def make_gateway_handler(engine: GatewayEngine):
    """Creates a BaseHTTPRequestHandler subclass bound to the given GatewayEngine."""

    class UniversalGatewayHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # pragma: no cover
            if os.environ.get("GATEWAY_VERBOSE"):
                super().log_message(format, *args)

        def _send_cors_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _send_json(self, status: int, payload: Dict[str, Any], extra_headers: Optional[Dict[str, str]] = None) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            if extra_headers:
                for k, v in extra_headers.items():
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _send_sse_chunks(self, chunks: list[bytes], extra_headers: Optional[Dict[str, str]] = None) -> None:
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            if extra_headers:
                for k, v in extra_headers.items():
                    self.send_header(k, v)
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(chunk)
                self.wfile.flush()

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._send_cors_headers()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?")[0].rstrip("/") or "/"

            if path in {"/", "/health"}:
                muscle_info = (
                    f"{engine.muscle_runtime.provider.name}/{engine.muscle_runtime.model_id}"
                    if engine.muscle_runtime else "disabled"
                )
                brain_info = (
                    f"{engine.brain_runtime.provider.name}/{engine.brain_runtime.model_id}"
                    if engine.brain_runtime else f"passthrough ({engine.upstream_url or 'caller-key'})"
                )
                self._send_json(200, {
                    "status": "ok",
                    "service": "dynamic-dual-model-universal-gateway",
                    "version": "9.1.0",
                    "muscle_tier": muscle_info,
                    "brain_tier": brain_info,
                    "endpoints": ["/v1/chat/completions", "/v1/responses", "/v1/messages", "/v1/models", "/v1/stats", "/health"],
                })
                return

            if path == "/v1/stats":
                self._send_json(200, engine.stats.snapshot())
                return

            if path == "/v1/models":
                models = [
                    {"id": "dual-mode-auto", "object": "model", "owned_by": "dynamic-dual-model"},
                ]
                if engine.muscle_runtime:
                    models.append({
                        "id": engine.muscle_runtime.model_id,
                        "object": "model",
                        "owned_by": engine.muscle_runtime.provider.name,
                    })
                if engine.brain_runtime:
                    models.append({
                        "id": engine.brain_runtime.model_id,
                        "object": "model",
                        "owned_by": engine.brain_runtime.provider.name,
                    })
                self._send_json(200, {"object": "list", "data": models})
                return

            self._send_json(404, _build_error_payload(PROTOCOL_OPENAI_CHAT, f"Unknown endpoint: {path}", "not_found"))

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?")[0].rstrip("/")
            protocol = ROUTE_TO_PROTOCOL.get(path)
            if protocol is None:
                self._send_json(404, _build_error_payload(PROTOCOL_OPENAI_CHAT, f"Unknown endpoint: {path}", "not_found"))
                return

            try:
                content_len = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
                payload = json.loads(raw_body.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("JSON request body must be an object.")
            except Exception as e:
                self._send_json(400, _build_error_payload(protocol, f"Invalid JSON request: {e}", "invalid_request_error"))
                return

            norm_req = normalize_request(protocol, payload)

            # 1. Try Cache / Muscle Memory locally
            local_decision = engine.try_local_execution(norm_req)
            if local_decision is not None:
                engine.stats.record(local_decision, protocol, norm_req.latest_user_text)
                headers_out = {
                    "X-DualMode-Pathway": local_decision.pathway,
                    "X-DualMode-Model": local_decision.model_used,
                    "X-DualMode-Tokens-Saved": str(local_decision.estimated_tokens_saved),
                }
                if norm_req.stream:
                    chunks = format_sse_stream(
                        protocol,
                        norm_req.model,
                        local_decision.text,
                        local_decision.input_tokens,
                        local_decision.output_tokens,
                    )
                    self._send_sse_chunks(chunks, extra_headers=headers_out)
                else:
                    body = format_json_response(
                        protocol,
                        norm_req.model,
                        local_decision.text,
                        local_decision.input_tokens,
                        local_decision.output_tokens,
                        pathway=local_decision.pathway,
                    )
                    self._send_json(200, body, extra_headers=headers_out)
                return

            # 2. Forward to Full Brain (either configured provider adapter or native HTTP passthrough)
            if engine.brain_mode == "provider" and engine.brain_runtime is not None:
                try:
                    brain_dec = engine.execute_brain_fallback(norm_req)
                    engine.stats.record(brain_dec, protocol, norm_req.latest_user_text)
                    headers_out = {
                        "X-DualMode-Pathway": brain_dec.pathway,
                        "X-DualMode-Model": brain_dec.model_used,
                        "X-DualMode-Tokens-Saved": "0",
                    }
                    if norm_req.stream:
                        chunks = format_sse_stream(
                            protocol,
                            norm_req.model,
                            brain_dec.text,
                            brain_dec.input_tokens,
                            brain_dec.output_tokens,
                        )
                        self._send_sse_chunks(chunks, extra_headers=headers_out)
                    else:
                        body = format_json_response(
                            protocol,
                            norm_req.model,
                            brain_dec.text,
                            brain_dec.input_tokens,
                            brain_dec.output_tokens,
                            pathway=brain_dec.pathway,
                        )
                        self._send_json(200, body, extra_headers=headers_out)
                except Exception as e:
                    self._send_json(502, _build_error_payload(protocol, f"Brain provider error: {e}", "upstream_error"))
                return

            # Native HTTP passthrough (preserves exact caller tools, thinking, streaming, and auth)
            req_headers = {k: v for k, v in self.headers.items()}
            target_url, fwd_headers = resolve_upstream_endpoint(protocol, req_headers, engine.upstream_url)
            if not target_url:
                self._send_json(
                    401,
                    _build_error_payload(
                        protocol,
                        "No upstream API credentials found for Brain passthrough. "
                        "Provide an Authorization / x-api-key header, set OPENAI_API_KEY / ANTHROPIC_API_KEY, "
                        "or start the gateway with --brain-provider ollama.",
                        "authentication_error",
                    ),
                )
                return

            t0 = time.perf_counter()
            upstream_req = urllib.request.Request(
                target_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=fwd_headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(upstream_req, timeout=180.0) as up_resp:
                    up_status = up_resp.status
                    up_ct = up_resp.headers.get("Content-Type", "application/json")
                    self.send_response(up_status)
                    self._send_cors_headers()
                    self.send_header("Content-Type", up_ct)
                    self.send_header("X-DualMode-Pathway", "brain_upstream")
                    self.send_header("Connection", "close")
                    self.end_headers()

                    in_tok = 0
                    out_tok = 0
                    if norm_req.stream or "text/event-stream" in up_ct.lower():
                        while True:
                            chunk = up_resp.read(4096)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            self.wfile.flush()
                    else:
                        raw_out = up_resp.read()
                        self.wfile.write(raw_out)
                        try:
                            parsed = json.loads(raw_out.decode("utf-8"))
                            usage = parsed.get("usage") or {}
                            in_tok = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
                            out_tok = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
                        except Exception:
                            pass

                    dt = (time.perf_counter() - t0) * 1000
                    engine.stats.record(
                        GatewayDecision(
                            handled_locally=False,
                            pathway="brain_upstream",
                            reason="Forwarded natively to Brain upstream.",
                            model_used=norm_req.model,
                            input_tokens=in_tok,
                            output_tokens=out_tok,
                            latency_ms=dt,
                        ),
                        protocol,
                        norm_req.latest_user_text,
                    )
            except urllib.error.HTTPError as he:
                err_body = he.read()
                self.send_response(he.code)
                self._send_cors_headers()
                self.send_header("Content-Type", he.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(err_body)))
                self.end_headers()
                self.wfile.write(err_body)
            except Exception as e:
                self._send_json(502, _build_error_payload(protocol, f"Upstream connection failed: {e}", "upstream_error"))

    return UniversalGatewayHandler


def print_client_env_instructions(host: str, port: int) -> None:
    base = f"http://{host}:{port}"
    print(f"\n=== Universal Dual-Mode Gateway Ready: {base} ===")
    print("\n# Windows PowerShell (any OpenAI-compatible tool / Codex / SDK):")
    print(f'  $env:OPENAI_BASE_URL="{base}/v1"')
    print(f'  $env:ANTHROPIC_BASE_URL="{base}"')
    print("\n# macOS / Linux (bash / zsh):")
    print(f'  export OPENAI_BASE_URL="{base}/v1"')
    print(f'  export ANTHROPIC_BASE_URL="{base}"')
    print("\n# Compatible Clients:")
    print(f"  - OpenAI Chat Completions : {base}/v1/chat/completions")
    print(f"  - OpenAI Responses API    : {base}/v1/responses")
    print(f"  - Anthropic Messages API  : {base}/v1/messages")
    print(f"  - Live Savings Dashboard  : {base}/v1/stats\n")


def start_server(
    host: str = "127.0.0.1",
    port: int = 8787,
    engine: Optional[GatewayEngine] = None,
) -> Tuple[ThreadingHTTPServer, GatewayEngine]:
    """Starts the ThreadingHTTPServer and returns (server, engine)."""
    gw_engine = engine or GatewayEngine.from_config()
    handler_cls = make_gateway_handler(gw_engine)
    server = ThreadingHTTPServer((host, port), handler_cls)
    return server, gw_engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Universal Dual-Mode HTTP/SSE Gateway Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8787, help="Bind port (default: 8787)")
    parser.add_argument("--muscle-provider", default=None, help="Muscle tier provider (ollama, mock, openai, anthropic, gemini, openrouter, none)")
    parser.add_argument("--muscle-model", default=None, help="Model ID for Muscle tier")
    parser.add_argument("--brain-provider", default="passthrough", help="Brain tier mode: passthrough (default) or provider name (ollama, openai, anthropic, openrouter, gemini, mock)")
    parser.add_argument("--brain-model", default=None, help="Model ID for Brain tier when --brain-provider != passthrough")
    parser.add_argument("--upstream-url", default=None, help="Custom upstream base URL for passthrough")
    parser.add_argument("--print-env", action="store_true", help="Print cross-platform client environment setup and exit")
    args = parser.parse_args()

    if args.print_env:
        print_client_env_instructions(args.host, args.port)
        sys.exit(0)

    gw = GatewayEngine.from_config(
        muscle_provider=args.muscle_provider,
        muscle_model=args.muscle_model,
        brain_provider=args.brain_provider,
        brain_model=args.brain_model,
        upstream_url=args.upstream_url,
    )
    server, _ = start_server(host=args.host, port=args.port, engine=gw)
    print_client_env_instructions(args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Universal Dual-Mode Gateway.")
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
