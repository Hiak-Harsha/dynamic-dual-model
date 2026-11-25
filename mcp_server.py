"""
mcp_server.py — Zero-Dependency Stdio MCP Server for Universal Dual-Mode Tools.

Allows any MCP-compatible host (Antigravity / agy CLI, Claude Desktop, Cursor,
VS Code, Windsurf, etc.) on Windows, macOS, or Linux to invoke Muscle Memory
deterministic execution, routing inspection, skill lookup, and provider status.

Usage:
  python mcp_server.py
"""

from __future__ import annotations
import json
import sys
from typing import Any, Dict, List, Optional

from base import ConversationState, ProviderResponse, ResponseCache
from execution_state import ExecutionState, COMPLETED_VERIFIED
from muscle_engine import MuscleExecutor
from procedural_memory import ProceduralMemory
from provider_registry import build_default_registry
from router import route
from success_contracts import resolve_contract


MCP_PROTOCOL_VERSION = "2024-11-05"

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "dual_mode_route",
        "description": "Evaluate a prompt with the Dynamic Dual-Mode multi-axis router (pathway, risk, novelty, context/tool/freshness flags, deterministic match).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "User prompt to classify."}
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "muscle_execute",
        "description": "Execute safe deterministic Muscle Memory (AST arithmetic, JSON validation, text statistics, exact cache) at 0 LLM tokens.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Prompt to execute deterministically."}
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "procedural_skill_lookup",
        "description": "Search trusted procedural skills stored in ProceduralMemory for matching workflows.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Task description or query to match against skills."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "provider_status",
        "description": "List all registered model providers (mock, ollama, openai, anthropic, openrouter, gemini) and their live availability.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


class DualModeMCPServer:
    """In-process JSON-RPC 2.0 MCP handler for stdio transport and unit testing."""

    def __init__(
        self,
        cache: Optional[ResponseCache] = None,
        memory: Optional[ProceduralMemory] = None,
    ):
        self.cache = cache or ResponseCache()
        self.memory = memory or ProceduralMemory()
        self.executor = MuscleExecutor(provider=None, memory=self.memory, cache=self.cache)

    def handle_message(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = msg.get("method")
        req_id = msg.get("id")
        params = msg.get("params") or {}

        # Notifications have no id and expect no response
        if req_id is None:
            return None

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "dynamic-dual-model-mcp",
                        "version": "9.1.0",
                    },
                },
            }

        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOL_DEFINITIONS}}

        if method == "tools/call":
            name = params.get("name", "")
            arguments = params.get("arguments") or {}
            try:
                text_out, is_error = self._call_tool(name, arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": text_out}],
                        "isError": is_error,
                    },
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Tool execution error: {e}"}],
                        "isError": True,
                    },
                }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    def _call_tool(self, name: str, arguments: Dict[str, Any]) -> tuple[str, bool]:
        if name == "dual_mode_route":
            prompt = str(arguments.get("prompt", ""))
            decision = route(prompt, ConversationState(), cache=self.cache, memory=self.memory, provider=None)
            return json.dumps(decision.to_dict(), indent=2), False

        if name == "muscle_execute":
            prompt = str(arguments.get("prompt", ""))
            decision = route(prompt, ConversationState(), cache=self.cache, memory=self.memory, provider=None)
            if decision.pathway not in {"cache", "muscle"} or (decision.pathway == "muscle" and not decision.procedure_match):
                return json.dumps({
                    "handled": False,
                    "pathway": decision.pathway,
                    "reason": "Prompt requires LLM / Full Brain reasoning; not eligible for 0-token deterministic execution.",
                }, indent=2), False

            exec_state = ExecutionState(objective=prompt)
            res = self.executor.execute(prompt, exec_state, decision=decision, conv_state=ConversationState())
            contract = resolve_contract(prompt)
            c_res = contract.evaluate(exec_state, res.text, exec_state)
            if res.status == COMPLETED_VERIFIED and (c_res.verified or c_res.task_satisfied):
                self.cache.put(prompt, ProviderResponse(text=res.text, tool_calls=[], stop_reason="end_turn", raw=None))
                return json.dumps({
                    "handled": True,
                    "status": res.status,
                    "skill_used": res.skill_used,
                    "tokens_used": 0,
                    "output": res.text,
                    "verified_facts": exec_state.verified_facts,
                }, indent=2), False
            return json.dumps({
                "handled": False,
                "status": res.status,
                "reason": res.reason or c_res.reason,
            }, indent=2), False

        if name == "procedural_skill_lookup":
            query = str(arguments.get("query", ""))
            skill = self.memory.retrieve(query, only_trusted=True)
            if skill is None:
                return json.dumps({"matched": False, "skill": None}, indent=2), False
            return json.dumps({
                "matched": True,
                "skill": {
                    "id": skill.id,
                    "name": skill.name,
                    "procedure_type": skill.procedure_type,
                    "trust_tier": skill.trust_tier,
                    "verified_success_count": skill.verified_success_count,
                    "required_tools": skill.required_tools,
                },
            }, indent=2), False

        if name == "provider_status":
            reg = build_default_registry()
            return json.dumps({"providers": reg.provider_info()}, indent=2), False

        return f"Unknown tool: {name}", True


def serve_stdio() -> None:  # pragma: no cover
    server = DualModeMCPServer()
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception as e:
            err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"Parse error: {e}"}}
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()
            continue

        resp = server.handle_message(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    serve_stdio()
