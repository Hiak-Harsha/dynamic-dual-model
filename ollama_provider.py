"""
ollama_provider.py — Local model provider adapter for Ollama.

Enables 100% $0 local model execution without cloud API keys or subscriptions.
Communicates with the local Ollama daemon via HTTP REST API (/api/chat).

Conforms to ModelProvider interface:
- classify: route user message using local model
- route_classify: multi-axis structured routing dict
- muscle_memory_reply: fast, short reply (max 120 tokens, no tools)
- full_brain_step: multi-turn agentic loop with tool calls and effort-to-token mapping
- summarize_history / summarize_history_step: context compression
- append_assistant_turn, append_tool_results, append_user_turn: native message history
- is_server_reachable: fast non-blocking probe to verify daemon is running
"""

from __future__ import annotations
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from base import (
    ModelProvider,
    ProviderCapabilities,
    ProviderResponse,
    ToolCall,
    ToolDef,
    count_tokens,
)


def is_ollama_reachable(host: str = "http://localhost:11434", timeout: float = 0.25) -> bool:
    """Probes the Ollama daemon with a fast timeout to check if it is running."""
    try:
        url = f"{host.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, headers={"User-Agent": "DualModeAgent/9.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


class OllamaProvider(ModelProvider):
    """Local model provider backed by an Ollama HTTP instance."""

    name: str = "ollama"

    def __init__(
        self,
        model: str = "llama3.2",
        host: Optional[str] = None,
        timeout: float = 60.0,
        request_fn: Optional[Any] = None,
    ):
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout
        # Optional custom requester for deterministic unit test mocking
        self._request_fn = request_fn

    @property
    def capabilities(self) -> ProviderCapabilities:
        # Tool support depends on model families (llama3.1+, qwen2.5+, mistral+ support tools)
        supports_tools = not any(
            small in self.model.lower() for small in ["tiny", "phi-1", "gemma-2b"]
        )
        return ProviderCapabilities(
            supports_reasoning_effort=False,
            supports_prompt_cache=False,  # Ollama standard API does not report cache hits
            supports_structured_output=True,
            supports_native_tool_history=True,
            supports_reasoning_usage=False,
            supports_tool_calls=supports_tools,
        )

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON POST request to the Ollama server."""
        if self._request_fn is not None:
            return self._request_fn(endpoint, payload)

        url = f"{self.host}{endpoint}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "DualModeAgent/9.1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read().decode("utf-8")
                return json.loads(data)
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Failed to connect to Ollama at {self.host}: {e}. "
                "Ensure Ollama is running (`ollama serve`)."
            ) from e

    # ------------------------------------------------------------------
    # Routing / Classification
    # ------------------------------------------------------------------
    def classify(self, user_message: str, recent_summary: str) -> str:
        prompt = (
            "Classify whether the following user message should be handled by 'muscle_memory' "
            "(simple, routine greeting, thanks, trivial factual question, or deterministic request) "
            "or 'full_brain' (multi-step task, coding, tool usage, complex reasoning, or debugging).\n\n"
            f"User message: {user_message}\n"
            f"Context: {recent_summary}\n\n"
            "Respond ONLY with 'muscle_memory' or 'full_brain'."
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_predict": 32,
                "temperature": 0.0,
            },
        }
        try:
            res = self._post("/api/chat", payload)
            content = res.get("message", {}).get("content", "").strip().lower()
            if "muscle_memory" in content:
                return "muscle_memory"
            return "full_brain"
        except Exception:
            # Fallback heuristic if local server errors
            low = user_message.strip().lower()
            if low in {"hi", "hello", "hey", "thanks", "thank you", "help"}:
                return "muscle_memory"
            return "full_brain"

    def route_classify(self, user_message: str, recent_summary: str) -> Dict[str, Any]:
        prompt = (
            "Analyze this user request for cognitive architecture routing.\n"
            f"User message: {user_message}\n"
            f"Context: {recent_summary}\n\n"
            "Output JSON with keys: pathway ('muscle' or 'full_brain'), confidence (0.0-1.0), "
            "tools_required (bool), risk ('low', 'medium', or 'high'), novelty (0.0-1.0), reasoning (str)."
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "format": "json",
            "stream": False,
            "options": {
                "num_predict": 128,
                "temperature": 0.0,
            },
        }
        try:
            res = self._post("/api/chat", payload)
            content = res.get("message", {}).get("content", "")
            data = json.loads(content)
            # Normalize keys
            pathway = data.get("pathway", "muscle" if self.classify(user_message, recent_summary) == "muscle_memory" else "full_brain")
            if pathway not in ("muscle", "full_brain"):
                pathway = "muscle" if "muscle" in str(pathway) else "full_brain"

            in_tokens = res.get("prompt_eval_count", count_tokens(prompt))
            out_tokens = res.get("eval_count", count_tokens(content))
            resp = ProviderResponse(
                text=content,
                tool_calls=[],
                stop_reason="end_turn",
                raw=res,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
            )
            return {
                "pathway": pathway,
                "confidence": float(data.get("confidence", 0.85)),
                "context_required": False,
                "tools_required": bool(data.get("tools_required", pathway == "full_brain")),
                "freshness_required": False,
                "novelty": float(data.get("novelty", 0.2 if pathway == "muscle" else 0.7)),
                "risk": str(data.get("risk", "low" if pathway == "muscle" else "medium")),
                "reasoning": str(data.get("reasoning", "Ollama model-based route classification.")),
                "_response": resp,
            }
        except Exception:
            c = self.classify(user_message, recent_summary)
            is_muscle = c == "muscle_memory"
            return {
                "pathway": "muscle" if is_muscle else "full_brain",
                "confidence": 0.85 if is_muscle else 0.90,
                "context_required": False,
                "tools_required": not is_muscle,
                "freshness_required": False,
                "novelty": 0.2 if is_muscle else 0.7,
                "risk": "low" if is_muscle else "medium",
                "reasoning": f"Fallback Ollama classification as {c}.",
                "_response": None,
            }

    # ------------------------------------------------------------------
    # Muscle Memory
    # ------------------------------------------------------------------
    def muscle_memory_reply(self, user_message: str) -> ProviderResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_message}],
            "stream": False,
            "options": {
                "num_predict": 120,
                "temperature": 0.0,
            },
        }
        res = self._post("/api/chat", payload)
        text = res.get("message", {}).get("content", "")
        in_tok = res.get("prompt_eval_count", count_tokens(user_message))
        out_tok = res.get("eval_count", count_tokens(text))

        return ProviderResponse(
            text=text,
            tool_calls=[],
            stop_reason="end_turn",
            raw=res,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )

    # ------------------------------------------------------------------
    # Full Brain
    # ------------------------------------------------------------------
    def full_brain_step(
        self,
        history: List[Any],
        tools: List[ToolDef],
        effort: str,
    ) -> ProviderResponse:
        # Map effort to max output tokens
        effort_token_map = {
            "low": 256,
            "medium": 1024,
            "high": 4096,
            "xhigh": 8192,
            "max": 8192,
        }
        max_tokens = effort_token_map.get(effort, 1024)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": history,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.1,
            },
        }

        # Translate tools to Ollama function definitions if supported
        if tools and self.capabilities.supports_tool_calls:
            ollama_tools = []
            for t in tools:
                ollama_tools.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                })
            payload["tools"] = ollama_tools

        res = self._post("/api/chat", payload)
        msg = res.get("message", {})
        text = msg.get("content", "") or ""

        # Parse tool calls
        raw_tools = msg.get("tool_calls", [])
        parsed_tools: List[ToolCall] = []
        for i, rt in enumerate(raw_tools):
            fn = rt.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"raw": args}
            parsed_tools.append(ToolCall(
                id=f"ollama_tc_{i}",
                name=name,
                arguments=args if isinstance(args, dict) else {},
            ))

        stop_reason = "tool_use" if parsed_tools else "end_turn"
        in_tok = res.get("prompt_eval_count", count_tokens(json.dumps(history)))
        out_tok = res.get("eval_count", count_tokens(text))

        return ProviderResponse(
            text=text,
            tool_calls=parsed_tools,
            stop_reason=stop_reason,
            raw=res,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )

    # ------------------------------------------------------------------
    # Summary Compression
    # ------------------------------------------------------------------
    def summarize_history(self, history_text: str) -> str:
        prompt = (
            "Summarize the following conversation history into 3 to 5 concise factual sentences. "
            "Retain all facts, user preferences, and completed task states.\n\n"
            f"{history_text}"
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_predict": 300,
                "temperature": 0.0,
            },
        }
        res = self._post("/api/chat", payload)
        return res.get("message", {}).get("content", "").strip()

    # ------------------------------------------------------------------
    # History Plumbing
    # ------------------------------------------------------------------
    def append_assistant_turn(self, history: List[Any], resp: ProviderResponse) -> None:
        turn: Dict[str, Any] = {"role": "assistant", "content": resp.text}
        if resp.tool_calls:
            turn["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }
                for tc in resp.tool_calls
            ]
        history.append(turn)

    def append_tool_results(
        self, history: List[Any], tool_calls: List[ToolCall], results: Dict[str, str]
    ) -> None:
        for tc in tool_calls:
            content = results.get(tc.id, "")
            history.append({
                "role": "tool",
                "content": content,
            })

    def append_user_turn(self, history: List[Any], text: str) -> None:
        history.append({"role": "user", "content": text})
