"""
gemini_provider.py — Provider adapter for Google Gemini models.

Supports Gemini models (default: gemini-1.5-flash / gemini-2.0-flash)
via the Google Generative Language v1beta REST API.
Operates with zero extra dependencies using the standard library urllib.request.

Conforms to ModelProvider interface:
- classify: route message using low-effort generation
- route_classify: multi-axis structured classification
- muscle_memory_reply: fast, short reply (max 120 output tokens)
- full_brain_step: multi-turn agentic loop with tool declarations and function calling
- summarize_history / summarize_history_step: context compression
- append_assistant_turn, append_tool_results, append_user_turn: Gemini contents representation
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


class GeminiProvider(ModelProvider):
    """Google Gemini provider adapter via REST API."""

    name: str = "gemini"

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        request_fn: Optional[Any] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.timeout = timeout
        self._request_fn = request_fn

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_reasoning_effort=False,
            supports_prompt_cache=False,
            supports_structured_output=True,
            supports_native_tool_history=True,
            supports_reasoning_usage=False,
            supports_tool_calls=True,
        )

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._request_fn is not None:
            return self._request_fn(self.model, payload)

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. "
                "Set GEMINI_API_KEY or provide api_key in constructor."
            )

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
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
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(f"Gemini API error (HTTP {e.code}): {err_body}") from e
        except urllib.error.URLError as e:
            raise ConnectionError(f"Network error connecting to Gemini API: {e}") from e

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
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 32},
        }
        try:
            res = self._post(payload)
            cand = res.get("candidates", [{}])[0]
            parts = cand.get("content", {}).get("parts", [{}])
            text = parts[0].get("text", "").strip().lower()
            if "muscle_memory" in text:
                return "muscle_memory"
            return "full_brain"
        except Exception:
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
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 160,
                "responseMimeType": "application/json",
            },
        }
        try:
            res = self._post(payload)
            cand = res.get("candidates", [{}])[0]
            parts = cand.get("content", {}).get("parts", [{}])
            text = parts[0].get("text", "").strip()
            data = json.loads(text)

            pathway = data.get("pathway", "muscle" if self.classify(user_message, recent_summary) == "muscle_memory" else "full_brain")
            if pathway not in ("muscle", "full_brain"):
                pathway = "muscle" if "muscle" in str(pathway) else "full_brain"

            usage = res.get("usageMetadata", {})
            in_tok = usage.get("promptTokenCount", count_tokens(prompt))
            out_tok = usage.get("candidatesTokenCount", count_tokens(text))

            resp = ProviderResponse(
                text=text,
                tool_calls=[],
                stop_reason="end_turn",
                raw=res,
                input_tokens=in_tok,
                output_tokens=out_tok,
            )
            return {
                "pathway": pathway,
                "confidence": float(data.get("confidence", 0.85)),
                "context_required": False,
                "tools_required": bool(data.get("tools_required", pathway == "full_brain")),
                "freshness_required": False,
                "novelty": float(data.get("novelty", 0.2 if pathway == "muscle" else 0.7)),
                "risk": str(data.get("risk", "low" if pathway == "muscle" else "medium")),
                "reasoning": str(data.get("reasoning", "Gemini structured classification.")),
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
                "reasoning": f"Fallback Gemini classification as {c}.",
                "_response": None,
            }

    # ------------------------------------------------------------------
    # Muscle Memory
    # ------------------------------------------------------------------
    def muscle_memory_reply(self, user_message: str) -> ProviderResponse:
        payload = {
            "contents": [{"role": "user", "parts": [{"text": user_message}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 120},
        }
        res = self._post(payload)
        cand = res.get("candidates", [{}])[0]
        parts = cand.get("content", {}).get("parts", [{}])
        text = parts[0].get("text", "") if parts else ""

        usage = res.get("usageMetadata", {})
        in_tok = usage.get("promptTokenCount", count_tokens(user_message))
        out_tok = usage.get("candidatesTokenCount", count_tokens(text))

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
        effort_tokens = {
            "low": 512,
            "medium": 2048,
            "high": 4096,
            "xhigh": 8192,
            "max": 8192,
        }
        max_tokens = effort_tokens.get(effort, 2048)

        payload: Dict[str, Any] = {
            "contents": history,
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": max_tokens,
            },
        }

        if tools and self.capabilities.supports_tool_calls:
            gemini_tools = []
            for t in tools:
                gemini_tools.append({
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                })
            payload["tools"] = [{"functionDeclarations": gemini_tools}]

        res = self._post(payload)
        cand = res.get("candidates", [{}])[0]
        parts = cand.get("content", {}).get("parts", [])

        text = ""
        tool_calls: List[ToolCall] = []
        for i, p in enumerate(parts):
            if "text" in p:
                text += p["text"]
            if "functionCall" in p:
                fc = p["functionCall"]
                tool_calls.append(ToolCall(
                    id=f"gemini_tc_{i}",
                    name=fc.get("name", ""),
                    arguments=fc.get("args", {}),
                ))

        stop_reason = "tool_use" if tool_calls else "end_turn"
        usage = res.get("usageMetadata", {})
        in_tok = usage.get("promptTokenCount", count_tokens(json.dumps(history)))
        out_tok = usage.get("candidatesTokenCount", count_tokens(text))

        return ProviderResponse(
            text=text,
            tool_calls=tool_calls,
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
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 300},
        }
        res = self._post(payload)
        cand = res.get("candidates", [{}])[0]
        parts = cand.get("content", {}).get("parts", [{}])
        return parts[0].get("text", "").strip() if parts else ""

    # ------------------------------------------------------------------
    # History Plumbing
    # ------------------------------------------------------------------
    def append_assistant_turn(self, history: List[Any], resp: ProviderResponse) -> None:
        parts: List[Dict[str, Any]] = []
        if resp.text:
            parts.append({"text": resp.text})
        for tc in resp.tool_calls:
            parts.append({
                "functionCall": {
                    "name": tc.name,
                    "args": tc.arguments,
                }
            })
        history.append({"role": "model", "parts": parts})

    def append_tool_results(
        self, history: List[Any], tool_calls: List[ToolCall], results: Dict[str, str]
    ) -> None:
        parts: List[Dict[str, Any]] = []
        for tc in tool_calls:
            res_val = results.get(tc.id, "")
            parts.append({
                "functionResponse": {
                    "name": tc.name,
                    "response": {"output": res_val},
                }
            })
        history.append({"role": "function", "parts": parts})

    def append_user_turn(self, history: List[Any], text: str) -> None:
        history.append({"role": "user", "parts": [{"text": text}]})
