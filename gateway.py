"""
gateway.py — Universal Dual-Mode Gateway Router, Protocol Normalizer & Telemetry.

Provides a vendor-neutral, cross-platform gateway layer compatible with:
- OpenAI Chat Completions API (/v1/chat/completions)
- OpenAI Responses API (/v1/responses)
- Anthropic Messages API (/v1/messages)

Routes safe deterministic, cached, and routine requests through the local
Muscle Memory tier (0 tokens or cheap/local model) while forwarding complex,
tool-using, context-dependent, or high-risk requests to the configured
Full Brain / flagship upstream with full native fidelity.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from base import ConversationState, ProviderResponse, ResponseCache, count_tokens
from execution_state import ExecutionState, COMPLETED_VERIFIED
from muscle_engine import MuscleExecutor
from procedural_memory import ProceduralMemory
from provider_registry import build_default_registry
from router import (
    CONTEXTUAL_ACK_PATTERN,
    DESTRUCTIVE_PATTERN,
    FIRST_PERSON_HISTORY,
    FRESHNESS_PATTERN,
    GREETING_EXPANDED,
    INJECTION_PATTERN,
    REFERENCE_PATTERN,
    TOOL_PATTERN,
    RouteDecision,
    route,
)
from success_contracts import resolve_contract
from universal_runtime import CognitionLedger, UniversalModelRuntime


PROTOCOL_OPENAI_CHAT = "openai_chat"
PROTOCOL_OPENAI_RESPONSES = "openai_responses"
PROTOCOL_ANTHROPIC = "anthropic"

DEFAULT_UPSTREAM_URLS = {
    "openai": "https://api.openai.com",
    "anthropic": "https://api.anthropic.com",
    "openrouter": "https://openrouter.ai/api",
    "ollama": "http://localhost:11434",
}

ESTIMATED_FLAGSHIP_SAVED_TOKENS_FLOOR = 600


@dataclass
class NormalizedRequest:
    """Canonical representation of an incoming OpenAI or Anthropic API request."""
    protocol: str
    model: str
    latest_user_text: str
    system_text: str
    turn_count: int
    has_tools_attached: bool
    has_tool_history: bool
    stream: bool
    estimated_input_tokens: int
    raw_payload: Dict[str, Any]


@dataclass
class GatewayDecision:
    """Outcome of gateway routing and execution."""
    handled_locally: bool
    pathway: str  # "cache" | "muscle_deterministic" | "muscle_model" | "brain_upstream"
    reason: str
    text: str = ""
    model_used: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_tokens_saved: int = 0
    latency_ms: float = 0.0
    escalated_from_muscle: bool = False


@dataclass
class GatewayStats:
    """Thread-safe telemetry tracking token, cost, and latency savings."""
    total_requests: int = 0
    cache_hits: int = 0
    muscle_deterministic_hits: int = 0
    muscle_model_hits: int = 0
    brain_upstream_forwards: int = 0
    escalations_to_brain: int = 0
    muscle_input_tokens: int = 0
    muscle_output_tokens: int = 0
    brain_input_tokens: int = 0
    brain_output_tokens: int = 0
    estimated_tokens_saved: int = 0
    recent_events: List[Dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, decision: GatewayDecision, protocol: str, prompt_preview: str) -> None:
        with self._lock:
            self.total_requests += 1
            if decision.pathway == "cache":
                self.cache_hits += 1
            elif decision.pathway == "muscle_deterministic":
                self.muscle_deterministic_hits += 1
            elif decision.pathway == "muscle_model":
                self.muscle_model_hits += 1
                self.muscle_input_tokens += decision.input_tokens
                self.muscle_output_tokens += decision.output_tokens
            else:
                self.brain_upstream_forwards += 1
                self.brain_input_tokens += decision.input_tokens
                self.brain_output_tokens += decision.output_tokens

            if decision.escalated_from_muscle:
                self.escalations_to_brain += 1
            self.estimated_tokens_saved += max(0, decision.estimated_tokens_saved)

            self.recent_events.append({
                "protocol": protocol,
                "pathway": decision.pathway,
                "handled_locally": decision.handled_locally,
                "model_used": decision.model_used,
                "tokens_saved": decision.estimated_tokens_saved,
                "latency_ms": round(decision.latency_ms, 2),
                "reason": decision.reason,
                "prompt_preview": prompt_preview[:80],
            })
            if len(self.recent_events) > 50:
                self.recent_events = self.recent_events[-50:]

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            local_hits = self.cache_hits + self.muscle_deterministic_hits + self.muscle_model_hits
            offload_pct = (local_hits / self.total_requests * 100.0) if self.total_requests else 0.0
            # Approximate flagship rate avoided ($3/M in + $15/M out blended ~$0.006/1k tokens)
            est_usd_saved = (self.estimated_tokens_saved / 1000.0) * 0.006
            return {
                "total_requests": self.total_requests,
                "local_muscle_or_cache_hits": local_hits,
                "offload_rate_pct": round(offload_pct, 1),
                "breakdown": {
                    "cache": self.cache_hits,
                    "muscle_deterministic": self.muscle_deterministic_hits,
                    "muscle_model": self.muscle_model_hits,
                    "brain_upstream": self.brain_upstream_forwards,
                    "escalations_to_brain": self.escalations_to_brain,
                },
                "tokens": {
                    "muscle_input": self.muscle_input_tokens,
                    "muscle_output": self.muscle_output_tokens,
                    "brain_input": self.brain_input_tokens,
                    "brain_output": self.brain_output_tokens,
                    "estimated_flagship_tokens_saved": self.estimated_tokens_saved,
                },
                "estimated_usd_saved": round(est_usd_saved, 4),
                "recent_events": list(self.recent_events[-10:]),
            }


def _extract_text_from_content(content: Any) -> Tuple[str, bool]:
    """Extracts plain text and detects embedded tool blocks from string or block lists."""
    if isinstance(content, str):
        return content, False
    if not isinstance(content, list):
        return "", False

    texts: List[str] = []
    has_tool_block = False
    for block in content:
        if isinstance(block, str):
            texts.append(block)
        elif isinstance(block, dict):
            btype = str(block.get("type", "")).lower()
            if btype in {"tool_use", "tool_result", "function_call", "function_call_output"}:
                has_tool_block = True
            elif btype in {"text", "input_text", "output_text"} and isinstance(block.get("text"), str):
                texts.append(block["text"])
            elif isinstance(block.get("content"), str):
                texts.append(block["content"])
    return "\n".join(t for t in texts if t), has_tool_block


def normalize_request(protocol: str, payload: Dict[str, Any]) -> NormalizedRequest:
    """Normalizes OpenAI Chat, OpenAI Responses, and Anthropic Messages payloads."""
    model = str(payload.get("model") or "default")
    stream = bool(payload.get("stream", False))
    has_tools_attached = bool(payload.get("tools") or payload.get("functions"))
    has_tool_history = False
    system_parts: List[str] = []
    user_texts: List[str] = []
    turn_count = 0

    if protocol == PROTOCOL_ANTHROPIC:
        sys_text, _ = _extract_text_from_content(payload.get("system", ""))
        if sys_text:
            system_parts.append(sys_text)
        messages = payload.get("messages") or []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).lower()
            turn_count += 1
            txt, has_tb = _extract_text_from_content(msg.get("content", ""))
            if has_tb:
                has_tool_history = True
            if role == "user" and txt.strip() and not has_tb:
                user_texts.append(txt.strip())

    elif protocol == PROTOCOL_OPENAI_RESPONSES:
        instr = payload.get("instructions")
        if isinstance(instr, str) and instr.strip():
            system_parts.append(instr.strip())
        inp = payload.get("input", "")
        if isinstance(inp, str):
            turn_count = 1
            if inp.strip():
                user_texts.append(inp.strip())
        elif isinstance(inp, list):
            for item in inp:
                if not isinstance(item, dict):
                    continue
                itype = str(item.get("type", "")).lower()
                if itype in {"function_call", "function_call_output"}:
                    has_tool_history = True
                    continue
                role = str(item.get("role", "user")).lower()
                turn_count += 1
                txt, has_tb = _extract_text_from_content(item.get("content", ""))
                if has_tb:
                    has_tool_history = True
                if role in {"system", "developer"} and txt.strip():
                    system_parts.append(txt.strip())
                elif role == "user" and txt.strip() and not has_tb:
                    user_texts.append(txt.strip())

    else:  # PROTOCOL_OPENAI_CHAT
        messages = payload.get("messages") or []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).lower()
            if role == "tool" or msg.get("tool_calls") or msg.get("function_call"):
                has_tool_history = True
                turn_count += 1
                continue
            turn_count += 1
            txt, has_tb = _extract_text_from_content(msg.get("content", ""))
            if has_tb:
                has_tool_history = True
            if role in {"system", "developer"} and txt.strip():
                system_parts.append(txt.strip())
            elif role == "user" and txt.strip() and not has_tb:
                user_texts.append(txt.strip())

    latest_user_text = user_texts[-1] if user_texts else ""
    system_text = "\n".join(system_parts)
    estimated_input_tokens = count_tokens(json.dumps(payload, default=str))

    return NormalizedRequest(
        protocol=protocol,
        model=model,
        latest_user_text=latest_user_text,
        system_text=system_text,
        turn_count=turn_count,
        has_tools_attached=has_tools_attached,
        has_tool_history=has_tool_history,
        stream=stream,
        estimated_input_tokens=estimated_input_tokens,
        raw_payload=payload,
    )


def format_json_response(
    protocol: str,
    model: str,
    text: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    pathway: str = "muscle",
) -> Dict[str, Any]:
    """Formats a completed local Muscle/Cache answer into the caller's exact API schema."""
    req_id = uuid.uuid4().hex[:20]
    created = int(time.time())

    if protocol == PROTOCOL_ANTHROPIC:
        return {
            "id": f"msg_{req_id}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            "dual_mode_pathway": pathway,
        }

    if protocol == PROTOCOL_OPENAI_RESPONSES:
        return {
            "id": f"resp_{req_id}",
            "object": "response",
            "created_at": created,
            "status": "completed",
            "model": model,
            "output": [
                {
                    "id": f"msg_{req_id}",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "output_text": text,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            "dual_mode_pathway": pathway,
        }

    # OpenAI Chat Completions
    return {
        "id": f"chatcmpl-{req_id}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "dual_mode_pathway": pathway,
    }


def format_sse_stream(
    protocol: str,
    model: str,
    text: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> List[bytes]:
    """Formats a local Muscle/Cache answer as valid SSE chunks for streaming clients."""
    req_id = uuid.uuid4().hex[:20]
    created = int(time.time())
    chunks: List[bytes] = []

    def _emit(event: Optional[str], data: Any) -> None:
        payload = data if isinstance(data, str) else json.dumps(data)
        if event:
            chunks.append(f"event: {event}\ndata: {payload}\n\n".encode("utf-8"))
        else:
            chunks.append(f"data: {payload}\n\n".encode("utf-8"))

    if protocol == PROTOCOL_ANTHROPIC:
        _emit("message_start", {
            "type": "message_start",
            "message": {
                "id": f"msg_{req_id}",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        })
        _emit("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
        _emit("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}})
        _emit("content_block_stop", {"type": "content_block_stop", "index": 0})
        _emit("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": output_tokens}})
        _emit("message_stop", {"type": "message_stop"})
        return chunks

    if protocol == PROTOCOL_OPENAI_RESPONSES:
        resp_obj = format_json_response(protocol, model, text, input_tokens, output_tokens)
        _emit("response.created", {"type": "response.created", "response": {"id": resp_obj["id"], "object": "response", "status": "in_progress", "model": model}})
        _emit("response.output_text.delta", {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": text})
        _emit("response.completed", {"type": "response.completed", "response": resp_obj})
        return chunks

    # OpenAI Chat Completions SSE
    _emit(None, {
        "id": f"chatcmpl-{req_id}",
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
    })
    _emit(None, {
        "id": f"chatcmpl-{req_id}",
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    })
    _emit(None, "[DONE]")
    return chunks


class GatewayEngine:
    """Universal dual-mode request router and executor.

    Handles Cache & Muscle Memory requests locally and escalates/forwards
    complex, coding, tool, contextual, and high-risk turns to Brain upstream.
    """

    def __init__(
        self,
        muscle_runtime: Optional[UniversalModelRuntime] = None,
        brain_runtime: Optional[UniversalModelRuntime] = None,
        brain_mode: str = "passthrough",
        upstream_url: Optional[str] = None,
        cache: Optional[ResponseCache] = None,
        memory: Optional[ProceduralMemory] = None,
        ledger: Optional[CognitionLedger] = None,
    ):
        self.ledger = ledger or CognitionLedger()
        self.cache = cache or ResponseCache()
        self.memory = memory or ProceduralMemory()
        self.muscle_runtime = muscle_runtime
        self.brain_runtime = brain_runtime
        self.brain_mode = brain_mode  # "passthrough" | "provider"
        self.upstream_url = upstream_url
        self.stats = GatewayStats()
        self._muscle_executor = MuscleExecutor(
            provider=muscle_runtime.provider if muscle_runtime else None,
            memory=self.memory,
            cache=self.cache,
        )

    @classmethod
    def from_config(
        cls,
        muscle_provider: Optional[str] = None,
        muscle_model: Optional[str] = None,
        brain_provider: str = "passthrough",
        brain_model: Optional[str] = None,
        upstream_url: Optional[str] = None,
    ) -> "GatewayEngine":
        """Builds a GatewayEngine using ProviderRegistry with sensible cross-platform defaults."""
        registry = build_default_registry()
        ledger = CognitionLedger()

        # Auto-select zero-cost local muscle tier if available, else mock if explicit
        chosen_muscle = muscle_provider
        if not chosen_muscle:
            ollama_factory = registry.get_factory("ollama")
            if ollama_factory and ollama_factory.is_available():
                chosen_muscle = "ollama"
            else:
                chosen_muscle = "mock"

        muscle_rt: Optional[UniversalModelRuntime] = None
        if chosen_muscle != "none":
            muscle_rt = registry.create_runtime(chosen_muscle, model=muscle_model, ledger=ledger)

        brain_rt: Optional[UniversalModelRuntime] = None
        brain_mode = "passthrough"
        resolved_upstream = upstream_url
        if brain_provider and brain_provider != "passthrough":
            brain_rt = registry.create_runtime(brain_provider, model=brain_model, ledger=ledger)
            brain_mode = "provider"
            if not resolved_upstream:
                resolved_upstream = DEFAULT_UPSTREAM_URLS.get(brain_provider)

        return cls(
            muscle_runtime=muscle_rt,
            brain_runtime=brain_rt,
            brain_mode=brain_mode,
            upstream_url=resolved_upstream,
            ledger=ledger,
        )

    def evaluate_route(self, req: NormalizedRequest) -> Tuple[bool, Optional[RouteDecision], str]:
        """Determines whether a normalized request can be safely handled by Cache or Muscle.

        Returns (can_try_local, route_decision, reason).
        """
        text = req.latest_user_text.strip()
        if not text:
            return False, None, "Empty user message; forwarding to Brain upstream."

        if req.has_tool_history:
            return False, None, "Active tool-call history present; preserving Brain loop."

        if INJECTION_PATTERN.search(text):
            return False, None, "Prompt injection signal; strictly routed to Brain."

        if DESTRUCTIVE_PATTERN.search(text):
            return False, None, "Destructive intent detected; strictly routed to Brain."

        if FRESHNESS_PATTERN.search(text):
            return False, None, "Freshness-sensitive query; routed to Brain."

        if REFERENCE_PATTERN.search(text) or FIRST_PERSON_HISTORY.search(text) or CONTEXTUAL_ACK_PATTERN.match(text):
            return False, None, "Context-dependent turn; routed to Brain."

        # If caller attached tools (e.g., Claude Code, Codex, Cursor) and prompt mentions files/code, never hijack
        if req.has_tools_attached and TOOL_PATTERN.search(text):
            return False, None, "Coding/file tool request with attached tools; routed to Brain."

        # Run authoritative router deterministic gates (without spending tokens on classifier)
        conv_state = ConversationState()
        decision = route(
            user_message=text,
            state=conv_state,
            cache=self.cache,
            memory=self.memory,
            provider=None,  # Deterministic gates only; avoid classifier cost on gateway hot path
        )

        if decision.pathway == "cache":
            return True, decision, "Exact match in zero-token ResponseCache."

        if decision.pathway == "muscle" and decision.procedure_match:
            # Deterministic Tier 1 (AST math, JSON validation, text stats) is safe if no active tool loop
            if not req.has_tools_attached or decision.procedure_match in {
                "deterministic_arithmetic",
                "deterministic_json_validation",
                "deterministic_text_statistics",
            }:
                return True, decision, f"Deterministic Muscle procedure: {decision.procedure_match}."

        # Greetings / routine acks can always use local Muscle even if coding agent attached 20 tools
        if GREETING_EXPANDED.match(text) and self.muscle_runtime is not None:
            return True, decision, "Routine greeting/acknowledgment eligible for local Muscle."

        # Standalone non-tool simple query on a plain chat/QA client
        if (
            not req.has_tools_attached
            and req.turn_count <= 2
            and decision.pathway == "muscle"
            and decision.risk == "low"
            and decision.novelty <= 0.25
            and self.muscle_runtime is not None
        ):
            return True, decision, "Stateless low-risk simple query eligible for local Muscle."

        return False, decision, "Complex, multi-turn, or tool-capable request; forwarding to Brain."

    def try_local_execution(self, req: NormalizedRequest) -> Optional[GatewayDecision]:
        """Attempts safe Tier 0 (cache) or Tier 1/2 (Muscle) execution.

        Returns GatewayDecision if handled with verified/acceptable quality, or None to forward to Brain.
        """
        t0 = time.perf_counter()
        can_local, decision, reason = self.evaluate_route(req)
        if not can_local or decision is None:
            return None

        text = req.latest_user_text.strip()
        exec_state = ExecutionState(objective=text)

        # Tier 0: ResponseCache
        if decision.pathway == "cache":
            cached = self.cache.lookup(text)
            if cached is not None:
                dt = (time.perf_counter() - t0) * 1000
                saved = max(ESTIMATED_FLAGSHIP_SAVED_TOKENS_FLOOR, req.estimated_input_tokens)
                return GatewayDecision(
                    handled_locally=True,
                    pathway="cache",
                    reason=reason,
                    text=cached.text,
                    model_used="response-cache",
                    input_tokens=0,
                    output_tokens=0,
                    estimated_tokens_saved=saved,
                    latency_ms=dt,
                )

        # Tier 1 / Tier 2 via MuscleExecutor
        m_res = self._muscle_executor.execute(
            user_message=text,
            exec_state=exec_state,
            decision=decision,
            conv_state=ConversationState(),
        )

        # Deterministic execution verified by SuccessContract
        if decision.procedure_match and m_res.status == COMPLETED_VERIFIED:
            contract = resolve_contract(text, provider=self.muscle_runtime)
            c_res = contract.evaluate(exec_state, m_res.text, exec_state)
            if c_res.verified or c_res.task_satisfied:
                resp_obj = ProviderResponse(
                    text=m_res.text,
                    tool_calls=[],
                    stop_reason="end_turn",
                    raw=None,
                    input_tokens=0,
                    output_tokens=0,
                )
                self.cache.put(text, resp_obj)
                dt = (time.perf_counter() - t0) * 1000
                saved = max(ESTIMATED_FLAGSHIP_SAVED_TOKENS_FLOOR, req.estimated_input_tokens)
                return GatewayDecision(
                    handled_locally=True,
                    pathway="muscle_deterministic",
                    reason=reason,
                    text=m_res.text,
                    model_used="muscle-deterministic",
                    input_tokens=0,
                    output_tokens=0,
                    estimated_tokens_saved=saved,
                    latency_ms=dt,
                )

        # Model-backed Muscle reply (only if Muscle succeeded cleanly without fake tool stubs)
        if m_res.is_completed and m_res.text.strip():
            candidate = m_res.text.strip()
            if self.muscle_runtime and hasattr(self.muscle_runtime.provider, "_looks_like_hallucinated_tool_json"):
                if self.muscle_runtime.provider._looks_like_hallucinated_tool_json(candidate):
                    return None
            dt = (time.perf_counter() - t0) * 1000
            muscle_tokens = m_res.input_tokens + m_res.output_tokens
            # If the client had a big system prompt + tool definitions, Muscle bypassed all of it
            saved = max(0, req.estimated_input_tokens - muscle_tokens)
            if GREETING_EXPANDED.match(text):
                self.cache.put(
                    text,
                    ProviderResponse(text=candidate, tool_calls=[], stop_reason="end_turn", raw=None),
                )
            model_name = self.muscle_runtime.model_id if self.muscle_runtime else "muscle"
            return GatewayDecision(
                handled_locally=True,
                pathway="muscle_model",
                reason=reason,
                text=candidate,
                model_used=model_name,
                input_tokens=m_res.input_tokens,
                output_tokens=m_res.output_tokens,
                estimated_tokens_saved=saved,
                latency_ms=dt,
            )

        return None

    def execute_brain_fallback(self, req: NormalizedRequest) -> GatewayDecision:
        """Executes a request via configured Brain provider adapter when not using HTTP passthrough."""
        t0 = time.perf_counter()
        if self.brain_runtime is None:
            raise RuntimeError("No brain_runtime configured for non-passthrough execution.")

        history: List[Dict[str, Any]] = []
        if req.system_text:
            history.append({"role": "system", "content": req.system_text})
        # Reconstruct plain message turns cleanly for adapter fallback
        if req.protocol == PROTOCOL_OPENAI_CHAT:
            for m in req.raw_payload.get("messages") or []:
                if isinstance(m, dict) and m.get("role") in {"user", "assistant"}:
                    txt, _ = _extract_text_from_content(m.get("content", ""))
                    if txt:
                        history.append({"role": m["role"], "content": txt})
        elif req.protocol == PROTOCOL_ANTHROPIC:
            for m in req.raw_payload.get("messages") or []:
                if isinstance(m, dict) and m.get("role") in {"user", "assistant"}:
                    txt, _ = _extract_text_from_content(m.get("content", ""))
                    if txt:
                        history.append({"role": m["role"], "content": txt})
        else:
            if req.latest_user_text:
                history.append({"role": "user", "content": req.latest_user_text})

        resp = self.brain_runtime.full_brain_step(history=history, tools=[], effort="medium")
        dt = (time.perf_counter() - t0) * 1000
        return GatewayDecision(
            handled_locally=False,
            pathway="brain_upstream",
            reason="Executed via configured Brain provider adapter.",
            text=resp.text,
            model_used=self.brain_runtime.model_id,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            estimated_tokens_saved=0,
            latency_ms=dt,
        )


def resolve_upstream_endpoint(
    protocol: str,
    headers: Dict[str, str],
    configured_upstream_url: Optional[str] = None,
) -> Tuple[Optional[str], Dict[str, str]]:
    """Determines the target upstream URL and auth headers for native passthrough."""
    lower_headers = {k.lower(): v for k, v in headers.items()}
    fwd_headers: Dict[str, str] = {"Content-Type": "application/json"}

    if protocol == PROTOCOL_ANTHROPIC:
        base = (configured_upstream_url or os.environ.get("ANTHROPIC_UPSTREAM_URL") or DEFAULT_UPSTREAM_URLS["anthropic"]).rstrip("/")
        path = "/v1/messages"
        api_key = lower_headers.get("x-api-key") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None, {}
        fwd_headers["x-api-key"] = api_key
        fwd_headers["anthropic-version"] = lower_headers.get("anthropic-version", "2023-06-01")
        if "anthropic-beta" in lower_headers:
            fwd_headers["anthropic-beta"] = lower_headers["anthropic-beta"]
        return f"{base}{path}", fwd_headers

    # OpenAI Chat or Responses
    base = (configured_upstream_url or os.environ.get("OPENAI_UPSTREAM_URL") or DEFAULT_UPSTREAM_URLS["openai"]).rstrip("/")
    if base.endswith("/openai"):
        path = "/responses" if protocol == PROTOCOL_OPENAI_RESPONSES else "/chat/completions"
    else:
        if base.endswith("/v1"):
            base = base[:-3]
        path = "/v1/responses" if protocol == PROTOCOL_OPENAI_RESPONSES else "/v1/chat/completions"
    auth = lower_headers.get("authorization")
    if not auth:
        env_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        if env_key:
            auth = f"Bearer {env_key}"
    if not auth and "localhost" not in base and "127.0.0.1" not in base:
        return None, {}
    if auth:
        fwd_headers["Authorization"] = auth
    return f"{base}{path}", fwd_headers
