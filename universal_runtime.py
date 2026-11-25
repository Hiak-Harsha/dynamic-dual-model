"""
universal_runtime.py — Universal Model Runtime & CognitionLedger.

Per the Non-Negotiable Architectural Rules:
- All model communication MUST pass through this runtime.
- Provider-specific behavior is NEVER implemented inside Router, Muscle,
  Full Brain, Verification, Handoff, SuccessContract, ExecutionState, or Controller.
- The dual-mode architecture (Muscle + Brain) is the invariant.
  Providers and models are replaceable execution backends.

Components:
- UniversalModelRuntime: Single gateway for ALL model calls. Performs capability
  negotiation, invokes the correct provider method, and records every call
  in the CognitionLedger.
- CapabilityNegotiation: Resolves requested model features against provider
  capabilities, applying graceful fallbacks.
- CognitionLedger: Immutable, append-only audit trail of every model invocation
  with full provenance (who called, why, which provider, which model, tokens, latency).
- ModelCapabilityCard: Declarative capability manifest per provider+model pair.
"""

from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from base import (
    ModelProvider,
    ProviderCapabilities,
    ProviderResponse,
    ToolDef,
    count_tokens,
)


# ============================================================
# ModelCapabilityCard — declarative per-provider+model manifest
# ============================================================
@dataclass
class ModelCapabilityCard:
    """Declares what a specific provider+model combination supports.

    This is the contract between a provider adapter and the runtime.
    The runtime NEVER inspects provider internals — it reads this card.
    """
    provider_name: str
    model_id: str

    # Feature support flags
    supports_tool_calls: bool = True
    supports_structured_output: bool = True
    supports_reasoning_effort: bool = False
    supports_prompt_cache: bool = False
    supports_native_tool_history: bool = True
    supports_reasoning_usage: bool = False
    supports_streaming: bool = False
    supports_thinking: bool = False

    # Token limits
    max_input_tokens: int = 128000
    max_output_tokens: int = 8192
    min_output_tokens: int = 16

    # Effort levels supported (empty = all)
    supported_effort_levels: List[str] = field(
        default_factory=lambda: ["low", "medium", "high"]
    )

    # Cost per 1K tokens (input, output) for accounting
    input_cost_per_1k: float = 0.003
    output_cost_per_1k: float = 0.015

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# CognitionLedger — immutable audit trail
# ============================================================
@dataclass
class LedgerEntry:
    """Single immutable record of a model invocation."""
    entry_id: str
    timestamp: str
    phase: str          # "routing" | "muscle" | "brain" | "verification" | "summary" | "handoff"
    caller: str         # module/function that initiated the call
    provider_name: str
    model_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    latency_ms: float = 0.0
    effort: str = "low"
    tool_count: int = 0
    stop_reason: str = ""
    request_id: Optional[str] = None
    error: Optional[str] = None
    capability_fallbacks: List[str] = field(default_factory=list)

    def billable_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["billable_tokens"] = self.billable_tokens()
        return d


class CognitionLedger:
    """Append-only audit trail of every model invocation.

    The ledger is the single source of truth for:
    - Total token consumption across all phases
    - Per-phase cost attribution
    - Capability fallback tracking
    - Provider call provenance
    """

    def __init__(self):
        self._entries: List[LedgerEntry] = []

    def record(
        self,
        phase: str,
        caller: str,
        provider_name: str,
        model_id: str,
        resp: Optional[ProviderResponse] = None,
        effort: str = "low",
        latency_ms: float = 0.0,
        capability_fallbacks: Optional[List[str]] = None,
        error: Optional[str] = None,
    ) -> LedgerEntry:
        """Records a model invocation and returns the ledger entry."""
        entry = LedgerEntry(
            entry_id=str(uuid.uuid4())[:12],
            timestamp=datetime.now(timezone.utc).isoformat(),
            phase=phase,
            caller=caller,
            provider_name=provider_name,
            model_id=model_id,
            input_tokens=resp.input_tokens if resp else 0,
            output_tokens=resp.output_tokens if resp else 0,
            cache_read_tokens=resp.cache_read_tokens if resp else 0,
            cache_write_tokens=resp.cache_write_tokens if resp else 0,
            reasoning_tokens=resp.reasoning_tokens if resp else 0,
            latency_ms=latency_ms,
            effort=effort,
            tool_count=len(resp.tool_calls) if resp else 0,
            stop_reason=resp.stop_reason if resp else "",
            request_id=resp.request_id if resp else None,
            error=error,
            capability_fallbacks=capability_fallbacks or [],
        )
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> List[LedgerEntry]:
        return list(self._entries)

    def phase_summary(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate token/cost stats per phase."""
        summary: Dict[str, Dict[str, Any]] = {}
        for e in self._entries:
            if e.phase not in summary:
                summary[e.phase] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "billable_tokens": 0,
                    "cache_read_tokens": 0,
                    "total_latency_ms": 0.0,
                    "errors": 0,
                }
            s = summary[e.phase]
            s["calls"] += 1
            s["input_tokens"] += e.input_tokens
            s["output_tokens"] += e.output_tokens
            s["billable_tokens"] += e.billable_tokens()
            s["cache_read_tokens"] += e.cache_read_tokens
            s["total_latency_ms"] += e.latency_ms
            if e.error:
                s["errors"] += 1
        return summary

    def total_billable_tokens(self) -> int:
        return sum(e.billable_tokens() for e in self._entries)

    def total_calls(self) -> int:
        return len(self._entries)

    def fallback_count(self) -> int:
        return sum(1 for e in self._entries if e.capability_fallbacks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls(),
            "total_billable_tokens": self.total_billable_tokens(),
            "fallback_count": self.fallback_count(),
            "phase_summary": self.phase_summary(),
            "entries": [e.to_dict() for e in self._entries],
        }


# ============================================================
# CapabilityNegotiation — resolve requested features vs provider
# ============================================================
class CapabilityNegotiation:
    """Resolves requested model features against a ModelCapabilityCard.

    Returns an EffectiveConfig with the resolved parameters and any
    fallbacks that were applied. The runtime uses this to adjust
    provider call parameters WITHOUT the caller needing to know
    provider-specific details.
    """

    @staticmethod
    def negotiate(
        card: ModelCapabilityCard,
        requested_effort: str = "medium",
        requested_max_output_tokens: Optional[int] = None,
        requested_tools: Optional[List[ToolDef]] = None,
        requested_thinking: bool = False,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Returns (effective_config, fallbacks_applied).

        effective_config contains the resolved parameters.
        fallbacks_applied lists human-readable descriptions of any downgrades.
        """
        fallbacks: List[str] = []
        config: Dict[str, Any] = {}

        # Effort negotiation
        if requested_effort in card.supported_effort_levels:
            config["effort"] = requested_effort
        else:
            # Downgrade to nearest supported level
            effort_order = ["low", "medium", "high", "xhigh", "max"]
            req_idx = effort_order.index(requested_effort) if requested_effort in effort_order else 1
            available = [e for e in effort_order if e in card.supported_effort_levels]
            if available:
                # Pick the closest available level at or below requested
                resolved = available[0]
                for e in available:
                    if effort_order.index(e) <= req_idx:
                        resolved = e
                config["effort"] = resolved
                fallbacks.append(f"effort:{requested_effort}→{resolved}")
            else:
                config["effort"] = "medium"
                fallbacks.append(f"effort:{requested_effort}→medium (no supported levels)")

        # Output token negotiation
        if requested_max_output_tokens is not None:
            clamped = max(card.min_output_tokens, min(requested_max_output_tokens, card.max_output_tokens))
            config["max_output_tokens"] = clamped
            if clamped != requested_max_output_tokens:
                fallbacks.append(f"max_output_tokens:{requested_max_output_tokens}→{clamped}")
        else:
            config["max_output_tokens"] = card.max_output_tokens

        # Tool support negotiation
        if requested_tools:
            if card.supports_tool_calls:
                config["tools_enabled"] = True
            else:
                config["tools_enabled"] = False
                fallbacks.append("tools:requested→disabled (provider does not support tools)")
        else:
            config["tools_enabled"] = False

        # Thinking/reasoning negotiation
        if requested_thinking:
            if card.supports_thinking:
                config["thinking_enabled"] = True
            else:
                config["thinking_enabled"] = False
                fallbacks.append("thinking:requested→disabled")
        else:
            config["thinking_enabled"] = False

        # Reasoning effort support
        config["reasoning_effort_supported"] = card.supports_reasoning_effort

        # Prompt cache support
        config["prompt_cache_supported"] = card.supports_prompt_cache

        return config, fallbacks


# ============================================================
# UniversalModelRuntime — single gateway for ALL model calls
# ============================================================
class UniversalModelRuntime:
    """Universal gateway for all model communication.

    Invariant: ALL model calls from Router, Muscle, Brain, Verification,
    Handoff, and Controller MUST pass through this runtime.

    Responsibilities:
    1. Capability negotiation against the active provider's card
    2. Delegation to the correct provider method
    3. Recording every call in the CognitionLedger
    4. Applying provider-agnostic pre/post processing
    """

    def __init__(
        self,
        provider: ModelProvider,
        capability_card: Optional[ModelCapabilityCard] = None,
        ledger: Optional[CognitionLedger] = None,
    ):
        self._provider = provider
        self._card = capability_card or self._infer_capability_card(provider)
        self._ledger = ledger or CognitionLedger()

    @property
    def provider(self) -> ModelProvider:
        """Direct access to the underlying provider (for history plumbing)."""
        return self._provider

    @property
    def provider_name(self) -> str:
        return self._provider.name

    @property
    def name(self) -> str:
        return self._provider.name

    @property
    def model_id(self) -> str:
        return self._provider.model

    @property
    def model(self) -> str:
        return self._provider.model

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._provider.capabilities

    @property
    def capability_card(self) -> ModelCapabilityCard:
        return self._card

    @property
    def ledger(self) -> CognitionLedger:
        return self._ledger

    @staticmethod
    def _infer_capability_card(provider: ModelProvider) -> ModelCapabilityCard:
        """Infers a ModelCapabilityCard from provider.capabilities and known defaults."""
        caps = provider.capabilities
        name = provider.name
        model = provider.model

        # Provider-specific defaults
        if name == "openai":
            return ModelCapabilityCard(
                provider_name=name,
                model_id=model,
                supports_tool_calls=caps.supports_tool_calls,
                supports_structured_output=caps.supports_structured_output,
                supports_reasoning_effort=True,
                supports_prompt_cache=True,
                supports_native_tool_history=caps.supports_native_tool_history,
                supports_reasoning_usage=True,
                supports_thinking=False,
                supported_effort_levels=["low", "medium", "high"],
                max_input_tokens=128000,
                max_output_tokens=16384,
                min_output_tokens=16,
                input_cost_per_1k=0.003,
                output_cost_per_1k=0.015,
            )
        elif name == "anthropic":
            return ModelCapabilityCard(
                provider_name=name,
                model_id=model,
                supports_tool_calls=caps.supports_tool_calls,
                supports_structured_output=caps.supports_structured_output,
                supports_reasoning_effort=True,
                supports_prompt_cache=True,
                supports_native_tool_history=caps.supports_native_tool_history,
                supports_reasoning_usage=False,
                supports_thinking=True,
                supported_effort_levels=["low", "medium", "high", "xhigh", "max"],
                max_input_tokens=200000,
                max_output_tokens=8192,
                min_output_tokens=8,
                input_cost_per_1k=0.003,
                output_cost_per_1k=0.015,
            )
        elif name == "ollama":
            return ModelCapabilityCard(
                provider_name=name,
                model_id=model,
                supports_tool_calls=caps.supports_tool_calls,
                supports_structured_output=caps.supports_structured_output,
                supports_reasoning_effort=False,
                supports_prompt_cache=False,
                supports_native_tool_history=caps.supports_native_tool_history,
                supports_reasoning_usage=False,
                supports_thinking=False,
                supported_effort_levels=["low", "medium", "high"],
                max_input_tokens=128000,
                max_output_tokens=8192,
                min_output_tokens=16,
                input_cost_per_1k=0.0,
                output_cost_per_1k=0.0,
            )
        elif name == "gemini":
            return ModelCapabilityCard(
                provider_name=name,
                model_id=model,
                supports_tool_calls=caps.supports_tool_calls,
                supports_structured_output=caps.supports_structured_output,
                supports_reasoning_effort=False,
                supports_prompt_cache=False,
                supports_native_tool_history=caps.supports_native_tool_history,
                supports_reasoning_usage=False,
                supports_thinking=True,
                supported_effort_levels=["low", "medium", "high", "max"],
                max_input_tokens=1000000,
                max_output_tokens=8192,
                min_output_tokens=16,
                input_cost_per_1k=0.00015,
                output_cost_per_1k=0.0006,
            )
        elif name == "openrouter":
            return ModelCapabilityCard(
                provider_name=name,
                model_id=model,
                supports_tool_calls=caps.supports_tool_calls,
                supports_structured_output=caps.supports_structured_output,
                supports_reasoning_effort=False,
                supports_prompt_cache=False,
                supports_native_tool_history=caps.supports_native_tool_history,
                supports_reasoning_usage=False,
                supports_thinking=False,
                supported_effort_levels=["low", "medium", "high"],
                max_input_tokens=128000,
                max_output_tokens=8192,
                min_output_tokens=16,
                input_cost_per_1k=0.002,
                output_cost_per_1k=0.008,
            )
        else:
            # Mock or unknown provider — conservative defaults
            return ModelCapabilityCard(
                provider_name=name,
                model_id=model,
                supports_tool_calls=caps.supports_tool_calls,
                supports_structured_output=caps.supports_structured_output,
                supports_reasoning_effort=caps.supports_reasoning_effort,
                supports_prompt_cache=caps.supports_prompt_cache,
                supports_native_tool_history=caps.supports_native_tool_history,
                supports_reasoning_usage=caps.supports_reasoning_usage,
                supported_effort_levels=["low", "medium", "high"],
                max_input_tokens=128000,
                max_output_tokens=8192,
                min_output_tokens=1,
                input_cost_per_1k=0.0,
                output_cost_per_1k=0.0,
            )

    # ------------------------------------------------------------------
    # Phase: Classification / Routing
    # ------------------------------------------------------------------
    def classify(self, user_message: str, recent_summary: str) -> Tuple[str, ProviderResponse]:
        """Route classification through the runtime with ledger recording."""
        t0 = time.perf_counter()
        result = self._provider.classify(user_message, recent_summary)
        latency = (time.perf_counter() - t0) * 1000

        # Create a synthetic ProviderResponse for classify (providers return str)
        resp = ProviderResponse(
            text=result,
            tool_calls=[],
            stop_reason="end_turn",
            raw=None,
            input_tokens=count_tokens(user_message + recent_summary),
            output_tokens=count_tokens(result),
        )
        self._ledger.record(
            phase="routing",
            caller="runtime.classify",
            provider_name=self._provider.name,
            model_id=self._provider.model,
            resp=resp,
            effort="low",
            latency_ms=latency,
        )
        return result, resp

    def route_classify(self, user_message: str, recent_summary: str) -> Dict[str, Any]:
        """Structured multi-axis routing through the runtime with ledger recording."""
        t0 = time.perf_counter()
        result = self._provider.route_classify(user_message, recent_summary)
        latency = (time.perf_counter() - t0) * 1000

        provider_resp = result.get("_response")
        self._ledger.record(
            phase="routing",
            caller="runtime.route_classify",
            provider_name=self._provider.name,
            model_id=self._provider.model,
            resp=provider_resp,
            effort="low",
            latency_ms=latency,
        )
        return result

    # ------------------------------------------------------------------
    # Phase: Muscle Memory
    # ------------------------------------------------------------------
    def muscle_memory_reply(self, user_message: str) -> ProviderResponse:
        """Muscle memory execution through the runtime."""
        config, fallbacks = CapabilityNegotiation.negotiate(
            self._card,
            requested_effort="low",
            requested_max_output_tokens=120,
        )

        t0 = time.perf_counter()
        resp = self._provider.muscle_memory_reply(user_message)
        latency = (time.perf_counter() - t0) * 1000

        self._ledger.record(
            phase="muscle",
            caller="runtime.muscle_memory_reply",
            provider_name=self._provider.name,
            model_id=self._provider.model,
            resp=resp,
            effort="low",
            latency_ms=latency,
            capability_fallbacks=fallbacks,
        )
        return resp

    # ------------------------------------------------------------------
    # Phase: Full Brain
    # ------------------------------------------------------------------
    def full_brain_step(
        self,
        history: List[Any],
        tools: List[ToolDef],
        effort: str,
    ) -> ProviderResponse:
        """Single Full Brain step through the runtime."""
        config, fallbacks = CapabilityNegotiation.negotiate(
            self._card,
            requested_effort=effort,
            requested_tools=tools if tools else None,
        )
        effective_effort = config.get("effort", effort)

        t0 = time.perf_counter()
        resp = self._provider.full_brain_step(history, tools, effective_effort)
        latency = (time.perf_counter() - t0) * 1000

        self._ledger.record(
            phase="brain",
            caller="runtime.full_brain_step",
            provider_name=self._provider.name,
            model_id=self._provider.model,
            resp=resp,
            effort=effective_effort,
            latency_ms=latency,
            capability_fallbacks=fallbacks,
        )
        return resp

    # ------------------------------------------------------------------
    # Phase: Summary Compression
    # ------------------------------------------------------------------
    def summarize_history_step(self, history_text: str) -> ProviderResponse:
        """History compression through the runtime."""
        t0 = time.perf_counter()
        resp = self._provider.summarize_history_step(history_text)
        latency = (time.perf_counter() - t0) * 1000

        self._ledger.record(
            phase="summary",
            caller="runtime.summarize_history_step",
            provider_name=self._provider.name,
            model_id=self._provider.model,
            resp=resp,
            effort="low",
            latency_ms=latency,
        )
        return resp

    def summarize_history(self, history_text: str) -> str:
        """Summarize conversation history through runtime."""
        return self.summarize_history_step(history_text).text

    # ------------------------------------------------------------------
    # History plumbing — delegated directly (no model call involved)
    # ------------------------------------------------------------------
    def append_assistant_turn(self, history: List[Any], resp: ProviderResponse) -> None:
        self._provider.append_assistant_turn(history, resp)

    def append_tool_results(
        self, history: List[Any], tool_calls: List[Any], results: Dict[str, str]
    ) -> None:
        self._provider.append_tool_results(history, tool_calls, results)

    def append_user_turn(self, history: List[Any], text: str) -> None:
        self._provider.append_user_turn(history, text)

    # ------------------------------------------------------------------
    # Runtime introspection
    # ------------------------------------------------------------------
    def runtime_status(self) -> Dict[str, Any]:
        """Returns runtime health and accounting summary."""
        return {
            "provider": self._provider.name,
            "model": self._provider.model,
            "capability_card": self._card.to_dict(),
            "ledger_summary": {
                "total_calls": self._ledger.total_calls(),
                "total_billable_tokens": self._ledger.total_billable_tokens(),
                "fallback_count": self._ledger.fallback_count(),
                "phase_summary": self._ledger.phase_summary(),
            },
        }
