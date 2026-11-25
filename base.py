"""
base.py — provider-agnostic core types.

Every model backend (Anthropic, OpenAI, ...) implements the ModelProvider
interface below. Nothing outside this file should need to know which
vendor is being used.

DESIGN NOTE — SAME-MODEL DUAL MODE:
This architecture uses ONE model for everything. The classifier, Muscle
Memory, and Full Brain all call the same model ID. They are differentiated
only by:
  - effort level   (low → classifier/muscle, medium/high/xhigh → full brain)
  - max_tokens     (8 → classifier, 120 → muscle, 8000 → full brain)
  - tools          (none → classifier/muscle, full toolset → full brain)
  - context        (bare message → muscle, full history → full brain)

This is NOT a FrugalGPT cascade where a cheap model triages to a flagship.
It is a single model operating at different intensity levels depending on
how much the turn actually needs.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from time import time
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================
# Canonical (vendor-neutral) tool definition
# ============================================================
@dataclass
class ToolDef:
    name: str
    description: str
    parameters: Dict[str, Any]  # a JSON Schema "object" definition


# ============================================================
# Canonical response shape every adapter must return
# ============================================================
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ProviderResponse:
    text: str                          # assistant's text, may be ""
    tool_calls: List[ToolCall]         # empty if no tool use
    stop_reason: str                   # "end_turn" | "tool_use"
    raw: Any                           # original SDK object, for logging
    # CONTRACT (both adapters must follow this — it's what makes stats
    # comparable across vendors, since Anthropic and OpenAI report cache
    # accounting with DIFFERENT semantics natively):
    #   input_tokens       = TOTAL prompt tokens billed for this call,
    #                         cache hits and cache writes INCLUDED. This is
    #                         OpenAI's native meaning of prompt/input_tokens
    #                         already; Anthropic's native input_tokens means
    #                         only the FRESH (non-cached) portion, so the
    #                         Anthropic adapter must add cache_read +
    #                         cache_write into it before returning.
    #   cache_read_tokens  = subset of input_tokens that hit the cache.
    #   cache_write_tokens = subset of input_tokens newly written to cache.
    # SessionStats below assumes this contract; do not add cache_read/write
    # on top of input_tokens when computing totals, or you'll double-count.
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0          # reasoning/thinking tokens if reported by provider
    request_id: Optional[str] = None   # provider request ID for telemetry
    # Provider-native representation of THIS turn's assistant message,
    # already in the shape that provider expects to see echoed back in
    # history (content blocks for Anthropic, output items for OpenAI, etc).
    # Treated as an opaque blob by everything outside the adapter.
    native_assistant_turn: Any = None


# ============================================================
# V5 Execution Budget & Usage Tracking
# ============================================================
@dataclass
class ExecutionBudget:
    """Execution bounds enforced across both Muscle and Brain execution paths."""
    max_model_calls: int = 12
    max_input_tokens: int = 32000
    max_output_tokens: int = 4000
    max_tool_calls: int = 12
    max_wall_time: float = 60.0
    max_state_tokens: int = 4000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_model_calls": self.max_model_calls,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_tool_calls": self.max_tool_calls,
            "max_wall_time": self.max_wall_time,
            "max_state_tokens": self.max_state_tokens,
        }


@dataclass
class BrainBudget:
    """Abstract reasoning intensity and resource budget for Full Brain."""
    effort: str = "medium"  # "low" | "medium" | "high" | "xhigh"
    max_output_tokens: int = 4000
    max_tool_iterations: int = 8
    enable_thinking: bool = False
    thinking_budget: Optional[int] = None


@dataclass
class ProviderCapabilities:
    """Provider feature capabilities and active configuration per V6 Section AW."""
    supports_reasoning_effort: bool = False
    supports_prompt_cache: bool = False
    supports_structured_output: bool = True
    supports_native_tool_history: bool = True
    supports_reasoning_usage: bool = False
    supports_tool_calls: bool = True
    requested_config: Dict[str, Any] = field(default_factory=dict)
    effective_config: Dict[str, Any] = field(default_factory=dict)
    fallback_applied: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskUsage:
    """Rigorous per-task token and resource accounting across all sub-phases."""
    task_id: str = ""
    routing_tokens: int = 0
    muscle_tokens: int = 0
    handoff_tokens: int = 0
    brain_tokens: int = 0
    verification_tokens: int = 0
    summary_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_billable_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    estimated_context_tokens: int = 0
    estimated_handoff_tokens: int = 0
    source: str = "simulated"  # "live" | "simulated"
    routing_recorded: bool = False
    recorded_request_ids: List[str] = field(default_factory=list)
    reuse_count: int = 0
    recomputation_count: int = 0
    verification_repeat_count: int = 0
    legitimate_repeat_count: int = 0

    def record_call(self, phase: str, resp: ProviderResponse) -> None:
        if resp.request_id and resp.request_id in self.recorded_request_ids:
            # Already recorded this exact request ID
            return
        if resp.request_id:
            self.recorded_request_ids.append(resp.request_id)

        self.model_calls += 1
        in_tok = resp.input_tokens
        out_tok = resp.output_tokens
        self.total_input_tokens += in_tok
        self.total_output_tokens += out_tok
        self.total_billable_tokens += (in_tok + out_tok)
        self.cache_read_tokens += resp.cache_read_tokens
        self.cache_write_tokens += resp.cache_write_tokens

        if phase == "routing":
            self.routing_tokens += (in_tok + out_tok)
            self.routing_recorded = True
        elif phase == "muscle":
            self.muscle_tokens += (in_tok + out_tok)
        elif phase == "handoff":
            self.handoff_tokens += (in_tok + out_tok)
        elif phase == "brain":
            self.brain_tokens += (in_tok + out_tok)
        elif phase == "verification":
            self.verification_tokens += (in_tok + out_tok)
        elif phase == "summary":
            self.summary_tokens += (in_tok + out_tok)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "routing_tokens": self.routing_tokens,
            "routing_recorded": self.routing_recorded,
            "muscle_tokens": self.muscle_tokens,
            "handoff_tokens": self.handoff_tokens,
            "brain_tokens": self.brain_tokens,
            "verification_tokens": self.verification_tokens,
            "summary_tokens": self.summary_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_billable_tokens": self.total_billable_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "reuse_count": self.reuse_count,
            "recomputation_count": self.recomputation_count,
            "verification_repeat_count": self.verification_repeat_count,
            "legitimate_repeat_count": self.legitimate_repeat_count,
            "source": self.source,
        }


# ============================================================
# V5 Logical Native History Interaction Groups
# ============================================================
@dataclass
class UserTurn:
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class AssistantTurn:
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    native_data: Any = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ToolInteractionGroup:
    """Atomic group binding an assistant tool_use turn to its matching tool results."""
    assistant_turn: AssistantTurn
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


FULL_BRAIN_CONTEXT_BUDGET: int = 16000  # Token budget for full brain history


# ============================================================
# Token estimation helper
# ============================================================
_TIKTOKEN_ENC = None
try:
    import importlib
    _tiktoken = importlib.import_module("tiktoken")
    _TIKTOKEN_ENC = _tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_ENC = None


def count_tokens(text: Any) -> int:
    """Accurate token counting via tiktoken if available, else chars/4 heuristic."""
    if not text:
        return 0
    s = text if isinstance(text, str) else str(text)
    if _TIKTOKEN_ENC is not None:
        try:
            return len(_TIKTOKEN_ENC.encode(s))
        except Exception:
            pass
    return max(1, len(s) // 4)


# Alias for backward compatibility
estimate_tokens = count_tokens


@dataclass
class CacheEntry:
    response: ProviderResponse
    timestamp: float
    context_free: bool = True


# ============================================================
# Response cache for Muscle Memory (exact-match, in-memory)
# ============================================================
class ResponseCache:
    """Zero-token response cache for Muscle Memory. Keyed on the normalized
    user message.

    CRITICAL SAFETY RULE: Context-dependent inputs ('yes', 'no', 'continue',
    'sure', 'do it', etc.) MUST NEVER be cached. Caching is strictly gated
    by an allow-list of verifiable context-free utterances.

    TTL-based eviction with a max-size cap. In-memory only — no stale
    cache issues across sessions, no disk I/O."""

    # Explicit allow-list of context-free utterances safe to cache
    ALLOW_LIST = {
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "thanks", "thank you", "thx", "ty", "cheers",
        "what is 2+2", "what's 2+2", "2+2", "2 + 2",
        "what is 2 + 2?", "what's 2 + 2?", "what is 2+2?", "what's 2+2?"
    }

    # Explicit deny-list of context-dependent utterances that MUST NEVER hit cache
    DENY_LIST = {
        "yes", "no", "yep", "nope", "continue", "go on", "proceed", "sure",
        "ok", "okay", "k", "do it", "confirm", "cancel", "stop", "more", "next"
    }

    def __init__(self, max_size: int = 256, ttl_seconds: float = 600.0):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._store: Dict[str, CacheEntry] = {}

    def _normalize(self, text: str) -> str:
        import re
        t = text.strip().lower()
        t = re.sub(r"[!?.,]+$", "", t).strip()
        t = re.sub(r"\s+", " ", t)
        synonyms = {
            "thank you": "thanks",
            "thx": "thanks",
            "ty": "thanks",
        }
        return synonyms.get(t, t)

    def is_cacheable(self, user_message: str) -> bool:
        norm = self._normalize(user_message)
        if norm in self.DENY_LIST:
            return False
        return norm in self.ALLOW_LIST

    def _key(self, user_message: str) -> str:
        return sha256(self._normalize(user_message).encode()).hexdigest()

    def get(self, user_message: str) -> Optional[ProviderResponse]:
        if not self.is_cacheable(user_message):
            return None
        k = self._key(user_message)
        entry = self._store.get(k)
        if entry is None:
            return None
        if time() - entry.timestamp > self.ttl:
            del self._store[k]
            return None
        return entry.response

    def lookup(self, user_message: str) -> Optional[ProviderResponse]:
        """Alias for get() per V6 router/muscle cache contract."""
        return self.get(user_message)

    def put(self, user_message: str, resp: ProviderResponse, context_free: bool = True) -> None:
        if not self.is_cacheable(user_message):
            return
        if len(self._store) >= self.max_size:
            # Evict oldest entry
            oldest_key = min(self._store, key=lambda k: self._store[k].timestamp)
            del self._store[oldest_key]
        self._store[self._key(user_message)] = CacheEntry(
            response=resp, timestamp=time(), context_free=context_free
        )


# ============================================================
# Conversation state (vendor-neutral)
# ============================================================
@dataclass
class ConversationState:
    # LLM-generated compressed synopsis of older history. Populated by
    # the summary compressor (runs every N full_brain turns or on mode
    # switch from full_brain → muscle_memory).
    running_summary: str = ""

    # Cheap, model-free rolling buffer shared by both modes. Updated
    # after every turn regardless of which path handled it, so a mode
    # switch doesn't lose recent context entirely.
    recent_plain_turns: List[tuple] = field(default_factory=list)  # [(role, text), ...]
    max_recent_plain_turns: int = 8  # entries, i.e. ~4 exchanges
    max_recent_tokens: int = 2000    # Token budget for recent plain turns

    # Count of consecutive full_brain turns since the last summary
    # compression. Reset to 0 after each compression.
    full_brain_turns_since_summary: int = 0

    def total_recent_tokens(self) -> int:
        return sum(estimate_tokens(t[1]) for t in self.recent_plain_turns)

    def add_plain_turn(self, role: str, text: str) -> None:
        if not text:
            return
        self.recent_plain_turns.append((role, text))
        # Trim by count
        if len(self.recent_plain_turns) > self.max_recent_plain_turns:
            self.recent_plain_turns = self.recent_plain_turns[-self.max_recent_plain_turns:]
        # Trim by token budget
        while len(self.recent_plain_turns) > 1 and self.total_recent_tokens() > self.max_recent_tokens:
            self.recent_plain_turns.pop(0)

    def recent_context_str(self) -> str:
        """Continuity context to seed Full Brain when it starts fresh
        after time spent in Muscle Memory."""
        parts = [f"{role}: {text}" for role, text in self.recent_plain_turns]
        combined = "\n".join(parts)
        if self.running_summary:
            combined = f"{self.running_summary}\n{combined}" if combined else self.running_summary
        return combined

    # Provider-native turn history, kept SEPARATELY per mode because
    # Muscle Memory intentionally does not see Full Brain's tool-call
    # history (that's the whole point of the cheap path).
    full_brain_history: List[Any] = field(default_factory=list)
    max_history_items: int = 40  # native items, not "turns" — trim by adapter
    full_brain_context_budget: int = 16000  # Token budget for full brain history

    def estimate_full_brain_tokens(self) -> int:
        total = estimate_tokens(self.running_summary)
        for item in self.full_brain_history:
            total += estimate_tokens(item)
        return total

    def _is_tool_result(self, item: Any) -> bool:
        if isinstance(item, dict):
            if item.get("role") == "tool":
                return True
            content = item.get("content")
            if isinstance(content, list) and any(
                isinstance(c, dict) and c.get("type") in ("tool_result", "tool_response") for c in content
            ):
                return True
        return False

    def _has_tool_calls(self, item: Any) -> bool:
        if isinstance(item, dict):
            if item.get("tool_calls"):
                return True
            content = item.get("content")
            if isinstance(content, list) and any(
                isinstance(c, dict) and c.get("type") == "tool_use" for c in content
            ):
                return True
        return False

    def _atomic_trim_head(self, min_items_to_drop: int = 1) -> None:
        """Drops oldest interaction units while preserving tool call -> tool result integrity."""
        dropped = 0
        while self.full_brain_history and (dropped < min_items_to_drop or len(self.full_brain_history) > 2):
            item = self.full_brain_history.pop(0)
            dropped += 1
            # If an assistant turn with tool calls was dropped, drop its immediate tool results as well
            if self._has_tool_calls(item):
                while self.full_brain_history and self._is_tool_result(self.full_brain_history[0]):
                    self.full_brain_history.pop(0)
                    dropped += 1
            if dropped >= min_items_to_drop:
                break

        # Sanity check: Head must never start with an orphaned tool result
        while self.full_brain_history and self._is_tool_result(self.full_brain_history[0]):
            self.full_brain_history.pop(0)

    def reset_full_brain_if_too_long(self):
        """Truncates history by count and tokens while preserving atomic interaction groups."""
        if len(self.full_brain_history) > self.max_history_items:
            self._atomic_trim_head(len(self.full_brain_history) - self.max_history_items)
        while len(self.full_brain_history) > 2 and self.estimate_full_brain_tokens() > self.full_brain_context_budget:
            self._atomic_trim_head(1)

    def apply_summary_truncation(self, new_summary: str, keep_tail_items: int = 4) -> None:
        """Replaces summarized history with the new summary and retains only the recent tail."""
        self.running_summary = new_summary
        if len(self.full_brain_history) > keep_tail_items:
            to_drop = len(self.full_brain_history) - keep_tail_items
            self._atomic_trim_head(to_drop)
        self.full_brain_turns_since_summary = 0


# ============================================================
# Usage tracking (vendor-neutral)
# ============================================================

# Approximate per-token costs in USD (input / output) — update these when
# pricing changes. Used only for the session report estimate; NOT for
# billing. These are illustrative defaults as of Sept 2026.
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # (input_cost_per_1k, output_cost_per_1k)
    "claude-sonnet-5":              (0.003, 0.015),
    "claude-haiku-4-5-20251001":    (0.0008, 0.004),
    "gpt-5.5":                      (0.003, 0.015),
    "gpt-5.4-mini":                 (0.0004, 0.0016),
}


@dataclass
class CachedResponseMetadata:
    """Historical telemetry for an item retrieved from ResponseCache."""
    original_input_tokens: int = 0
    original_output_tokens: int = 0
    original_request_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class TurnStats:
    mode: str
    provider: str
    model: str
    effort: str
    input_tokens: int
    output_tokens: int
    cache_read: int = 0
    cache_write: int = 0
    cached_response: bool = False  # True if served from ResponseCache (0 API tokens)
    cached_metadata: Optional[CachedResponseMetadata] = None
    latency_ms: float = 0.0
    # Field distinguishing mock/test-run rows from live-API rows.
    # "mock" is the default for MockProvider; real adapters pass their own
    # provider.name ("anthropic"/"openai") through source below.
    source: str = "mock"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "source": self.source,
            "mode": self.mode,
            "provider": self.provider,
            "model": self.model,
            "effort": self.effort,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read": self.cache_read,
            "cache_write": self.cache_write,
            "cached_response": self.cached_response,
            "cached_metadata": asdict(self.cached_metadata) if self.cached_metadata else None,
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass
class SessionStats:
    turns: List[TurnStats] = field(default_factory=list)
    task_usages: List[TaskUsage] = field(default_factory=list)

    def total_tokens_consumed(self) -> int:
        """Returns the total number of billable input + output tokens consumed across all recorded turns."""
        return sum(t.input_tokens + t.output_tokens for t in self.turns)

    def record_task_usage(self, usage: TaskUsage, log_file: Optional[str] = None) -> None:
        self.task_usages.append(usage)
        if log_file is None:
            log_file = os.environ.get("AGENT_TASK_LOG_FILE", os.path.join("logs", "task_usages.jsonl"))
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(usage.to_dict()) + "\n")
        except Exception:
            pass

    def add(self, mode: str, provider: str, resp: ProviderResponse,
            model: str = "", effort: str = "", cached_response: bool = False,
            latency_ms: float = 0.0, log_file: Optional[str] = None,
            source: Optional[str] = None):
        if log_file is None:
            log_file = os.environ.get("AGENT_LOG_FILE", os.path.join("logs", "session_turns.jsonl"))

        is_cache_hit = cached_response or (mode == "cache")
        if is_cache_hit:
            cached_meta = CachedResponseMetadata(
                original_input_tokens=resp.input_tokens,
                original_output_tokens=resp.output_tokens,
                original_request_id=getattr(resp, "request_id", None),
            )
            in_tok = 0
            out_tok = 0
            c_read = 0
            c_write = 0
        else:
            cached_meta = None
            in_tok = resp.input_tokens
            out_tok = resp.output_tokens
            c_read = resp.cache_read_tokens
            c_write = resp.cache_write_tokens

        turn = TurnStats(
            mode=mode,
            provider=provider,
            model=model,
            effort=effort,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read=c_read,
            cache_write=c_write,
            cached_response=is_cache_hit,
            cached_metadata=cached_meta,
            latency_ms=latency_ms,
            source=(source if source is not None
                    else ("live" if provider in {"anthropic", "openai"} else "mock")),
        )
        self.turns.append(turn)

        if log_file:
            try:
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(turn.to_dict()) + "\n")
            except Exception:
                pass

    def cached_input_token_ratio(self) -> float:
        """Fraction of total input tokens served from prompt cache."""
        total_in = self.total_input_tokens()
        total_read = sum(t.cache_read for t in self.turns)
        return (total_read / total_in) if total_in else 0.0

    def requests_with_cache_hit_ratio(self) -> float:
        """Fraction of non-zero API requests that had at least one cache read token."""
        api_turns = [t for t in self.turns if not t.cached_response]
        if not api_turns:
            return 0.0
        hits = sum(1 for t in api_turns if t.cache_read > 0)
        return hits / len(api_turns)

    def report(self) -> str:
        if not self.turns:
            return "No turns recorded."
        cache_turns = [t for t in self.turns if t.mode == "cache" or t.cached_response]
        muscle = [t for t in self.turns if t.mode == "muscle_memory" and not t.cached_response]
        brain = [t for t in self.turns if t.mode == "full_brain"]
        classifier = [t for t in self.turns if t.mode == "classify"]

        def avg(ts):
            return sum(t.input_tokens + t.output_tokens for t in ts) / len(ts) if ts else 0

        total_billable = sum(t.input_tokens for t in self.turns)
        total_read = sum(t.cache_read for t in self.turns)
        hit_rate = (total_read / total_billable) if total_billable else 0.0

        # Cost estimation
        total_cost = 0.0
        for t in self.turns:
            pricing = MODEL_PRICING.get(t.model, (0.003, 0.015))
            total_cost += (t.input_tokens * pricing[0] + t.output_tokens * pricing[1]) / 1000

        non_classify = [t for t in self.turns if t.mode not in {"classify", "summary"}]
        total_non_classify = len(non_classify)
        total_user_tasks = len(self.task_usages) if self.task_usages else total_non_classify

        lines = [
            "========== Session Report ==========",
            f"User Tasks             : {total_user_tasks}",
            f"Total turns (user)     : {total_non_classify}",
            f"  Muscle Memory        : {len(muscle)} ({len(muscle)/total_non_classify:.0%})" if total_non_classify else "",
            f"  Full Brain           : {len(brain)} ({len(brain)/total_non_classify:.0%})" if total_non_classify else "",
            f"  Cached (0-token)     : {len(cache_turns)}",
            f"Classifier calls       : {len(classifier)} (skipped by pre-filter: {total_non_classify - len(classifier)})" if total_non_classify else "",
            f"Avg tokens (Muscle)    : {avg(muscle):.0f}",
            f"Avg tokens (Brain)     : {avg(brain):.0f}",
            f"Cached Input Token %   : {self.cached_input_token_ratio():.1%}",
            f"Request Cache Hit %    : {self.requests_with_cache_hit_ratio():.1%}",
            f"Prompt cache hit %     : {hit_rate:.1%}",
            f"Est. session cost      : ${total_cost:.4f}",
            "====================================",
        ]
        return "\n".join(line for line in lines if line)

    def total_input_tokens(self) -> int:
        return sum(t.input_tokens for t in self.turns)

    def total_output_tokens(self) -> int:
        return sum(t.output_tokens for t in self.turns)

    def total_tokens_consumed(self) -> int:
        return sum(t.input_tokens + t.output_tokens for t in self.turns)

    def prompt_cache_hit_rate(self) -> float:
        total_in = self.total_input_tokens()
        total_read = sum(t.cache_read for t in self.turns)
        return (total_read / total_in) if total_in else 0.0

    def estimated_cost_usd(self) -> float:
        total_cost = 0.0
        for t in self.turns:
            pricing = MODEL_PRICING.get(t.model, (0.003, 0.015))
            total_cost += (t.input_tokens * pricing[0] + t.output_tokens * pricing[1]) / 1000
        return total_cost


# ============================================================
# The interface every provider adapter implements
# ============================================================
class ModelProvider(ABC):
    """One instance per provider. Uses a SINGLE model ID — classifier,
    Muscle Memory, and Full Brain are differentiated only by effort
    level, max tokens, tool availability, and context size."""

    name: str = "base"
    model: str = ""  # the single model ID used for everything

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Declared provider capabilities per V6 Section AW."""
        return ProviderCapabilities(
            supports_reasoning_effort=False,
            supports_prompt_cache=False,
            supports_structured_output=True,
            supports_native_tool_history=True,
            supports_reasoning_usage=False,
            supports_tool_calls=True,
        )

    @abstractmethod
    def classify(self, user_message: str, recent_summary: str) -> str:
        """Return exactly 'muscle_memory' or 'full_brain'.
        Uses the SAME model with effort=low and minimal context."""

    def route_classify(self, user_message: str, recent_summary: str) -> Dict[str, Any]:
        """Structured multi-axis classification returning a JSON-compatible dict:
        pathway (muscle|full_brain), confidence, context_required, tools_required,
        freshness_required, novelty (0.0-1.0), risk (low|medium|high), reasoning.
        Uses the SAME model with effort=low. Default implementation delegates to classify()."""
        classification = self.classify(user_message, recent_summary)
        is_muscle = classification == "muscle_memory"
        return {
            "pathway": "muscle" if is_muscle else "full_brain",
            "confidence": 0.85 if is_muscle else 0.90,
            "context_required": False,
            "tools_required": not is_muscle,
            "freshness_required": False,
            "novelty": 0.3 if is_muscle else 0.7,
            "risk": "low" if is_muscle else "medium",
            "reasoning": f"Fallback route classified as {classification}.",
        }

    @abstractmethod
    def muscle_memory_reply(self, user_message: str) -> ProviderResponse:
        """Same model, effort=low, max_tokens=120, no tools, bare message only.
        No context injected — that's the whole point of the cheap path."""

    @abstractmethod
    def full_brain_step(
        self,
        history: List[Any],
        tools: List[ToolDef],
        effort: str,
    ) -> ProviderResponse:
        """One step of the agentic loop. Same model, configurable effort,
        full tools, full history. Caller loops while stop_reason == 'tool_use'."""

    def summarize_history_step(self, history_text: str) -> ProviderResponse:
        """Compress a block of conversation history returning actual ProviderResponse token metrics."""
        summary = self.summarize_history(history_text)
        return ProviderResponse(
            text=summary,
            tool_calls=[],
            stop_reason="end_turn",
            raw=None,
            input_tokens=count_tokens(history_text),
            output_tokens=count_tokens(summary),
        )

    @abstractmethod
    def summarize_history(self, history_text: str) -> str:
        """Compress a block of conversation history into 3-5 sentences.
        Uses the same model with effort=low. Called by the orchestrator
        every N full_brain turns or on mode switch."""

    @abstractmethod
    def append_assistant_turn(self, history: List[Any], resp: ProviderResponse) -> None:
        """Mutates history in place, adding this turn's assistant output
        in the provider's native shape."""

    @abstractmethod
    def append_tool_results(
        self, history: List[Any], tool_calls: List[ToolCall], results: Dict[str, str]
    ) -> None:
        """Mutates history in place, adding tool_call -> result pairs in
        the provider's native shape (results keyed by ToolCall.id)."""

    @abstractmethod
    def append_user_turn(self, history: List[Any], text: str) -> None:
        """Mutates history in place with a plain user message."""
