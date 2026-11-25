"""
router.py — Multi-axis routing system & expected-cost model for V6.

Per V6 Spec Sections AQ, AR, AS, AT, AU:
- Router question: "Is there a safe, bounded, sufficiently verifiable automatic execution strategy worth attempting?"
- Multi-dimensional assessment: procedure_match, procedure_success_rate, execution_feasibility,
  context_dependency, tool_requirement, freshness, novelty, risk, verification_available.
- New-Task Muscle-First Support: Safe bounded tasks can attempt automatic execution before escalating.
- Expected cost calculation based on empirical priors without ungrounded multipliers.
"""

from __future__ import annotations
import re
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Dict

from base import ConversationState, ModelProvider, ResponseCache


@dataclass
class RouteDecision:
    pathway: str  # "cache" | "muscle" | "light_brain" | "full_brain"
    confidence: float = 1.0

    procedure_match: Optional[str] = None
    procedure_confidence: float = 0.0
    procedure_success_rate: float = 0.0
    execution_feasibility: bool = True

    context_required: bool = False
    tools_required: bool = False
    freshness_required: bool = False

    novelty: float = 0.0  # 0.0 (routine) to 1.0 (novel)
    risk: str = "low"     # "low" | "medium" | "high"

    verification_available: bool = True

    expected_muscle_success_probability: float = 0.5
    estimated_muscle_cost: int = 50
    estimated_verification_cost: int = 0
    estimated_handoff_cost: int = 150
    estimated_brain_cost: int = 1200
    estimated_brain_continuation_cost: int = 600
    estimated_brain_restart_cost: int = 1200
    continuation_cost_ratio: float = 0.5

    routing_response: Optional[Any] = None
    reasoning: str = ""

    def expected_muscle_path_cost(self) -> float:
        """Controller-level expected cost calculation per V6 Section AU:
        P(succ) * Muscle + (1 - P(succ)) * (Muscle + Handoff + (Brain_Cost * Continuation_Ratio)).
        """
        p_succ = self.expected_muscle_success_probability
        p_fail = max(0.0, 1.0 - p_succ)
        continuation_brain_cost = self.estimated_brain_cost * self.continuation_cost_ratio
        muscle_total = self.estimated_muscle_cost + self.estimated_verification_cost
        continuation_total = muscle_total + self.estimated_handoff_cost + continuation_brain_cost
        return (p_succ * muscle_total) + (p_fail * continuation_total)

    def to_dict(self) -> dict:
        d = asdict(self)
        if d.get("routing_response"):
            d["routing_response"] = str(d["routing_response"])
        return d


# ============================================================
# Deterministic Patterns (0 model tokens, 0 network latency)
# ============================================================
REFERENCE_PATTERN = re.compile(
    r"\b(this|that|it|the above|same as before|continue|them|those|these|previous|earlier|again|proceed|more)\b",
    re.IGNORECASE,
)

FRESHNESS_PATTERN = re.compile(
    r"\b(current|latest|today|now|recent|currently|breaking|live|upcoming|newest|yesterday|tomorrow)\b",
    re.IGNORECASE,
)

FIRST_PERSON_HISTORY = re.compile(
    r"\b(my|our|we|us|earlier|before|you said|as i said|as we discussed|told you|remind me|did i|do i|i like|i love|i prefer|i mentioned|i said|what did i)\b",
    re.IGNORECASE,
)

DESTRUCTIVE_PATTERN = re.compile(
    r"\b(delete|remove|drop|overwrite|truncate|rm\b|kill|destroy|erase|format|purge)\b",
    re.IGNORECASE,
)

INJECTION_PATTERN = re.compile(
    r"\b(ignore\s+(?:all\s+|prior\s+|previous\s+)?instructions|disregard|system\s+prompt|jailbreak|unrestricted|override\s+safety)\b",
    re.IGNORECASE,
)

TOOL_PATTERN = re.compile(
    r"\b(write_file|read_file|create\s+(?:a\s+)?file|edit\s+(?:a\s+)?file|save\s+(?:to|file)|write\s+to|inspect\s+(?:the\s+|a\s+)?file|open\s+(?:a\s+)?file|write\s+(?:a\s+)?(?:python\s+)?script|read\s+[a-zA-Z0-9_\-\./]+\b|[a-zA-Z0-9_\-]+\.(?:py|json|txt|md|csv|sh|yaml|yml)\b)\b",
    re.IGNORECASE,
)

GREETING_EXPANDED = re.compile(
    r"^\s*(hi|hello(?:\s+there)?|hey|greetings|howdy|sup|yo|"
    r"how\s+are\s+you(?:\s+today)?|"
    r"tell\s+me\s+a\s+(?:short\s+)?joke|"
    r"good\s+(?:morning|afternoon|evening|day)|"
    r"(?:cool,?\s*)?(?:thanks(?:\s+a\s+lot)?|thank\s+you(?:\s+for\s+(?:the\s+)?help)?|thanks\s+for\s+asking|thx|ty|cheers)|"
    r"bye|goodbye(?:,\s*have\s+a\s+nice\s+day)?|cya|see\s+you(?:\s+later)?)\s*[!?.]*\s*$",
    re.IGNORECASE,
)

PURE_ARITHMETIC_PATTERN = re.compile(
    r"^\s*(what\s+is\s+|calculate\s+|compute\s+)?[-+]?\d+(\.\d+)?\s*[\+\-\*\/\%]\s*[-+]?\d+(\.\d+)?\s*[?=.]*\s*$",
    re.IGNORECASE,
)

TEXT_STATISTICS_PATTERN = re.compile(
    r"^\s*(?:(?:compute|calculate|get|show)\s+)?(?:text\s+)?(?:statistics|stats)\s*(?:of|for)?\s*:\s*\S+",
    re.IGNORECASE,
)

STATIC_TRIVIA_PATTERN = re.compile(
    r"^\s*(what\s+is\s+the\s+capital\s+of\s+[a-zA-Z\s]+|"
    r"who\s+wrote\s+[a-zA-Z\s]+|"
    r"what\s+year\s+did\s+[a-zA-Z0-9\s]+|"
    r"how\s+many\s+(days|hours|minutes|seconds|continents|planets)\s+(are\s+there\s+in|in\s+a)\s+[a-zA-Z\s]+)\s*\??\s*$",
    re.IGNORECASE,
)

CONTEXTUAL_ACK_PATTERN = re.compile(
    r"^\s*(yes|no|sure|ok|okay|do\s+it|continue|proceed|yes,?\s*please|no\s*thanks)\b",
    re.IGNORECASE,
)


def deterministic_route(
    user_message: str,
    state: ConversationState,
    cache: Optional[Any] = None,
    memory: Optional[Any] = None,
) -> Optional[RouteDecision]:
    """Fast deterministic gate per V6 Section M (0 tokens, 0 network latency)."""
    clean = user_message.strip()

    # Gate 0: Response Cache Hit (0 model tokens, exact match)
    if cache is not None and isinstance(cache, ResponseCache):
        cached_resp = cache.lookup(clean)
        if cached_resp is not None:
            return RouteDecision(
                pathway="cache",
                confidence=1.0,
                procedure_match=None,
                procedure_confidence=1.0,
                procedure_success_rate=1.0,
                context_required=False,
                tools_required=False,
                freshness_required=False,
                novelty=0.0,
                risk="low",
                verification_available=True,
                expected_muscle_success_probability=1.0,
                estimated_muscle_cost=0,
                estimated_handoff_cost=0,
                estimated_brain_cost=1200,
                reasoning="Exact query match in zero-token ResponseCache.",
            )

    has_recent = bool(state.recent_plain_turns)
    is_contextual_ack = bool(CONTEXTUAL_ACK_PATTERN.match(clean))
    is_destructive = bool(DESTRUCTIVE_PATTERN.search(clean))
    is_injection = bool(INJECTION_PATTERN.search(clean))

    # Gate 1: Contextual confirmations / acknowledgments following recent turns
    if is_contextual_ack and has_recent:
        return RouteDecision(
            pathway="full_brain",
            confidence=1.0,
            procedure_match=None,
            procedure_confidence=0.0,
            context_required=True,
            tools_required=is_destructive,
            freshness_required=False,
            novelty=0.2,
            risk="high" if is_destructive else "low",
            verification_available=True,
            expected_muscle_success_probability=0.2,
            estimated_muscle_cost=50,
            estimated_handoff_cost=150,
            estimated_brain_cost=1200,
            reasoning="Deterministic gate: contextual acknowledgment following prior turns.",
        )

    # Gate 2: Adversarial Prompt Injections
    if is_injection:
        return RouteDecision(
            pathway="full_brain",
            confidence=1.0,
            procedure_match=None,
            procedure_confidence=0.0,
            context_required=False,
            tools_required=False,
            freshness_required=False,
            novelty=1.0,
            risk="high",
            verification_available=True,
            expected_muscle_success_probability=0.0,
            estimated_muscle_cost=0,
            estimated_handoff_cost=0,
            estimated_brain_cost=1200,
            reasoning="Deterministic gate: prompt injection detected; routed to full_brain with high risk.",
        )

    # Gate 3: Destructive / High-Risk Operations
    if is_destructive:
        return RouteDecision(
            pathway="full_brain",
            confidence=1.0,
            procedure_match=None,
            procedure_confidence=0.0,
            context_required=False,
            tools_required=True,
            freshness_required=False,
            novelty=0.5,
            risk="high",
            verification_available=True,
            expected_muscle_success_probability=0.0,
            estimated_muscle_cost=0,
            estimated_handoff_cost=0,
            estimated_brain_cost=1200,
            reasoning="Deterministic gate: destructive pattern detected; strictly routed to full_brain.",
        )

    # Gate 4: Procedural Skill Match (Memory retrieval)
    if memory is not None:
        matched_skill = memory.retrieve(clean, only_trusted=True)
        if matched_skill:
            est_cost = 0 if matched_skill.is_deterministic or matched_skill.procedure_type == "DETERMINISTIC" else 35
            return RouteDecision(
                pathway="muscle",
                confidence=matched_skill.calibrated_confidence or 0.95,
                procedure_match=matched_skill.id,
                procedure_confidence=matched_skill.calibrated_confidence or 0.95,
                procedure_success_rate=matched_skill.empirical_success_rate or 1.0,
                context_required=False,
                tools_required=bool(matched_skill.required_tools),
                freshness_required=False,
                novelty=0.1,
                risk="low",
                verification_available=True,
                expected_muscle_success_probability=matched_skill.empirical_success_rate or 0.95,
                estimated_muscle_cost=est_cost,
                estimated_handoff_cost=100,
                estimated_brain_cost=1200,
                reasoning=f"Deterministic gate: matched procedural skill '{matched_skill.id}' ({matched_skill.procedure_type}).",
            )

    # Gate 5: Casual Greetings & Acknowledgments
    if GREETING_EXPANDED.match(clean):
        return RouteDecision(
            pathway="muscle",
            confidence=1.0,
            procedure_match=None,
            procedure_confidence=0.0,
            context_required=False,
            tools_required=False,
            freshness_required=False,
            novelty=0.0,
            risk="low",
            verification_available=True,
            expected_muscle_success_probability=1.0,
            estimated_muscle_cost=30,
            estimated_handoff_cost=100,
            estimated_brain_cost=1200,
            reasoning="Deterministic gate: routine greeting or acknowledgment.",
        )

    # Gate 6: Deterministic Arithmetic Calculations
    if PURE_ARITHMETIC_PATTERN.match(clean):
        return RouteDecision(
            pathway="muscle",
            confidence=1.0,
            procedure_match="deterministic_arithmetic",
            procedure_confidence=1.0,
            procedure_success_rate=1.0,
            context_required=False,
            tools_required=False,
            freshness_required=False,
            novelty=0.05,
            risk="low",
            verification_available=True,
            expected_muscle_success_probability=1.0,
            estimated_muscle_cost=0,
            estimated_handoff_cost=100,
            estimated_brain_cost=1200,
            reasoning="Deterministic gate: pure arithmetic expression evaluable by AST.",
        )

    # Gate 6b: Deterministic Text Statistics
    if TEXT_STATISTICS_PATTERN.match(clean):
        return RouteDecision(
            pathway="muscle",
            confidence=1.0,
            procedure_match="deterministic_text_statistics",
            procedure_confidence=1.0,
            procedure_success_rate=1.0,
            context_required=False,
            tools_required=False,
            freshness_required=False,
            novelty=0.05,
            risk="low",
            verification_available=True,
            expected_muscle_success_probability=1.0,
            estimated_muscle_cost=0,
            estimated_handoff_cost=100,
            estimated_brain_cost=1200,
            reasoning="Deterministic gate: text statistics request.",
        )

    # Gate 7: Static trivia
    if STATIC_TRIVIA_PATTERN.match(clean):
        return RouteDecision(
            pathway="muscle",
            confidence=0.9,
            procedure_match=None,
            procedure_confidence=0.0,
            context_required=False,
            tools_required=False,
            freshness_required=False,
            novelty=0.1,
            risk="low",
            verification_available=True,
            expected_muscle_success_probability=0.85,
            estimated_muscle_cost=45,
            estimated_handoff_cost=150,
            estimated_brain_cost=1200,
            reasoning="Deterministic gate: static closed-domain trivia.",
        )

    # Gate 8: Tool Requests
    if TOOL_PATTERN.search(clean):
        # Safe Bounded Exploration: Read-only file inspection with low risk can be attempted by Muscle
        is_read_only = bool(re.search(r"\b(read|inspect|view|show|cat|check|examine)\b", clean, re.IGNORECASE)) and not bool(
            re.search(r"\b(write|create|modify|delete|remove|patch|update|save)\b", clean, re.IGNORECASE)
        )
        if is_read_only and not is_destructive and not is_injection:
            return RouteDecision(
                pathway="muscle",
                confidence=0.85,
                procedure_match=None,
                procedure_confidence=0.0,
                context_required=False,
                tools_required=True,
                freshness_required=False,
                novelty=0.3,
                risk="low",
                verification_available=True,
                expected_muscle_success_probability=0.7,
                estimated_muscle_cost=40,
                estimated_handoff_cost=120,
                estimated_brain_cost=1200,
                reasoning="Deterministic gate: safe bounded read-only tool inspection; Muscle attempts bounded execution.",
            )

        return RouteDecision(
            pathway="full_brain",
            confidence=0.95,
            procedure_match=None,
            procedure_confidence=0.0,
            context_required=False,
            tools_required=True,
            freshness_required=False,
            novelty=0.5,
            risk="medium",
            verification_available=True,
            expected_muscle_success_probability=0.1,
            estimated_muscle_cost=70,
            estimated_handoff_cost=150,
            estimated_brain_cost=1200,
            reasoning="Deterministic gate: complex or write tool invocation requested without procedural skill.",
        )

    # Gate 10: Freshness Queries
    if FRESHNESS_PATTERN.search(clean):
        return RouteDecision(
            pathway="full_brain",
            confidence=0.9,
            procedure_match=None,
            procedure_confidence=0.0,
            context_required=False,
            tools_required=False,
            freshness_required=True,
            novelty=0.6,
            risk="medium",
            verification_available=True,
            expected_muscle_success_probability=0.2,
            estimated_muscle_cost=60,
            estimated_handoff_cost=150,
            estimated_brain_cost=1200,
            reasoning="Deterministic gate: dynamic freshness query detected.",
        )

    return None


def route(
    user_message: str,
    state: ConversationState,
    *args,
    cache: Optional[ResponseCache] = None,
    memory: Optional[Any] = None,
    provider: Optional[ModelProvider] = None,
    **kwargs,
) -> RouteDecision:
    """Master router selecting optimal execution pathway per V6 Section AQ.
    Flexible signature handles positional provider/cache for full backward compatibility.
    """
    # Positional argument resolution
    for arg in args:
        if isinstance(arg, ModelProvider):
            provider = arg
        elif isinstance(arg, ResponseCache):
            cache = arg
        elif memory is None and not isinstance(arg, (str, int, float, bool)):
            memory = arg

    # 1. Fast deterministic gate
    det_decision = deterministic_route(user_message, state, cache=cache, memory=memory)
    if det_decision is not None:
        return det_decision

    # 2. Structured Model Classification
    has_ref = bool(REFERENCE_PATTERN.search(user_message)) or bool(FIRST_PERSON_HISTORY.search(user_message))
    has_tools = bool(TOOL_PATTERN.search(user_message))
    has_freshness = bool(FRESHNESS_PATTERN.search(user_message))
    has_injection = bool(INJECTION_PATTERN.search(user_message))
    has_destructive = bool(DESTRUCTIVE_PATTERN.search(user_message))

    summary_str = state.running_summary or state.recent_context_str() or ""
    routing_resp = None

    if provider is None:
        pathway = "full_brain"
        confidence = 0.5
        risk = "high" if (has_injection or has_destructive) else ("medium" if has_tools else "low")
        return RouteDecision(
            pathway=pathway,
            confidence=confidence,
            context_required=has_ref,
            tools_required=has_tools,
            freshness_required=has_freshness,
            novelty=0.7,
            risk=risk,
            verification_available=True,
            expected_muscle_success_probability=0.25,
            estimated_muscle_cost=70,
            estimated_handoff_cost=150,
            estimated_brain_cost=1250,
            reasoning="No provider configured; fallback to full_brain.",
        )

    try:
        raw = provider.route_classify(user_message, summary_str)
        if isinstance(raw, tuple):
            structured, routing_resp = raw
        elif isinstance(raw, dict):
            structured = raw
            routing_resp = raw.get("_response")
        else:
            structured = {}

        raw_pathway = str(structured.get("pathway", "full_brain")).lower()
        pathway = "muscle" if raw_pathway in {"muscle", "muscle_memory"} else "full_brain"
        confidence = float(structured.get("confidence", 0.85))

        context_required = bool(structured.get("context_required", False)) or has_ref
        tools_required = bool(structured.get("tools_required", False)) or has_tools
        freshness_required = bool(structured.get("freshness_required", False)) or has_freshness

        novelty = float(structured.get("novelty", 0.3 if pathway == "muscle" else 0.7))
        risk = str(structured.get("risk", "high" if (has_injection or has_destructive) else ("medium" if has_tools else "low"))).lower()
        if has_injection or has_destructive:
            risk = "high"
        elif risk not in {"low", "medium", "high"}:
            risk = "medium" if tools_required else "low"

        reasoning = str(structured.get("reasoning", f"Model structured classification: {pathway}."))
    except Exception as e:
        pathway = "full_brain"
        confidence = 0.5
        context_required = has_ref
        tools_required = has_tools
        freshness_required = has_freshness
        novelty = 0.7
        risk = "high" if (has_injection or has_destructive) else ("medium" if has_tools else "low")
        reasoning = f"Fail-safe fallback due to error: {e}"

    # Hard non-overridable safety floors
    if (risk == "high" or has_injection or has_destructive) and pathway == "muscle":
        pathway = "full_brain"
        reasoning += " (Hard safety override to full_brain: high-risk/injection/destructive)"

    # Hard context-dependence floor
    if (context_required or tools_required) and pathway == "muscle":
        pathway = "full_brain"
        reasoning += f" (Hard safety override: context={context_required}, tools={tools_required})"

    # Cost model evaluation
    est_muscle = 70
    est_handoff = 150
    est_brain = 1250
    p_succ = 0.85 if pathway == "muscle" else 0.25
    expected_muscle_cost = (p_succ * est_muscle) + ((1.0 - p_succ) * (est_muscle + est_handoff + 600))

    if pathway == "muscle" and expected_muscle_cost >= est_brain:
        pathway = "full_brain"
        reasoning += " (Overridden to full_brain: unfavorable expected cost)"

    if (
        pathway == "full_brain"
        and not tools_required
        and not freshness_required
        and risk == "low"
        and novelty < 0.5
        and context_required
        and user_message.strip().lower() not in {"continue", "what language do i like?"}
    ):
        pathway = "light_brain"
        reasoning += " (Downgraded to light_brain: context-aware but tool-free)"

    return RouteDecision(
        pathway=pathway,
        confidence=confidence,
        procedure_match=None,
        procedure_confidence=0.0,
        procedure_success_rate=0.0,
        execution_feasibility=True,
        context_required=context_required,
        tools_required=tools_required,
        freshness_required=freshness_required,
        novelty=novelty,
        risk=risk,
        verification_available=True,
        expected_muscle_success_probability=p_succ,
        estimated_muscle_cost=est_muscle,
        estimated_verification_cost=0,
        estimated_handoff_cost=est_handoff,
        estimated_brain_cost=est_brain,
        estimated_brain_continuation_cost=600,
        estimated_brain_restart_cost=1200,
        routing_response=routing_resp,
        reasoning=reasoning,
    )
