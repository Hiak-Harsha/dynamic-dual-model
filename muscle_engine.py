"""
muscle_engine.py — 4-tier Muscle execution engine.

Per V6 Spec Sections I, J, Z, AA, AB:
- class MuscleExecutor:
    Tier 0: Response cache (zero tokens, exact match)
    Tier 1: Deterministic procedures (zero model tokens, pure code)
    Tier 2: Validated procedural skills:
        DETERMINISTIC: run Python procedure without LLM.
        TOOL_SEQUENCE: run explicitly allowed tools according to stored procedure under SkillToolPolicy.
        HYBRID: deterministic computation + bounded model adaptation.
        MODEL_ADAPTIVE: use LLM only when procedure execution requires adaptation.
    Tier 3: Bounded adaptive procedural execution (strict budget, uncertainty escalation)
- Terminal States: COMPLETED_VERIFIED, COMPLETED_UNVERIFIED, PARTIAL, HANDOFF_REQUIRED, BLOCKED, FAILED.
  Only COMPLETED_VERIFIED may directly complete the task.
- Explicit budget enforcement passed from controller.
"""

from __future__ import annotations
import ast
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List

from base import ConversationState, ModelProvider, ProviderResponse, ResponseCache, ExecutionBudget
from execution_state import ExecutionState, EvidenceLevel, COMPLETED_VERIFIED, TaskStatus
from procedural_memory import Skill, ProceduralMemory, ProcedureType
from router import RouteDecision
from tools import execute_tool, execute_tool_with_record, SkillToolPolicy, _resolve_in_sandbox
from procedure_executor import Procedure, ProcedureStep, ProcedureExecutor

# Terminal States per V6 Section Z
COMPLETED_UNVERIFIED = "completed_unverified"
PARTIAL = "partial"
HANDOFF_REQUIRED = "handoff_required"
BLOCKED = "blocked"
FAILED = "failed"


@dataclass
class MuscleResult:
    status: str  # COMPLETED_VERIFIED | COMPLETED_UNVERIFIED | PARTIAL | HANDOFF_REQUIRED | BLOCKED | FAILED
    confidence: float = 1.0
    text: str = ""
    partial_result: str = ""
    reason: str = ""
    execution_state: ExecutionState = field(default_factory=ExecutionState)
    skill_used: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def requires_handoff(self) -> bool:
        return self.status in {HANDOFF_REQUIRED, PARTIAL, "handoff", "handoff_required"}

    @property
    def is_completed(self) -> bool:
        return self.status in {COMPLETED_VERIFIED, COMPLETED_UNVERIFIED, "completed"}


# ============================================================
# Tier 1: Deterministic Execution Functions (0 LLM Tokens)
# ============================================================
def _run_deterministic_json(user_message: str, state: ExecutionState) -> Tuple[bool, str]:
    state.add_completed_step("Attempted deterministic JSON validation")
    clean = user_message.strip()

    block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
    candidate = block_match.group(1).strip() if block_match else clean

    if not (candidate.startswith("{") and candidate.endswith("}")) and not (
        candidate.startswith("[") and candidate.endswith("]")
    ):
        raw_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", candidate)
        if raw_match:
            candidate = raw_match.group(1).strip()

    try:
        parsed = json.loads(candidate)
        state.add_observation(f"JSON parsed successfully: type={type(parsed).__name__}")
        state.add_verified_fact(
            f"JSON parsed successfully ({type(parsed).__name__} with {len(parsed)} elements)",
            source="deterministic",
            method="json.loads",
        )
        report = (
            f"Valid JSON payload ({type(parsed).__name__}).\n"
            f"Element count: {len(parsed)}\n"
            f"Keys / Schema: {list(parsed.keys()) if isinstance(parsed, dict) else 'Array item count: ' + str(len(parsed))}"
        )
        return True, report
    except Exception as e:
        state.add_observation(f"JSON parsing failed: {e}")
        state.add_verified_fact(
            f"JSON validation completed: Input contains invalid JSON ({e})",
            source="deterministic",
            method="json.loads",
        )
        return True, f"Invalid JSON syntax detected: {e}"


def _run_deterministic_text_stats(user_message: str, state: ExecutionState) -> Tuple[bool, str]:
    state.add_completed_step("Attempted deterministic text statistics")
    clean = user_message.strip()
    words = clean.split()
    lines = clean.splitlines()
    chars = len(clean)

    report = (
        f"Text Statistics:\n"
        f"- Lines: {len(lines)}\n"
        f"- Words: {len(words)}\n"
        f"- Characters: {chars}"
    )
    state.add_observation(f"Text stats calculated: {len(words)} words, {len(lines)} lines")
    state.add_verified_fact(f"Text statistics computed: {len(words)} words, {chars} characters", source="deterministic", method="str_methods")
    return True, report


def _run_deterministic_arithmetic(user_message: str, state: ExecutionState) -> Tuple[bool, str]:
    state.add_completed_step("Attempted safe AST arithmetic evaluation: deterministic_arithmetic")
    state.metadata["is_deterministic"] = True
    clean = user_message.strip()

    raw = re.sub(r"^(what\s+is|calculate|compute)\s+", "", clean, flags=re.IGNORECASE).rstrip("?=.")
    expr = raw.strip()

    if len(expr) > 200:
        return False, "Expression exceeds maximum length cap (200 characters)"

    try:
        node = ast.parse(expr, mode="eval")
    except Exception as e:
        return False, f"Could not evaluate expression: {e}"

    node_count = sum(1 for _ in ast.walk(node))
    if node_count > 50:
        return False, f"Expression complexity exceeds maximum AST node limit ({node_count}/50)"

    allowed_operators = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    MAX_EXPONENT = 1000

    def _eval_node(n):
        if isinstance(n, ast.Expression):
            return _eval_node(n.body)
        elif isinstance(n, ast.Constant):
            if isinstance(n.value, bool):
                raise ValueError("Boolean literals are not allowed in arithmetic expressions")
            if isinstance(n.value, (int, float)):
                if abs(n.value) > 1e100:
                    raise ValueError("Numeric literal magnitude exceeds safe limit (1e100)")
                return n.value
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            val = _eval_node(n.operand)
            return +val if isinstance(n.op, ast.UAdd) else -val
        elif isinstance(n, ast.BinOp) and isinstance(n.op, allowed_operators):
            left = _eval_node(n.left)
            right = _eval_node(n.right)
            if isinstance(n.op, ast.Add):
                res = left + right
            elif isinstance(n.op, ast.Sub):
                res = left - right
            elif isinstance(n.op, ast.Mult):
                res = left * right
            elif isinstance(n.op, ast.Div):
                if right == 0:
                    raise ZeroDivisionError("division by zero")
                res = left / right
            elif isinstance(n.op, ast.FloorDiv):
                if right == 0:
                    raise ZeroDivisionError("integer division by zero")
                res = left // right
            elif isinstance(n.op, ast.Mod):
                if right == 0:
                    raise ZeroDivisionError("modulo by zero")
                res = left % right
            elif isinstance(n.op, ast.Pow):
                if abs(right) > MAX_EXPONENT:
                    raise ValueError(f"Exponent {right} exceeds MAX_EXPONENT cap ({MAX_EXPONENT})")
                if abs(left) > 1e10 and right > 10:
                    raise ValueError("Exponential operation exceeds safe resource limits")
                res = left ** right

            if abs(res) > 1e100:
                raise ValueError("Intermediate result exceeds safe numeric bounds (1e100)")
            return res

        raise ValueError("Not a purely numeric arithmetic expression")

    try:
        val = _eval_node(node)
        formatted = int(val) if isinstance(val, float) and val.is_integer() else val
        state.add_observation(f"AST arithmetic evaluated: {expr} = {formatted}")
        state.add_verified_fact(f"Mathematical evaluation: {expr} = {formatted}", source="deterministic", method="ast_eval")
        return True, f"Result: {formatted}"
    except Exception as e:
        state.add_observation(f"Arithmetic evaluation failed: {e}")
        return False, f"Could not evaluate: Not a purely numeric arithmetic expression ({e})"


_run_deterministic_ast_arithmetic = _run_deterministic_arithmetic


def _run_tool_sequence_procedure(
    skill: Skill,
    user_message: str,
    state: ExecutionState,
) -> Tuple[bool, str]:
    """Executes a bounded TOOL_SEQUENCE skill under SkillToolPolicy per V6 Section J & AB."""
    policy = SkillToolPolicy(
        allowed_tools=skill.required_tools or ["read_file"],
        max_calls=2,
        allow_write=False,
    )

    state.add_completed_step(f"Initiated tool sequence procedure: {skill.id}")

    path_match = re.search(r"['\"]?([\w\-\.\/]+\.(?:py|json|txt|md|csv))['\"]?", user_message)
    if not path_match:
        return False, "Could not extract valid file path argument for tool sequence"

    path_arg = path_match.group(1).strip("'\"")
    state.current_step = f"Executing read_file on '{path_arg}'"

    can_reuse, cached_out, rep_type = state.check_tool_reuse("read_file", {"path": path_arg})
    if can_reuse and cached_out:
        state.recomputation_count += 1
        state.recomputation_types["recomputation"] += 1
        state.add_observation(f"Reused verified fresh tool output for '{path_arg}'")
        state.set_frontier(current="read_file_cached", next_action="complete", remaining=[])
        return True, cached_out

    result, record = execute_tool_with_record(
        "read_file",
        {"path": path_arg},
        call_id=f"muscle_call_{skill.id}",
        policy=policy,
        current_calls=len(state.tool_actions),
    )

    state.record_tool_action({"name": "read_file", "arguments": {"path": path_arg}, "id": record.call_id})
    state.record_tool_result(record.to_dict())

    if record.success:
        state.record_state_freshness(path_arg, result)
        state.tool_state[f"file_read:{path_arg}"] = True
        state.add_observation(f"File '{path_arg}' read successfully via procedural skill", source="tool")
        state.add_verified_fact(f"File inspection completed for {path_arg}", source="execution", method="read_file_hash", artifact_hash=record.artifact_after_hash)
        state.set_frontier(current="tool_sequence_completed", next_action="complete", remaining=[])
        return True, result
    else:
        state.add_error(f"Tool sequence execution failed: {record.error}")
        state.set_frontier(current="tool_sequence_failed", next_action="handoff_to_brain", blocked=record.error)
        return False, result


DETERMINISTIC_REGISTRY = {
    "deterministic_json_validation": _run_deterministic_json,
    "deterministic_text_statistics": _run_deterministic_text_stats,
    "deterministic_ast_arithmetic": _run_deterministic_arithmetic,
    "deterministic_arithmetic": _run_deterministic_arithmetic,
}


# ============================================================
# Core MuscleExecutor Class per V6 Section I
# ============================================================
class MuscleExecutor:
    """Multi-tier procedural executor for automatic execution."""

    def __init__(
        self,
        provider: Optional[ModelProvider] = None,
        memory: Optional[ProceduralMemory] = None,
        cache: Optional[ResponseCache] = None,
    ):
        self.provider = provider
        self.memory = memory or ProceduralMemory()
        self.cache = cache or ResponseCache()

    def execute(
        self,
        user_message: str,
        exec_state: ExecutionState,
        decision: Optional[RouteDecision] = None,
        conv_state: Optional[ConversationState] = None,
        budget: Optional[ExecutionBudget] = None,
    ) -> MuscleResult:
        """Executes a task through the Muscle pathway across Tiers 0–3."""
        if budget is not None:
            exec_state.budget = budget
        if decision is None:
            decision = RouteDecision(pathway="muscle", risk="low", novelty=0.0)
        if conv_state is None:
            conv_state = ConversationState()

        # ------------------------------------------------------------
        # TIER 0: Response Cache (Exact Reusable Output, 0 Tokens)
        # ------------------------------------------------------------
        cached_resp = self.cache.lookup(user_message)
        if cached_resp is not None:
            exec_state.add_completed_step("Delivered exact response from zero-token ResponseCache")
            exec_state.add_verified_fact("Exact cached response invariant satisfied", source="cache", method="exact_match")
            exec_state.status = COMPLETED_VERIFIED
            exec_state.confidence = 1.0
            exec_state.set_frontier(current="cache_delivery", next_action="complete", remaining=[])
            return MuscleResult(
                status=COMPLETED_VERIFIED,
                confidence=1.0,
                text=cached_resp.text,
                partial_result=cached_resp.text,
                execution_state=exec_state,
                skill_used="cache_tier_0",
                input_tokens=0,
                output_tokens=0,
            )

        # Direct deterministic handler from router decision (e.g. arithmetic, json validation)
        if decision.procedure_match and decision.procedure_match in DETERMINISTIC_REGISTRY:
            handler = DETERMINISTIC_REGISTRY[decision.procedure_match]
            success, text = handler(user_message, exec_state)
            if success:
                exec_state.status = COMPLETED_VERIFIED
                exec_state.confidence = 1.0
                exec_state.metadata["is_deterministic"] = True
                exec_state.set_frontier(current="deterministic_completed", next_action="complete", remaining=[])
                return MuscleResult(
                    status=COMPLETED_VERIFIED,
                    confidence=1.0,
                    text=text,
                    partial_result=text,
                    execution_state=exec_state,
                    skill_used=decision.procedure_match,
                    input_tokens=0,
                    output_tokens=0,
                )

        # ------------------------------------------------------------
        # TIER 1 & TIER 2: Procedural Skills
        # ------------------------------------------------------------
        matched_skill = self.memory.retrieve(user_message, only_trusted=True)

        if matched_skill:
            if matched_skill.required_tools and matched_skill.procedure_type != ProcedureType.TOOL_SEQUENCE.value:
                exec_state.status = HANDOFF_REQUIRED
                exec_state.mark_unresolved(
                    f"Skill '{matched_skill.id}' requires tools {matched_skill.required_tools} — unavailable in Muscle mode"
                )
                return MuscleResult(
                    status=HANDOFF_REQUIRED,
                    confidence=0.6,
                    text="",
                    partial_result="",
                    reason=f"Skill '{matched_skill.id}' requires tool access; routing to Full Brain.",
                    execution_state=exec_state,
                    skill_used=matched_skill.id,
                    input_tokens=0,
                    output_tokens=0,
                )

            # Type A: DETERMINISTIC
            if (
                matched_skill.is_deterministic
                or matched_skill.procedure_type == ProcedureType.DETERMINISTIC.value
            ):
                handler_key = matched_skill.deterministic_handler or matched_skill.procedure
                handler = DETERMINISTIC_REGISTRY.get(handler_key)
                if handler:
                    success, text = handler(user_message, exec_state)
                    if success:
                        exec_state.status = COMPLETED_VERIFIED
                        exec_state.confidence = 1.0
                        exec_state.set_frontier(current="deterministic_completed", next_action="complete", remaining=[])
                        return MuscleResult(
                            status=COMPLETED_VERIFIED,
                            confidence=1.0,
                            text=text,
                            partial_result=text,
                            execution_state=exec_state,
                            skill_used=matched_skill.id,
                            input_tokens=0,
                            output_tokens=0,
                        )
                    else:
                        exec_state.status = HANDOFF_REQUIRED
                        exec_state.confidence = 0.4
                        exec_state.set_frontier(current="deterministic_failed", next_action="handoff_to_brain", blocked=text)
                        return MuscleResult(
                            status=HANDOFF_REQUIRED,
                            confidence=0.4,
                            partial_result=text,
                            reason=f"Deterministic skill '{matched_skill.id}' could not complete: {text}",
                            execution_state=exec_state,
                            skill_used=matched_skill.id,
                            input_tokens=0,
                            output_tokens=0,
                        )

            # Type B: TOOL_SEQUENCE via ProcedureExecutor
            if matched_skill.procedure_type == ProcedureType.TOOL_SEQUENCE.value:
                proc = None
                if matched_skill.procedure and matched_skill.procedure.strip().startswith("{"):
                    try:
                        data = json.loads(matched_skill.procedure)
                        if "steps" in data and data["steps"]:
                            proc = Procedure.from_dict(data)
                    except Exception:
                        pass

                if proc is not None:
                    executor = ProcedureExecutor(policy=SkillToolPolicy(allowed_tools=matched_skill.required_tools))
                    success, text, _ = executor.execute(proc, {"user_input": user_message}, exec_state)
                else:
                    success, text = _run_tool_sequence_procedure(matched_skill, user_message, exec_state)

                if success:
                    exec_state.status = COMPLETED_VERIFIED
                    exec_state.confidence = matched_skill.calibrated_confidence or 0.9
                    return MuscleResult(
                        status=COMPLETED_VERIFIED,
                        confidence=exec_state.confidence,
                        text=text,
                        partial_result=text,
                        execution_state=exec_state,
                        skill_used=matched_skill.id,
                        input_tokens=0,
                        output_tokens=0,
                    )
                else:
                    exec_state.status = HANDOFF_REQUIRED
                    exec_state.confidence = 0.4
                    return MuscleResult(
                        status=HANDOFF_REQUIRED,
                        confidence=0.4,
                        text="",
                        partial_result=text,
                        reason=f"Skill '{matched_skill.id}' tool sequence failed: {text}",
                        execution_state=exec_state,
                        skill_used=matched_skill.id,
                        input_tokens=0,
                        output_tokens=0,
                    )

            # Type C & D: HYBRID / MODEL_ADAPTIVE
            if self.provider is not None and not matched_skill.required_tools:
                prompt = (
                    f"[PROCEDURAL SKILL: {matched_skill.id} v{matched_skill.version}]\n"
                    f"Procedure:\n{matched_skill.procedure}\n\n"
                    f"Task: {user_message}"
                )
                resp = self.provider.muscle_memory_reply(prompt)
                exec_state.usage.record_call("muscle", resp)
                exec_state.add_completed_step(f"Executed procedural skill {matched_skill.id} (v{matched_skill.version})")
                exec_state.add_observation(f"Output received from skill {matched_skill.id}")
                exec_state.confidence = matched_skill.calibrated_confidence
                exec_state.status = COMPLETED_VERIFIED
                exec_state.set_frontier(current="skill_model_adaptive_completed", next_action="complete", remaining=[])
                return MuscleResult(
                    status=COMPLETED_VERIFIED,
                    confidence=matched_skill.calibrated_confidence,
                    text=resp.text,
                    partial_result=resp.text,
                    execution_state=exec_state,
                    skill_used=matched_skill.id,
                    input_tokens=resp.input_tokens,
                    output_tokens=resp.output_tokens,
                )

        # ------------------------------------------------------------
        # TIER 3: Bounded Adaptive Procedural Execution
        # ------------------------------------------------------------
        if self.provider is None:
            exec_state.status = HANDOFF_REQUIRED
            return MuscleResult(
                status=HANDOFF_REQUIRED,
                confidence=0.0,
                reason="No model provider available for Tier 3 execution",
                execution_state=exec_state,
            )

        # Signal A: Context Contradiction Check
        if decision.context_required and not conv_state.recent_plain_turns:
            exec_state.status = HANDOFF_REQUIRED
            exec_state.confidence = 0.2
            exec_state.mark_unresolved("Task requires prior context but no recent turns are available in memory buffer")
            exec_state.set_frontier(current="tier3_blocked", next_action="handoff_to_brain", blocked="Missing context")
            return MuscleResult(
                status=HANDOFF_REQUIRED,
                confidence=0.2,
                text="",
                partial_result="",
                reason="Context contradiction: context_required was True but no context exists in state buffer.",
                execution_state=exec_state,
                skill_used=None,
                input_tokens=0,
                output_tokens=0,
            )

        # Enforce budget before execution
        exceeded, limit_msg = exec_state.is_budget_exceeded()
        if exceeded:
            exec_state.status = HANDOFF_REQUIRED
            exec_state.mark_unresolved(f"Muscle budget exhausted: {limit_msg}")
            exec_state.set_frontier(current="tier3_budget_exceeded", next_action="handoff_to_brain", blocked=limit_msg)
            return MuscleResult(
                status=HANDOFF_REQUIRED,
                confidence=0.3,
                reason=limit_msg,
                execution_state=exec_state,
                skill_used=None,
            )

        # Inject recent 1-2 turns if context is required
        if decision.context_required and conv_state.recent_plain_turns:
            recent_tail = conv_state.recent_plain_turns[-2:]
            context_block = "\n".join(f"{r}: {t}" for r, t in recent_tail)
            prompt = f"[Context]:\n{context_block}\n\nUser: {user_message}"
        else:
            prompt = user_message

        resp = self.provider.muscle_memory_reply(prompt)
        exec_state.usage.record_call("muscle", resp)
        exec_state.add_completed_step("Completed bounded Muscle Memory reply")
        exec_state.add_observation("Direct response produced")

        reply_clean = resp.text.strip().lower()

        # Signal B: Hedging / Uncertainty Language Deny-List
        HEDGING_PATTERNS = [
            r"\bi('m| am) not sure\b",
            r"\bi('m| am) not entirely sure\b",
            r"\bi think\b",
            r"\bmight be\b",
            r"\bi don't have enough (information|context)\b",
            r"\bi cannot (determine|verify|answer)\b",
            r"\bit depends on (the context|information not provided)\b",
            r"\bcould you (please )?clarify\b",
            r"\bas an ai\b",
            r"\bi lack the context\b",
        ]
        for hp in HEDGING_PATTERNS:
            if re.search(hp, reply_clean):
                exec_state.status = HANDOFF_REQUIRED
                exec_state.confidence = 0.35
                exec_state.mark_unresolved(f"Model expressed hedging or uncertainty: '{hp}'")
                exec_state.set_frontier(current="tier3_hedging_detected", next_action="handoff_to_brain", blocked=hp)
                return MuscleResult(
                    status=HANDOFF_REQUIRED,
                    confidence=0.35,
                    text=resp.text,
                    partial_result=resp.text,
                    reason=f"Model output contained hedging/uncertainty marker matching pattern: '{hp}'",
                    execution_state=exec_state,
                    skill_used=None,
                    input_tokens=resp.input_tokens,
                    output_tokens=resp.output_tokens,
                )

        # Signal C: Structural Response Length Ratio
        is_substantive_req = any(w in user_message.lower() for w in ["explain", "code", "write", "list", "describe", "steps"])
        if is_substantive_req and len(reply_clean.split()) < 5:
            exec_state.status = HANDOFF_REQUIRED
            exec_state.confidence = 0.4
            exec_state.mark_unresolved("Substantive task yielded overly truncated response in Muscle Memory")
            exec_state.set_frontier(current="tier3_truncated", next_action="handoff_to_brain", blocked="Truncated output")
            return MuscleResult(
                status=HANDOFF_REQUIRED,
                confidence=0.4,
                text=resp.text,
                partial_result=resp.text,
                reason="Response length was suspiciously truncated for a substantive request.",
                execution_state=exec_state,
                skill_used=None,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
            )

        exec_state.status = COMPLETED_UNVERIFIED
        exec_state.confidence = 0.7
        exec_state.set_frontier(current="tier3_unverified", next_action="verification_gate", remaining=[])
        return MuscleResult(
            status=COMPLETED_UNVERIFIED,
            confidence=0.7,
            text=resp.text,
            partial_result=resp.text,
            execution_state=exec_state,
            skill_used=None,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
        )


def execute_muscle(*args, **kwargs) -> MuscleResult:
    """Flexible functional entrypoint wrapping MuscleExecutor for full backward compatibility
    with both legacy and V6 signatures.
    """
    user_message = None
    conv_state = None
    decision = None
    memory = None
    provider = None
    cache = None
    budget = None
    exec_state = None

    if len(args) >= 1 and isinstance(args[0], str):
        user_message = args[0]
        if len(args) > 1: conv_state = args[1]
        if len(args) > 2: decision = args[2]
        if len(args) > 3: memory = args[3]
        if len(args) > 4: provider = args[4]
        if len(args) > 5: cache = args[5]
        if len(args) > 6: budget = args[6]
        if len(args) > 7: exec_state = args[7]
    elif len(args) >= 1:
        provider = args[0]
        if len(args) > 1: conv_state = args[1]
        if len(args) > 2: exec_state = args[2]
        if len(args) > 3: user_message = args[3]
        if len(args) > 4: decision = args[4]
        if len(args) > 5: cache = args[5]
        if len(args) > 6: memory = args[6]
        if len(args) > 7: budget = args[7]

    if "user_message" in kwargs: user_message = kwargs["user_message"]
    if "provider" in kwargs: provider = kwargs["provider"]
    if "conv_state" in kwargs: conv_state = kwargs["conv_state"]
    if "decision" in kwargs: decision = kwargs["decision"]
    if "memory" in kwargs: memory = kwargs["memory"]
    if "cache" in kwargs: cache = kwargs["cache"]
    if "budget" in kwargs: budget = kwargs["budget"]
    if "exec_state" in kwargs: exec_state = kwargs["exec_state"]

    if user_message is None:
        user_message = ""
    if exec_state is None:
        exec_state = ExecutionState(objective=user_message)
    if conv_state is None:
        conv_state = ConversationState()
    if decision is None:
        decision = RouteDecision(pathway="muscle", risk="low", novelty=0.0)

    executor = MuscleExecutor(provider=provider, memory=memory, cache=cache)
    return executor.execute(
        user_message=user_message,
        exec_state=exec_state,
        decision=decision,
        conv_state=conv_state,
        budget=budget,
    )
