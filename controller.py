"""
controller.py — Central lifecycle controller and orchestration layer for V9.

Per V6 Spec Sections C, E, F, O, P, Q, Z + V9 Universal Model Runtime:
- Single authoritative task lifecycle:
  START -> ExecutionState -> Intent/Safety Analysis -> Procedure Retrieval -> Route -> Execute -> Verify -> Success OR Handoff -> Brain continuation if needed -> Verify -> Final Result -> Learning Observer.
- Authoritative TaskUsage accounting: every provider call recorded.
- TrajectoryAnalyzer & LearnabilityFilter: Extracts genuine execution trajectories;
  rejects ordinary conversational Q&A; never names skills by first two words.

V9 ADDITION: All model calls route through UniversalModelRuntime, which performs
capability negotiation and records every call in the CognitionLedger.
"""

from __future__ import annotations
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from base import (
    ConversationState,
    ModelProvider,
    ProviderResponse,
    ResponseCache,
    SessionStats,
    TaskUsage,
    ExecutionBudget,
    BrainBudget,
    count_tokens,
)
from brain import BrainEngine, BrainTrajectory, select_effort
from execution_state import ExecutionState, HandoffState
from handoff import build_handoff
from muscle_engine import MuscleExecutor, MuscleResult, COMPLETED_VERIFIED, COMPLETED_UNVERIFIED, HANDOFF_REQUIRED
from procedural_memory import ProceduralMemory, Skill, ProcedureType, normalize_intent_family
from router import route, RouteDecision
from verification import verify, VerificationResult
from success_contracts import (
    SuccessContract,
    ContractResult,
    resolve_contract,
    NoVerificationAvailableContract,
    FileExistsContract,
    JsonValidContract,
    ReadOperationCompletedContract,
    WriteOperationCompletedContract,
)
from universal_runtime import UniversalModelRuntime, CognitionLedger, ModelCapabilityCard


@dataclass
class TaskContext:
    """Encapsulates the lifecycle context of a single user task."""
    task_id: str
    user_input: str
    state: ConversationState
    exec_state: ExecutionState
    decision: Optional[RouteDecision] = None
    final_text: str = ""
    verification_result: Optional[VerificationResult] = None
    success_contract: Optional[SuccessContract] = None
    contract_result: Optional[ContractResult] = None
    skill_used: Optional[str] = None
    success: bool = False
    start_time: float = field(default_factory=time.perf_counter)
    latency_ms: float = 0.0


def is_task_learnable(task: TaskContext, trajectory: Optional[BrainTrajectory]) -> Tuple[bool, str]:
    """Evaluates whether a task trajectory is suitable for procedural memory extraction per V6 Section P."""
    decision = task.decision
    if decision and decision.risk in {"high", "destructive"}:
        return False, "Task is high-risk/destructive"

    if not task.success or "[stopped:" in (task.final_text or ""):
        return False, "Task did not complete successfully"

    intent_fam, _ = normalize_intent_family(task.user_input)
    if intent_fam in {"conversational_greeting", "conversational_acknowledgment", "destructive_operation", "unclassified_intent"}:
        return False, f"Intent family '{intent_fam}' is conversational or non-procedural"

    has_tools = bool(task.exec_state.tool_actions)
    has_verified = bool(task.exec_state.verified_facts)
    has_steps = len(task.exec_state.completed_steps) >= 2

    if not (has_tools or has_verified or has_steps):
        return False, "No repeatable procedural actions, tool steps, or verified facts found"

    return True, ""


class AgentController:
    """Central controller coordinating routing, execution pathways, verification, and learning.

    V9: All model communication is mediated through UniversalModelRuntime.
    The runtime performs capability negotiation and records every call
    in the CognitionLedger. Provider-specific behavior never leaks into
    this controller.
    """

    def __init__(
        self,
        provider: Optional[ModelProvider] = None,
        state: Optional[ConversationState] = None,
        stats: Optional[SessionStats] = None,
        cache: Optional[ResponseCache] = None,
        memory: Optional[ProceduralMemory] = None,
        brain: Optional[BrainEngine] = None,
        summary_interval: int = 8,
        runtime: Optional[UniversalModelRuntime] = None,
    ):
        # V9: UniversalModelRuntime is the single gateway for all model calls.
        # If not provided, auto-create one wrapping the raw provider.
        if runtime is not None:
            self._runtime = runtime
            self.provider = runtime.provider
        elif provider is not None:
            self._runtime = UniversalModelRuntime(provider=provider)
            self.provider = provider
        else:
            raise ValueError("Either provider or runtime must be supplied to AgentController.")

        self.state = state or ConversationState()
        self.stats = stats or SessionStats()
        self.cache = cache or ResponseCache()
        self.memory = memory or ProceduralMemory()
        self.brain = brain or BrainEngine()
        self.muscle_executor = MuscleExecutor(provider=self._runtime, memory=self.memory, cache=self.cache)
        self.summary_interval = summary_interval
        self.last_mode = "init"

    @property
    def runtime(self) -> UniversalModelRuntime:
        """Access the Universal Model Runtime."""
        return self._runtime

    @property
    def ledger(self) -> CognitionLedger:
        """Access the CognitionLedger for audit/accounting."""
        return self._runtime.ledger

    def _finalize_task_state(self, task: TaskContext, reply: str, skill_used: Optional[str] = None) -> None:
        """Authoritative terminal state determination strictly driven by SuccessContract evaluation."""
        contract = task.success_contract
        if contract is None or isinstance(contract, NoVerificationAvailableContract):
            # Specialize contract based on execution state and physical artifacts per V8.1 Section 8
            written = [k for k in task.exec_state.tool_state if k.startswith("file_written:")]
            read_acts = [k for k in task.exec_state.tool_state if k.startswith("file_accessed:") or k.startswith("file_read:")]
            if written:
                contract = WriteOperationCompletedContract(path=written[0].split(":", 1)[1])
            elif read_acts and not any(w in task.user_input.lower() for w in ["diagnose", "fix", "repair", "modify", "rewrite", "debug", "refactor", "clean"]):
                contract = ReadOperationCompletedContract()
            elif "```json" in reply or (reply.strip().startswith("{") and reply.strip().endswith("}")) or (reply.strip().startswith("[") and reply.strip().endswith("]")):
                try:
                    cand = reply.strip()
                    if "```json" in cand:
                        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cand)
                        cand = m.group(1).strip() if m else cand
                    json.loads(cand)
                    contract = JsonValidContract()
                except Exception:
                    pass
            elif contract is None:
                contract = resolve_contract(task.user_input, provider=self._runtime)

        task.success_contract = contract
        contract_res = contract.evaluate(task, reply, task.exec_state)
        task.contract_result = contract_res
        task.verification_result = contract_res
        task.final_text = reply
        if skill_used:
            task.skill_used = skill_used

        if not contract_res.execution_completed:
            task.exec_state.status = "failed"
            task.success = False
            task.exec_state.add_error(f"Execution did not complete: {contract_res.reason}")
        elif contract_res.verified:
            task.exec_state.status = COMPLETED_VERIFIED
            task.success = True
            task.exec_state.add_verified_fact(contract_res.evidence, source="contract", method=contract_res.contract_type)
        elif contract_res.task_satisfied:
            task.exec_state.status = COMPLETED_UNVERIFIED
            task.success = True
        else:
            task.exec_state.status = "failed"
            task.success = False
            task.exec_state.add_error(f"Success contract unsatisfied: {contract_res.reason}")

    def start(self, user_input: str, contract: Optional[SuccessContract] = None) -> TaskContext:
        """Initializes a new task lifecycle context."""
        exec_state = ExecutionState(objective=user_input)
        resolved_contract = contract or resolve_contract(user_input, provider=self._runtime)
        return TaskContext(
            task_id=exec_state.task_id,
            user_input=user_input,
            state=self.state,
            exec_state=exec_state,
            success_contract=resolved_contract,
        )

    def route_task(self, task: TaskContext) -> RouteDecision:
        """Evaluates routing dimensions and determines execution pathway."""
        decision = route(
            user_message=task.user_input,
            state=self.state,
            cache=self.cache,
            memory=self.memory,
            provider=self._runtime,
        )
        task.decision = decision
        if decision.routing_response is not None and not task.exec_state.usage.routing_recorded:
            task.exec_state.usage.record_call("routing", decision.routing_response)
            self.stats.add(
                "classify",
                self.provider.name,
                decision.routing_response,
                model=self.provider.model,
                effort="none",
                latency_ms=getattr(decision, "routing_latency_ms", 0.0),
            )
        return decision

    def execute_task(self, task: TaskContext) -> str:
        """Executes task following the single authoritative lifecycle per V6 Section E."""
        if not task.decision:
            task.decision = self.route_task(task)
        decision = task.decision
        if decision.routing_response is not None and not task.exec_state.usage.routing_recorded:
            task.exec_state.usage.record_call("routing", decision.routing_response)
            self.stats.add(
                "classify",
                self.provider.name,
                decision.routing_response,
                model=self.provider.model,
                effort="none",
                latency_ms=getattr(decision, "routing_latency_ms", 0.0),
            )

        # Pathway: CACHE
        if decision.pathway == "cache":
            t0 = time.perf_counter()
            cached = self.cache.lookup(task.user_input)
            latency = (time.perf_counter() - t0) * 1000
            if cached is not None:
                self._finalize_task_state(task, cached.text, skill_used="cache_tier_0")
                task.exec_state.add_verified_fact("Exact cached response invariant satisfied", source="cache", method="exact_match")

                # Cache hit: 0 current provider calls, 0 current billable tokens
                task.exec_state.usage.cache_hit = True
                self.stats.add("cache", self.provider.name, cached, model=self.provider.model, effort="none", cached_response=True, latency_ms=latency)
                self.state.add_plain_turn("user", task.user_input)
                self.state.add_plain_turn("assistant", cached.text)
                self.record_usage(task)
                self.compress_summary_if_needed(task)
                return cached.text

        # Pathway: MUSCLE
        if decision.pathway == "muscle":
            t0 = time.perf_counter()
            muscle_res: MuscleResult = self.muscle_executor.execute(
                user_message=task.user_input,
                exec_state=task.exec_state,
                decision=decision,
                conv_state=self.state,
                budget=task.exec_state.budget,
            )
            muscle_latency = (time.perf_counter() - t0) * 1000
            task.exec_state = muscle_res.execution_state

            # Evaluate authoritative SuccessContract on Muscle result
            contract = task.success_contract or resolve_contract(task.user_input, provider=self._runtime)
            contract_res = contract.evaluate(task, muscle_res.text, task.exec_state)
            task.contract_result = contract_res

            # Authoritative check: Proceed if independently verified, or if conversational turn with satisfied task
            is_verified = contract_res.verified
            is_conversational_complete = (
                contract_res.task_satisfied
                and (
                    "routine greeting" in getattr(decision, "reasoning", "")
                    or (getattr(decision, "risk", "") == "low" and not getattr(decision, "tools_required", False) and getattr(decision, "novelty", 0.0) == 0.0)
                )
            )

            if (is_verified or is_conversational_complete) and muscle_res.status not in ("handoff_required", "blocked", "failed"):
                self._finalize_task_state(task, muscle_res.text, skill_used=muscle_res.skill_used)

                if muscle_res.skill_used and muscle_res.skill_used in self.memory.skills:
                    self.memory.reinforce(muscle_res.skill_used, input_variant=task.user_input)

                # Cache deterministic results
                resp_obj = ProviderResponse(
                    text=muscle_res.text,
                    tool_calls=[],
                    stop_reason="end_turn",
                    raw=None,
                    input_tokens=muscle_res.input_tokens,
                    output_tokens=muscle_res.output_tokens,
                )
                if (
                    muscle_res.skill_used
                    and muscle_res.skill_used in self.memory.skills
                    and self.memory.skills[muscle_res.skill_used].is_deterministic
                ):
                    self.cache.put(task.user_input, resp_obj)

                self.stats.add("muscle_memory", self.provider.name, resp_obj, model=self.provider.model, effort="low", latency_ms=muscle_latency)
                self.state.add_plain_turn("user", task.user_input)
                self.state.add_plain_turn("assistant", muscle_res.text)
                self.record_usage(task)
                self.compress_summary_if_needed(task)
                return muscle_res.text

            # Muscle unverified or incomplete: Handoff to Brain continuation
            # Record Muscle attempt tokens so SessionStats matches TaskUsage exactly
            if muscle_res.input_tokens > 0 or muscle_res.output_tokens > 0:
                resp_obj = ProviderResponse(
                    text=muscle_res.text,
                    tool_calls=[],
                    stop_reason="end_turn",
                    raw=None,
                    input_tokens=muscle_res.input_tokens,
                    output_tokens=muscle_res.output_tokens,
                )
                self.stats.add("muscle_memory", self.provider.name, resp_obj, model=self.provider.model, effort="low", latency_ms=muscle_latency)

            if muscle_res.skill_used and muscle_res.skill_used in self.memory.skills:
                self.memory.penalize(muscle_res.skill_used, error_reason=contract_res.reason)

            handoff = build_handoff(
                task.exec_state,
                reason=f"Verification not satisfied ({contract_res.reason})" if not contract_res.verified else muscle_res.reason,
                suggested_next_step="Continue execution from established Task Frontier and resolve remaining items.",
                procedure_info=muscle_res.skill_used,
            )

            effort = select_effort(decision.risk, decision.novelty)
            t0 = time.perf_counter()

            def on_continuation_step(step_resp: ProviderResponse, step_lat: float):
                self.stats.add("full_brain", self.provider.name, step_resp, model=self.provider.model, effort=effort, latency_ms=step_lat)

            reply, last_resp, cont_exec_state = self.brain.continue_from_handoff(
                self._runtime, self.state, task.user_input, handoff, effort=effort, on_step=on_continuation_step
            )
            task.exec_state = cont_exec_state
            if last_resp.stop_reason == "end_turn":
                task.exec_state.errors = []

            self._finalize_task_state(task, reply)

            self.state.add_plain_turn("user", task.user_input)
            self.state.add_plain_turn("assistant", reply)

            if task.success:
                self.observe_and_extract_procedure(task)

            self.record_usage(task)
            self.compress_summary_if_needed(task)
            return reply

        # Pathway: LIGHT_BRAIN or FULL_BRAIN
        effort = "low" if decision.pathway == "light_brain" else select_effort(decision.risk, decision.novelty)
        t0 = time.perf_counter()

        def on_brain_step(step_resp: ProviderResponse, step_lat: float):
            self.stats.add("full_brain", self.provider.name, step_resp, model=self.provider.model, effort=effort, latency_ms=step_lat)

        reply, last_resp, brain_exec_state = self.brain.execute_turn(
            self._runtime, self.state, task.user_input, effort=effort, exec_state=task.exec_state, on_step=on_brain_step
        )
        task.exec_state = brain_exec_state
        if last_resp.stop_reason == "end_turn":
            task.exec_state.errors = []

        self._finalize_task_state(task, reply)

        self.state.add_plain_turn("user", task.user_input)
        self.state.add_plain_turn("assistant", reply)

        if task.success:
            self.observe_and_extract_procedure(task)

        self.record_usage(task)
        self.compress_summary_if_needed(task)
        return reply

    def observe_and_extract_procedure(self, task: TaskContext) -> None:
        """Extracts a reusable execution procedure from Brain trajectory per V6 Section O & P."""
        trajectory = self.brain.last_trajectory
        learnable, reason = is_task_learnable(task, trajectory)
        if not learnable:
            return

        intent_fam, intent_name = normalize_intent_family(task.user_input)
        clean_words = re.sub(r"[^\w\s]+", "", task.user_input.lower()).split()
        action = clean_words[0] if clean_words else "task"
        skill_id = f"proc_{intent_fam}_{action}"

        # If already registered, reinforce with input variant
        if skill_id in self.memory.skills:
            self.memory.reinforce(skill_id, input_variant=task.user_input)
            return

        tools_called = list(set(a.get("name") for a in task.exec_state.tool_actions if isinstance(a, dict) and a.get("name")))
        proc_type = ProcedureType.TOOL_SEQUENCE.value if tools_called else ProcedureType.MODEL_ADAPTIVE.value

        # Filter out conversational LLM steps per V7 Phase 8
        conversational_indicators = ["brain deliberative", "stop_reason=", "resumed execution from", "handoff"]
        steps = [
            s for s in task.exec_state.completed_steps
            if not any(ind in s.lower() for ind in conversational_indicators)
        ]
        if not steps and trajectory:
            steps = [
                s for s in trajectory.state_transitions
                if not any(ind in s.lower() for ind in conversational_indicators)
            ]

        # If tools were invoked, build a structured action graph procedure with parameterization
        if tools_called and task.exec_state.tool_actions:
            action_graph_steps = []
            extracted_vars: Dict[str, str] = {}
            extract_counter = 1

            # Detect file paths or names in user input for parameterization
            path_pattern = r"['\"]?([a-zA-Z0-9_\-\.\/]+\.[a-zA-Z0-9_]{1,6})['\"]?"
            path_matches = re.findall(path_pattern, task.user_input)
            if path_matches:
                primary_path = path_matches[0]
                action_graph_steps.append({
                    "id": f"extract_step_{extract_counter}",
                    "step_type": "EXTRACT_REGEX",
                    "pattern": path_pattern,
                    "source_var": "user_input",
                    "output_var": "extracted_path",
                })
                extracted_vars[primary_path] = "${extracted_path}"
                extract_counter += 1

            for idx, a in enumerate(task.exec_state.tool_actions):
                if isinstance(a, dict) and a.get("name"):
                    raw_args = a.get("arguments", {})
                    param_args = {}
                    if isinstance(raw_args, dict):
                        for k, v in raw_args.items():
                            val_str = str(v)
                            for literal, token in extracted_vars.items():
                                if literal in val_str:
                                    val_str = val_str.replace(literal, token)
                            param_args[k] = val_str
                    else:
                        param_args = raw_args

                    action_graph_steps.append({
                        "id": f"step_{idx+1}",
                        "step_type": "TOOL_CALL",
                        "tool": a.get("name"),
                        "arguments": param_args,
                        "output_var": f"out_{idx+1}",
                    })
            if action_graph_steps:
                action_graph_steps.append({
                    "id": f"step_{len(action_graph_steps)+1}",
                    "step_type": "RETURN",
                    "return_expr": f"${{out_{len(task.exec_state.tool_actions)}}}",
                })
                proc_str = json.dumps({
                    "id": skill_id,
                    "name": f"Action Graph for {intent_name}",
                    "steps": action_graph_steps,
                })
                if not steps:
                    steps = [f"Execute tool {a.get('name')}" for a in task.exec_state.tool_actions if isinstance(a, dict)]
            else:
                proc_str = json.dumps({"intent": intent_fam, "steps": steps, "required_tools": tools_called})
        else:
            # Without tools or verifiable execution steps, do not distill conversational noise into procedural memory
            if not steps:
                return
            proc_str = json.dumps({"intent": intent_fam, "steps": steps, "required_tools": tools_called})

        patterns = [
            r"^\s*" + re.escape(action) + r"\b.*",
            r".*\b" + re.escape(intent_fam.replace("_", " ")) + r"\b.*",
        ]

        self.memory.register_candidate(
            skill_id=skill_id,
            name=f"Learned {intent_name}",
            intent_family=intent_fam,
            semantic_intent=f"Reusable procedure for {intent_name}",
            trigger_patterns=patterns,
            preconditions=[f"Input matches intent family {intent_fam}"],
            non_applicability_conditions=["Destructive actions or malformed inputs"],
            steps=steps,
            procedure_type=proc_type,
            procedure=proc_str,
            required_tools=tools_called,
            verification=task.verification_result.tier_used if task.verification_result else "structural",
            is_deterministic=False,
        )

    def record_usage(self, task: TaskContext) -> None:
        """Records comprehensive TaskUsage into SessionStats."""
        task.exec_state.usage.task_id = task.task_id
        task.exec_state.usage.source = "live" if self.provider.name in {"anthropic", "openai"} else "simulated"
        self.stats.record_task_usage(task.exec_state.usage)
        task.latency_ms = (time.perf_counter() - task.start_time) * 1000

    def compress_summary_if_needed(self, task: Optional[Any] = None, force: bool = False) -> None:
        """Compresses conversation history into running summary if interval reached or forced."""
        if not force and self.state.full_brain_turns_since_summary < self.summary_interval:
            return

        if not self.state.full_brain_history and not self.state.recent_plain_turns:
            return

        text_to_compress = self.state.recent_context_str()
        if not text_to_compress and self.state.full_brain_history:
            text_to_compress = "\n".join(str(item) for item in self.state.full_brain_history)

        if not text_to_compress:
            return

        t0 = time.perf_counter()
        resp = self._runtime.summarize_history_step(text_to_compress)
        latency = (time.perf_counter() - t0) * 1000

        self.state.apply_summary_truncation(resp.text)
        self.stats.add("summary", self.provider.name, resp, model=self.provider.model, effort="low", latency_ms=latency)
        if task is not None and hasattr(task, "exec_state"):
            task.exec_state.usage.record_call("summary", resp)

    def shutdown(self) -> None:
        """Flushes memory and persists state on shutdown."""
        self.memory.flush()
