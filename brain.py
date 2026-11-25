"""
brain.py — Deliberative Brain execution engine, trajectory capture, and frontier continuation executor.

Per V6 Spec Sections O, AD, AE, AH, AI, AJ:
- BrainTrajectory: captures objective, tool calls, arguments, results, observations, decisions, verified facts, postconditions.
- Avoids verified recomputation: reuses verified fresh tool outputs instead of executing redundant calls.
- Continuation from Frontier: continues directly on the same ExecutionState.
- Structured continuation: Objective represented once, frontier clearly exposed.
- Atomic native tool history: tool calls and tool results always paired.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from base import ConversationState, ModelProvider, ProviderResponse, ToolCall, ToolDef, count_tokens
from execution_state import ExecutionState, HandoffState, EvidenceLevel, canonical_tool_args
from handoff import render_handoff_prompt
from tools import TOOLS, execute_tool, execute_tool_with_record, get_file_content_hash

MAX_TOOL_ITERATIONS = 8


@dataclass
class BrainTrajectory:
    """Rigorous execution trajectory recorded during a Brain turn per V6 Section O."""
    task_id: str
    objective: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    decisions: List[Dict[str, str]] = field(default_factory=list)
    verified_facts: List[str] = field(default_factory=list)
    verification_records: List[Dict[str, Any]] = field(default_factory=list)
    state_transitions: List[str] = field(default_factory=list)
    postconditions: Dict[str, Any] = field(default_factory=dict)
    final_output: str = ""
    success: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def select_effort(decision_or_risk: Any = "low", novelty: float = 0.0) -> str:
    """Dynamic effort selection based on task risk and novelty."""
    from router import RouteDecision
    if isinstance(decision_or_risk, RouteDecision):
        risk = decision_or_risk.risk
        nov = decision_or_risk.novelty
    elif isinstance(decision_or_risk, str):
        risk = decision_or_risk
        nov = novelty
    else:
        risk = getattr(decision_or_risk, "risk", "low")
        nov = getattr(decision_or_risk, "novelty", 0.0)

    if risk == "high":
        return "high"
    if nov >= 0.8:
        return "high"
    if nov >= 0.3 or risk == "medium":
        return "medium"
    return "low"


class BrainEngine:
    """Deliberative reasoning engine managing multi-turn tool loops, state continuation, and trajectory recording."""

    def __init__(self, tools: Optional[List[ToolDef]] = None, max_tool_iterations: int = MAX_TOOL_ITERATIONS):
        self.tools = tools or TOOLS
        self.max_tool_iterations = max_tool_iterations
        self.last_trajectory: Optional[BrainTrajectory] = None

    def execute_turn(
        self,
        provider: ModelProvider,
        state: ConversationState,
        user_message: str,
        effort: str = "medium",
        handoff_preamble: Optional[str] = None,
        exec_state: Optional[ExecutionState] = None,
        budget: Optional[Any] = None,
        on_step: Optional[Any] = None,
        allow_tools: bool = True,
    ) -> Tuple[str, ProviderResponse, ExecutionState]:
        """Executes a Full Brain deliberative turn with full tool loop and updates exec_state."""
        if exec_state is None:
            exec_state = ExecutionState(objective=user_message)

        trajectory = BrainTrajectory(
            task_id=exec_state.task_id,
            objective=exec_state.objective or user_message,
        )
        self.last_trajectory = trajectory

        effective_effort = getattr(budget, "effort", effort) if budget else effort
        effective_iterations = getattr(budget, "max_tool_iterations", self.max_tool_iterations) if budget else self.max_tool_iterations

        # Structured continuation per V6 Section AH: do not duplicate user message if handoff provided
        if handoff_preamble:
            full_user_message = handoff_preamble
        elif not state.full_brain_history and state.recent_plain_turns:
            context = state.recent_context_str()
            full_user_message = f"[CONTEXT FROM PRIOR TURNS]:\n{context}\n\nTask: {user_message}"
        else:
            full_user_message = user_message

        provider.append_user_turn(state.full_brain_history, full_user_message)
        state.full_brain_turns_since_summary += 1

        tool_failure_counts: Dict[str, int] = exec_state.tool_failures
        last_resp = ProviderResponse(text="", tool_calls=[], stop_reason="end_turn", raw=None)

        for iteration in range(effective_iterations):
            # Enforce budget before initiating model step
            exceeded, limit_msg = exec_state.is_budget_exceeded()
            if exceeded:
                exec_state.status = "partial"
                exec_state.mark_unresolved(f"Execution budget exceeded before iteration {iteration + 1}: {limit_msg}")
                warning = f"[stopped: {limit_msg}]"
                last_resp.text = f"{last_resp.text}\n\n{warning}" if last_resp.text else warning
                trajectory.final_output = last_resp.text
                trajectory.success = False
                return last_resp.text, last_resp, exec_state

            active_tools = self.tools if allow_tools else []
            t0 = time.perf_counter()
            resp = provider.full_brain_step(state.full_brain_history, active_tools, effort=effective_effort)
            latency = (time.perf_counter() - t0) * 1000
            last_resp = resp

            exec_state.usage.record_call("brain", resp)
            if resp.tool_calls:
                step_desc = f"Deliberative tool execution: {', '.join(c.name for c in resp.tool_calls)}"
            else:
                step_desc = "Deliberative reasoning conclusion reached"
            exec_state.add_completed_step(step_desc)
            trajectory.state_transitions.append(step_desc)

            if on_step is not None:
                try:
                    on_step(resp, latency)
                except Exception:
                    pass

            provider.append_assistant_turn(state.full_brain_history, resp)

            if resp.stop_reason == "end_turn" or not resp.tool_calls:
                exec_state.status = "completed_unverified" if not exec_state.errors else "failed"
                exec_state.partial_result = resp.text
                exec_state.set_frontier(current="brain_turn_completed", next_action="evaluate_contract", remaining=[])
                trajectory.final_output = resp.text
                trajectory.success = not bool(exec_state.errors)
                trajectory.verified_facts = list(exec_state.verified_facts)
                trajectory.observations = list(exec_state.observations)
                trajectory.decisions = list(exec_state.decisions)
                return resp.text, resp, exec_state

            # Handle atomic tool call sequence per V8.1 Section 10
            tool_results_dict: Dict[str, str] = {}
            unexecuted_reason: Optional[str] = None

            for call in resp.tool_calls:
                if unexecuted_reason is not None:
                    not_exec_res = json.dumps({"status": "NOT_EXECUTED", "reason": unexecuted_reason})
                    tool_results_dict[call.id] = not_exec_res
                    exec_state.record_tool_result({
                        "tool_call_id": call.id,
                        "name": call.name,
                        "output": not_exec_res,
                        "success": False,
                        "error_code": "NOT_EXECUTED",
                        "error_message": unexecuted_reason,
                    })
                    continue

                exceeded, limit_msg = exec_state.is_budget_exceeded()
                if exceeded:
                    exec_state.status = "partial"
                    exec_state.mark_unresolved(f"Tool budget exceeded for '{call.name}': {limit_msg}")
                    unexecuted_reason = "tool_budget_exceeded"
                    not_exec_res = json.dumps({"status": "NOT_EXECUTED", "reason": unexecuted_reason})
                    tool_results_dict[call.id] = not_exec_res
                    exec_state.record_tool_result({
                        "tool_call_id": call.id,
                        "name": call.name,
                        "output": not_exec_res,
                        "success": False,
                        "error_code": "NOT_EXECUTED",
                        "error_message": unexecuted_reason,
                    })
                    continue

                canonical_args = canonical_tool_args(call.arguments)
                call_dict = {"name": call.name, "arguments": call.arguments, "id": call.id, "canonical": canonical_args}
                trajectory.tool_calls.append(call_dict)

                # Recomputation check per V8.1 Section 2
                can_reuse, cached_output, rep_category = exec_state.check_tool_reuse(call.name, call.arguments)
                if can_reuse and cached_output is not None:
                    if rep_category == "reuse":
                        exec_state.reuse_count += 1
                        exec_state.recomputation_types["reuse"] = exec_state.recomputation_types.get("reuse", 0) + 1
                    else:
                        exec_state.recomputation_count += 1
                        exec_state.recomputation_types[rep_category] = exec_state.recomputation_types.get(rep_category, 0) + 1
                    exec_state.add_observation(f"Reused verified result for tool '{call.name}'")
                    result = cached_output
                    exec_state.record_tool_result({
                        "tool_call_id": call.id,
                        "name": call.name,
                        "output": result,
                        "success": True,
                    })
                else:
                    exec_state.record_tool_action(call_dict, phase="brain", action_type="tool_call")

                    if tool_failure_counts.get(call.name, 0) >= 2:
                        result = (
                            f"FATAL: Tool '{call.name}' has failed {tool_failure_counts[call.name]} consecutive times. "
                            f"Do not call this tool again. Explain the issue directly to the user."
                        )
                        exec_state.record_tool_result({
                            "tool_call_id": call.id,
                            "name": call.name,
                            "output": result,
                            "success": False,
                            "error_code": "CONSECUTIVE_FAILURES",
                        })
                    else:
                        result, record = execute_tool_with_record(call.name, call.arguments, call_id=call.id)
                        trajectory.tool_results.append(record.to_dict())

                        rec_dict = record.to_dict()
                        rec_dict["tool_call_id"] = call.id
                        rec_dict["name"] = call.name
                        rec_dict["output"] = result
                        exec_state.record_tool_result(rec_dict)

                        if not record.success:
                            tool_failure_counts[call.name] = tool_failure_counts.get(call.name, 0) + 1
                            exec_state.add_error(f"Tool {call.name} error: {record.error}")
                        else:
                            tool_failure_counts[call.name] = 0
                            exec_state.add_observation(f"Tool {call.name} executed successfully", source="tool")
                            if call.name == "write_file" and isinstance(call.arguments, dict) and "path" in call.arguments:
                                path_val = call.arguments["path"]
                                exec_state.tool_state[f"file_written:{path_val}"] = True
                                if "content" in call.arguments:
                                    exec_state.record_state_freshness(path_val, call.arguments["content"], operation="write")
                                trajectory.postconditions[f"file_written:{path_val}"] = record.artifact_after_hash
                            elif call.name == "read_file" and isinstance(call.arguments, dict) and "path" in call.arguments:
                                path_val = call.arguments["path"]
                                exec_state.tool_state[f"file_read:{path_val}"] = True
                                exec_state.record_state_freshness(path_val, result, operation="read")

                tool_results_dict[call.id] = result

            # Append atomic tool results to native history
            provider.append_tool_results(state.full_brain_history, resp.tool_calls, tool_results_dict)

            if unexecuted_reason is not None:
                # Break turn loop due to budget exhaustion
                break

        warning = "[stopped: too many tool iterations]"
        last_resp.text = f"{last_resp.text}\n\n{warning}" if last_resp.text else warning
        exec_state.status = "partial"
        exec_state.mark_unresolved("Hard cap on tool iterations reached")
        trajectory.final_output = last_resp.text
        trajectory.success = False
        return last_resp.text, last_resp, exec_state

    def continue_from_handoff(
        self,
        provider: ModelProvider,
        state: ConversationState,
        user_message: str,
        handoff: HandoffState,
        effort: str = "medium",
        budget: Optional[Any] = None,
        on_step: Optional[Any] = None,
    ) -> Tuple[str, ProviderResponse, ExecutionState]:
        """Continues execution seamlessly from a Muscle handoff without restarting cognition per V8.1 Section 13."""
        handoff_prompt = render_handoff_prompt(handoff, max_tokens=400)
        handoff.execution_state.add_completed_step("Full Brain resumed execution from Muscle Memory handoff")
        existing_next = handoff.execution_state.next_action or "resolve_remaining_work"
        handoff.execution_state.current_step = existing_next

        return self.execute_turn(
            provider=provider,
            state=state,
            user_message=user_message,
            effort=effort,
            handoff_preamble=handoff_prompt,
            exec_state=handoff.execution_state,
            budget=budget,
            on_step=on_step,
        )

    run_turn = execute_turn
