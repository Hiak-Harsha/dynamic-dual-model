"""
procedure_executor.py — Deterministic Action-Graph Procedural Executor for V7.

Executes structured procedural action graphs without passing stored procedure text
as an LLM prompt. Implements variable binding, tool invocation under SkillToolPolicy,
regex extraction, JSON transformation, condition branching, and exact failure point
handoff recording per V7 Phase 7.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import re
import os
from typing import Any, Dict, List, Optional, Tuple

from execution_state import ExecutionState, HANDOFF_REQUIRED, COMPLETED_VERIFIED
from tools import execute_tool_with_record, SkillToolPolicy, _resolve_in_sandbox, get_file_content_hash


class StepType(str, Enum):
    TOOL_CALL = "TOOL_CALL"
    CONSTANT = "CONSTANT"
    EXTRACT_REGEX = "EXTRACT_REGEX"
    TRANSFORM_JSON = "TRANSFORM_JSON"
    VERIFY_CONDITION = "VERIFY_CONDITION"
    BRANCH = "BRANCH"
    RETURN = "RETURN"


@dataclass
class ProcedureStep:
    id: str
    step_type: str
    tool: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    output_var: Optional[str] = None
    pattern: Optional[str] = None
    source_var: Optional[str] = None
    json_path: Optional[str] = None
    condition: Optional[Dict[str, Any]] = None
    on_true_goto: Optional[str] = None
    on_false_goto: Optional[str] = None
    return_expr: Optional[str] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProcedureStep:
        return cls(
            id=str(data.get("id", "step_0")),
            step_type=str(data.get("step_type", StepType.TOOL_CALL.value)).upper(),
            tool=data.get("tool"),
            arguments=data.get("arguments", {}),
            output_var=data.get("output_var"),
            pattern=data.get("pattern"),
            source_var=data.get("source_var"),
            json_path=data.get("json_path"),
            condition=data.get("condition"),
            on_true_goto=data.get("on_true_goto"),
            on_false_goto=data.get("on_false_goto"),
            return_expr=data.get("return_expr"),
            description=data.get("description", ""),
        )


@dataclass
class Procedure:
    id: str
    name: str = ""
    steps: List[ProcedureStep] = field(default_factory=list)
    policy: Optional[SkillToolPolicy] = None
    stop_conditions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "steps": [s.to_dict() for s in self.steps],
            "stop_conditions": self.stop_conditions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Procedure:
        raw_steps = data.get("steps", [])
        steps = [
            s if isinstance(s, ProcedureStep) else ProcedureStep.from_dict(s)
            for s in raw_steps
        ]
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            steps=steps,
            stop_conditions=data.get("stop_conditions", []),
        )


class ProcedureExecutor:
    """Interprets and executes structured procedural action graphs."""

    def __init__(self, policy: Optional[SkillToolPolicy] = None):
        self.policy = policy or SkillToolPolicy()

    def _resolve_template_vars(self, val: Any, env: Dict[str, Any]) -> Any:
        if isinstance(val, str):
            def replacer(m):
                var_name = m.group(1).strip()
                if var_name in env:
                    return str(env[var_name])
                # Support dot-notation access like input.path or params.id
                parts = var_name.split(".")
                curr = env
                found = True
                for p in parts:
                    if isinstance(curr, dict) and p in curr:
                        curr = curr[p]
                    else:
                        found = False
                        break
                if found:
                    return str(curr)
                return m.group(0)
            return re.sub(r"\$\{([^}]+)\}", replacer, val)
        elif isinstance(val, dict):
            return {k: self._resolve_template_vars(v, env) for k, v in val.items()}
        elif isinstance(val, list):
            return [self._resolve_template_vars(i, env) for i in val]
        return val

    def execute(
        self,
        procedure: Procedure,
        initial_env: Dict[str, Any],
        state: ExecutionState,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Executes procedure step-by-step against execution_state.
        Returns (success: bool, final_output: str, final_env: Dict[str, Any]).
        """
        env: Dict[str, Any] = dict(initial_env)
        # External policy authorization: permissions come strictly from trusted external policy per V8.1 Section 3
        policy = self.policy
        observed_tools = list(dict.fromkeys([s.tool for s in procedure.steps if s.tool]))
        authorized_tools = list(policy.allowed_tools) if (policy and policy.allowed_tools) else []

        step_map: Dict[str, int] = {s.id: idx for idx, s in enumerate(procedure.steps)}
        idx = 0
        total_steps = len(procedure.steps)
        max_loop_iterations = max(50, total_steps * 5)
        iterations = 0

        while idx < total_steps:
            iterations += 1
            if iterations > max_loop_iterations:
                state.add_error("Procedural execution loop limit exceeded (infinite loop protection)")
                state.status = HANDOFF_REQUIRED
                state.set_frontier(current="loop_exceeded", next_action="handoff_to_brain", blocked="Loop limit")
                return False, "Procedural loop limit exceeded", env

            step = procedure.steps[idx]
            step_type = step.step_type.upper()
            state.current_step = f"Executing step '{step.id}' ({step_type}): {step.description or step.tool or ''}"

            # ------------------------------------------------------------
            # STEP: TOOL_CALL
            # ------------------------------------------------------------
            if step_type == StepType.TOOL_CALL.value:
                tool_name = step.tool or ""
                resolved_args = self._resolve_template_vars(step.arguments, env)

                # Authorization check: external policy strictly controls tool invocation
                if tool_name not in authorized_tools:
                    err_msg = f"Tool '{tool_name}' not authorized by external policy (authorized: {authorized_tools}, observed: {observed_tools})"
                    state.add_error(err_msg)
                    state.status = HANDOFF_REQUIRED
                    remaining = [s.id for s in procedure.steps[idx:]]
                    state.set_frontier(
                        current=f"unauthorized_tool:{tool_name}",
                        next_action="handoff_to_brain",
                        blocked=err_msg,
                        remaining=remaining,
                    )
                    return False, err_msg, env

                # Check tool reuse / recomputation per V8.1 Section 2
                can_reuse, cached_out, rep_type = state.check_tool_reuse(tool_name, resolved_args)
                if can_reuse and cached_out is not None:
                    if rep_type == "reuse":
                        state.reuse_count += 1
                        state.recomputation_types["reuse"] = state.recomputation_types.get("reuse", 0) + 1
                    else:
                        state.recomputation_count += 1
                        state.recomputation_types[rep_type] = state.recomputation_types.get(rep_type, 0) + 1
                    state.add_observation(f"Reused verified fresh tool output for step '{step.id}' ({tool_name})")
                    if step.output_var:
                        env[step.output_var] = cached_out
                    idx += 1
                    continue

                # Execute with authoritative record
                call_id = f"proc_{procedure.id}_{step.id}"
                out_str, record = execute_tool_with_record(
                    name=tool_name,
                    arguments=resolved_args,
                    call_id=call_id,
                    policy=policy,
                    current_calls=len(state.tool_actions),
                )

                state.record_tool_action(
                    {
                        "name": tool_name,
                        "arguments": resolved_args,
                        "id": record.call_id,
                        "step_id": step.id,
                    },
                    phase="muscle",
                    action_type="tool_call",
                )
                rec_dict = record.to_dict()
                rec_dict["name"] = tool_name
                rec_dict["output"] = out_str
                state.record_tool_result(rec_dict)

                if not record.success:
                    # Exact failure point recorded into execution state
                    err_msg = record.error or out_str
                    state.add_error(f"Step '{step.id}' tool '{tool_name}' failed: {err_msg}")
                    state.status = HANDOFF_REQUIRED
                    remaining = [s.id for s in procedure.steps[idx:]]
                    state.set_frontier(
                        current=f"step_{step.id}_failed",
                        next_action="handoff_to_brain",
                        blocked=err_msg,
                        remaining=remaining,
                    )
                    return False, err_msg, env

                # Record freshness if file operation
                if tool_name in {"read_file", "write_file"} and "path" in resolved_args:
                    p = resolved_args["path"]
                    state.record_state_freshness(p, out_str)
                    state.tool_state[f"file_accessed:{p}"] = True

                state.add_completed_step(f"Step '{step.id}' completed: {tool_name}")
                if step.output_var:
                    env[step.output_var] = out_str
                env["last_output"] = out_str
                env["output"] = out_str
                idx += 1

            # ------------------------------------------------------------
            # STEP: CONSTANT
            # ------------------------------------------------------------
            elif step_type == StepType.CONSTANT.value:
                val = self._resolve_template_vars(step.arguments.get("value", ""), env)
                if step.output_var:
                    env[step.output_var] = val
                state.add_completed_step(f"Step '{step.id}' set constant: {step.output_var}")
                idx += 1

            # ------------------------------------------------------------
            # STEP: EXTRACT_REGEX
            # ------------------------------------------------------------
            elif step_type == StepType.EXTRACT_REGEX.value:
                src_text = str(env.get(step.source_var or "user_input", ""))
                pattern = step.pattern or ""
                m = re.search(pattern, src_text)
                if not m:
                    err_msg = f"Step '{step.id}' regex match failed for pattern: {pattern}"
                    state.add_error(err_msg)
                    state.status = HANDOFF_REQUIRED
                    state.set_frontier(
                        current=f"step_{step.id}_regex_failed",
                        next_action="handoff_to_brain",
                        blocked=err_msg,
                        remaining=[s.id for s in procedure.steps[idx:]],
                    )
                    return False, err_msg, env

                matched_val = m.group(1) if m.groups() else m.group(0)
                if step.output_var:
                    env[step.output_var] = matched_val
                state.add_completed_step(f"Step '{step.id}' regex extracted: {step.output_var}='{matched_val}'")
                idx += 1

            # ------------------------------------------------------------
            # STEP: TRANSFORM_JSON
            # ------------------------------------------------------------
            elif step_type == StepType.TRANSFORM_JSON.value:
                src_text = str(env.get(step.source_var or "", ""))
                try:
                    parsed = json.loads(src_text)
                    if step.json_path:
                        for k in step.json_path.split("."):
                            if isinstance(parsed, dict) and k in parsed:
                                parsed = parsed[k]
                    if step.output_var:
                        env[step.output_var] = parsed
                    state.add_completed_step(f"Step '{step.id}' parsed JSON successfully")
                    idx += 1
                except Exception as e:
                    err_msg = f"Step '{step.id}' JSON transformation failed: {e}"
                    state.add_error(err_msg)
                    state.status = HANDOFF_REQUIRED
                    state.set_frontier(
                        current=f"step_{step.id}_json_failed",
                        next_action="handoff_to_brain",
                        blocked=err_msg,
                        remaining=[s.id for s in procedure.steps[idx:]],
                    )
                    return False, err_msg, env

            # ------------------------------------------------------------
            # STEP: VERIFY_CONDITION
            # ------------------------------------------------------------
            elif step_type == StepType.VERIFY_CONDITION.value:
                cond = step.condition or {}
                check_type = cond.get("type", "")
                passed = False

                if check_type == "file_exists":
                    p = self._resolve_template_vars(cond.get("path", ""), env)
                    resolved = _resolve_in_sandbox(p)
                    passed = bool(resolved and os.path.exists(resolved))
                elif check_type == "non_empty":
                    v = env.get(cond.get("var", ""), "")
                    passed = bool(v and str(v).strip())
                elif check_type == "equals":
                    v1 = self._resolve_template_vars(cond.get("actual", ""), env)
                    v2 = self._resolve_template_vars(cond.get("expected", ""), env)
                    passed = (v1 == v2)

                if passed:
                    state.add_verified_fact(f"Condition '{check_type}' verified in step {step.id}", source="procedure")
                    if step.on_true_goto and step.on_true_goto in step_map:
                        idx = step_map[step.on_true_goto]
                    else:
                        idx += 1
                else:
                    state.add_error(f"Verification condition '{check_type}' failed in step {step.id}")
                    if step.on_false_goto and step.on_false_goto in step_map:
                        idx = step_map[step.on_false_goto]
                    else:
                        state.status = HANDOFF_REQUIRED
                        state.set_frontier(
                            current=f"step_{step.id}_condition_failed",
                            next_action="handoff_to_brain",
                            blocked=f"Condition {check_type} unsatisfied",
                            remaining=[s.id for s in procedure.steps[idx:]],
                        )
                        return False, f"Condition failed in step {step.id}", env

            # ------------------------------------------------------------
            # STEP: BRANCH
            # ------------------------------------------------------------
            elif step_type == StepType.BRANCH.value:
                target = step.on_true_goto
                if target and target in step_map:
                    idx = step_map[target]
                else:
                    idx += 1

            # ------------------------------------------------------------
            # STEP: RETURN
            # ------------------------------------------------------------
            elif step_type == StepType.RETURN.value:
                ret_val = self._resolve_template_vars(step.return_expr or "${output}", env)
                state.add_completed_step(f"Step '{step.id}' returned final output")
                state.status = COMPLETED_VERIFIED
                state.set_frontier(current="completed", next_action="complete", remaining=[])
                return True, str(ret_val), env

            else:
                idx += 1

        # Completed all steps without explicit return
        last_out = env.get("output", "") or env.get("result", "") or "Procedure completed successfully"
        state.status = COMPLETED_VERIFIED
        state.set_frontier(current="completed", next_action="complete", remaining=[])
        return True, str(last_out), env
