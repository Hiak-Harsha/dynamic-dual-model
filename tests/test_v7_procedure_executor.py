"""
test_v7_procedure_executor.py
Validates Action-Graph Procedural Execution, step-by-step state tracking,
exact failure point handoff, and conversational artifact quarantine per V7 Phase 7 & 8.
"""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from execution_state import ExecutionState, HANDOFF_REQUIRED, COMPLETED_VERIFIED
from procedure_executor import Procedure, ProcedureStep, ProcedureExecutor, StepType
from procedural_memory import ProceduralMemory, Skill
from tools import _resolve_in_sandbox, SkillToolPolicy


def test_procedure_executor_action_graph_success():
    """Validates step sequence execution: EXTRACT_REGEX -> TOOL_CALL (read_file) -> RETURN."""
    fname = "proc_exec_test.txt"
    fpath = _resolve_in_sandbox(fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write("Line 1\nLine 2\nLine 3")

    try:
        proc = Procedure(
            id="test_read_proc",
            name="Test Read Procedure",
            steps=[
                ProcedureStep(
                    id="step_1",
                    step_type=StepType.EXTRACT_REGEX.value,
                    pattern=r"file:\s*([\w\-\.]+)",
                    source_var="user_input",
                    output_var="target_file",
                ),
                ProcedureStep(
                    id="step_2",
                    step_type=StepType.TOOL_CALL.value,
                    tool="read_file",
                    arguments={"path": "${target_file}"},
                    output_var="file_text",
                ),
                ProcedureStep(
                    id="step_3",
                    step_type=StepType.RETURN.value,
                    return_expr="Content: ${file_text}",
                ),
            ],
        )

        state = ExecutionState(objective="Inspect file: proc_exec_test.txt")
        executor = ProcedureExecutor(policy=SkillToolPolicy(allowed_tools=["read_file"]))
        success, output, env = executor.execute(proc, {"user_input": "Please inspect file: proc_exec_test.txt"}, state)

        assert success is True
        assert "Line 1" in output
        assert state.status == COMPLETED_VERIFIED
        assert len(state.tool_actions) == 1
        assert len(state.tool_results) == 1
        assert state.tool_results[0]["success"] is True
        assert state.frontier["current"] == "completed"

    finally:
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except Exception:
                pass


def test_procedure_executor_records_exact_failure_point_and_handoff():
    """Validates that when a tool step fails, exact step failure and remaining steps are recorded."""
    proc = Procedure(
        id="test_fail_proc",
        name="Test Fail Procedure",
        steps=[
            ProcedureStep(
                id="step_1",
                step_type=StepType.CONSTANT.value,
                arguments={"value": "initialized"},
                output_var="status",
            ),
            ProcedureStep(
                id="step_2",
                step_type=StepType.TOOL_CALL.value,
                tool="read_file",
                arguments={"path": "nonexistent_file_abc.txt"},
                output_var="missing_content",
            ),
            ProcedureStep(
                id="step_3",
                step_type=StepType.RETURN.value,
                return_expr="${missing_content}",
            ),
        ],
    )

    state = ExecutionState(objective="Read nonexistent file")
    executor = ProcedureExecutor(policy=SkillToolPolicy(allowed_tools=["read_file"]))
    success, err_msg, env = executor.execute(proc, {}, state)

    assert success is False
    assert state.status == HANDOFF_REQUIRED
    # Exact step failure recorded in frontier
    assert state.frontier["current"] == "step_step_2_failed"
    assert state.frontier["next_action"] == "handoff_to_brain"
    assert "step_2" in state.frontier["remaining"]
    assert "step_3" in state.frontier["remaining"]
    assert len(state.errors) >= 1
    assert "step_2" in state.errors[0].lower() or "read_file" in state.errors[0].lower()


def test_procedure_executor_rejects_unauthorized_tools():
    """Validates that ProcedureExecutor does NOT self-grant tool permissions and strictly rejects unauthorized tools."""
    proc = Procedure(
        id="test_unauth_proc",
        name="Test Unauthorized Procedure",
        steps=[
            ProcedureStep(
                id="step_1",
                step_type=StepType.TOOL_CALL.value,
                tool="write_file",
                arguments={"path": "secret.txt", "content": "leak"},
            ),
        ],
    )
    # External policy authorizes only read_file, NOT write_file
    restricted_policy = SkillToolPolicy(allowed_tools=["read_file"])
    executor = ProcedureExecutor(policy=restricted_policy)
    state = ExecutionState(objective="Attempt write without authorization")
    success, err_msg, _ = executor.execute(proc, {}, state)

    assert success is False
    assert state.status == HANDOFF_REQUIRED
    assert "not authorized" in err_msg.lower() or "unauthorized" in err_msg.lower()
    assert state.frontier["current"] == "unauthorized_tool:write_file"
    assert state.frontier["next_action"] == "handoff_to_brain"


def test_procedural_memory_quarantines_conversational_llm_artifacts():
    """Validates Phase 8: Skills containing conversational text like 'Brain deliberative step' are quarantined."""
    mem = ProceduralMemory(storage_path="test_temp_skills.json")

    invalid_skill = Skill(
        id="bad_conversational_skill",
        steps=[
            "Attempted deterministic JSON validation",
            "Brain deliberative step 1 (stop_reason=end_turn)",
        ],
        procedure='{"steps": ["Brain deliberative step 1 (stop_reason=end_turn)"]}',
    )

    valid, reason = mem.is_valid_action_graph_skill(invalid_skill)
    assert valid is False
    assert "conversational" in reason.lower()
