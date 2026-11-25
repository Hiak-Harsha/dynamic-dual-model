"""
tests/test_v6_procedural_replay.py — Tests for V6 procedural replay and tool policies.

Per V6 Spec Sections I, J, Z, AA, AB:
- DETERMINISTIC skills replay with 0 model tokens.
- TOOL_SEQUENCE skills execute within SkillToolPolicy and track state freshness.
- Writes are disallowed in Muscle unless explicitly permitted.
- Malformed inputs trigger graceful handoff rather than silent failures.
"""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from base import ConversationState, ModelProvider, ProviderResponse, ResponseCache
from execution_state import ExecutionState
from muscle_engine import MuscleExecutor, COMPLETED_VERIFIED, HANDOFF_REQUIRED
from procedural_memory import ProceduralMemory, Skill, ProcedureType
from router import RouteDecision
from tools import SkillToolPolicy, execute_tool_with_record, _resolve_in_sandbox
from mock_provider import MockProvider


def test_deterministic_procedure_zero_tokens():
    memory = ProceduralMemory()
    executor = MuscleExecutor(memory=memory)
    state = ExecutionState(objective="Calculate 12 * 12")

    result = executor.execute("Calculate 12 * 12", state)
    assert result.status == COMPLETED_VERIFIED
    assert "144" in result.text
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert len(state.verified_facts) >= 1
    assert state.frontier["current_step"] == "deterministic_completed"


def test_tool_sequence_under_policy(tmp_path):
    # Setup test file in sandbox
    test_file = _resolve_in_sandbox("procedural_test.txt")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("Line 1\nLine 2\nKey: Value")

    try:
        memory = ProceduralMemory()
        executor = MuscleExecutor(memory=memory)
        state = ExecutionState(objective="read procedural_test.txt")

        # safe_read_file_procedure is seeded in ProceduralMemory
        result = executor.execute("read procedural_test.txt", state)
        assert result.status == COMPLETED_VERIFIED
        assert "Key: Value" in result.text
        assert state.tool_state.get("file_read:procedural_test.txt") is True
        assert len(state.verified_facts) >= 1
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)


def test_tool_policy_blocks_unauthorized_write():
    policy = SkillToolPolicy(allowed_tools=["read_file"], allow_write=False)
    res, rec = execute_tool_with_record("write_file", {"path": "blocked.txt", "content": "bad"}, policy=policy)
    assert not rec.success
    assert "not permitted by SkillToolPolicy" in rec.error


def test_malformed_json_validation_reports_syntax_error():
    memory = ProceduralMemory()
    executor = MuscleExecutor(memory=memory)
    state = ExecutionState(objective="validate broken json")

    result = executor.execute('validate json: {"broken": true,', state)
    assert result.status == COMPLETED_VERIFIED
    assert "Invalid JSON syntax detected" in result.text
