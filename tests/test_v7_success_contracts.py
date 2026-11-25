"""
test_v7_success_contracts.py
Comprehensive validation for V7 Success Contracts, strict Level 4 verification,
factual grounding verification, artifact freshness hash verification,
and arithmetic bounded safety.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from base import ProviderResponse
from execution_state import ExecutionState, COMPLETED_VERIFIED, COMPLETED_UNVERIFIED
from success_contracts import (
    ExactValueContract,
    JsonValidContract,
    JsonInvalidContract,
    PythonSyntaxContract,
    FileExistsContract,
    FileHashContract,
    EvidenceRequiredContract,
    resolve_contract,
)
from verification import verify
from tools import _resolve_in_sandbox, execute_tool_structured, SkillToolPolicy
from muscle_engine import _run_deterministic_arithmetic
from mock_provider import MockProvider


def test_level_4_model_verifier_rejects_deceptive_pass_substring():
    """Validates Bug 5 fix: 'FAIL - do not pass' must NOT pass verification."""
    class DeceptiveProvider(MockProvider):
        def muscle_memory_reply(self, prompt: str):
            # Model explicitly says FAIL - do not pass
            return ProviderResponse(
                text="FAIL - do not pass because key requirements were missed.",
                tool_calls=[],
                stop_reason="end_turn",
                raw=None,
                input_tokens=40,
                output_tokens=15,
            )

    provider = DeceptiveProvider()
    state = ExecutionState(objective="Verify code quality")
    res = verify("result text", "Verify code quality", state, verification_mode="level_4", provider=provider)
    assert res.passed is False
    assert res.is_factual_truth is False


def test_level_4_model_verifier_accepts_strict_json_pass():
    """Validates strict structured JSON PASS parsing."""
    class StructuredPassProvider(MockProvider):
        def muscle_memory_reply(self, prompt: str):
            return ProviderResponse(
                text='```json\n{"verdict": "PASS", "confidence": 0.98, "reason": "All checks satisfied"}\n```',
                tool_calls=[],
                stop_reason="end_turn",
                raw=None,
                input_tokens=50,
                output_tokens=25,
            )

    provider = StructuredPassProvider()
    state = ExecutionState(objective="Verify calculation")
    res = verify("result text", "Verify calculation", state, verification_mode="level_4", provider=provider)
    assert res.passed is True
    assert res.confidence == 0.98
    assert res.reason == "All checks satisfied"


def test_level_4_model_verifier_rejects_json_fail_and_uncertain():
    """Validates that structured JSON with FAIL or UNCERTAIN verdict fails verification."""
    class StructuredUncertainProvider(MockProvider):
        def muscle_memory_reply(self, prompt: str):
            return ProviderResponse(
                text='{"verdict": "UNCERTAIN", "confidence": 0.4, "reason": "Insufficient evidence"}',
                tool_calls=[],
                stop_reason="end_turn",
                raw=None,
                input_tokens=50,
                output_tokens=25,
            )

    provider = StructuredUncertainProvider()
    state = ExecutionState(objective="Verify calculation")
    res = verify("result text", "Verify calculation", state, verification_mode="level_4", provider=provider)
    assert res.passed is False


def test_level_0_rejects_mismatched_value():
    """Validates Bug 6 fix: Level 0 does not accept arbitrary non-empty text when expected_value mismatches."""
    state = ExecutionState(objective="Compute 2+2")
    res = verify("Result: 5", "Compute 2+2", state, verification_mode="level_0", expected_value=4)
    assert res.passed is False
    assert "does not match" in res.reason.lower() or "mismatch" in res.record.evidence.lower()


def test_level_3_factual_question_ungrounded_is_not_factual_truth():
    """Validates Bug 7 fix: Factual questions without independent evidence do not get certified as truth."""
    state = ExecutionState(objective="What is the capital of Australia?")
    res = verify("Sydney", "What is the capital of Australia?", state, verification_mode="consistency")
    assert res.passed is True  # No contradiction detected with empty state
    assert res.is_factual_truth is False  # But NOT certified factual truth!


def test_artifact_freshness_rejects_reuse_when_disk_content_changes():
    """Validates Bug 3 fix: modifying file from A to B prevents stale reuse."""
    fname = "freshness_test_artifact.txt"
    fpath = _resolve_in_sandbox(fname)

    # Initial state: File contains "Version A"
    with open(fpath, "w", encoding="utf-8") as f:
        f.write("Version A")

    try:
        state = ExecutionState(objective="Process artifact")
        state.record_tool_action({"name": "read_file", "arguments": {"path": fname}})
        state.record_tool_result({"name": "read_file", "output": "Version A"})
        state.record_state_freshness(fname, "Version A")

        # Initial check: disk has Version A, state freshness has Version A
        can_reuse, out, rep_type = state.check_tool_reuse("read_file", {"path": fname})
        assert can_reuse is True
        assert out == "Version A"

        # Now mutate file on disk to "Version B"
        with open(fpath, "w", encoding="utf-8") as f:
            f.write("Version B")

        # Freshness check must detect hash change and reject reuse!
        can_reuse_after, out_after, rep_type_after = state.check_tool_reuse("read_file", {"path": fname})
        assert can_reuse_after is False
        assert rep_type_after == "legitimate_repeat"

        # Now test deletion: file deleted from disk
        os.remove(fpath)
        can_reuse_del, _, rep_type_del = state.check_tool_reuse("read_file", {"path": fname})
        assert can_reuse_del is False
        assert rep_type_del == "legitimate_repeat"

    finally:
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except Exception:
                pass


def test_arithmetic_exponent_and_boolean_safety():
    """Validates Bug 4 fix: Exponents > 1000 and booleans are rejected in AST arithmetic."""
    state = ExecutionState(objective="Compute arithmetic")

    # Exponent > 1000
    ok, err = _run_deterministic_arithmetic("2 ** 1001", state)
    assert ok is False
    assert "MAX_EXPONENT" in err or "exceeds" in err

    # Boolean rejected
    ok_bool, err_bool = _run_deterministic_arithmetic("True + 1", state)
    assert ok_bool is False
    assert "Boolean" in err_bool or "not allowed" in err_bool


def test_structured_tool_result():
    """Validates Phase 10: execute_tool_structured returns structured ToolResult."""
    # Read nonexistent file
    res = execute_tool_structured("read_file", {"path": "nonexistent_file_xyz.txt"})
    assert res.success is False
    assert res.error_code == "EXECUTION_ERROR"

    # Unknown tool
    res_unknown = execute_tool_structured("nonexistent_tool", {})
    assert res_unknown.success is False
    assert res_unknown.error_code == "UNKNOWN_TOOL"


def test_immutable_skill_tool_policy():
    """Validates Phase 9: SkillToolPolicy enforces allowed tools and path boundaries."""
    policy = SkillToolPolicy(
        allowed_tools=["read_file"],
        max_calls=2,
        allow_write=False,
        allowed_paths=["safe_dir/"],
    )

    # Write tool disallowed
    ok, reason = policy.check_permitted("write_file", {"path": "safe_dir/test.txt"}, 0)
    assert ok is False

    # Unauthorized path disallowed
    ok, reason = policy.check_permitted("read_file", {"path": "other_dir/test.txt"}, 0)
    assert ok is False
    assert "authorized allowed_paths" in reason

    # Max calls limit
    ok, reason = policy.check_permitted("read_file", {"path": "safe_dir/test.txt"}, 2)
    assert ok is False
    assert "Maximum tool calls" in reason
