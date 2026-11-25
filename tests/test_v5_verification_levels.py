"""
tests/test_v5_verification_levels.py — Tests for V5 Verification Engine (Levels 0 through 4).

Per V5 Spec Sections 12 & 13:
- Level 0: Deterministic invariant
- Level 1: Structural verification (JSON, Python AST)
- Level 2: Execution verification (artifacts on disk, tool outcomes)
- Level 3: Evidence/consistency verification (claims vs observations, hedging detection)
- Level 4: Model-based verification
- Produces VerificationRecord stored inside ExecutionState.
"""

from __future__ import annotations
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from execution_state import ExecutionState
from verification import verify, VerificationResult, VerificationRecord
from mock_provider import MockProvider
from tools import _resolve_in_sandbox, execute_tool


def test_level_0_deterministic_invariant():
    """Validates Level 0 deterministic invariant verification."""
    state = ExecutionState(objective="Calculate 12 * 8")
    res = verify("96", "Calculate 12 * 8", state, verification_mode="level_0")
    assert res.passed
    assert res.tier_used == "deterministic"
    assert len(state.verification_records) == 1
    assert state.verification_records[0]["level"] == 0
    assert "96" in state.verified_facts[0]

    # Empty result fails Level 0
    res_fail = verify("", "Calculate 12 * 8", state, verification_mode="level_0")
    assert not res_fail.passed


def test_level_1_structural_verification():
    """Validates Level 1 structural verification against JSON and Python AST."""
    state = ExecutionState(objective="Output user config as JSON")

    # Valid JSON
    res_json = verify('{"user": "alice", "active": true}', "Output user config as JSON", state, verification_mode="level_1")
    assert res_json.passed
    assert res_json.tier_used == "structural"
    assert "JSON payload parsed successfully" in state.verified_facts[-1]

    # Invalid JSON fails
    res_json_fail = verify('{"user": "alice", active: true}', "Output user config as JSON", state, verification_mode="level_1")
    assert not res_json_fail.passed

    # Valid Python AST
    py_code = "```python\ndef greet(name):\n    return f'Hello, {name}'\n```"
    res_py = verify(py_code, "Write greet function", state, verification_mode="level_1")
    assert res_py.passed
    assert "Python code blocks verified syntax-clean" in state.verified_facts[-1]

    # Invalid Python AST fails
    py_code_fail = "```python\ndef broken(\n```"
    res_py_fail = verify(py_code_fail, "Write broken function", state, verification_mode="level_1")
    assert not res_py_fail.passed


def test_level_2_execution_verification():
    """Validates Level 2 execution verification against artifacts on disk."""
    state = ExecutionState(objective="Write configuration file")
    test_rel = "test_v5_exec_verify.txt"
    execute_tool("write_file", {"path": test_rel, "content": "config_val=42", "overwrite": True})

    state.tool_state[f"file_written:{test_rel}"] = True
    res = verify("File written successfully.", "Write configuration file", state, verification_mode="level_2")
    assert res.passed
    assert res.tier_used == "execution"

    # Missing file fails Level 2
    state.tool_state["file_written:nonexistent_ghost_file.txt"] = True
    res_fail = verify("File written successfully.", "Write configuration file", state, verification_mode="level_2")
    assert not res_fail.passed
    assert "missing on disk" in res_fail.reason.lower() or "was not found" in res_fail.reason.lower()

    # Cleanup
    real = _resolve_in_sandbox(test_rel)
    if os.path.exists(real):
        os.remove(real)


def test_level_3_evidence_consistency_and_hedging():
    """Validates Level 3 consistency verification: detects hedging language and unresolved items."""
    state = ExecutionState(objective="Verify production database hostname")
    state.add_observation("Database hostname is db.internal.net")

    # Consistent clean claim
    res = verify("The production database hostname is db.internal.net.", "Verify database", state, verification_mode="level_3")
    assert res.passed
    assert res.tier_used == "consistency"

    # Hedging language fails Level 3
    res_hedging = verify("I think the database might be db.internal.net, but I am not entirely sure.", "Verify database", state, verification_mode="level_3")
    assert not res_hedging.passed
    assert "hedging" in res_hedging.reason.lower()

    # Unresolved work in state fails Level 3
    state.mark_unresolved("Confirm backup database replica")
    res_unresolved = verify("Database hostname is db.internal.net.", "Verify database", state, verification_mode="level_3")
    assert not res_unresolved.passed
    assert "unresolved" in res_unresolved.reason.lower()


def test_level_4_model_based_verification():
    """Validates Level 4 model-based verification using Provider."""
    class PassingJudgeProvider(MockProvider):
        def muscle_memory_reply(self, prompt: str):
            from base import ProviderResponse
            return ProviderResponse(text="PASS: The summary accurately reflects the task objective.", tool_calls=[], stop_reason="end_turn", raw=None, input_tokens=40, output_tokens=15)

    provider = PassingJudgeProvider()
    state = ExecutionState(objective="Summarize release notes")
    res = verify("Version 5.0 introduces dual-path execution.", "Summarize release notes", state, verification_mode="level_4", provider=provider)
    assert res.passed
    assert res.tier_used == "model"
    assert res.tokens_used == 55
    assert len(state.verification_records) == 1
    assert state.verification_records[0]["level"] == 4
