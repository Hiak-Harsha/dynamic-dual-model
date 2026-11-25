"""
tests/test_v8_success_contracts_exhaustive.py
Exhaustive validation of all 13 Success Contracts in success_contracts.py:
1. ExactValueContract
2. TextStatisticsContract
3. JsonValidContract
4. JsonInvalidContract
5. PythonSyntaxContract
6. FileExistsContract
7. FileHashContract
8. FileContentMatchContract
9. ToolPostconditionContract
10. StructuralContract
11. EvidenceRequiredContract
12. ModelRubricContract
13. NoVerificationAvailableContract
"""

import hashlib
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from execution_state import ExecutionState
from success_contracts import (
    ContractType,
    ContractResult,
    ExactValueContract,
    TextStatisticsContract,
    JsonValidContract,
    JsonInvalidContract,
    PythonSyntaxContract,
    FileExistsContract,
    FileHashContract,
    FileContentMatchContract,
    ToolPostconditionContract,
    StructuralContract,
    EvidenceRequiredContract,
    ModelRubricContract,
    NoVerificationAvailableContract,
    resolve_contract,
)
from tools import _resolve_in_sandbox
from mock_provider import MockProvider


# ============================================================
# 1. ExactValueContract Tests
# ============================================================
def test_exact_value_numeric_match():
    c = ExactValueContract(expected_value=42)
    state = ExecutionState()
    res = c.evaluate(None, "Result: 42", state)
    assert res.task_satisfied is True
    assert res.verified is True
    assert res.confidence == 1.0


def test_exact_value_numeric_mismatch():
    c = ExactValueContract(expected_value=42)
    state = ExecutionState()
    res = c.evaluate(None, "Result: 99", state)
    assert res.task_satisfied is False
    assert res.verified is False


def test_exact_value_string_match():
    c = ExactValueContract(expected_value="Sydney")
    state = ExecutionState()
    res = c.evaluate(None, "Result: Sydney", state)
    assert res.task_satisfied is True
    assert res.verified is True


def test_exact_value_empty_text():
    c = ExactValueContract(expected_value=42)
    state = ExecutionState()
    res = c.evaluate(None, "   ", state)
    assert res.execution_completed is False
    assert res.verified is False


def test_exact_value_unsupplied_expected_value_without_deterministic_proof():
    """Security audit: unsupplied expected_value without AST provenance cannot certify verified."""
    c = ExactValueContract(expected_value=None)
    state = ExecutionState()
    res = c.evaluate(None, "Result: Sydney", state)
    assert res.task_satisfied is True
    assert res.verified is False  # Must remain unverified!
    assert "cannot certify correctness" in res.reason.lower()


def test_exact_value_unsupplied_expected_value_with_deterministic_proof():
    """Deterministic AST arithmetic execution state allows verified extraction."""
    c = ExactValueContract(expected_value=None)
    state = ExecutionState()
    state.add_completed_step("Attempted safe AST arithmetic evaluation: deterministic_arithmetic")
    state.metadata["is_deterministic"] = True
    res = c.evaluate(None, "Result: 1200", state)
    assert res.task_satisfied is True
    assert res.verified is True
    assert res.confidence == 1.0


# ============================================================
# 2. TextStatisticsContract Tests
# ============================================================
def test_text_statistics_valid():
    c = TextStatisticsContract()
    state = ExecutionState()
    text = "Text Statistics:\n- Lines: 3\n- Words: 12\n- Characters: 80"
    res = c.evaluate(None, text, state)
    assert res.task_satisfied is True
    assert res.verified is True


def test_text_statistics_malformed():
    c = TextStatisticsContract()
    state = ExecutionState()
    res = c.evaluate(None, "There are some lines and words in this string.", state)
    assert res.task_satisfied is False
    assert res.verified is False


# ============================================================
# 3. JsonValidContract Tests
# ============================================================
def test_json_valid_contract():
    c = JsonValidContract()
    state = ExecutionState()
    res = c.evaluate(None, '```json\n{"status": "ok", "count": 5}\n```', state)
    assert res.task_satisfied is True
    assert res.verified is True


def test_json_valid_contract_invalid():
    c = JsonValidContract()
    state = ExecutionState()
    res = c.evaluate(None, '```json\n{"status": "ok", "count": \n```', state)
    assert res.task_satisfied is False
    assert res.verified is False


# ============================================================
# 4. JsonInvalidContract Tests
# ============================================================
def test_json_invalid_contract():
    c = JsonInvalidContract()
    state = ExecutionState()
    res = c.evaluate(None, 'Invalid JSON syntax detected: Expecting value at line 1', state)
    assert res.task_satisfied is True
    assert res.verified is True


def test_json_invalid_contract_when_json_is_actually_valid():
    c = JsonInvalidContract()
    state = ExecutionState()
    res = c.evaluate(None, '```json\n{"valid": true}\n```', state)
    assert res.task_satisfied is False
    assert res.verified is False


# ============================================================
# 5. PythonSyntaxContract Tests
# ============================================================
def test_python_syntax_contract_valid():
    c = PythonSyntaxContract()
    state = ExecutionState()
    code = "```python\ndef add(a, b):\n    return a + b\n```"
    res = c.evaluate(None, code, state)
    assert res.task_satisfied is True
    assert res.verified is True


def test_python_syntax_contract_syntax_error():
    c = PythonSyntaxContract()
    state = ExecutionState()
    code = "```python\ndef add(a, b\n    return a + b\n```"
    res = c.evaluate(None, code, state)
    assert res.task_satisfied is False
    assert res.verified is False


# ============================================================
# 6. FileExistsContract Tests
# ============================================================
def test_file_exists_contract():
    fname = "test_contract_file_exists.txt"
    fpath = _resolve_in_sandbox(fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write("content")

    try:
        c = FileExistsContract(path=fname)
        state = ExecutionState()
        res = c.evaluate(None, "Created file", state)
        assert res.task_satisfied is True
        assert res.verified is True

        c_nonexistent = FileExistsContract(path="nonexistent_xyz.txt")
        res_non = c_nonexistent.evaluate(None, "Checked file", state)
        assert res_non.task_satisfied is False
        assert res_non.verified is False
    finally:
        if os.path.exists(fpath):
            os.remove(fpath)


# ============================================================
# 7. FileHashContract Tests
# ============================================================
def test_file_hash_contract():
    fname = "test_contract_hash.txt"
    fpath = _resolve_in_sandbox(fname)
    content = "deterministic content for hashing 123"
    expected_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)

    try:
        c = FileHashContract(path=fname, expected_hash=expected_sha)
        state = ExecutionState()
        res = c.evaluate(None, "File saved", state)
        assert res.task_satisfied is True
        assert res.verified is True

        # Wrong hash
        c_bad = FileHashContract(path=fname, expected_hash="00000000000000000000000000000000")
        res_bad = c_bad.evaluate(None, "File saved", state)
        assert res_bad.task_satisfied is False
        assert res_bad.verified is False
    finally:
        if os.path.exists(fpath):
            os.remove(fpath)


# ============================================================
# 8. FileContentMatchContract Tests
# ============================================================
def test_file_content_match_contract():
    fname = "test_contract_snippet.txt"
    fpath = _resolve_in_sandbox(fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write("database_host=localhost\ndatabase_port=5432\nenv=prod")

    try:
        c = FileContentMatchContract(path=fname, expected_snippet="database_port=5432")
        state = ExecutionState()
        res = c.evaluate(None, "File created", state)
        assert res.task_satisfied is True
        assert res.verified is True

        c_missing = FileContentMatchContract(path=fname, expected_snippet="timeout=999")
        res_mis = c_missing.evaluate(None, "File created", state)
        assert res_mis.task_satisfied is False
        assert res_mis.verified is False
    finally:
        if os.path.exists(fpath):
            os.remove(fpath)


# ============================================================
# 9. ToolPostconditionContract Tests
# ============================================================
def test_tool_postcondition_contract():
    c = ToolPostconditionContract(tool_name="read_file", expected_value="localhost")
    state = ExecutionState()
    # Missing tool result
    res_fail = c.evaluate(None, "Tool called", state)
    assert res_fail.task_satisfied is False

    # Successful tool result with value
    state.record_tool_result({"name": "read_file", "output": "host=localhost", "success": True})
    res_pass = c.evaluate(None, "Tool called", state)
    assert res_pass.task_satisfied is True
    assert res_pass.verified is True


# ============================================================
# 10. StructuralContract Tests
# ============================================================
def test_structural_contract():
    c = StructuralContract(expected_structure="json")
    state = ExecutionState()
    res = c.evaluate(None, '{"status": 200}', state)
    assert res.task_satisfied is True
    assert res.verified is True

    res_fail = c.evaluate(None, 'Plain text without structure', state)
    assert res_fail.task_satisfied is False
    assert res_fail.verified is False


# ============================================================
# 11. EvidenceRequiredContract Tests
# ============================================================
def test_evidence_required_contract_ungrounded():
    c = EvidenceRequiredContract()
    state = ExecutionState()
    # No verified evidence in state
    res = c.evaluate(None, "The capital of France is Paris.", state)
    assert res.task_satisfied is True
    assert res.verified is False  # Must be unverified without external grounding!


def test_evidence_required_contract_grounded():
    c = EvidenceRequiredContract()
    state = ExecutionState()
    state.add_verified_fact("Capital verified from atlas", source="external", method="lookup")
    res = c.evaluate(None, "The capital of France is Paris.", state)
    assert res.task_satisfied is True
    assert res.verified is True


# ============================================================
# 12. ModelRubricContract Tests
# ============================================================
def test_model_rubric_contract():
    prov = MockProvider()
    c = ModelRubricContract(rubric="Must be polite", provider=prov)
    state = ExecutionState()
    res = c.evaluate(None, "Thank you very much, have a great day!", state)
    assert res.task_satisfied is True
    assert res.verified is True


def test_model_rubric_contract_no_provider():
    c = ModelRubricContract(rubric="Must be polite", provider=None)
    state = ExecutionState()
    res = c.evaluate(None, "Hello", state)
    assert res.task_satisfied is True
    assert res.verified is False


# ============================================================
# 13. NoVerificationAvailableContract Tests
# ============================================================
def test_no_verification_available_contract():
    c = NoVerificationAvailableContract()
    state = ExecutionState()
    res = c.evaluate(None, "General creative response", state)
    assert res.execution_completed is True
    assert res.task_satisfied is True
    assert res.verified is False


# ============================================================
# Contract Resolution Factory Tests
# ============================================================
def test_resolve_contract_factory():
    assert isinstance(resolve_contract("what is 48 * 25?"), ExactValueContract)
    assert isinstance(resolve_contract("calculate word count of file"), TextStatisticsContract)
    assert isinstance(resolve_contract("validate json: {}"), JsonValidContract)
    assert isinstance(resolve_contract("check if json is malformed"), JsonInvalidContract)
    assert isinstance(resolve_contract("Write a python function add(a, b)"), PythonSyntaxContract)
    assert isinstance(resolve_contract("What is the capital of Australia?"), EvidenceRequiredContract)
    assert isinstance(resolve_contract("Verify file_exists 'data.json'"), FileExistsContract)
    assert isinstance(resolve_contract("Verify that 'config.json' contains 'env=prod'"), FileContentMatchContract)
    assert isinstance(resolve_contract("open-ended creative writing story"), NoVerificationAvailableContract)
