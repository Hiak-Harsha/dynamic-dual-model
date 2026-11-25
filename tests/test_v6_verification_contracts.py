"""
tests/test_v6_verification_contracts.py — Tests for multi-level verification contracts.

Per V6 Spec Sections AL, AM, AN, AO, AP:
- Level 0: Deterministic Invariant (non-empty, bounds)
- Level 1: Structural verification (JSON, Python AST)
- Level 2: Execution verification (sandbox files exist, hash verified, no errors)
- Level 3: Evidence & consistency verification:
    Strictly establishes 'no detected contradiction'.
    MUST NOT create a verified fact claiming the answer is correct!
- Level 4: Model-based verification
"""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from execution_state import ExecutionState
from tools import _resolve_in_sandbox
from verification import verify, VerificationResult


def test_level_0_deterministic_invariant():
    state = ExecutionState(objective="1 + 1")
    res = verify("2", "1 + 1", state, verification_mode="deterministic")
    assert res.passed is True
    assert res.tier_used == "deterministic"
    assert len(state.verified_facts) == 1

    empty_state = ExecutionState(objective="empty")
    res_empty = verify("", "empty", empty_state, verification_mode="deterministic")
    assert res_empty.passed is False


def test_level_1_structural_verification():
    state = ExecutionState(objective="parse json")
    res = verify('{"status": 200, "ok": true}', "parse json", state, verification_mode="structural")
    assert res.passed is True
    assert res.tier_used == "structural"

    broken_state = ExecutionState(objective="bad json")
    res_bad = verify('{"status": 200,', "bad json", broken_state, verification_mode="structural")
    assert res_bad.passed is False


def test_level_2_execution_verification():
    fname = "verif_exec_test.txt"
    fpath = _resolve_in_sandbox(fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write("test content")

    try:
        state = ExecutionState(objective="write file")
        state.tool_state[f"file_written:{fname}"] = True
        res = verify("Wrote file successfully", "write file", state, verification_mode="execution")
        assert res.passed is True
        assert res.tier_used == "execution"
    finally:
        if os.path.exists(fpath):
            os.remove(fpath)


def test_level_3_consistency_does_not_manufacture_fact():
    state = ExecutionState(objective="factual claim")
    state.add_observation("User stated favorite fruit is mango")

    # Result has no contradictions
    res = verify("Your favorite fruit is mango.", "factual claim", state, verification_mode="consistency")
    assert res.passed is True
    # Per V6 Section AN: Must NOT claim 'answer is verified correct' as a verified fact
    for vf in state.verified_facts:
        assert "consistent" not in vf.lower()
    # It must be recorded in observations
    assert any("zero contradictions" in obs.lower() for obs in state.observations)
