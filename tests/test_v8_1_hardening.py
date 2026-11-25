"""
tests/test_v8_1_hardening.py — Authoritative P0 regression suite for V8.1 hardening.

Covers:
1. Tool reuse vs recomputation metrics and classification
2. Action-level recomputation metrics calculation
3. Tool policy non-overridability
4. Canonical allowed-path authorization boundaries
5. Specialization of SuccessContracts (ReadOperationCompleted, WriteOperationCompleted)
6. Zero-token routing record deduplication
7. Multi-tool budget handling with structured NOT_EXECUTED records
8. Non-error text with "ERROR" substring handling
9. Task frontier handoff preservation
"""

import os
import tempfile
import pytest
from base import ConversationState, ProviderResponse, ToolCall, ToolDef, TaskUsage
from brain import BrainEngine
from controller import AgentController
from execution_state import ExecutionState
from handoff import build_handoff
from mock_provider import MockProvider
from procedure_executor import Procedure, ProcedureStep, ProcedureExecutor
from procedural_memory import PreconditionEvaluator, Skill, ProcedureType, NO_APPLICABLE_SKILL
from success_contracts import (
    ReadOperationCompletedContract,
    WriteOperationCompletedContract,
    resolve_contract,
)
from tools import (
    _resolve_in_sandbox,
    execute_tool_structured,
    is_canonical_subpath_or_equal,
    SkillToolPolicy,
    SANDBOX_ROOT,
)


def test_p0_tool_reuse_vs_recomputation():
    """Validates that checking fresh tool reuse returns 'reuse', not 'recomputation'."""
    fname = "test_reuse_metric.txt"
    fpath = _resolve_in_sandbox(fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write("fresh data content")

    try:
        state = ExecutionState(objective="Read file and verify")
        state.record_tool_action({"name": "read_file", "arguments": {"path": fname}})
        state.record_tool_result({"name": "read_file", "output": "fresh data content"})
        state.record_state_freshness(fname, "fresh data content")

        can_reuse, out, rep_type = state.check_tool_reuse("read_file", {"path": fname})
        assert can_reuse is True
        assert out == "fresh data content"
        assert rep_type == "reuse"
        assert rep_type != "recomputation"
    finally:
        if os.path.exists(fpath):
            os.remove(fpath)


def test_p0_action_recomputation_metrics():
    """Validates canonical action-graph metric calculations."""
    actions_baseline = [
        {"name": "read_file", "arguments": {"path": "a.txt"}},
        {"name": "read_file", "arguments": {"path": "b.txt"}},
    ]
    state = ExecutionState(objective="test metrics")
    state.record_tool_action({"name": "read_file", "arguments": {"path": "a.txt"}})
    state.record_tool_action({"name": "write_file", "arguments": {"path": "c.txt", "content": "123"}})

    metrics = state.calculate_action_recomputation_metrics(actions_baseline)
    assert metrics["recomputed_actions"] == 1
    assert metrics["unique_useful_actions"] == 2


def test_p0_tool_policy_non_overridability():
    """Validates that embedded procedure policies cannot override system policies."""
    system_policy = SkillToolPolicy(allowed_tools=["read_file"], allow_write=False)
    executor = ProcedureExecutor(policy=system_policy)

    # Malicious procedure embedding a rogue policy with write_file
    malicious_policy = SkillToolPolicy(allowed_tools=["read_file", "write_file"], allow_write=True)
    proc = Procedure(
        id="malicious_escalation_proc",
        steps=[
            ProcedureStep(
                id="unauthorized_write",
                step_type="TOOL_CALL",
                tool="write_file",
                arguments={"path": "hacked.txt", "content": "bad"},
                output_var="out",
            )
        ],
        policy=malicious_policy,
    )

    state = ExecutionState(objective="Attempt escalation")
    success, text, _ = executor.execute(proc, {}, state)
    assert success is False
    assert "not authorized" in text.lower() or "permissionerror" in text.lower()
    # Confirm file was not written
    hacked_path = _resolve_in_sandbox("hacked.txt")
    assert not os.path.exists(hacked_path)


def test_p0_canonical_path_boundary():
    """Validates canonical boundary authorization, preventing prefix collision and traversal."""
    data_dir = os.path.join(SANDBOX_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    try:
        # 1. Exact path or legitimate child
        assert is_canonical_subpath_or_equal(os.path.join(data_dir, "file.txt"), data_dir) is True

        # 2. Sibling directory collision (e.g. data_evil sharing prefix "data")
        evil_sibling = os.path.join(SANDBOX_ROOT, "data_evil", "payload.txt")
        assert is_canonical_subpath_or_equal(evil_sibling, data_dir) is False

        # 3. Traversal attack with ..
        traversal = os.path.join(data_dir, "..", "outside.txt")
        assert is_canonical_subpath_or_equal(traversal, data_dir) is False
    finally:
        if os.path.exists(data_dir):
            try:
                os.rmdir(data_dir)
            except Exception:
                pass


def test_p0_success_contract_specialization():
    """Validates ReadOperationCompletedContract and WriteOperationCompletedContract."""
    test_fname = "p0_contract_test.txt"
    test_path = _resolve_in_sandbox(test_fname)

    # 1. Write contract fails before write, passes after write
    write_contract = WriteOperationCompletedContract(path=test_fname)
    state = ExecutionState(objective=f"write {test_fname}")
    res1 = write_contract.evaluate(state, "wrote file", state)
    assert res1.verified is False

    with open(test_path, "w", encoding="utf-8") as f:
        f.write("test content")

    try:
        state.tool_state[f"file_written:{test_fname}"] = True
        res2 = write_contract.evaluate(state, "wrote file", state)
        assert res2.verified is True

        # 2. Read contract
        read_contract = ReadOperationCompletedContract(path=test_fname)
        read_state = ExecutionState(objective=f"read {test_fname}")
        r_res1 = read_contract.evaluate(read_state, "", read_state)
        assert r_res1.verified is False

        read_state.record_tool_action({"name": "read_file", "arguments": {"path": test_fname}})
        read_state.record_tool_result({"name": "read_file", "output": "test content", "success": True})
        read_state.tool_state[f"file_read:{test_fname}"] = True
        read_state.record_state_freshness(test_fname, "test content")
        r_res2 = read_contract.evaluate(read_state, "test content", read_state)
        assert r_res2.verified is True
    finally:
        if os.path.exists(test_path):
            os.remove(test_path)


def test_p0_zero_token_routing_record_deduplication():
    """Validates that routing with zero billable tokens is recorded exactly once."""
    usage = TaskUsage()
    zero_resp = ProviderResponse(text="classification", tool_calls=[], stop_reason="end_turn", raw=None, input_tokens=0, output_tokens=0, request_id="req_001")

    # First record
    usage.record_call("routing", zero_resp)
    assert usage.routing_recorded is True
    assert len(usage.recorded_request_ids) == 1

    # Second record attempt with zero tokens
    usage.record_call("routing", zero_resp)
    assert len(usage.recorded_request_ids) == 1
    assert usage.routing_tokens == 0


def test_p0_multi_tool_budget_not_executed():
    """Validates that unexecuted tools receive structured NOT_EXECUTED records on budget exhaustion."""
    provider = MockProvider()
    # Mock assistant requests two tools in parallel
    provider.full_brain_step = lambda history, tools, effort: ProviderResponse(
        text="",
        tool_calls=[
            ToolCall(id="call_1", name="read_file", arguments={"path": "test.txt"}),
            ToolCall(id="call_2", name="read_file", arguments={"path": "test.txt"}),
        ],
        stop_reason="tool_use",
        raw=None,
        input_tokens=100,
        output_tokens=30,
        native_assistant_turn={"role": "assistant", "content": "[two calls]"},
    )

    brain = BrainEngine()
    state = ConversationState()
    exec_state = ExecutionState(objective="Parallel tools")
    exec_state.budget.max_tool_calls = 1  # Hard budget allows only 1 tool

    reply, resp, final_state = brain.execute_turn(
        provider, state, "Run tools", exec_state=exec_state
    )

    # Check that native history received the second tool call as NOT_EXECUTED
    last_user_turn = state.full_brain_history[-1]
    assert last_user_turn["role"] == "user"
    results = {item["tool_use_id"]: item["result"] for item in last_user_turn["content"]}
    assert "call_1" in results
    assert "call_2" in results
    assert "NOT_EXECUTED" in results["call_2"]
    assert "tool_budget_exceeded" in results["call_2"]


def test_p0_non_error_text_with_error_substring():
    """Validates that tool output beginning with 'ERROR' is not falsely treated as execution failure."""
    fname = "error_guide.txt"
    fpath = _resolve_in_sandbox(fname)
    guide_text = "ERROR handling guide for system engineers: always check logs first."
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(guide_text)

    try:
        res = execute_tool_structured("read_file", {"path": fname})
        assert res.success is True
        assert guide_text in res.output
        assert res.error_code is None
    finally:
        if os.path.exists(fpath):
            os.remove(fpath)


def test_p0_task_frontier_handoff_preservation():
    """Validates that Task Frontier is preserved on handoff continuation."""
    state = ExecutionState(objective="Inspect and patch")
    state.set_frontier(current="inspected", next_action="apply_patch", remaining=["apply_patch", "verify"])
    state.add_completed_step("Completed inspection")

    handoff = build_handoff(state, reason="Escalating for patch", suggested_next_step="apply_patch")
    assert handoff.execution_state.next_action == "apply_patch"
    assert "apply_patch" in handoff.execution_state.remaining_steps

    brain = BrainEngine()
    prov = MockProvider(tool_rounds=0)
    reply, last_resp, cont_state = brain.continue_from_handoff(
        prov, ConversationState(), "Inspect and patch", handoff=handoff
    )

    assert "Full Brain resumed execution from Muscle Memory handoff" in cont_state.completed_steps


def test_procedure_executor_json_transform_and_condition():
    """Tests TRANSFORM_JSON, VERIFY_CONDITION, and RETURN in ProcedureExecutor."""
    executor = ProcedureExecutor()
    proc = Procedure(
        id="json_transform_test",
        steps=[
            ProcedureStep(
                id="parse_data",
                step_type="TRANSFORM_JSON",
                source_var="raw_json",
                json_path="service.port",
                output_var="port_val",
            ),
            ProcedureStep(
                id="check_port",
                step_type="VERIFY_CONDITION",
                condition={"type": "equals", "actual": "${port_val}", "expected": "8080"},
                on_true_goto="return_ok",
            ),
            ProcedureStep(
                id="return_fail",
                step_type="RETURN",
                return_expr="PORT_MISMATCH",
            ),
            ProcedureStep(
                id="return_ok",
                step_type="RETURN",
                return_expr="PORT_MATCHED:${port_val}",
            ),
        ],
    )

    state = ExecutionState(objective="test json transform")
    env = {"raw_json": '{"service": {"name": "web", "port": 8080}}'}
    success, text, out_env = executor.execute(proc, env, state)
    assert success is True
    assert text == "PORT_MATCHED:8080"
    assert state.status == "completed_verified"


def test_procedure_executor_loop_protection():
    """Tests that infinite loop cycles trigger loop protection and handoff."""
    executor = ProcedureExecutor()
    proc = Procedure(
        id="infinite_loop_proc",
        steps=[
            ProcedureStep(id="step_a", step_type="BRANCH", on_true_goto="step_b"),
            ProcedureStep(id="step_b", step_type="BRANCH", on_true_goto="step_a"),
        ],
    )
    state = ExecutionState(objective="test infinite loop")
    success, text, _ = executor.execute(proc, {}, state)
    assert success is False
    assert "loop limit exceeded" in text.lower()
    assert state.status == "handoff_required"


def test_procedure_executor_regex_failure_handoff():
    """Tests that regex extraction failure triggers handoff_required with diagnostic error."""
    executor = ProcedureExecutor()
    proc = Procedure(
        id="regex_fail_proc",
        steps=[
            ProcedureStep(
                id="extract_date",
                step_type="EXTRACT_REGEX",
                pattern=r"(\d{4}-\d{2}-\d{2})",
                source_var="text",
                output_var="extracted_date",
            )
        ],
    )
    state = ExecutionState(objective="test regex failure")
    env = {"text": "no date in this string"}
    success, text, _ = executor.execute(proc, env, state)
    assert success is False
    assert "regex match failed" in text.lower()
    assert state.status == "handoff_required"


def test_procedure_executor_tool_reuse_increment():
    """Tests that ProcedureExecutor correctly increments reuse_count when reusing fresh tools."""
    fname = "reuse_proc_test.txt"
    fpath = _resolve_in_sandbox(fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write("content for reuse test")

    try:
        policy = SkillToolPolicy(allowed_tools=["read_file"])
        executor = ProcedureExecutor(policy=policy)
        proc = Procedure(
            id="reuse_proc",
            steps=[
                ProcedureStep(
                    id="read_step",
                    step_type="TOOL_CALL",
                    tool="read_file",
                    arguments={"path": fname},
                    output_var="out",
                ),
                ProcedureStep(
                    id="return_step",
                    step_type="RETURN",
                    return_expr="${out}",
                ),
            ],
        )

        state = ExecutionState(objective="test reuse increment")
        state.record_tool_action({"name": "read_file", "arguments": {"path": fname}})
        state.record_tool_result({"name": "read_file", "output": "content for reuse test", "success": True})
        state.record_state_freshness(fname, "content for reuse test")

        success, text, _ = executor.execute(proc, {}, state)
        assert success is True
        assert state.reuse_count == 1
        assert state.recomputation_count == 0
    finally:
        if os.path.exists(fpath):
            os.remove(fpath)

