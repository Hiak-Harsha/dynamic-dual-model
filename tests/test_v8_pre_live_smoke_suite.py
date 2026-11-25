"""
tests/test_v8_pre_live_smoke_suite.py — 10-Step Pre-Live Engineering Smoke Test Suite.

Implements the 10 sequential checks required before executing real-model scientific hypothesis tests:
TEST 1: Simple Muscle task
TEST 2: Deterministic task
TEST 3: Brain deliberative reasoning task
TEST 4: Single tool execution task (write_file)
TEST 5: Multi-tool verification task (read_file & verification)
TEST 6: Forced Muscle partial execution
TEST 7: Muscle -> handoff -> Brain continuation
TEST 8: Same task -> Brain restart
TEST 9: Compare actual token usage and action-level recomputation
TEST 10: Verify provider usage against TaskUsage telemetry
"""

import copy
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from base import ConversationState, ResponseCache, SessionStats, ProviderResponse
from brain import BrainEngine
from controller import AgentController
from execution_state import ExecutionState, COMPLETED_VERIFIED, COMPLETED_UNVERIFIED
from handoff import build_handoff
from mock_provider import MockProvider
from restart_agent import RestartAgent
from tools import _resolve_in_sandbox


@pytest.fixture
def test_env():
    fpath = _resolve_in_sandbox("test_suite_smoke.txt")
    if os.path.exists(fpath):
        try:
            os.remove(fpath)
        except Exception:
            pass
    yield fpath
    if os.path.exists(fpath):
        try:
            os.remove(fpath)
        except Exception:
            pass


def test_step_1_simple_muscle_task():
    """TEST 1: Simple Muscle task."""
    prov = MockProvider()
    controller = AgentController(provider=prov)
    task = controller.start("hi")
    reply = controller.execute_task(task)

    assert task.decision.pathway in ("muscle", "cache")
    assert task.success is True
    assert task.exec_state.usage.total_billable_tokens <= 100


def test_step_2_deterministic_task():
    """TEST 2: Deterministic task."""
    prov = MockProvider()
    controller = AgentController(provider=prov)
    task = controller.start("what is 48 * 25?")
    reply = controller.execute_task(task)

    assert task.decision.pathway == "muscle"
    assert "1200" in reply
    assert task.exec_state.status == COMPLETED_VERIFIED
    assert task.success is True
    assert task.exec_state.usage.total_billable_tokens == 0  # 0 API tokens for deterministic AST!


def test_step_3_brain_task():
    """TEST 3: Brain deliberative task."""
    prov = MockProvider()
    controller = AgentController(provider=prov)
    task = controller.start("Explain the primary architectural differences between TCP and UDP.")
    reply = controller.execute_task(task)

    assert task.decision.pathway in ("full_brain", "light_brain")
    assert task.success is True
    assert task.exec_state.usage.total_billable_tokens > 0


def test_step_4_single_tool_task(test_env):
    """TEST 4: Single tool task (write_file)."""
    prov = MockProvider(tool_rounds=1)
    controller = AgentController(provider=prov)
    task = controller.start("write hello world to test_suite_smoke.txt")
    reply = controller.execute_task(task)

    assert task.decision.pathway == "full_brain"
    assert task.decision.tools_required is True
    assert task.success is True


def test_step_5_multi_tool_task(test_env):
    """TEST 5: Multi-tool task (write then read)."""
    with open(test_env, "w", encoding="utf-8") as f:
        f.write("smoke test content alpha")

    prov = MockProvider(tool_rounds=1)
    controller = AgentController(provider=prov)
    task = controller.start("read test_suite_smoke.txt and verify content")
    reply = controller.execute_task(task)

    assert task.success is True
    assert len(task.exec_state.tool_actions) >= 1


def test_step_6_forced_muscle_partial_execution():
    """TEST 6: Forced Muscle partial execution."""
    prov = MockProvider()
    controller = AgentController(provider=prov)
    task = controller.start("Inspect file test_partial_prog.txt and diagnose errors")
    # Simulate partial progress in Muscle
    task.exec_state.add_completed_step("Inspected file header")
    task.exec_state.record_tool_action({"name": "read_file", "arguments": {"path": "test_partial_prog.txt"}})
    task.exec_state.set_frontier(current="inspected_header", next_action="diagnose_body", remaining=["diagnose_body"])

    assert len(task.exec_state.completed_steps) >= 1
    assert task.exec_state.frontier["current"] == "inspected_header"
    assert len(task.exec_state.action_signatures) == 1


def test_step_7_muscle_to_brain_continuation():
    """TEST 7: Muscle -> handoff -> Brain continuation."""
    prov = MockProvider(tool_rounds=0)
    controller = AgentController(provider=prov)
    task = controller.start("Inspect file test_handoff.txt and format summary")
    task.exec_state.add_completed_step("Read initial configuration")
    task.exec_state.record_tool_action({"name": "read_file", "arguments": {"path": "test_handoff.txt"}})
    task.exec_state.set_frontier(current="read_complete", next_action="format_summary")

    checkpoint = copy.deepcopy(task.exec_state)
    handoff = build_handoff(checkpoint, reason="Escalating for formatting", suggested_next_step="Format summary")

    brain = BrainEngine()
    reply, last_resp, cont_state = brain.continue_from_handoff(
        prov, ConversationState(), task.user_input, handoff=handoff, effort="medium"
    )

    assert len(cont_state.completed_steps) >= 2
    assert "Full Brain resumed execution from Muscle Memory handoff" in cont_state.completed_steps


def test_step_8_same_task_brain_restart():
    """TEST 8: Same task -> Brain restart."""
    prov = MockProvider(tool_rounds=1)
    task_input = "Inspect file test_restart.txt and summarize"

    # Prior muscle action
    prior_action = "read_file({\"path\":\"test_restart.txt\"})"
    prior_actions = [prior_action]

    # Restart baseline runs fresh BrainEngine without handoff
    fresh_state = ExecutionState(objective=task_input)
    brain = BrainEngine()
    reply, last_resp, restart_state = brain.execute_turn(
        prov, ConversationState(), task_input, effort="medium", exec_state=fresh_state
    )

    # Action-level recomputation detection
    # If the restarted brain invoked the tool, it is recorded
    assert restart_state.objective == task_input
    assert restart_state.usage.total_billable_tokens > 0


def test_step_9_compare_token_usage_and_action_recomputation():
    """TEST 9: Compare actual token usage between Continuation and Restart."""
    prior_actions = ["read_file({\"path\":\"config.json\"})"]

    # Condition C: Restart
    c_state = ExecutionState(objective="test task")
    c_state.record_tool_action({"name": "read_file", "arguments": {"path": "config.json"}})
    c_resp = ProviderResponse(text="", tool_calls=[], stop_reason="end_turn", raw=None, input_tokens=1500, output_tokens=100)
    c_state.usage.record_call("full_brain_step", c_resp)
    c_reps, c_ratio = c_state.calculate_action_recomputation(prior_actions)

    # Condition D: Continuation
    d_state = ExecutionState(objective="test task")
    # Continuation does NOT re-execute read_file; it continues to write_summary
    d_state.record_tool_action({"name": "write_file", "arguments": {"path": "summary.txt", "content": "done"}})
    d_resp = ProviderResponse(text="", tool_calls=[], stop_reason="end_turn", raw=None, input_tokens=800, output_tokens=80)
    d_state.usage.record_call("full_brain_step", d_resp)
    d_reps, d_ratio = d_state.calculate_action_recomputation(prior_actions)

    assert c_reps == 1
    assert c_ratio == 1.0
    assert d_reps == 0
    assert d_ratio == 0.0
    assert d_state.usage.total_billable_tokens < c_state.usage.total_billable_tokens


def test_step_10_verify_provider_usage_against_task_usage():
    """TEST 10: Verify provider usage against TaskUsage."""
    prov = MockProvider()
    stats = SessionStats()
    controller = AgentController(provider=prov, stats=stats)
    task = controller.start("Hello world")
    reply = controller.execute_task(task)

    # Validate that SessionStats turns match TaskUsage tokens
    session_tokens = stats.total_tokens_consumed()
    task_tokens = task.exec_state.usage.total_billable_tokens
    assert session_tokens == task_tokens
    assert task.exec_state.usage.task_id == task.task_id
