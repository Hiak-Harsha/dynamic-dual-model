"""
tests/test_v5_handoff_strengthening.py — Tests for prioritized, budget-enforced continuation handoff.

Per V5 Specification & Master Implementation Prompt:
1. Handoff block contains in priority order:
   - User goal
   - Verification failure reason
   - Observations & facts already established
   - Files touched / tool results
   - Unresolved items
   - Explicit "continue from here" instruction
2. Enforces hard token budget (< 400 tokens).
3. Failed muscle action produces a handoff containing failure reason and target path.
4. Full Brain continuation receives handoff preamble and does not restart discovery.
"""

from __future__ import annotations
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from base import ConversationState, ProviderResponse, ToolCall, count_tokens
from execution_state import ExecutionState
from handoff import build_handoff, render_handoff_prompt, format_prioritized_handoff
from controller import AgentController
from mock_provider import MockProvider


def test_handoff_contains_all_prioritized_fields_in_order():
    """Validates that render_handoff_prompt formats fields in exact required priority order."""
    state = ExecutionState(objective="Process batch data files")
    state.add_verified_fact("Verified schema version 2.1", source="validation")
    state.observations.append("Found 3 input CSV files in directory")
    state.tool_state["file_written:data_out.csv"] = True
    state.tool_state["file_read:raw_input.csv"] = True
    state.mark_unresolved("Header validation on data_out.csv pending")
    state.errors.append("CSV row 42 had missing delimiter")

    handoff = build_handoff(
        state=state,
        reason="Verification failed: Schema mismatch on row 42",
        suggested_next_step="Inspect row 42 in raw_input.csv and correct delimiter",
    )

    prompt = render_handoff_prompt(handoff, max_tokens=400)

    # Assert all prioritized sections are present
    assert "Objective: Process batch data files" in prompt
    assert "Verified Facts" in prompt
    assert "Verified schema version 2.1" in prompt
    assert "Files Touched & Tool Results:" in prompt
    assert "file_written:data_out.csv: True" in prompt
    assert "Unresolved Items (ACTION REQUIRED):" in prompt
    assert "Header validation on data_out.csv pending" in prompt
    assert "Errors Encountered:" in prompt
    assert "CSV row 42 had missing delimiter" in prompt
    assert "CONTINUATION DIRECTIVE:" in prompt
    assert "Do not repeat verified completed work or re-execute successful tools" in prompt

    # Verify order of sections
    idx_goal = prompt.find("Objective: Process batch data files")
    idx_reason = prompt.find("Verification Failure Reason:")
    idx_facts = prompt.find("Verified Facts")
    idx_tools = prompt.find("Files Touched & Tool Results:")
    idx_unresolved = prompt.find("Unresolved Items")
    idx_directive = prompt.find("CONTINUATION DIRECTIVE:")

    assert idx_goal < idx_reason < idx_facts < idx_tools < idx_unresolved < idx_directive


def test_handoff_enforces_hard_token_budget():
    """Validates that a heavily bloated execution state is strictly truncated to < 400 tokens."""
    state = ExecutionState(objective="Massive data processing task with excessive log telemetry")
    # Add large quantities of telemetry
    for i in range(50):
        state.add_verified_fact(f"Verified invariant #{i}: computation node {i} initialized cleanly and returned exit code 0", source="test")
        state.observations.append(f"Observation #{i}: Network packet {i} received with sequence number {i * 1024} bytes from host")
        state.tool_state[f"file_written:output_chunk_{i}.bin"] = True
        state.unresolved.append(f"Unresolved partition chunk #{i} needs cross-validation check")

    handoff = build_handoff(
        state=state,
        reason="Verification failure: Memory limit exceeded on batch worker",
        suggested_next_step="Resume from partition 45",
    )

    rendered = render_handoff_prompt(handoff, max_tokens=400)
    tokens = count_tokens(rendered)

    # Must strictly satisfy hard token budget
    assert tokens <= 400, f"Rendered handoff exceeded 400 token cap: {tokens} tokens"
    # Essential headers and directives must never be dropped
    assert "Objective:" in rendered
    assert "Verification Failure Reason:" in rendered
    assert "CONTINUATION DIRECTIVE:" in rendered


def test_failed_muscle_produces_handoff_with_failure_and_target_path():
    """Validates that a failed muscle execution produces a continuation handoff
    containing the explicit failure reason and target file path."""
    state = ExecutionState(objective="Write configuration to config/system.json")
    state.tool_state["file_written:config/system.json"] = False
    state.errors.append("Permission denied: cannot write to config/system.json")
    state.mark_unresolved("Retry write with elevated permissions")

    handoff = build_handoff(
        state=state,
        reason="Level 2 execution verification failed: file config/system.json not created",
        suggested_next_step="Check directory permissions and retry write",
    )

    block = render_handoff_prompt(handoff, max_tokens=400)
    assert "config/system.json" in block
    assert "Level 2 execution verification failed" in block
    assert "Permission denied: cannot write to config/system.json" in block


def test_full_brain_continuation_receives_handoff_and_avoids_rediscovery():
    """Validates that Full Brain continuation receives the handoff preamble,
    has access to established facts, and proceeds without re-running verified work."""
    captured_histories = []
    provider = MockProvider()

    def step_hook(history, tools, effort):
        captured_histories.append(list(history))
        return ProviderResponse(
            text="Completed continuation successfully based on verified facts.",
            tool_calls=[],
            stop_reason="end_turn",
            raw=None,
            input_tokens=800,
            output_tokens=40,
        )

    provider.full_brain_step = step_hook
    ctrl = AgentController(provider=provider)

    # Setup task with prior established state
    task = ctrl.start("Continue processing invoice items")
    ctrl.route_task(task)

    # Simulate established facts from prior muscle steps
    task.exec_state.add_verified_fact("Invoice #9823 customer verified as Acme Corp", source="db")
    task.exec_state.tool_state["file_read:invoices/9823.json"] = True
    task.exec_state.mark_unresolved("Calculate sales tax for Acme Corp invoice")

    handoff = build_handoff(
        state=task.exec_state,
        reason="Muscle memory reached calculation step requiring tax rule validation",
    )

    ctrl.brain.continue_from_handoff(
        provider=ctrl.provider,
        state=ctrl.state,
        user_message=task.user_input,
        handoff=handoff,
        effort="medium",
    )

    # Verify that the continuation prompt was injected into Full Brain history
    assert len(captured_histories) > 0
    injected_user_prompt = captured_histories[0][-1]["content"]
    assert "[CONTINUATION HANDOFF FROM MUSCLE MEMORY]" in injected_user_prompt
    assert "Invoice #9823 customer verified as Acme Corp" in injected_user_prompt
    assert "file_read:invoices/9823.json: True" in injected_user_prompt
    assert "[CONTINUATION DIRECTIVE:" in injected_user_prompt
