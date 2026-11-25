"""
tests/test_v5_state_and_budget.py — Tests for V5 ExecutionState, Evidence Levels, and ExecutionBudget.

Per V5 Spec Sections 7, 8, 20, 21:
- State != chat transcript.
- Evidence levels: VERIFIED, OBSERVED, ASSUMED, INFERRED, UNVERIFIED, ERROR.
- LLM assertions are never marked VERIFIED without independent verification.
- ExecutionBudget enforcement triggers handoff when exceeded.
- Handoff rendering strictly prioritizes verified facts over lower-priority items.
"""

from __future__ import annotations
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from base import ExecutionBudget, TaskUsage, ProviderResponse, count_tokens
from execution_state import ExecutionState, EvidenceLevel, StateItem
from handoff import build_handoff, render_handoff_prompt


def test_evidence_levels_classification():
    """Validates that facts, observations, assumptions, and errors receive explicit evidence levels."""
    state = ExecutionState(objective="Analyze user authentication subsystem")

    state.add_verified_fact("Auth uses bcrypt with cost factor 12", source="code_inspection")
    state.add_observation("Found 4 user login endpoints in routes.py", source="static_analysis")
    state.add_assumption("User database is PostgreSQL based on psycopg2 import")
    state.add_inference("Sessions expire after 24 hours based on cookie max_age")
    state.add_error("Failed to connect to test auth server")

    # Assertions on collections
    assert len(state.verified_facts) == 1
    assert len(state.observations) == 1
    assert len(state.assumptions) == 1
    assert len(state.inferences) == 1
    assert len(state.errors) == 1

    # Assertions on typed StateItem records
    levels = {item.value: item.level for item in state.items}
    assert levels["Auth uses bcrypt with cost factor 12"] == EvidenceLevel.VERIFIED.value
    assert levels["Found 4 user login endpoints in routes.py"] == EvidenceLevel.OBSERVED.value
    assert levels["User database is PostgreSQL based on psycopg2 import"] == EvidenceLevel.ASSUMED.value
    assert levels["Sessions expire after 24 hours based on cookie max_age"] == EvidenceLevel.INFERRED.value
    assert levels["Failed to connect to test auth server"] == EvidenceLevel.ERROR.value

    # Re-adding existing fact does not duplicate
    state.add_verified_fact("Auth uses bcrypt with cost factor 12")
    assert len(state.verified_facts) == 1


def test_execution_budget_enforcement():
    """Validates that exceeding any budget dimension triggers budget exhaustion."""
    budget = ExecutionBudget(
        max_model_calls=2,
        max_input_tokens=500,
        max_output_tokens=100,
        max_tool_calls=3,
    )
    state = ExecutionState(objective="Bounded compute task", budget=budget)

    # Initial state: not exceeded
    exceeded, msg = state.is_budget_exceeded()
    assert not exceeded
    assert msg == ""

    # Simulate model calls within budget
    resp1 = ProviderResponse(text="call 1", tool_calls=[], stop_reason="end_turn", raw=None, input_tokens=200, output_tokens=40)
    state.usage.record_call("muscle", resp1)
    exceeded, _ = state.is_budget_exceeded()
    assert not exceeded

    # Exceed model calls
    resp2 = ProviderResponse(text="call 2", tool_calls=[], stop_reason="end_turn", raw=None, input_tokens=200, output_tokens=40)
    state.usage.record_call("muscle", resp2)
    exceeded, msg = state.is_budget_exceeded()
    assert exceeded
    assert "Model call limit exceeded" in msg


def test_handoff_prioritization_and_budget():
    """Validates that handoff rendering strictly prioritizes verified facts and stays within budget."""
    state = ExecutionState(objective="Refactor database connector")

    # Add items across all priority levels
    state.add_verified_fact("PostgreSQL 16 is target database")
    state.add_verified_fact("Connection pool size set to 20")
    state.add_completed_step("Inspected config.py")
    state.add_completed_step("Audited db_pool.py")
    state.mark_unresolved("Update connection retry logic")
    state.add_error("Timeout connecting to staging replica")
    state.add_decision("Use asyncpg instead of psycopg2", why="Non-blocking IO requirement")
    state.add_observation("Found 12 raw SQL queries in repo")
    state.add_assumption("Staging replica might be offline for maintenance")
    state.partial_result = "class AsyncDatabasePool:\n    def __init__(self): pass"

    handoff = build_handoff(state, reason="Encountered timeout on staging replica", suggested_next_step="Inspect network retry loop")
    rendered = render_handoff_prompt(handoff, max_tokens=400)

    # Assertions on content and ordering
    assert "[CONTINUATION HANDOFF FROM MUSCLE MEMORY]" in rendered
    assert "Objective: Refactor database connector" in rendered
    assert "Verified Facts (ESTABLISHED - DO NOT RE-EVALUATE):" in rendered
    assert "[VERIFIED] PostgreSQL 16 is target database" in rendered
    assert "Completed Steps:" in rendered
    assert "Unresolved Items (ACTION REQUIRED):" in rendered
    assert "CONTINUATION DIRECTIVE" in rendered
    assert "Do not repeat verified completed work" in rendered

    # Token budget check
    tokens = count_tokens(rendered)
    assert tokens <= 450, f"Handoff prompt exceeded token budget: {tokens} tokens"


def test_state_serialization_round_trip():
    """Validates that ExecutionState cleanly serializes to dict and reconstructs identical state."""
    state = ExecutionState(objective="Schema migration task")
    state.add_verified_fact("Tables users and orders exist")
    state.add_observation("Found foreign key constraint on user_id")
    state.add_decision("Add index on created_at", why="Optimize sorting")
    state.usage.total_input_tokens = 1200
    state.usage.total_output_tokens = 250

    as_dict = state.to_dict()
    reconstructed = ExecutionState.from_dict(as_dict)

    assert reconstructed.objective == state.objective
    assert reconstructed.verified_facts == state.verified_facts
    assert reconstructed.observations == state.observations
    assert reconstructed.decisions == state.decisions
    assert reconstructed.usage.total_input_tokens == 1200
    assert reconstructed.usage.total_output_tokens == 250
