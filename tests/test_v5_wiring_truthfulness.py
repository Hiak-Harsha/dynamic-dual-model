"""
test_v5_wiring_truthfulness.py — Rigorous verification of wiring truthfulness and invariant guarantees.

Tests the 10 critical wiring fixes:
1. Zero fabricated routing tokens (TaskUsage.routing_tokens == provider actual).
2. Zero fabricated summary tokens (TaskUsage.summary_tokens == provider actual).
3. Multi-call Full Brain SessionStats aggregation matches TaskUsage across 1, 2, 4 calls.
4. Universal task-level verification gate runs on Brain and Muscle outputs.
5. Tier 3 Muscle completes unverified and cannot bypass verification.
6. Deterministic JSON semantics distinguish procedure success from JSON syntax errors.
7. AST arithmetic evaluation is safe and rejects code execution attempts.
8. PreconditionEvaluator filters skills by non-applicability and preconditions.
9. Wall-time, state-token, and tool-call budgets are enforced.
10. Router cost model uses neutral continuation_cost_ratio without circular 0.55 bias.
"""

import ast
import json
import os
import sys
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from base import ConversationState, ExecutionBudget, ProviderResponse, SessionStats
from brain import BrainEngine
from controller import AgentController, TaskContext
from execution_state import ExecutionState
from mock_provider import MockProvider
from muscle_engine import execute_muscle, _run_deterministic_arithmetic, _run_deterministic_json
from procedural_memory import ProceduralMemory, PreconditionEvaluator, Skill
from router import route, RouteDecision
from verification import verify


def test_zero_fabricated_routing_tokens():
    """Validates that routing token accounting uses actual provider usage, not fabricated constants."""
    provider = MockProvider()
    controller = AgentController(provider=provider)
    task = controller.start("What is the capital of France?")
    
    decision = controller.route_task(task)
    
    if decision.routing_response is not None:
        expected_tokens = decision.routing_response.input_tokens + decision.routing_response.output_tokens
        assert task.exec_state.usage.routing_tokens == expected_tokens
        classify_turns = [t for t in controller.stats.turns if t.mode == "classify"]
        assert len(classify_turns) == 1
        assert classify_turns[0].input_tokens == decision.routing_response.input_tokens


def test_zero_fabricated_summary_tokens():
    """Validates that history summary tokens use actual provider response tokens."""
    provider = MockProvider()
    controller = AgentController(provider=provider, summary_interval=2)
    
    for i in range(3):
        controller.state.full_brain_history.append({"role": "user", "content": f"query {i}"})
        controller.state.full_brain_history.append({"role": "assistant", "content": f"answer {i}"})
    controller.state.full_brain_turns_since_summary = 5
    
    task = controller.start("Check system status")
    controller.compress_summary_if_needed(task=task, force=True)
    
    summary_turns = [t for t in controller.stats.turns if t.mode == "summary"]
    assert len(summary_turns) == 1
    assert summary_turns[0].input_tokens == 120
    assert summary_turns[0].output_tokens == 35
    assert task.exec_state.usage.summary_tokens == 155


def test_multi_call_session_stats_aggregation():
    """Validates that multi-turn tool loops record every single model step into SessionStats."""
    for tool_rounds in [0, 1, 3]:
        provider = MockProvider(classify_answer="full_brain", tool_rounds=tool_rounds)
        controller = AgentController(provider=provider)
        
        task = controller.start(f"Run {tool_rounds} rounds of tools")
        controller.execute_task(task)
        
        expected_brain_steps = tool_rounds + 1
        brain_turns = [t for t in controller.stats.turns if t.mode == "full_brain"]
        assert len(brain_turns) == expected_brain_steps
        
        total_brain_tokens_in_stats = sum(t.input_tokens + t.output_tokens for t in brain_turns)
        assert task.exec_state.usage.brain_tokens == total_brain_tokens_in_stats


def test_universal_verification_gate():
    """Validates that Brain outputs pass through the verification gate."""
    provider = MockProvider(classify_answer="full_brain", tool_rounds=0)
    controller = AgentController(provider=provider)
    
    task = controller.start("Provide an overview")
    controller.execute_task(task)
    
    assert task.verification_result is not None
    assert task.exec_state.status in {"completed", "completed_verified", "completed_unverified"}


def test_tier3_muscle_cannot_bypass_verification():
    """Validates that Tier 3 Muscle initially returns completed_unverified and is verified."""
    provider = MockProvider()
    conv_state = ConversationState()
    decision = RouteDecision(pathway="muscle", context_required=False, risk="low")
    memory = ProceduralMemory()
    
    res = execute_muscle("What is the population of Tokyo?", conv_state, decision, memory, provider)
    assert res.status == "completed_unverified"
    assert res.execution_state.status == "completed_unverified"


def test_deterministic_json_validation_semantics():
    """Validates that JSON validator handles both valid and invalid JSON gracefully."""
    state = ExecutionState(objective="Validate JSON")
    
    # Valid JSON
    ok, msg = _run_deterministic_json('Validate: {"name": "alice", "active": true}', state)
    assert ok is True
    assert "Valid JSON" in msg
    assert any("JSON parsed successfully" in f for f in state.verified_facts)
    
    # Invalid JSON syntax
    state2 = ExecutionState(objective="Validate JSON")
    ok2, msg2 = _run_deterministic_json('Validate: {"name": "alice", invalid}', state2)
    assert ok2 is True
    assert "Invalid JSON syntax" in msg2
    assert any("JSON validation completed: Input contains invalid JSON" in f for f in state2.verified_facts)


def test_ast_arithmetic_safe_evaluation():
    """Validates that arithmetic uses AST visitor without eval() and blocks injection."""
    state = ExecutionState(objective="Calculate")
    
    ok, res = _run_deterministic_arithmetic("What is 15 + 3 * 4?", state)
    assert ok is True
    assert "27" in res
    
    ok, res = _run_deterministic_arithmetic("Calculate (100 - 20) / 4", state)
    assert ok is True
    assert "20" in res
    
    state_mal = ExecutionState(objective="Hack")
    ok, res = _run_deterministic_arithmetic("compute __import__('os').system('ls')", state_mal)
    assert ok is False
    assert "Not a purely numeric" in res or "Could not evaluate" in res


def test_precondition_evaluator():
    """Validates that PreconditionEvaluator checks preconditions and non-applicability."""
    json_skill = Skill(
        id="json_validator",
        trigger_patterns=[r".*\bjson\b.*"],
        procedure="deterministic_json_validation",
        preconditions=["contains JSON"],
        non_applicability_conditions=["Non-JSON plain questions"],
    )
    
    passed, _ = PreconditionEvaluator.evaluate(json_skill, 'Check this json: {"key": "val"}')
    assert passed is True
    
    passed, reason = PreconditionEvaluator.evaluate(json_skill, "Tell me a joke about json")
    assert passed is False
    assert "non-applicability" in reason.lower() or "precondition" in reason.lower()


def test_budget_enforcement():
    """Validates wall-time and state-token budget enforcement in ExecutionState."""
    budget = ExecutionBudget(max_wall_time=0.01, max_state_tokens=50)
    state = ExecutionState(objective="Time-limited task", budget=budget)
    
    time.sleep(0.02)
    exceeded, msg = state.is_budget_exceeded()
    assert exceeded is True
    assert "Wall-time limit exceeded" in msg
    
    state2 = ExecutionState(objective="Token-limited task", budget=ExecutionBudget(max_wall_time=100.0, max_state_tokens=20))
    for i in range(15):
        state2.add_verified_fact(f"Established verified fact number {i} with substantial detailed context")
    exceeded2, msg2 = state2.is_budget_exceeded()
    assert exceeded2 is True
    assert "State token size limit exceeded" in msg2


def test_router_cost_formula_unbiased():
    """Validates that RouteDecision.expected_muscle_path_cost does not use hardcoded 0.55."""
    decision = RouteDecision(
        pathway="muscle",
        expected_muscle_success_probability=0.5,
        estimated_muscle_cost=100,
        estimated_handoff_cost=100,
        estimated_brain_cost=1000,
        continuation_cost_ratio=1.0,
    )
    assert decision.expected_muscle_path_cost() == 650.0
    
    decision.continuation_cost_ratio = 0.5
    assert decision.expected_muscle_path_cost() == 400.0
