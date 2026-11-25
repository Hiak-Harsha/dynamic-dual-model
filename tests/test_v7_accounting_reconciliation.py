"""
test_v7_accounting_reconciliation.py
Tests authoritative token accounting, model routing recording, zero-billable cache hits,
and session reconciliation per V7 Phase 2 and Phase 3.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from base import ProviderResponse, ResponseCache, SessionStats, TurnStats, CachedResponseMetadata
from controller import AgentController
from mock_provider import MockProvider


def test_model_routing_recorded_in_task_usage_and_session_stats():
    """Validates that a forced model routing call (e.g. 150 in + 30 out)
    is recorded into TaskUsage.routing_tokens, model_calls, and SessionStats."""
    # MockProvider configured to return a routing classifier response of 150 in + 30 out
    provider = MockProvider(
        classify_answer="full_brain",
        classify_input_tokens=150,
        classify_output_tokens=30,
    )
    controller = AgentController(provider=provider)

    task = controller.start("Can you explain the detailed design of our microservice architecture?")
    decision = controller.route_task(task)

    assert decision.routing_response is not None
    assert task.exec_state.usage.routing_tokens == 180
    assert task.exec_state.usage.model_calls == 1
    assert task.exec_state.usage.total_input_tokens == 150
    assert task.exec_state.usage.total_output_tokens == 30
    assert task.exec_state.usage.total_billable_tokens == 180

    # Verify SessionStats contains exactly one classifier call
    classify_turns = [t for t in controller.stats.turns if t.mode == "classify"]
    assert len(classify_turns) == 1
    assert classify_turns[0].input_tokens == 150
    assert classify_turns[0].output_tokens == 30


def test_cache_hit_zero_billable_tokens_in_session_stats():
    """Validates that a cached response with historical 50+20 tokens
    contributes ZERO current billable tokens to TaskUsage and SessionStats."""
    provider = MockProvider()
    cache = ResponseCache()
    cached_resp = ProviderResponse(
        text="cached answer",
        tool_calls=[],
        stop_reason="end_turn",
        raw=None,
        input_tokens=50,
        output_tokens=20,
    )
    cache.put("hello", cached_resp)

    controller = AgentController(provider=provider, cache=cache)
    task = controller.start("hello")
    reply = controller.execute_task(task)

    assert reply == "cached answer"
    # TaskUsage verification
    assert task.exec_state.usage.model_calls == 0
    assert task.exec_state.usage.total_input_tokens == 0
    assert task.exec_state.usage.total_output_tokens == 0
    assert task.exec_state.usage.total_billable_tokens == 0
    assert task.exec_state.usage.cache_hit is True

    # SessionStats verification
    assert len(controller.stats.turns) == 1
    cache_turn = controller.stats.turns[0]
    assert cache_turn.mode == "cache"
    assert cache_turn.cached_response is True
    assert cache_turn.input_tokens == 0
    assert cache_turn.output_tokens == 0
    assert controller.stats.total_tokens_consumed() == 0

    # Historical metadata preserved separately
    assert cache_turn.cached_metadata is not None
    assert cache_turn.cached_metadata.original_input_tokens == 50
    assert cache_turn.cached_metadata.original_output_tokens == 20


def test_session_stats_reconciles_with_task_usage():
    """Validates that for any executed task, SessionStats tokens match TaskUsage actual calls."""
    provider = MockProvider(classify_answer="full_brain", tool_rounds=0)
    controller = AgentController(provider=provider)

    task = controller.start("Summarize the system telemetry")
    controller.execute_task(task)

    # Sum of TaskUsage model calls and tokens must equal non-cache SessionStats
    total_task_tokens = task.exec_state.usage.total_billable_tokens
    total_session_tokens = controller.stats.total_tokens_consumed()
    assert total_task_tokens == total_session_tokens
