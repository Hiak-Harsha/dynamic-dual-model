"""
tests/test_v5_accounting.py — Tests for Complete Multi-Call Token Accounting across All Sub-Phases.

Per V5 Spec Sections 26, 27, 28, 29:
- Mandatory complete accounting: Every model call across all iterations must contribute usage.
- Multi-iteration tool loops count all model calls, not just the final turn.
- TaskUsage tracks routing, muscle, brain, verification, and summary tokens separately.
- Zero-token response cache hits contribute 0 API tokens.
"""

from __future__ import annotations
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from base import ConversationState, ProviderResponse, ResponseCache, SessionStats, TaskUsage, ToolCall
from controller import AgentController
from mock_provider import MockProvider


def test_task_usage_records_all_subphases():
    """Validates that TaskUsage tracks routing, muscle, brain, verification, and summary tokens."""
    usage = TaskUsage(task_id="t123")
    assert usage.total_billable_tokens == 0
    assert usage.model_calls == 0

    # 1. Routing call
    r_resp = ProviderResponse(text="route", tool_calls=[], stop_reason="end_turn", raw=None, input_tokens=150, output_tokens=30)
    usage.record_call("routing", r_resp)
    assert usage.routing_tokens == 180
    assert usage.model_calls == 1

    # 2. Muscle call
    m_resp = ProviderResponse(text="muscle", tool_calls=[], stop_reason="end_turn", raw=None, input_tokens=60, output_tokens=25)
    usage.record_call("muscle", m_resp)
    assert usage.muscle_tokens == 85
    assert usage.model_calls == 2

    # 3. Brain call
    b_resp = ProviderResponse(text="brain", tool_calls=[], stop_reason="end_turn", raw=None, input_tokens=1200, output_tokens=80)
    usage.record_call("brain", b_resp)
    assert usage.brain_tokens == 1280
    assert usage.model_calls == 3

    # 4. Verification call
    v_resp = ProviderResponse(text="verify", tool_calls=[], stop_reason="end_turn", raw=None, input_tokens=70, output_tokens=15)
    usage.record_call("verification", v_resp)
    assert usage.verification_tokens == 85
    assert usage.model_calls == 4

    # 5. Summary call
    s_resp = ProviderResponse(text="summary", tool_calls=[], stop_reason="end_turn", raw=None, input_tokens=500, output_tokens=60)
    usage.record_call("summary", s_resp)
    assert usage.summary_tokens == 560
    assert usage.model_calls == 5

    total_expected = 180 + 85 + 1280 + 85 + 560
    assert usage.total_billable_tokens == total_expected
    assert usage.total_input_tokens == (150 + 60 + 1200 + 70 + 500)
    assert usage.total_output_tokens == (30 + 25 + 80 + 15 + 60)


def test_multi_call_tool_loop_accounting():
    """Validates that a multi-call tool loop counts every model iteration."""
    # MockProvider configured for 3 tool rounds (i.e. 4 total model calls: 3 tool use + 1 final)
    provider = MockProvider(classify_answer="full_brain", tool_rounds=3)
    controller = AgentController(provider=provider)

    task = controller.start("Investigate system logs and compile report")
    controller.execute_task(task)

    # 4 calls in tool loop
    assert task.exec_state.usage.model_calls >= 4
    # Each mock brain step produces 1200 in + ~40/80 out
    assert task.exec_state.usage.brain_tokens >= 4000
    assert task.exec_state.usage.tool_calls == 3


def test_cache_hit_zero_billable_tokens():
    """Validates that a ResponseCache hit contributes zero billable tokens to the task."""
    provider = MockProvider()
    cache = ResponseCache()
    cached_resp = ProviderResponse(text="hello!", tool_calls=[], stop_reason="end_turn", raw=None, input_tokens=50, output_tokens=10)
    cache.put("hello", cached_resp)

    controller = AgentController(provider=provider, cache=cache)
    task = controller.start("hello")
    reply = controller.execute_task(task)

    assert reply == "hello!"
    assert task.exec_state.usage.model_calls == 0
    assert task.exec_state.usage.total_billable_tokens == 0
    assert task.skill_used == "cache_tier_0"
