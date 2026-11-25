"""
tests/test_tier3_uncertainty.py — Verification of Phase 3 Real Uncertainty Detection.

Validates that Tier 3 of muscle_engine.py actually emits status="handoff_required":
1. When context is required but no context is in the memory buffer (context contradiction).
2. When the response contains hedging or uncertainty markers ("I'm not sure", "it depends", etc.).
3. When a substantive request receives a suspiciously truncated response.
4. While clean conversational responses complete normally.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from base import ConversationState, ProviderResponse
from router import RouteDecision
from muscle_engine import execute_muscle, MuscleResult
from procedural_memory import ProceduralMemory
from mock_provider import MockProvider


def test_tier3_context_contradiction():
    """Test Condition 1: context_required is True, but no turns exist in buffer."""
    provider = MockProvider()
    conv_state = ConversationState()  # Empty buffer
    decision = RouteDecision(pathway="muscle", context_required=True, risk="low")
    memory = ProceduralMemory()

    res: MuscleResult = execute_muscle("What language do I like?", conv_state, decision, memory, provider)
    assert res.status == "handoff_required", f"Expected handoff_required, got {res.status}"
    assert res.confidence == 0.2
    assert "Context contradiction" in res.reason
    print("PASS: Tier 3 context contradiction correctly triggers handoff.")


def test_tier3_hedging_detection():
    """Test Condition 2: Model output contains hedging or lack of information."""
    hedging_replies = [
        "I'm not sure about that detail.",
        "I don't have enough information to answer accurately.",
        "It depends on the context of your codebase.",
        "Could you please clarify what you mean?",
    ]

    for reply in hedging_replies:
        provider = MockProvider()
        # Mock reply returns hedging phrase
        provider.muscle_memory_reply = lambda msg: ProviderResponse(
            text=reply, tool_calls=[], stop_reason="end_turn", raw=None, input_tokens=40, output_tokens=15
        )
        conv_state = ConversationState()
        decision = RouteDecision(pathway="muscle", context_required=False, risk="low")
        memory = ProceduralMemory()

        res: MuscleResult = execute_muscle("Is port 8080 open?", conv_state, decision, memory, provider)
        assert res.status == "handoff_required", f"Failed to catch hedging: '{reply}'"
        assert res.confidence == 0.35
        assert "hedging/uncertainty marker" in res.reason

    print("PASS: Tier 3 hedging language markers reliably detected and handed off.")


def test_tier3_truncation_detection():
    """Test Condition 3: Substantive request yields truncated response."""
    provider = MockProvider()
    provider.muscle_memory_reply = lambda msg: ProviderResponse(
        text="No idea",  # 2 words on an 'explain' task
        tool_calls=[], stop_reason="end_turn", raw=None, input_tokens=40, output_tokens=3
    )
    conv_state = ConversationState()
    decision = RouteDecision(pathway="muscle", context_required=False, risk="low")
    memory = ProceduralMemory()

    res: MuscleResult = execute_muscle("Explain the architecture in detail", conv_state, decision, memory, provider)
    assert res.status == "handoff_required"
    assert res.confidence == 0.4
    assert "truncated" in res.reason
    print("PASS: Tier 3 truncated substantive reply detected and handed off.")


def test_tier3_clean_completion():
    """Test Condition 4: Clean response completes normally."""
    provider = MockProvider()
    conv_state = ConversationState()
    decision = RouteDecision(pathway="muscle", context_required=False, risk="low")
    memory = ProceduralMemory()
    memory.skills.clear()

    res: MuscleResult = execute_muscle("Hello there!", conv_state, decision, memory, provider)
    assert res.status in {"completed", "completed_unverified"}
    assert res.is_completed
    assert res.confidence >= 0.7
    print("PASS: Tier 3 clean responses complete with high confidence.")


if __name__ == "__main__":
    test_tier3_context_contradiction()
    test_tier3_hedging_detection()
    test_tier3_truncation_detection()
    test_tier3_clean_completion()
    print("\nALL TIER 3 UNCERTAINTY TESTS PASSED!")
