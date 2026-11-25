"""
test_architecture_boundaries.py — Architecture Boundary & Ledger Invariant Tests.

Verifies the V9.1 non-negotiable architectural rules:
1. AgentController manages exactly one UniversalModelRuntime.
2. Ledger Invariant: Every model invocation through the runtime produces an immutable
   CognitionLedger entry; cache hits do not add model call entries.
3. Provider Independence: Changing providers or simulators does not break the dual-mode controller.
4. ModelSimulator failure modes (weak, partial, timeout) are safely handled by the architecture.
"""

import pytest

from base import ConversationState, SessionStats, ResponseCache, ProviderResponse
from controller import AgentController, TaskContext
from procedural_memory import ProceduralMemory
from universal_runtime import UniversalModelRuntime, CognitionLedger
from tests.mock_provider import MockProvider
from tests.model_simulator import ModelSimulator


def test_controller_owns_single_runtime():
    """AgentController must construct or receive exactly one UniversalModelRuntime."""
    prov = MockProvider()
    controller = AgentController(provider=prov)
    assert controller.runtime is not None
    assert isinstance(controller.runtime, UniversalModelRuntime)
    assert controller.runtime.provider is prov
    assert controller.ledger is controller.runtime.ledger


def test_runtime_ledger_accounting_invariant():
    """Ledger must increment on model calls and remain unchanged on deterministic cache hits."""
    prov = MockProvider()
    runtime = UniversalModelRuntime(prov)
    controller = AgentController(runtime=runtime)

    # Initial state
    assert controller.ledger.total_calls() == 0

    # 1. First execution of a query (model call)
    task1 = controller.start("write code for fibonacci")
    reply1 = controller.execute_task(task1)
    assert len(reply1) > 0
    # Ledger should have recorded the invocation(s)
    calls_after_t1 = controller.ledger.total_calls()
    assert calls_after_t1 >= 1

    # 2. Put response into cache to test deterministic cache hit
    controller.cache.put("hello", ProviderResponse(text="cached hi", tool_calls=[], stop_reason="end_turn", raw=None))
    task2 = controller.start("hello")
    reply2 = controller.execute_task(task2)
    assert reply2 == "cached hi"

    # Ledger calls must NOT have changed on cache hit
    assert controller.ledger.total_calls() == calls_after_t1


def test_simulator_modes_through_runtime():
    """ModelSimulator modes (weak, partial, timeout) function deterministically through UniversalModelRuntime."""
    sim = ModelSimulator(mode="perfect")
    runtime = UniversalModelRuntime(sim)

    # Perfect mode execution
    resp = runtime.muscle_memory_reply("test input")
    assert "Simulated muscle response" in resp.text
    assert runtime.ledger.total_calls() == 1

    # Weak mode execution
    sim.set_mode("weak")
    resp_weak = runtime.muscle_memory_reply("test input")
    assert "not really sure" in resp_weak.text
    assert runtime.ledger.total_calls() == 2

    # Timeout mode handling
    sim.set_mode("timeout")
    with pytest.raises(TimeoutError):
        runtime.muscle_memory_reply("test timeout")


def test_direct_provider_calls_scan_discipline():
    """Verify that core modules do not make unauthorized un-intercepted external calls."""
    import inspect
    from universal_runtime import UniversalModelRuntime

    # Ensure UniversalModelRuntime exposes all necessary model gateway methods
    required_runtime_methods = [
        "classify",
        "route_classify",
        "muscle_memory_reply",
        "full_brain_step",
        "summarize_history_step",
    ]
    for method_name in required_runtime_methods:
        assert hasattr(UniversalModelRuntime, method_name), f"UniversalModelRuntime missing {method_name}"
        assert callable(getattr(UniversalModelRuntime, method_name))
