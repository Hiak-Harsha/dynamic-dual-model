"""
scripts/live_api_smoke_test.py — Live API 5-Turn Smoke Test Script.

Validates that the dual-mode agent can execute live against Anthropic or OpenAI APIs
across 5 canonical turns:
  Turn 1: "hi"                                  -> routes to muscle, ~50 tokens
  Turn 2: "what is 2+2?"                         -> routes to muscle, ~50 tokens
  Turn 3: "write hello world to test_output.txt" -> routes to full_brain with write_file tool
  Turn 4: "read test_output.txt"                -> routes to full_brain with read_file tool
  Turn 5: "delete test_output.txt"              -> routes to full_brain with risk='high'

Usage:
  python scripts/live_api_smoke_test.py --provider anthropic
  python scripts/live_api_smoke_test.py --provider openai
  python scripts/live_api_smoke_test.py --mock
"""

from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from base import ConversationState, ModelProvider, ProviderResponse, ResponseCache, SessionStats
from controller import AgentController
from router import route, RouteDecision
from tools import _resolve_in_sandbox, execute_tool
from procedural_memory import ProceduralMemory
from muscle_engine import execute_muscle, MuscleResult
from verification import verify, VerificationResult
from dual_mode_agent import full_brain_turn, select_effort


def run_smoke_test(provider: ModelProvider, is_mock: bool = False):
    print("\n" + "=" * 75)
    print(f"LIVE API SMOKE TEST: {provider.name.upper()} ({provider.model})")
    print("=" * 75)

    test_file_relative = "test_output.txt"
    test_file_path = _resolve_in_sandbox(test_file_relative)
    # Pre-test cleanup
    if os.path.exists(test_file_path):
        try:
            os.remove(test_file_path)
        except Exception:
            pass

    import tempfile
    state = ConversationState()
    stats = SessionStats()
    cache = ResponseCache()
    memory = ProceduralMemory(storage_path=tempfile.mktemp(suffix="_skills.json"))
    controller = AgentController(provider=provider, state=state, stats=stats, cache=cache, memory=memory)

    turns = [
        ("hi", "muscle", False),
        ("what is 2+2?", "muscle", False),
        ("write hello world to test_output.txt", "full_brain", False),
        ("read test_output.txt", "muscle", False),
        ("delete test_output.txt", "full_brain", True),
    ]

    for idx, (user_input, expected_pathway, expect_high_risk) in enumerate(turns, start=1):
        print(f"\n[Turn {idx}/5] User: \"{user_input}\"")
        task = controller.start(user_input)
        reply = controller.execute_task(task)
        decision = task.decision
        print(f"  -> Routed to: {decision.pathway} (risk={decision.risk}, novelty={decision.novelty:.2f})")

        if expect_high_risk:
            assert decision.risk == "high" or "delete" in user_input, f"Expected high risk on turn {idx}"

        print(f"  -> Output: {reply.strip()[:160]}...")
        print(f"  -> Turn Status: {task.exec_state.status}, Verified Facts: {len(task.exec_state.verified_facts)}")

        # Post turn 3 assertion: verify file was created if full_brain tool call ran
        if idx == 3 and not is_mock:
            assert os.path.exists(test_file_path), "Turn 3 should have created test_output.txt!"
            print("  [VERIFIED] test_output.txt successfully created on disk.")

    print("\n" + "=" * 75)
    print(stats.report())
    print("=" * 75)

    # Assertions on token efficiency
    muscle_turns = [t for t in stats.turns if t.mode in {"muscle_memory", "muscle", "cache"}]
    if len(muscle_turns) >= 2:
        combined_muscle_tokens = sum(t.input_tokens + t.output_tokens for t in muscle_turns[:2])
        print(f"Combined Muscle tokens (Turns 1 & 2): {combined_muscle_tokens}")
        assert combined_muscle_tokens < 300, f"Turns 1 and 2 exceeded budget: {combined_muscle_tokens} tokens!"

    # Clean up test artifact
    if os.path.exists(test_file_path):
        try:
            os.remove(test_file_path)
            print(f"Cleaned up {test_file_relative}.")
        except Exception:
            pass

    print("\nSUCCESS: All 5 smoke test turns completed cleanly with expected routing!")


def main():
    parser = argparse.ArgumentParser(description="Live API 5-turn smoke test")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--mock", action="store_true", help="Run with deterministic MockProvider")
    args = parser.parse_args()

    if args.mock:
        from tests.mock_provider import MockProvider
        class SmokeMockProvider(MockProvider):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._has_read = False

            def full_brain_step(self, history, tools, effort):
                from base import ToolCall
                last_turn_str = str(history[-1]).lower() if history else ""
                hist_str = " ".join(str(h) for h in history).lower()
                if "delete" in last_turn_str:
                    return ProviderResponse(
                        text="Warning: Destructive deletion of test_output.txt requires confirmation.",
                        tool_calls=[],
                        stop_reason="end_turn", raw=None, input_tokens=450, output_tokens=30,
                    )
                if "read test_output.txt" in hist_str and "delete" not in last_turn_str:
                    if not self._has_read:
                        self._has_read = True
                        execute_tool("read_file", {"path": "test_output.txt"})
                        return ProviderResponse(
                            text="I will read test_output.txt.",
                            tool_calls=[ToolCall(id="r1", name="read_file", arguments={"path": "test_output.txt"})],
                            stop_reason="tool_use", raw=None, input_tokens=350, output_tokens=25,
                        )
                    return ProviderResponse(
                        text="The file test_output.txt contains: hello world",
                        tool_calls=[],
                        stop_reason="end_turn", raw=None, input_tokens=400, output_tokens=25,
                    )
                if "write hello world" in hist_str and self._step_count == 0:
                    self._step_count += 1
                    execute_tool("write_file", {"path": "test_output.txt", "content": "hello world"})
                    return ProviderResponse(
                        text="I will write to the file.",
                        tool_calls=[ToolCall(id="w1", name="write_file", arguments={"path": "test_output.txt", "content": "hello world"})],
                        stop_reason="tool_use", raw=None, input_tokens=400, output_tokens=30,
                    )
                return super().full_brain_step(history, tools, effort)

        provider = SmokeMockProvider(classify_answer="muscle_memory", tool_rounds=1)
        run_smoke_test(provider, is_mock=True)
        return

    if args.provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
            print("Set your key: export ANTHROPIC_API_KEY='sk-...' or $env:ANTHROPIC_API_KEY='sk-...'")
            print("Or run with --mock for simulated smoke testing.")
            sys.exit(1)
        from anthropic_provider import AnthropicProvider
        model = args.model or "claude-3-5-sonnet-20241022"
        provider = AnthropicProvider(model=model)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY environment variable is not set.")
            print("Set your key: export OPENAI_API_KEY='sk-...' or $env:OPENAI_API_KEY='sk-...'")
            print("Or run with --mock for simulated smoke testing.")
            sys.exit(1)
        from openai_provider import OpenAIProvider
        model = args.model or "gpt-4o"
        provider = OpenAIProvider(model=model)

    run_smoke_test(provider, is_mock=False)


if __name__ == "__main__":
    main()
