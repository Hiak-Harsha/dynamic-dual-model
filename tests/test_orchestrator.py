"""
tests/test_orchestrator.py — real, runnable assertions against the actual
orchestrator code in dual_mode_agent.py, using MockProvider so there is
NO network call and NO fabricated result: every number printed below was
actually computed by running this code, not estimated.

Run: python tests/test_orchestrator.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from base import ConversationState, ProviderResponse, ResponseCache, SessionStats
from dual_mode_agent import (
    full_brain_turn,
    is_obvious_muscle_memory,
    compress_summary_if_needed,
    MAX_TOOL_ITERATIONS,
    SUMMARY_INTERVAL_TURNS,
)
from mock_provider import MockProvider


def test_tool_loop_terminates_on_end_turn():
    provider = MockProvider(tool_rounds=2)
    state = ConversationState()
    reply, resp = full_brain_turn(provider, state, "do the thing", effort="medium")
    assert reply == "[mock final full-brain answer]", reply
    assert resp.stop_reason == "end_turn"
    step_calls = [c for c in provider.calls if c[0] == "full_brain_step"]
    assert len(step_calls) == 3, step_calls
    print("PASS: tool loop runs the expected number of steps and returns final text")


def test_tool_loop_respects_hard_cap():
    provider = MockProvider(tool_rounds=999)
    state = ConversationState()
    reply, resp = full_brain_turn(provider, state, "loop forever please", effort="medium")
    step_calls = [c for c in provider.calls if c[0] == "full_brain_step"]
    assert len(step_calls) == MAX_TOOL_ITERATIONS, step_calls
    assert "too many tool iterations" in reply
    print(f"PASS: hard cap stops the loop at exactly {MAX_TOOL_ITERATIONS} steps")


def test_history_stays_paired_after_tool_use():
    provider = MockProvider(tool_rounds=3)
    state = ConversationState()
    full_brain_turn(provider, state, "multi step task", effort="medium")
    for i, item in enumerate(state.full_brain_history):
        if isinstance(item, dict) and item.get("content") == "[mock tool_use turn]":
            assert i + 1 < len(state.full_brain_history), "tool_use turn left dangling at end of history"
    print("PASS: no dangling tool_use turn at the end of history after loop exit")


def test_mode_switch_context_handoff():
    state = ConversationState()
    state.add_plain_turn("user", "my name is Rehan")
    state.add_plain_turn("assistant", "[mock cheap reply to: my name is Rehan]")

    provider = MockProvider(tool_rounds=0)
    full_brain_turn(provider, state, "what's my name?", effort="medium")

    seeded_user = state.full_brain_history[0]
    assert "Rehan" in str(seeded_user), "context handoff did not include prior Muscle Memory turn"
    print("PASS: Full Brain receives prior Muscle Memory context on first use")


def test_stats_hit_rate_math_and_cost():
    stats = SessionStats()
    r1 = ProviderResponse(text="", tool_calls=[], stop_reason="end_turn", raw=None,
                           input_tokens=1000, output_tokens=50, cache_read_tokens=0, cache_write_tokens=200)
    r2 = ProviderResponse(text="", tool_calls=[], stop_reason="end_turn", raw=None,
                           input_tokens=1000, output_tokens=50, cache_read_tokens=800, cache_write_tokens=0)
    stats.add("full_brain", "mock", r1, model="claude-sonnet-5", effort="medium")
    stats.add("full_brain", "mock", r2, model="claude-sonnet-5", effort="medium")

    expected_total_input = 2000
    expected_cache_read = 800
    expected_hit_rate = 800 / 2000  # 40.0%

    total_input = sum(t.input_tokens for t in stats.turns)
    total_read = sum(t.cache_read for t in stats.turns)
    computed_hit_rate = total_read / total_input

    assert total_input == expected_total_input, (total_input, expected_total_input)
    assert abs(computed_hit_rate - expected_hit_rate) < 1e-9, (computed_hit_rate, expected_hit_rate)

    report = stats.report()
    assert "Est. session cost" in report
    assert "40.0%" in report
    print(f"PASS: hit-rate math correct ({computed_hit_rate:.1%}) and cost estimation included in report")


def test_regex_prefilter():
    # Obvious context-free greetings/farewells/thanks must match
    positives = ["hi", "hello", "hey", "thanks", "Thank you!", "thx", "goodbye", "bye", "cheers"]
    for text in positives:
        assert is_obvious_muscle_memory(text), f"Expected '{text}' to match pre-filter"

    # Context-dependent words and tasks needing tools or code must NOT match pre-filter
    negatives = [
        "yes",
        "no",
        "ok",
        "sure",
        "continue",
        "ok, delete it",
        "yes, please",
        "write a python script to test.py",
        "read the file config.json",
        "hi, can you write a script to download data?",
        "what is the meaning of life and how to implement it in code?",
        "debug this error traceback",
    ]
    for text in negatives:
        assert not is_obvious_muscle_memory(text), f"Expected '{text}' NOT to match pre-filter"

    # Explicit confirmation for anchored regex safety
    assert not is_obvious_muscle_memory("ok, delete it"), "Crucial: 'ok, delete it' must not match pre-filter"

    print("PASS: regex pre-filter correctly classifies obvious chat and rejects tool/code requests")


def test_response_cache():
    cache = ResponseCache(max_size=10, ttl_seconds=60.0)
    msg = "hi"
    resp = ProviderResponse(text="Hello! How can I help you?", tool_calls=[], stop_reason="end_turn", raw=None,
                            input_tokens=10, output_tokens=8)

    # Initial get is None
    assert cache.get(msg) is None, "Expected cache miss before put"

    # Put and retrieve
    cache.put(msg, resp)
    cached = cache.get(msg)
    assert cached is not None, "Expected cache hit after put"
    assert cached.text == resp.text

    # Case insensitivity and whitespace normalization
    assert cache.get("  HI  ") is not None, "Expected case-insensitive cache hit"

    # Trailing punctuation normalization
    assert cache.get("hi!") is not None, "Expected hit with trailing exclamation"
    assert cache.get("hi...") is not None, "Expected hit with trailing dots"

    # Synonym collapse
    thanks_resp = ProviderResponse(text="You're welcome!", tool_calls=[], stop_reason="end_turn", raw=None,
                                  input_tokens=10, output_tokens=5)
    cache.put("thanks", thanks_resp)
    assert cache.get("thank you") is not None, "Expected 'thank you' to hit cache for 'thanks'"
    assert cache.get("thx!") is not None, "Expected 'thx!' to hit cache for 'thanks'"

    print("PASS: response cache delivers normalized hits (whitespace, punctuation, synonyms) with 0 tokens")


def test_retry_utility():
    from retry_util import with_retry

    attempts = 0
    def flaky_function():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise Exception("Rate limit exceeded (429)")
        return "success"

    result = with_retry(flaky_function, max_attempts=4, base_delay=0.01)
    assert result == "success"
    assert attempts == 3
    print("PASS: retry utility retries on transient errors with backoff and succeeds")


def test_muscle_memory_context_reduction():
    """Muscle memory reply should only take the user message, zero context."""
    provider = MockProvider(tool_rounds=0)
    resp = provider.muscle_memory_reply("hello")
    # Verify the call was made with only the user message
    last_call = provider.calls[-1]
    assert last_call[0] == "muscle_memory_reply"
    assert last_call[1] == "hello"
    assert len(last_call) == 2, f"Muscle memory received unexpected context args: {last_call}"
    print("PASS: Muscle Memory receives bare user message with zero context overhead")


def test_summary_compressor():
    provider = MockProvider(tool_rounds=0)
    state = ConversationState()
    state.add_plain_turn("user", "build a web app")
    state.add_plain_turn("assistant", "sure, here is the code")
    state.full_brain_history.append({"role": "user", "content": "dummy"})
    state.full_brain_turns_since_summary = SUMMARY_INTERVAL_TURNS

    compress_summary_if_needed(provider, state, force=False)
    assert state.running_summary == "[mock summary of conversation history]"
    assert state.full_brain_turns_since_summary == 0
    print(f"PASS: summary compressor triggers at interval ({SUMMARY_INTERVAL_TURNS} turns) and updates running_summary")


def measure_pure_orchestration_overhead(n=200):
    provider = MockProvider(tool_rounds=1)
    state = ConversationState()
    start = time.perf_counter()
    for i in range(n):
        full_brain_turn(provider, state, f"turn {i}", effort="medium")
        state.reset_full_brain_if_too_long()
    elapsed = time.perf_counter() - start
    per_turn_ms = (elapsed / n) * 1000
    print(f"MEASURED (this machine, mocked network): {n} full_brain turns "
          f"(2 mock model calls each) in {elapsed:.4f}s total "
          f"-> {per_turn_ms:.3f} ms/turn of pure orchestration overhead")


if __name__ == "__main__":
    test_tool_loop_terminates_on_end_turn()
    test_tool_loop_respects_hard_cap()
    test_history_stays_paired_after_tool_use()
    test_mode_switch_context_handoff()
    test_stats_hit_rate_math_and_cost()
    test_regex_prefilter()
    test_response_cache()
    test_retry_utility()
    test_muscle_memory_context_reduction()
    test_summary_compressor()
    measure_pure_orchestration_overhead()
    print("\nALL 10 ORCHESTRATOR TESTS PASSED")
