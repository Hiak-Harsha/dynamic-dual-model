"""
tests/test_phase0_safety.py — Verification of Phase 0 safety-critical fixes:
1. ResponseCache allow-list (blocks 'yes', 'no', 'continue', 'yep', etc.)
2. ConversationState token budgeting (max_recent_tokens, full_brain_context_budget)
3. Summary truncation (summary replaces old history, does not accumulate indefinitely)
4. Safe & atomic write_file (refuses overwrite unless overwrite=True, writes via tempfile)
5. Clean context preamble (no synthetic 'assistant: Understood.' fake history)
"""

import sys, os, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from base import ResponseCache, ConversationState, ProviderResponse, estimate_tokens
from tools import execute_tool, _resolve_in_sandbox, SANDBOX_ROOT
from dual_mode_agent import full_brain_turn, is_obvious_muscle_memory, compress_summary_if_needed
from mock_provider import MockProvider


def test_cache_safety_allow_list():
    cache = ResponseCache()
    dummy_resp = ProviderResponse(text="Hello there!", tool_calls=[], stop_reason="end_turn", raw=None)
    
    # 1. Allowed items get cached
    cache.put("hello", dummy_resp)
    assert cache.get("hello") is not None
    assert cache.get("Hello!") is not None  # normalized match

    # 2. Context-dependent tokens MUST NEVER get cached
    deny_words = ["yes", "no", "yep", "nope", "continue", "do it", "sure", "ok", "okay"]
    for word in deny_words:
        cache.put(word, dummy_resp)
        assert cache.get(word) is None, f"Dangerous token '{word}' was retrieved from cache!"
        assert not cache.is_cacheable(word), f"Dangerous token '{word}' reported as cacheable!"

    print("PASS: ResponseCache strictly enforces allow-list and blocks context-dependent tokens.")


def test_conversation_state_token_budget():
    state = ConversationState(max_recent_tokens=50, max_recent_plain_turns=10)
    
    # Add several plain turns
    state.add_plain_turn("user", "short text")
    assert len(state.recent_plain_turns) == 1
    
    # Add a huge message that exceeds the token budget (~4 chars per token)
    huge_text = "word " * 200  # 1000 chars ~ 250 tokens > 50 tokens budget
    state.add_plain_turn("assistant", huge_text)
    
    # Older items must be trimmed to satisfy token budget
    assert state.total_recent_tokens() <= 260  # Only latest huge text kept, prior short text popped
    assert len(state.recent_plain_turns) == 1
    
    print("PASS: ConversationState respects token-based budget for recent turns.")


def test_summary_replacement_truncates_history():
    provider = MockProvider(tool_rounds=0)
    state = ConversationState()
    
    # Populate history with 10 turns
    for i in range(10):
        provider.append_user_turn(state.full_brain_history, f"turn {i}")
    
    assert len(state.full_brain_history) == 10
    state.full_brain_turns_since_summary = 8
    
    compress_summary_if_needed(provider, state, force=True)
    
    # After compression, history should be truncated to keep_tail_items (4)
    assert len(state.full_brain_history) <= 4, f"Expected <= 4 items, got {len(state.full_brain_history)}"
    assert state.running_summary != ""
    assert state.full_brain_turns_since_summary == 0
    print("PASS: Summary compression truncates old history rather than accumulating indefinitely.")


def test_safe_and_atomic_write_file():
    test_filename = "phase0_safety_test.txt"
    test_content_v1 = "Initial content v1"
    test_content_v2 = "Overwritten content v2"
    
    # Ensure clean state
    full_path = _resolve_in_sandbox(test_filename)
    if os.path.exists(full_path):
        os.remove(full_path)
        
    # 1. Initial write without overwrite flag should succeed
    res1 = execute_tool("write_file", {"path": test_filename, "content": test_content_v1})
    assert "OK:" in res1, res1
    assert os.path.exists(full_path)
    with open(full_path, "r", encoding="utf-8") as f:
        assert f.read() == test_content_v1

    # 2. Second write without overwrite flag MUST fail safely
    res2 = execute_tool("write_file", {"path": test_filename, "content": test_content_v2})
    assert "ERROR: File" in res2 and "already exists" in res2, res2
    # Verify content was NOT modified
    with open(full_path, "r", encoding="utf-8") as f:
        assert f.read() == test_content_v1

    # 3. Third write with overwrite=True MUST succeed atomically
    res3 = execute_tool("write_file", {"path": test_filename, "content": test_content_v2, "overwrite": True})
    assert "OK:" in res3, res3
    with open(full_path, "r", encoding="utf-8") as f:
        assert f.read() == test_content_v2

    # Clean up with retry for Windows filesystem file locks
    for _ in range(5):
        try:
            if os.path.exists(full_path):
                os.remove(full_path)
            break
        except PermissionError:
            time.sleep(0.05)
    print("PASS: write_file refuses overwrite by default and writes atomically with overwrite=True.")


def test_no_synthetic_assistant_turn():
    provider = MockProvider(tool_rounds=0)
    state = ConversationState()
    state.add_plain_turn("user", "my favorite color is teal")
    state.add_plain_turn("assistant", "I will remember that!")

    full_brain_turn(provider, state, "what color do I like?")

    # Verify history has exactly 2 turns: 1 user turn (with context preamble), 1 assistant response
    assert len(state.full_brain_history) == 2, f"Expected 2 items, got {len(state.full_brain_history)}"
    
    # First turn is user turn with context preamble
    user_turn = state.full_brain_history[0]
    user_text = str(user_turn)
    assert "[CONTEXT FROM PRIOR TURNS]" in user_text
    assert "teal" in user_text
    assert "what color do I like?" in user_text

    # Second turn is real assistant response
    assistant_turn = state.full_brain_history[1]
    assert "Understood." not in str(assistant_turn)
    print("PASS: No synthetic 'assistant: Understood.' turn injected into conversation history.")


if __name__ == "__main__":
    test_cache_safety_allow_list()
    test_conversation_state_token_budget()
    test_summary_replacement_truncates_history()
    test_safe_and_atomic_write_file()
    test_no_synthetic_assistant_turn()
    print("\nALL PHASE 0 SAFETY TESTS PASSED!")
