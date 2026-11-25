"""
test_ollama_gemini_providers.py — Unit tests for Ollama and Gemini provider adapters.

Tests offline mock-requester paths, error recovery, option formatting, and tool schemas.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from base import ProviderResponse, ToolCall, ToolDef
from ollama_provider import OllamaProvider, is_ollama_reachable
from gemini_provider import GeminiProvider


def test_ollama_reachable_offline():
    """is_ollama_reachable returns False when server is offline without raising."""
    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        assert is_ollama_reachable(timeout=0.01) is False


def test_ollama_reachable_online():
    """is_ollama_reachable returns True when server responds with 200."""
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp
    with patch("urllib.request.urlopen", return_value=mock_resp):
        assert is_ollama_reachable(timeout=0.01) is True


def test_ollama_effort_to_token_mapping():
    """Effort levels correctly map to num_predict in Ollama requests."""
    captured_payloads = []

    def mock_post(endpoint, payload):
        captured_payloads.append(payload)
        return {
            "message": {"content": "ok", "tool_calls": []},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

    provider = OllamaProvider(request_fn=mock_post)
    
    # Low effort -> 256 tokens
    provider.full_brain_step([], [], effort="low")
    assert captured_payloads[-1]["options"]["num_predict"] == 256

    # High effort -> 4096 tokens
    provider.full_brain_step([], [], effort="high")
    assert captured_payloads[-1]["options"]["num_predict"] == 4096


def test_ollama_muscle_reply_caps_120_tokens():
    """Muscle memory replies in Ollama cap output to 120 tokens."""
    captured = []
    def mock_post(endpoint, payload):
        captured.append(payload)
        return {"message": {"content": "muscle output"}, "prompt_eval_count": 10, "eval_count": 5}

    provider = OllamaProvider(request_fn=mock_post)
    resp = provider.muscle_memory_reply("fast query")
    assert captured[-1]["options"]["num_predict"] == 120
    assert resp.text == "muscle output"
    assert resp.cache_read_tokens == 0
    assert resp.cache_write_tokens == 0


def test_ollama_connection_error_handling():
    """Ollama handles URLError with descriptive ConnectionError."""
    import urllib.error
    provider = OllamaProvider(host="http://localhost:99999")
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Refused")):
        with pytest.raises(ConnectionError, match="Ensure Ollama is running"):
            provider._post("/api/chat", {"test": 1})


def test_gemini_missing_api_key_raises():
    """GeminiProvider raises ValueError if GEMINI_API_KEY is not set."""
    with patch.dict("os.environ", {}, clear=True):
        provider = GeminiProvider(api_key="")
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            provider._post({"test": 1})


def test_gemini_tool_declaration_formatting():
    """Tools are mapped to Gemini functionDeclarations structure."""
    captured = []
    def mock_post(payload):
        captured.append(payload)
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "write_file", "args": {"path": "a.txt"}}}
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 10},
        }

    provider = GeminiProvider(api_key="mock_key", request_fn=lambda m, p: mock_post(p))
    tools = [
        ToolDef(name="write_file", description="write", parameters={"type": "object", "properties": {"path": {"type": "string"}}})
    ]
    resp = provider.full_brain_step([], tools, effort="medium")
    assert resp.stop_reason == "tool_use"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "write_file"
    assert captured[-1]["tools"][0]["functionDeclarations"][0]["name"] == "write_file"


def test_gemini_history_plumbing():
    """Gemini contents structure accurately appends model and function response turns."""
    provider = GeminiProvider(api_key="mock_key")
    history = []
    provider.append_user_turn(history, "hello gemini")
    assert history[0]["role"] == "user"
    assert history[0]["parts"][0]["text"] == "hello gemini"

    tc = ToolCall(id="tc_0", name="search", arguments={"q": "dual mode"})
    resp = ProviderResponse(text="searching", tool_calls=[tc], stop_reason="tool_use", raw=None)
    provider.append_assistant_turn(history, resp)
    assert history[1]["role"] == "model"
    assert history[1]["parts"][1]["functionCall"]["name"] == "search"

    provider.append_tool_results(history, [tc], {"tc_0": "search result"})
    assert history[2]["role"] == "function"
    assert history[2]["parts"][0]["functionResponse"]["name"] == "search"


def test_ollama_hallucinated_tool_json_retries_without_tools():
    """When Ollama emits fake tool-call JSON in plain text, full_brain_step retries without tools."""
    captured_payloads = []

    def mock_post(endpoint, payload):
        captured_payloads.append(payload)
        if "tools" in payload:
            return {
                "message": {
                    "content": '{"name": "explain_caching", "parameters": {"topic": "tokens"}}',
                    "tool_calls": [],
                },
                "prompt_eval_count": 50,
                "eval_count": 20,
            }
        return {
            "message": {"content": "Procedural caching avoids repeated LLM calls."},
            "prompt_eval_count": 40,
            "eval_count": 12,
        }

    provider = OllamaProvider(request_fn=mock_post)
    tools = [ToolDef(name="read_file", description="read", parameters={"type": "object", "properties": {}})]
    resp = provider.full_brain_step([{"role": "user", "content": "Explain caching"}], tools, effort="medium")
    assert len(captured_payloads) == 2
    assert "tools" in captured_payloads[0]
    assert "tools" not in captured_payloads[1]
    assert resp.text == "Procedural caching avoids repeated LLM calls."


def test_ollama_zero_cost_accounting():
    """SessionStats records $0.00 cost for local Ollama turns."""
    from base import SessionStats
    stats = SessionStats()
    stats.add(
        "full_brain",
        "ollama",
        ProviderResponse(text="ok", tool_calls=[], stop_reason="end_turn", raw=None, input_tokens=500, output_tokens=100),
        model="llama3.2",
        log_file="",
    )
    assert stats.estimated_cost_usd() == 0.0
    assert "$0.0000" in stats.report()


def test_statistics_of_routes_deterministically_to_muscle():
    """'statistics of: ...' routes to deterministic_text_statistics without counting the command prefix."""
    import tempfile
    from controller import AgentController
    from procedural_memory import ProceduralMemory
    from tests.mock_provider import MockProvider

    ctrl = AgentController(
        provider=MockProvider(),
        memory=ProceduralMemory(storage_path=tempfile.mktemp(suffix="_stats_skills.json")),
    )
    task = ctrl.start("statistics of: The quick brown fox jumps over the lazy dog.")
    reply = ctrl.execute_task(task)
    assert task.decision.pathway == "muscle"
    assert task.success is True
    assert task.exec_state.usage.total_billable_tokens == 0
    assert "Words: 9" in reply
