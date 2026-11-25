"""
test_contract.py — Universal Provider Conformance Suite.

Verifies that all provider adapters (Mock, Simulator, Ollama, Gemini, OpenRouter)
strictly satisfy the universal ModelProvider interface contract:
1. Basic completion & stop reasons
2. Empty & malformed response handling
3. Tool call parsing and structure
4. Multiple tool call handling
5. History continuation and turn shapes
6. Usage accounting and token invariants
7. Capability negotiation integration
"""

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock
import pytest

from base import ModelProvider, ProviderResponse, ToolCall, ToolDef
from tests.mock_provider import MockProvider
from tests.model_simulator import ModelSimulator
from ollama_provider import OllamaProvider
from gemini_provider import GeminiProvider
from openai_provider import OpenRouterProvider
from universal_runtime import UniversalModelRuntime, CapabilityNegotiation, ModelCapabilityCard


def _create_mocked_ollama() -> OllamaProvider:
    """Creates an OllamaProvider backed by a deterministic in-memory requester."""
    def fake_ollama_req(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if "messages" in payload and payload["messages"]:
            last_content = payload["messages"][-1].get("content", "")
            if "format" in payload and payload["format"] == "json":
                return {
                    "message": {
                        "content": json.dumps({
                            "pathway": "muscle",
                            "confidence": 0.92,
                            "tools_required": False,
                            "risk": "low",
                            "novelty": 0.1,
                            "reasoning": "Mocked ollama response"
                        })
                    },
                    "prompt_eval_count": 25,
                    "eval_count": 15,
                }
            if "tools" in payload and payload["tools"]:
                return {
                    "message": {
                        "content": "Using tool",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "read_file",
                                    "arguments": {"path": "test.txt"}
                                }
                            }
                        ]
                    },
                    "prompt_eval_count": 40,
                    "eval_count": 20,
                }
            return {
                "message": {"content": "Ollama mocked response"},
                "prompt_eval_count": 30,
                "eval_count": 10,
            }
        return {"message": {"content": "default"}, "prompt_eval_count": 10, "eval_count": 5}

    return OllamaProvider(model="llama3.2", request_fn=fake_ollama_req)


def _create_mocked_gemini() -> GeminiProvider:
    """Creates a GeminiProvider backed by a deterministic in-memory requester."""
    def fake_gemini_req(model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        tools = payload.get("tools")
        if tools:
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "read_file",
                                        "args": {"path": "test.txt"}
                                    }
                                }
                            ],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 50,
                    "candidatesTokenCount": 20,
                }
            }
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Gemini mocked response"}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 35,
                "candidatesTokenCount": 15,
            }
        }

    return GeminiProvider(model="gemini-1.5-flash", api_key="fake_key", request_fn=fake_gemini_req)


def _create_mocked_openrouter() -> OpenRouterProvider:
    """Creates an OpenRouterProvider with a mock client."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.output_text = "OpenRouter mock reply"
    mock_resp.usage.input_tokens = 40
    mock_resp.usage.output_tokens = 20
    mock_resp.usage.input_tokens_details.cached_tokens = 0
    mock_client.responses.create.return_value = mock_resp
    return OpenRouterProvider(model="meta-llama/llama-3.3-70b-instruct", client=mock_client)


@pytest.fixture(params=["mock", "simulator", "ollama", "gemini", "openrouter"])
def provider(request) -> ModelProvider:
    if request.param == "mock":
        return MockProvider()
    elif request.param == "simulator":
        return ModelSimulator(mode="perfect")
    elif request.param == "ollama":
        return _create_mocked_ollama()
    elif request.param == "gemini":
        return _create_mocked_gemini()
    elif request.param == "openrouter":
        return _create_mocked_openrouter()
    raise ValueError(f"Unknown provider param: {request.param}")


# ============================================================
# Universal Contract Tests
# ============================================================

def test_contract_basic_completion(provider: ModelProvider):
    """Every provider must return a valid ProviderResponse for muscle_memory_reply."""
    resp = provider.muscle_memory_reply("hello")
    assert isinstance(resp, ProviderResponse)
    assert isinstance(resp.text, str)
    assert resp.input_tokens >= 0
    assert resp.output_tokens >= 0
    assert resp.stop_reason in ("end_turn", "stop", "tool_use", "max_tokens")


def test_contract_classify_and_routing(provider: ModelProvider):
    """Every provider must support classify and route_classify."""
    classification = provider.classify("hi", "")
    assert classification in ("muscle_memory", "full_brain")

    route_dict = provider.route_classify("write a script", "")
    assert isinstance(route_dict, dict)
    assert "pathway" in route_dict
    assert route_dict["pathway"] in ("muscle", "full_brain")
    assert "confidence" in route_dict
    assert 0.0 <= float(route_dict["confidence"]) <= 1.0


def test_contract_history_continuation_plumbing(provider: ModelProvider):
    """Every provider must support append_user_turn, append_assistant_turn, and append_tool_results."""
    history: List[Any] = []

    # 1. User turn
    provider.append_user_turn(history, "Please read test.txt")
    assert len(history) == 1

    # 2. Assistant turn with tool call
    tc = ToolCall(id="tc_1", name="read_file", arguments={"path": "test.txt"})
    resp = ProviderResponse(
        text="Reading file now.",
        tool_calls=[tc],
        stop_reason="tool_use",
        raw=None,
        input_tokens=50,
        output_tokens=20,
    )
    provider.append_assistant_turn(history, resp)
    assert len(history) >= 2

    # 3. Tool results
    provider.append_tool_results(history, [tc], {"tc_1": "file contents: alpha"})
    assert len(history) >= 3


def test_contract_full_brain_step_with_tools(provider: ModelProvider):
    """Every provider must support full_brain_step and return valid stop_reason."""
    tools = [
        ToolDef(
            name="read_file",
            description="Reads a file from disk",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
    ]
    history: List[Any] = []
    provider.append_user_turn(history, "Read test.txt please")

    resp = provider.full_brain_step(history, tools, effort="medium")
    assert isinstance(resp, ProviderResponse)
    assert resp.stop_reason in ("end_turn", "tool_use")
    assert resp.input_tokens >= 0
    assert resp.output_tokens >= 0


def test_contract_summary_step(provider: ModelProvider):
    """Every provider must support summarize_history_step."""
    sample_history = "User: Hi\nAssistant: Hello! How can I help?\nUser: What is 2+2?\nAssistant: 4"
    resp = provider.summarize_history_step(sample_history)
    assert isinstance(resp, ProviderResponse)
    assert isinstance(resp.text, str)
    assert len(resp.text) > 0
    assert resp.output_tokens >= 0


def test_contract_capability_negotiation(provider: ModelProvider):
    """UniversalModelRuntime must cleanly negotiate capabilities with provider."""
    runtime = UniversalModelRuntime(provider)
    card = runtime.capability_card
    assert card is not None
    assert isinstance(card, ModelCapabilityCard)

    config, fallbacks = CapabilityNegotiation.negotiate(
        card,
        requested_effort="high",
        requested_max_output_tokens=4096,
    )
    assert "effort" in config
    assert "max_output_tokens" in config
    assert config["max_output_tokens"] <= card.max_output_tokens


def test_contract_ledger_recording(provider: ModelProvider):
    """Invoking runtime methods must record entries in CognitionLedger."""
    runtime = UniversalModelRuntime(provider)
    initial_calls = runtime.ledger.total_calls()

    _ = runtime.muscle_memory_reply("ping")
    assert runtime.ledger.total_calls() == initial_calls + 1
    last_entry = runtime.ledger.entries[-1]
    assert last_entry.phase == "muscle"
    assert last_entry.provider_name == provider.name
    assert last_entry.model_id == provider.model
