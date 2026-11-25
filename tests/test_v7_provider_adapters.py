"""
test_v7_provider_adapters.py
Unit tests for AnthropicProvider and OpenAIProvider adapters using mock API clients.
Exercises classification, routing, muscle memory replies, full brain steps, tool parsing,
prompt cache token accounting, and fallback branches to achieve high line coverage.
"""

from __future__ import annotations
import json
import os
import sys
from unittest.mock import MagicMock
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from anthropic_provider import AnthropicProvider, _safe_messages_create, _to_anthropic_tool
from openai_provider import OpenAIProvider, _safe_responses_create, _to_openai_tool
from base import ToolDef, ToolCall, ProviderResponse


# ============================================================================
# AnthropicProvider Tests
# ============================================================================

def test_anthropic_classify_and_route():
    mock_client = MagicMock()
    # Mock for classify
    mock_resp_1 = MagicMock()
    mock_resp_1.content = [MagicMock(text="muscle_memory")]
    # Mock for route_classify
    mock_resp_2 = MagicMock()
    mock_resp_2.content = [MagicMock(text=json.dumps({
        "pathway": "muscle",
        "confidence": 0.95,
        "context_required": False,
        "tools_required": False,
        "freshness_required": False,
        "novelty": 0.1,
        "risk": "low",
        "reasoning": "Simple greeting"
    }))]
    mock_resp_2.usage.input_tokens = 40
    mock_resp_2.usage.output_tokens = 20
    mock_resp_2.usage.cache_read_input_tokens = 0
    mock_resp_2.usage.cache_creation_input_tokens = 0

    mock_client.messages.create.side_effect = [mock_resp_1, mock_resp_2]

    provider = AnthropicProvider(client=mock_client)
    cls_result = provider.classify("hello", recent_summary="prior context")
    assert cls_result == "muscle_memory"

    route_dict = provider.route_classify("hello", recent_summary="")
    assert route_dict["pathway"] == "muscle"
    assert route_dict["confidence"] == 0.95
    assert "_response" in route_dict


def test_anthropic_muscle_memory_reply_and_tokens():
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text="Muscle answer")]
    mock_resp.usage.input_tokens = 50
    mock_resp.usage.output_tokens = 15
    mock_resp.usage.cache_read_input_tokens = 10
    mock_resp.usage.cache_creation_input_tokens = 5
    mock_resp.stop_reason = "end_turn"
    mock_client.messages.create.return_value = mock_resp

    provider = AnthropicProvider(client=mock_client)
    res = provider.muscle_memory_reply("test input")

    assert res.text == "Muscle answer"
    assert res.input_tokens == 50
    assert res.output_tokens == 15
    assert res.cache_read_tokens == 10
    assert res.cache_write_tokens == 5


def test_anthropic_full_brain_with_tools():
    mock_client = MagicMock()
    mock_block = MagicMock()
    mock_block.type = "tool_use"
    mock_block.name = "read_file"
    mock_block.input = {"path": "test.txt"}
    mock_block.id = "call_123"

    mock_resp = MagicMock()
    mock_resp.content = [mock_block]
    mock_resp.usage.input_tokens = 120
    mock_resp.usage.output_tokens = 30
    mock_resp.usage.cache_read_input_tokens = 0
    mock_resp.usage.cache_creation_input_tokens = 0
    mock_resp.stop_reason = "tool_use"
    mock_client.messages.create.return_value = mock_resp

    provider = AnthropicProvider(client=mock_client)
    tools = [ToolDef(name="read_file", description="reads file", parameters={"type": "object"})]
    history = [{"role": "user", "content": "read test.txt"}]
    res = provider.full_brain_step(history, tools, effort="high")

    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "read_file"
    assert res.tool_calls[0].arguments == {"path": "test.txt"}

    # Plumb history
    provider.append_assistant_turn(history, res)
    assert len(history) == 2
    provider.append_tool_results(history, res.tool_calls, {"call_123": "content"})
    assert len(history) == 3


def test_anthropic_summarize():
    mock_client = MagicMock()
    mock_resp_sum = MagicMock()
    mock_resp_sum.content = [MagicMock(text="Summary text")]
    mock_resp_sum.usage.input_tokens = 40
    mock_resp_sum.usage.output_tokens = 12
    mock_resp_sum.usage.cache_read_input_tokens = 0
    mock_resp_sum.usage.cache_creation_input_tokens = 0
    mock_client.messages.create.return_value = mock_resp_sum

    provider = AnthropicProvider(client=mock_client)
    s_text = provider.summarize_history("old history")
    assert s_text == "Summary text"


def test_safe_messages_create_fallback():
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        TypeError("unexpected keyword argument 'output_config'"),
        MagicMock(content=[MagicMock(text="ok")]),
    ]
    res = _safe_messages_create(mock_client, model="test", output_config={"effort": "low"})
    assert res.content[0].text == "ok"


# ============================================================================
# OpenAIProvider Tests
# ============================================================================

def test_openai_classify_and_route():
    mock_client = MagicMock()
    mock_resp_1 = MagicMock()
    mock_resp_1.output = [MagicMock(content=[MagicMock(text="full_brain")])]
    mock_resp_1.output_text = "full_brain"

    mock_resp_2 = MagicMock()
    mock_resp_2.output_text = json.dumps({
        "pathway": "full_brain",
        "confidence": 0.88,
        "context_required": True,
        "tools_required": True,
        "freshness_required": False,
        "novelty": 0.7,
        "risk": "medium",
        "reasoning": "Requires tool invocation"
    })
    mock_resp_2.usage.input_tokens = 50
    mock_resp_2.usage.output_tokens = 25
    mock_resp_2.usage.input_token_details.cached_tokens = 0

    mock_client.responses.create.side_effect = [mock_resp_1, mock_resp_2]

    provider = OpenAIProvider(client=mock_client)
    cls_res = provider.classify("write a script", recent_summary="some summary")
    assert cls_res == "full_brain"

    route_dict = provider.route_classify("write a script", recent_summary="")
    assert route_dict["pathway"] == "full_brain"
    assert route_dict["confidence"] == 0.88


def test_openai_muscle_memory_reply_and_tokens():
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.output_text = "OpenAI Muscle reply"
    mock_resp.usage.input_tokens = 80
    mock_resp.usage.output_tokens = 25
    mock_resp.usage.input_tokens_details.cached_tokens = 20
    mock_client.responses.create.return_value = mock_resp

    provider = OpenAIProvider(client=mock_client)
    res = provider.muscle_memory_reply("quick prompt")

    assert res.text == "OpenAI Muscle reply"
    assert res.input_tokens == 80
    assert res.output_tokens == 25
    assert res.cache_read_tokens == 20


def test_openai_full_brain_with_tools():
    mock_client = MagicMock()
    mock_tool_item = MagicMock()
    mock_tool_item.type = "function_call"
    mock_tool_item.name = "write_file"
    mock_tool_item.arguments = json.dumps({"path": "out.txt", "content": "hello"})
    mock_tool_item.call_id = "op_call_1"

    mock_resp = MagicMock()
    mock_resp.output = [mock_tool_item]
    mock_resp.output_text = ""
    mock_resp.usage.input_tokens = 100
    mock_resp.usage.output_tokens = 40
    mock_resp.usage.input_token_details.cached_tokens = 0
    mock_client.responses.create.return_value = mock_resp

    provider = OpenAIProvider(client=mock_client)
    tools = [ToolDef(name="write_file", description="writes file", parameters={"type": "object"})]
    history = [{"role": "user", "content": "write out.txt"}]
    res = provider.full_brain_step(history, tools, effort="medium")

    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "write_file"
    assert res.tool_calls[0].arguments["path"] == "out.txt"

    # Plumb history
    provider.append_assistant_turn(history, res)
    assert len(history) == 2
    provider.append_tool_results(history, res.tool_calls, {"op_call_1": "ok"})
    assert len(history) == 3


def test_openai_summarize():
    mock_client = MagicMock()
    mock_resp_sum = MagicMock()
    mock_resp_sum.output_text = "OpenAI Summary text"
    mock_resp_sum.usage.input_tokens = 45
    mock_resp_sum.usage.output_tokens = 15
    mock_resp_sum.usage.input_token_details.cached_tokens = 0
    mock_client.responses.create.return_value = mock_resp_sum

    provider = OpenAIProvider(client=mock_client)
    s_text = provider.summarize_history("old history")
    assert s_text == "OpenAI Summary text"


def test_safe_responses_create_fallback():
    mock_client = MagicMock()
    mock_client.responses.create.side_effect = [
        TypeError("unexpected keyword argument 'reasoning'"),
        MagicMock(output=[MagicMock(content=[MagicMock(text="fallback ok")])], output_text="fallback ok"),
    ]
    res = _safe_responses_create(mock_client, model="test", reasoning={"effort": "low"})
    assert res.output_text == "fallback ok"
