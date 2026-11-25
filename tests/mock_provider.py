"""
tests/mock_provider.py — deterministic MockProvider matching the
SAME-MODEL ModelProvider interface.
"""

from __future__ import annotations
from typing import Any, Dict, List

import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from base import ModelProvider, ProviderResponse, ToolCall, ToolDef


class MockProvider(ModelProvider):
    name = "mock"
    model = "mock-sonnet-5"

    def __init__(
        self,
        classify_answer="muscle_memory",
        tool_rounds=0,
        classify_input_tokens: int = 150,
        classify_output_tokens: int = 30,
        model: str = "mock-sonnet-5",
    ):
        self.classify_answer = classify_answer
        self.tool_rounds = tool_rounds
        self.classify_input_tokens = classify_input_tokens
        self.classify_output_tokens = classify_output_tokens
        self.model = model
        self._step_count = 0
        self.calls = []

    def reset(self) -> None:
        self._step_count = 0
        self.calls.clear()

    def classify(self, user_message: str, recent_summary: str) -> str:
        self.calls.append(("classify", user_message, recent_summary))
        return self.classify_answer

    def route_classify(self, user_message: str, recent_summary: str) -> Dict[str, Any]:
        self.calls.append(("route_classify", user_message, recent_summary))
        low = user_message.lower()
        if any(w in low for w in ["write", "file", "delete", "code", "debug", "refactor", "database", "drop", "table"]):
            is_muscle = False
        else:
            is_muscle = (self.classify_answer == "muscle_memory")
        return {
            "pathway": "muscle" if is_muscle else "full_brain",
            "confidence": 0.90 if is_muscle else 0.95,
            "context_required": bool(any(w in low for w in ["earlier", "previous", "that", "it", "continue"])),
            "tools_required": not is_muscle,
            "freshness_required": bool(any(w in low for w in ["today", "current", "latest", "now"])),
            "novelty": 0.2 if is_muscle else 0.7,
            "risk": "high" if any(w in low for w in ["delete", "drop", "purge"]) else ("medium" if not is_muscle else "low"),
            "reasoning": f"Mock structured classification (is_muscle={is_muscle}).",
            "_response": ProviderResponse(
                text=f"Mock route: {'muscle' if is_muscle else 'full_brain'}",
                tool_calls=[],
                stop_reason="end_turn",
                raw=None,
                input_tokens=self.classify_input_tokens,
                output_tokens=self.classify_output_tokens,
            ),
        }

    def muscle_memory_reply(self, user_message: str) -> ProviderResponse:
        self.calls.append(("muscle_memory_reply", user_message))
        if "rubric" in user_message.lower() or "verdict" in user_message.lower():
            text = json.dumps({"verdict": "PASS", "confidence": 0.95, "reason": "Mock rubric verified"})
        else:
            text = f"[mock cheap reply to: {user_message}]"
        return ProviderResponse(
            text=text,
            tool_calls=[], stop_reason="end_turn", raw=None,
            input_tokens=50, output_tokens=20,
        )

    def full_brain_step(self, history: List[Any], tools: List[ToolDef], effort: str) -> ProviderResponse:
        self._step_count += 1
        self.calls.append(("full_brain_step", len(history), effort))
        if self._step_count <= self.tool_rounds:
            tool_name = "read_file"
            tool_args = {"path": "test.txt"}
            for item in history:
                text_val = ""
                if isinstance(item, dict):
                    content = item.get("content", "")
                    if isinstance(content, str):
                        text_val = content
                elif isinstance(item, str):
                    text_val = item
                if "write" in text_val.lower():
                    import re
                    m = re.search(r"write\s+(?:'|\")?(.+?)(?:'|\")?\s+to\s+([a-zA-Z0-9_\-\.]+)", text_val, re.IGNORECASE)
                    if m:
                        tool_name = "write_file"
                        tool_args = {"path": m.group(2), "content": m.group(1), "overwrite": True}
                        break
                    elif "test_suite_smoke.txt" in text_val:
                        tool_name = "write_file"
                        tool_args = {"path": "test_suite_smoke.txt", "content": "hello world", "overwrite": True}
                        break

            return ProviderResponse(
                text="", tool_calls=[ToolCall(id=f"call_{self._step_count}", name=tool_name,
                                              arguments=tool_args)],
                stop_reason="tool_use", raw=None,
                input_tokens=1200, output_tokens=40,
                cache_read_tokens=1000 if self._step_count > 1 else 0,
                cache_write_tokens=200 if self._step_count == 1 else 0,
                native_assistant_turn={"role": "assistant", "content": "[mock tool_use turn]"},
            )
        return ProviderResponse(
            text="[mock final full-brain answer]", tool_calls=[], stop_reason="end_turn", raw=None,
            input_tokens=1200, output_tokens=80,
            cache_read_tokens=1000, cache_write_tokens=0,
            native_assistant_turn={"role": "assistant", "content": "[mock final turn]"},
        )

    def summarize_history_step(self, history_text: str) -> ProviderResponse:
        self.calls.append(("summarize_history", len(history_text)))
        return ProviderResponse(
            text="[mock summary of conversation history]",
            tool_calls=[],
            stop_reason="end_turn",
            raw=None,
            input_tokens=120,
            output_tokens=35,
        )

    def summarize_history(self, history_text: str) -> str:
        return self.summarize_history_step(history_text).text

    def append_assistant_turn(self, history: List[Any], resp: ProviderResponse) -> None:
        history.append(resp.native_assistant_turn)

    def append_tool_results(self, history: List[Any], tool_calls: List[ToolCall], results: Dict[str, str]) -> None:
        history.append({"role": "user", "content": [{"tool_use_id": tc.id, "result": results.get(tc.id, "")} for tc in tool_calls]})

    def append_user_turn(self, history: List[Any], text: str) -> None:
        history.append({"role": "user", "content": text})
