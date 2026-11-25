"""
anthropic_provider.py — SAME-MODEL dual-mode adapter for Anthropic.

This is NOT a cheap-model / flagship-model cascade. Every call —
classifier, Muscle Memory, Full Brain, and summary compressor — uses
the SAME model ID. They are differentiated only by:

  - effort    : low (classifier/muscle/summary) vs medium/high/xhigh (brain)
  - max_tokens: 8 (classifier), 120 (muscle), 300 (summary), 8000 (brain)
  - tools     : none (classifier/muscle/summary) vs full toolset (brain)
  - context   : bare message (muscle) vs full history (brain)

Current as of Sept 2026 — verify against https://docs.claude.com:

  - Model: "claude-sonnet-5" for everything. The previous architecture
    used Haiku for cheap calls — that was the wrong design.
  - Effort: top-level output_config={"effort": "low"|"medium"|"high"|
    "xhigh"|"max"}. NOT nested inside thinking.
  - Thinking: thinking={"type": "adaptive"} on Full Brain ONLY. Omitted
    on classifier/muscle/summary where we don't want reasoning overhead.
  - Sonnet 5 rejects temperature / top_p / top_k with 400 if non-default.
  - Prompt caching: cache_control={"type": "ephemeral"} on the system
    block and on the LAST tool. Minimum cacheable length ~1,024 tokens.
    The Full Brain system prompt is padded to ~1,200 tokens to cross this.
"""

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

try:
    from anthropic import Anthropic  # type: ignore[import-not-found, import-untyped]
except ImportError:
    Anthropic = None  # type: ignore

from base import ModelProvider, ProviderResponse, ToolCall, ToolDef

from retry_util import with_retry

CACHE = {"type": "ephemeral"}


# The Full Brain system prompt, padded past Anthropic's ~1,024-token
# minimum cacheable length so prompt caching actually activates.
# This is STATIC — no timestamps, user IDs, or session data.
FULL_BRAIN_SYSTEM_PROMPT = (
    "You are an expert software engineer, systems architect, and problem "
    "solver with access to file tools. Your capabilities include reading, "
    "writing, and analyzing code files, debugging issues, implementing "
    "features, and providing technical guidance.\n\n"

    "## Core Behavior\n"
    "- PREFER using tools over describing what to do. If a user asks you "
    "to write code, actually write it to a file — don't just show it.\n"
    "- Be thorough but stay focused on the requested result.\n"
    "- When reading files, scan for the specific information needed rather "
    "than summarizing the entire file.\n"
    "- When writing files, include complete, working code — never use "
    "placeholders like '// ... rest of code ...'.\n\n"

    "## Coding Style Guide\n"
    "Follow these conventions unless the existing codebase uses different ones:\n"
    "- Use descriptive variable and function names (no single-letter names "
    "except loop counters and well-known conventions like x/y coordinates).\n"
    "- Include docstrings for public functions and classes.\n"
    "- Handle errors explicitly — never silently swallow exceptions.\n"
    "- Prefer composition over inheritance where practical.\n"
    "- Keep functions focused: one function, one responsibility.\n"
    "- Use type hints in Python; use TypeScript over plain JavaScript.\n"
    "- Constants should be UPPER_SNAKE_CASE and defined at module level.\n"
    "- Imports should be sorted: stdlib first, then third-party, then local.\n"
    "- Maximum line length: 100 characters (soft limit, readability first).\n"
    "- Use f-strings for string formatting in Python 3.6+.\n\n"

    "## Tool Usage Guidelines\n"
    "- Always read a file before modifying it, unless you just wrote it.\n"
    "- When writing files, use the full path including any subdirectories.\n"
    "- If a tool call fails, report the error clearly and suggest an "
    "alternative approach rather than retrying blindly.\n"
    "- Prefer small, incremental file changes over rewriting entire files.\n"
    "- When creating new files, include appropriate file headers and "
    "documentation.\n\n"

    "## Output Format\n"
    "- For code explanations, use fenced code blocks with language tags.\n"
    "- For multi-step plans, use numbered lists.\n"
    "- Keep prose concise — favor bullet points over paragraphs.\n"
    "- When reporting tool results, summarize the outcome rather than "
    "echoing the full output verbatim.\n"
    "- If the user's request is ambiguous, ask ONE clarifying question "
    "rather than guessing — but lean toward doing the obvious thing if "
    "it's 80%+ likely what they meant.\n\n"

    "## Error Handling & Debugging\n"
    "- When debugging, start by reading the relevant file(s) and error "
    "messages before proposing fixes.\n"
    "- Explain the root cause of bugs, not just the fix.\n"
    "- When a fix is non-obvious, add a comment explaining WHY, not just "
    "WHAT the code does.\n"
    "- Test your assumptions: if you think a file exists, read it first; "
    "if you think a function has a certain signature, verify it.\n\n"

    "## Sandboxing and Security Guarantees\n"
    "- All file read and write operations are strictly confined to the workspace sandbox root.\n"
    "- Any attempt to navigate outside the sandbox using path traversal sequences (such as parent "
    "directory references or absolute paths outside the target directory) will be rejected by the security layer.\n"
    "- Destructive actions—including file deletions, table drops, and major schema alterations—are treated as "
    "high-risk operations. Always verify paths and ensure atomic write operations using temporary replacement files.\n"
    "- Never write partial implementations or mock placeholders when modifying user code.\n\n"

    "## Quality Parity and Adaptive Execution\n"
    "- This agent operates under strict quality parity constraints: Muscle Memory is used exclusively "
    "when equivalent quality to Full Brain is mathematically or deterministically assured.\n"
    "- Full Brain mode is engaged whenever tasks require multi-step reasoning, complex file operations, "
    "live external data, or deep structural analysis.\n"
    "- The agent maintains an ExecutionState tracker recording completed steps, observations, and tool outcomes. "
    "If a task transitions between execution modes, a compact continuation handoff block (< 400 tokens) is "
    "transferred to avoid redundant re-discovery and prevent lost context.\n"
    "- Tool execution errors must be diagnosed systematically. If a tool fails consecutively, do not repeat "
    "the same call blindly; reassess parameters, verify file permissions, or propose an alternative strategy.\n"
    "- Procedural memory learns successful execution patterns from verified Full Brain completions. Maintain "
    "clean, self-contained procedures and adhere to deterministic verification standards.\n"
    "- Session statistics record turn counts, token expenditure, and prompt cache hit rates. Never degrade "
    "solution quality to conserve tokens—Rule R1 requires that efficiency is achieved strictly through intelligent "
    "routing and caching, never through degraded responses.\n\n"

    "## Operational Constraints and Invariants\n"
    "- Maintain backward compatibility across interfaces and preserve existing public signatures.\n"
    "- Adhere to the principle of least privilege in tool execution: only invoke tools when required for the task.\n"
    "- Ensure that every tool result is verified before declaring task completion.\n"
    "- When presenting results to the user, distinguish between verified facts and model inferences.\n"
    "- Preserve deterministic formatting for machine-readable outputs, including JSON payloads and structured error blocks.\n"
    "- This system prompt is intentionally structured and padded to exceed the 1,024-token prompt caching activation "
    "threshold to guarantee that prompt caching activates on every Full Brain turn and prefix tokens are not re-billed unnecessarily."
)


def _to_anthropic_tool(t: ToolDef) -> Dict[str, Any]:
    return {"name": t.name, "description": t.description, "input_schema": t.parameters}


def _safe_messages_create(client, **kwargs):
    """Executes messages.create with graceful degradation if optional kwargs like output_config are unsupported."""
    try:
        return client.messages.create(**kwargs)
    except TypeError as e:
        if "output_config" in kwargs and "unexpected keyword" in str(e).lower():
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("output_config", None)
            return client.messages.create(**fallback_kwargs)
        raise


class AnthropicProvider(ModelProvider):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5", client: Optional[Any] = None):
        if client is not None:
            self.client = client
        elif Anthropic is None:
            raise ImportError(
                "The 'anthropic' package is not installed. "
                "Install it with: pip install 'anthropic>=0.40.0'"
            )
        else:
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    # ---------------- classification ----------------
    def classify(self, user_message: str, recent_summary: str) -> str:
        system = (
            "You are a strict single-word classifier.\n"
            "muscle_memory: greetings, acknowledgments, small talk, simple "
            "factual questions needing no tools, files, or code.\n"
            "full_brain: anything needing code, file operations, review, "
            "multi-step reasoning, or tool use.\n"
            "If genuinely unsure, answer full_brain — a wrong 'muscle_memory' "
            "guess silently strands the user with a tool-free reply.\n"
            "Reply with exactly one word: muscle_memory or full_brain."
        )
        messages = []
        if recent_summary.strip():
            # Non-conversational context preamble: inject as structured prefix
            # on the user message. NEVER as a fabricated assistant turn.
            messages.append({"role": "user", "content": (
                f"[CONTEXT FOR CLASSIFICATION — do not respond to this, "
                f"use it only to inform your classification decision]\n"
                f"{recent_summary}\n\n"
                f"[MESSAGE TO CLASSIFY]\n{user_message}"
            )})
        else:
            messages.append({"role": "user", "content": user_message})

        resp = with_retry(lambda: self.client.messages.create(
            model=self.model,         # SAME model — just with effort=low
            max_tokens=8,
            system=system,
            messages=messages,
            output_config={"effort": "low"},  # minimal effort for classification
        ))
        decision = resp.content[0].text.strip().lower() if resp.content else ""
        if "muscle_memory" in decision and "full_brain" not in decision:
            return "muscle_memory"
        return "full_brain"

    def route_classify(self, user_message: str, recent_summary: str) -> Dict[str, Any]:
        """Queries the same model with effort=low requesting a structured JSON RouteDecision."""
        system = (
            "You are a strict multi-axis task routing engine. Analyze the incoming user request "
            "and classify it across orthogonal dimensions.\n\n"
            "Respond ONLY with a valid JSON object adhering strictly to this schema:\n"
            "{\n"
            '  "pathway": "muscle" | "full_brain",\n'
            '  "confidence": <float 0.0-1.0>,\n'
            '  "context_required": <boolean: true if user refers to prior turns/decisions/files>,\n'
            '  "tools_required": <boolean: true if tool/file operations or live execution is needed>,\n'
            '  "freshness_required": <boolean: true if current/live external facts are required>,\n'
            '  "novelty": <float 0.0-1.0: 0.0=routine/familiar, 1.0=unprecedented/complex>,\n'
            '  "risk": "low" | "medium" | "high",\n'
            '  "reasoning": "<concise single-sentence justification>"\n'
            "}\n\n"
            "Rules:\n"
            "- pathway: 'muscle' ONLY for simple greetings, acknowledgments, basic math/trivia needing NO tools.\n"
            "- pathway: 'full_brain' for code, file ops, debugging, multi-step tasks, or high-risk actions.\n"
            "- If unsure, choose 'full_brain'. Output raw JSON only, no markdown fencing."
        )
        messages = []
        if recent_summary.strip():
            messages.append({"role": "user", "content": (
                f"[CONTEXT FOR ROUTING — do not respond to this, use only to inform classification]\n"
                f"{recent_summary}\n\n"
                f"[MESSAGE TO ROUTE]\n{user_message}"
            )})
        else:
            messages.append({"role": "user", "content": user_message})

        resp = with_retry(lambda: self.client.messages.create(
            model=self.model,
            max_tokens=250,
            system=system,
            messages=messages,
            output_config={"effort": "low"},
        ))
        raw_text = resp.content[0].text.strip() if resp.content else ""
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw_text = "\n".join(lines).strip()

        u = resp.usage
        fresh = getattr(u, "input_tokens", 0) or 0
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
        provider_resp = ProviderResponse(
            text=raw_text,
            tool_calls=[],
            stop_reason="end_turn",
            raw=resp,
            input_tokens=fresh + cache_read + cache_write,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        )

        import json
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict) and "pathway" in parsed:
                parsed["_response"] = provider_resp
                return parsed
        except Exception:
            pass
        fallback = super().route_classify(user_message, recent_summary)
        fallback["_response"] = provider_resp
        return fallback

    # ---------------- muscle memory ----------------
    def muscle_memory_reply(self, user_message: str) -> ProviderResponse:
        system = (
            "You are a concise assistant. Reply in 1-3 short sentences. "
            "Never write code or reference files in this mode."
        )
        # AGGRESSIVE CONTEXT REDUCTION: only the bare user message.
        # No recent_context_str, no running_summary. The whole point of
        # Muscle Memory is near-zero context → near-minimum input tokens.
        messages = [{"role": "user", "content": user_message}]

        resp = with_retry(lambda: _safe_messages_create(
            self.client,
            model=self.model,         # SAME model
            max_tokens=120,
            system=system,
            messages=messages,
            output_config={"effort": "low"},  # minimal effort
        ))
        u = resp.usage
        return ProviderResponse(
            text=resp.content[0].text if resp.content else "",
            tool_calls=[],
            stop_reason="end_turn",
            raw=resp,
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )

    # ---------------- full brain ----------------
    def full_brain_step(self, history: List[Any], tools: List[ToolDef], effort: str) -> ProviderResponse:
        system = [{"type": "text", "text": FULL_BRAIN_SYSTEM_PROMPT, "cache_control": CACHE}]

        anthropic_tools = [_to_anthropic_tool(t) for t in tools]
        if anthropic_tools:
            anthropic_tools[-1]["cache_control"] = CACHE

        resp = with_retry(lambda: _safe_messages_create(
            self.client,
            model=self.model,         # SAME model — just with higher effort
            max_tokens=8000,
            system=system,
            tools=anthropic_tools,
            messages=history,
            output_config={"effort": effort},
        ))

        tool_calls = [
            ToolCall(id=b.id, name=b.name, arguments=b.input)
            for b in resp.content if b.type == "tool_use"
        ]
        text = "".join(b.text for b in resp.content if b.type == "text")
        u = resp.usage
        fresh = getattr(u, "input_tokens", 0) or 0
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
        return ProviderResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason="tool_use" if resp.stop_reason == "tool_use" else "end_turn",
            raw=resp,
            input_tokens=fresh + cache_read + cache_write,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            native_assistant_turn={"role": "assistant", "content": resp.content},
        )

    # ---------------- summary compressor ----------------
    def summarize_history_step(self, history_text: str) -> ProviderResponse:
        system = (
            "Compress the following conversation excerpt into 3-5 concise "
            "sentences capturing the key topics discussed, decisions made, "
            "and any important context. Omit greetings and filler."
        )
        resp = with_retry(lambda: self.client.messages.create(
            model=self.model,         # SAME model, effort=low
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": history_text}],
            output_config={"effort": "low"},
        ))
        text = resp.content[0].text.strip() if resp.content else ""
        u = resp.usage
        fresh = getattr(u, "input_tokens", 0) or 0
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
        return ProviderResponse(
            text=text,
            tool_calls=[],
            stop_reason="end_turn",
            raw=resp,
            input_tokens=fresh + cache_read + cache_write,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        )

    def summarize_history(self, history_text: str) -> str:
        return self.summarize_history_step(history_text).text

    # ---------------- history plumbing ----------------
    def append_assistant_turn(self, history: List[Any], resp: ProviderResponse) -> None:
        history.append(resp.native_assistant_turn)

    def append_tool_results(self, history: List[Any], tool_calls: List[ToolCall], results: Dict[str, str]) -> None:
        content = [
            {"type": "tool_result", "tool_use_id": tc.id, "content": results.get(tc.id, "")}
            for tc in tool_calls
        ]
        history.append({"role": "user", "content": content})

    def append_user_turn(self, history: List[Any], text: str) -> None:
        history.append({"role": "user", "content": text})
