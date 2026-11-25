---
name: dynamic-dual-model
description: >-
  Use this skill when routing prompts between deterministic Muscle Memory and
  Full Brain execution, executing deterministic AST math/JSON/text-stats checks
  at 0 tokens, looking up trusted procedural skills, or running the Universal
  Dual-Mode HTTP/SSE Gateway and MCP tools.
---

# Dynamic Dual-Mode Model Architecture

Use the `dynamic-dual-model` MCP tools (`dual_mode_route`, `muscle_execute`,
`procedural_skill_lookup`, `provider_status`) or launch the Universal HTTP/SSE
Gateway (`dual-mode-gateway`) so routine, deterministic, and cached prompts run
via Muscle Memory while coding, tool-using, contextual, and high-risk turns run
on Full Brain.

## When to Use MCP Tools

1. **Deterministic 0-token execution (`muscle_execute`):**
   - Arithmetic expressions (`what is 48 * 25?`)
   - JSON validation (`validate json: {...}`)
   - Text statistics (`statistics of: ...`)
2. **Routing & safety inspection (`dual_mode_route`):**
   - Inspect `pathway`, `risk`, `novelty`, `tools_required`, `context_required`,
     `freshness_required`, and deterministic matches before dispatching a model.
3. **Skill memory (`procedural_skill_lookup`):**
   - Pass a query to match a trusted skill, or leave `query` empty to list all
     trusted/eligible skills.
4. **Provider readiness (`provider_status`):**
   - Check which local (`ollama`, `mock`) and cloud (`openai`, `anthropic`,
     `openrouter`, `gemini`) backends are currently available.

## Connecting Any OpenAI/Anthropic Client Without Cloning

Start the universal gateway directly via `uvx` or `pip`:

```bash
uvx --from git+https://github.com/Hiak-Harsha/dynamic-dual-model dual-mode-gateway --port 8787
```

Or install once via `pip`:

```bash
pip install git+https://github.com/Hiak-Harsha/dynamic-dual-model.git
dual-mode-gateway --port 8787
```

Then set:

```bash
export OPENAI_BASE_URL="http://127.0.0.1:8787/v1"
export ANTHROPIC_BASE_URL="http://127.0.0.1:8787"
```
