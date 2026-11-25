# Dynamic Dual-Mode Model Architecture (`v9.1.0`)

An adaptive dual-path LLM agent architecture that routes routine, deterministic, and learned procedural tasks through a lightweight **Muscle Memory** engine while escalating complex, context-dependent, high-risk, or uncertain tasks to a deliberative **Full Brain** — using structured **Continuation Handoffs** (`HandoffState`) instead of discarding partial work.

> **Status:** Offline hardened & verified (`269/269` pytest tests passing, `build_manifest.json` v9.1.0). Packaged benchmark artifacts are simulation-validated (`MockProvider` / `ModelSimulator`); live API validation scripts are wired and ready for execution with API keys or local Ollama.

---

## Core Architectural Principles

1. **R1 — Quality Parity, Not Quality Trade-off:** Muscle Memory never knowingly ships degraded answers to save tokens; outputs pass deterministic verification and explicit success contracts before completion.
2. **R2 — Orthogonal Routing Axes:** The router independently evaluates difficulty, context-dependency, freshness, and risk rather than relying on naive length heuristics.
3. **R3 — Continuation, Not Restart:** When Muscle Memory hits uncertainty or requires broader tool reasoning, it emits a compact structured `HandoffState` so Full Brain continues from verified progress instead of restarting from scratch.
4. **R4 — State != Chat Transcript:** Progress, tool effects, disk SHA-256 freshness, and unresolved items live in structured `ExecutionState` and `TaskFrontier`.
5. **R5 — Gated Procedural Learning:** Learned skills progress through gated trust tiers (`candidate` -> `trusted` -> `eligible`) and are filtered against conversational artifacts; consecutive failures trigger demotion.

---

## Architecture Overview

```text
User Prompt
    |
    v
Multi-Axis Router (router.py)
    |-- Deterministic Gates (allow-listed cache, greetings, destructive verbs,
    |                        context references, freshness words)
    '-- Optional Low-Effort Classifier Fallback
    |
    +--> [Tier 0: Response Cache] (0 tokens)
    |
    +--> [Muscle Pathway: muscle_engine.py]
    |       - Tier 0: Exact response cache (0 tokens)
    |       - Tier 1: Deterministic Python execution (AST math, JSON validation, text stats)
    |       - Tier 2: Procedural skill replay / DAG execution (procedure_executor.py)
    |       - Tier 3: Bounded adaptive execution under strict token/tool budgets
    |         |
    |         v
    |    Verification & Success Contracts (verification.py, success_contracts.py)
    |         |-- COMPLETED_VERIFIED --> Reinforce skill & return
    |         '-- HANDOFF / PARTIAL  --> Build HandoffState (handoff.py)
    |                                        |
    +--> [Full Brain Pathway: brain.py] <----+
            - Resumes from HandoffState frontier without repeating verified steps
            - Sandboxed tools (tools.py) + dynamic effort scaling
            - All model calls routed via UniversalModelRuntime + CognitionLedger
```

---

## Quick Start

### 1. Environment Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Core dependencies in `requirements.txt`:
- `anthropic>=0.40.0`
- `openai>=1.50.0`
- `pytest>=7.0.0`
- `pytest-cov>=4.0.0`

### 2. List Available Providers

```powershell
python dual_mode_agent.py --list-providers
```

### 3. Run the Interactive CLI (`dual_mode_agent.py`)

```powershell
# Zero-cost offline deterministic mock provider
python dual_mode_agent.py --provider mock

# Zero-cost local execution via Ollama (requires `ollama serve` on localhost:11434)
python dual_mode_agent.py --provider ollama --model llama3.2

# Cloud providers (set the corresponding API key first)
python dual_mode_agent.py --provider openai --model gpt-4o
python dual_mode_agent.py --provider anthropic
python dual_mode_agent.py --provider gemini
python dual_mode_agent.py --provider openrouter
```

Interactive REPL commands:
- `report` — print session token, cache, and cost telemetry
- `quit` or `exit` — shut down cleanly and print final session report

Optional session quota environment variables:
- `AGENT_TOKEN_WARN_LIMIT` (default: `40000`)
- `AGENT_TOKEN_HARD_LIMIT` (default: `90000`)

---

## Supported Providers (`provider_registry.py`)

All providers plug into `UniversalModelRuntime` (`universal_runtime.py`) via `ModelCapabilityCard` declarations and record invocations in `CognitionLedger`:

| Provider | Default Model | Requirement | Notes |
|---|---|---|---|
| `mock` | `mock-sonnet-5` | None (`$0`) | Deterministic offline provider for tests & smoke runs |
| `ollama` | `llama3.2` | Local daemon (`localhost:11434`) | Zero-cost local model execution |
| `openai` | `gpt-4o` | `OPENAI_API_KEY` | OpenAI Responses / Chat adapter (`openai_provider.py`) |
| `anthropic` | `claude-sonnet-5` | `ANTHROPIC_API_KEY` | Anthropic Messages adapter (`anthropic_provider.py`) |
| `openrouter` | `meta-llama/llama-3.3-70b-instruct` | `OPENROUTER_API_KEY` | OpenAI-compatible OpenRouter adapter |
| `gemini` | `gemini-1.5-flash` | `GEMINI_API_KEY` | Google Generative Language REST adapter (`gemini_provider.py`) |

---

## Testing, Verification & Audit Sync

```powershell
# 1. Bytecode / syntax check across all files
python -m compileall -q .

# 2. Full automated test suite (269 tests)
python -m pytest -q

# 3. Universal Provider Conformance suite (35 tests)
python -m pytest tests/provider_contract/ -q

# 4. Line coverage report
python -m pytest tests/ --cov=. --cov-report=term -q

# 5. Verify zero drift across pytest count, build_manifest.json SHA-256 hashes, and docs
python scripts/verify_audit_sync.py
```

---

## Validation & Benchmark Runners

### Live / Mock Functional Validation (`scripts/live_validation.py`)
Suites: `basic`, `deterministic`, `tool`, `handoff`, `restart_resume`, or `all`.

```powershell
python scripts/live_validation.py --provider mock --test all
python scripts/live_validation.py --provider ollama --model llama3.2 --test all
python scripts/live_validation.py --provider anthropic --test all
python scripts/live_validation.py --provider openai --test deterministic --limit 2
```

### 5-Turn Smoke Test (`scripts/live_api_smoke_test.py`)

```powershell
python scripts/live_api_smoke_test.py --mock
python scripts/live_api_smoke_test.py --provider ollama --model llama3.2
python scripts/live_api_smoke_test.py --provider anthropic
python scripts/live_api_smoke_test.py --provider openai
```

### A/B Token & Accuracy Benchmark (`scripts/ab_runner.py`)

```powershell
python scripts/ab_runner.py --mock
python scripts/ab_runner.py --live --provider ollama --model llama3.2
python scripts/ab_runner.py --live --provider anthropic
python scripts/ab_runner.py --live --provider openai --model gpt-4o
```

### 4-Condition Scientific Experiment (`benchmark/scientific_ab_runner.py`)
Compares **Condition A** (Brain-only), **Condition B** (Muscle-only), **Condition C** (Adaptive with Restart), and **Condition D** (Adaptive with Continuation):

```powershell
python benchmark/scientific_ab_runner.py --mock --limit 18
python benchmark/scientific_ab_runner.py --live --provider ollama --model llama3.2 --limit 18
python benchmark/scientific_ab_runner.py --live --provider openai --limit 18
```

*(Note: Existing artifacts generated in `results/` and `experiments/` are simulation baselines produced with `MockProvider` and `ModelSimulator`; live LLM validation requires active API credits or a running Ollama instance.)*

---

## Repository Structure

| Path | Purpose |
|---|---|
| `controller.py` | Authoritative `AgentController` lifecycle entrypoint |
| `universal_runtime.py` | `UniversalModelRuntime`, `ModelCapabilityCard`, capability negotiation, `CognitionLedger` |
| `provider_registry.py` | `ProviderRegistry` & `ProviderFactory` registrations |
| `router.py` | Deterministic safety/context/freshness gates & model classifier routing |
| `muscle_engine.py` | 4-tier `MuscleExecutor` (cache, deterministic Python, procedural skills, bounded adaptive) |
| `brain.py` | `BrainEngine` deliberative reasoning, dynamic effort selection, continuation execution |
| `handoff.py` | `HandoffState` distillation & continuation prompt builder |
| `execution_state.py` | Structured `ExecutionState`, `TaskFrontier`, disk SHA-256 reuse checks |
| `procedural_memory.py` | Gated skill store (`seed_skills.json`, `skills.json`) & quarantine filtering |
| `procedure_executor.py` | Deterministic procedural DAG executor under `SkillToolPolicy` |
| `verification.py` | Multi-tier verification checks |
| `success_contracts.py` | Explicit terminal `SuccessContract` validators |
| `tools.py` | Path-canonical sandboxed file tools, atomic writes, untrusted data delimiters |
| `base.py` | Shared state, cache, budgets, usage accounting, provider base classes |
| `dual_mode_agent.py` | Interactive CLI REPL |
| `restart_agent.py` | Pure-restart control baseline for scientific benchmarks |
| `retry_util.py` | Exponential backoff with jitter for provider calls |
| `anthropic_provider.py`, `openai_provider.py`, `ollama_provider.py`, `gemini_provider.py`, `mock_provider.py` | Provider adapters |
| `benchmark/` | 4-condition scientific A/B experiment runner |
| `scripts/` | Live validation, smoke tests, cache verification, A/B runner, CI audit sync |
| `tests/` | 269 unit, integration, adversarial, evaluation, and provider-contract tests |
| `docs/` | Detailed audit and readiness reports |

---

## Documentation

- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — Core v4/v5 architectural rules and baseline experiments
- [IMPLEMENTATION_PLAN_V4.md](IMPLEMENTATION_PLAN_V4.md) — v4 specification reference
- [docs/CURRENT_STATE_AUDIT.md](docs/CURRENT_STATE_AUDIT.md) — Module coverage, bug resolution record, and diagnostic baseline
- [docs/FINAL_V8_1_READINESS.md](docs/FINAL_V8_1_READINESS.md) — V9.1 Universal Runtime readiness report & 32-area verification matrix
- [examples/v5_trace_examples.md](examples/v5_trace_examples.md) — Execution trace examples
