"""
provider_registry.py — Provider Registry for the Universal Model Runtime.

Central registry where provider adapters self-register with their
ModelCapabilityCards. The runtime and CLI use this to discover, select,
and instantiate providers without hardcoded if/elif chains.

Per Non-Negotiable Rule: No giant `if provider ==` blocks.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Type

from base import ModelProvider
from universal_runtime import ModelCapabilityCard, UniversalModelRuntime, CognitionLedger


# ============================================================
# ProviderFactory — callable that creates a provider instance
# ============================================================
class ProviderFactory:
    """Wraps a provider class with its default model and capability card."""

    def __init__(
        self,
        name: str,
        provider_class: Type[ModelProvider],
        default_model: str,
        capability_card_factory: Callable[[str], ModelCapabilityCard],
        requires_api_key: Optional[str] = None,
        description: str = "",
        is_available_fn: Optional[Callable[[], bool]] = None,
    ):
        self.name = name
        self.provider_class = provider_class
        self.default_model = default_model
        self.capability_card_factory = capability_card_factory
        self.requires_api_key = requires_api_key
        self.description = description
        self.is_available_fn = is_available_fn

    def create(self, model: Optional[str] = None, **kwargs) -> ModelProvider:
        """Instantiate the provider with the specified or default model."""
        resolved_model = model or self.default_model
        return self.provider_class(model=resolved_model, **kwargs)

    def create_card(self, model: Optional[str] = None) -> ModelCapabilityCard:
        """Create a capability card for the specified or default model."""
        resolved_model = model or self.default_model
        return self.capability_card_factory(resolved_model)

    def is_available(self) -> bool:
        """Check if this provider can be instantiated (API key present, SDK installed, server reachable)."""
        if self.is_available_fn is not None:
            return self.is_available_fn()
        if self.requires_api_key:
            import os
            if not os.environ.get(self.requires_api_key):
                return False
        try:
            # Check if the provider class can be imported/used
            return self.provider_class is not None
        except Exception:
            return False

    def status_label(self) -> str:
        """User-friendly availability status string."""
        if self.is_available():
            if self.name == "ollama":
                return "AVAILABLE (localhost:11434)"
            elif self.name == "mock":
                return "AVAILABLE ($0 offline)"
            return "AVAILABLE"
        else:
            if self.name == "ollama":
                return "OFFLINE (start with 'ollama serve')"
            elif self.requires_api_key:
                return f"NEEDS {self.requires_api_key}"
            return "UNAVAILABLE"


# ============================================================
# ProviderRegistry — central registry of all providers
# ============================================================
class ProviderRegistry:
    """Central registry for provider adapters.

    Providers register themselves with a name, factory, and capability card.
    The registry eliminates hardcoded provider selection logic.
    """

    def __init__(self):
        self._factories: Dict[str, ProviderFactory] = {}

    def register(self, factory: ProviderFactory) -> None:
        """Register a provider factory."""
        self._factories[factory.name] = factory

    def get_factory(self, name: str) -> Optional[ProviderFactory]:
        """Get a registered factory by name."""
        return self._factories.get(name)

    def list_providers(self) -> List[str]:
        """List all registered provider names."""
        return list(self._factories.keys())

    def list_available(self) -> List[str]:
        """List providers that can be instantiated right now."""
        return [name for name, f in self._factories.items() if f.is_available()]

    def create_provider(self, name: str, model: Optional[str] = None, **kwargs) -> ModelProvider:
        """Create a provider instance by name."""
        factory = self._factories.get(name)
        if factory is None:
            available = ", ".join(self._factories.keys())
            raise ValueError(
                f"Unknown provider '{name}'. Available: {available}"
            )
        return factory.create(model=model, **kwargs)

    def create_runtime(
        self,
        name: str,
        model: Optional[str] = None,
        ledger: Optional[CognitionLedger] = None,
        **kwargs,
    ) -> UniversalModelRuntime:
        """Create a full UniversalModelRuntime for the specified provider."""
        factory = self._factories.get(name)
        if factory is None:
            available = ", ".join(self._factories.keys())
            raise ValueError(
                f"Unknown provider '{name}'. Available: {available}"
            )
        provider = factory.create(model=model, **kwargs)
        card = factory.create_card(model=model)
        return UniversalModelRuntime(
            provider=provider,
            capability_card=card,
            ledger=ledger or CognitionLedger(),
        )

    def provider_info(self) -> List[Dict[str, Any]]:
        """Returns info about all registered providers."""
        info = []
        for name, f in self._factories.items():
            info.append({
                "name": name,
                "default_model": f.default_model,
                "available": f.is_available(),
                "status_label": f.status_label(),
                "requires_api_key": f.requires_api_key,
                "description": f.description,
            })
        return info


# ============================================================
# Built-in provider registrations
# ============================================================
def _openai_card_factory(model: str) -> ModelCapabilityCard:
    return ModelCapabilityCard(
        provider_name="openai",
        model_id=model,
        supports_tool_calls=True,
        supports_structured_output=True,
        supports_reasoning_effort=True,
        supports_prompt_cache=True,
        supports_native_tool_history=True,
        supports_reasoning_usage=True,
        supports_thinking=False,
        supported_effort_levels=["low", "medium", "high"],
        max_input_tokens=128000,
        max_output_tokens=16384,
        min_output_tokens=16,
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
    )


def _anthropic_card_factory(model: str) -> ModelCapabilityCard:
    return ModelCapabilityCard(
        provider_name="anthropic",
        model_id=model,
        supports_tool_calls=True,
        supports_structured_output=True,
        supports_reasoning_effort=True,
        supports_prompt_cache=True,
        supports_native_tool_history=True,
        supports_reasoning_usage=False,
        supports_thinking=True,
        supported_effort_levels=["low", "medium", "high", "xhigh", "max"],
        max_input_tokens=200000,
        max_output_tokens=8192,
        min_output_tokens=8,
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
    )


def _mock_card_factory(model: str) -> ModelCapabilityCard:
    return ModelCapabilityCard(
        provider_name="mock",
        model_id=model,
        supports_tool_calls=True,
        supports_structured_output=True,
        supports_reasoning_effort=False,
        supports_prompt_cache=False,
        supports_native_tool_history=True,
        supports_reasoning_usage=False,
        supports_thinking=False,
        supported_effort_levels=["low", "medium", "high"],
        max_input_tokens=128000,
        max_output_tokens=8192,
        min_output_tokens=1,
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
    )


def _ollama_card_factory(model: str) -> ModelCapabilityCard:
    supports_tools = not any(s in model.lower() for s in ["tiny", "phi-1", "gemma-2b"])
    return ModelCapabilityCard(
        provider_name="ollama",
        model_id=model,
        supports_tool_calls=supports_tools,
        supports_structured_output=True,
        supports_reasoning_effort=False,
        supports_prompt_cache=False,
        supports_native_tool_history=True,
        supports_reasoning_usage=False,
        supports_thinking=False,
        supported_effort_levels=["low", "medium", "high"],
        max_input_tokens=128000,
        max_output_tokens=8192,
        min_output_tokens=16,
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
    )


def _openrouter_card_factory(model: str) -> ModelCapabilityCard:
    return ModelCapabilityCard(
        provider_name="openrouter",
        model_id=model,
        supports_tool_calls=True,
        supports_structured_output=True,
        supports_reasoning_effort=False,
        supports_prompt_cache=False,
        supports_native_tool_history=True,
        supports_reasoning_usage=False,
        supports_thinking=False,
        supported_effort_levels=["low", "medium", "high"],
        max_input_tokens=128000,
        max_output_tokens=8192,
        min_output_tokens=16,
        input_cost_per_1k=0.002,
        output_cost_per_1k=0.008,
    )


def _gemini_card_factory(model: str) -> ModelCapabilityCard:
    return ModelCapabilityCard(
        provider_name="gemini",
        model_id=model,
        supports_tool_calls=True,
        supports_structured_output=True,
        supports_reasoning_effort=False,
        supports_prompt_cache=False,
        supports_native_tool_history=True,
        supports_reasoning_usage=False,
        supports_thinking=True,
        supported_effort_levels=["low", "medium", "high", "max"],
        max_input_tokens=1000000,
        max_output_tokens=8192,
        min_output_tokens=16,
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.0006,
    )


def build_default_registry() -> ProviderRegistry:
    """Builds the registry with all built-in provider factories."""
    registry = ProviderRegistry()

    # OpenAI
    try:
        from openai_provider import OpenAIProvider
        registry.register(ProviderFactory(
            name="openai",
            provider_class=OpenAIProvider,
            default_model="gpt-4o",
            capability_card_factory=_openai_card_factory,
            requires_api_key="OPENAI_API_KEY",
            description="OpenAI GPT models via Responses API",
        ))
    except ImportError:
        pass

    # Anthropic
    try:
        from anthropic_provider import AnthropicProvider
        registry.register(ProviderFactory(
            name="anthropic",
            provider_class=AnthropicProvider,
            default_model="claude-sonnet-5",
            capability_card_factory=_anthropic_card_factory,
            requires_api_key="ANTHROPIC_API_KEY",
            description="Anthropic Claude models via Messages API",
        ))
    except ImportError:
        pass

    # Mock (always available)
    try:
        from tests.mock_provider import MockProvider
        registry.register(ProviderFactory(
            name="mock",
            provider_class=MockProvider,
            default_model="mock-sonnet-5",
            capability_card_factory=_mock_card_factory,
            requires_api_key=None,
            description="Deterministic mock provider for testing ($0, no API key needed)",
        ))
    except ImportError:
        pass

    # Ollama (local HTTP daemon, $0, no API key)
    try:
        from ollama_provider import OllamaProvider, is_ollama_reachable
        registry.register(ProviderFactory(
            name="ollama",
            provider_class=OllamaProvider,
            default_model="llama3.2",
            capability_card_factory=_ollama_card_factory,
            requires_api_key=None,
            is_available_fn=is_ollama_reachable,
            description="Local Ollama models via HTTP ($0, no API key needed)",
        ))
    except ImportError:
        pass

    # OpenRouter (OpenAI-compatible routing)
    try:
        from openai_provider import OpenRouterProvider
        registry.register(ProviderFactory(
            name="openrouter",
            provider_class=OpenRouterProvider,
            default_model="meta-llama/llama-3.3-70b-instruct",
            capability_card_factory=_openrouter_card_factory,
            requires_api_key="OPENROUTER_API_KEY",
            description="OpenRouter unified model routing API",
        ))
    except ImportError:
        pass

    # Gemini (Google Generative Language REST API)
    try:
        from gemini_provider import GeminiProvider
        registry.register(ProviderFactory(
            name="gemini",
            provider_class=GeminiProvider,
            default_model="gemini-1.5-flash",
            capability_card_factory=_gemini_card_factory,
            requires_api_key="GEMINI_API_KEY",
            description="Google Gemini models via REST API",
        ))
    except ImportError:
        pass

    return registry
