"""Root-level export for MockProvider to eliminate PYTHONPATH packaging fragility."""
from tests.mock_provider import MockProvider

__all__ = ["MockProvider"]
