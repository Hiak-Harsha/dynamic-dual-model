"""
tests/test_dynamic_effort.py — Tests for the select_effort() function.

Verifies that effort is dynamically selected based on RouteDecision's
novelty and risk axes, replacing the previously hardcoded 'medium'.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dual_mode_agent import select_effort
from router import RouteDecision


def test_high_risk_yields_high_effort():
    decision = RouteDecision(pathway="full_brain", risk="high", novelty=0.1)
    assert select_effort(decision) == "high", "High risk should always yield high effort"
    print("PASS: high risk -> high effort")


def test_high_novelty_yields_high_effort():
    decision = RouteDecision(pathway="full_brain", risk="low", novelty=0.85)
    assert select_effort(decision) == "high", "High novelty (>=0.8) should yield high effort"
    print("PASS: high novelty -> high effort")


def test_low_novelty_low_risk_yields_low_effort():
    decision = RouteDecision(pathway="muscle", risk="low", novelty=0.2)
    assert select_effort(decision) == "low", "Low novelty + low risk should yield low effort"
    print("PASS: low novelty + low risk -> low effort")


def test_medium_novelty_medium_risk_yields_medium_effort():
    decision = RouteDecision(pathway="full_brain", risk="medium", novelty=0.5)
    assert select_effort(decision) == "medium", "Medium novelty + medium risk should yield medium effort"
    print("PASS: medium conditions -> medium effort")


def test_boundary_novelty_03_low_risk_yields_medium():
    """novelty=0.3 is NOT < 0.3, so it shouldn't qualify for low effort."""
    decision = RouteDecision(pathway="full_brain", risk="low", novelty=0.3)
    assert select_effort(decision) == "medium", "novelty=0.3 (boundary) should yield medium, not low"
    print("PASS: boundary novelty=0.3 -> medium effort")


def test_boundary_novelty_08_yields_high():
    """novelty=0.8 should trigger high effort."""
    decision = RouteDecision(pathway="full_brain", risk="low", novelty=0.8)
    assert select_effort(decision) == "high", "novelty=0.8 (boundary) should yield high effort"
    print("PASS: boundary novelty=0.8 -> high effort")


def test_risk_overrides_novelty():
    """High risk should yield high effort even with low novelty."""
    decision = RouteDecision(pathway="full_brain", risk="high", novelty=0.0)
    assert select_effort(decision) == "high", "High risk should override low novelty"
    print("PASS: risk=high overrides novelty=0.0")


if __name__ == "__main__":
    test_high_risk_yields_high_effort()
    test_high_novelty_yields_high_effort()
    test_low_novelty_low_risk_yields_low_effort()
    test_medium_novelty_medium_risk_yields_medium_effort()
    test_boundary_novelty_03_low_risk_yields_medium()
    test_boundary_novelty_08_yields_high()
    test_risk_overrides_novelty()
    print("\nALL DYNAMIC EFFORT TESTS PASSED!")
