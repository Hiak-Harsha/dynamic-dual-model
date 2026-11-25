"""
test_v7_scientific_runner.py
Automated validation of the 4-condition controlled scientific experiment runner per V7 Phase 16.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmark.scientific_ab_runner import run_4_condition_experiment
from mock_provider import MockProvider


def test_scientific_4_condition_runner():
    """Validates that all 4 conditions execute cleanly, produce valid hypothesis metrics,
    and confirm Quality Parity and Token Savings."""
    provider = MockProvider()
    report = run_4_condition_experiment(provider)

    assert "conditions" in report
    conds = report["conditions"]
    assert "A_Brain_Only" in conds
    assert "B_Muscle_Only" in conds
    assert "C_Adaptive_Restart" in conds
    assert "D_Adaptive_Continuation" in conds

    # Metrics verification
    metrics = report["hypothesis_metrics"]
    assert "token_savings_d_vs_a_pct" in metrics
    assert "token_savings_d_vs_c_pct" in metrics
    assert "quality_parity_confirmed" in metrics
    assert metrics["quality_parity_confirmed"] is True
    assert metrics["token_savings_d_vs_a_pct"] > 0
    assert metrics["token_savings_d_vs_c_pct"] > 0
